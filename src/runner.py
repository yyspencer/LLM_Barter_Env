"""
runner.py
 
Experiment execution layer for the LLM barter experiment.
 
This module owns the full experiment loop:
  for each round:
      run preference probes if scheduled
      pair agents
      run negotiation for each pair
      decide whether trade happens (commitment phase)
      validate and execute trades
      update inventories
      log everything
 
Entry points:
  run_mock_experiment(cfg)   -- full pipeline with deterministic mock agents
  run_gpt_experiment(cfg)    -- full pipeline with real GPT-5.4 API calls
 
Both modes share the same round/probe/logging infrastructure. The only
difference is the agent callables passed into the shared loop.
"""
 
from __future__ import annotations
 
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
 
from config import LoadedConfig
from display_order import order_for_run
from logger import RunLogger, build_pair_id
from mock_agent import (
    mock_commitment_decision,
    mock_negotiation_action,
    mock_preference_probe,
)
from pairing import generate_pairing_rounds, resolve_total_rounds
from prompt_render import format_goods_dict
from shadow_trades import (
    build_shadow_proposed_trade,
    feasible_specs_for_inventory,
    generate_shadow_trade_specs_from_config,
    is_player_eligible,
)
from utility import (
    apply_trade_to_inventory,
    player_starting_utility_summary,
    predicted_marginal_utilities,
    shifted_cobb_douglas,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Inventory = Dict[str, int]

# Agent callables — each returns the appropriate response dict.
NegotiationFn      = Callable[..., Dict[str, Any]]
CommitmentFn       = Callable[..., Dict[str, Any]]
ProbeFn            = Callable[..., Dict[str, Any]]
ShadowCommitmentFn = Callable[..., Dict[str, Any]]
 
 
# ---------------------------------------------------------------------------
# Trade validation
# ---------------------------------------------------------------------------
 
def validate_trade(
    proposer_id: str,
    responder_id: str,
    proposed_trade: Mapping[str, Any],
    inventories: Mapping[str, Inventory],
    cfg: LoadedConfig,
) -> Tuple[bool, str]:
    """
    Validate a proposed trade against the experiment's trade rules.
    Returns (ok: bool, reason: str).
    """
    goods = set(cfg.experiment.market.goods)
    rules = cfg.experiment.trade_rules
 
    if not proposed_trade:
        return False, "proposed_trade is empty or None"
 
    give    = proposed_trade.get("give", {})
    receive = proposed_trade.get("receive", {})
 
    if not give or not receive:
        return False, "Trade must have non-empty give and receive sides"
 
    unknown_give    = set(give.keys()) - goods
    unknown_receive = set(receive.keys()) - goods
    if unknown_give or unknown_receive:
        return False, f"Unknown goods: give={unknown_give}, receive={unknown_receive}"
 
    for side_name, side in [("give", give), ("receive", receive)]:
        for good, qty in side.items():
            if not isinstance(qty, int) or qty < 0:
                return False, f"{side_name}.{good} must be a nonneg int, got {qty}"
 
    action_space = cfg.experiment.mechanism.action_space

    if action_space == "one_for_one":
        give_total    = sum(give.values())
        receive_total = sum(receive.values())
        if give_total != 1 or receive_total != 1:
            return (
                False,
                f"one_for_one requires exactly 1 unit each side; "
                f"got give={give_total}, receive={receive_total}",
            )
        if len(give) != 1 or len(receive) != 1:
            return False, "one_for_one requires exactly one good on each side"

    elif action_space == "small_bundle":
        # Any (bounded) quantity is fine, but each side must consist of a
        # single good type -- no mixing goods within a side.
        if len(give) != 1 or len(receive) != 1:
            return (
                False,
                "small_bundle requires exactly one good type on each side "
                f"(got give={list(give.keys())}, receive={list(receive.keys())})",
            )

    elif action_space == "any_bundle":
        # No shape restriction beyond the generic checks below (multiple
        # goods per side are allowed).
        pass

    give_total    = sum(give.values())
    receive_total = sum(receive.values())
    if give_total < rules.min_units_per_side or receive_total < rules.min_units_per_side:
        return False, f"Trade below min_units_per_side={rules.min_units_per_side}"
    if give_total > rules.max_units_per_side or receive_total > rules.max_units_per_side:
        return False, f"Trade exceeds max_units_per_side={rules.max_units_per_side}"
 
    if rules.enforce_inventory_constraints:
        proposer_inv = inventories[proposer_id]
        for good, qty in give.items():
            if proposer_inv.get(good, 0) < qty:
                return (
                    False,
                    f"Proposer {proposer_id} cannot give {qty}×{good} "
                    f"(has {proposer_inv.get(good, 0)})",
                )
        responder_inv = inventories[responder_id]
        for good, qty in receive.items():
            if responder_inv.get(good, 0) < qty:
                return (
                    False,
                    f"Responder {responder_id} cannot give {qty}×{good} "
                    f"(has {responder_inv.get(good, 0)})",
                )
 
    return True, "ok"
 
 
def execute_trade(
    proposer_id: str,
    responder_id: str,
    proposed_trade: Mapping[str, Any],
    inventories: Dict[str, Inventory],
) -> None:
    """
    Apply a validated, accepted trade to both players' inventories in-place.
 
    proposed_trade is from the proposer's perspective:
      give    = what proposer gives    = what responder receives
      receive = what proposer receives = what responder gives
    """
    give    = proposed_trade["give"]
    receive = proposed_trade["receive"]
 
    inventories[proposer_id] = apply_trade_to_inventory(
        inventory=inventories[proposer_id],
        give=give,
        receive=receive,
        enforce_nonnegative=True,
    )
    inventories[responder_id] = apply_trade_to_inventory(
        inventory=inventories[responder_id],
        give=receive,
        receive=give,
        enforce_nonnegative=True,
    )
 
 
# ---------------------------------------------------------------------------
# Shadow trades
# ---------------------------------------------------------------------------

def run_shadow_trades_for_player(
    player: Any,
    inventory: Inventory,
    round_index: int,
    label: str,
    cfg: LoadedConfig,
    logger: RunLogger,
    shadow_commitment_fn: Optional[ShadowCommitmentFn],
    specs: List[Dict[str, Any]],
    display_order: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Send one player the full battery of feasible shadow trades (see
    shadow_trades.py) and record each accept/reject response.

    Each shadow trade is built and sent through ``shadow_commitment_fn``
    exactly like a real commitment decision — same prompt shape, generic
    fictitious partner — so the agent has no reason to treat it differently.
    The response is only ever logged (transcript + shadow_trades.json + the
    returned list, which callers attach to the concurrent probe record); it
    is never applied to inventories and never fed into negotiation history,
    trade history, or the bulletin board, so it cannot influence any later
    prompt.

    Returns [] if shadow trades are disabled, this player is not in the
    configured player_ids, or shadow_commitment_fn is None (e.g. the random
    baseline, which has no LLM to ask).
    """
    st_cfg = cfg.experiment.preference_drift.shadow_trades
    if shadow_commitment_fn is None or not st_cfg.enabled:
        return []
    if not is_player_eligible(player.id, st_cfg.player_ids):
        return []

    goods = cfg.experiment.market.goods
    shift = cfg.experiment.utility.terminal_utility.shift
    utility_before = shifted_cobb_douglas(
        inventory=inventory, weights=player.utility_weights, shift=shift, goods=goods,
    )

    feasible = feasible_specs_for_inventory(specs, inventory)
    records: List[Dict[str, Any]] = []

    for spec in feasible:
        proposed_trade = build_shadow_proposed_trade(spec)
        decision = shadow_commitment_fn(
            player=player,
            proposed_trade=proposed_trade,
            round_index=round_index,
            shadow_id=spec["shadow_id"],
            display_order=display_order,
        )

        # Hypothetical only — this is never applied to the real inventory,
        # just computed to report what accepting this offer *would* have
        # done, alongside the agent's actual accept/reject response.
        inventory_after = apply_trade_to_inventory(
            inventory=inventory, give=spec["give"], receive=spec["receive"], goods=goods,
        )
        utility_after = shifted_cobb_douglas(
            inventory=inventory_after, weights=player.utility_weights, shift=shift, goods=goods,
        )

        record = {
            "round_index": round_index,
            "probe_label": label,
            "player_id": player.id,
            "display_name": player.display_name,
            "shadow_id": spec["shadow_id"],
            "focal_good": spec["focal_good"],
            "counter_good": spec["counter_good"],
            "direction": spec["direction"],
            "player_give": dict(spec["give"]),
            "player_receive": dict(spec["receive"]),
            "inventory_before": dict(inventory),
            "inventory_after": inventory_after,
            "utility_before": utility_before,
            "utility_after": utility_after,
            "decision": decision.get("decision"),
            "reasoning_summary": decision.get("reasoning_summary", ""),
        }
        records.append(record)
        logger.log_shadow_trade(record)
        logger.append_transcript(
            f"[SHADOW {label}] {player.display_name}: {spec['direction']} offer — "
            f"give {format_goods_dict(spec['give'])}, receive {format_goods_dict(spec['receive'])} "
            f"=> {record['decision']}\n"
        )

    return records


# ---------------------------------------------------------------------------
# Preference probes
# ---------------------------------------------------------------------------

def run_preference_probes(
    players: List[Any],
    inventories: Dict[str, Inventory],
    round_index: int,
    logger: RunLogger,
    cfg: LoadedConfig,
    probe_fn: ProbeFn,
    shadow_commitment_fn: Optional[ShadowCommitmentFn] = None,
    display_order: Optional[List[str]] = None,
    label_override: Optional[str] = None,
    trade_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    bulletin_board: Optional[List[str]] = None,
    broadcast: bool = False,
) -> None:
    """
    Run preference elicitation probes for all players.

    round_index=0  -> pre-run
    round_index=-1 -> post-run
    round_index=N  -> mid-experiment after round N

    label_override, if provided, replaces the auto-derived label. Probe-only
    mode uses this to produce labels like ``probe_1``, ``probe_2``, ... that
    don't get confused with trade-round numbers.

    probe_fn signature: (player, round_index) -> dict

    shadow_commitment_fn, if given, sends every player the configured
    battery of hypothetical shadow trades (see run_shadow_trades_for_player)
    alongside their probe. Pass None to skip shadow trades entirely.
    """
    if label_override is not None:
        label = label_override
    else:
        label = (
            "pre_run"  if round_index == 0
            else "post_run" if round_index == -1
            else f"round_{round_index}"
        )

    shadow_specs = (
        generate_shadow_trade_specs_from_config(cfg)
        if shadow_commitment_fn is not None
        else []
    )

    for player in players:
        response = probe_fn(
            player=player,
            round_index=round_index,
            display_order=display_order,
            trade_history=(trade_history or {}).get(player.id),
            bulletin_board=bulletin_board,
            broadcast=broadcast,
        )

        if response is None:
            logger.append_transcript(
                f"[PROBE {label}] {player.display_name}: SKIPPED (parse error)\n"
            )
            continue

        current_inventory = dict(inventories[player.id])
        predicted_marginal_utility = predicted_marginal_utilities(
            inventory=current_inventory,
            weights=player.utility_weights,
            goods=cfg.experiment.market.goods,
            shift=cfg.experiment.utility.terminal_utility.shift,
        )

        shadow_trade_records = run_shadow_trades_for_player(
            player=player,
            inventory=current_inventory,
            round_index=round_index,
            label=label,
            cfg=cfg,
            logger=logger,
            shadow_commitment_fn=shadow_commitment_fn,
            specs=shadow_specs,
            display_order=display_order,
        )

        probe_record = {
            "round_index": round_index,
            "probe_label": label,
            "player_id": player.id,
            "display_name": player.display_name,
            "current_inventory": current_inventory,
            "predicted_marginal_utility": predicted_marginal_utility,
            "display_order": list(display_order) if display_order else None,
            "response": response,
            "shadow_trades": shadow_trade_records,
        }
        logger.log_preference_probe(probe_record)
        logger.append_transcript(
        f"[PROBE {label}] {player.display_name}:\n"
        f"    current_inventory={current_inventory}\n"
        f"    predicted_marginal_utility={predicted_marginal_utility}\n"
        f"    ratings_inventory={response['ratings_inventory']}\n"
        f"    ratings_general={response['ratings_general']}\n"
        f"    desired_bundle={response['desired_bundle']}\n"
        )
 
 
# ---------------------------------------------------------------------------
# Single pair negotiation — shared logic
# ---------------------------------------------------------------------------
 
def run_pair_negotiation(
    player_a: Any,
    player_b: Any,
    inventories: Dict[str, Inventory],
    round_index: int,
    pair_id: str,
    cfg: LoadedConfig,
    logger: RunLogger,
    negotiation_fn: NegotiationFn,
    commitment_fn: CommitmentFn,
    display_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run one pair's negotiation using the provided agent callables.
 
    negotiation_fn(player, partner, negotiation_history, turn_index,
                   round_index, pair_id) -> dict
    commitment_fn(player, proposed_trade, round_index, pair_id) -> dict
 
    Returns a result dict with keys:
      trade_accepted, proposed_trade, proposer_id, responder_id,
      negotiation_log, rejection_reason
    """
    turns_per_agent = cfg.experiment.mechanism.negotiation_turns_per_agent
    max_turns = turns_per_agent * 2
 
    negotiation_history: List[Dict[str, Any]] = []
    players = [player_a, player_b]
 
    result: Dict[str, Any] = {
        "trade_accepted": False,
        "proposed_trade": None,
        "proposer_id": None,
        "responder_id": None,
        "negotiation_log": negotiation_history,
        "rejection_reason": "No agreement reached within turn limit",
    }
 
    for turn in range(max_turns):
        current = players[turn % 2]
        other   = players[(turn + 1) % 2]
 
        action = negotiation_fn(
            player=current,
            partner=other,
            negotiation_history=negotiation_history,
            turn_index=turn,
            round_index=round_index,
            pair_id=pair_id,
            display_order=display_order,
        )
 
        msg_record = {
            "turn": turn,
            "speaker_id": current.id,
            "speaker": current.display_name,
            "action_type": action["action_type"],
            "message": action.get("message_to_partner", ""),
            "proposed_trade": action.get("proposed_trade"),
            "accept_trade": action.get("accept_trade"),
        }
        negotiation_history.append(msg_record)
 
        logger.log_event(
            event_type="negotiation_turn",
            payload={"pair_id": pair_id, "turn": turn, "action": action},
            round_index=round_index,
            player_id=current.id,
            pair_id=pair_id,
        )
        logger.append_transcript(
            f"  Turn {turn + 1} [{current.display_name}] "
            f"({action['action_type']}): {action.get('message_to_partner', '')}\n"
        )
 
        action_type = action["action_type"]

        # --- "accept" is NOT a valid standalone negotiation action ---
        # Acceptance happens only in the commitment phase, at the moment a
        # trade is offered to a player. An agent cannot accept an earlier
        # offer on its own turn. If an agent emits "accept" anyway — often a
        # sign it placed its real offer in the wrong field — treat it as a
        # harmless no-op and continue. Crucially, we NEVER execute a stale
        # prior offer, which is the bug this prevents.
        if action_type == "accept":
            logger.append_transcript(
                f"  [ACCEPT IGNORED: {current.display_name} used \"accept\" "
                f"on their own turn. Offers can only be accepted at the "
                f"moment they are made, not after the fact. Nothing was "
                f"traded.]\n"
            )
            negotiation_history.append({
                "turn": turn,
                "speaker_id": current.id,
                "speaker": current.display_name,
                "action_type": "system_note",
                "message": (
                    f"{current.display_name} tried to ACCEPT on their own "
                    f"turn, but offers can only be accepted at the instant "
                    f"they are offered (when the partner makes the offer, "
                    f"the other player is asked to accept or reject it right "
                    f"then). You cannot accept an earlier offer after the "
                    f"fact. Nothing was traded. If you want a particular "
                    f"trade, propose it yourself with action_type \"offer\" "
                    f"and a proposed_trade. The negotiation continues."
                ),
                "proposed_trade": None,
                "accept_trade": None,
                "source": "accept_failed",
            })
            continue
 
        # --- Offer / counteroffer: run commitment phase (binary) ---
        if action_type in ("offer", "counteroffer") and action.get("proposed_trade"):
            proposed = action["proposed_trade"]
 
            ok, reason = validate_trade(
                proposer_id=current.id,
                responder_id=other.id,
                proposed_trade=proposed,
                inventories=inventories,
                cfg=cfg,
            )
            if not ok:
                logger.append_transcript(
                    f"  [INVALID TRADE from {current.display_name}: {reason}]\n"
                )
                result["rejection_reason"] = f"Invalid trade: {reason}"
                continue
 
            commitment = commitment_fn(
                player=other,
                proposed_trade=proposed,
                round_index=round_index,
                pair_id=pair_id,
                partner_name=current.display_name,
                negotiation_history=negotiation_history,
                display_order=display_order,
            )
            decision = commitment["decision"]
            reasoning = commitment.get("reasoning_summary", "")
 
            logger.log_event(
                event_type="commitment_decision",
                payload={
                    "pair_id": pair_id,
                    "proposer_id": current.id,
                    "responder_id": other.id,
                    "proposed_trade": proposed,
                    "decision": decision,
                },
                round_index=round_index,
                pair_id=pair_id,
            )
            logger.append_transcript(
                f"  [COMMITMENT] {other.display_name}: "
                f"{decision} — {reasoning}\n"
            )
 
            if decision == "accept":
                result.update({
                    "trade_accepted": True,
                    "proposed_trade": proposed,
                    "proposer_id": current.id,
                    "responder_id": other.id,
                    "rejection_reason": None,
                })
                break
 
            # decision == "reject": record an explicit outcome entry in the
            # shared negotiation history so that on later turns the responder
            # can see it already declined this exact offer and why. This
            # prevents the agent from accidentally re-accepting an offer it
            # previously rejected at commitment time.
            give_str = format_goods_dict(proposed.get("give", {}))
            receive_str = format_goods_dict(proposed.get("receive", {}))
            negotiation_history.append({
                "turn": turn,
                "speaker_id": other.id,
                "speaker": other.display_name,
                "action_type": "commitment_decision",
                "decision": "reject",
                "message": (
                    f"{other.display_name} reviewed {current.display_name}'s "
                    f"offer ({current.display_name} gives {give_str}; "
                    f"{current.display_name} receives {receive_str}) and "
                    f"DECLINED it. Reason: "
                    f"{reasoning or '(no reason given)'}. "
                    f"That exact offer is now off the table and should not be "
                    f"accepted later. The negotiation continues — a different "
                    f"offer may still be proposed."
                ),
                "proposed_trade": None,
                "accept_trade": None,
                "source": "commitment_outcome",
            })
            continue
 
        # --- no_trade or reject ---
        if action_type in ("no_trade", "reject"):
            result["rejection_reason"] = f"{current.display_name} chose {action_type}"
            break
 
    return result
 
 
# ---------------------------------------------------------------------------
# Single round — shared logic
# ---------------------------------------------------------------------------
 
def run_round(
    round_index: int,
    pairs: List[Tuple[str, str]],
    player_map: Dict[str, Any],
    inventories: Dict[str, Inventory],
    cfg: LoadedConfig,
    logger: RunLogger,
    trade_history: Dict[str, List[Dict[str, Any]]],
    negotiation_fn: NegotiationFn,
    commitment_fn: CommitmentFn,
    bulletin_board: Optional[List[str]] = None,
    display_order: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run one experiment round for all pairs. Returns trade result records.

    If ``bulletin_board`` is provided, a fixed-format market bulletin is
    appended to it after every accepted trade (broadcast condition). The
    same list is shared with the negotiation agents so they see all market
    activity so far when broadcast is enabled.
    """
    round_trades: List[Dict[str, Any]] = []
    # Bulletins generated this round are buffered and only merged into the
    # shared (agent-visible) bulletin_board at END of round, so agents in
    # round N see bulletins through round N-1 only.
    pending_bulletins: List[str] = []
 
    logger.append_transcript(f"\n{'=' * 60}\nROUND {round_index}\n{'=' * 60}\n")
    logger.log_event(
        event_type="round_started",
        payload={"round_index": round_index, "num_pairs": len(pairs)},
        round_index=round_index,
    )
 
    for player_a_id, player_b_id in pairs:
        player_a = player_map[player_a_id]
        player_b = player_map[player_b_id]
        pair_id  = build_pair_id(round_index, player_a_id, player_b_id)
 
        logger.append_transcript(
            f"\nPair: {player_a.display_name} vs {player_b.display_name} [{pair_id}]\n"
            f"  {player_a.display_name} inventory: {dict(inventories[player_a_id])}\n"
            f"  {player_b.display_name} inventory: {dict(inventories[player_b_id])}\n"
        )
        logger.log_event(
            event_type="pair_started",
            payload={
                "pair_id": pair_id,
                "player_a": player_a_id,
                "player_b": player_b_id,
                "inventory_a": dict(inventories[player_a_id]),
                "inventory_b": dict(inventories[player_b_id]),
            },
            round_index=round_index,
            pair_id=pair_id,
        )
 
        pair_result = run_pair_negotiation(
            player_a=player_a,
            player_b=player_b,
            inventories=inventories,
            round_index=round_index,
            pair_id=pair_id,
            cfg=cfg,
            logger=logger,
            negotiation_fn=negotiation_fn,
            commitment_fn=commitment_fn,
            display_order=display_order,
        )
 
        if pair_result["trade_accepted"]:
            proposed     = pair_result["proposed_trade"]
            proposer_id  = pair_result["proposer_id"]
            responder_id = pair_result["responder_id"]
 
            inv_before_a = dict(inventories[player_a_id])
            inv_before_b = dict(inventories[player_b_id])
 
            execute_trade(proposer_id, responder_id, proposed, inventories)
 
            util_before_a = shifted_cobb_douglas(inv_before_a, dict(player_a.utility_weights))
            util_after_a  = shifted_cobb_douglas(inventories[player_a_id], dict(player_a.utility_weights))
            util_before_b = shifted_cobb_douglas(inv_before_b, dict(player_b.utility_weights))
            util_after_b  = shifted_cobb_douglas(inventories[player_b_id], dict(player_b.utility_weights))
 
            trade_record = {
                "round_index": round_index,
                "pair_id": pair_id,
                "proposer_id": proposer_id,
                "responder_id": responder_id,
                "proposed_trade": proposed,
                "accepted": True,
                "inventory_before": {
                    player_a_id: inv_before_a,
                    player_b_id: inv_before_b,
                },
                "inventory_after": {
                    player_a_id: dict(inventories[player_a_id]),
                    player_b_id: dict(inventories[player_b_id]),
                },
                "utility_before": {player_a_id: util_before_a, player_b_id: util_before_b},
                "utility_after":  {player_a_id: util_after_a,  player_b_id: util_after_b},
            }
            logger.log_trade(trade_record)
 
            for pid in [player_a_id, player_b_id]:
                partner_id = player_b_id if pid == player_a_id else player_a_id
                trade_history.setdefault(pid, []).append({
                    "round_index": round_index,
                    "pair_id": pair_id,
                    "partner_id": partner_id,
                    "partner_name": player_map[partner_id].display_name,
                    "decision": "accepted",
                    "give":    proposed["give"]    if pid == proposer_id else proposed["receive"],
                    "receive": proposed["receive"] if pid == proposer_id else proposed["give"],
                })
 
            round_trades.append(trade_record)
            logger.append_transcript(
                f"  TRADE ACCEPTED: give={proposed['give']}, receive={proposed['receive']}\n"
                f"  {player_a.display_name}: "
                f"{inv_before_a} → {dict(inventories[player_a_id])} "
                f"(utility {util_before_a:.4f} → {util_after_a:.4f})\n"
                f"  {player_b.display_name}: "
                f"{inv_before_b} → {dict(inventories[player_b_id])} "
                f"(utility {util_before_b:.4f} → {util_after_b:.4f})\n"
            )

            # Broadcast condition: append a fixed-format market bulletin for
            # every accepted trade. The "fair and necessary" framing is
            # intentional — it injects social framing to probe for preference
            # drift. The proposer gave `give` and received `receive`.
            if bulletin_board is not None:
                # Optional player filter: only broadcast trades involving
                # specific agents (e.g. broadcast_filter_players: [player_1]).
                # When null/empty, all trades are broadcast.
                bfp = cfg.experiment.mechanism.broadcast_filter_players
                if bfp and proposer_id not in bfp and responder_id not in bfp:
                    pass  # neither participant is in the filter — skip
                else:
                    proposer_name = player_map[proposer_id].display_name
                    responder_name = player_map[responder_id].display_name
                    give_str = format_goods_dict(proposed["give"])
                    receive_str = format_goods_dict(proposed["receive"])

                    # Pick which template to use based on bulletin_rules.
                    # A rule matches when its `player` is a participant
                    # AND that player gains the good in `gains`. The
                    # proposer gains everything in `receive`; the
                    # responder gains everything in `give`. First matching
                    # rule wins; fallback is market_bulletin_template.
                    template_name = "market_bulletin_template"
                    rules = cfg.experiment.mechanism.bulletin_rules or []
                    for rule in rules:
                        if rule.player == proposer_id and rule.gains in proposed["receive"]:
                            template_name = rule.template
                            break
                        if rule.player == responder_id and rule.gains in proposed["give"]:
                            template_name = rule.template
                            break

                    template = getattr(cfg.prompts, template_name, None)
                    if template is None:
                        logger.append_transcript(
                            f"  [BULLETIN WARNING] template '{template_name}' "
                            f"not found in prompts.yaml; using default.\n"
                        )
                        template = cfg.prompts.market_bulletin_template

                    bulletin = template.format(
                        proposer_name=proposer_name,
                        give_str=give_str,
                        receive_str=receive_str,
                        responder_name=responder_name,
                    )
                    # Buffer it; visible to agents only at round end.
                    pending_bulletins.append(bulletin)
 
        else:
            reason = pair_result.get("rejection_reason", "unknown")
            no_trade_record = {
                "round_index": round_index,
                "pair_id": pair_id,
                "accepted": False,
                "rejection_reason": reason,
                "proposed_trade": pair_result.get("proposed_trade"),
            }
            logger.log_trade(no_trade_record)
            round_trades.append(no_trade_record)
            logger.append_transcript(f"  NO TRADE: {reason}\n")
 
        logger.log_event(
            event_type="pair_completed",
            payload={
                "pair_id": pair_id,
                "trade_accepted": pair_result["trade_accepted"],
                "rejection_reason": pair_result.get("rejection_reason"),
            },
            round_index=round_index,
            pair_id=pair_id,
        )
 
    # End of round: publish this round's bulletins so they become visible
    # to all agents starting next round.
    if bulletin_board is not None and pending_bulletins:
        bulletin_board.extend(pending_bulletins)
        logger.append_transcript(
            f"\n[MARKET BULLETINS POSTED AT END OF ROUND {round_index} "
            f"(visible to all agents from next round)]\n"
        )
        for b in pending_bulletins:
            logger.append_transcript(f"  {b}\n")

    logger.log_event(
        event_type="round_completed",
        payload={
            "round_index": round_index,
            "trades_accepted": sum(1 for t in round_trades if t["accepted"]),
            "trades_rejected": sum(1 for t in round_trades if not t["accepted"]),
        },
        round_index=round_index,
    )
 
    return round_trades
 
 
# ---------------------------------------------------------------------------
# Probe schedule
# ---------------------------------------------------------------------------
 
def should_probe(
    round_index: int,
    cfg: LoadedConfig,
    total_rounds: Optional[int] = None,
) -> bool:
    """Return True if a mid-experiment probe should run after this round.

    Modes:
      - "interval_rounds": fire every `interval_rounds` completed rounds
        (after rounds r, 2r, 3r, ...). Recurring.
      - "midpoint": fire exactly once, after the middle round of the run.
        The middle is (total_rounds + 1) // 2, so a 5-round run probes
        after round 3, a 4-round run after round 2. Requires total_rounds.
    """
    if not cfg.experiment.preference_drift.enabled:
        return False
    sched = cfg.experiment.preference_drift.probe_schedule

    if sched.mode == "interval_rounds":
        return round_index > 0 and round_index % sched.interval_rounds == 0

    if sched.mode == "midpoint":
        if not total_rounds or total_rounds < 1:
            return False
        midpoint = (total_rounds + 1) // 2
        return round_index == midpoint

    return False


# ---------------------------------------------------------------------------
# Washout schedule
# ---------------------------------------------------------------------------

def base_total_rounds(exp_cfg: Any, num_players: int) -> int:
    """
    Number of rounds in the "normal" schedule, before any washout rounds are
    appended. Pure function of config + player count so it can be computed
    identically wherever it's needed (the main loop and the negotiation/
    commitment closures both need it, and they run in different functions).
    """
    return resolve_total_rounds(
        num_players=num_players,
        num_rounds=exp_cfg.rounds.max_rounds_override,
        round_multiplier=exp_cfg.rounds.round_multiplier,
    )


def washout_round_indices(exp_cfg: Any, num_players: int) -> set[int]:
    """Absolute round indices that belong to the washout block, if enabled."""
    washout_cfg = exp_cfg.washout
    if not washout_cfg.enabled or washout_cfg.num_rounds <= 0:
        return set()
    base = base_total_rounds(exp_cfg, num_players)
    return set(range(base + 1, base + washout_cfg.num_rounds + 1))


def washout_probe_round_indices(exp_cfg: Any, num_players: int) -> set[int]:
    """Absolute round indices after which a washout probe should fire."""
    washout_cfg = exp_cfg.washout
    if not washout_cfg.enabled or washout_cfg.num_rounds <= 0:
        return set()
    base = base_total_rounds(exp_cfg, num_players)
    return {base + offset for offset in washout_cfg.probe_after_washout_rounds}


def effective_broadcast(exp_cfg: Any, num_players: int, round_index: int) -> bool:
    """
    Whether the market bulletin should be shown/recorded for this round or
    probe. Same as mechanism.broadcast_completed_trades, except forced off
    during washout rounds when washout.disable_broadcast is set — even
    though the underlying mechanism config is unchanged for the rest of the
    run. round_index=0 (pre-run) and -1 (post-run) never fall in the
    washout block, so they're unaffected.
    """
    base_broadcast = exp_cfg.mechanism.broadcast_completed_trades
    washout_cfg = exp_cfg.washout
    if (
        base_broadcast
        and washout_cfg.enabled
        and washout_cfg.disable_broadcast
        and round_index in washout_round_indices(exp_cfg, num_players)
    ):
        return False
    return base_broadcast


# ---------------------------------------------------------------------------
# Shared experiment loop
# ---------------------------------------------------------------------------
 
def _run_experiment_loop(
    cfg: LoadedConfig,
    logger: RunLogger,
    mode: str,
    negotiation_fn: NegotiationFn,
    commitment_fn: CommitmentFn,
    probe_fn: ProbeFn,
    shadow_commitment_fn: Optional[ShadowCommitmentFn] = None,
    enable_probes: bool = True,
    bulletin_board: Optional[List[str]] = None,
) -> RunLogger:
    """
    Core loop shared by all run modes.

    Callers construct agent callables and pass them in.
    This function owns all scheduling, inventory management, and summary writing.

    When ``enable_probes`` is False, pre/mid/post preference probes are skipped
    regardless of the drift config. This is used by run modes that don't
    involve an LLM (e.g. the random baseline), where probes have no meaning.
    shadow_commitment_fn is threaded through unchanged; passing None (the
    default) skips shadow trades regardless of shadow_trades.enabled.
    """
    exp_cfg = cfg.experiment
    goods   = exp_cfg.market.goods
    players = cfg.players.players
 
    player_map: Dict[str, Any]           = {p.id: p for p in players}
    inventories: Dict[str, Inventory]    = {p.id: dict(p.inventory) for p in players}
    trade_history: Dict[str, List[Dict]] = {}

    # Run-wide counterbalancing of goods display order (see display_order.py).
    run_index = exp_cfg.experiment.run_index
    display_order = order_for_run(run_index)
    logger.log_event("display_order_assigned", {
        "run_index": run_index,
        "display_order": display_order,
    })
    logger.append_transcript(
        f"\n[DISPLAY ORDER for this run: {', '.join(display_order)}]\n"
    )

    # Starting state
    starting_utilities = []
    logger.append_transcript("STARTING INVENTORIES AND UTILITIES\n" + "=" * 60 + "\n")
    for player in players:
        summary = player_starting_utility_summary(player, goods, shift=1.0)
        starting_utilities.append(summary)
        logger.log_event("player_starting_state", payload=summary, player_id=player.id)
        logger.append_transcript(
            f"  {player.display_name} ({player.id}): "
            f"inv={summary['inventory']}, U={summary['utility']:.4f}\n"
        )
 
    num_players = len(players)
    washout_cfg = exp_cfg.washout
    washout_rounds = washout_round_indices(exp_cfg, num_players)
    washout_probe_rounds = washout_probe_round_indices(exp_cfg, num_players)

    # Pre-run probe
    drift_cfg = exp_cfg.preference_drift
    if enable_probes and drift_cfg.enabled and drift_cfg.probe_schedule.include_pre_probe:
        logger.append_transcript("\n[PRE-RUN PREFERENCE PROBES]\n")
        pre_run_broadcast = effective_broadcast(exp_cfg, num_players, round_index=0)
        run_preference_probes(
            players=players,
            inventories=inventories,
            round_index=0,
            logger=logger,
            cfg=cfg,
            probe_fn=probe_fn,
            shadow_commitment_fn=shadow_commitment_fn,
            display_order=display_order,
            trade_history=trade_history,
            bulletin_board=bulletin_board if pre_run_broadcast else None,
            broadcast=pre_run_broadcast,
        )

    # Pairing schedule — extended with washout_cfg.num_rounds extra rounds
    # (same pairing mechanism, just appended) when washout is enabled.
    total_scheduled_rounds = base_total_rounds(exp_cfg, num_players) + len(washout_rounds)
    pairing_rounds = generate_pairing_rounds(
        player_ids=[p.id for p in players],
        num_rounds=total_scheduled_rounds,
        round_multiplier=exp_cfg.rounds.round_multiplier,
        reshuffle_between_cycles=exp_cfg.pairing.reshuffle_between_runs,
        seed=exp_cfg.experiment.seed,
    )

    if washout_rounds:
        logger.append_transcript(
            f"\n[WASHOUT ENABLED: {washout_cfg.num_rounds} extra rounds appended "
            f"(rounds {min(washout_rounds)}-{max(washout_rounds)}). "
            f"Broadcast disabled for these rounds: {washout_cfg.disable_broadcast}. "
            f"Washout probes scheduled after rounds: {sorted(washout_probe_rounds)}]\n"
        )

    all_trade_records: List[Dict[str, Any]] = []
    total_rounds = len(pairing_rounds)

    for pr in pairing_rounds:
        round_index = pr.round_index
        round_broadcast = effective_broadcast(exp_cfg, num_players, round_index)

        round_trades = run_round(
            round_index=round_index,
            pairs=pr.pairs,
            player_map=player_map,
            inventories=inventories,
            cfg=cfg,
            logger=logger,
            trade_history=trade_history,
            negotiation_fn=negotiation_fn,
            commitment_fn=commitment_fn,
            bulletin_board=bulletin_board if round_broadcast else None,
            display_order=display_order,
        )
        all_trade_records.extend(round_trades)

        # Mid-run preference probe. This runs AFTER the round's trades have
        # been negotiated and applied to inventories, so a probe labelled
        # "after round N" genuinely reflects the state once round N is
        # complete (not the state going into round N). Washout rounds add
        # their own probe points (washout_probe_rounds) on top of the
        # normal interval/midpoint schedule.
        if (
            enable_probes
            and (
                should_probe(round_index, cfg, total_rounds=total_rounds)
                or round_index in washout_probe_rounds
            )
        ):
            washout_note = " (washout)" if round_index in washout_rounds else ""
            logger.append_transcript(
                f"\n[MID-RUN PREFERENCE PROBE — after round {round_index}{washout_note}]\n"
            )
            run_preference_probes(
                players=players,
                inventories=inventories,
                round_index=round_index,
                logger=logger,
                cfg=cfg,
                probe_fn=probe_fn,
                shadow_commitment_fn=shadow_commitment_fn,
                display_order=display_order,
                trade_history=trade_history,
                bulletin_board=bulletin_board if round_broadcast else None,
                broadcast=round_broadcast,
            )

        n_stop = exp_cfg.stopping.stop_if_no_trades_for_n_rounds
        if n_stop is not None:
            recent = all_trade_records[-(n_stop * max(len(pr.pairs), 1)):]
            if recent and not any(t["accepted"] for t in recent):
                logger.log_event(
                    "early_stop",
                    payload={"reason": f"No trades for {n_stop} consecutive rounds"},
                    round_index=round_index,
                )
                logger.append_transcript(
                    f"\n[EARLY STOP] No trades for {n_stop} consecutive rounds.\n"
                )
                break
 
    # Post-run probe
    if enable_probes and drift_cfg.enabled and drift_cfg.probe_schedule.include_post_probe:
        logger.append_transcript("\n[POST-RUN PREFERENCE PROBES]\n")
        post_run_broadcast = effective_broadcast(exp_cfg, num_players, round_index=-1)
        run_preference_probes(
            players=players,
            inventories=inventories,
            round_index=-1,
            logger=logger,
            cfg=cfg,
            probe_fn=probe_fn,
            shadow_commitment_fn=shadow_commitment_fn,
            display_order=display_order,
            trade_history=trade_history,
            bulletin_board=bulletin_board if post_run_broadcast else None,
            broadcast=post_run_broadcast,
        )
 
    # Final summary
    final_utilities = []
    for player in players:
        u = shifted_cobb_douglas(inventories[player.id], dict(player.utility_weights))
        final_utilities.append({
            "player_id": player.id,
            "display_name": player.display_name,
            "final_inventory": dict(inventories[player.id]),
            "final_utility": u,
        })
 
    total_accepted = sum(1 for t in all_trade_records if t["accepted"])
    total_rejected = sum(1 for t in all_trade_records if not t["accepted"])
 
    summary = {
        "experiment_name": exp_cfg.experiment.name,
        "mode": mode,
        "num_rounds": len(pairing_rounds),
        "num_players": exp_cfg.market.num_players,
        "goods": goods,
        "seed": exp_cfg.experiment.seed,
        "run_index": run_index,
        "display_order": display_order,
        "total_trades_accepted": total_accepted,
        "total_trades_rejected": total_rejected,
        "starting_utilities": starting_utilities,
        "final_utilities": final_utilities,
        "total_welfare_change": sum(
            f["final_utility"] - s["utility"]
            for f, s in zip(final_utilities, starting_utilities)
        ),
    }
    logger.set_summary(summary)
 
    logger.append_transcript(
        f"\n{'=' * 60}\nFINAL SUMMARY\n{'=' * 60}\n"
        f"Trades accepted: {total_accepted}\n"
        f"Trades rejected: {total_rejected}\n"
    )
    for f, s in zip(final_utilities, starting_utilities):
        delta = f["final_utility"] - s["utility"]
        logger.append_transcript(
            f"  {f['display_name']}: "
            f"U {s['utility']:.4f} → {f['final_utility']:.4f} "
            f"(Δ {delta:+.4f}), inv={f['final_inventory']}\n"
        )
 
    logger.log_event("experiment_completed", {"summary": summary})
    logger.finalize()
    return logger
 
 
# ---------------------------------------------------------------------------
# Mock experiment entry point
# ---------------------------------------------------------------------------
 
def run_mock_experiment(cfg: LoadedConfig) -> RunLogger:
    """Run the full experiment with deterministic mock agents (no API calls)."""
    exp_cfg = cfg.experiment
 
    logger = RunLogger.create(
        output_dir=exp_cfg.logging.output_dir,
        experiment_name=exp_cfg.experiment.name,
        filenames=dict(exp_cfg.logging.filenames),
    )
    logger.log_event("experiment_started", {
        "experiment_name": exp_cfg.experiment.name,
        "mode": "mock",
        "num_players": exp_cfg.market.num_players,
        "goods": exp_cfg.market.goods,
        "seed": exp_cfg.experiment.seed,
    })
    logger.save_config_files(
        config_paths=[
            "configs/experiment.yaml",
            "configs/models.yaml",
            "configs/players.yaml",
            "configs/prompts.yaml",
        ],
        loaded_config=cfg,
    )
 
    # The mock agent reads from player.inventory directly, which is the
    # config-loaded starting inventory. The loop manages its own inventories
    # dict separately, so we bind a snapshot at call time via default args.
    starting_inventories = {p.id: dict(p.inventory) for p in cfg.players.players}
 
    # Market bulletin board (broadcast condition). Mock agents don't read
    # prompts, so this won't change their behaviour, but it keeps the
    # bulletin mechanics exercised and visible in the transcript.
    broadcast = cfg.experiment.mechanism.broadcast_completed_trades
    bulletin_board: Optional[List[str]] = [] if broadcast else None

    def negotiation_fn(player, partner, negotiation_history, turn_index,
                       round_index, pair_id, _inv=starting_inventories, **_ignored):
        # Use the live inventories snapshot passed via the loop's closure
        return mock_negotiation_action(
            player_id=player.id,
            inventory=_inv.get(player.id, dict(player.inventory)),
            weights=dict(player.utility_weights),
            round_index=round_index,
            partner_id=partner.id,
            negotiation_history=negotiation_history,
            turn_index=turn_index,
        )
 
    def commitment_fn(player, proposed_trade, round_index, pair_id,
                      _inv=starting_inventories, **_ignored):
        return mock_commitment_decision(
            player_id=player.id,
            inventory=_inv.get(player.id, dict(player.inventory)),
            weights=dict(player.utility_weights),
            proposed_trade=proposed_trade,
        )
 
    def probe_fn(player, round_index, **_ignored):
        return mock_preference_probe(
            player_id=player.id,
            weights=dict(player.utility_weights),
        )

    def shadow_commitment_fn(player, proposed_trade, round_index, shadow_id,
                             display_order=None, _inv=starting_inventories, **_ignored):
        return mock_commitment_decision(
            player_id=player.id,
            inventory=_inv.get(player.id, dict(player.inventory)),
            weights=dict(player.utility_weights),
            proposed_trade=proposed_trade,
        )

    # Wrap execute_trade to keep the mock agent's inventory snapshot in sync
    # (the mock agent uses utility math on the live inventory for decisions)
    original_execute = execute_trade
 
    def synced_execute(proposer_id, responder_id, proposed_trade, inventories):
        original_execute(proposer_id, responder_id, proposed_trade, inventories)
        starting_inventories[proposer_id]  = dict(inventories[proposer_id])
        starting_inventories[responder_id] = dict(inventories[responder_id])
 
    import runner as _self
    _self.execute_trade = synced_execute
 
    try:
        result = _run_experiment_loop(
            cfg=cfg,
            logger=logger,
            mode="mock",
            negotiation_fn=negotiation_fn,
            commitment_fn=commitment_fn,
            probe_fn=probe_fn,
            shadow_commitment_fn=shadow_commitment_fn,
            bulletin_board=bulletin_board if broadcast else None,
        )
    finally:
        _self.execute_trade = original_execute

    print(f"\nMock run complete. Output written to: {logger.run_dir}")
    return result
 
 
# ---------------------------------------------------------------------------
# GPT experiment entry point
# ---------------------------------------------------------------------------
 
def run_gpt_experiment(cfg: LoadedConfig) -> RunLogger:
    """
    Run the full experiment with real GPT-5.4 API calls.
 
    Requires:
      - OPENAI_API_KEY in .env (or environment)
      - prompts.yaml loaded (require_prompts=True, the default)
      - model_id 'gpt' present and enabled in models.yaml
    """
    from openai import OpenAI
    from openai_agent import (
        gpt_commitment_decision,
        gpt_negotiation_action,
        gpt_preference_probe,
    )
 
    exp_cfg = cfg.experiment
    goods   = exp_cfg.market.goods
    prompts = cfg.prompts
 
    if prompts is None:
        raise ValueError("prompts.yaml must be loaded for a GPT run.")
 
    gpt_spec = cfg.models.get_model("gpt")
 
    if not gpt_spec.api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file as OPENAI_API_KEY=sk-..."
        )
 
    client = OpenAI(api_key=gpt_spec.api_key)
 
    logger = RunLogger.create(
        output_dir=exp_cfg.logging.output_dir,
        experiment_name=exp_cfg.experiment.name,
        filenames=dict(exp_cfg.logging.filenames),
    )
    logger.log_event("experiment_started", {
        "experiment_name": exp_cfg.experiment.name,
        "mode": "gpt",
        "model": gpt_spec.model,
        "num_players": exp_cfg.market.num_players,
        "goods": goods,
        "seed": exp_cfg.experiment.seed,
    })
    logger.save_config_files(
        config_paths=[
            "configs/experiment.yaml",
            "configs/models.yaml",
            "configs/players.yaml",
            "configs/prompts.yaml",
        ],
        loaded_config=cfg,
    )
 
    # Live inventory mirror — the GPT agent needs to see current inventories
    # when building prompts (world state). We sync this after every trade.
    live_inv: Dict[str, Inventory] = {
        p.id: dict(p.inventory) for p in cfg.players.players
    }

    # Shared public market bulletin board for the broadcast condition.
    # run_round appends a bulletin after every accepted trade; the
    # negotiation closure below reads the same list so agents see all
    # market activity so far. Gated by broadcast_completed_trades.
    broadcast = cfg.experiment.mechanism.broadcast_completed_trades
    bulletin_board: Optional[List[str]] = [] if broadcast else None
    num_players = len(cfg.players.players)

    def _round_broadcast(round_index: int) -> bool:
        # Same as `broadcast`, except forced off during washout rounds when
        # washout.disable_broadcast is set (see effective_broadcast).
        return effective_broadcast(exp_cfg, num_players, round_index)

    def negotiation_fn(player, partner, negotiation_history, turn_index,
                       round_index, pair_id, display_order=None):
        # Patch player.inventory so prompt_render sees the live state
        player.inventory = live_inv[player.id]
        round_broadcast = _round_broadcast(round_index)
        return gpt_negotiation_action(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner=partner,
            negotiation_history=negotiation_history,
            turn_index=turn_index,
            model_spec=gpt_spec,
            client=client,
            logger=logger,
            pair_id=pair_id,
            display_order=display_order,
            action_space=exp_cfg.mechanism.action_space,
            board_history=bulletin_board if round_broadcast else None,
            broadcast=round_broadcast,
        )

    def commitment_fn(player, proposed_trade, round_index, pair_id,
                      partner_name="your partner", negotiation_history=None,
                      display_order=None):
        player.inventory = live_inv[player.id]
        round_broadcast = _round_broadcast(round_index)
        return gpt_commitment_decision(
            player=player,
            prompts=prompts,
            goods=goods,
            proposed_trade=proposed_trade,
            model_spec=gpt_spec,
            client=client,
            logger=logger,
            round_index=round_index,
            pair_id=pair_id,
            display_order=display_order,
            partner_name=partner_name,
            negotiation_history=negotiation_history,
            board_history=bulletin_board if round_broadcast else None,
            broadcast=round_broadcast,
        )
 
    def probe_fn(player, round_index, trade_history=None,
                 bulletin_board=None, broadcast=False, display_order=None):
        # Patch live inventory so the probe context shows the up-to-date state.
        player.inventory = live_inv[player.id]
        return gpt_preference_probe(
            player=player,
            prompts=prompts,
            model_spec=gpt_spec,
            client=client,
            logger=logger,
            round_index=round_index,
            display_order=display_order,
            goods=goods,
            trade_history=trade_history,
            board_history=bulletin_board,
            broadcast=broadcast,
        )

    def shadow_commitment_fn(player, proposed_trade, round_index, shadow_id, display_order=None):
        # Same prompt shape as a real commitment decision, generic
        # fictitious partner, tagged "shadow_commitment" in the logs so it's
        # never conflated with a real commitment decision.
        player.inventory = live_inv[player.id]
        round_broadcast = _round_broadcast(round_index)
        return gpt_commitment_decision(
            player=player,
            prompts=prompts,
            goods=goods,
            proposed_trade=proposed_trade,
            model_spec=gpt_spec,
            client=client,
            logger=logger,
            round_index=round_index,
            pair_id=f"shadow_{shadow_id}",
            display_order=display_order,
            partner_name="another trader in the market",
            negotiation_history=None,
            board_history=bulletin_board if round_broadcast else None,
            broadcast=round_broadcast,
            prompt_type="shadow_commitment",
            output_type="shadow_commitment",
        )

    # Wrap execute_trade to keep live_inv in sync with the loop's inventories
    original_execute = execute_trade
 
    def synced_execute(proposer_id, responder_id, proposed_trade, inventories):
        original_execute(proposer_id, responder_id, proposed_trade, inventories)
        live_inv[proposer_id]  = dict(inventories[proposer_id])
        live_inv[responder_id] = dict(inventories[responder_id])
 
    import runner as _self
    _self.execute_trade = synced_execute
 
    try:
        result = _run_experiment_loop(
            cfg=cfg,
            logger=logger,
            mode="gpt",
            negotiation_fn=negotiation_fn,
            commitment_fn=commitment_fn,
            probe_fn=probe_fn,
            shadow_commitment_fn=shadow_commitment_fn,
            bulletin_board=bulletin_board,
        )
    finally:
        _self.execute_trade = original_execute

    print(f"\nGPT run complete. Output written to: {logger.run_dir}")
    return result

# ---------------------------------------------------------------------------
# Random-baseline experiment entry point
# ---------------------------------------------------------------------------

def run_random_experiment(cfg: LoadedConfig) -> RunLogger:
    """
    Baseline run where every agent trades RANDOMLY but FEASIBLY.

    On its first turn of each pair, the active player enumerates every
    feasible 1-for-1 trade — i.e. every (give_good, receive_good) pair where
    the proposer holds at least 1 of give_good, the partner holds at least 1
    of receive_good, and the two goods are different — and proposes one of
    those uniformly at random. The responder always accepts. If no feasible
    1-for-1 trade exists between the two current inventories, the pair
    simply doesn't trade that round.

    The result is the strongest "no intelligence" baseline: every pair makes
    a trade whenever any 1-for-1 swap is possible, but the choice of trade
    is uniformly random with no regard for either side's preferences. No LLM
    is involved and no preference probes are taken.

    The randomness is seeded from cfg.experiment.experiment.seed so a given
    config produces a reproducible random transcript.
    """
    import random as _random

    exp_cfg = cfg.experiment
    goods   = exp_cfg.market.goods

    rng = _random.Random(exp_cfg.experiment.seed)

    logger = RunLogger.create(
        output_dir=exp_cfg.logging.output_dir,
        experiment_name=exp_cfg.experiment.name + "_random",
        filenames=dict(exp_cfg.logging.filenames),
    )
    logger.log_event("experiment_started", {
        "experiment_name": exp_cfg.experiment.name + "_random",
        "mode": "random",
        "num_players": exp_cfg.market.num_players,
        "goods": goods,
        "seed": exp_cfg.experiment.seed,
    })
    logger.save_config_files(
        config_paths=[
            "configs/experiment.yaml",
            "configs/models.yaml",
            "configs/players.yaml",
            "configs/prompts.yaml",
        ],
        loaded_config=cfg,
    )

    # Live inventory mirror so each turn's random offer reflects what BOTH
    # players actually hold at that moment. The proposer reads its own
    # inventory and the partner's so the proposed trade is always feasible
    # for both sides.
    live_inv: Dict[str, Inventory] = {
        p.id: dict(p.inventory) for p in cfg.players.players
    }

    def negotiation_fn(player, partner, negotiation_history, turn_index,
                       round_index, pair_id, **_ignored):
        """First turn: propose a uniformly random feasible 1-for-1 trade.
        Subsequent turns: end the negotiation (single-shot per pair)."""
        if turn_index > 0:
            return {
                "action_type": "no_trade",
                "message_to_partner": "",
                "proposed_trade": None,
                "accept_trade": None,
                "reasoning_summary": "random baseline: single-shot per pair",
            }

        my_inv      = live_inv[player.id]
        partner_inv = live_inv[partner.id]
        my_goods      = [g for g, q in my_inv.items()      if q >= 1]
        partner_goods = [g for g, q in partner_inv.items() if q >= 1]

        # Every (give, receive) pair that is feasible right now: proposer
        # holds the give-good, partner holds the receive-good, and the two
        # are different. Uniform random over this set is the natural
        # "random feasible 1-for-1 trade".
        feasible = [
            (g, h) for g in my_goods for h in partner_goods if g != h
        ]
        if not feasible:
            return {
                "action_type": "no_trade",
                "message_to_partner": "",
                "proposed_trade": None,
                "accept_trade": None,
                "reasoning_summary": (
                    "random baseline: no feasible 1-for-1 trade exists "
                    "between these two inventories"
                ),
            }
        give_good, receive_good = rng.choice(feasible)
        return {
            "action_type": "offer",
            "message_to_partner": f"Random offer: 1×{give_good} for 1×{receive_good}.",
            "proposed_trade": {"give": {give_good: 1}, "receive": {receive_good: 1}},
            "accept_trade": None,
            "reasoning_summary": (
                "random baseline: uniformly random feasible offer "
                "constructed from both inventories"
            ),
        }

    def commitment_fn(player, proposed_trade, round_index, pair_id, **_ignored):
        """Always accept. The runner has already validated feasibility before
        commitment_fn is called, and random offers are constructed to be
        feasible for both sides, so there is no reason for the baseline to
        decline."""
        return {
            "decision": "accept",
            "reasoning_summary": "random baseline: always accept feasible offers",
        }

    def probe_fn(player, round_index):
        # Probes are disabled for random runs; this only exists to satisfy
        # the signature of _run_experiment_loop.
        raise RuntimeError("probe_fn should not be called in random mode")

    # Keep live_inv in sync with applied trades (same pattern as mock/gpt).
    original_execute = execute_trade

    def synced_execute(proposer_id, responder_id, proposed_trade, inventories):
        original_execute(proposer_id, responder_id, proposed_trade, inventories)
        live_inv[proposer_id]  = dict(inventories[proposer_id])
        live_inv[responder_id] = dict(inventories[responder_id])

    import runner as _self
    _self.execute_trade = synced_execute

    try:
        result = _run_experiment_loop(
            cfg=cfg,
            logger=logger,
            mode="random",
            negotiation_fn=negotiation_fn,
            commitment_fn=commitment_fn,
            probe_fn=probe_fn,
            enable_probes=False,
        )
    finally:
        _self.execute_trade = original_execute

    print(f"\nRandom-baseline run complete. Output written to: {logger.run_dir}")
    return result


# ---------------------------------------------------------------------------
# Probe-only experiment entry point
# ---------------------------------------------------------------------------

def run_probe_only_experiment(
    cfg: LoadedConfig,
    count: int = 10,
    with_context: bool = False,
) -> RunLogger:
    """
    Run only preference probes — no trading, no pairing, no commitment.

    Used to measure preference drift in the LLM in the absence of any
    interaction between agents. Inventories stay at their starting values
    throughout. Each player is probed ``count`` times.

    Two modes controlled by ``with_context``:

    **without context** (default, ``with_context=False``):
        Each probe is an independent fresh API call — ``[system, user]``.
        The model has no memory of its own prior answers. This isolates
        whatever drift comes from the model's own non-determinism.

    **with context** (``with_context=True``):
        Each probe extends a growing per-player conversation:
        ``[system, user_1, assistant_1, user_2, assistant_2, ..., user_N]``.
        The model can see every answer it has given so far and may anchor
        to or diverge from them. This tests whether exposure to prior
        self-responses amplifies or suppresses drift.

    Results land in ``preference_probes.json`` with labels
    ``probe_1`` ... ``probe_N`` so the existing analysis tooling works
    without modification.
    """
    from openai import OpenAI
    from openai_agent import (
        gpt_commitment_decision,
        gpt_preference_probe,
        gpt_preference_probe_contextual,
    )
    from prompt_render import build_preference_probe_messages

    if count < 1:
        raise ValueError(f"probe-only count must be >= 1, got {count}")

    exp_cfg = cfg.experiment
    prompts = cfg.prompts

    if prompts is None:
        raise ValueError("prompts.yaml must be loaded for a probe-only run.")

    gpt_spec = cfg.models.get_model("gpt")
    if not gpt_spec.api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file as OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=gpt_spec.api_key)

    mode_tag    = "probe_only_context" if with_context else "probe_only"
    mode_label  = "with context" if with_context else "no context"
    exp_name    = exp_cfg.experiment.name + f"_{mode_tag}"

    logger = RunLogger.create(
        output_dir=exp_cfg.logging.output_dir,
        experiment_name=exp_name,
        filenames=dict(exp_cfg.logging.filenames),
    )
    logger.log_event("experiment_started", {
        "experiment_name": exp_name,
        "mode": mode_tag,
        "model": gpt_spec.model,
        "num_players": exp_cfg.market.num_players,
        "probe_count": count,
        "with_context": with_context,
        "seed": exp_cfg.experiment.seed,
    })
    logger.save_config_files(
        config_paths=[
            "configs/experiment.yaml",
            "configs/models.yaml",
            "configs/players.yaml",
            "configs/prompts.yaml",
        ],
        loaded_config=cfg,
    )

    players = cfg.players.players
    inventories: Dict[str, Inventory] = {p.id: dict(p.inventory) for p in players}

    display_order = order_for_run(exp_cfg.experiment.run_index)
    logger.log_event("display_order_assigned", {
        "run_index": exp_cfg.experiment.run_index,
        "display_order": display_order,
    })

    logger.append_transcript(
        "STARTING INVENTORIES (no trading in probe-only mode)\n"
        + "=" * 60 + "\n"
    )
    for player in players:
        summary = player_starting_utility_summary(player, exp_cfg.market.goods, shift=1.0)
        logger.append_transcript(
            f"  {player.display_name} ({player.id}): "
            f"inv={summary['inventory']}, U={summary['utility']:.4f}\n"
        )

    logger.append_transcript(
        f"\n[PROBE-ONLY RUN: {count} repeated probes per player, "
        f"no inter-agent interaction, context={with_context}]\n"
    )
    if with_context:
        logger.append_transcript(
            "[Context mode: each probe extends the player's own growing "
            "conversation — the model sees all its prior answers.]\n"
        )
    else:
        logger.append_transcript(
            "[No-context mode: each probe is an independent fresh call — "
            "the model has no memory of prior answers.]\n"
        )

    # Per-player conversation history, only used in with_context mode.
    # Each entry is a pair of {"role":"user",...} + {"role":"assistant",...}
    # messages from the previous iteration.
    player_histories: Dict[str, List[Dict[str, str]]] = {
        p.id: [] for p in players
    }

    def shadow_commitment_fn(player, proposed_trade, round_index, shadow_id, display_order=None):
        return gpt_commitment_decision(
            player=player,
            prompts=prompts,
            goods=exp_cfg.market.goods,
            proposed_trade=proposed_trade,
            model_spec=gpt_spec,
            client=client,
            logger=logger,
            round_index=round_index,
            pair_id=f"shadow_{shadow_id}",
            display_order=display_order,
            partner_name="another trader in the market",
            negotiation_history=None,
            board_history=None,
            broadcast=False,
            prompt_type="shadow_commitment",
            output_type="shadow_commitment",
        )

    shadow_specs = generate_shadow_trade_specs_from_config(cfg)

    for i in range(1, count + 1):
        logger.append_transcript(f"\n--- Probe iteration {i} of {count} ---\n")

        for player in players:
            if with_context:
                # Build the user message so we can append it to history.
                base_msgs    = build_preference_probe_messages(player, prompts, display_order)
                user_msg     = base_msgs[1]

                parsed, raw  = gpt_preference_probe_contextual(
                    player=player,
                    prompts=prompts,
                    model_spec=gpt_spec,
                    client=client,
                    logger=logger,
                    round_index=i,
                    prior_history=player_histories[player.id],
                    display_order=display_order,
                )
                # Append this Q&A turn so the next iteration sees it.
                player_histories[player.id].extend([
                    user_msg,
                    {"role": "assistant", "content": raw},
                ])
            else:
                parsed = gpt_preference_probe(
                    player=player,
                    prompts=prompts,
                    model_spec=gpt_spec,
                    client=client,
                    logger=logger,
                    round_index=i,
                    display_order=display_order,
                )

            # Log and write to transcript (same path for both modes).
            current_inventory = dict(inventories[player.id])
            predicted_marginal_utility = predicted_marginal_utilities(
                inventory=current_inventory,
                weights=player.utility_weights,
                goods=exp_cfg.market.goods,
                shift=exp_cfg.utility.terminal_utility.shift,
            )

            shadow_trade_records = run_shadow_trades_for_player(
                player=player,
                inventory=current_inventory,
                round_index=i,
                label=f"probe_{i}",
                cfg=cfg,
                logger=logger,
                shadow_commitment_fn=shadow_commitment_fn,
                specs=shadow_specs,
                display_order=display_order,
            )

            probe_record = {
                "round_index": i,
                "probe_label": f"probe_{i}",
                "player_id": player.id,
                "display_name": player.display_name,
                "current_inventory": current_inventory,
                "predicted_marginal_utility": predicted_marginal_utility,
                "display_order": list(display_order),
                "response": parsed,
                "with_context": with_context,
                "shadow_trades": shadow_trade_records,
            }
            logger.log_preference_probe(probe_record)
            logger.append_transcript(
                f"[PROBE probe_{i}] {player.display_name}:\n"
                f"    current_inventory={current_inventory}\n"
                f"    predicted_marginal_utility={predicted_marginal_utility}\n"
                f"    ratings_inventory={parsed['ratings_inventory']}\n"
                f"    ratings_general={parsed['ratings_general']}\n"
                f"    desired_bundle={parsed['desired_bundle']}\n"
            )

    summary = {
        "experiment_name": exp_name,
        "mode": mode_tag,
        "with_context": with_context,
        "num_players": exp_cfg.market.num_players,
        "probe_count": count,
        "model": gpt_spec.model,
        "seed": exp_cfg.experiment.seed,
    }
    logger.set_summary(summary)
    logger.log_event("experiment_completed", {"summary": summary})
    logger.finalize()

    print(f"\nProbe-only run ({mode_label}, {count} probes per player) complete. "
          f"Output written to: {logger.run_dir}")
    return logger