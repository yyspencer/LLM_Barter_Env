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
from logger import RunLogger, build_pair_id
from mock_agent import (
    mock_commitment_decision,
    mock_negotiation_action,
    mock_preference_probe,
)
from pairing import generate_pairing_rounds
from prompt_render import format_goods_dict
from utility import (
    apply_trade_to_inventory,
    player_starting_utility_summary,
    shifted_cobb_douglas,
)
 
 
# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
 
Inventory = Dict[str, int]
 
# Agent callables — each returns the appropriate response dict.
NegotiationFn = Callable[..., Dict[str, Any]]
CommitmentFn  = Callable[..., Dict[str, Any]]
ProbeFn       = Callable[..., Dict[str, Any]]
 
 
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
 
    if cfg.experiment.mechanism.action_space == "one_for_one":
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
# Preference probes
# ---------------------------------------------------------------------------
 
def run_preference_probes(
    players: List[Any],
    inventories: Dict[str, Inventory],
    round_index: int,
    logger: RunLogger,
    cfg: LoadedConfig,
    probe_fn: ProbeFn,
) -> None:
    """
    Run preference elicitation probes for all players.
 
    round_index=0  -> pre-run
    round_index=-1 -> post-run
    round_index=N  -> mid-experiment after round N
 
    probe_fn signature: (player, round_index) -> dict
    """
    label = (
        "pre_run"  if round_index == 0
        else "post_run" if round_index == -1
        else f"round_{round_index}"
    )
 
    for player in players:
        response = probe_fn(player=player, round_index=round_index)
 
        probe_record = {
            "round_index": round_index,
            "probe_label": label,
            "player_id": player.id,
            "display_name": player.display_name,
            "current_inventory": dict(inventories[player.id]),
            "response": response,
        }
        logger.log_preference_probe(probe_record)
        logger.append_transcript(
            f"[PROBE {label}] {player.display_name}: "
            f"ratings={response['ratings']}, "
            f"desired={response['desired_bundle_6_units']}\n"
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
 
        # --- Explicit accept of partner's last offer ---
        if action_type == "accept" and action.get("accept_trade"):
            last_offer = None
            for entry in reversed(negotiation_history[:-1]):
                if entry["speaker_id"] == other.id and entry.get("proposed_trade"):
                    last_offer = entry["proposed_trade"]
                    break
 
            if last_offer is None:
                # The partner never made a real offer (e.g. they only
                # described a trade in a message). The accept cannot go
                # through, but this must NOT end the negotiation — just
                # skip this turn and let negotiation continue.
                logger.append_transcript(
                    f"  [ACCEPT IGNORED: {current.display_name} tried to "
                    f"accept, but {other.display_name} has no standing "
                    f"offer. A trade must be made with action_type "
                    f"\"offer\", not described in a message.]\n"
                )
                negotiation_history.append({
                    "turn": turn,
                    "speaker_id": current.id,
                    "speaker": current.display_name,
                    "action_type": "system_note",
                    "message": (
                        f"{current.display_name} attempted to ACCEPT, but "
                        f"there is no standing offer from "
                        f"{other.display_name} to accept. Nothing was "
                        f"traded. To put a real offer on the table an agent "
                        f"must use action_type \"offer\" with a "
                        f"proposed_trade — describing a trade only in a "
                        f"message does NOT create an offer. The negotiation "
                        f"continues."
                    ),
                    "proposed_trade": None,
                    "accept_trade": None,
                    "source": "accept_failed",
                })
                continue
 
            ok, reason = validate_trade(
                proposer_id=other.id,
                responder_id=current.id,
                proposed_trade=last_offer,
                inventories=inventories,
                cfg=cfg,
            )
            if ok:
                result.update({
                    "trade_accepted": True,
                    "proposed_trade": last_offer,
                    "proposer_id": other.id,
                    "responder_id": current.id,
                    "rejection_reason": None,
                })
                break
 
            # The accept referenced a real offer but the resulting trade is
            # infeasible (e.g. the accepter cannot supply their side). Do
            # not end the negotiation — skip the turn and continue.
            logger.append_transcript(
                f"  [ACCEPT IGNORED: validation failed — {reason}]\n"
            )
            negotiation_history.append({
                "turn": turn,
                "speaker_id": current.id,
                "speaker": current.display_name,
                "action_type": "system_note",
                "message": (
                    f"{current.display_name} attempted to ACCEPT "
                    f"{other.display_name}'s offer, but that trade is not "
                    f"feasible ({reason}). Nothing was traded. The "
                    f"negotiation continues; a feasible trade may still be "
                    f"proposed."
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
) -> List[Dict[str, Any]]:
    """Run one experiment round for all pairs. Returns trade result records."""
    round_trades: List[Dict[str, Any]] = []
 
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
# Shared experiment loop
# ---------------------------------------------------------------------------
 
def _run_experiment_loop(
    cfg: LoadedConfig,
    logger: RunLogger,
    mode: str,
    negotiation_fn: NegotiationFn,
    commitment_fn: CommitmentFn,
    probe_fn: ProbeFn,
) -> RunLogger:
    """
    Core loop shared by all run modes.
 
    Callers construct agent callables and pass them in.
    This function owns all scheduling, inventory management, and summary writing.
    """
    exp_cfg = cfg.experiment
    goods   = exp_cfg.market.goods
    players = cfg.players.players
 
    player_map: Dict[str, Any]           = {p.id: p for p in players}
    inventories: Dict[str, Inventory]    = {p.id: dict(p.inventory) for p in players}
    trade_history: Dict[str, List[Dict]] = {}
 
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
 
    # Pre-run probe
    drift_cfg = exp_cfg.preference_drift
    if drift_cfg.enabled and drift_cfg.probe_schedule.include_pre_probe:
        logger.append_transcript("\n[PRE-RUN PREFERENCE PROBES]\n")
        run_preference_probes(
            players=players,
            inventories=inventories,
            round_index=0,
            logger=logger,
            cfg=cfg,
            probe_fn=probe_fn,
        )
 
    # Pairing schedule
    pairing_rounds = generate_pairing_rounds(
        player_ids=[p.id for p in players],
        num_rounds=exp_cfg.rounds.max_rounds_override,
        round_multiplier=exp_cfg.rounds.round_multiplier,
        reshuffle_between_cycles=exp_cfg.pairing.reshuffle_between_runs,
        seed=exp_cfg.experiment.seed,
    )
 
    all_trade_records: List[Dict[str, Any]] = []
    total_rounds = len(pairing_rounds)
 
    for pr in pairing_rounds:
        round_index = pr.round_index
 
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
        )
        all_trade_records.extend(round_trades)
 
        # Mid-run preference probe. This runs AFTER the round's trades have
        # been negotiated and applied to inventories, so a probe labelled
        # "after round N" genuinely reflects the state once round N is
        # complete (not the state going into round N).
        if should_probe(round_index, cfg, total_rounds=total_rounds):
            logger.append_transcript(
                f"\n[MID-RUN PREFERENCE PROBE — after round {round_index}]\n"
            )
            run_preference_probes(
                players=players,
                inventories=inventories,
                round_index=round_index,
                logger=logger,
                cfg=cfg,
                probe_fn=probe_fn,
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
    if drift_cfg.enabled and drift_cfg.probe_schedule.include_post_probe:
        logger.append_transcript("\n[POST-RUN PREFERENCE PROBES]\n")
        run_preference_probes(
            players=players,
            inventories=inventories,
            round_index=-1,
            logger=logger,
            cfg=cfg,
            probe_fn=probe_fn,
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
 
    def negotiation_fn(player, partner, negotiation_history, turn_index,
                       round_index, pair_id, _inv=starting_inventories):
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
                      _inv=starting_inventories):
        return mock_commitment_decision(
            player_id=player.id,
            inventory=_inv.get(player.id, dict(player.inventory)),
            weights=dict(player.utility_weights),
            proposed_trade=proposed_trade,
        )
 
    def probe_fn(player, round_index):
        return mock_preference_probe(
            player_id=player.id,
            weights=dict(player.utility_weights),
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
 
    def negotiation_fn(player, partner, negotiation_history, turn_index,
                       round_index, pair_id):
        # Patch player.inventory so prompt_render sees the live state
        player.inventory = live_inv[player.id]
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
        )
 
    def commitment_fn(player, proposed_trade, round_index, pair_id):
        player.inventory = live_inv[player.id]
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
        )
 
    def probe_fn(player, round_index):
        return gpt_preference_probe(
            player=player,
            prompts=prompts,
            model_spec=gpt_spec,
            client=client,
            logger=logger,
            round_index=round_index,
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
        )
    finally:
        _self.execute_trade = original_execute
 
    print(f"\nGPT run complete. Output written to: {logger.run_dir}")
    return result