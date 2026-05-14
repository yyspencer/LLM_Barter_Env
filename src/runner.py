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
 
Current support: mock_run mode only (no real LLM API calls).
Real provider support will be added once the mock pipeline is verified.
 
Entry points:
  run_mock_experiment(cfg)   -- runs with deterministic mock agents
"""
 
from __future__ import annotations
 
import copy
from typing import Any, Dict, List, Mapping, Optional, Tuple
 
from config import LoadedConfig
from logger import RunLogger, build_pair_id, utc_timestamp
from mock_agent import (
    mock_commitment_decision,
    mock_negotiation_action,
    mock_preference_probe,
)
from pairing import generate_pairing_rounds
from utility import (
    apply_trade_to_inventory,
    player_starting_utility_summary,
    shifted_cobb_douglas,
)
 
 
# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
 
Inventory = Dict[str, int]
PlayerState = Dict[str, Any]     # live mutable state for one player during a run
 
 
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
 
    Checks:
    - proposed_trade has give/receive keys
    - give and receive contain only known goods
    - quantities are nonnegative integers
    - one_for_one constraint (if action_space == "one_for_one")
    - proposer has sufficient inventory to give
    """
    goods = set(cfg.experiment.market.goods)
    rules = cfg.experiment.trade_rules
 
    if not proposed_trade:
        return False, "proposed_trade is empty or None"
 
    give = proposed_trade.get("give", {})
    receive = proposed_trade.get("receive", {})
 
    if not give or not receive:
        return False, "Trade must have non-empty give and receive sides"
 
    # Check goods are known
    unknown_give = set(give.keys()) - goods
    unknown_receive = set(receive.keys()) - goods
    if unknown_give or unknown_receive:
        return False, f"Unknown goods: give={unknown_give}, receive={unknown_receive}"
 
    # Check quantities are nonnegative integers
    for side_name, side in [("give", give), ("receive", receive)]:
        for good, qty in side.items():
            if not isinstance(qty, int) or qty < 0:
                return False, f"{side_name}.{good} must be a nonneg int, got {qty}"
 
    # one_for_one constraint
    if cfg.experiment.mechanism.action_space == "one_for_one":
        give_total = sum(give.values())
        receive_total = sum(receive.values())
        if give_total != 1 or receive_total != 1:
            return (
                False,
                f"one_for_one requires exactly 1 unit each side; "
                f"got give={give_total}, receive={receive_total}",
            )
        if len(give) != 1 or len(receive) != 1:
            return False, "one_for_one requires exactly one good on each side"
 
    # min/max units per side
    give_total = sum(give.values())
    receive_total = sum(receive.values())
    if give_total < rules.min_units_per_side or receive_total < rules.min_units_per_side:
        return False, f"Trade below min_units_per_side={rules.min_units_per_side}"
    if give_total > rules.max_units_per_side or receive_total > rules.max_units_per_side:
        return False, f"Trade exceeds max_units_per_side={rules.max_units_per_side}"
 
    # Inventory constraint: proposer must be able to give
    if rules.enforce_inventory_constraints:
        proposer_inv = inventories[proposer_id]
        for good, qty in give.items():
            if proposer_inv.get(good, 0) < qty:
                return (
                    False,
                    f"Proposer {proposer_id} cannot give {qty}×{good} "
                    f"(has {proposer_inv.get(good, 0)})",
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
      give  = what proposer gives  = what responder receives
      receive = what proposer receives = what responder gives
    """
    give = proposed_trade["give"]
    receive = proposed_trade["receive"]
 
    inventories[proposer_id] = apply_trade_to_inventory(
        inventory=inventories[proposer_id],
        give=give,
        receive=receive,
        enforce_nonnegative=True,
    )
    inventories[responder_id] = apply_trade_to_inventory(
        inventory=inventories[responder_id],
        give=receive,       # responder gives what proposer receives
        receive=give,       # responder receives what proposer gives
        enforce_nonnegative=True,
    )
 
 
# ---------------------------------------------------------------------------
# Preference probe runner
# ---------------------------------------------------------------------------
 
def run_preference_probes(
    players: List[Any],
    inventories: Dict[str, Inventory],
    round_index: int,
    logger: RunLogger,
    cfg: LoadedConfig,
    mode: str = "mock",
) -> None:
    """
    Run preference elicitation probes for all players.
 
    round_index=0 → pre-run probe
    round_index=-1 → post-run probe
    round_index=N → mid-experiment probe after round N
    """
    label = (
        "pre_run" if round_index == 0
        else "post_run" if round_index == -1
        else f"round_{round_index}"
    )
 
    for player in players:
        if mode == "mock":
            response = mock_preference_probe(
                player_id=player.id,
                weights=dict(player.utility_weights),
            )
        else:
            raise NotImplementedError(f"Preference probe mode '{mode}' not yet supported.")
 
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
# Single pair negotiation
# ---------------------------------------------------------------------------
 
def run_pair_negotiation_mock(
    player_a: Any,
    player_b: Any,
    inventories: Dict[str, Inventory],
    round_index: int,
    pair_id: str,
    cfg: LoadedConfig,
    logger: RunLogger,
) -> Dict[str, Any]:
    """
    Run one pair's negotiation using deterministic mock agents.
 
    Returns a result dict:
      {
        "trade_accepted": bool,
        "proposed_trade": dict | None,
        "proposer_id": str | None,
        "responder_id": str | None,
        "negotiation_log": list,
        "rejection_reason": str | None,
      }
    """
    goods = cfg.experiment.market.goods
    turns_per_agent = cfg.experiment.mechanism.negotiation_turns_per_agent
    max_turns = turns_per_agent * 2   # total back-and-forth turns
 
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
        current_player = players[turn % 2]
        other_player = players[(turn + 1) % 2]
 
        action = mock_negotiation_action(
            player_id=current_player.id,
            inventory=dict(inventories[current_player.id]),
            weights=dict(current_player.utility_weights),
            round_index=round_index,
            partner_id=other_player.id,
            negotiation_history=negotiation_history,
            turn_index=turn,
        )
 
        msg_record = {
            "turn": turn,
            "speaker_id": current_player.id,
            "speaker": current_player.display_name,
            "action_type": action["action_type"],
            "message": action.get("message_to_partner", ""),
            "proposed_trade": action.get("proposed_trade"),
            "accept_trade": action.get("accept_trade"),
        }
        negotiation_history.append(msg_record)
 
        logger.log_event(
            event_type="negotiation_turn",
            payload={
                "pair_id": pair_id,
                "turn": turn,
                "action": action,
            },
            round_index=round_index,
            player_id=current_player.id,
            pair_id=pair_id,
        )
 
        transcript_line = (
            f"  Turn {turn + 1} [{current_player.display_name}] "
            f"({action['action_type']}): {action.get('message_to_partner', '')}\n"
        )
        logger.append_transcript(transcript_line)
 
        action_type = action["action_type"]
 
        # --- Early accept (agent explicitly accepts partner's last offer) ---
        if action_type == "accept" and action.get("accept_trade"):
            # Find the last proposed trade from the other player
            last_offer = None
            for entry in reversed(negotiation_history[:-1]):
                if entry["speaker_id"] == other_player.id and entry.get("proposed_trade"):
                    last_offer = entry["proposed_trade"]
                    break
 
            if last_offer is None:
                result["rejection_reason"] = "accept with no prior offer"
                break
 
            ok, reason = validate_trade(
                proposer_id=other_player.id,
                responder_id=current_player.id,
                proposed_trade=last_offer,
                inventories=inventories,
                cfg=cfg,
            )
            if ok:
                result.update({
                    "trade_accepted": True,
                    "proposed_trade": last_offer,
                    "proposer_id": other_player.id,
                    "responder_id": current_player.id,
                    "rejection_reason": None,
                })
            else:
                result["rejection_reason"] = f"Validation failed on accept: {reason}"
            break
 
        # --- Offer or counteroffer: run commitment phase ---
        if action_type in ("offer", "counteroffer") and action.get("proposed_trade"):
            proposed = action["proposed_trade"]
 
            ok, reason = validate_trade(
                proposer_id=current_player.id,
                responder_id=other_player.id,
                proposed_trade=proposed,
                inventories=inventories,
                cfg=cfg,
            )
            if not ok:
                logger.append_transcript(
                    f"  [INVALID TRADE from {current_player.display_name}: {reason}]\n"
                )
                result["rejection_reason"] = f"Invalid trade: {reason}"
                # Don't break — let negotiation continue
                continue
 
            # Run commitment decision for the responder
            commitment = mock_commitment_decision(
                player_id=other_player.id,
                inventory=dict(inventories[other_player.id]),
                weights=dict(other_player.utility_weights),
                proposed_trade=proposed,
            )
 
            logger.log_event(
                event_type="commitment_decision",
                payload={
                    "pair_id": pair_id,
                    "proposer_id": current_player.id,
                    "responder_id": other_player.id,
                    "proposed_trade": proposed,
                    "decision": commitment["decision"],
                },
                round_index=round_index,
                pair_id=pair_id,
            )
 
            logger.append_transcript(
                f"  [COMMITMENT] {other_player.display_name}: "
                f"{commitment['decision']} — {commitment['reasoning_summary']}\n"
            )
 
            if commitment["decision"] == "accept":
                result.update({
                    "trade_accepted": True,
                    "proposed_trade": proposed,
                    "proposer_id": current_player.id,
                    "responder_id": other_player.id,
                    "rejection_reason": None,
                })
                break
 
            # Rejected — continue negotiating
            continue
 
        # --- no_trade or reject: stop this pair ---
        if action_type in ("no_trade", "reject"):
            result["rejection_reason"] = (
                f"{current_player.display_name} chose {action_type}"
            )
            break
 
    return result
 
 
# ---------------------------------------------------------------------------
# Single round
# ---------------------------------------------------------------------------
 
def run_round_mock(
    round_index: int,
    pairs: List[Tuple[str, str]],
    player_map: Dict[str, Any],
    inventories: Dict[str, Inventory],
    cfg: LoadedConfig,
    logger: RunLogger,
    trade_history: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Run one experiment round.
 
    Returns a list of trade result records for this round.
    """
    goods = cfg.experiment.market.goods
    round_trades: List[Dict[str, Any]] = []
 
    logger.append_transcript(
        f"\n{'=' * 60}\n"
        f"ROUND {round_index}\n"
        f"{'=' * 60}\n"
    )
 
    logger.log_event(
        event_type="round_started",
        payload={"round_index": round_index, "num_pairs": len(pairs)},
        round_index=round_index,
    )
 
    for player_a_id, player_b_id in pairs:
        player_a = player_map[player_a_id]
        player_b = player_map[player_b_id]
        pair_id = build_pair_id(round_index, player_a_id, player_b_id)
 
        logger.append_transcript(
            f"\nPair: {player_a.display_name} vs {player_b.display_name} "
            f"[{pair_id}]\n"
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
 
        pair_result = run_pair_negotiation_mock(
            player_a=player_a,
            player_b=player_b,
            inventories=inventories,
            round_index=round_index,
            pair_id=pair_id,
            cfg=cfg,
            logger=logger,
        )
 
        if pair_result["trade_accepted"]:
            proposed = pair_result["proposed_trade"]
            proposer_id = pair_result["proposer_id"]
            responder_id = pair_result["responder_id"]
 
            inv_before_a = dict(inventories[player_a_id])
            inv_before_b = dict(inventories[player_b_id])
 
            execute_trade(
                proposer_id=proposer_id,
                responder_id=responder_id,
                proposed_trade=proposed,
                inventories=inventories,
            )
 
            util_before_a = shifted_cobb_douglas(
                inv_before_a, dict(player_a.utility_weights), shift=1.0
            )
            util_after_a = shifted_cobb_douglas(
                inventories[player_a_id], dict(player_a.utility_weights), shift=1.0
            )
            util_before_b = shifted_cobb_douglas(
                inv_before_b, dict(player_b.utility_weights), shift=1.0
            )
            util_after_b = shifted_cobb_douglas(
                inventories[player_b_id], dict(player_b.utility_weights), shift=1.0
            )
 
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
                "utility_before": {
                    player_a_id: util_before_a,
                    player_b_id: util_before_b,
                },
                "utility_after": {
                    player_a_id: util_after_a,
                    player_b_id: util_after_b,
                },
            }
 
            logger.log_trade(trade_record)
 
            for pid in [player_a_id, player_b_id]:
                trade_history.setdefault(pid, []).append({
                    "round_index": round_index,
                    "pair_id": pair_id,
                    "partner_id": player_b_id if pid == player_a_id else player_a_id,
                    "partner_name": (
                        player_b.display_name if pid == player_a_id else player_a.display_name
                    ),
                    "decision": "accepted",
                    "give": proposed["give"] if pid == proposer_id else proposed["receive"],
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
 
            logger.append_transcript(
                f"  NO TRADE: {reason}\n"
            )
 
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
# Probe schedule checker
# ---------------------------------------------------------------------------
 
def should_probe(round_index: int, cfg: LoadedConfig) -> bool:
    """Return True if a mid-experiment probe should run after this round."""
    if not cfg.experiment.preference_drift.enabled:
        return False
    sched = cfg.experiment.preference_drift.probe_schedule
    if sched.mode == "interval_rounds":
        return round_index > 0 and round_index % sched.interval_rounds == 0
    return False
 
 
# ---------------------------------------------------------------------------
# Main mock experiment entry point
# ---------------------------------------------------------------------------
 
def run_mock_experiment(cfg: LoadedConfig) -> RunLogger:
    """
    Run the full experiment with deterministic mock agents.
 
    Steps:
      1. Create RunLogger, save config snapshot
      2. Initialise live inventories from players.yaml
      3. Log starting utilities
      4. Pre-run preference probe
      5. Generate full pairing schedule
      6. For each round:
           a. Run probe if scheduled
           b. Run all pairs (negotiation + commitment)
           c. Validate and execute accepted trades
           d. Update inventories
           e. Log round summary
      7. Post-run preference probe
      8. Save final summary
      9. Finalize logger
 
    Returns the RunLogger so the caller can inspect or extend.
    """
    exp_cfg = cfg.experiment
    goods = exp_cfg.market.goods
 
    # 1. Logger setup
    logger = RunLogger.create(
        output_dir=exp_cfg.logging.output_dir,
        experiment_name=exp_cfg.experiment.name,
        filenames=dict(exp_cfg.logging.filenames),
    )
 
    logger.log_event("experiment_started", {
        "experiment_name": exp_cfg.experiment.name,
        "mode": "mock",
        "num_players": exp_cfg.market.num_players,
        "goods": goods,
        "seed": exp_cfg.experiment.seed,
    })
 
    # Save config files (no API keys in output dir)
    logger.save_config_files(
        config_paths=[
            "configs/experiment.yaml",
            "configs/models.yaml",
            "configs/players.yaml",
            "configs/prompts.yaml",
        ],
        loaded_config=cfg,
    )
 
    # 2. Initialise live inventories (deep copy so we don't mutate config)
    players = cfg.players.players
    player_map: Dict[str, Any] = {p.id: p for p in players}
    inventories: Dict[str, Inventory] = {
        p.id: dict(p.inventory) for p in players
    }
    trade_history: Dict[str, List[Dict[str, Any]]] = {}
 
    # 3. Log starting state
    starting_utilities = []
    logger.append_transcript("STARTING INVENTORIES AND UTILITIES\n" + "=" * 60 + "\n")
    for player in players:
        summary = player_starting_utility_summary(player, goods, shift=1.0)
        starting_utilities.append(summary)
        logger.log_event(
            event_type="player_starting_state",
            payload=summary,
            player_id=player.id,
        )
        logger.append_transcript(
            f"  {player.display_name} ({player.id}): "
            f"inv={summary['inventory']}, "
            f"U={summary['utility']:.4f}\n"
        )
 
    # 4. Pre-run preference probe
    if exp_cfg.preference_drift.enabled and exp_cfg.preference_drift.probe_schedule.include_pre_probe:
        logger.append_transcript("\n[PRE-RUN PREFERENCE PROBES]\n")
        run_preference_probes(
            players=players,
            inventories=inventories,
            round_index=0,
            logger=logger,
            cfg=cfg,
            mode="mock",
        )
 
    # 5. Pairing schedule
    player_ids = [p.id for p in players]
    pairing_rounds = generate_pairing_rounds(
        player_ids=player_ids,
        num_rounds=exp_cfg.rounds.max_rounds_override,
        round_multiplier=exp_cfg.rounds.round_multiplier,
        reshuffle_between_cycles=exp_cfg.pairing.reshuffle_between_runs,
        seed=exp_cfg.experiment.seed,
    )
 
    all_trade_records: List[Dict[str, Any]] = []
 
    # 6. Main experiment loop
    for pr in pairing_rounds:
        round_index = pr.round_index
 
        # 6a. Mid-experiment probe
        if should_probe(round_index, cfg):
            label = f"\n[MID-RUN PREFERENCE PROBE — after round {round_index}]\n"
            logger.append_transcript(label)
            run_preference_probes(
                players=players,
                inventories=inventories,
                round_index=round_index,
                logger=logger,
                cfg=cfg,
                mode="mock",
            )
 
        # 6b–e. Run the round
        round_trades = run_round_mock(
            round_index=round_index,
            pairs=pr.pairs,
            player_map=player_map,
            inventories=inventories,
            cfg=cfg,
            logger=logger,
            trade_history=trade_history,
        )
        all_trade_records.extend(round_trades)
 
        # Optional early stop
        stopping = exp_cfg.stopping
        if stopping.stop_if_no_trades_for_n_rounds is not None:
            n = stopping.stop_if_no_trades_for_n_rounds
            recent = all_trade_records[-n * len(pr.pairs):]
            if recent and not any(t["accepted"] for t in recent):
                logger.log_event(
                    event_type="early_stop",
                    payload={"reason": f"No trades for {n} consecutive rounds"},
                    round_index=round_index,
                )
                logger.append_transcript(
                    f"\n[EARLY STOP] No trades for {n} consecutive rounds.\n"
                )
                break
 
    # 7. Post-run preference probe
    if exp_cfg.preference_drift.enabled and exp_cfg.preference_drift.probe_schedule.include_post_probe:
        logger.append_transcript("\n[POST-RUN PREFERENCE PROBES]\n")
        run_preference_probes(
            players=players,
            inventories=inventories,
            round_index=-1,
            logger=logger,
            cfg=cfg,
            mode="mock",
        )
 
    # 8. Final summary
    final_utilities = []
    for player in players:
        u = shifted_cobb_douglas(
            inventories[player.id], dict(player.utility_weights), shift=1.0
        )
        final_utilities.append({
            "player_id": player.id,
            "display_name": player.display_name,
            "final_inventory": dict(inventories[player.id]),
            "final_utility": u,
        })
 
    total_trades = sum(1 for t in all_trade_records if t["accepted"])
    total_no_trade = sum(1 for t in all_trade_records if not t["accepted"])
 
    summary = {
        "experiment_name": exp_cfg.experiment.name,
        "mode": "mock",
        "num_rounds": len(pairing_rounds),
        "num_players": exp_cfg.market.num_players,
        "goods": goods,
        "seed": exp_cfg.experiment.seed,
        "total_trades_accepted": total_trades,
        "total_trades_rejected": total_no_trade,
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
        f"Trades accepted: {total_trades}\n"
        f"Trades rejected: {total_no_trade}\n"
    )
    for f, s in zip(final_utilities, starting_utilities):
        delta = f["final_utility"] - s["utility"]
        logger.append_transcript(
            f"  {f['display_name']}: "
            f"U {s['utility']:.4f} → {f['final_utility']:.4f} "
            f"(Δ {delta:+.4f}), "
            f"inv={f['final_inventory']}\n"
        )
 
    logger.log_event("experiment_completed", {"summary": summary})
 
    # 9. Finalize
    logger.finalize()
 
    print(f"\nMock run complete. Output written to: {logger.run_dir}")
    return logger