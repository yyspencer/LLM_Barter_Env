"""
mock_agent.py
 
Deterministic mock agent for testing the full experiment pipeline
without making any real LLM API calls.
 
Strategy (simple but economically sensible):
  - Offer 1 unit of your lowest-weight good you actually own
    in exchange for 1 unit of the highest-weight good you don't
    already have in abundance.
  - Accept any trade that improves your utility (or leaves it flat).
  - Preference probe responses are derived directly from utility weights.
 
This is intentionally minimal. Its only purpose is to let runner.py
exercise the full pipeline end-to-end before real providers are wired in.
"""
 
from __future__ import annotations
 
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple
 
from utility import (
    apply_trade_to_inventory,
    shifted_cobb_douglas,
)
 
 
Inventory = Dict[str, int]
Weights = Dict[str, float]
 
 
# ---------------------------------------------------------------------------
# Core decision helpers
# ---------------------------------------------------------------------------
 
def _goods_by_weight_ascending(weights: Weights) -> List[str]:
    """Return goods sorted from lowest to highest weight."""
    return sorted(weights.keys(), key=lambda g: weights[g])
 
 
def _goods_by_weight_descending(weights: Weights) -> List[str]:
    """Return goods sorted from highest to lowest weight."""
    return sorted(weights.keys(), key=lambda g: weights[g], reverse=True)
 
 
def _pick_give_good(
    inventory: Inventory,
    weights: Weights,
) -> Optional[str]:
    """
    Pick the good to give away: lowest-weight good that we own at least 1 of.
    """
    for good in _goods_by_weight_ascending(weights):
        if inventory.get(good, 0) >= 1:
            return good
    return None
 
 
def _pick_receive_good(
    inventory: Inventory,
    weights: Weights,
    exclude: Optional[str] = None,
) -> Optional[str]:
    """
    Pick the good to ask for: highest-weight good that isn't the one we're giving.
    """
    for good in _goods_by_weight_descending(weights):
        if good == exclude:
            continue
        return good
    return None
 
 
def _trade_improves_utility(
    inventory: Inventory,
    weights: Weights,
    give: Mapping[str, int],
    receive: Mapping[str, int],
    shift: float = 1.0,
) -> bool:
    """
    Return True if applying this trade leaves utility >= current utility.
    """
    try:
        new_inventory = apply_trade_to_inventory(
            inventory=inventory,
            give=give,
            receive=receive,
            enforce_nonnegative=True,
        )
    except ValueError:
        return False
 
    before = shifted_cobb_douglas(inventory, weights, shift=shift)
    after = shifted_cobb_douglas(new_inventory, weights, shift=shift)
    return after >= before
 
 
# ---------------------------------------------------------------------------
# Negotiation action
# ---------------------------------------------------------------------------
 
def mock_negotiation_action(
    player_id: str,
    inventory: Inventory,
    weights: Weights,
    round_index: int,
    partner_id: str,
    negotiation_history: Optional[List[Mapping[str, Any]]] = None,
    turn_index: int = 0,
) -> Dict[str, Any]:
    """
    Return a deterministic negotiation action dict.
 
    On turn 0 (first message), always try to make an offer.
    On subsequent turns, inspect the partner's last message:
      - If they proposed a trade and it improves our utility, accept.
      - If we have a better offer, counter.
      - Otherwise, reject with a no_trade.
 
    Return format mirrors the prompts.yaml negotiation response schema:
      action_type, message_to_partner, proposed_trade, accept_trade,
      reasoning_summary
    """
    negotiation_history = negotiation_history or []
 
    # --- Check if partner made an offer we should respond to ---
    if negotiation_history:
        last_msg = negotiation_history[-1]
        last_speaker = last_msg.get("speaker_id") or last_msg.get("speaker")
        last_trade = last_msg.get("proposed_trade")
 
        if last_speaker != player_id and last_trade:
            give_side = last_trade.get("give", {})
            receive_side = last_trade.get("receive", {})
 
            # Their give becomes our receive, their receive becomes our give
            our_give = receive_side
            our_receive = give_side
 
            if _trade_improves_utility(inventory, weights, our_give, our_receive):
                return {
                    "action_type": "accept",
                    "message_to_partner": (
                        f"I accept your offer. "
                        f"I will give {_fmt(our_give)} for {_fmt(our_receive)}."
                    ),
                    "proposed_trade": last_trade,
                    "accept_trade": True,
                    "reasoning_summary": (
                        "This trade improves my utility given my current inventory."
                    ),
                }
 
    # --- Try to make our own offer ---
    give_good = _pick_give_good(inventory, weights)
    receive_good = _pick_receive_good(inventory, weights, exclude=give_good)
 
    if give_good is None or receive_good is None or give_good == receive_good:
        return {
            "action_type": "no_trade",
            "message_to_partner": "I have nothing useful to offer right now.",
            "proposed_trade": None,
            "accept_trade": False,
            "reasoning_summary": "No beneficial trade is possible given my current inventory.",
        }
 
    proposed = {
        "give": {give_good: 1},
        "receive": {receive_good: 1},
    }
 
    # Sanity check: only offer if it helps or is neutral
    if not _trade_improves_utility(inventory, weights, proposed["give"], proposed["receive"]):
        return {
            "action_type": "no_trade",
            "message_to_partner": "I don't see a mutually beneficial trade at this time.",
            "proposed_trade": None,
            "accept_trade": False,
            "reasoning_summary": "Proposed trade would not improve my utility.",
        }
 
    return {
        "action_type": "offer",
        "message_to_partner": (
            f"I propose: I give you 1×{give_good} in exchange for 1×{receive_good}."
        ),
        "proposed_trade": proposed,
        "accept_trade": None,
        "reasoning_summary": (
            f"I value {receive_good} more than {give_good} given my current weights and inventory."
        ),
    }
 
 
# ---------------------------------------------------------------------------
# Commitment decision
# ---------------------------------------------------------------------------
 
def mock_commitment_decision(
    player_id: str,
    inventory: Inventory,
    weights: Weights,
    proposed_trade: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Return a commitment decision (accept/reject) for a finalised trade proposal.
 
    proposed_trade format:
      { "give": {"A": 1}, "receive": {"B": 1} }
    from the perspective of the player who originally proposed it.
    In the commitment phase, we are the *other* player, so we flip sides.
    """
    give_side = proposed_trade.get("give", {})
    receive_side = proposed_trade.get("receive", {})
 
    # We receive what they give, we give what they receive
    our_give = receive_side
    our_receive = give_side
 
    if _trade_improves_utility(inventory, weights, our_give, our_receive):
        return {
            "decision": "accept",
            "reasoning_summary": "This trade improves my utility.",
        }
 
    return {
        "decision": "reject",
        "reasoning_summary": "This trade does not improve my utility.",
    }
 
 
# ---------------------------------------------------------------------------
# Preference probe
# ---------------------------------------------------------------------------
 
def mock_preference_probe(
    player_id: str,
    weights: Weights,
) -> Dict[str, Any]:
    """
    Return a deterministic preference probe response derived from utility weights.
 
    Ratings are scaled from weights (0–1) onto a 1–10 integer scale.
    The desired bundle allocates 6 units proportionally to weights.
    """
    goods = sorted(weights.keys())
    max_weight = max(weights.values()) if weights else 1.0
 
    ratings: Dict[str, int] = {}
    for good in goods:
        # Scale weight to 1–10
        scaled = (weights[good] / max_weight) * 9 + 1
        ratings[good] = max(1, min(10, round(scaled)))
 
    # Allocate 6 units proportionally
    total_weight = sum(weights.values())
    raw_alloc = {g: (weights[g] / total_weight) * 6 for g in goods}
 
    # Floor and fix remainder
    alloc = {g: int(raw_alloc[g]) for g in goods}
    remainder = 6 - sum(alloc.values())
    # Give remainder to the highest-weight good
    if remainder > 0:
        top_good = max(goods, key=lambda g: weights[g])
        alloc[top_good] += remainder
 
    top = max(goods, key=lambda g: weights[g])
    explanation = (
        f"I prioritise {top} most strongly, "
        f"with weights {', '.join(f'{g}:{weights[g]:.2f}' for g in goods)}."
    )
 
    return {
        "ratings_inventory": dict(ratings),
        "ratings_general": dict(ratings),
        "desired_bundle": alloc,
        "one_sentence_explanation": explanation,
    }
 
 
# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------
 
def _fmt(goods_dict: Mapping[str, int]) -> str:
    if not goods_dict:
        return "nothing"
    return ", ".join(f"{qty}×{good}" for good, qty in goods_dict.items())