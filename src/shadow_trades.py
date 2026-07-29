"""
shadow_trades.py

Generates and evaluates "shadow" commitment offers: hypothetical one-off
trade proposals sent to every player alongside a preference probe, framed
exactly like a real commitment decision.

Why this exists:
Real negotiated trades only cover whatever bundle two paired agents happen
to land on, which can leave some goods (e.g. a good that's rarely scarce
enough to trade for) with very few observed accept/reject data points. A
shadow trade sidesteps that by directly asking "would you accept this
specific offer?" for a configurable battery of offers, without waiting for
one to arise organically in negotiation.

Design constraints (see ShadowTradesConfig in config.py):
- Purely hypothetical: never applied to inventories, never appended to
  negotiation history, trade history, or the market bulletin board, so it
  cannot leak into any later prompt.
- Modular: which goods, which quantities, which directions, and which
  players are all config-driven.
- Only feasible offers are sent: the player must actually hold enough of
  whatever they'd be asked to give up, and the offer must satisfy the
  experiment's own trade-shape rules (action space, min/max units per side).

This module does not call any LLM APIs and does not mutate game state.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


Inventory = Mapping[str, int]


def _trade_shape_allowed(
    give: Mapping[str, int],
    receive: Mapping[str, int],
    action_space: str,
    min_units_per_side: int,
    max_units_per_side: int,
) -> bool:
    """
    Mirrors the shape checks in runner.validate_trade (minus the two-sided
    inventory check, since a shadow trade's counterparty is fictitious).
    """
    give_total = sum(give.values())
    receive_total = sum(receive.values())

    if give_total < min_units_per_side or receive_total < min_units_per_side:
        return False
    if give_total > max_units_per_side or receive_total > max_units_per_side:
        return False

    if action_space in ("one_for_one", "small_bundle"):
        if len(give) != 1 or len(receive) != 1:
            return False
    if action_space == "one_for_one":
        if give_total != 1 or receive_total != 1:
            return False

    return True


def generate_shadow_trade_specs(
    goods: Iterable[str],
    focal_goods: Iterable[str],
    focal_quantity: int,
    counter_quantities: Iterable[int],
    directions: Iterable[str],
    action_space: str,
    min_units_per_side: int,
    max_units_per_side: int,
    counter_goods: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build the (player-independent) battery of candidate shadow trade specs.

    Each spec is expressed from the player's own perspective:
      give    = what the player would give up
      receive = what the player would receive
    and carries direction/focal/counter metadata for logging.

    "acquire" specs offer the player the focal good in exchange for the
    counter good; "surrender" specs offer the counter good in exchange for
    the focal good. Specs that don't fit the experiment's trade-shape rules
    (action space, min/max units per side) are skipped entirely — feasibility
    against a specific player's inventory is checked separately.
    """
    goods = list(goods)
    all_directions = set(directions)

    specs: List[Dict[str, Any]] = []
    for focal in focal_goods:
        candidate_counters = (
            list(counter_goods) if counter_goods is not None
            else [g for g in goods if g != focal]
        )
        for counter in candidate_counters:
            if counter == focal:
                continue
            for qty in counter_quantities:
                if "acquire" in all_directions:
                    give = {counter: qty}
                    receive = {focal: focal_quantity}
                    if _trade_shape_allowed(
                        give, receive, action_space, min_units_per_side, max_units_per_side
                    ):
                        specs.append({
                            "shadow_id": f"acquire_{focal_quantity}{focal}_for_{qty}{counter}",
                            "focal_good": focal,
                            "counter_good": counter,
                            "direction": "acquire",
                            "give": give,
                            "receive": receive,
                        })
                if "surrender" in all_directions:
                    give = {focal: focal_quantity}
                    receive = {counter: qty}
                    if _trade_shape_allowed(
                        give, receive, action_space, min_units_per_side, max_units_per_side
                    ):
                        specs.append({
                            "shadow_id": f"surrender_{focal_quantity}{focal}_for_{qty}{counter}",
                            "focal_good": focal,
                            "counter_good": counter,
                            "direction": "surrender",
                            "give": give,
                            "receive": receive,
                        })

    return specs


def generate_shadow_trade_specs_from_config(cfg: Any) -> List[Dict[str, Any]]:
    """Convenience wrapper that pulls all parameters from a LoadedConfig."""
    st_cfg = cfg.experiment.preference_drift.shadow_trades
    return generate_shadow_trade_specs(
        goods=cfg.experiment.market.goods,
        focal_goods=st_cfg.focal_goods,
        focal_quantity=st_cfg.focal_quantity,
        counter_quantities=st_cfg.counter_quantities,
        directions=st_cfg.directions,
        action_space=cfg.experiment.mechanism.action_space,
        min_units_per_side=cfg.experiment.trade_rules.min_units_per_side,
        max_units_per_side=cfg.experiment.trade_rules.max_units_per_side,
        counter_goods=st_cfg.counter_goods,
    )


def is_player_eligible(player_id: str, player_ids: Optional[Iterable[str]]) -> bool:
    """player_ids=None means every player is eligible."""
    return player_ids is None or player_id in set(player_ids)


def feasible_specs_for_inventory(
    specs: Iterable[Mapping[str, Any]],
    inventory: Inventory,
) -> List[Dict[str, Any]]:
    """
    Keep only specs the player can actually fulfil: they must currently hold
    at least as much of each "give" good as the spec asks for. The
    fictitious counterparty is assumed to always have the "receive" side
    available, since no real partner inventory is involved.
    """
    feasible = []
    for spec in specs:
        give = spec["give"]
        if all(inventory.get(good, 0) >= qty for good, qty in give.items()):
            feasible.append(dict(spec))
    return feasible


def build_shadow_proposed_trade(spec: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Convert a player-perspective spec into a proposer-perspective
    proposed_trade dict, matching the convention used everywhere else
    (execute_trade, render_commitment_prompt, etc.):

        proposed_trade["give"]    = what the (fictitious) proposer gives
                                     = what the player would receive
        proposed_trade["receive"] = what the proposer receives
                                     = what the player would give up
    """
    return {
        "give": dict(spec["receive"]),
        "receive": dict(spec["give"]),
    }
