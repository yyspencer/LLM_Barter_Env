"""
utility.py

Utility functions for the LLM barter experiment.

Current purpose:
- Compute each player's latent terminal utility from inventory + utility weights.
- Use shifted Cobb-Douglas by default:
      U_i(x) = product_g (x_g + shift) ^ alpha_ig
- Provide helpers for utility deltas and human-readable summaries.

This module does not call any LLM APIs and does not mutate game state.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional


Inventory = Mapping[str, int]
Weights = Mapping[str, float]


def validate_inventory_and_weights(
    inventory: Inventory,
    weights: Weights,
    goods: Optional[Iterable[str]] = None,
    require_weight_sum_one: bool = False,
    tolerance: float = 1e-6,
) -> None:
    """
    Validate that inventory and weights are compatible.
    """
    inventory_keys = set(inventory.keys())
    weight_keys = set(weights.keys())

    if goods is not None:
        goods_set = set(goods)

        missing_inventory = goods_set - inventory_keys
        extra_inventory = inventory_keys - goods_set
        if missing_inventory or extra_inventory:
            raise ValueError(
                "Inventory goods mismatch. "
                f"Missing: {sorted(missing_inventory)}, "
                f"Extra: {sorted(extra_inventory)}"
            )

        missing_weights = goods_set - weight_keys
        extra_weights = weight_keys - goods_set
        if missing_weights or extra_weights:
            raise ValueError(
                "Utility weight goods mismatch. "
                f"Missing: {sorted(missing_weights)}, "
                f"Extra: {sorted(extra_weights)}"
            )
    else:
        if inventory_keys != weight_keys:
            raise ValueError(
                "Inventory and utility weights must contain the same goods. "
                f"Inventory goods: {sorted(inventory_keys)}, "
                f"Weight goods: {sorted(weight_keys)}"
            )

    for good, amount in inventory.items():
        if not isinstance(amount, int):
            raise TypeError(f"Inventory quantity for {good} must be int, got {type(amount)}.")
        if amount < 0:
            raise ValueError(f"Inventory quantity for {good} must be nonnegative, got {amount}.")

    # Utility weights may be any value (any sum, zero, or negative). No
    # constraint is enforced here; the only structural requirement is that
    # weights cover exactly the configured goods, checked above.
    if require_weight_sum_one:
        total = sum(weights.values())
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Utility weights must sum to 1. Current sum={total}.")


def shifted_cobb_douglas(
    inventory: Inventory,
    weights: Weights,
    shift: float = 1.0,
    goods: Optional[Iterable[str]] = None,
    validate: bool = True,
) -> float:
    """
    Compute shifted Cobb-Douglas utility:

        U(x) = product_g (x_g + shift) ^ alpha_g

    The shift prevents utility from becoming zero when an agent has zero units
    of a positively weighted good.
    """
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}.")

    if validate:
        validate_inventory_and_weights(
            inventory=inventory,
            weights=weights,
            goods=goods,
            require_weight_sum_one=False,
        )

    selected_goods = list(goods) if goods is not None else list(weights.keys())

    utility = 1.0
    for good in selected_goods:
        amount = inventory[good]
        weight = weights[good]
        utility *= (amount + shift) ** weight

    return utility


def log_shifted_cobb_douglas(
    inventory: Inventory,
    weights: Weights,
    shift: float = 1.0,
    goods: Optional[Iterable[str]] = None,
    validate: bool = True,
) -> float:
    """
    Compute log shifted Cobb-Douglas utility:

        log U(x) = sum_g alpha_g * log(x_g + shift)
    """
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}.")

    if validate:
        validate_inventory_and_weights(
            inventory=inventory,
            weights=weights,
            goods=goods,
            require_weight_sum_one=False,
        )

    selected_goods = list(goods) if goods is not None else list(weights.keys())

    total = 0.0
    for good in selected_goods:
        amount = inventory[good]
        weight = weights[good]
        total += weight * math.log(amount + shift)

    return total


def utility_delta(
    before_inventory: Inventory,
    after_inventory: Inventory,
    weights: Weights,
    shift: float = 1.0,
    goods: Optional[Iterable[str]] = None,
) -> float:
    """
    Compute terminal utility change:

        delta = U(after) - U(before)
    """
    before = shifted_cobb_douglas(
        inventory=before_inventory,
        weights=weights,
        shift=shift,
        goods=goods,
        validate=True,
    )
    after = shifted_cobb_douglas(
        inventory=after_inventory,
        weights=weights,
        shift=shift,
        goods=goods,
        validate=True,
    )
    return after - before


def log_utility_delta(
    before_inventory: Inventory,
    after_inventory: Inventory,
    weights: Weights,
    shift: float = 1.0,
    goods: Optional[Iterable[str]] = None,
) -> float:
    """
    Compute log utility change:

        delta_log = log U(after) - log U(before)
    """
    before = log_shifted_cobb_douglas(
        inventory=before_inventory,
        weights=weights,
        shift=shift,
        goods=goods,
        validate=True,
    )
    after = log_shifted_cobb_douglas(
        inventory=after_inventory,
        weights=weights,
        shift=shift,
        goods=goods,
        validate=True,
    )
    return after - before


def predicted_marginal_utilities(
    inventory: Inventory,
    weights: Weights,
    goods: Optional[Iterable[str]] = None,
    shift: float = 1.0,
) -> Dict[str, float]:
    """
    Predicted marginal utility of one additional unit of each good, read off
    the latent Cobb-Douglas utility function:

        delta_U_g = U(..., x_g + 1, ...) - U(..., x_g, ...)

    This is the objective counterpart to a preference probe's elicited
    valuation for that good. Comparing the two per good, per probe, is what
    lets later analysis tell rational inventory-driven revaluation apart
    from valuation shifts the inventory alone doesn't explain.
    """
    selected_goods = list(goods) if goods is not None else list(weights.keys())
    base = shifted_cobb_douglas(inventory, weights, shift=shift, goods=selected_goods)

    deltas: Dict[str, float] = {}
    for good in selected_goods:
        bumped_inventory = dict(inventory)
        bumped_inventory[good] = bumped_inventory.get(good, 0) + 1
        after = shifted_cobb_douglas(bumped_inventory, weights, shift=shift, goods=selected_goods)
        deltas[good] = after - base

    return deltas


def apply_trade_to_inventory(
    inventory: Inventory,
    give: Mapping[str, int],
    receive: Mapping[str, int],
    goods: Optional[Iterable[str]] = None,
    enforce_nonnegative: bool = True,
) -> Dict[str, int]:
    """
    Return a new inventory after applying a proposed trade.

    This does not mutate the original inventory.
    """
    new_inventory = dict(inventory)

    valid_goods = set(goods) if goods is not None else set(new_inventory.keys())

    for good, amount in give.items():
        if good not in valid_goods:
            raise ValueError(f"Unknown good in give: {good}")
        if not isinstance(amount, int) or amount < 0:
            raise ValueError(f"Give quantity for {good} must be nonnegative int, got {amount}.")
        new_inventory[good] = new_inventory.get(good, 0) - amount

    for good, amount in receive.items():
        if good not in valid_goods:
            raise ValueError(f"Unknown good in receive: {good}")
        if not isinstance(amount, int) or amount < 0:
            raise ValueError(f"Receive quantity for {good} must be nonnegative int, got {amount}.")
        new_inventory[good] = new_inventory.get(good, 0) + amount

    if enforce_nonnegative:
        negatives = {g: q for g, q in new_inventory.items() if q < 0}
        if negatives:
            raise ValueError(f"Trade would create negative inventory: {negatives}")

    if goods is not None:
        for good in goods:
            new_inventory.setdefault(good, 0)

    return new_inventory


def format_inventory(inventory: Inventory, goods: Optional[Iterable[str]] = None) -> str:
    """Format an inventory as 'A x 1, B x 0, C x 3'."""
    selected_goods = list(goods) if goods is not None else list(inventory.keys())
    return ", ".join(f"{good} x {inventory.get(good, 0)}" for good in selected_goods)


def format_weights(weights: Weights, goods: Optional[Iterable[str]] = None) -> str:
    """Format utility weights as 'A:0.70, B:0.20, C:0.10'."""
    selected_goods = list(goods) if goods is not None else list(weights.keys())
    return ", ".join(f"{good}:{weights.get(good, 0):.2f}" for good in selected_goods)


def player_starting_utility_summary(
    player,
    goods: Iterable[str],
    shift: float = 1.0,
) -> Dict[str, object]:
    """
    Build a small summary dict for a loaded PlayerSpec-like object.

    This is intentionally duck-typed so it works with the Pydantic PlayerSpec
    from config.py without importing it.
    """
    utility = shifted_cobb_douglas(
        inventory=player.inventory,
        weights=player.utility_weights,
        shift=shift,
        goods=goods,
        validate=True,
    )
    log_utility = log_shifted_cobb_douglas(
        inventory=player.inventory,
        weights=player.utility_weights,
        shift=shift,
        goods=goods,
        validate=True,
    )

    return {
        "player_id": player.id,
        "display_name": player.display_name,
        "role": getattr(player, "role", None),
        "model_id": player.model_id,
        "inventory": dict(player.inventory),
        "utility_weights": dict(player.utility_weights),
        "utility": utility,
        "log_utility": log_utility,
    }


if __name__ == "__main__":
    # this part is just for demo
    goods = ["A", "B", "C"]
    inventory = {"A": 1, "B": 0, "C": 3}
    weights = {"A": 0.7, "B": 0.2, "C": 0.1}

    print("Inventory:", format_inventory(inventory, goods))
    print("Weights:", format_weights(weights, goods))
    print("Utility:", shifted_cobb_douglas(inventory, weights, goods=goods))
    print("Log utility:", log_shifted_cobb_douglas(inventory, weights, goods=goods))