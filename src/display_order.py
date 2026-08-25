"""
display_order.py

Run-wide counterbalancing of goods display order.

Follows the 6-cycle pattern (all permutations of A, B, C):
    Run 1  -> A, B, C
    Run 2  -> B, C, A
    Run 3  -> C, A, B
    Run 4  -> A, C, B
    Run 5  -> B, A, C
    Run 6  -> C, B, A
    (pattern repeats: run 7 -> A, B, C, run 8 -> B, C, A, ...)

The order is fixed for the duration of a single run and applies to every
agent, every probe, every inventory display, and every ordered JSON schema
within that run.
"""

from __future__ import annotations
from typing import List

# Explicit table so intent is auditable: the 6 permutations of A, B, C.
ORDER_PATTERN: List[List[str]] = [
    ["A", "B", "C"],  # run 1
    ["B", "C", "A"],  # run 2
    ["C", "A", "B"],  # run 3
    ["A", "C", "B"],  # run 4
    ["B", "A", "C"],  # run 5
    ["C", "B", "A"],  # run 6
]


def order_for_run(run_index: int) -> List[str]:
    """
    Return the display order for a given 1-indexed run number.

    Run 7 -> same as Run 1 (pattern cycles with period 6).
    """
    if run_index < 1:
        raise ValueError(f"run_index must be >= 1, got {run_index}")
    return list(ORDER_PATTERN[(run_index - 1) % len(ORDER_PATTERN)])