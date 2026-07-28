"""
display_order.py

Run-wide counterbalancing of goods display order.

Follows the 8-cycle pattern:
    Run 1  -> A, B, C
    Run 2  -> B, C, A
    Run 3  -> C, A, B
    Run 4  -> A, C, B
    Run 5  -> B, A, C
    Run 6  -> C, B, A
    Run 7  -> A, B, C
    Run 8  -> B, C, A
    (pattern repeats)

The order is fixed for the duration of a single run and applies to every
agent, every probe, every inventory display, and every ordered JSON schema
within that run.
"""

from __future__ import annotations
from typing import List

# Explicit table so intent is auditable; do NOT compute this from permutations,
# because the specification is a specific sequence, not "all 6 permutations".
ORDER_PATTERN: List[List[str]] = [
    ["A", "B", "C"],  # run 1
    ["B", "C", "A"],  # run 2
    ["C", "A", "B"],  # run 3
    ["A", "C", "B"],  # run 4
    ["B", "A", "C"],  # run 5
    ["C", "B", "A"],  # run 6
    ["A", "B", "C"],  # run 7
    ["B", "C", "A"],  # run 8
]


def order_for_run(run_index: int) -> List[str]:
    """
    Return the display order for a given 1-indexed run number.

    Run 9 -> same as Run 1 (pattern cycles with period 8).
    """
    if run_index < 1:
        raise ValueError(f"run_index must be >= 1, got {run_index}")
    return list(ORDER_PATTERN[(run_index - 1) % len(ORDER_PATTERN)])