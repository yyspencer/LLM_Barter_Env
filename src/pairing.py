"""
pairing.py

Round-robin disjoint-pair scheduler for the barter experiment.

Design goals:
- In each simulation round, each player appears in at most one pair.
- Pair negotiations within a round can therefore be run in parallel.
- Within one full round-robin cycle, no pair is repeated.
- A new cycle only begins after the previous cycle has covered all pairings.
- If the requested number of rounds is smaller than one full cycle, return a
  non-repeating prefix of the schedule.

Terminology:
- For an even number of players, one full round-robin cycle has n - 1 rounds.
- For an odd number of players, one full cycle has n rounds because one player
  receives a bye each round.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


PlayerId = str
Pair = Tuple[PlayerId, PlayerId]
RoundPairs = List[Pair]
Schedule = List[RoundPairs]


@dataclass(frozen=True)
class PairingRound:
    """One simulation round containing disjoint player pairs."""
    round_index: int
    cycle_index: int
    pairs: RoundPairs


def _validate_player_ids(player_ids: Sequence[PlayerId]) -> None:
    if len(player_ids) < 2:
        raise ValueError("At least two players are required to generate pairings.")

    if len(set(player_ids)) != len(player_ids):
        raise ValueError(f"Duplicate player IDs found: {player_ids}")


def full_cycle_length(num_players: int) -> int:
    """
    Return the number of rounds required for one full round-robin cycle.

    Even n:
        n - 1 rounds.
    Odd n:
        n rounds, because one player has a bye each round.
    """
    if num_players < 2:
        raise ValueError("num_players must be at least 2.")

    return num_players - 1 if num_players % 2 == 0 else num_players


def _circle_method_one_cycle(player_ids: Sequence[PlayerId]) -> Schedule:
    """
    Generate one full round-robin cycle using the standard circle method.

    If the number of players is odd, a BYE slot is added internally.
    Pairs involving BYE are omitted.
    """
    _validate_player_ids(player_ids)

    roster: List[Optional[str]] = list(player_ids)

    if len(roster) % 2 == 1:
        roster.append(None)

    num_slots = len(roster)
    num_rounds = num_slots - 1

    schedule: Schedule = []

    for _ in range(num_rounds):
        round_pairs: RoundPairs = []

        for i in range(num_slots // 2):
            p1 = roster[i]
            p2 = roster[num_slots - 1 - i]

            if p1 is not None and p2 is not None:
                round_pairs.append((p1, p2))

        schedule.append(round_pairs)

        # Keep the first player fixed and rotate the rest clockwise.
        roster = [roster[0]] + [roster[-1]] + roster[1:-1]

    return schedule


def _shuffle_player_order(
    player_ids: Sequence[PlayerId],
    rng: random.Random,
) -> List[PlayerId]:
    shuffled = list(player_ids)
    rng.shuffle(shuffled)
    return shuffled


def resolve_total_rounds(
    num_players: int,
    num_rounds: Optional[int] = None,
    round_multiplier: int = 1,
) -> int:
    """
    Resolve the exact total round count generate_round_robin_disjoint_pairs
    would use, without generating the schedule itself. Mirrors the sizing
    rule used there: an explicit num_rounds wins; otherwise it's
    round_multiplier full cycles.
    """
    if round_multiplier < 1:
        raise ValueError("round_multiplier must be >= 1.")
    if num_rounds is not None:
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1 when provided.")
        return num_rounds
    return full_cycle_length(num_players) * round_multiplier


def generate_round_robin_disjoint_pairs(
    player_ids: Sequence[PlayerId],
    num_rounds: Optional[int] = None,
    round_multiplier: int = 1,
    reshuffle_between_cycles: bool = True,
    seed: Optional[int] = None,
) -> Schedule:
    """
    Generate a round-robin disjoint-pair schedule.

    Parameters
    ----------
    player_ids:
        Sequence of player IDs.

    num_rounds:
        Exact number of simulation rounds to return.
        If None, returns round_multiplier full cycles.

    round_multiplier:
        Number of full round-robin cycles to generate when num_rounds is None.
        Ignored if num_rounds is provided.

    reshuffle_between_cycles:
        If True, reshuffle player order before each new full cycle. This changes
        round ordering and pair placement, but each cycle still contains every
        pair exactly once.

    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    Schedule:
        A list of rounds. Each round is a list of disjoint pairs.

    Notes
    -----
    - Within each full cycle, no pair repeats.
    - If num_rounds is shorter than one cycle, the returned partial schedule
      still contains no repeated pairs.
    - If num_rounds exceeds one cycle, additional rounds begin only after the
      previous cycle completes.
    """
    _validate_player_ids(player_ids)

    if round_multiplier < 1:
        raise ValueError("round_multiplier must be >= 1.")

    cycle_len = full_cycle_length(len(player_ids))

    if num_rounds is None:
        total_rounds = cycle_len * round_multiplier
    else:
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1 when provided.")
        total_rounds = num_rounds

    rng = random.Random(seed)
    result: Schedule = []

    cycle_index = 0

    while len(result) < total_rounds:
        if cycle_index == 0 or reshuffle_between_cycles:
            cycle_players = _shuffle_player_order(player_ids, rng)
        else:
            cycle_players = list(player_ids)

        cycle_schedule = _circle_method_one_cycle(cycle_players)

        remaining = total_rounds - len(result)
        result.extend(cycle_schedule[:remaining])

        cycle_index += 1

    return result


def generate_pairing_rounds(
    player_ids: Sequence[PlayerId],
    num_rounds: Optional[int] = None,
    round_multiplier: int = 1,
    reshuffle_between_cycles: bool = True,
    seed: Optional[int] = None,
) -> List[PairingRound]:
    """
    Same as generate_round_robin_disjoint_pairs, but returns metadata objects
    with round_index and cycle_index.
    """
    schedule = generate_round_robin_disjoint_pairs(
        player_ids=player_ids,
        num_rounds=num_rounds,
        round_multiplier=round_multiplier,
        reshuffle_between_cycles=reshuffle_between_cycles,
        seed=seed,
    )

    cycle_len = full_cycle_length(len(player_ids))
    rounds: List[PairingRound] = []

    for idx, pairs in enumerate(schedule):
        rounds.append(
            PairingRound(
                round_index=idx + 1,
                cycle_index=(idx // cycle_len) + 1,
                pairs=pairs,
            )
        )

    return rounds


def flatten_pairs(schedule: Schedule) -> List[Pair]:
    """Flatten a schedule into a list of pairs."""
    return [pair for round_pairs in schedule for pair in round_pairs]


def canonical_pair(pair: Pair) -> Pair:
    """Return a pair in sorted/canonical order for duplicate checking."""
    a, b = pair
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def has_duplicate_pairs_within_cycle(
    schedule: Schedule,
    num_players: int,
) -> bool:
    """
    Check whether any pair repeats within each round-robin cycle.

    This should return False for schedules produced by this module.
    """
    cycle_len = full_cycle_length(num_players)

    for start in range(0, len(schedule), cycle_len):
        cycle = schedule[start:start + cycle_len]
        seen: set[Pair] = set()

        for pair in flatten_pairs(cycle):
            cp = canonical_pair(pair)
            if cp in seen:
                return True
            seen.add(cp)

    return False


def format_schedule(schedule: Schedule) -> str:
    """Format a schedule for readable console output."""
    lines: List[str] = []

    for round_idx, round_pairs in enumerate(schedule, start=1):
        lines.append(f"Round {round_idx}:")
        for p1, p2 in round_pairs:
            lines.append(f"  {p1} vs {p2}")
        if not round_pairs:
            lines.append("  No pairs")
        lines.append("")

    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    demo_players = [f"player_{i}" for i in range(1, 7)]
    demo_schedule = generate_round_robin_disjoint_pairs(
        demo_players,
        num_rounds=None,
        round_multiplier=1,
        reshuffle_between_cycles=False,
        seed=42,
    )
    print(format_schedule(demo_schedule))
