"""
main.py

Dry-run entry point for the LLM barter experiment.

Current purpose:
1. Load and validate configs.
2. Print config summary.
3. Generate the round-robin disjoint-pair schedule.
4. Print the pair schedule.

This does not call any LLM APIs yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config, print_config_summary
from pairing import (
    format_schedule,
    full_cycle_length,
    generate_round_robin_disjoint_pairs,
    has_duplicate_pairs_within_cycle,
)


def resolve_num_rounds(cfg) -> int | None:
    """
    Determine the exact number of rounds to generate.

    If max_rounds_override is set in experiment.yaml, use it.
    Otherwise return None, which tells pairing.py to use:
        round_multiplier * full_cycle_length(num_players)
    """
    override = cfg.experiment.rounds.max_rounds_override
    if override is not None:
        return int(override)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run loader and pairing scheduler for LLM_Barter_Env."
    )

    parser.add_argument(
        "--experiment",
        default="configs/experiment.yaml",
        help="Path to experiment.yaml",
    )
    parser.add_argument(
        "--models",
        default="configs/models.yaml",
        help="Path to models.yaml",
    )
    parser.add_argument(
        "--players",
        default="configs/players.yaml",
        help="Path to players.yaml",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file containing API keys",
    )
    parser.add_argument(
        "--skip-api-key-check",
        action="store_true",
        help="Load configs without requiring API keys to exist.",
    )

    args = parser.parse_args()

    cfg = load_config(
        experiment_path=Path(args.experiment),
        models_path=Path(args.models),
        players_path=Path(args.players),
        env_path=Path(args.env),
        require_api_keys=not args.skip_api_key_check,
    )

    print_config_summary(cfg)

    player_ids = [player.id for player in cfg.players.players]
    num_players = len(player_ids)
    cycle_len = full_cycle_length(num_players)

    exact_num_rounds = resolve_num_rounds(cfg)
    round_multiplier = cfg.experiment.rounds.round_multiplier
    seed = cfg.experiment.experiment.seed

    schedule = generate_round_robin_disjoint_pairs(
        player_ids=player_ids,
        num_rounds=exact_num_rounds,
        round_multiplier=round_multiplier,
        reshuffle_between_cycles=cfg.experiment.pairing.reshuffle_between_runs,
        seed=seed,
    )

    total_rounds = len(schedule)
    total_pair_interactions = sum(len(round_pairs) for round_pairs in schedule)

    print("\n" + "=" * 60)
    print("PAIRING SCHEDULE")
    print("=" * 60)
    print(f"Players: {num_players}")
    print(f"Full cycle length: {cycle_len} rounds")
    print(f"Generated rounds: {total_rounds}")
    print(f"Total pair interactions: {total_pair_interactions}")
    print(f"Pairing mode: {cfg.experiment.pairing.mode}")
    print(f"Execution mode: {cfg.experiment.mechanism.execution_mode}")
    print(f"Synchronization: {cfg.experiment.mechanism.synchronization}")

    has_dupes = has_duplicate_pairs_within_cycle(schedule, num_players)
    print(f"Duplicate pairs within cycle: {'yes' if has_dupes else 'no'}")

    print("-" * 60)
    print(format_schedule(schedule))
    print("=" * 60)

    print("\nDry run complete. No model APIs were called.")


if __name__ == "__main__":
    main()
