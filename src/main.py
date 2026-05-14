"""
main.py
 
Entry point for the LLM barter experiment.
 
Modes:
  --dry-run    Validate configs and print the pairing schedule. No API calls.
  --mock-run   Run the full experiment with deterministic mock agents. No API calls.
  --run        Run the full experiment with real LLM providers. (Not yet implemented.)
 
Usage:
  python src/main.py --dry-run
  python src/main.py --mock-run
  python src/main.py --mock-run --skip-api-key-check
"""
 
from __future__ import annotations
 
import argparse
import sys
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
 
 
def cmd_dry_run(cfg) -> None:
    """Print config summary and pairing schedule. No API calls."""
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
 
 
def cmd_mock_run(cfg) -> None:
    """Run the full experiment loop with deterministic mock agents."""
    from runner import run_mock_experiment
    run_mock_experiment(cfg)
 
 
def cmd_run(cfg) -> None:
    """Run with real LLM providers. Not yet implemented."""
    print("ERROR: --run mode is not yet implemented. Use --mock-run to test the pipeline.")
    sys.exit(1)
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Barter Experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --dry-run    Validate configs and print the pairing schedule (no API calls).
  --mock-run   Run the full experiment with deterministic mock agents (no API calls).
  --run        Run with real LLM providers (not yet implemented).
        """,
    )
 
    # Mode flags (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and preview pairing schedule. No API calls.",
    )
    mode_group.add_argument(
        "--mock-run",
        action="store_true",
        help="Run full experiment with deterministic mock agents. No API calls.",
    )
    mode_group.add_argument(
        "--run",
        action="store_true",
        help="Run full experiment with real LLM providers. (Not yet implemented.)",
    )
 
    # Config paths
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
        help="Load configs without requiring API keys (useful for --mock-run).",
    )
 
    args = parser.parse_args()
 
    # Mock run never needs real API keys
    require_keys = not args.skip_api_key_check and not args.mock_run
 
    cfg = load_config(
        experiment_path=Path(args.experiment),
        models_path=Path(args.models),
        players_path=Path(args.players),
        env_path=Path(args.env),
        require_api_keys=require_keys,
    )
 
    if args.dry_run:
        cmd_dry_run(cfg)
    elif args.mock_run:
        cmd_mock_run(cfg)
    elif args.run:
        cmd_run(cfg)
 
 
if __name__ == "__main__":
    main()