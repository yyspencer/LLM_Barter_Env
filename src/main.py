"""
main.py
 
Entry point for the LLM barter experiment.
 
Modes:
  --dry-run      Validate configs and print the pairing schedule. No API calls.
  --mock-run     Run the full experiment with deterministic mock agents. No API calls.
  --run          Run the full experiment with real LLM providers.
  --random       Run a no-intelligence baseline where agents trade randomly. No API calls.
  --probe-only   Run only preference probes (no trading) to measure LLM drift.
 
Usage:
  python src/main.py --dry-run
  python src/main.py --mock-run
  python src/main.py --random
  python src/main.py --probe-only --probe-only-count 10
  python src/main.py --run
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
 
from config import load_config, print_config_summary
from display_order import order_for_run
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
    """Run with real GPT-5.4 API calls."""
    from runner import run_gpt_experiment
    run_gpt_experiment(cfg)


def cmd_random_run(cfg) -> None:
    """Random-baseline run: agents trade uniformly at random. No API calls."""
    from runner import run_random_experiment
    run_random_experiment(cfg)


def cmd_probe_only(cfg, count: int, with_context: bool) -> None:
    """Probe-only run: repeated preference probes with no trading."""
    from runner import run_probe_only_experiment
    run_probe_only_experiment(cfg, count=count, with_context=with_context)
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Barter Experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --dry-run      Validate configs and preview pairing schedule (no API calls).
  --mock-run     Run full experiment with deterministic mock agents (no API calls).
  --run          Run full experiment with real LLM providers.
  --random       Random-baseline run for comparing against intelligent agents (no API calls).
  --probe-only   Run only preference probes (no trading) to measure LLM drift.
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
        help="Run full experiment with real LLM providers.",
    )
    mode_group.add_argument(
        "--random",
        action="store_true",
        help=("Random-baseline run: every agent proposes a uniformly random "
              "1-for-1 trade and accepts/rejects via coin flip. No API calls."),
    )
    mode_group.add_argument(
        "--probe-only",
        action="store_true",
        help=("Run only preference probes (no trading) to measure LLM drift. "
              "Requires OPENAI_API_KEY. Number of probes set via "
              "--probe-only-count (default 10)."),
    )
 
    # Mode-specific options
    parser.add_argument(
        "--probe-only-count",
        type=int,
        default=10,
        help="Number of probe iterations per player in --probe-only mode (default: 10).",
    )
    parser.add_argument(
        "--probe-only-context",
        action="store_true",
        default=False,
        help=(
            "In --probe-only mode, include each player's prior probe responses "
            "in the context window of subsequent probes. Without this flag each "
            "probe is an independent fresh call. With this flag the conversation "
            "grows as [system, Q1, A1, Q2, A2, ..., QN], so the model sees all "
            "its own previous answers. Run both conditions to compare drift with "
            "and without self-memory. Output dirs are suffixed _probe_only vs "
            "_probe_only_context so they are always kept separate."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help=("Number of repeated runs to execute back-to-back (default: 1). "
              "Each run gets its own output directory; the seed is incremented "
              "per run so pairings and random choices differ. Useful for "
              "averaging out variance, especially with --random and --run. "
              "Ignored by --dry-run."),
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
    parser.add_argument(
        "--broadcast",
        action="store_true",
        default=False,
        help=(
            "Force the broadcast condition on, overriding experiment.yaml's "
            "mechanism.broadcast_completed_trades. Under broadcast, every "
            "accepted trade appends a fixed market bulletin "
            "(\"Market Update: Agent X exchanged ... with Agent Y. Trade "
            "described by participants as 'fair and necessary.'\"), and the "
            "running bulletin board is shown in all negotiation, commitment, "
            "and preference probe prompts."
        ),
    )

    args = parser.parse_args()

    if args.num_runs < 1:
        parser.error(f"--num-runs must be >= 1 (got {args.num_runs})")

    # API keys are needed for any run that actually calls an LLM.
    # That's --run and --probe-only. Everything else is offline.
    needs_keys_by_default = args.run or args.probe_only
    require_keys = needs_keys_by_default and not args.skip_api_key_check

    cfg = load_config(
        experiment_path=Path(args.experiment),
        models_path=Path(args.models),
        players_path=Path(args.players),
        env_path=Path(args.env),
        require_api_keys=require_keys,
    )

    # --broadcast overrides the config value. We only flip TRUE -> the flag
    # is opt-in; the default is whatever experiment.yaml says.
    if args.broadcast:
        cfg.experiment.mechanism.broadcast_completed_trades = True

    # --dry-run is just a config sanity check; repeating it is pointless,
    # so it ignores --num-runs and runs once.
    if args.dry_run:
        cmd_dry_run(cfg)
        return

    # For every other mode, run --num-runs times. Each iteration bumps the
    # seed (so pairing schedules and random-baseline RNG actually differ)
    # and tags the experiment name with the run index (so the output dirs
    # are obviously grouped). The base values are restored at the end so
    # callers reusing cfg afterward see no side effects.
    base_seed = cfg.experiment.experiment.seed
    base_name = cfg.experiment.experiment.name

    # Snapshot the starting inventories from players.yaml so each run
    # starts fresh regardless of how the previous run mutated player objects.
    # (The mock/gpt runners write live post-trade inventories back onto
    # player.inventory during a run, so without this reset run 2 would
    # start from run 1's end-state — agents at their Nash equilibrium.)
    starting_inventories = {
        p.id: dict(p.inventory) for p in cfg.players.players
    }

    try:
        for i in range(args.num_runs):
            # Restore every player to their yaml-defined starting inventory.
            for p in cfg.players.players:
                p.inventory = dict(starting_inventories[p.id])

            if args.num_runs > 1:
                cfg.experiment.experiment.seed = base_seed + i
                cfg.experiment.experiment.name = f"{base_name}_run_{i + 1}"

            # Assign the counterbalancing run index. This is 1-indexed so
            # run_index=1 maps to ORDER_PATTERN[0] = ['A', 'B', 'C'].
            cfg.experiment.experiment.run_index = i + 1

            bar = "=" * 60
            print(f"\n{bar}\n=== Run {cfg.experiment.experiment.run_index} of "
                  f"{args.num_runs} (seed={cfg.experiment.experiment.seed}, "
                  f"display_order={order_for_run(cfg.experiment.experiment.run_index)}) ===\n{bar}")

            if args.mock_run:
                cmd_mock_run(cfg)
            elif args.run:
                cmd_run(cfg)
            elif args.random:
                cmd_random_run(cfg)
            elif args.probe_only:
                cmd_probe_only(cfg, count=args.probe_only_count,
                               with_context=args.probe_only_context)
    finally:
        cfg.experiment.experiment.seed = base_seed
        cfg.experiment.experiment.name = base_name
        cfg.experiment.experiment.run_index = 1
 
 
if __name__ == "__main__":
    main()