"""
optimality_cache.py

Exact parallel optimal-allocation solver with checkpoint caching.

Given a barter experiment run directory, compute:

    optimal_welfare = max_all_feasible_allocations sum_i U_i(x_i)

where total supply of each good is fixed by the starting inventories in
config_snapshot/players.yaml.

The result is cached by a deterministic hash of player IDs, utility weights,
starting inventories, total supply, goods, and utility settings.

Usage:
    python optimality_cache.py --run-dir ../runs/my_run_folder
    python optimality_cache.py --run-dir ../runs/my_run_folder --workers 16
    python optimality_cache.py --run-dir ../runs/my_run_folder --max-combinations 50000000
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as exc:
    raise ImportError("Please install PyYAML: pip install pyyaml") from exc


AllocationVector = Tuple[int, ...]


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_players_from_run(run_dir: str | Path) -> List[Dict[str, Any]]:
    players_path = Path(run_dir) / "config_snapshot" / "players.yaml"
    players_yaml = load_yaml(players_path)
    players = players_yaml.get("players", [])
    if not players:
        raise ValueError(f"No players found in {players_path}")
    return players


def load_summary_from_run(run_dir: str | Path):
    return load_json(Path(run_dir) / "summary.json", default=None)


def shifted_cobb_douglas(
    inventory: Mapping[str, int],
    weights: Mapping[str, float],
    shift: float = 1.0,
) -> float:
    utility = 1.0
    for good, alpha in weights.items():
        utility *= (inventory.get(good, 0) + shift) ** alpha
    return float(utility)


def total_initial_goods(players: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    goods = sorted(players[0]["inventory"].keys())
    totals = {g: 0 for g in goods}
    for p in players:
        for g, q in p["inventory"].items():
            totals[g] += int(q)
    return totals


def generate_allocations_for_one_good(total: int, n_players: int) -> Iterator[AllocationVector]:
    """All nonnegative integer vectors of length n_players that sum to total."""
    if n_players == 1:
        yield (total,)
        return
    for x in range(total + 1):
        for rest in generate_allocations_for_one_good(total - x, n_players - 1):
            yield (x,) + rest


def count_allocations_one_good(total: int, n_players: int) -> int:
    return math.comb(total + n_players - 1, n_players - 1)


def estimate_search_size(players: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals = total_initial_goods(players)
    n_players = len(players)
    per_good_counts = {
        g: count_allocations_one_good(total, n_players)
        for g, total in totals.items()
    }
    total_combinations = 1
    for c in per_good_counts.values():
        total_combinations *= c
    return {
        "n_players": n_players,
        "goods": sorted(totals.keys()),
        "total_supply": totals,
        "per_good_counts": per_good_counts,
        "total_combinations": total_combinations,
    }


def canonical_problem_spec(
    players: Sequence[Mapping[str, Any]],
    shift: float = 1.0,
    utility_type: str = "shifted_cobb_douglas",
) -> Dict[str, Any]:
    players_sorted = sorted(players, key=lambda p: p["id"])
    player_specs = []
    for p in players_sorted:
        player_specs.append({
            "id": p["id"],
            "role": p.get("role"),
            "inventory": {k: int(v) for k, v in sorted(p["inventory"].items())},
            "utility_weights": {
                k: float(v) for k, v in sorted(p["utility_weights"].items())
            },
            "utility_type": p.get("utility_type", utility_type),
        })

    totals = total_initial_goods(players_sorted)
    return {
        "objective": "maximize_sum_player_utilities",
        "utility_type": utility_type,
        "shift": float(shift),
        "goods": sorted(totals.keys()),
        "total_supply": {k: int(v) for k, v in sorted(totals.items())},
        "players": player_specs,
    }


def fingerprint_problem(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunked(seq: Sequence[Any], chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i:i + chunk_size])


def evaluate_first_good_chunk(
    *,
    first_good: str,
    first_alloc_chunk: List[AllocationVector],
    remaining_goods: List[str],
    remaining_allocs_by_good: Dict[str, List[AllocationVector]],
    players_compact: List[Dict[str, Any]],
    shift: float,
) -> Dict[str, Any]:
    n_players = len(players_compact)
    goods = [first_good] + remaining_goods
    remaining_lists = [remaining_allocs_by_good[g] for g in remaining_goods]

    best_welfare = -float("inf")
    best_allocation = None
    checked = 0

    for first_alloc in first_alloc_chunk:
        combo_iter = itertools.product(*remaining_lists) if remaining_lists else [tuple()]

        for remaining_combo in combo_iter:
            alloc_vectors = (first_alloc,) + tuple(remaining_combo)

            inventories = []
            for i in range(n_players):
                inv = {g: int(alloc_vectors[g_idx][i]) for g_idx, g in enumerate(goods)}
                inventories.append(inv)

            welfare = 0.0
            for p, inv in zip(players_compact, inventories):
                welfare += shifted_cobb_douglas(inv, p["utility_weights"], shift=shift)

            checked += 1

            if welfare > best_welfare:
                best_welfare = welfare
                best_allocation = inventories

    return {
        "best_welfare": best_welfare,
        "best_allocation": best_allocation,
        "checked": checked,
    }


def exact_optimal_welfare_parallel(
    players: Sequence[Mapping[str, Any]],
    shift: float = 1.0,
    workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    max_combinations: Optional[int] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    players_sorted = sorted(players, key=lambda p: p["id"])
    n_players = len(players_sorted)

    estimate = estimate_search_size(players_sorted)
    goods = estimate["goods"]
    totals = estimate["total_supply"]
    total_combinations = int(estimate["total_combinations"])

    if progress:
        print("Search estimate:", json.dumps(estimate, indent=2), flush=True)

    if max_combinations is not None and total_combinations > max_combinations:
        raise RuntimeError(
            f"Search space has {total_combinations:,} combinations, exceeding "
            f"max_combinations={max_combinations:,}. Refusing exact search."
        )

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    per_good_counts = estimate["per_good_counts"]
    split_good = max(goods, key=lambda g: per_good_counts[g])
    remaining_goods = [g for g in goods if g != split_good]

    first_allocs = list(generate_allocations_for_one_good(totals[split_good], n_players))
    remaining_allocs_by_good = {
        g: list(generate_allocations_for_one_good(totals[g], n_players))
        for g in remaining_goods
    }

    if chunk_size is None:
        target_tasks = max(workers * 8, 1)
        chunk_size = max(1, math.ceil(len(first_allocs) / target_tasks))

    chunks = list(chunked(first_allocs, chunk_size))

    players_compact = [
        {
            "id": p["id"],
            "utility_weights": {
                k: float(v) for k, v in sorted(p["utility_weights"].items())
            },
        }
        for p in players_sorted
    ]

    start = time.time()
    checked_total = 0
    best_welfare = -float("inf")
    best_allocation = None
    completed_tasks = 0

    if progress:
        print(
            f"Starting exact parallel search: split_good={split_good}, "
            f"workers={workers}, tasks={len(chunks)}, "
            f"total={total_combinations:,}",
            flush=True,
        )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                evaluate_first_good_chunk,
                first_good=split_good,
                first_alloc_chunk=chunk,
                remaining_goods=remaining_goods,
                remaining_allocs_by_good=remaining_allocs_by_good,
                players_compact=players_compact,
                shift=shift,
            )
            for chunk in chunks
        ]

        for fut in as_completed(futures):
            result = fut.result()
            completed_tasks += 1
            checked_total += int(result["checked"])

            if result["best_welfare"] > best_welfare:
                best_welfare = float(result["best_welfare"])
                best_allocation = result["best_allocation"]

            if progress:
                elapsed = time.time() - start
                rate = checked_total / elapsed if elapsed > 0 else 0.0
                eta = (total_combinations - checked_total) / rate if rate > 0 else float("inf")
                print(
                    f"[{completed_tasks}/{len(chunks)} tasks] "
                    f"checked={checked_total:,}/{total_combinations:,} "
                    f"({checked_total / total_combinations:.2%}); "
                    f"best={best_welfare:.8f}; "
                    f"rate={rate:,.0f}/s; ETA={eta / 60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - start
    return {
        "optimal_welfare": best_welfare,
        "optimal_allocation": best_allocation,
        "checked_combinations": checked_total,
        "total_combinations": total_combinations,
        "elapsed_seconds": elapsed,
        "workers": workers,
        "split_good": split_good,
        "exact": True,
    }


def compute_or_load_optimality(
    run_dir: str | Path,
    cache_dir: str | Path = "optimality_cache",
    shift: float = 1.0,
    workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    max_combinations: Optional[int] = None,
    force: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    players = load_players_from_run(run_dir)

    spec = canonical_problem_spec(players, shift=shift)
    fp = fingerprint_problem(spec)
    cache_path = cache_dir / f"{fp}.json"

    if cache_path.exists() and not force:
        if progress:
            print(f"Loading cached optimality: {cache_path}", flush=True)
        cached = load_json(cache_path)
        cached["cache_hit"] = True
        cached["cache_path"] = str(cache_path)
        return cached

    if progress:
        print(f"No cache found. Computing optimality for run: {run_dir}", flush=True)
        print(f"Fingerprint: {fp}", flush=True)
        print(f"Cache path: {cache_path}", flush=True)

    result = exact_optimal_welfare_parallel(
        players=players,
        shift=shift,
        workers=workers,
        chunk_size=chunk_size,
        max_combinations=max_combinations,
        progress=progress,
    )

    output = {
        "fingerprint": fp,
        "problem_spec": spec,
        "run_dir_source": str(run_dir),
        "created_at_unix": time.time(),
        **result,
    }

    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_path.replace(cache_path)

    output["cache_hit"] = False
    output["cache_path"] = str(cache_path)
    return output


def realized_welfare_from_summary(run_dir: str | Path) -> Optional[float]:
    summary = load_summary_from_run(run_dir)
    if not summary:
        return None
    final_utils = summary.get("final_utilities", [])
    if not final_utils:
        return None
    return sum(float(x["final_utility"]) for x in final_utils)


def starting_welfare_from_summary(run_dir: str | Path) -> Optional[float]:
    summary = load_summary_from_run(run_dir)
    if not summary:
        return None
    starting_utils = summary.get("starting_utilities", [])
    if not starting_utils:
        return None
    return sum(float(x["utility"]) for x in starting_utils)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to one run output folder.")
    parser.add_argument("--cache-dir", default="optimality_cache")
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists.")
    args = parser.parse_args()

    result = compute_or_load_optimality(
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        shift=args.shift,
        workers=args.workers,
        chunk_size=args.chunk_size,
        max_combinations=args.max_combinations,
        force=args.force,
        progress=True,
    )

    print("\n=== OPTIMALITY RESULT ===")
    print(f"Cache hit: {result.get('cache_hit')}")
    print(f"Cache path: {result.get('cache_path')}")
    print(f"Optimal welfare: {result['optimal_welfare']:.8f}")
    print(f"Checked combinations: {result.get('checked_combinations'):,}")
    print(f"Elapsed seconds: {result.get('elapsed_seconds')}")

    starting = starting_welfare_from_summary(args.run_dir)
    realized = realized_welfare_from_summary(args.run_dir)

    if starting is not None:
        print(f"Starting welfare: {starting:.8f}")
    if realized is not None:
        print(f"Realized welfare: {realized:.8f}")
        print(f"Efficiency ratio: {realized / result['optimal_welfare']:.8f}")


if __name__ == "__main__":
    main()
