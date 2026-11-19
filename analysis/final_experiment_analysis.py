#!/usr/bin/env python3
"""Confirmatory analysis for the final 3 x 3 market-bulletin experiment.

Design as recorded
----------------------------------
Factor 1, market bulletin, three levels:

    Control   -- bulletin board disabled
    Scarcity  -- "Good B is becoming harder to obtain ..."
    Expert    -- "Market Analyst Bulletin: ... Good B ... strategically
                  important good."

Factor 2, model, three levels: ``claude-sonnet-5`` (Anthropic),
``gpt-5.6-terra`` (OpenAI), ``gemini-3.1-pro-preview`` (Google).

Forty independent runs per cell were collected, 3 x 3 x 40 = 360; the count
that survives the plan's technical-exclusion rules is printed in the report
header.  Each run seats six agents: two Builders (player_1/2, endowment
A1 B0 C3), two Weavers (player_3/4, A3 B1 C0), and two Merchants
(player_5/6, A1 B1 C1, equal Cobb-Douglas weights).  Ten trading rounds are
followed by a four-round washout block (rounds 11-14) in which the bulletin
board is withdrawn.

Measures, exactly as the plan defines them
------------------------------------------
Every measure below is read from the probe schema
(``preference_probes.json``) and the standardized shadow-offer battery
(``shadow_trades.json``):

    Pb_general    ratings_general[B]   - mean(ratings_general[A, C])
                  the inventory-independent general-importance B index
    bundle_B      desired_bundle[B], out of the six units the probe allows
    Pb_bundle     desired_bundle[B]    - mean(desired_bundle[A, C])
    Pb_inventory  ratings_inventory[B] - mean(ratings_inventory[A, C])
                  state-dependent valuation
    Pb_predicted  the same contrast over ``predicted_marginal_utility``,
                  the latent shifted-Cobb-Douglas marginal utility at the
                  agent's actual inventory
    accept        accept/reject on each standardized shadow offer, analysed
                  separately for Acquire-B and Surrender-B
    Policy_B      P(accept | acquire B) - P(accept | surrender B)
    reservation   the rung of the surrender ladder at which a Merchant first
                  accepts (1 unit, 2 units, or never), averaged over the two
                  counter goods

Hypotheses (Prereg2)
--------------------
    H1  operational policy moves with the bulletin: 1) acquire-B acceptance
        up, 2) surrender-B acceptance down.  (The reservation value was H1.3
        in the original plan; Prereg2 dropped it, so it is exploratory here.)
    H2  general stated preference does not durably move: 1) ideal bundle,
        2) general-importance index
    H3  policy effects exist while stated-preference changes stay inside the
        +/-0.5 equivalence bounds -- judged from the H1 and H2 tests, with no
        separate model
    H4  effects weaken after the bulletin is withdrawn, tested separately for
        acquiring B, surrendering B, general stated preference and
        state-dependent valuation, by round-14 contrasts and round 10 -> 14
        changes

What follows Prereg2 exactly
----------------------------
*   Round is a categorical factor with levels 0, 5, 10, 12, 14 (reference 0).
*   The stated-preference and state-dependent models contain Condition x
    Round x Model and B presentation position, with random intercepts for run
    and Merchant within run.  Pooled condition contrasts are averages of the
    three model-specific contrasts, i.e. estimated marginal means over Model.
*   The state-dependent index is modelled without a holdings covariate, as
    Prereg2 specifies.  Because the bulletins change holdings and the index
    moves with them, the report adds an exact-inventory comparison (treated
    vs Control Merchants holding the same A, B and C) as a robustness check,
    and splits it into the three ratings as a post hoc check.
*   Planned contrasts: Expert - Control, Scarcity - Control, Expert - Scarcity
    at round 10; round 0 -> 10 difference-in-differences; round-14 contrasts
    and round 10 -> 14 changes.  Holm adjustment within three families
    (primary policy, general preference, washout); two-sided alpha = .05;
    odds ratios for the logistic model; TOST against +/-0.5 for the two
    general-preference measures.

Departures from Prereg2
-----------------------
Each is printed in the report so a reader of the output cannot miss it.

1.  Estimation of the policy model.  ``statsmodels`` has no frequentist
    binomial GLMM, so acceptance is a binomial GLM with cluster-robust
    (sandwich) covariance on independent run; the Merchant and snapshot
    levels are absorbed into the standard errors.  Pass ``--glmm`` to add a
    Bayesian binomial mixed model with run and Merchant-within-run random
    intercepts as a robustness check.
2.  Model factor in the policy model.  Prereg2 asks for Condition x Round x
    Direction x Model.  The policy model keeps Model as a main effect: with
    about 40 runs per cell and a rare outcome, the four-way model has cells
    with no accepts at all (as single-model fits did in the pilot).
    Model-specific policy effects are reported as run-level differences in
    acceptance rates, labelled exploratory.
3.  Ideal bundle.  No beta-binomial mixed model is available here, so
    B-out-of-six is fitted as a binomial GLM with run-clustered covariance,
    and the equivalence test is run on the units scale from a linear mixed
    model, because the bound (+/-0.5 B units) is stated in units.  When every
    answer is identical the measure is reported as exactly invariant instead
    of being fitted.
4.  Bulletin delivery is audited from ``raw_prompts.jsonl`` for every run,
    because the YAML snapshot in each run directory does not record the
    setting that was actually loaded.  A treated run must carry its bulletin
    in every prompt of rounds 1-10, and no run may show a bulletin at round 0,
    in the washout, or one belonging to another condition; a run that breaks
    any of these is excluded under Prereg2's "failure to apply the assigned
    experimental condition" rule.  Round 0 is therefore a clean baseline.
5.  Degenerate cells are detected from the data, never assumed.  A design
    cell of the shadow battery with no accepts (or no rejects) is perfectly
    separated and corrupts the cluster-robust covariance, so it is dropped
    from the logistic fit, reported, and the contrasts that needed it are
    given on the probability scale instead.  Where a stated measure has zero
    variance in one condition, the mixed-model contrasts are accompanied by
    run-level Welch and permutation tests.
6.  Every planned policy contrast is also reported as a run-level difference
    in acceptance rates.  These are secondary: the logistic contrasts carry
    the confirmatory tests.

Runs are read from ``data/<Model>/<Condition>/<run>`` by default
(``--base-dir`` overrides it).  Outputs land in
``analysis/analysis_outputs/final_experiment``:

    report.txt          the primary record: a computed summary of H1-H4, a
                        legend, then annotated tables for every analysis
    contrasts.csv       every contrast in the report, keyed by label and family
    dissociation.csv    the H3 common-scale table
    runs.csv, probes.csv, shadow_offers.csv
                        the tidy frames every analysis reads

Progress goes to stderr and stays out of the report.  This script draws no
figures; ``final_experiment_figures.py`` draws them into the same folder,
loading the data through this module.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import textwrap
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths and experiment design
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = REPO_DIR / "data"
DEFAULT_OUT_DIR = REPO_DIR / "analysis" / "analysis_outputs" / "final_experiment"

FOCAL_GOOD = "B"
GOODS = ["A", "B", "C"]
BUNDLE_TOTAL = 6

# Run directories are laid out as <base>/<Model>/<Condition>/<run>, with the
# folder names spelled exactly as in MODELS and CONDITIONS.
CONDITIONS = ["Control", "Scarcity", "Expert"]
CONDITION_BLURB = {
    "Control": "Control (no bulletin)",
    "Scarcity": "Scarcity (Good B is becoming harder to obtain)",
    "Expert": "Expert (analyst recommendation on Good B)",
}
# Long form only; the plan's contrasts are all against Control.
REFERENCE_CONDITION = "Control"
TREATMENTS = ["Scarcity", "Expert"]

MODELS = ["Claude", "OpenAI", "Gemini"]

ROLE_BY_PLAYER = {
    "player_1": "Builder",
    "player_2": "Builder",
    "player_3": "Weaver",
    "player_4": "Weaver",
    "player_5": "Merchant",
    "player_6": "Merchant",
}

# Probe rounds, all of them in the Prereg2 confirmatory set.  Round 0 is a
# clean pre-treatment baseline; the bulletin is shown on rounds 1-10 only.
PROBE_ROUNDS = [0, 5, 10, 12, 14]
TREATMENT_END = 10  # last round on which the bulletin board is shown
PRIMARY_ROUND = 10  # the plan's "treatment on" time point
PERSISTENCE_ROUND = 14

DIRECTIONS = ["acquire", "surrender"]
DIRECTION_LABEL = {"acquire": "Acquire B", "surrender": "Surrender B"}

# Equivalence bounds, from the plan, on the scale each measure is stated in.
EQUIV_BOUND_GENERAL = 0.5  # rating points on Pb_general
EQUIV_BOUND_BUNDLE = 0.5  # B units out of six
ALPHA = 0.05
N_PERMUTATIONS = 20000

# H3 needs one scale that every measure can be read on.  A baseline-SD
# standardisation is unavailable because several measures have zero variance at
# round 0.  The width of each measuring instrument is available for all of
# them and assumes nothing about the sample: a 1-10 rating contrast spans -9
# to +9, a probability spans 0 to 1, the six-unit bundle spans 0 to 6, and the
# B-protection index spans -1 to +1.
INSTRUMENT_RANGE = {
    "accept": 1.0,
    "Policy_B": 2.0,
    "Pb_general": 18.0,
    "Pb_inventory": 18.0,
    "bundle_B": float(BUNDLE_TOTAL),
}

class Tee:
    """Mirror everything printed into a transcript file in the output folder."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str) -> int:
        self._stdout.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_RUN_FILES = (
    "preference_probes.json",
    "shadow_trades.json",
    "trades.json",
    "summary.json",
)


def discover_run_dirs(base_dir: Path) -> list[tuple[str, str, Path]]:
    """Return ``(model, condition, run_dir)`` for every run under ``base_dir``.

    The layout is ``<base_dir>/<Model>/<Condition>/<run>``.  Folders are
    walked in sorted order, so the row order of every frame -- and with it the
    floating-point detail of the near-degenerate fits -- is reproducible.
    """
    found: list[tuple[str, str, Path]] = []
    for model in sorted(MODELS):
        for condition in sorted(CONDITIONS):
            condition_dir = base_dir / model / condition
            if not condition_dir.is_dir():
                print(f"  WARNING: no run directory {condition_dir}")
                continue
            found.extend(
                (model, condition, run_dir)
                for run_dir in sorted(condition_dir.iterdir())
                if run_dir.is_dir()
            )
    if not found:
        raise FileNotFoundError(f"No runs discovered under {base_dir}")
    return found


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _index(mapping: Any, good: str) -> float:
    """B minus the mean of A and C, the plan's relative-preference index."""
    if not isinstance(mapping, dict):
        return np.nan
    try:
        values = {key: float(mapping.get(key, np.nan)) for key in GOODS}
    except (TypeError, ValueError):
        return np.nan
    others = [values[other] for other in GOODS if other != good]
    return values[good] - float(np.mean(others))


def _positive(mapping: Any) -> dict[str, float]:
    if not isinstance(mapping, dict):
        return {}
    return {
        key: float(value)
        for key, value in mapping.items()
        if isinstance(value, (int, float)) and value > 0
    }


# Distinctive phrase of each bulletin.
BULLETIN_PATTERNS = {
    "Scarcity": "Good B is becoming harder to obtain",
    "Expert": "Market Analyst Bulletin",
}


def audit_exposure(run_dir: Path, condition: str) -> dict[str, Any]:
    """Read what the agents were actually shown, prompt by prompt.

    Returns the share of bulletin-phase prompts (rounds 1-10) that carry the
    assigned bulletin, the count of prompts carrying any *other* bulletin, and
    the counts of round-0 and washout (rounds 11-14) prompts carrying one.  This is
    the only reliable record of delivery: ``experiment.yaml`` in the snapshot
    shows ``broadcast_completed_trades: false`` for every run, treated or not.
    """
    assigned = BULLETIN_PATTERNS.get(condition)
    on_total = on_assigned = foreign = 0
    baseline_total = baseline_hits = washout_total = washout_hits = 0
    with (run_dir / "raw_prompts.jsonl").open(
        encoding="utf-8", errors="replace"
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            # The round is needed, so the line is parsed; the bulletin itself is
            # matched on the raw line, which is far cheaper than walking messages.
            head = line[:400]
            marker = head.find('"round_index":')
            if marker < 0:
                round_index = int(json.loads(line).get("round_index") or 0)
            else:
                digits = head[marker + 14 :].lstrip()
                number = ""
                for character in digits:
                    if character.isdigit():
                        number += character
                    else:
                        break
                round_index = int(number) if number else 0
            has_assigned = bool(assigned) and assigned in line
            has_foreign = any(
                pattern in line
                for name, pattern in BULLETIN_PATTERNS.items()
                if name != condition
            )
            if round_index == 0:
                baseline_total += 1
                baseline_hits += has_assigned or has_foreign
            elif round_index <= TREATMENT_END:
                on_total += 1
                on_assigned += has_assigned
                foreign += has_foreign
            else:
                washout_total += 1
                washout_hits += has_assigned or has_foreign
    return {
        "exposure_share": on_assigned / on_total if on_total else np.nan,
        "foreign_bulletin_prompts": foreign,
        "round0_bulletin_prompts": baseline_hits,
        "round0_prompts": baseline_total,
        "washout_bulletin_prompts": washout_hits,
        "washout_prompts": washout_total,
        "bulletin_phase_prompts": on_total,
    }


def condition_applied(condition: str, audit: dict[str, Any]) -> str | None:
    """Why a run fails the plan's "assigned condition applied" rule, if it does."""
    if audit["foreign_bulletin_prompts"]:
        return (
            f"{audit['foreign_bulletin_prompts']} prompts carry a bulletin that "
            f"is not the {condition} one"
        )
    if audit["round0_bulletin_prompts"]:
        return (
            f"{audit['round0_bulletin_prompts']} round-0 prompts carry a "
            "bulletin, so round 0 is not a baseline"
        )
    if condition == REFERENCE_CONDITION:
        return None
    if audit["exposure_share"] < 1.0:
        return (
            f"only {audit['exposure_share']:.1%} of round 1-10 prompts carry "
            "the assigned bulletin"
        )
    if audit["washout_bulletin_prompts"]:
        return (
            f"{audit['washout_bulletin_prompts']} washout prompts still carry "
            "the bulletin"
        )
    return None


def load_dataset(base_dir: Path) -> dict[str, pd.DataFrame]:
    """Build the three tidy frames every analysis below reads.

    ``probes``   one row per agent per probe round
    ``shadow``   one row per standardized shadow offer
    ``runs``     one row per run, carrying the audited bulletin exposure
    """
    probe_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    excluded: list[tuple[Path, str]] = []

    for model, condition, run_dir in discover_run_dirs(base_dir):
        missing = [
            name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()
        ]
        if missing:
            excluded.append((run_dir, f"missing {', '.join(missing)}"))
            continue
        try:
            probes = _read_json(run_dir / "preference_probes.json")
            shadow = _read_json(run_dir / "shadow_trades.json")
            trades = _read_json(run_dir / "trades.json")
            summary = _read_json(run_dir / "summary.json")
        except (json.JSONDecodeError, OSError) as error:
            excluded.append((run_dir, f"unreadable log ({error})"))
            continue

        run_id = f"{model}|{condition}|{run_dir.name}"

        # Guard the two things the plan lists as invalidating a run: the
        # assigned condition must have been applied, and the agents must have
        # been initialized with the prespecified endowments and roles.
        roles = {
            entry["player_id"]: entry["role"] for entry in summary["starting_utilities"]
        }
        if roles != ROLE_BY_PLAYER:
            excluded.append((run_dir, "roles not as prespecified"))
            continue

        audit = audit_exposure(run_dir, condition)
        reason = condition_applied(condition, audit)
        if reason:
            excluded.append((run_dir, f"assigned condition not applied: {reason}"))
            continue
        run_rows.append(
            {
                "run_id": run_id,
                "model": model,
                "condition": condition,
                "run_dir": str(run_dir),
                "trades_accepted": int(summary.get("total_trades_accepted") or 0),
                "trades_rejected": int(summary.get("total_trades_rejected") or 0),
                **audit,
            }
        )

        for probe in probes:
            player = probe.get("player_id")
            response = probe.get("response")
            if not isinstance(response, dict):
                continue
            order = list(probe.get("display_order") or GOODS)
            bundle = response.get("desired_bundle") or {}
            bundle_total = sum(
                float(value)
                for value in bundle.values()
                if isinstance(value, (int, float))
            )
            probe_rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "condition": condition,
                    "player_id": player,
                    "role": ROLE_BY_PLAYER.get(player, "?"),
                    "round": int(probe["round_index"]),
                    "b_position": (
                        order.index(FOCAL_GOOD) + 1 if FOCAL_GOOD in order else np.nan
                    ),
                    "Pb_general": _index(response.get("ratings_general"), FOCAL_GOOD),
                    "Pb_inventory": _index(
                        response.get("ratings_inventory"), FOCAL_GOOD
                    ),
                    "Pb_bundle": _index(bundle, FOCAL_GOOD),
                    "bundle_B": float(bundle.get(FOCAL_GOOD, np.nan)),
                    "bundle_total": bundle_total,
                    "Pb_predicted": _index(
                        probe.get("predicted_marginal_utility"), FOCAL_GOOD
                    ),
                    "held_B": float(
                        (probe.get("current_inventory") or {}).get(FOCAL_GOOD, np.nan)
                    ),
                    # The full holding, so treated and Control Merchants can be
                    # compared at identical inventories.
                    "inventory_state": ",".join(
                        str((probe.get("current_inventory") or {}).get(good, ""))
                        for good in GOODS
                    ),
                    **{
                        f"rating_inventory_{good}": float(
                            (response.get("ratings_inventory") or {}).get(good, np.nan)
                        )
                        for good in GOODS
                    },
                    **{
                        f"rating_general_{good}": float(
                            (response.get("ratings_general") or {}).get(good, np.nan)
                        )
                        for good in GOODS
                    },
                    **{
                        f"bundle_{good}": float(bundle.get(good, np.nan))
                        for good in GOODS
                    },
                }
            )

        for offer in shadow:
            give = _positive(offer.get("player_give"))
            receive = _positive(offer.get("player_receive"))
            direction = str(offer.get("direction") or "")
            if direction not in DIRECTIONS:
                continue
            focal_units = (
                receive.get(FOCAL_GOOD, 0.0)
                if direction == "acquire"
                else give.get(FOCAL_GOOD, 0.0)
            )
            counter_units = (
                sum(give.values()) if direction == "acquire" else sum(receive.values())
            )
            decision = str(offer.get("decision") or "").lower()
            if focal_units <= 0 or decision not in {"accept", "reject"}:
                continue
            units_received = sum(receive.values())
            units_surrendered = sum(give.values())
            utility_before = float(offer.get("utility_before", np.nan))
            utility_after = float(offer.get("utility_after", np.nan))
            if not (utility_before > 0 and utility_after > 0):
                continue
            inventory = offer.get("inventory_before") or {}
            shadow_rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "condition": condition,
                    "player_id": offer.get("player_id"),
                    "round": int(offer["round_index"]),
                    "shadow_id": offer.get("shadow_id"),
                    "direction": direction,
                    "counter_good": str(offer.get("counter_good") or ""),
                    "units_received": units_received,
                    "units_surrendered": units_surrendered,
                    # Economic attractiveness, signed by direction already:
                    # >0 means the offer hands over more units than it takes.
                    "log_ratio": math.log(units_received / units_surrendered),
                    # Counter-good units per focal unit -- the rung of the ladder.
                    "price": counter_units / focal_units,
                    "dlogU": math.log(utility_after) - math.log(utility_before),
                    "accept": 1 if decision == "accept" else 0,
                    "held_B": float(inventory.get(FOCAL_GOOD, np.nan)),
                    "snapshot": f"{offer.get('player_id')}@{offer['round_index']}",
                }
            )

    probes_df = pd.DataFrame(probe_rows)
    shadow_df = pd.DataFrame(shadow_rows)
    runs_df = pd.DataFrame(run_rows)

    for frame in (probes_df, shadow_df, runs_df):
        frame["condition"] = pd.Categorical(
            frame["condition"], categories=CONDITIONS, ordered=True
        )
        frame["model"] = pd.Categorical(frame["model"], categories=MODELS, ordered=True)
    for frame in (probes_df, shadow_df):
        frame["round_f"] = pd.Categorical(
            ["r" + str(value) for value in frame["round"]],
            categories=["r" + str(value) for value in PROBE_ROUNDS],
            ordered=True,
        )
    shadow_df["direction"] = pd.Categorical(
        shadow_df["direction"], categories=DIRECTIONS, ordered=True
    )

    return {
        "probes": probes_df,
        "shadow": shadow_df,
        "runs": runs_df,
        "excluded": pd.DataFrame(
            [{"run_dir": str(path), "reason": reason} for path, reason in excluded],
            columns=["run_dir", "reason"],
        ),
    }


# ---------------------------------------------------------------------------
# Derived per-snapshot measures
# ---------------------------------------------------------------------------

def reservation_frame(shadow_df: pd.DataFrame) -> pd.DataFrame:
    """Reservation rung for surrendering B, per Merchant per probe.

    The battery walks a two-rung ladder for each counter good: 1 unit, then 2
    units, for the same single unit of B.  The reservation value is the first
    rung the Merchant accepts -- 1, 2, or 3 for "no rung on this ladder" --
    and the measure is the mean over the two counter goods, so it is only
    defined where the whole ladder is feasible (that is, the Merchant holds
    at least one B).  Non-monotone ladders (accepts 1 unit, refuses 2) are
    counted and reported rather than smoothed away.
    """
    surrender = shadow_df[shadow_df["direction"] == "surrender"]
    rows: list[dict[str, Any]] = []
    grouped = surrender.groupby(
        ["run_id", "model", "condition", "player_id", "round"], observed=True
    )
    for (run_id, model, condition, player_id, round_index), block in grouped:
        rungs: list[float] = []
        violations = 0
        for counter_good, ladder in block.groupby("counter_good", observed=True):
            by_price = {
                float(row.price): int(row.accept) for row in ladder.itertuples()
            }
            if {1.0, 2.0} - set(by_price):
                continue
            if by_price[1.0]:
                rungs.append(1.0)
            elif by_price[2.0]:
                rungs.append(2.0)
            else:
                rungs.append(3.0)
            if by_price[1.0] and not by_price[2.0]:
                violations += 1
        if len(rungs) < 2:
            continue
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "condition": condition,
                "player_id": player_id,
                "round": int(round_index),
                "reservation": float(np.mean(rungs)),
                "ladder_violations": violations,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["condition"] = pd.Categorical(
        frame["condition"], categories=CONDITIONS, ordered=True
    )
    frame["model"] = pd.Categorical(frame["model"], categories=MODELS, ordered=True)
    frame["round_f"] = pd.Categorical(
        ["r" + str(value) for value in frame["round"]],
        categories=["r" + str(value) for value in PROBE_ROUNDS],
        ordered=True,
    )
    return frame


def policy_index_frame(shadow_df: pd.DataFrame) -> pd.DataFrame:
    """B-protection index per Merchant per probe, on the probability scale."""
    grouped = shadow_df.groupby(
        ["run_id", "model", "condition", "player_id", "round", "direction"],
        observed=True,
    )["accept"].mean()
    wide = grouped.unstack("direction")
    wide = wide.dropna(subset=DIRECTIONS)
    frame = wide.reset_index()
    frame["Policy_B"] = frame["acquire"] - frame["surrender"]
    return frame


# ---------------------------------------------------------------------------
# Contrast machinery
# ---------------------------------------------------------------------------

def _fixed_effects(result: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fixed-effect names, estimates, and covariance, for GLM or MixedLM.

    ``MixedLMResults.params`` appends the variance components, which must not
    be swept into a contrast, so the fixed block is sliced off explicitly.
    """
    if hasattr(result, "fe_params"):
        names = list(result.fe_params.index)
        estimates = np.asarray(result.fe_params, dtype=float)
        covariance = np.asarray(result.cov_params(), dtype=float)[
            : len(names), : len(names)
        ]
    else:
        names = list(result.params.index)
        estimates = np.asarray(result.params, dtype=float)
        covariance = np.asarray(result.cov_params(), dtype=float)
    return names, estimates, covariance


def contrast(
    result: Any,
    weights: dict[str, float],
    label: str,
    family: str,
    *,
    exponentiate: bool = False,
) -> dict[str, Any]:
    """Estimate, 95% CI, 90% CI, and two-sided p for one linear combination.

    A name absent from the design is treated as a zero coefficient, which is
    what the treatment-coded reference cell is; asking for it explicitly
    keeps every contrast below readable as "this cell minus that cell".
    """
    names, estimates, covariance = _fixed_effects(result)
    position = {name: index for index, name in enumerate(names)}
    vector = np.zeros(len(names))
    for name, weight in weights.items():
        if name in position:
            vector[position[name]] += weight
        elif not name.endswith("__reference__"):
            # Unknown names are a coding error, not a silently dropped cell.
            raise KeyError(f"{name!r} is not a parameter of this model")
    estimate = float(vector @ estimates)
    standard_error = float(math.sqrt(max(0.0, vector @ covariance @ vector)))
    if standard_error == 0.0:
        return {
            "label": label,
            "family": family,
            "estimate": estimate,
            "se": np.nan,
            "z": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci90_low": np.nan,
            "ci90_high": np.nan,
        }
    z = estimate / standard_error
    row = {
        "label": label,
        "family": family,
        "estimate": estimate,
        "se": standard_error,
        "z": z,
        "p": float(2 * stats.norm.sf(abs(z))),
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "ci90_low": estimate - 1.645 * standard_error,
        "ci90_high": estimate + 1.645 * standard_error,
    }
    if exponentiate:
        row["odds_ratio"] = math.exp(estimate)
        row["or_low"] = math.exp(row["ci_low"])
        row["or_high"] = math.exp(row["ci_high"])
    return row


def holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Holm-adjust p-values within each declared family, in place."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    for family_rows in by_family.values():
        testable = [row for row in family_rows if np.isfinite(row.get("p", np.nan))]
        order = sorted(testable, key=lambda row: row["p"])
        count = len(order)
        running = 0.0
        for index, row in enumerate(order):
            adjusted = (count - index) * row["p"]
            running = max(running, min(1.0, adjusted))
            row["p_holm"] = running
        for row in family_rows:
            row.setdefault("p_holm", np.nan)
    return rows


def tost(estimate: float, standard_error: float, bound: float) -> dict[str, float]:
    """Two one-sided tests against +/-``bound``; equivalence when p < alpha."""
    if not np.isfinite(standard_error) or standard_error <= 0:
        return {"p_lower": np.nan, "p_upper": np.nan, "p_tost": np.nan}
    p_lower = float(stats.norm.sf((estimate + bound) / standard_error))
    p_upper = float(stats.norm.sf((bound - estimate) / standard_error))
    return {
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
    }


def cell(*parts: str) -> str:
    return ":".join(parts)


def cell_term(variable: str, level: str) -> str:
    """Patsy's name for one level of a treatment-coded categorical."""
    return f"C({variable})[T.{level}]"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _prepare_cells(
    frame: pd.DataFrame, columns: Sequence[str], reference: str
) -> pd.DataFrame:
    """Add a single ``cell`` factor spanning the interaction of ``columns``.

    The plan's saturated Condition x Round x Direction model is exactly a
    cell-means model, and writing it that way makes every planned contrast a
    plain difference of cells instead of a sum of four interaction terms that
    has to be reassembled from patsy's naming.
    """
    frame = frame.copy()
    frame["cell"] = [
        cell(*[str(value) for value in row])
        for row in frame[list(columns)].itertuples(index=False)
    ]
    levels = sorted(set(frame["cell"]))
    if reference not in levels:
        raise ValueError(f"reference cell {reference!r} absent from the data")
    levels = [reference] + [level for level in levels if level != reference]
    frame["cell"] = pd.Categorical(frame["cell"], categories=levels, ordered=False)
    return frame


def separated_cells(
    frame: pd.DataFrame, columns: Sequence[str], outcome: str
) -> pd.DataFrame:
    """Design cells in which the binary outcome never varies.

    A cell with no accepts (or no rejects) is perfectly separated: its logit
    is unbounded, its contribution to the Hessian is degenerate, and leaving
    it in corrupts the whole cluster-robust covariance rather than just its
    own row.  The plan's exclusion rules cover invalid observations, not
    structural zeros, so these cells are dropped from the *fit* only, with
    the counts reported and the affected contrasts handled separately.
    """
    counts = (
        frame.groupby(list(columns), observed=True)[outcome]
        .agg(["sum", "count"])
        .reset_index()
    )
    return counts[(counts["sum"] == 0) | (counts["sum"] == counts["count"])]


def fit_acceptance_model(
    shadow_df: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, str]:
    """The plan's primary model: acceptance of standardized shadow offers.

    Saturated in Condition x Round x Direction (as a cell-means factor), with
    the plan's covariates -- offer attractiveness, counter good, and the
    latent-utility change from accepting -- and model family as a blocking
    factor.  Covariance is cluster-robust on independent run.  Perfectly
    separated cells are dropped from the fit and returned to the caller so
    the contrasts that would have used them can be reported as such.
    """
    data = shadow_df[shadow_df["round"].isin(PROBE_ROUNDS)].copy()
    dropped = separated_cells(data, ("condition", "round_f", "direction"), "accept")
    if len(dropped):
        keys = {
            (str(row.condition), str(row.round_f), str(row.direction))
            for row in dropped.itertuples()
        }
        keep = [
            (str(condition), str(round_f), str(direction)) not in keys
            for condition, round_f, direction in zip(
                data["condition"], data["round_f"], data["direction"]
            )
        ]
        data = data[keep]
    # The plan's reference cell may itself be one of the separated ones.
    reference = cell(REFERENCE_CONDITION, "r0", "acquire")
    surviving = {
        cell(str(condition), str(round_f), str(direction))
        for condition, round_f, direction in zip(
            data["condition"], data["round_f"], data["direction"]
        )
    }
    if reference not in surviving:
        reference = cell(REFERENCE_CONDITION, f"r{PRIMARY_ROUND}", "acquire")
    data = _prepare_cells(data, ("condition", "round_f", "direction"), reference)
    formula = (
        "accept ~ C(cell) + log_ratio + C(counter_good) + dlogU + C(model)"
    )
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": np.asarray(data["run_id"].astype("category").cat.codes)},
    )
    return result, data, dropped, reference


def fit_acceptance_glmm(data: pd.DataFrame) -> Any:
    """Bayesian binomial mixed model, run and Merchant-within-run intercepts.

    Robustness check only: it is the closest thing this stack has to the
    preregistered GLMM, and it reports posterior standard deviations rather
    than frequentist p-values.
    """
    frame = data.copy()
    frame["merchant"] = frame["run_id"] + "|" + frame["player_id"]
    formula = "accept ~ C(cell) + log_ratio + C(counter_good) + dlogU + C(model)"
    variance = {"run": "0 + C(run_id)", "merchant": "0 + C(merchant)"}
    model = sm.BinomialBayesMixedGLM.from_formula(formula, variance, frame)
    return model.fit_vb(verbose=False)


def fit_preference_lmm(
    probes_df: pd.DataFrame, outcome: str
) -> tuple[Any, pd.DataFrame]:
    """Linear mixed model for a Merchant stated-preference index.

    Fixed effects: Condition x Round (cell means) and B presentation
    position.  Random intercepts for run and for Merchant nested within run,
    as preregistered.
    """
    data = probes_df[
        (probes_df["role"] == "Merchant")
        & probes_df["round"].isin(PROBE_ROUNDS)
        & probes_df[outcome].notna()
    ]
    data = _prepare_cells(
        data, ("condition", "round_f", "model"), PREF_REFERENCE
    )
    data = data.copy()
    data["merchant"] = data["player_id"]
    formula = f"{outcome} ~ C(cell) + C(b_position)"
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["run_id"],
        re_formula="1",
        vc_formula={"merchant": "0 + C(merchant)"},
    )
    try:
        result = model.fit(reml=True, method="lbfgs")
        if not getattr(result, "converged", True):
            raise ValueError("did not converge")
    except (ValueError, np.linalg.LinAlgError):
        # Preregistered fallback: drop the lowest random intercept, keep run.
        print(
            f"  NOTE: {outcome}: Merchant-within-run variance component was "
            "singular; refitting with the run intercept only"
        )
        model = smf.mixedlm(formula, data=data, groups=data["run_id"], re_formula="1")
        result = model.fit(reml=True, method="lbfgs")
    return result, data


def fit_bundle_binomial(probes_df: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """B units out of six as a binomial GLM with run-clustered covariance.

    Stands in for the preregistered beta-binomial mixed model; the clustered
    sandwich absorbs the overdispersion the beta-binomial would have carried.
    """
    data = probes_df[
        (probes_df["role"] == "Merchant")
        & probes_df["round"].isin(PROBE_ROUNDS)
        & probes_df["bundle_B"].notna()
        & (probes_df["bundle_total"] == BUNDLE_TOTAL)
    ]
    data = _prepare_cells(
        data, ("condition", "round_f", "model"), PREF_REFERENCE
    )
    data = data.copy()
    data["successes"] = data["bundle_B"]
    data["failures"] = BUNDLE_TOTAL - data["bundle_B"]
    formula = "successes + failures ~ C(cell) + C(b_position)"
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": np.asarray(data["run_id"].astype("category").cat.codes)},
    )
    return result, data


def fit_state_dependent_lmm(probes_df: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """State-dependent valuation, as Prereg2 specifies it.

    Condition x Round x Model and B presentation position, random intercepts
    for run and Merchant within run, and no holdings covariate: Prereg2
    dropped the original plan's predicted-marginal-utility adjustment.  The
    index moves with holdings, which the bulletins change, so the report pairs
    this model with an exact-inventory comparison (``matched_valuation_rows``).
    """
    data = probes_df[
        (probes_df["role"] == "Merchant")
        & probes_df["round"].isin(PROBE_ROUNDS)
        & probes_df["Pb_inventory"].notna()
    ]
    data = _prepare_cells(
        data, ("condition", "round_f", "model"), PREF_REFERENCE
    )
    data = data.copy()
    data["merchant"] = data["player_id"]
    formula = "Pb_inventory ~ C(cell) + C(b_position)"
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["run_id"],
        re_formula="1",
        vc_formula={"merchant": "0 + C(merchant)"},
    )
    try:
        result = model.fit(reml=True, method="lbfgs")
        if not getattr(result, "converged", True):
            raise ValueError("did not converge")
    except (ValueError, np.linalg.LinAlgError):
        print(
            "  NOTE: state-dependent model: Merchant variance component was "
            "singular; refitting with the run intercept only"
        )
        model = smf.mixedlm(formula, data=data, groups=data["run_id"], re_formula="1")
        result = model.fit(reml=True, method="lbfgs")
    return result, data


def fit_reservation_lmm(reservation_df: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Reservation rung for surrendering B, Condition x Round cell means."""
    data = reservation_df[reservation_df["round"].isin(PROBE_ROUNDS)]
    data = _prepare_cells(
        data, ("condition", "round_f", "model"), PREF_REFERENCE
    )
    data = data.copy()
    data["merchant"] = data["player_id"]
    formula = "reservation ~ C(cell)"
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data["run_id"],
        re_formula="1",
        vc_formula={"merchant": "0 + C(merchant)"},
    )
    try:
        result = model.fit(reml=True, method="lbfgs")
        if not getattr(result, "converged", True):
            raise ValueError("did not converge")
    except (ValueError, np.linalg.LinAlgError):
        print(
            "  NOTE: reservation model: Merchant variance component was "
            "singular; refitting with the run intercept only"
        )
        model = smf.mixedlm(formula, data=data, groups=data["run_id"], re_formula="1")
        result = model.fit(reml=True, method="lbfgs")
    return result, data


# ---------------------------------------------------------------------------
# Planned contrasts
# ---------------------------------------------------------------------------

def _policy_cell(condition: str, round_index: int, direction: str) -> str:
    return cell_term("cell", cell(condition, f"r{round_index}", direction))


def _pref_cell(condition: str, round_index: int, model: str) -> str:
    return cell_term("cell", cell(condition, f"r{round_index}", model))


PREF_REFERENCE = cell(REFERENCE_CONDITION, "r0", MODELS[0])
RESERVATION_FAMILY = "exploratory reservation (dropped in Prereg2)"


def _pref_pair(
    left: str, right: str, round_index: int, sign: float = 1.0
) -> dict[str, float]:
    """``left`` minus ``right`` at one round, averaged over the three models.

    The stated-preference models are saturated in Condition x Round x Model,
    as Prereg2 specifies, so a pooled condition contrast is the equally
    weighted mean of the three model-specific contrasts -- the
    estimated-marginal-means contrast Prereg2 names.
    """
    weights: dict[str, float] = defaultdict(float)
    for model_name in MODELS:
        weights[_pref_cell(left, round_index, model_name)] += sign / len(MODELS)
        weights[_pref_cell(right, round_index, model_name)] -= sign / len(MODELS)
    return dict(weights)


def _pair_weights(
    cell_name: Any, left: tuple, right: tuple, sign: float = 1.0
) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    weights[cell_name(*left)] += sign
    weights[cell_name(*right)] -= sign
    return dict(weights)


def _merge(*weight_sets: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = defaultdict(float)
    for weights in weight_sets:
        for name, weight in weights.items():
            merged[name] += weight
    return {name: weight for name, weight in merged.items() if abs(weight) > 1e-12}


def policy_contrasts(
    result: Any, dropped: pd.DataFrame, reference: str
) -> list[dict[str, Any]]:
    """Every operational-policy contrast the plan names, in its three families.

    A contrast that would need a perfectly separated cell is reported as not
    estimable on the logit scale rather than silently dropped; the
    probability-scale companions below cover the same comparison.
    """
    rows: list[dict[str, Any]] = []
    missing = {
        cell(str(row.condition), str(row.round_f), str(row.direction))
        for row in dropped.itertuples()
    }
    reference_term = cell_term("cell", reference)

    def add(weights: dict[str, float], label: str, family: str) -> None:
        needed = {
            name[len("C(cell)[T.") : -1]
            for name in weights
            if name.startswith("C(cell)[T.")
        }
        blocked = sorted(needed & missing)
        if blocked:
            rows.append(
                {
                    "label": label,
                    "family": family,
                    "estimate": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci90_low": np.nan,
                    "ci90_high": np.nan,
                    "note": "not estimable on the logit scale; separated cell(s) "
                    + ", ".join(blocked),
                }
            )
            return
        weights = {
            name: weight for name, weight in weights.items() if name != reference_term
        }
        rows.append(contrast(result, weights, label, family, exponentiate=True))

    pairs = [("Scarcity", "Control"), ("Expert", "Control"), ("Expert", "Scarcity")]

    # Family 1: the primary round-10 condition contrasts, by direction.
    for direction in DIRECTIONS:
        for left, right in pairs:
            add(
                _pair_weights(
                    _policy_cell,
                    (left, PRIMARY_ROUND, direction),
                    (right, PRIMARY_ROUND, direction),
                ),
                f"{DIRECTION_LABEL[direction]} @ r{PRIMARY_ROUND}: {left} - {right}",
                "primary policy",
            )
    # Family 1 also holds the plan's difference-in-differences, round 0 to 10.
    for direction in DIRECTIONS:
        for left, right in pairs:
            add(
                _merge(
                    _pair_weights(
                        _policy_cell,
                        (left, PRIMARY_ROUND, direction),
                        (right, PRIMARY_ROUND, direction),
                    ),
                    _pair_weights(
                        _policy_cell,
                        (left, 0, direction),
                        (right, 0, direction),
                        sign=-1.0,
                    ),
                ),
                f"{DIRECTION_LABEL[direction]} DiD r0->r{PRIMARY_ROUND}: "
                f"{left} - {right}",
                "primary policy",
            )
    # B-protection index: the acquire-minus-surrender gap, in log odds.
    for left, right in pairs:
        add(
            _merge(
                _pair_weights(
                    _policy_cell,
                    (left, PRIMARY_ROUND, "acquire"),
                    (right, PRIMARY_ROUND, "acquire"),
                ),
                _pair_weights(
                    _policy_cell,
                    (left, PRIMARY_ROUND, "surrender"),
                    (right, PRIMARY_ROUND, "surrender"),
                    sign=-1.0,
                ),
            ),
            f"B-protection index @ r{PRIMARY_ROUND}: {left} - {right}",
            # Prereg2 makes the index descriptive, so it has its own family
            # and does not enlarge the confirmatory one.
            "descriptive B-protection index",
        )

    # Family 3: persistence after the bulletin is withdrawn.
    for direction in DIRECTIONS:
        for left, right in pairs:
            add(
                _pair_weights(
                    _policy_cell,
                    (left, PERSISTENCE_ROUND, direction),
                    (right, PERSISTENCE_ROUND, direction),
                ),
                f"{DIRECTION_LABEL[direction]} @ r{PERSISTENCE_ROUND}: "
                f"{left} - {right}",
                "washout",
            )
    for direction in DIRECTIONS:
        for treatment in TREATMENTS:
            add(
                _merge(
                    _pair_weights(
                        _policy_cell,
                        (treatment, PERSISTENCE_ROUND, direction),
                        (REFERENCE_CONDITION, PERSISTENCE_ROUND, direction),
                    ),
                    _pair_weights(
                        _policy_cell,
                        (treatment, PRIMARY_ROUND, direction),
                        (REFERENCE_CONDITION, PRIMARY_ROUND, direction),
                        sign=-1.0,
                    ),
                ),
                f"{DIRECTION_LABEL[direction]} decay r{PRIMARY_ROUND}->"
                f"r{PERSISTENCE_ROUND}: {treatment} - Control",
                "washout",
            )
    return rows


def run_level_difference(
    frame: pd.DataFrame,
    value_col: str,
    left: str,
    right: str,
    round_index: int,
    *,
    baseline_round: int | None = None,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 20260917,
) -> dict[str, Any]:
    """Difference of run-level means, with a Welch CI and a permutation p.

    The plan's unit of independence is the simulation run, so a comparison
    taken at that level needs no distributional assumption about the
    individual observations.  This is how the probability-scale policy
    contrasts and the robustness checks on the near-degenerate stated
    measures are computed: it stays valid where a model whose residual
    variance is zero in one condition does not.  With ``baseline_round`` the
    quantity differenced is each run's change from that round.
    """
    rounds = [round_index] if baseline_round is None else [baseline_round, round_index]
    subset = frame[
        frame["round"].isin(rounds) & frame["condition"].isin([left, right])
    ]
    per_run = (
        subset.groupby(["condition", "run_id", "round"], observed=True)[value_col]
        .mean()
        .unstack("round")
    )
    if baseline_round is None:
        values = per_run[round_index].dropna()
    else:
        values = (per_run[round_index] - per_run[baseline_round]).dropna()
    frames = values.reset_index()
    frames.columns = ["condition", "run_id", "value"]
    a = frames[frames["condition"] == left]["value"].to_numpy(dtype=float)
    b = frames[frames["condition"] == right]["value"].to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"estimate": np.nan, "se": np.nan, "p": np.nan, "p_perm": np.nan}
    estimate = float(a.mean() - b.mean())
    standard_error = float(
        math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    )
    if standard_error > 0:
        welch = stats.ttest_ind(a, b, equal_var=False)
        p_value = float(welch.pvalue)
    else:
        p_value = np.nan

    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    if np.ptp(pooled) == 0:
        p_perm = 1.0
    else:
        draws = np.empty(n_permutations)
        for index in range(n_permutations):
            shuffled = rng.permutation(pooled)
            draws[index] = shuffled[: len(a)].mean() - shuffled[len(a) :].mean()
        p_perm = float((np.abs(draws) >= abs(estimate) - 1e-12).mean())

    return {
        "estimate": estimate,
        "se": standard_error,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "p": p_value,
        "p_perm": p_perm,
        "n_left": len(a),
        "n_right": len(b),
    }


def policy_risk_differences(shadow_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Probability-scale companions to the logistic policy contrasts.

    These are the contrasts the plan asks for, computed on the scale that
    survives the round-0 structural zero: a run-level risk difference in
    P(accept), at round 10 and as a change from round 0.
    """
    rows: list[dict[str, Any]] = []
    pairs = [("Scarcity", "Control"), ("Expert", "Control"), ("Expert", "Scarcity")]
    for direction in DIRECTIONS:
        subset = shadow_df[shadow_df["direction"] == direction]
        for left, right in pairs:
            level = run_level_difference(
                subset, "accept", left, right, PRIMARY_ROUND
            )
            rows.append(
                {
                    "label": f"{DIRECTION_LABEL[direction]} @ r{PRIMARY_ROUND} "
                    f"(risk diff): {left} - {right}",
                    "family": "primary policy (risk difference)",
                    **level,
                }
            )
            did = run_level_difference(
                subset, "accept", left, right, PRIMARY_ROUND, baseline_round=0
            )
            rows.append(
                {
                    "label": f"{DIRECTION_LABEL[direction]} DiD r0->r"
                    f"{PRIMARY_ROUND} (risk diff): {left} - {right}",
                    "family": "primary policy (risk difference)",
                    **did,
                }
            )
    return rows


def off_plan_round_rows(
    shadow_df: pd.DataFrame,
    probes_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    round_index: int,
    family: str,
) -> list[dict[str, Any]]:
    """Run-level condition contrasts at a round the plan does not test.

    Used for two rounds.  Round 0, the pre-treatment baseline: no bulletin has
    been shown and every Merchant still holds A1 B1 C1, so these contrasts are
    a balance check and should be null.  Round 12, two rounds into the
    washout: the first look at behaviour once the bulletin is gone.
    Run-level throughout, because several of these cells have no accepts at
    all.
    """
    merchants = probes_df[probes_df["role"] == "Merchant"]
    outcomes = [
        ("Acquire B", shadow_df[shadow_df["direction"] == "acquire"], "accept"),
        ("Surrender B", shadow_df[shadow_df["direction"] == "surrender"], "accept"),
        ("B-protection index", policy_df, "Policy_B"),
        ("Pb_general", merchants, "Pb_general"),
        ("Pb_inventory", merchants, "Pb_inventory"),
    ]
    rows: list[dict[str, Any]] = []
    for name, frame, column in outcomes:
        for left, right in (
            ("Scarcity", "Control"),
            ("Expert", "Control"),
            ("Expert", "Scarcity"),
        ):
            rows.append(
                {
                    "label": f"{name} @ r{round_index}: {left} - {right}",
                    "family": family,
                    **run_level_difference(frame, column, left, right, round_index),
                }
            )
    return rows


def preference_contrasts(
    result: Any, outcome_label: str, family: str, bound: float | None
) -> list[dict[str, Any]]:
    """Round-10, DiD, round-14 and round 10 -> 14 contrasts, plus TOST.

    Every contrast is averaged over the three model families (see
    ``_pref_pair``).
    """
    rows: list[dict[str, Any]] = []
    pairs = [("Scarcity", "Control"), ("Expert", "Control"), ("Expert", "Scarcity")]
    reference_term = cell_term("cell", PREF_REFERENCE)

    def add(weights: dict[str, float], label: str, fam: str) -> dict[str, Any]:
        weights = {
            name: weight
            for name, weight in weights.items()
            if name != reference_term and abs(weight) > 1e-12
        }
        row = contrast(result, weights, label, fam)
        if bound is not None:
            row.update(tost(row["estimate"], row["se"], bound))
            row["equiv_bound"] = bound
            row["equivalent"] = bool(
                np.isfinite(row["p_tost"]) and row["p_tost"] < ALPHA
            )
        rows.append(row)
        return row

    for round_index, fam in (
        (PRIMARY_ROUND, family),
        (PERSISTENCE_ROUND, "washout"),
    ):
        for left, right in pairs:
            add(
                _pref_pair(left, right, round_index),
                f"{outcome_label} @ r{round_index}: {left} - {right}",
                fam,
            )
    for left, right in pairs:
        add(
            _merge(
                _pref_pair(left, right, PRIMARY_ROUND),
                _pref_pair(left, right, 0, sign=-1.0),
            ),
            f"{outcome_label} DiD r0->r{PRIMARY_ROUND}: {left} - {right}",
            family,
        )
    # Prereg2 also asks for round 10 -> 14 changes on these outcomes.
    for treatment in TREATMENTS:
        add(
            _merge(
                _pref_pair(treatment, REFERENCE_CONDITION, PERSISTENCE_ROUND),
                _pref_pair(treatment, REFERENCE_CONDITION, PRIMARY_ROUND, sign=-1.0),
            ),
            f"{outcome_label} decay r{PRIMARY_ROUND}->r{PERSISTENCE_ROUND}: "
            f"{treatment} - Control",
            "washout",
        )
    return rows


def matched_gap(
    probes_df: pd.DataFrame,
    value_col: str,
    treatment: str,
    round_index: int,
) -> dict[str, Any]:
    """Treatment minus Control at identical inventories, at one round.

    Each treated Merchant is compared with the mean of Control Merchants that
    hold exactly the same A, B and C at the same round, so no assumption is
    needed about how holdings map to ratings.  Treated Merchants whose holding
    never occurs in Control are dropped (the share kept is returned).  The
    interval treats the run as the unit: differences are averaged within run
    before the across-run mean and its normal-approximation 95% interval.
    """
    merchants = probes_df[
        (probes_df["role"] == "Merchant") & (probes_df["round"] == round_index)
    ]
    treated = merchants[merchants["condition"] == treatment]
    control = merchants[merchants["condition"] == REFERENCE_CONDITION]
    reference = control.groupby("inventory_state", observed=True)[value_col].mean()
    kept = treated[treated["inventory_state"].isin(reference.index)]
    if kept.empty:
        return {"estimate": np.nan, "se": np.nan, "p": np.nan}
    differences = kept[value_col].to_numpy(dtype=float) - reference.loc[
        kept["inventory_state"]
    ].to_numpy(dtype=float)
    per_run = pd.Series(differences, index=kept["run_id"].to_numpy()).groupby(
        level=0
    ).mean()
    estimate = float(per_run.mean())
    standard_error = (
        float(per_run.std(ddof=1) / math.sqrt(len(per_run)))
        if len(per_run) > 1
        else np.nan
    )
    z = estimate / standard_error if standard_error and standard_error > 0 else np.nan
    return {
        "estimate": estimate,
        "se": standard_error,
        "ci_low": estimate - 1.96 * standard_error,
        "ci_high": estimate + 1.96 * standard_error,
        "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        "share_matched": len(kept) / len(treated) if len(treated) else np.nan,
    }


def matched_valuation_rows(probes_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Exact-inventory gaps for the state-dependent index and its three ratings.

    Robustness for the H4.4 model, which cannot separate the bulletin from the
    holdings it changes, and the post hoc check of which ratings move.
    """
    rows: list[dict[str, Any]] = []
    outcomes = [
        ("Pb_inventory", "state-dependent B index"),
        ("rating_inventory_B", "rating of B"),
        ("rating_inventory_A", "rating of A"),
        ("rating_inventory_C", "rating of C"),
    ]
    for column, name in outcomes:
        family = (
            "state-dependent robustness (exact inventory match)"
            if column == "Pb_inventory"
            else "exploratory rating components (exact inventory match)"
        )
        for round_index in (5, PRIMARY_ROUND, 12, PERSISTENCE_ROUND):
            for treatment in TREATMENTS:
                rows.append(
                    {
                        "label": f"{name} @ r{round_index} (matched): "
                        f"{treatment} - Control",
                        "family": family,
                        **matched_gap(probes_df, column, treatment, round_index),
                    }
                )
    return rows


def per_model_policy_contrasts(shadow_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Exploratory: round-0 and round-10 policy contrasts within each model family.

    These are run-level risk differences rather than a per-family refit of the
    logistic model.  Within a single family the latent-utility covariate the
    plan requires quasi-separates acceptance -- for Google, ``dlogU`` picks up
    a coefficient of about 550 with a standard error in the hundreds of
    thousands, which drives the cluster-robust covariance to NaN and makes
    every contrast from that fit uninterpretable.  A run-level difference in
    P(accept) needs no such covariate, is estimable in all three families, and
    is on a scale that can be compared across families with different base
    rates.  The plan declares no confirmatory model-by-condition test, so
    nothing confirmatory rests on this.
    """
    rows: list[dict[str, Any]] = []
    for round_index in (0, PRIMARY_ROUND):
        for model_name in MODELS:
            subset = shadow_df[shadow_df["model"] == model_name]
            for direction in DIRECTIONS:
                block = subset[subset["direction"] == direction]
                for treatment in TREATMENTS:
                    result = run_level_difference(
                        block,
                        "accept",
                        treatment,
                        REFERENCE_CONDITION,
                        round_index,
                        n_permutations=4000,
                    )
                    rows.append(
                        {
                            "label": f"{model_name} | "
                            f"{DIRECTION_LABEL[direction]} @ r{round_index}: "
                            f"{treatment} - Control",
                            "family": "exploratory per-model",
                            "model": model_name,
                            "direction": direction,
                            "treatment": treatment,
                            "round": round_index,
                            **result,
                        }
                    )
    return rows


def degenerate_contrast_rows(
    outcome_label: str, note: str, bound: float | None
) -> list[dict[str, Any]]:
    """Contrast rows for a measure that never varies, stated as exactly zero.

    Fitting a model to a constant would manufacture standard errors out of
    nothing, so the rows say what is true: the difference is zero, exactly,
    and if the plan set an equivalence bound the zero sits inside it.
    """
    rows: list[dict[str, Any]] = []
    pairs = [("Scarcity", "Control"), ("Expert", "Control"), ("Expert", "Scarcity")]
    specifications = [(PRIMARY_ROUND, "general preference"), (PERSISTENCE_ROUND, "washout")]
    for round_index, family in specifications:
        for left, right in pairs:
            rows.append(
                {
                    "label": f"{outcome_label} @ r{round_index}: {left} - {right}",
                    "family": family,
                    "estimate": 0.0,
                    "se": 0.0,
                    "z": np.nan,
                    "p": np.nan,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "ci90_low": 0.0,
                    "ci90_high": 0.0,
                    "note": note,
                    "degenerate": True,
                    **(
                        {
                            "equiv_bound": bound,
                            "equivalent": True,
                            "p_tost": 0.0,
                            "p_lower": 0.0,
                            "p_upper": 0.0,
                        }
                        if bound is not None
                        else {}
                    ),
                }
            )
    for left, right in pairs:
        rows.append(
            {
                "label": f"{outcome_label} DiD r0->r{PRIMARY_ROUND}: {left} - {right}",
                "family": "general preference",
                "estimate": 0.0,
                "se": 0.0,
                "z": np.nan,
                "p": np.nan,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "ci90_low": 0.0,
                "ci90_high": 0.0,
                "note": note,
                "degenerate": True,
                **(
                    {
                        "equiv_bound": bound,
                        "equivalent": True,
                        "p_tost": 0.0,
                        "p_lower": 0.0,
                        "p_upper": 0.0,
                    }
                    if bound is not None
                    else {}
                ),
            }
        )
    return rows


def preference_robustness_rows(
    probes_df: pd.DataFrame, outcome: str, outcome_label: str
) -> list[dict[str, Any]]:
    """Run-level Welch and permutation checks on a stated-preference contrast.

    The general-importance index has zero variance in Control at every round,
    so the mixed model's pooled residual variance is not a description of
    either condition.  These rows make the same comparison without that
    assumption.
    """
    merchants = probes_df[probes_df["role"] == "Merchant"]
    rows: list[dict[str, Any]] = []
    pairs = [("Scarcity", "Control"), ("Expert", "Control"), ("Expert", "Scarcity")]
    for round_index in (PRIMARY_ROUND, PERSISTENCE_ROUND):
        for left, right in pairs:
            result = run_level_difference(
                merchants, outcome, left, right, round_index
            )
            rows.append(
                {
                    "label": f"{outcome_label} @ r{round_index} (run level): "
                    f"{left} - {right}",
                    "family": "stated-preference robustness",
                    **result,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# H3: policy against stated preference on a common scale
# ---------------------------------------------------------------------------

def common_scale_contrast(
    frame: pd.DataFrame,
    value_col: str,
    treatment: str,
    scale: float,
    round_index: int,
) -> dict[str, Any]:
    """Treatment minus Control at one round, as a share of the instrument range.

    The plan asks whether the operational-policy effect exceeds the
    stated-preference effect, which needs a scale both can be read on.  The
    width of the measuring instrument exists for every measure and assumes
    nothing about the sample, unlike a baseline SD, which is zero for several
    of them.  The quantity is the level contrast, not a change from round 0:
    conditions are randomized and every run starts from the same state, so the
    level contrast is the treatment effect.
    """
    result = run_level_difference(
        frame, value_col, treatment, REFERENCE_CONDITION, round_index,
        n_permutations=2000,
    )
    if not np.isfinite(result["estimate"]):
        return {**result, "share": np.nan, "share_low": np.nan, "share_high": np.nan}
    return {
        **result,
        "scale": scale,
        "share": result["estimate"] / scale,
        "share_low": result["ci_low"] / scale,
        "share_high": result["ci_high"] / scale,
    }


def dissociation_table(
    shadow_df: pd.DataFrame,
    probes_df: pd.DataFrame,
    policy_df: pd.DataFrame,
) -> pd.DataFrame:
    """H3: the treatment effect on every measure, on one common scale."""
    merchants = probes_df[probes_df["role"] == "Merchant"]
    acquire = shadow_df[shadow_df["direction"] == "acquire"]
    surrender = shadow_df[shadow_df["direction"] == "surrender"]

    outcomes: list[tuple[str, str, pd.DataFrame, str]] = [
        ("Policy: accept acquiring B", "operational", acquire, "accept"),
        ("Policy: accept surrendering B", "operational", surrender, "accept"),
        ("Policy: B-protection index", "operational", policy_df, "Policy_B"),
        ("Preference: general-importance B", "stated", merchants, "Pb_general"),
        ("Preference: ideal-bundle B units", "stated", merchants, "bundle_B"),
        (
            "Valuation: state-dependent B (raw)",
            "state-dependent",
            merchants,
            "Pb_inventory",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for label, kind, subset, column in outcomes:
        for treatment in TREATMENTS:
            for round_index in (0, PRIMARY_ROUND):
                rows.append(
                    {
                        "outcome": label,
                        "kind": kind,
                        "treatment": treatment,
                        "round": round_index,
                        "measure": column,
                        **common_scale_contrast(
                            subset,
                            column,
                            treatment,
                            INSTRUMENT_RANGE[column],
                            round_index,
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

REPORT_WIDTH = 100
PAIRS = [("Scarcity", "Control"), ("Expert", "Control"), ("Expert", "Scarcity")]


def mean_ci(values: Any) -> tuple[float, float, float]:
    """Mean and a normal-approximation 95% CI over the finite values."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(array, ddof=1) / math.sqrt(len(array)))
    return mean, mean - half_width, mean + half_width


def note(text: str, indent: int = 2) -> None:
    """Print a paragraph of annotation, wrapped to the report width."""
    prefix = " " * indent
    print(
        textwrap.fill(
            " ".join(text.split()),
            width=REPORT_WIDTH,
            initial_indent=prefix,
            subsequent_indent=prefix,
        )
    )


def section(number: str, title: str) -> None:
    print("\n")
    print("=" * REPORT_WIDTH)
    print(f"{number}. {title}" if number else title)
    print("=" * REPORT_WIDTH)


def subsection(title: str) -> None:
    print(f"\n{title}")
    print("-" * min(len(title), REPORT_WIDTH))


def fmt_p(p: Any) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _finite(row: dict[str, Any] | None, key: str) -> bool:
    return row is not None and np.isfinite(row.get(key, np.nan))


def _stars(row: dict[str, Any]) -> str:
    p = row.get("p_holm", np.nan)
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return ""


def effect_text(row: dict[str, Any], odds: bool) -> str:
    """Estimate and 95% CI: an odds ratio for logistic rows, else a difference."""
    if odds and "odds_ratio" in row:
        return (
            f"OR {row['odds_ratio']:.3g} [{row['or_low']:.3g}, "
            f"{row['or_high']:.3g}]"
        )
    return f"{row['estimate']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"


def short_effect(row: dict[str, Any] | None) -> str:
    """Estimate with its Holm p, for the one-line summaries."""
    if row is None:
        return "missing"
    if row.get("degenerate"):
        return "exactly 0"
    if not _finite(row, "se"):
        return "not estimable"
    value = (
        f"OR {row['odds_ratio']:.3g}" if "odds_ratio" in row
        else f"{row['estimate']:+.3f}"
    )
    return f"{value} (p_holm {fmt_p(row.get('p_holm'))})"


def print_contrasts(
    rows: Sequence[dict[str, Any]],
    title: str,
    odds: bool,
    notes: Sequence[str] = (),
) -> None:
    """One annotated table of contrasts.

    Columns: significance stars (Holm-adjusted p only), the contrast label as
    it appears in ``contrasts.csv``, the estimate with its 95% CI, the raw p,
    the Holm-adjusted p where the row belongs to a Holm family, and the
    permutation p where one was computed.  Anything else -- an equivalence
    test, a matching rate, a reason a row has no test -- follows at the end.
    """
    if not rows:
        return
    subsection(title)
    for text in notes:
        note(text)
    width = max(len(row["label"]) for row in rows)
    effect_width = 30
    print(
        f"\n  {'':3s} {'contrast':<{width}s}  {'estimate [95% CI]':>{effect_width}s}"
        f"  {'p':>7s}  {'p_holm':>7s}  {'p_perm':>7s}"
    )
    for row in rows:
        lead = f"  {_stars(row):3s} {row['label']:<{width}s}  "
        if row.get("degenerate"):
            detail = "exactly 0: the measure never varies"
            if row.get("equiv_bound") is not None:
                detail += f"; inside +/-{row['equiv_bound']:.1f} by inspection"
            print(lead + detail)
            continue
        if not _finite(row, "estimate") or (
            row.get("note") and not _finite(row, "se")
        ):
            print(lead + str(row.get("note") or "not estimable: too few runs"))
            continue
        if not _finite(row, "se"):
            print(
                lead
                + f"{row['estimate']:+.3f}  no test: zero standard error "
                "(the outcome is almost constant)"
            )
            continue
        if row["se"] == 0:
            print(
                lead
                + f"{row['estimate']:>+{effect_width}.3f}  no test: every run "
                "in both groups has the same value"
            )
            continue
        line = (
            lead
            + f"{effect_text(row, odds):>{effect_width}s}"
            + f"  {fmt_p(row.get('p')):>7s}"
            + f"  {fmt_p(row['p_holm']) if _finite(row, 'p_holm') else '':>7s}"
            + f"  {fmt_p(row['p_perm']) if _finite(row, 'p_perm') else '':>7s}"
        )
        if _finite(row, "p_tost"):
            verdict = "equivalent" if row["equivalent"] else "NOT shown equivalent"
            line += f"  TOST {fmt_p(row['p_tost'])}: {verdict}"
        if _finite(row, "share_matched"):
            line += f"  ({row['share_matched']:.0%} of treated Merchants matched)"
        print(line.rstrip())


def print_legend() -> None:
    section("", "HOW TO READ THIS REPORT")
    entries = [
        (
            "Unit",
            "The independent simulation run (40 per model x condition cell). "
            "Every descriptive mean and interval below is taken over run-level "
            "means, never over raw observations.",
        ),
        (
            "Rounds",
            "Probes are taken at rounds 0, 5, 10, 12 and 14. The bulletin is "
            "shown on rounds 1-10; round 0 is the pre-treatment baseline and "
            "rounds 11-14 are the washout. r10 is round 10, the primary time "
            "point; r14 is the end of the washout.",
        ),
        (
            "Labels",
            "'<outcome> @ r10: Expert - Control' is Expert minus Control at "
            "round 10. 'DiD r0->r10' is the round 0 -> 10 change in the first "
            "condition minus the same change in the second. 'decay r10->r14' "
            "is the gap at round 14 minus the gap at round 10, so an effect "
            "that fades has a decay of the opposite sign (an odds ratio on "
            "the other side of 1). Labels match the 'label' column of "
            "contrasts.csv.",
        ),
        (
            "Estimates",
            "OR is an odds ratio from the logistic acceptance model: above 1 "
            "means the first condition accepts more often. Any other estimate "
            "is a difference on the measure's own scale: a probability for "
            "acceptance (0.087 = 8.7 percentage points), rating points for "
            "the B indices, units of B for the ideal bundle.",
        ),
        (
            "Intervals",
            "95% confidence intervals, in square brackets.",
        ),
        (
            "p-values",
            "p is the raw two-sided p-value. p_holm is Holm-adjusted within "
            "the family named in the table title (primary policy, general "
            "preference, washout, plus a few descriptive families); a blank "
            "p_holm means the row is outside every family. p_perm is a "
            "permutation test on run-level means.",
        ),
        (
            "Stars",
            "*** p_holm < .001, ** < .01, * < .05. Rows outside a Holm family "
            "never get stars, however small their raw p.",
        ),
        (
            "TOST",
            "Two one-sided tests against the preregistered +/-0.5 bound. "
            "'equivalent' (TOST p < .05) means the 90% CI lies wholly inside "
            "the bound. A difference can be both significant and equivalent: "
            "real, but too small to matter by the plan's standard.",
        ),
        (
            "Measures",
            "Pb_general: general importance rating of B minus the mean of A "
            "and C (1-10 scale, so -9 to +9). bundle_B: units of B in the "
            "Merchant's ideal six-unit bundle. Pb_inventory: the same index "
            "for ratings given the Merchant's current inventory. "
            "B-protection index (Policy_B): P(accept acquiring B) minus "
            "P(accept surrendering B). Reservation: the first rung (1 or 2 "
            "units, 3 = never) at which a Merchant gives up a unit of B.",
        ),
    ]
    for name, text in entries:
        lines = textwrap.wrap(" ".join(text.split()), width=REPORT_WIDTH - 14)
        print(f"  {name:<11s} {lines[0]}")
        for line in lines[1:]:
            print(f"  {'':<11s} {line}")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _policy_row(by_label: dict[str, Any], direction: str, round_label: str,
                treatment: str, risk: bool = False) -> dict[str, Any] | None:
    tag = " (risk diff)" if risk else ""
    return by_label.get(
        f"{DIRECTION_LABEL[direction]} {round_label}{tag}: {treatment} - Control"
    )


def _bundle_label(by_label: dict[str, Any]) -> str:
    return (
        "bundle_B (units)"
        if f"bundle_B (units) @ r{PRIMARY_ROUND}: Scarcity - Control" in by_label
        else "bundle_B"
    )


def print_summary(rows: Sequence[dict[str, Any]]) -> None:
    """The four hypotheses, each read off its confirmatory tests.

    Every verdict here is computed from the rows reported in full later on;
    the decision rule is printed next to each one.
    """
    by_label = {row["label"]: row for row in rows}
    section("", "SUMMARY OF CONFIRMATORY RESULTS")
    note(
        "Each line is computed from the tables in sections 5-8, using the "
        "decision rule stated with it. Section numbers point to the full "
        "tables."
    )

    subsection(
        f"H1  The bulletin moves operational policy (round {PRIMARY_ROUND}; "
        "section 5)"
    )
    note(
        "Rule: supported when the logistic contrast against Control is "
        "significant after Holm (family 'primary policy') in the predicted "
        "direction. The run-level risk difference is shown beneath it; it "
        "does not adjust for offer terms or utility change, so the two can "
        "disagree."
    )
    for treatment in TREATMENTS:
        print(f"\n    {treatment} - Control")
        for number, direction, predicted, wording in (
            ("H1.1", "acquire", 1.0, "accepting offers that give B goes UP"),
            ("H1.2", "surrender", -1.0, "accepting offers that take B goes DOWN"),
        ):
            row = _policy_row(by_label, direction, f"@ r{PRIMARY_ROUND}", treatment)
            risk = _policy_row(
                by_label, direction, f"@ r{PRIMARY_ROUND}", treatment, risk=True
            )
            supported = bool(
                _finite(row, "p_holm")
                and row["p_holm"] < ALPHA
                and np.sign(row["estimate"]) == predicted
            )
            verdict = "SUPPORTED" if supported else "not supported"
            print(f"      {number} {wording:<40s} {verdict}")
            print(f"           logistic model   {short_effect(row)}")
            if risk is not None and _finite(risk, "estimate"):
                print(
                    f"           risk difference  {risk['estimate'] * 100:+.1f} "
                    f"percentage points (p_holm {fmt_p(risk.get('p_holm'))})"
                )

    subsection("H2  General stated preference does not move (section 6)")
    note(
        "Rule: supported when every round-10 and round 0 -> 10 contrast "
        "against Control is equivalent to zero within +/-0.5 (TOST p < .05). "
        "A contrast that is also significantly different from zero is "
        "flagged: the change is real but inside the bound."
    )
    bundle = _bundle_label(by_label)
    for treatment in TREATMENTS:
        checks = []
        for number, outcome, name in (
            ("H2.1", bundle, "ideal bundle, B units"),
            ("H2.2", "Pb_general", "general-importance index"),
        ):
            for kind in (f"@ r{PRIMARY_ROUND}", f"DiD r0->r{PRIMARY_ROUND}"):
                row = by_label.get(f"{outcome} {kind}: {treatment} - Control")
                checks.append((number, name, kind, row))
        supported = all(
            row is not None and bool(row.get("equivalent")) for *_, row in checks
        )
        print(
            f"\n    {treatment} - Control: "
            + ("SUPPORTED" if supported else "NOT supported")
        )
        for number, name, kind, row in checks:
            if row is None:
                print(f"      {number} {name:<26s} {kind:<11s} missing")
                continue
            if row.get("degenerate"):
                print(
                    f"      {number} {name:<26s} {kind:<11s} exactly 0 "
                    "(never varies): equivalent"
                )
                continue
            status = "equivalent" if row.get("equivalent") else "NOT equivalent"
            flag = (
                "  <- also significantly different from 0"
                if _finite(row, "p_holm") and row["p_holm"] < ALPHA
                else ""
            )
            print(
                f"      {number} {name:<26s} {kind:<11s} "
                f"{row['estimate']:+.3f} [{row['ci90_low']:+.3f}, "
                f"{row['ci90_high']:+.3f}] 90% CI: {status}{flag}"
            )

    subsection("H3  Policy moves while stated preference stays put (section 7)")
    note(
        "Rule (Prereg2): supported for a treatment when at least one H1 "
        "contrast is supported and every H2 contrast is equivalent."
    )
    for verdict in h3_verdicts(rows):
        policy = [direction for direction, _, flag in verdict["policy"] if flag]
        stated = all(flag for _, _, flag in verdict["preference"])
        print(
            f"\n    {verdict['treatment']} - Control: "
            + ("SUPPORTED" if verdict["supported"] else "NOT supported")
        )
        print(
            "      policy effect in the predicted direction: "
            + (", ".join(policy) if policy else "none")
        )
        print(
            "      stated preference equivalent to Control: "
            + ("yes, on every contrast" if stated else "no")
        )

    subsection(
        f"H4  Effects weaken once the bulletin is withdrawn "
        f"(r{PRIMARY_ROUND} vs r{PERSISTENCE_ROUND}; section 8)"
    )
    note(
        f"Each cell is the treatment - Control gap with its Holm p. The "
        f"round-{PRIMARY_ROUND} gap is from the H1/H2 families; the "
        f"round-{PERSISTENCE_ROUND} gap and the change are from the washout "
        "family. The reading states the facts H4 is judged on: whether there "
        "was an effect to weaken, whether it is still detectable at the end "
        "of the washout, and whether its decline is itself significant."
    )
    outcomes = [
        ("acquiring B", "Acquire B"),
        ("surrendering B", "Surrender B"),
        ("Pb_general", "Pb_general"),
        ("Pb_inventory", "Pb_inventory"),
    ]
    print(
        f"\n    {'outcome':<26s} {'gap at r' + str(PRIMARY_ROUND):<28s}"
        f"{'gap at r' + str(PERSISTENCE_ROUND):<28s}"
        f"{'change r10->r14'}"
    )
    for name, prefix in outcomes:
        for treatment in TREATMENTS:
            on = by_label.get(f"{prefix} @ r{PRIMARY_ROUND}: {treatment} - Control")
            off = by_label.get(
                f"{prefix} @ r{PERSISTENCE_ROUND}: {treatment} - Control"
            )
            change = by_label.get(
                f"{prefix} decay r{PRIMARY_ROUND}->r{PERSISTENCE_ROUND}: "
                f"{treatment} - Control"
            )
            print(
                f"    {name + ', ' + treatment:<26s} {short_effect(on):<28s}"
                f"{short_effect(off):<28s}{short_effect(change)}"
            )
            print(f"    {'':<26s} -> {h4_reading(on, off, change)}")


def _significant(row: dict[str, Any] | None) -> bool:
    return _finite(row, "p_holm") and row["p_holm"] < ALPHA


def h4_reading(on: Any, off: Any, change: Any) -> str:
    if not _significant(on):
        return f"no significant effect at r{PRIMARY_ROUND}, so nothing to weaken"
    parts = [
        f"effect at r{PRIMARY_ROUND}",
        (
            f"still significant at r{PERSISTENCE_ROUND}"
            if _significant(off)
            else f"no longer significant at r{PERSISTENCE_ROUND}"
        ),
    ]
    declined = _significant(change) and np.sign(change["estimate"]) == -np.sign(
        on["estimate"]
    )
    parts.append(
        "the decline is significant" if declined else "the decline is not significant"
    )
    return "; ".join(parts)


def print_data_section(data: dict[str, pd.DataFrame], base_dir: Path) -> None:
    probes_df, shadow_df, runs_df = data["probes"], data["shadow"], data["runs"]
    section("1", "DATA AND EXCLUSIONS")
    cells = runs_df.groupby(["model", "condition"], observed=True).size()
    print(f"  Source:             {base_dir}")
    per_cell = (
        f"{cells.min()}" if cells.min() == cells.max()
        else f"{cells.min()}-{cells.max()}"
    )
    print(f"  Runs analysed:      {len(runs_df)} ({per_cell} per model x condition cell)")
    excluded = data["excluded"]
    print(f"  Runs excluded:      {len(excluded)}")
    for row in excluded.itertuples():
        print(f"    {row.run_dir}: {row.reason}")
    note(
        "A run is excluded if a log file is missing or unreadable, if the "
        "agents' roles are not as prespecified, or if the bulletin audit in "
        "section 2 shows the assigned condition was not applied.",
        indent=4,
    )
    print(
        "\n  Runs per cell:      "
        + "   ".join(
            f"{model}: "
            + "/".join(str(cells.get((model, c), 0)) for c in CONDITIONS)
            for model in MODELS
        )
        + "   (Control/Scarcity/Expert)"
    )

    expected = len(runs_df) * len(ROLE_BY_PLAYER) * len(PROBE_ROUNDS)
    print(
        f"\n  Preference probes:  {len(probes_df)} of {expected} expected "
        f"({len(probes_df[probes_df['role'] == 'Merchant'])} from Merchants)"
    )
    print(f"    dropped for an unparseable response: {expected - len(probes_df)}")
    order = probes_df.groupby("b_position", observed=True).size()
    print(
        "    B display position: "
        + ", ".join(f"{int(key)}: {value}" for key, value in order.items())
        + " (counterbalanced; a covariate in the preference models)"
    )

    full_battery = 8
    snapshots = len(shadow_df.groupby(["run_id", "player_id", "round"], observed=True))
    print(
        f"\n  Shadow offers:      {len(shadow_df)} answered, from {snapshots} "
        f"Merchant-probe snapshots ({100 * len(shadow_df) / (snapshots * full_battery):.1f}% "
        f"of the nominal {snapshots * full_battery})"
    )
    feasible = (
        shadow_df.groupby(["direction", "price"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    for row in feasible.itertuples():
        print(
            f"    {DIRECTION_LABEL[row.direction]:<12s} for {row.price:.0f} "
            f"unit(s) of another good: {row.n}"
        )
    note(
        "Each Merchant probe presents up to eight standardized offers. An "
        "offer the Merchant could not complete with its current inventory is "
        "not shown and, as preregistered, is excluded rather than imputed. "
        "Offers to surrender B only exist when the Merchant holds B, and "
        "holdings are themselves affected by the bulletin, so the surrender "
        "sample is conditioned on a post-treatment state.",
        indent=4,
    )


def print_manipulation_check(runs_df: pd.DataFrame) -> None:
    section("2", "MANIPULATION CHECK: BULLETIN DELIVERY")
    note(
        "Read from raw_prompts.jsonl, prompt by prompt, because the YAML "
        "snapshot in each run directory does not record the setting that "
        "was loaded. Treated runs must show their bulletin in every prompt "
        "of rounds 1-10; no run may show a bulletin at round 0, in the "
        "washout, or one belonging to another condition. Runs that fail are "
        "excluded (section 1), so this table should read 100% / 0 / 0 for "
        "the bulletins and 0% / 0 / 0 for Control."
    )
    print(
        f"\n  {'condition':<10s} {'runs':>5s}   {'rounds 1-10: assigned':>22s}"
        f"   {'round 0: any':>14s}   {'washout: any':>14s}"
    )
    print(f"  {'':<10s} {'':>5s}   {'bulletin shown':>22s}   {'bulletin':>14s}   {'bulletin':>14s}")
    for condition in CONDITIONS:
        block = runs_df[runs_df["condition"] == condition]
        round0 = (
            f"{int(block['round0_bulletin_prompts'].sum())}/"
            f"{int(block['round0_prompts'].sum())}"
        )
        washout = (
            f"{int(block['washout_bulletin_prompts'].sum())}/"
            f"{int(block['washout_prompts'].sum())}"
        )
        print(
            f"  {condition:<10s} {len(block):>5d}   "
            f"{block['exposure_share'].fillna(0).mean():>21.1%}   "
            f"{round0:>14s}   {washout:>14s}"
        )
    note(
        "The last two columns count prompts. With round 0 clean, the round-0 "
        "contrasts in section 9 are a balance check and the round 0 -> 10 "
        "difference-in-differences measures the change the bulletin caused.",
    )

    subsection("Real trades accepted per run (mean)")
    trades = runs_df.groupby(["condition", "model"], observed=True)[
        "trades_accepted"
    ].mean()
    print(f"  {'condition':<10s}" + "".join(f"{m:>10s}" for m in MODELS))
    for condition in CONDITIONS:
        print(
            f"  {condition:<10s}"
            + "".join(
                f"{trades.get((condition, model_name), np.nan):>10.2f}"
                for model_name in MODELS
            )
        )


def _round_label(round_index: int) -> str:
    if round_index == 0:
        return "0 (baseline)"
    if round_index > TREATMENT_END:
        return f"{round_index} (washout)"
    return str(round_index)


def _ci_cell(values: Any, scale: float = 1.0, digits: int = 2, signed: bool = False) -> str:
    mean, lower, upper = mean_ci(values)
    if not np.isfinite(mean):
        return "n/a"
    sign = "+" if signed else ""
    return (
        f"{mean * scale:{sign}.{digits}f} "
        f"[{lower * scale:{sign}.{digits}f}, {upper * scale:{sign}.{digits}f}]"
    )


def print_descriptives(
    shadow_df: pd.DataFrame, probes_df: pd.DataFrame, policy_df: pd.DataFrame
) -> None:
    section("3", "DESCRIPTIVE MEANS BY CONDITION AND ROUND")
    note(
        "Means over run-level means with 95% intervals across runs, pooled "
        "over the three models. Merchants only."
    )

    def per_run(frame: pd.DataFrame, condition: str, round_index: int, column: str) -> pd.Series:
        block = frame[(frame["condition"] == condition) & (frame["round"] == round_index)]
        return block.groupby("run_id", observed=True)[column].mean()

    subsection("Shadow offers: share accepted (%)")
    note(
        "Acquire = an offer that gives the Merchant one unit of B for one or "
        "two units of another good; surrender = the reverse. The B-protection "
        "index is acquire minus surrender acceptance, on a -1 to +1 scale."
    )
    acquire = shadow_df[shadow_df["direction"] == "acquire"]
    surrender = shadow_df[shadow_df["direction"] == "surrender"]
    print(
        f"\n  {'condition':<10s} {'round':<14s} {'acquire B':>20s} "
        f"{'surrender B':>20s} {'B-protection':>22s}"
    )
    for condition in CONDITIONS:
        for round_index in PROBE_ROUNDS:
            print(
                f"  {condition:<10s} {_round_label(round_index):<14s} "
                f"{_ci_cell(per_run(acquire, condition, round_index, 'accept'), 100, 1):>20s} "
                f"{_ci_cell(per_run(surrender, condition, round_index, 'accept'), 100, 1):>20s} "
                f"{_ci_cell(per_run(policy_df, condition, round_index, 'Policy_B'), 1, 3, True):>22s}"
            )
        print()

    merchants = probes_df[probes_df["role"] == "Merchant"]
    subsection("Preference probes")
    note(
        "Pb_general and Pb_inventory are rating differences (B minus the "
        "mean of A and C) on a -9 to +9 scale; 0 means B is rated like the "
        "others. bundle_B is units of B in an ideal bundle of six; 2 is an "
        "even split."
    )
    print(
        f"\n  {'condition':<10s} {'round':<14s} {'Pb_general':>22s} "
        f"{'bundle_B':>22s} {'Pb_inventory':>22s}"
    )
    for condition in CONDITIONS:
        for round_index in PROBE_ROUNDS:
            print(
                f"  {condition:<10s} {_round_label(round_index):<14s} "
                f"{_ci_cell(per_run(merchants, condition, round_index, 'Pb_general'), 1, 3, True):>22s} "
                f"{_ci_cell(per_run(merchants, condition, round_index, 'bundle_B'), 1, 3):>22s} "
                f"{_ci_cell(per_run(merchants, condition, round_index, 'Pb_inventory'), 1, 3, True):>22s}"
            )
        print()

    subsection("How much the stated measures vary at all (Merchants, all rounds)")
    for outcome in ("Pb_general", "bundle_B", "Pb_inventory"):
        values = merchants[outcome].dropna()
        counts = values.value_counts().sort_index()
        if len(counts) <= 6:
            spread = ", ".join(
                f"{value:g}: {count}" for value, count in counts.items()
            )
        else:
            spread = f"{len(counts)} distinct values from {counts.index.min():g} to {counts.index.max():g}"
        print(f"  {outcome:<13s} SD {values.std():.4f}   {spread}")
    note(
        "A measure that barely varies leaves some contrasts below with "
        "nothing to test; those rows say so instead of showing a p-value."
    )


def print_models_section(results: dict[str, Any]) -> None:
    section("4", "MODELS FITTED")
    note(
        "Model estimates are reported as contrasts in sections 5-9; this "
        "section records what was fitted and anything that went wrong."
    )
    acceptance_result = results["acceptance_result"]
    acceptance_data = results["acceptance_data"]
    dropped = results["dropped"]

    subsection("Operational policy: binomial GLM on shadow-offer acceptance")
    note(
        "Condition x Round x Direction as cell means, plus offer "
        "attractiveness, the counter good, the latent-utility change and "
        "model family; covariance clustered on run. (Prereg2 asks for a "
        "binomial GLMM with Model in the interaction; see departures 1-2 in "
        "the script docstring.)"
    )
    print(
        f"\n  N = {int(acceptance_result.nobs)} offers, "
        f"{acceptance_data['run_id'].nunique()} run clusters, "
        f"{len(acceptance_result.params)} fixed parameters; reference cell "
        f"{results['reference']}"
    )
    covariates = {
        "log_ratio": "offer attractiveness, log(units received / surrendered)",
        "dlogU": "latent log-utility change from accepting",
    }
    for name, blurb in covariates.items():
        if name not in acceptance_result.params.index:
            continue
        spread = float(acceptance_data[name].std())
        print(
            f"  {name:<10s} b = {acceptance_result.params[name]:+.3f} "
            f"(p = {acceptance_result.pvalues[name]:.3g}); "
            f"{acceptance_result.params[name] * spread:+.2f} log-odds per SD "
            f"({spread:.3f}) -- {blurb}"
        )
    if not np.isfinite(np.asarray(acceptance_result.cov_params())).all():
        print(
            "  WARNING: the cluster-robust covariance is not finite; the "
            "logistic contrasts cannot be trusted"
        )

    print("\n  Design cells with no variation in acceptance:")
    if not len(dropped):
        print("    none: every cell contains both accepts and rejects")
    for row in dropped.itertuples():
        print(
            f"    {str(row.condition):<9s} {str(row.round_f):<4s} "
            f"{DIRECTION_LABEL[str(row.direction)]:<12s} "
            f"{int(row.sum)} accepts of {int(row.count)} offers"
        )
    if len(dropped):
        note(
            "These cells are perfectly separated: their log odds are "
            "unbounded, so they are dropped from the logistic fit. A contrast "
            "that needs one is marked 'not estimable' and its run-level "
            "probability-scale version is the one to read.",
            indent=4,
        )

    subsection("Stated preference and valuation")
    print(
        f"  Pb_general    linear mixed model, run and Merchant-within-run "
        f"intercepts; N = {results['general_n']}"
    )
    if results["bundle_constant"]:
        print("  bundle_B      not fitted: every Merchant answer is identical")
    else:
        print(
            f"  bundle_B      binomial GLM on B of six (run-clustered) and a "
            f"linear mixed model on units for the TOST; N = {results['bundle_n']}"
        )
    print(
        f"  Pb_inventory  linear mixed model, no holdings term (Prereg2); "
        f"N = {results['state_n']}"
    )
    print(
        f"  reservation   linear mixed model on the surrender ladder; "
        f"N = {results['reservation_n']} ladders, "
        f"{results['ladder_violations']} non-monotone (accept 1 unit, refuse 2)"
    )
    note(
        "Each preference model is saturated in Condition x Round x Model with "
        "B display position as a covariate; pooled contrasts are the average "
        "of the three model-specific contrasts."
    )

    subsection("Fitting messages")
    messages = [line for line in results["fit_log"].splitlines() if line.strip()]
    if messages:
        for line in messages:
            print(f"  {line.strip()}")
    else:
        print("  none: every model converged with its preregistered random effects")


def h3_verdicts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """H3 as Prereg2 states it, read off the H1 and H2 tests.

    A treatment supports H3 when at least one of its round-10 policy contrasts
    against Control is significant after Holm in the predicted direction, and
    every general-preference contrast against Control (round 10 and round
    0 -> 10, for both the general-importance index and the ideal bundle) is
    inside the +/-0.5 equivalence bound.
    """
    by_label = {row["label"]: row for row in rows}
    verdicts = []
    for treatment in TREATMENTS:
        policy = []
        for direction, predicted in (("Acquire B", 1.0), ("Surrender B", -1.0)):
            row = by_label.get(f"{direction} @ r{PRIMARY_ROUND}: {treatment} - Control")
            significant = bool(
                row
                and np.isfinite(row.get("p_holm", np.nan))
                and row["p_holm"] < ALPHA
                and np.sign(row["estimate"]) == predicted
            )
            policy.append((direction, row, significant))
        preference = []
        for outcome in ("Pb_general", "bundle_B (units)", "bundle_B"):
            for kind in (f"@ r{PRIMARY_ROUND}", f"DiD r0->r{PRIMARY_ROUND}"):
                row = by_label.get(f"{outcome} {kind}: {treatment} - Control")
                if row is not None and "equivalent" in row:
                    preference.append((f"{outcome} {kind}", row, bool(row["equivalent"])))
        verdicts.append(
            {
                "treatment": treatment,
                "policy": policy,
                "preference": preference,
                "supported": any(flag for _, _, flag in policy)
                and bool(preference)
                and all(flag for _, _, flag in preference),
            }
        )
    return verdicts


def print_h3(rows: Sequence[dict[str, Any]], dissociation: pd.DataFrame) -> None:
    section("7", "H3: POLICY MOVES, STATED PREFERENCE DOES NOT")
    note(
        "Prereg2 judges H3 from the H1 and H2 tests, with no model of its "
        "own: a treatment supports it when at least one round-10 policy "
        "contrast is significant after Holm in the predicted direction and "
        "every general-preference contrast is equivalent within +/-0.5."
    )
    for verdict in h3_verdicts(rows):
        print(
            f"\n  {verdict['treatment']} - Control: "
            + ("SUPPORTED" if verdict["supported"] else "NOT SUPPORTED")
        )
        for direction, row, flag in verdict["policy"]:
            if row is None or not np.isfinite(row.get("se", np.nan)):
                print(f"    policy  {direction:<30s} not estimable")
                continue
            print(
                f"    policy  {direction:<30s} "
                f"{'OR ' + format(row.get('odds_ratio', np.nan), '.3g'):<10s}"
                f"p_holm {fmt_p(row.get('p_holm')):<9s}"
                + ("effect in the predicted direction" if flag else "no qualifying effect")
            )
        for name, row, flag in verdict["preference"]:
            print(
                f"    stated  {name:<30s} {row['estimate']:<+10.3f}"
                f"{'':<16s}"
                + ("equivalent within +/-0.5" if flag else "NOT shown equivalent")
            )

    subsection("Descriptive: every measure on a common scale")
    note(
        "To compare the size of the policy and preference effects, each "
        "treatment - Control level contrast (run-level means) is divided by "
        "the width of its measuring instrument: 1.0 for P(accept), 2.0 for "
        "the B-protection index, 18 for a rating difference, 6 for the "
        "bundle. The state-dependent row is raw and moves with holdings; the "
        "holdings-matched version is in section 6. Not a test; shown for "
        "scale."
    )
    for treatment in TREATMENTS:
        for round_index in (0, PRIMARY_ROUND):
            print(f"\n  {treatment} - Control at round {round_index}")
            block = dissociation[
                (dissociation["treatment"] == treatment)
                & (dissociation["round"] == round_index)
            ]
            for row in block.itertuples():
                if not np.isfinite(row.estimate):
                    print(f"    {row.outcome:<36s} not estimable")
                    continue
                line = (
                    f"    {row.outcome:<36s} {row.estimate:+8.4f} "
                    f"[{row.ci_low:+8.4f}, {row.ci_high:+8.4f}]  "
                    f"= {100 * row.share:+6.2f}% of range"
                )
                line += (
                    f"  (Welch p {fmt_p(row.p)})" if np.isfinite(row.p)
                    else "  (no variance)"
                )
                print(line)


def constant_outcome(frame: pd.DataFrame, outcome: str) -> bool:
    values = frame[outcome].dropna().to_numpy(dtype=float)
    return len(values) > 0 and float(np.ptp(values)) == 0.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def compute(base_dir: Path, fit_glmm: bool) -> dict[str, Any]:
    """Load the runs and fit every model; returns everything the report needs.

    Messages printed while fitting (a singular variance component, a
    fallback fit) are captured and reported in the models section instead of
    landing wherever they happen to fall in the output.
    """
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        data = load_dataset(base_dir)
        probes_df, shadow_df = data["probes"], data["shadow"]
        policy_df = policy_index_frame(shadow_df)
        reservation_df = reservation_frame(shadow_df)
        merchants = probes_df[probes_df["role"] == "Merchant"]
        bundle_constant = constant_outcome(merchants, "bundle_B")

        print_progress("fitting the acceptance model")
        acceptance_result, acceptance_data, dropped, reference = fit_acceptance_model(
            shadow_df
        )
        policy_rows = policy_contrasts(acceptance_result, dropped, reference)
        print_progress("run-level policy contrasts")
        risk_rows = policy_risk_differences(shadow_df)
        baseline_rows = off_plan_round_rows(
            shadow_df, probes_df, policy_df, 0,
            "round 0 balance check (not preregistered)",
        )
        rebound_rows = off_plan_round_rows(
            shadow_df, probes_df, policy_df, 12, "round 12 (not preregistered)"
        )

        print_progress("fitting the preference models")
        general_result, general_data = fit_preference_lmm(probes_df, "Pb_general")
        general_rows = preference_contrasts(
            general_result, "Pb_general", "general preference", EQUIV_BOUND_GENERAL
        )
        general_robust_rows = preference_robustness_rows(
            probes_df, "Pb_general", "Pb_general"
        )

        bundle_n = 0
        if bundle_constant:
            note_text = (
                f"every Merchant answer is B = {BUNDLE_TOTAL // len(GOODS)} of "
                f"{BUNDLE_TOTAL}; zero variance in "
                f"{len(merchants['bundle_B'].dropna())} observations"
            )
            bundle_logit_rows: list[dict[str, Any]] = []
            bundle_unit_rows = degenerate_contrast_rows(
                "bundle_B", note_text, EQUIV_BOUND_BUNDLE
            )
            for row in bundle_unit_rows:
                if row["family"] == "general preference":
                    row["family"] = "bundle equivalence (TOST)"
        else:
            bundle_result, bundle_data = fit_bundle_binomial(probes_df)
            bundle_n = len(bundle_data)
            bundle_logit_rows = preference_contrasts(
                bundle_result, "bundle_B (logit)", "general preference", None
            )
            bundle_lmm_result, _ = fit_preference_lmm(probes_df, "bundle_B")
            bundle_unit_rows = preference_contrasts(
                bundle_lmm_result,
                "bundle_B (units)",
                "general preference",
                EQUIV_BOUND_BUNDLE,
            )
            for row in bundle_unit_rows:
                row["family"] = "bundle equivalence (TOST)"

        state_result, state_data = fit_state_dependent_lmm(probes_df)
        matched_rows = matched_valuation_rows(probes_df)
        state_rows = preference_contrasts(
            state_result, "Pb_inventory", "general preference", None
        )

        reservation_result, reservation_data = fit_reservation_lmm(reservation_df)
        # H1.3 was dropped in Prereg2: reported, but outside every Holm family.
        reservation_rows = preference_contrasts(
            reservation_result, "reservation", RESERVATION_FAMILY, None
        )
        for row in reservation_rows:
            row["family"] = RESERVATION_FAMILY

        all_rows = (
            policy_rows
            + risk_rows
            + baseline_rows
            + rebound_rows
            + general_rows
            + general_robust_rows
            + bundle_logit_rows
            + bundle_unit_rows
            + state_rows
        )
        holm(all_rows)
        all_rows = all_rows + reservation_rows + matched_rows

        print_progress("common-scale and per-model contrasts")
        dissociation = dissociation_table(shadow_df, probes_df, policy_df)
        per_model_rows = per_model_policy_contrasts(shadow_df)

        glmm_summary = None
        if fit_glmm:
            print_progress("fitting the Bayesian GLMM")
            glmm_summary = str(fit_acceptance_glmm(acceptance_data).summary())

    return {
        "data": data,
        "policy_df": policy_df,
        "acceptance_result": acceptance_result,
        "acceptance_data": acceptance_data,
        "dropped": dropped,
        "reference": reference,
        "bundle_constant": bundle_constant,
        "general_n": len(general_data),
        "bundle_n": bundle_n,
        "state_n": len(state_data),
        "reservation_n": len(reservation_data),
        "ladder_violations": (
            int(reservation_df["ladder_violations"].sum()) if len(reservation_df) else 0
        ),
        "policy_rows": policy_rows,
        "risk_rows": risk_rows,
        "baseline_rows": baseline_rows,
        "rebound_rows": rebound_rows,
        "general_rows": general_rows,
        "general_robust_rows": general_robust_rows,
        "bundle_logit_rows": bundle_logit_rows,
        "bundle_unit_rows": bundle_unit_rows,
        "state_rows": state_rows,
        "matched_rows": matched_rows,
        "reservation_rows": reservation_rows,
        "all_rows": all_rows,
        "dissociation": dissociation,
        "per_model_rows": per_model_rows,
        "glmm_summary": glmm_summary,
        "fit_log": log.getvalue(),
    }


def print_progress(message: str) -> None:
    """Progress for the terminal only; it never reaches report.txt."""
    print(f"[final_experiment_analysis] {message} ...", file=sys.stderr, flush=True)


def write_report(results: dict[str, Any], base_dir: Path) -> None:
    data = results["data"]
    probes_df, shadow_df, runs_df = data["probes"], data["shadow"], data["runs"]
    all_rows = results["all_rows"]
    in_family = lambda rows, family: [row for row in rows if row["family"] == family]

    print("=" * REPORT_WIDTH)
    print("FINAL EXPERIMENT: CONFIRMATORY ANALYSIS REPORT")
    print("=" * REPORT_WIDTH)
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Data:      {base_dir}")
    print(
        f"  Design:    3 bulletins (Control, Scarcity, Expert) x 3 models "
        f"({', '.join(MODELS)}); {len(runs_df)} runs analysed"
    )
    note(
        "This report is the primary record of the analysis. Every number in "
        "it is in the CSV files written alongside it (listed at the end), and "
        "the figures drawn by final_experiment_figures.py are plain averages "
        "of the same data. Analyses not in Prereg2 are labelled 'not "
        "preregistered', 'exploratory' or 'robustness' wherever they appear."
    )

    print_summary(all_rows)
    print_legend()
    print_data_section(data, base_dir)
    print_manipulation_check(runs_df)
    print_descriptives(shadow_df, probes_df, results["policy_df"])
    print_models_section(results)

    section("5", "H1: OPERATIONAL POLICY")
    note(
        "Do Merchants accept offers that give them B more often (H1.1), and "
        "offers that take B away less often (H1.2), when a bulletin is up? "
        f"The confirmatory tests are the round-{PRIMARY_ROUND} contrasts and "
        "the round 0 -> 10 differences-in-differences from the logistic "
        "model, Holm-adjusted together as the 'primary policy' family."
    )
    print_contrasts(
        in_family(results["policy_rows"], "primary policy"),
        "Logistic model, odds ratios (Holm family: primary policy)",
        odds=True,
        notes=[
            "An odds ratio above 1 means more acceptance than the comparison "
            "condition. A row marked 'not estimable' needs a design cell with "
            "no variation in acceptance (listed in section 4); the risk "
            "differences below cover the same comparison.",
        ],
    )
    print_contrasts(
        results["risk_rows"],
        "The same contrasts as run-level risk differences (Holm family: "
        "primary policy (risk difference))",
        odds=False,
        notes=[
            "Difference in the share of offers accepted, per run, then "
            "averaged (0.087 = 8.7 percentage points). Secondary to the "
            "logistic contrasts: it does not adjust for the offer's terms or "
            "the utility change, which differ between conditions because "
            "holdings do. Where acceptance at round 0 is zero in both "
            "conditions, the DiD equals the round-10 contrast.",
        ],
    )
    print_contrasts(
        in_family(results["policy_rows"], "descriptive B-protection index"),
        "B-protection index: the acquire odds ratio divided by the surrender "
        "odds ratio (descriptive in Prereg2; Holm within its own family)",
        odds=True,
    )
    print_contrasts(
        results["reservation_rows"],
        "Exploratory: reservation value for giving up B (H1.3 in the original "
        "plan, dropped in Prereg2; unadjusted)",
        odds=False,
        notes=[
            "Units of another good needed before a Merchant gives up one B "
            "(1, 2, or 3 = refuses both). Positive = B is harder to buy away.",
        ],
    )

    section("6", "H2: STATED PREFERENCE, AND STATE-DEPENDENT VALUATION")
    note(
        "Does what Merchants say about B change? H2 predicts it does not, "
        "and is tested by equivalence: each contrast must lie within +/-0.5 "
        "(rating points or B units). The 'general preference' Holm family "
        "holds the Pb_general and Pb_inventory contrasts."
    )
    print_contrasts(
        in_family(results["general_rows"], "general preference"),
        "H2.2 general-importance B index, mixed model (Holm family: general "
        "preference)",
        odds=False,
    )
    print_contrasts(
        results["general_robust_rows"],
        "H2.2 the same comparison at run level (robustness; Holm family: "
        "stated-preference robustness)",
        odds=False,
        notes=[
            "When one condition's answers barely vary, the mixed model's "
            "shared residual variance describes neither condition; Welch and "
            "permutation tests on run means do not need that assumption.",
        ],
    )
    print_contrasts(
        results["bundle_unit_rows"],
        "H2.1 ideal bundle, B units, linear mixed model for the equivalence "
        "test (Holm family: bundle equivalence)",
        odds=False,
    )
    print_contrasts(
        in_family(results["bundle_logit_rows"], "general preference"),
        "H2.1 ideal bundle, B of six, binomial GLM (log odds; Holm family: "
        "general preference)",
        odds=False,
        notes=[
            "When nearly every answer is the same, these log-odds differences "
            "sit at zero and some have no usable standard error; the units "
            "model above carries the equivalence test.",
        ],
    )
    print_contrasts(
        in_family(results["state_rows"], "general preference"),
        "State-dependent B index, Prereg2 model without a holdings term "
        "(Holm family: general preference)",
        odds=False,
        notes=[
            "This index is a rating given the Merchant's current inventory, "
            "and the bulletins change inventories, so a gap here can come "
            "from holdings rather than from the bulletin. The matched "
            "comparison below removes that.",
        ],
    )
    matched = results["matched_rows"]
    print_contrasts(
        [row for row in matched if row["label"].startswith("state-dependent")],
        "Robustness: state-dependent B index at identical inventories "
        "(unadjusted)",
        odds=False,
        notes=[
            "Each treated Merchant is compared with Control Merchants holding "
            "exactly the same A, B and C at the same round; differences are "
            "averaged within run first.",
        ],
    )
    print_contrasts(
        [row for row in matched if not row["label"].startswith("state-dependent")],
        "Post hoc: the three inventory-conditioned ratings at identical "
        "inventories (unadjusted)",
        odds=False,
        notes=[
            "Splits the index above into its parts (1-10 rating scale), to "
            "show whether a gap comes from the rating of B or of A and C.",
        ],
    )

    print_h3(all_rows, results["dissociation"])

    section("8", "H4: AFTER THE BULLETIN IS WITHDRAWN")
    note(
        f"The bulletin is removed after round {TREATMENT_END}. H4 predicts "
        "the effects weaken, judged by the round-14 contrasts and by the "
        "round 10 -> 14 change in each treatment - Control gap ('decay'), "
        "all Holm-adjusted together as the 'washout' family. A decay of the "
        "opposite sign to the round-10 effect means the effect shrank."
    )
    washout = in_family(all_rows, "washout")
    for prefix, title, odds in (
        ("Acquire B", "Accepting offers that give B (odds ratios)", True),
        ("Surrender B", "Accepting offers that take B (odds ratios)", True),
        ("Pb_general", "General-importance B index", False),
        ("bundle_B", "Ideal bundle, log odds", False),
        ("Pb_inventory", "State-dependent B index", False),
    ):
        print_contrasts(
            [row for row in washout if row["label"].startswith(prefix)],
            f"{title} (Holm family: washout)",
            odds=odds,
        )
    print_contrasts(
        results["rebound_rows"],
        "Round 12, two rounds after withdrawal (not preregistered; run level; "
        "Holm within its own family)",
        odds=False,
        notes=[
            "The first probe without the bulletin, between the round-10 and "
            "round-14 tests above.",
        ],
    )

    section("9", "BALANCE CHECK, MODEL HETEROGENEITY AND ROBUSTNESS")
    print_contrasts(
        results["baseline_rows"],
        "Round 0 balance check, before any bulletin or trade (not "
        "preregistered; run level)",
        odds=False,
        notes=[
            "No bulletin has been shown and every Merchant holds one unit of "
            "each good, so any difference here is chance. These should all "
            "be null.",
        ],
    )
    per_model = results["per_model_rows"]
    for round_index, purpose in (
        (0, "baseline balance within each model"),
        (PRIMARY_ROUND, "the round-10 policy effect within each model"),
    ):
        print_contrasts(
            [row for row in per_model if row["round"] == round_index],
            f"Exploratory: {purpose}, risk differences (unadjusted)",
            odds=False,
            notes=(
                [
                    "Prereg2 declares no confirmatory model x condition test. "
                    "These are run-level differences in P(accept) within one "
                    "model's 40 runs per condition; the logistic model cannot "
                    "be fitted per model (its utility covariate separates "
                    "acceptance within a single family).",
                ]
                if round_index == 0
                else []
            ),
        )
    if results["glmm_summary"] is not None:
        subsection("Robustness: Bayesian binomial mixed model (run + Merchant intercepts)")
        print(results["glmm_summary"])


def run_analysis(base_dir: Path, out_dir: Path, fit_glmm: bool) -> None:
    results = compute(base_dir, fit_glmm)
    write_report(results, base_dir)

    data = results["data"]
    files = {
        "report.txt": "this report",
        "contrasts.csv": "every contrast above, one row each, keyed by 'label' "
        "and 'family'",
        "dissociation.csv": "the common-scale table of section 7",
        "runs.csv": "one row per analysed run, with the bulletin audit",
        "probes.csv": "one row per agent per preference probe (all roles)",
        "shadow_offers.csv": "one row per answered standardized offer",
    }
    pd.DataFrame(results["all_rows"] + results["per_model_rows"]).to_csv(
        out_dir / "contrasts.csv", index=False
    )
    results["dissociation"].to_csv(out_dir / "dissociation.csv", index=False)
    data["runs"].to_csv(out_dir / "runs.csv", index=False)
    data["probes"].to_csv(out_dir / "probes.csv", index=False)
    data["shadow"].to_csv(out_dir / "shadow_offers.csv", index=False)

    section("", "FILES")
    print(f"  Written to {out_dir}:")
    for name, description in files.items():
        print(f"    {name:<18s} {description}")
    note(
        "Figures (h1_*.png to h4_*.png) are drawn into the same folder by "
        "final_experiment_figures.py.",
        indent=4,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory holding <Model>/<Condition>/<run> (default: data/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: analysis/analysis_outputs/final_experiment)",
    )
    parser.add_argument(
        "--glmm",
        action="store_true",
        help="Also fit the Bayesian binomial mixed model robustness check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(out_dir / "report.txt")
    sys.stdout = tee
    try:
        run_analysis(args.base_dir.resolve(), out_dir, args.glmm)
    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    main()
