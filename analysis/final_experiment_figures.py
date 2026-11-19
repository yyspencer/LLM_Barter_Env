#!/usr/bin/env python3
"""Six figures for the final experiment, one per hypothesis.

``final_experiment_analysis.py`` carries the registered statistical models and
writes the report and tables; it doesn't draw any figures, because these figures are specific to this particular data (hand-written descriptions of the results).

    h1_1_buying_b.png        H1.1  do Merchants say yes more often to buying B?
    h1_2_selling_b.png       H1.2  do they say yes less often to selling B?
    h1_3_price_of_b.png      H1.3  does it take more to buy B away from them?
    h2_stated_preference.png H2    does what they *say* they want change?
    h3_do_versus_say.png     H3    do actions move more than words?
    h4_after_removal.png     H4    does the effect outlast the bulletin?
                                  (buying, selling, and state-dependent valuation)

Everything plotted is a plain average over independent runs: the percentage of
standardized offers a Merchant accepted, the rating it gave, the bundle it
asked for. A band or whisker is the 95% interval of that average across runs.
The same exclusions as the confirmatory script apply, because the data is
loaded through it.

Outputs are ``analysis/analysis_outputs/final_experiment``, next to the
report and tables the analysis script writes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import final_experiment_analysis as fea

# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
PHASE_FILL = "#f0efec"

CONDITIONS = fea.CONDITIONS
COLOR = {"Control": "#6b6a66", "Scarcity": "#eb6834", "Expert": "#2a78d6"}
MARKER = {"Control": "o", "Scarcity": "s", "Expert": "^"}
CONDITION_NAME = {
    "Control": "No bulletin",
    "Scarcity": "Scarcity bulletin",
    "Expert": "Expert bulletin",
}
MODEL_NAME = {
    "Claude": "Claude Sonnet 5",
    "OpenAI": "GPT-5.6 Terra",
    "Gemini": "Gemini 3.1 Pro",
}
ROUNDS = fea.PROBE_ROUNDS
BULLETIN_END = fea.TREATMENT_END
SHOWN_ROUND = fea.PRIMARY_ROUND

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": "#b9b8b2",
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def run_average(
    frame: pd.DataFrame, value: str, condition: str, round_index: int
) -> tuple[float, float, float]:
    """Average of run-level means with its 95% interval across runs."""
    block = frame[(frame["condition"] == condition) & (frame["round"] == round_index)]
    return fea.mean_ci(block.groupby("run_id", observed=True)[value].mean())


def gap_to_control(
    frame: pd.DataFrame, value: str, condition: str, round_index: int
) -> tuple[float, float, float]:
    """Treated minus Control average at one round, with a 95% interval."""

    def per_run(name: str) -> np.ndarray:
        block = frame[(frame["condition"] == name) & (frame["round"] == round_index)]
        return block.groupby("run_id", observed=True)[value].mean().to_numpy()

    treated, control = per_run(condition), per_run("Control")
    if len(treated) < 2 or len(control) < 2:
        return np.nan, np.nan, np.nan
    gap = float(treated.mean() - control.mean())
    half = 1.96 * math.sqrt(
        treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)
    )
    return gap, gap - half, gap + half


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def mark_phases(ax: plt.Axes, label: bool = True) -> None:
    """Shade the rounds after the bulletin has been taken down."""
    ax.axvspan(BULLETIN_END, ROUNDS[-1] + 0.6, color=PHASE_FILL, lw=0, zorder=0)
    ax.axvline(BULLETIN_END, color="#9c9b95", lw=1, ls=(0, (4, 3)), zorder=1)
    if label:
        top = ax.get_ylim()[1]
        ax.text(
            BULLETIN_END / 2, top, "bulletin on the board",
            ha="center", va="top", fontsize=8, color=INK_SECONDARY,
        )
        ax.text(
            (BULLETIN_END + ROUNDS[-1] + 0.6) / 2, top, "bulletin removed",
            ha="center", va="top", fontsize=8, color=INK_SECONDARY,
        )


def style_time_axis(ax: plt.Axes, xlabel: bool = True) -> None:
    ax.set_xticks(ROUNDS)
    ax.set_xlim(ROUNDS[0] - 0.6, ROUNDS[-1] + 0.6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if xlabel:
        ax.set_xlabel("Trading round at which Merchants were asked")


def condition_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    scale: float = 100.0,
    end_labels: bool = False,
    small: bool = False,
    label_only: Sequence[str] | None = None,
) -> float:
    """One line per condition; returns the highest point drawn."""
    highest = 0.0
    ends: list[tuple[float, str]] = []
    for condition in CONDITIONS:
        stats = [run_average(frame, value, condition, r) for r in ROUNDS]
        means = [scale * s[0] for s in stats]
        lows = [scale * s[1] for s in stats]
        highs = [scale * s[2] for s in stats]
        ax.fill_between(ROUNDS, lows, highs, color=COLOR[condition], alpha=0.14, lw=0)
        ax.plot(
            ROUNDS, means,
            color=COLOR[condition], lw=1.6 if small else 2.2,
            marker=MARKER[condition], ms=4.5 if small else 7.5,
            mec=SURFACE, mew=1.0 if small else 1.6,
            label=CONDITION_NAME[condition], zorder=3,
        )
        highest = max(highest, np.nanmax(highs))
        ends.append((means[ROUNDS.index(SHOWN_ROUND)], condition))
    if end_labels:
        # Label the last round with the bulletin still up: it is the round the
        # hypothesis is about.  Labels are pushed apart when two lines meet, so
        # a number is never printed on top of another.
        chosen = sorted(
            entry for entry in ends if label_only is None or entry[1] in label_only
        )
        minimum_gap = 0.055 * highest
        previous = -np.inf
        for value_at, condition in chosen:
            height = max(value_at + 0.02 * highest, previous + minimum_gap)
            previous = height
            ax.text(
                SHOWN_ROUND - 0.3, height,
                f"{value_at:.0f}%" if value_at >= 1 else f"{value_at:.1f}%",
                ha="right", va="bottom", fontsize=9, color=INK, fontweight="bold",
            )
    return highest


def headline(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.965, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.905, subtitle, fontsize=10.5, color=INK_SECONDARY, va="top")


def footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.035, 0.018, text, fontsize=8, color=INK_SECONDARY, va="bottom")


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def runs_note(runs_df: pd.DataFrame) -> str:
    counts = runs_df.groupby("condition", observed=True).size()
    span = (
        f"{counts.min()}" if counts.min() == counts.max()
        else f"{counts.min()}-{counts.max()}"
    )
    return (
        f"Each point averages {span} independent market "
        "runs (two Merchants per run, three AI models pooled). Shaded band: 95% "
        "interval of that average."
    )


# ---------------------------------------------------------------------------
# H1.1 and H1.2: one template, two directions
# ---------------------------------------------------------------------------

def fig_offers(
    shadow_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    direction: str,
    title: str,
    subtitle: str,
    ylabel: str,
    path: Path,
) -> None:
    subset = shadow_df[shadow_df["direction"] == direction]
    fig = plt.figure(figsize=(13, 6.4))
    grid = fig.add_gridspec(
        3, 2, width_ratios=[2.15, 1], left=0.065, right=0.975,
        top=0.80, bottom=0.14, wspace=0.16, hspace=0.42,
    )
    main = fig.add_subplot(grid[:, 0])
    top = condition_lines(main, subset, "accept", end_labels=True)
    main.set_ylim(0, top * 1.22)
    mark_phases(main)
    style_time_axis(main)
    main.set_ylabel(ylabel)
    main.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    main.yaxis.set_major_locator(mticker.MaxNLocator(7, steps=[1, 2, 5, 10]))
    main.set_title("All three AI models together", loc="left", color=INK)
    main.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False)

    ceiling = 0.0
    panels = []
    for row, model in enumerate(fea.MODELS):
        ax = fig.add_subplot(grid[row, 1])
        ceiling = max(
            ceiling,
            condition_lines(ax, subset[subset["model"] == model], "accept", small=True),
        )
        ax.set_title(MODEL_NAME[model], loc="left", color=INK, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        style_time_axis(ax, xlabel=(row == len(fea.MODELS) - 1))
        if row == len(fea.MODELS) - 1:
            ax.set_xlabel("Trading round")
        panels.append(ax)
    for ax in panels:
        # One scale for the three models, so their sizes can be compared.
        ax.set_ylim(0, ceiling * 1.1)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3, steps=[1, 2, 5, 10]))
        mark_phases(ax, label=False)

    headline(fig, title, subtitle)
    footnote(fig, runs_note(runs_df))
    save(fig, path)


# ---------------------------------------------------------------------------
# H1.3: the price of B
# ---------------------------------------------------------------------------

def fig_price(
    shadow_df: pd.DataFrame, reservation_df: pd.DataFrame, runs_df: pd.DataFrame,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 6.2))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.5, 1], left=0.065, right=0.975,
        top=0.78, bottom=0.15, wspace=0.2,
    )
    frame = reservation_df.copy()
    frame["refuses_all"] = (frame["reservation"] >= 3.0).astype(float)

    ax = fig.add_subplot(grid[0, 0])
    condition_lines(ax, frame, "refuses_all", end_labels=True, label_only=("Control",))
    # The two bulletin lines finish within two points of each other, so they
    # share one label instead of two that would have to be pushed apart.
    treated = [
        100 * run_average(frame, "refuses_all", name, SHOWN_ROUND)[0]
        for name in fea.TREATMENTS
    ]
    ax.text(
        SHOWN_ROUND - 0.3, 100.4,
        f"{min(treated):.0f}-{max(treated):.0f}%",
        ha="right", va="bottom", fontsize=9, color=INK, fontweight="bold",
    )
    ax.set_ylim(60, 106)
    mark_phases(ax)
    style_time_axis(ax)
    ax.set_ylabel("Merchants who turn down every offer for their B")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_title(
        "Share of Merchants for whom no price on offer was enough",
        loc="left", color=INK,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.04), frameon=False)
    ax.text(
        ROUNDS[-1] + 0.5, 60.6, "axis starts at 60%", fontsize=8,
        color=INK_SECONDARY, ha="right",
    )

    bars = fig.add_subplot(grid[0, 1])
    surrender = shadow_df[
        (shadow_df["direction"] == "surrender") & (shadow_df["round"] == SHOWN_ROUND)
    ]
    width = 0.25
    for offset, condition in zip((-width, 0.0, width), CONDITIONS):
        for position, price in enumerate((1.0, 2.0)):
            mean, low, high = run_average(
                surrender[surrender["price"] == price], "accept", condition,
                SHOWN_ROUND,
            )
            x = position + offset
            bars.bar(
                x, 100 * mean, width=width - 0.02, color=COLOR[condition],
                label=CONDITION_NAME[condition] if position == 0 else None,
            )
            bars.errorbar(
                x, 100 * mean,
                yerr=[[100 * (mean - max(low, 0))], [100 * (high - mean)]],
                fmt="none", ecolor=INK_SECONDARY, elinewidth=1, capsize=2.5,
            )
            bars.annotate(
                f"{100 * mean:.0f}%" if mean >= 0.0095 else f"{100 * mean:.1f}%",
                (x, 100 * high), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=9, color=INK,
            )
    bars.set_xticks([0, 1])
    bars.set_xticklabels(["1 unit of A or C\nfor 1 B", "2 units of A or C\nfor 1 B"])
    bars.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    bars.set_ylabel("Offers accepted")
    bars.grid(axis="y", color=GRID, lw=0.8)
    bars.set_axisbelow(True)
    bars.tick_params(length=0)
    bars.set_title(
        f"Does doubling the payment help? (round {SHOWN_ROUND})", loc="left", color=INK
    )
    bars.legend(loc="upper left", frameon=False)

    headline(
        fig,
        "Most Merchants won't sell B at any price offered, bulletin or not",
        "Merchants holding B were offered 1 unit, then 2 units, of another good "
        "for it. Without a bulletin about one in ten would sell\nat some "
        "price by round 10; with either bulletin, about one in forty. That gap "
        "is small and not statistically reliable.",
    )
    footnote(
        fig,
        runs_note(runs_df)
        + " Only Merchants who still hold a unit of B can be asked.",
    )
    save(fig, path)


# ---------------------------------------------------------------------------
# H2: stated preference
# ---------------------------------------------------------------------------

def fig_stated(probes_df: pd.DataFrame, runs_df: pd.DataFrame, path: Path) -> None:
    merchants = probes_df[probes_df["role"] == "Merchant"].copy()
    merchants["rates_b_highest"] = (merchants["Pb_general"] > 0).astype(float)
    merchants["others"] = (
        merchants["rating_general_A"] + merchants["rating_general_C"]
    ) / 2

    fig = plt.figure(figsize=(13, 6.2))
    grid = fig.add_gridspec(
        1, 3, left=0.06, right=0.975, top=0.76, bottom=0.15, wspace=0.26
    )

    # Counts quoted in the panel titles, read from the data so a new collection
    # cannot leave a stale claim on the figure.
    answers = merchants["bundle_B"].notna()
    odd_bundles = int(
        (
            (merchants["bundle_A"] != 2)
            | (merchants["bundle_B"] != 2)
            | (merchants["bundle_C"] != 2)
        )[answers].sum()
    )
    rating_means = [
        run_average(merchants, "rating_general_B", condition, r)[0]
        for condition in CONDITIONS
        for r in ROUNDS
    ]
    rating_spread = math.ceil(10 * (max(rating_means) - min(rating_means))) / 10
    expert_peak = max(
        100 * run_average(merchants, "rates_b_highest", "Expert", r)[0]
        for r in ROUNDS
    )

    # Panel 1: the ideal bundle, as the units the Merchants actually named.
    ax = fig.add_subplot(grid[0, 0])
    at_round = merchants[merchants["round"] == SHOWN_ROUND]
    goods_color = {"A": "#c9c8c2", "B": "#4a3aa7", "C": "#c9c8c2"}
    for position, condition in enumerate(CONDITIONS):
        block = at_round[at_round["condition"] == condition]
        bottom = 0.0
        for good in fea.GOODS:
            units = float(block[f"bundle_{good}"].mean())
            ax.bar(
                position, units, bottom=bottom, width=0.62,
                color=goods_color[good], edgecolor=SURFACE, linewidth=2,
            )
            ax.text(
                position, bottom + units / 2, f"{good}: {units:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if good == "B" else INK,
            )
            bottom += units
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([CONDITION_NAME[c].replace(" ", "\n", 1) for c in CONDITIONS])
    ax.set_ylim(0, 6.2)
    ax.set_ylabel("Units asked for, out of 6")
    ax.tick_params(length=0)
    ax.set_title(
        '"Pick your ideal 6 units"\n'
        + (
            f"every answer was 2-2-2 (round {SHOWN_ROUND} shown)"
            if odd_bundles == 0
            else f"{answers.sum() - odd_bundles:,} of {answers.sum():,} answers "
            f"were 2-2-2 (round {SHOWN_ROUND} shown)"
        ),
        loc="left", color=INK,
    )

    # Panel 2: the ratings themselves, on the scale they were given on.
    ax = fig.add_subplot(grid[0, 1])
    for condition in CONDITIONS:
        stats = [run_average(merchants, "rating_general_B", condition, r) for r in ROUNDS]
        ax.plot(
            ROUNDS, [s[0] for s in stats], color=COLOR[condition], lw=2.2,
            marker=MARKER[condition], ms=7.5, mec=SURFACE, mew=1.6,
            label=CONDITION_NAME[condition], zorder=3,
        )
    others = [
        float(merchants[merchants["round"] == r]["others"].mean()) for r in ROUNDS
    ]
    ax.plot(
        ROUNDS, others, color=INK, lw=1.2, ls=(0, (2, 2)),
        label="Goods A and C (all runs)", zorder=4,
    )
    # Zoomed to a three-point window of the 1-10 scale: wide enough that the
    # lines still read as one bundle, tight enough to see the tenths between
    # them.  Every integer in view gets a tick, and the panel says it is zoomed.
    ax.set_ylim(5, 8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.grid(axis="y", which="minor", color=GRID, lw=0.5)
    ax.text(
        ROUNDS[-1] + 0.5, 5.04, "axis zoomed to 5-8 of the 1-10 scale",
        fontsize=8, color=INK_SECONDARY, ha="right",
    )
    mark_phases(ax)
    style_time_axis(ax, xlabel=False)
    ax.set_xlabel("Trading round")
    ax.set_ylabel("Rating, 1 (not valuable) to 10 (extremely)")
    ax.set_title(
        f'"How valuable is B to you?"\nall within {rating_spread:.1f} of a point',
        loc="left", color=INK,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=8)

    # Panel 3: how many Merchants budged at all.
    ax = fig.add_subplot(grid[0, 2])
    top = condition_lines(
        ax, merchants, "rates_b_highest", end_labels=True, label_only=("Expert",)
    )
    ax.set_ylim(0, max(top * 1.3, 14))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    mark_phases(ax)
    style_time_axis(ax, xlabel=False)
    ax.set_xlabel("Trading round")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylabel("Merchants rating B above A and C")
    ax.set_title(
        "How many rated B higher at all?\na few, only under Expert, only while it was up",
        loc="left", color=INK,
    )

    headline(
        fig,
        "What Merchants say they want barely moves",
        "Asked to set their holdings aside, Merchants name the same ideal bundle "
        "in every condition and rate B the same as the other goods.\nThe Expert "
        f"bulletin nudges a small minority (at most {expert_peak:.0f}%) to rate B "
        "one point higher, and that disappears when the bulletin does.",
    )
    footnote(fig, runs_note(runs_df))
    save(fig, path)


# ---------------------------------------------------------------------------
# H3: what they do against what they say
# ---------------------------------------------------------------------------

def fig_do_say(
    shadow_df: pd.DataFrame, probes_df: pd.DataFrame, runs_df: pd.DataFrame, path: Path
) -> None:
    merchants = probes_df[probes_df["role"] == "Merchant"]
    panels: list[dict[str, Any]] = [
        {
            "frame": shadow_df[shadow_df["direction"] == "acquire"],
            "value": "accept", "scale": 100.0, "unit": "%",
            "title": "Says yes to\nbuying B",
            "ylabel": "Offers accepted", "ylim": (0, 55),
        },
        {
            "frame": shadow_df[shadow_df["direction"] == "surrender"],
            "value": "accept", "scale": 100.0, "unit": "%",
            "title": "Says yes to\nselling B",
            "ylabel": "Offers accepted", "ylim": (0, 15),
        },
        {
            "frame": merchants, "value": "Pb_general", "scale": 1.0,
            "unit": "", "title": "How much higher B is rated\nthan the other goods",
            "ylabel": "Rating points (scale allows -9 to +9)", "ylim": (-9, 9),
        },
        {
            "frame": merchants, "value": "bundle_B", "scale": 1.0, "unit": "",
            "title": "Units of B in\nthe ideal bundle",
            "ylabel": "Units, out of 6", "ylim": (0, 6),
        },
    ]
    fig = plt.figure(figsize=(13, 6.6))
    grid = fig.add_gridspec(
        1, 4, left=0.055, right=0.985, top=0.68, bottom=0.27, wspace=0.42
    )
    for column, spec in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        values: dict[str, float] = {}
        for position, condition in enumerate(CONDITIONS):
            mean, low, high = run_average(
                spec["frame"], spec["value"], condition, SHOWN_ROUND
            )
            scale = spec["scale"]
            values[condition] = scale * mean
            ax.bar(position, scale * mean, width=0.64, color=COLOR[condition])
            if high > low:
                ax.errorbar(
                    position, scale * mean,
                    yerr=[[scale * (mean - max(low, 0))], [scale * (high - mean)]],
                    fmt="none", ecolor=INK_SECONDARY, elinewidth=1, capsize=2.5,
                )
            shown = scale * mean
            if spec["unit"] == "%":
                text = f"{shown:.0f}%" if shown >= 1 else f"{shown:.1f}%"
            else:
                text = f"{shown:.2f}"
            ax.annotate(
                text, (position, scale * max(high, mean)),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=10, color=INK, fontweight="bold",
            )
        ax.set_ylim(*spec["ylim"])
        if spec["ylim"][0] < 0:
            ax.axhline(0, color="#9c9b95", lw=1)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(["None", "Scarcity", "Expert"])
        ax.set_ylabel(spec["ylabel"])
        if spec["unit"] == "%":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(spec["title"], loc="left", color=INK, fontsize=10.5)

        # The plain-words reading of the panel, under the axis.
        base = values["Control"]
        lines = []
        for condition in fea.TREATMENTS:
            change = values[condition] - base
            if spec["unit"] == "%":
                lines.append(f"{condition}: {change:+.0f} points")
            else:
                lines.append(f"{condition}: {change:+.2f}")
        ax.text(
            0.0, -0.13, "vs no bulletin\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=9, color=INK_SECONDARY, va="top",
        )

    for x, label in ((0.055, "WHAT MERCHANTS DO"), (0.545, "WHAT MERCHANTS SAY")):
        fig.text(x, 0.775, label, fontsize=10, fontweight="bold", color=INK)
    fig.add_artist(
        plt.Line2D([0.515, 0.515], [0.08, 0.80], color="#b9b8b2", lw=1)
    )
    headline(
        fig,
        "Only the Scarcity bulletin clearly changes what Merchants do; neither "
        "changes what they say",
        f"The same Merchants, at round {SHOWN_ROUND}, with the bulletin still on "
        "the board. Left: their decisions on standardised offers. Right: their\n"
        "answers when asked directly about their preferences, drawn on the "
        "full range each question allows.",
    )
    footnote(
        fig,
        runs_note(runs_df)
        .replace("Each point", "Each bar")
        .replace("Shaded band", "Whisker"),
    )
    save(fig, path)


# ---------------------------------------------------------------------------
# H4: after the bulletin is removed
# ---------------------------------------------------------------------------

def gap_panel(
    ax: plt.Axes, frame: pd.DataFrame, value: str, scale: float
) -> float:
    """Treated-minus-Control lines for both bulletins; returns the widest point."""
    widest = 0.0
    for condition in fea.TREATMENTS:
        stats = [gap_to_control(frame, value, condition, r) for r in ROUNDS]
        gaps = [scale * stat[0] for stat in stats]
        lows = [scale * stat[1] for stat in stats]
        highs = [scale * stat[2] for stat in stats]
        ax.fill_between(ROUNDS, lows, highs, color=COLOR[condition], alpha=0.14, lw=0)
        ax.plot(
            ROUNDS, gaps, color=COLOR[condition], lw=2.2,
            marker=MARKER[condition], ms=7.5, mec=SURFACE, mew=1.6,
            label=CONDITION_NAME[condition], zorder=3,
        )
        widest = max(widest, np.nanmax(np.abs(lows + highs)))
    ax.axhline(
        0, color=COLOR["Control"], lw=1.6, zorder=2,
        label="No bulletin (the reference)",
    )
    style_time_axis(ax)
    return widest


def fig_after(
    shadow_df: pd.DataFrame,
    probes_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(17, 6.4))
    grid = fig.add_gridspec(
        1, 3, left=0.05, right=0.985, top=0.75, bottom=0.15, wspace=0.24
    )

    # Panels 1 and 2 share one percentage-point scale, so the buying and
    # selling gaps can be compared directly.
    policy_axes = []
    limit = 0.0
    for column, (direction, title) in enumerate(
        (
            ("acquire", "Buying B: gap to no-bulletin Merchants"),
            ("surrender", "Selling B: gap to no-bulletin Merchants"),
        )
    ):
        ax = fig.add_subplot(grid[0, column])
        subset = shadow_df[shadow_df["direction"] == direction]
        limit = max(limit, gap_panel(ax, subset, "accept", 100.0))
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel("Percentage points more (+) or fewer (-) offers accepted")
        policy_axes.append(ax)
    for ax in policy_axes:
        ax.set_ylim(-limit * 1.12, limit * 1.22)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f"))
        mark_phases(ax)
    policy_axes[0].legend(loc="lower left", frameon=False)

    # Panel 3 is in rating points, a different unit, so it has its own axis.
    # Each bulletin Merchant is compared with no-bulletin Merchants holding
    # exactly the same A, B and C, so differences in holdings cannot pass for
    # a bulletin effect (the bulletins change what Merchants end up holding).
    ax = fig.add_subplot(grid[0, 2])
    widest = 0.0
    for condition in fea.TREATMENTS:
        stats = [
            fea.matched_gap(probes_df, "Pb_inventory", condition, r) for r in ROUNDS
        ]
        gaps = [stat["estimate"] for stat in stats]
        lows = [stat.get("ci_low", stat["estimate"]) for stat in stats]
        highs = [stat.get("ci_high", stat["estimate"]) for stat in stats]
        lows = [g if not np.isfinite(lo) else lo for g, lo in zip(gaps, lows)]
        highs = [g if not np.isfinite(hi) else hi for g, hi in zip(gaps, highs)]
        ax.fill_between(ROUNDS, lows, highs, color=COLOR[condition], alpha=0.14, lw=0)
        ax.plot(
            ROUNDS, gaps, color=COLOR[condition], lw=2.2,
            marker=MARKER[condition], ms=7.5, mec=SURFACE, mew=1.6,
            label=CONDITION_NAME[condition], zorder=3,
        )
        widest = max(widest, np.nanmax(np.abs(lows + highs)))
    ax.axhline(0, color=COLOR["Control"], lw=1.6, zorder=2)
    style_time_axis(ax)
    span = max(widest, 0.3) * 1.25
    ax.set_ylim(-span, span)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))
    mark_phases(ax)
    ax.set_title(
        "Rating B above A and C given current holdings:\ngap to no-bulletin "
        "Merchants holding exactly the same goods",
        loc="left", color=INK,
    )
    ax.set_ylabel("Rating points higher (+) or lower (-), on the 1-10 scale")

    headline(
        fig,
        "After the bulletin comes down, buying and selling go back to normal",
        "Each line is how far Merchants who saw a bulletin are from Merchants "
        "who never did; on the grey line there is no difference. The buying "
        "and selling gaps\nclose once the bulletin is removed after round "
        f"{BULLETIN_END}. In how they value what they hold, bulletin Merchants "
        "differ by at most about 0.1 points: they rate every good slightly\n"
        "higher, A and C a little more than B, so B's lead over the other two "
        "goods shrinks rather than grows.",
    )
    footnote(
        fig,
        runs_note(runs_df)
        + ' Right panel: the "given your current inventory" rating of B minus '
        "the mean of A and C.",
    )
    save(fig, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-dir", type=Path, default=fea.DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=fea.DEFAULT_OUT_DIR)
    args = parser.parse_args()

    data = fea.load_dataset(args.base_dir.resolve())
    probes_df, shadow_df, runs_df = data["probes"], data["shadow"], data["runs"]
    print(f"Runs: {len(runs_df)} analysed, {len(data['excluded'])} excluded")
    out = args.out_dir.resolve()

    # The titles and subtitles below state findings.  They were written for the
    # October 2026 collection; re-read every figure against the report whenever
    # the runs change, because a sentence here cannot update itself.
    fig_offers(
        shadow_df, runs_df, "acquire",
        "The Scarcity bulletin makes Merchants buy B more often; the Expert one "
        "does not",
        "Each Merchant was offered a unit of B in exchange for 1 or 2 units of "
        "another good. The chart shows how often they accepted.\nExpert "
        "Merchants already hold more B by round 10, so these offers are worse "
        "deals for them, and they accept them no more often.",
        "Offers to buy B that were accepted",
        out / "h1_1_buying_b.png",
    )
    fig_offers(
        shadow_df, runs_df, "surrender",
        "Merchants rarely sell B anyway; with a bulletin up, almost never",
        "Each Merchant holding B was offered 1 or 2 units of another good for "
        "it. The chart shows how often they accepted.\nWithout a bulletin about "
        "2% of offers are accepted by round 10; with either bulletin, under 1%. "
        "Acceptance returns once the bulletin is removed.",
        "Offers to sell B that were accepted",
        out / "h1_2_selling_b.png",
    )
    fig_price(shadow_df, fea.reservation_frame(shadow_df), runs_df, out / "h1_3_price_of_b.png")
    fig_stated(probes_df, runs_df, out / "h2_stated_preference.png")
    fig_do_say(shadow_df, probes_df, runs_df, out / "h3_do_versus_say.png")
    fig_after(shadow_df, probes_df, runs_df, out / "h4_after_removal.png")


if __name__ == "__main__":
    main()
