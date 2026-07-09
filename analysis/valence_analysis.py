#!/usr/bin/env python3
"""
Valence test analysis — preference drift and trade behavior across all agent types.

Four conditions:
  - Extremely + Broadcast        ("extremely fair and necessary" / "extremely unfair and unnecessary")
  - Extremely + No Broadcast     (same wording, no bulletin)
  - Wonderful/Awful + Broadcast  ("wonderful" / "awful")
  - Wonderful/Awful + No Broadcast

Each condition has 10 runs.

Agent types and hard-coded baselines
  Builder  (agents 1 & 2): ratings A=10 B=5  C=1;  bundle A=5 B=1 C=0
  Weaver   (agents 3 & 4): ratings A=2  B=10 C=6;  bundle A=0 B=5 C=1
  Merchant (agents 5 & 6): ratings A=7  B=7  C=7;  bundle A=2 B=2 C=2

Builder agents label their own trades:
  acquiring A → "fair / wonderful"
  acquiring B → "unfair / awful"

Figures saved to analysis/ as:
  fig1_ratings_{type}.png          — probed good ratings over rounds
  fig2_bundle_{type}.png           — desired-bundle composition over rounds
  fig3_net_acquisition_{type}.png  — cumulative net acquisition per round
  fig4_trade_behavior_{type}.png   — acceptance prob, WTP, proposal targets
  fig5_broadcast_compare_{type}.png — broadcast vs no-broadcast (pooled wording)
  fig6_wording_compare_{type}.png  — "extremely" vs "wonderful/awful" (broadcast only)
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats as spstats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/qnguyen/Desktop/LLM_Barter_Env/runs/valence test")
OUT_DIR  = Path("/home/qnguyen/Desktop/LLM_Barter_Env/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Conditions ───────────────────────────────────────────────────────────────
CONDITIONS = {
    "Extremely\n+ Broadcast":          "extremely with broadcast",
    "Extremely\n+ No Broadcast":       "extremely without broadcast",
    "Wonderful/Awful\n+ Broadcast":    "wonderful awful with broadcast",
    "Wonderful/Awful\n+ No Broadcast": "wonderful awful without broadcast",
}
COND_LABELS = list(CONDITIONS.keys())
SHORT_LABELS = ["Ext+BC", "Ext−BC", "W/A+BC", "W/A−BC"]

BC_LABELS   = [l for l in COND_LABELS if "Broadcast" in l and "No" not in l]
NOBC_LABELS = [l for l in COND_LABELS if "No Broadcast" in l]
EXT_BC      = "Extremely\n+ Broadcast"
WA_BC       = "Wonderful/Awful\n+ Broadcast"

# ─── Agent type definitions ────────────────────────────────────────────────────
AGENT_TYPES = {
    "builder": {
        "ids":              {"player_1", "player_2"},
        "label":            "Builder (agents 1 & 2)",
        "short":            "builder",
        "baseline_ratings": {"A": 10, "B": 5, "C": 1},
        "baseline_bundle":  {"A": 5,  "B": 1, "C": 0},
    },
    "weaver": {
        "ids":              {"player_3", "player_4"},
        "label":            "Weaver (agents 3 & 4)",
        "short":            "weaver",
        "baseline_ratings": {"A": 2,  "B": 10, "C": 6},
        "baseline_bundle":  {"A": 0,  "B": 5,  "C": 1},
    },
    "merchant": {
        "ids":              {"player_5", "player_6"},
        "label":            "Merchant (agents 5 & 6)",
        "short":            "merchant",
        "baseline_ratings": {"A": 7,  "B": 7,  "C": 7},
        "baseline_bundle":  {"A": 2,  "B": 2,  "C": 2},
    },
}

GOODS        = ["A", "B", "C"]
PROBE_ROUNDS = [0, 2, 4, 6, 8, 10]
GCOL         = {"A": "#1565C0", "B": "#B71C1C", "C": "#1B5E20"}
CCOL         = ["#1565C0", "#FF6F00", "#1B5E20", "#6A1B9A"]   # one per condition

# ─── Wave 2: runs_7-7 (market-framing conditions, multi-unit trades) ──────────
# Unlike the valence-test 2x2 design, this batch is a single control plus
# three treatment wordings. The broadcast trigger also differs: control
# broadcasts Builder (agents 1 & 2) gaining A/B with "fair"/"unfair" moral
# language (same mechanic as the original "extremely" condition); the three
# treatments broadcast ONLY Weaver (agents 3 & 4) gaining Good B, always in
# B-favorable language, never moralizing, via three different rhetorical
# framings (scarcity, bandwagon/social-proof, expert authority).
BASE_DIR2 = Path("/home/qnguyen/Desktop/LLM_Barter_Env/runs/runs_7-7")

CONDITIONS2 = {
    "Control\n(Builder fair/unfair)": "control_runs",
    "Scarcity\n(Weaver framing)":     "condition_1",
    "Bandwagon\n(Weaver framing)":    "condition_2",
    "Expert\n(Weaver framing)":       "condition_3",
}
COND_LABELS2  = list(CONDITIONS2.keys())
SHORT_LABELS2 = ["Control", "Scarcity", "Bandwagon", "Expert"]
CCOL2         = ["#455A64", "#EF6C00", "#6A1B9A", "#00838F"]   # one per condition
CONTROL_LABEL2    = COND_LABELS2[0]
TREATMENT_LABELS2 = COND_LABELS2[1:]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_run(run_dir: Path) -> dict:
    run = {}
    for fname, key in [("preference_probes.json", "probes"), ("trades.json", "trades")]:
        p = run_dir / fname
        if p.exists():
            with open(p) as f:
                run[key] = json.load(f)
    ep = run_dir / "events.jsonl"
    if ep.exists():
        with open(ep) as f:
            run["events"] = [json.loads(line) for line in f]
    return run


def load_condition(cond_dir_name: str, base_dir: Path = BASE_DIR) -> list[dict]:
    d = base_dir / cond_dir_name
    return [load_run(sub) for sub in sorted(d.iterdir()) if sub.is_dir()]


ALL_RUNS: dict[str, list[dict]] = {
    label: load_condition(cdir) for label, cdir in CONDITIONS.items()
}

ALL_RUNS2: dict[str, list[dict]] = {
    label: load_condition(cdir, BASE_DIR2) for label, cdir in CONDITIONS2.items()
}

# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS HELPER
# ══════════════════════════════════════════════════════════════════════════════

def mean_ci(arr) -> tuple[float, float, float]:
    """(mean, lower 95% CI, upper 95% CI) for a 1-D array."""
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(arr))
    if n < 2:
        return m, m, m
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    t  = float(spstats.t.ppf(0.975, df=n - 1))
    return m, m - t * se, m + t * se

# ══════════════════════════════════════════════════════════════════════════════
# PROBE EXTRACTION  (generic: pass any set of agent IDs)
# ══════════════════════════════════════════════════════════════════════════════

def agent_probe_series(runs: list[dict], agent_ids: set[str], key: str
                       ) -> dict[int, np.ndarray]:
    """
    {round_index: array(n_runs, 3)} where axis-1 = [A, B, C].
    Each run row = mean over the agents in agent_ids.
    key: "ratings" | "desired_bundle"
    """
    per_round: dict[int, list] = defaultdict(list)
    for run in runs:
        if "probes" not in run:
            continue
        per_agent: dict[int, list] = defaultdict(list)
        for probe in run["probes"]:
            if probe["player_id"] not in agent_ids:
                continue
            r    = probe["round_index"]
            resp = probe["response"]
            if key == "ratings":
                vals = [resp["ratings"].get(g, np.nan) for g in GOODS]
            else:
                vals = [resp["desired_bundle_6_units"].get(g, 0) for g in GOODS]
            per_agent[r].append(vals)
        for r, vlist in per_agent.items():
            per_round[r].append(np.nanmean(vlist, axis=0))
    return {r: np.array(v) for r, v in sorted(per_round.items())}


def pooled_probe_series(cond_labels: list[str], agent_ids: set[str], key: str
                        ) -> dict[int, np.ndarray]:
    per_round: dict[int, list] = defaultdict(list)
    for lbl in cond_labels:
        for r, arr in agent_probe_series(ALL_RUNS[lbl], agent_ids, key).items():
            per_round[r].extend(arr.tolist())
    return {r: np.array(v) for r, v in sorted(per_round.items())}

# ══════════════════════════════════════════════════════════════════════════════
# TRADE BEHAVIOUR EXTRACTION  (generic)
# ══════════════════════════════════════════════════════════════════════════════

def agent_perspective(trade: dict, agent_ids: set[str]) -> tuple[dict, dict] | None:
    """(goods_received, goods_given) from the focal agents' point of view."""
    pt = trade.get("proposed_trade")
    if not pt:
        return None
    proposer  = trade.get("proposer_id", "")
    responder = trade.get("responder_id", "")
    give_d    = pt.get("give", {})
    recv_d    = pt.get("receive", {})
    if proposer in agent_ids:
        return recv_d, give_d
    if responder in agent_ids:
        return give_d, recv_d
    return None


def net_acquisition_series(runs: list[dict], agent_ids: set[str]
                           ) -> dict[int, np.ndarray]:
    """Cumulative net acquisition per round, {round_index: array(n_runs, 3)}."""
    per_round: dict[int, list] = defaultdict(list)
    for run in runs:
        if "trades" not in run:
            continue
        accepted   = [t for t in run["trades"] if t.get("accepted", False)]
        all_rounds = sorted({t["round_index"] for t in run["trades"]})
        cum = np.zeros(3)
        for r in all_rounds:
            for t in accepted:
                if t["round_index"] != r:
                    continue
                mp = agent_perspective(t, agent_ids)
                if mp is None:
                    continue
                recv, gave = mp
                for i, g in enumerate(GOODS):
                    cum[i] += recv.get(g, 0) - gave.get(g, 0)
            per_round[r].append(cum.copy())
    return {r: np.array(v) for r, v in sorted(per_round.items())}


def acceptance_data(runs: list[dict], agent_ids: set[str]) -> list[dict]:
    """Commitment decisions where a focal agent was the RESPONDER."""
    records = []
    for run in runs:
        if "events" not in run:
            continue
        for e in run["events"]:
            if e["event_type"] != "commitment_decision":
                continue
            p = e["payload"]
            if p.get("responder_id") not in agent_ids:
                continue
            pt = p.get("proposed_trade")
            if not pt:
                continue
            accepted = p.get("decision") == "accept"
            for g, qty in pt.get("give", {}).items():
                if qty and qty > 0:
                    records.append({
                        "good_acquired": g,
                        "accepted":      accepted,
                        "round":         e.get("round_index", -1),
                        "qty_acquired":  qty,
                        "give_dict":     pt.get("receive", {}),
                    })
    return records


def proposal_targets(runs: list[dict], agent_ids: set[str]) -> dict[str, int]:
    """What goods do focal agents seek as proposers (offer action type)."""
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        if "events" not in run:
            continue
        for e in run["events"]:
            if e["event_type"] != "negotiation_turn":
                continue
            if e.get("player_id") not in agent_ids:
                continue
            action = e["payload"]["action"]
            if action["action_type"] != "offer":
                continue
            pt = action.get("proposed_trade")
            if not pt:
                continue
            for g, qty in pt.get("receive", {}).items():
                if qty and qty > 0:
                    counts[g] += qty
    return dict(counts)


def wtp_data(runs: list[dict], agent_ids: set[str]) -> dict[str, dict[str, list]]:
    """Units of other goods given per unit of each good acquired."""
    result: dict[str, dict[str, list]] = {g: defaultdict(list) for g in GOODS}
    for run in runs:
        if "trades" not in run:
            continue
        for t in run["trades"]:
            if not t.get("accepted"):
                continue
            mp = agent_perspective(t, agent_ids)
            if mp is None:
                continue
            recv, gave = mp
            for g_acq, qty_acq in recv.items():
                if qty_acq <= 0:
                    continue
                for g_gave, qty_gave in gave.items():
                    if qty_gave > 0:
                        result[g_acq][g_gave].append(qty_gave / qty_acq)
    return result


def acc_prob_ci(records: list[dict], good: str) -> tuple[float, float, float]:
    vals = [r["accepted"] for r in records if r["good_acquired"] == good]
    if not vals:
        return np.nan, np.nan, np.nan
    n  = len(vals)
    p  = float(np.mean(vals))
    se = float(np.sqrt(p * (1 - p) / n)) if n > 1 else 0.0
    return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def mean_wtp(wtp: dict, good_acq: str) -> tuple[float, float]:
    all_r = [v for vals in wtp.get(good_acq, {}).values() for v in vals]
    if not all_r:
        return np.nan, 0.0
    return float(np.mean(all_r)), float(np.std(all_r, ddof=1) / np.sqrt(len(all_r)))


def proposal_fracs(targets: dict) -> dict[str, float]:
    total = sum(targets.values()) or 1
    return {g: targets.get(g, 0) / total for g in GOODS}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       130,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def _probe_subplot(ax, series: dict[int, np.ndarray], baselines: dict[str, float],
                   title: str, ylabel: str, xlabel: bool) -> None:
    """Populate one subplot for a probe-over-rounds panel."""
    for gi, g in enumerate(GOODS):
        rounds, mn_vals, lo_vals, hi_vals = [], [], [], []
        for r in sorted(series):
            m, lo, hi = mean_ci(series[r][:, gi])
            rounds.append(r); mn_vals.append(m); lo_vals.append(lo); hi_vals.append(hi)
        c = GCOL[g]
        ax.plot(rounds, mn_vals, color=c, lw=2, marker="o", ms=5, label=f"Good {g}")
        ax.fill_between(rounds, lo_vals, hi_vals, color=c, alpha=0.15)
        ax.axhline(baselines[g], color=c, lw=1, ls=":", alpha=0.55)

    ax.set_title(title.replace("\n", " "), fontweight="bold")
    ax.set_xticks(PROBE_ROUNDS)
    ax.set_xlim(-0.5, 10.5)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Round")


def fig_probe(series_per_cond: dict, baselines: dict[str, float],
              ylabel: str, suptitle: str, ylim: tuple, save_path: Path,
              yticks: list | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax_i, (lbl, series) in enumerate(series_per_cond.items()):
        ax = axes.flat[ax_i]
        _probe_subplot(ax, series, baselines, lbl,
                       ylabel if ax_i % 2 == 0 else "",
                       ax_i >= 2)
        ax.set_ylim(*ylim)
        if yticks:
            ax.set_yticks(yticks)

    good_patches = [mpatches.Patch(color=GCOL[g], label=f"Good {g}") for g in GOODS]
    base_lines   = [plt.Line2D([0], [0], color=GCOL[g], ls=":", alpha=0.7,
                               label=f"Baseline {g}={int(baselines[g])}")
                    for g in GOODS]
    fig.legend(handles=good_patches + base_lines, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_net(net_per_cond: dict, agent_label: str, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax_i, (lbl, series) in enumerate(net_per_cond.items()):
        ax = axes.flat[ax_i]
        for gi, g in enumerate(GOODS):
            rounds, mn_vals, lo_vals, hi_vals = [], [], [], []
            for r in sorted(series):
                m, lo, hi = mean_ci(series[r][:, gi])
                rounds.append(r); mn_vals.append(m); lo_vals.append(lo); hi_vals.append(hi)
            c = GCOL[g]
            ax.plot(rounds, mn_vals, color=c, lw=2, marker="o", ms=4, label=f"Good {g}")
            ax.fill_between(rounds, lo_vals, hi_vals, color=c, alpha=0.15)
        ax.axhline(0, color="grey", lw=1, ls="--", alpha=0.6)
        ax.set_title(lbl.replace("\n", " "), fontweight="bold")
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.5, 10.5)
        if ax_i % 2 == 0:
            ax.set_ylabel("Cumulative net units acquired")
        if ax_i >= 2:
            ax.set_xlabel("Round")

    handles = [mpatches.Patch(color=GCOL[g], label=f"Good {g}") for g in GOODS] + \
              [plt.Line2D([0], [0], color="grey", ls="--", label="Net = 0")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{agent_label}: Cumulative Net Acquisition of Goods\n"
        "(net = units acquired − units given; mean ± 95% CI across 10 runs)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_trade_behavior(acc_records_per_cond: dict,
                       proposals_per_cond: dict,
                       wtp_per_cond: dict,
                       agent_label: str,
                       save_path: Path,
                       cond_labels: list | None = None,
                       short_labels: list | None = None,
                       ccol: list | None = None,
                       panel_e_groups: list | None = None,
                       panel_e_title: str = "E  Acceptance rate over rounds\n(pooled across wording conditions)") -> None:
    cond_labels  = cond_labels  if cond_labels  is not None else COND_LABELS
    short_labels = short_labels if short_labels is not None else SHORT_LABELS
    ccol         = ccol         if ccol         is not None else CCOL
    if panel_e_groups is None:
        panel_e_groups = [
            ("With broadcast", BC_LABELS,   "#B71C1C"),
            ("No broadcast",   NOBC_LABELS, "#1565C0"),
        ]

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    ax_acc  = fig.add_subplot(gs[0, :2])
    ax_n    = fig.add_subplot(gs[0, 2])
    ax_wtp  = fig.add_subplot(gs[1, 0])
    ax_prop = fig.add_subplot(gs[1, 1])
    ax_time = fig.add_subplot(gs[1, 2])

    # Panel A — acceptance probability
    n_conds = len(cond_labels)
    bar_w   = 0.18
    offsets = np.linspace(-(len(GOODS) - 1) / 2 * bar_w,
                           (len(GOODS) - 1) / 2 * bar_w, len(GOODS))
    x_base  = np.arange(n_conds)

    for gi, g in enumerate(GOODS):
        ps, los, his = [], [], []
        for lbl in cond_labels:
            p, lo, hi = acc_prob_ci(acc_records_per_cond[lbl], g)
            ps.append(p); los.append(lo); his.append(hi)
        x = x_base + offsets[gi]
        err_lo = [max(0, p - lo) for p, lo in zip(ps, los)]
        err_hi = [max(0, hi - p) for p, hi in zip(ps, his)]
        ax_acc.bar(x, ps, width=bar_w, color=GCOL[g], label=f"Acquire {g}",
                   alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_acc.errorbar(x, ps, yerr=[err_lo, err_hi], fmt="none",
                        ecolor="black", elinewidth=1.2, capsize=3)

    ax_acc.set_xticks(x_base)
    ax_acc.set_xticklabels(short_labels)
    ax_acc.set_ylabel("P(accept)")
    ax_acc.set_ylim(0, 1.15)
    ax_acc.set_title("A  Acceptance probability (agent = responder)\nP(accept | acquiring good X)")
    ax_acc.legend(loc="upper right", fontsize=8)
    ax_acc.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Panel B — sample sizes table
    ax_n.axis("off")
    rows = [["Condition"] + [f"N(acq {g})" for g in GOODS]]
    for lbl, sh in zip(cond_labels, short_labels):
        row = [sh] + [str(sum(1 for r in acc_records_per_cond[lbl]
                              if r["good_acquired"] == g)) for g in GOODS]
        rows.append(row)
    tbl = ax_n.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#ECEFF1"); cell.set_text_props(fontweight="bold")
    ax_n.set_title("B  Sample sizes\n(commitment decisions as responder)",
                   fontsize=10, fontweight="bold")

    # Panel C — WTP
    bar_w2 = 0.2
    x2 = np.arange(len(GOODS))
    for ci, lbl in enumerate(cond_labels):
        wtp = wtp_per_cond[lbl]
        means, errs = [], []
        for g in GOODS:
            m, se = mean_wtp(wtp, g)
            means.append(m); errs.append(se)
        off = (ci - (n_conds - 1) / 2) * bar_w2
        ax_wtp.bar(x2 + off, means, width=bar_w2, color=ccol[ci],
                   label=short_labels[ci], alpha=0.85)
        ax_wtp.errorbar(x2 + off, means, yerr=errs, fmt="none",
                        ecolor="black", elinewidth=1, capsize=2.5)
    ax_wtp.set_xticks(x2)
    ax_wtp.set_xticklabels([f"Acquire {g}" for g in GOODS])
    ax_wtp.set_ylabel("Mean units given per unit acquired")
    ax_wtp.set_title("C  Willingness-to-pay\n(units of other goods given / unit of X received)")
    ax_wtp.legend(fontsize=7, ncol=2)
    ax_wtp.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.6)

    # Panel D — proposal targets
    bottoms = np.zeros(n_conds)
    for gi, g in enumerate(GOODS):
        heights = [proposal_fracs(proposals_per_cond[lbl])[g] for lbl in cond_labels]
        ax_prop.bar(range(n_conds), heights, bottom=bottoms,
                    color=GCOL[g], label=f"Seek {g}", alpha=0.9)
        bottoms += np.array(heights)
    ax_prop.set_xticks(range(n_conds))
    ax_prop.set_xticklabels(short_labels)
    ax_prop.set_ylabel("Fraction of outgoing offers")
    ax_prop.set_ylim(0, 1.05)
    ax_prop.set_title("D  Goods sought in proposals\n(as proposer; fraction of offer-actions)")
    ax_prop.legend(loc="upper right", fontsize=8)
    ax_prop.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Panel E — acceptance rate over rounds (grouped, e.g. broadcast vs no-broadcast)
    def pooled_acc_rounds(labels):
        pooled: dict[int, list] = defaultdict(list)
        for lbl in labels:
            for rec in acc_records_per_cond[lbl]:
                pooled[rec["round"]].append(int(rec["accepted"]))
        result = {}
        for r in sorted(pooled):
            vals = pooled[r]
            n = len(vals); p = float(np.mean(vals))
            se = float(np.sqrt(p * (1 - p) / max(n, 1)))
            result[r] = (p, p - 1.96 * se, p + 1.96 * se)
        return result

    for label_str, labels, color in panel_e_groups:
        series = pooled_acc_rounds(labels)
        if not series:
            continue
        rounds = sorted(series)
        ps  = [series[r][0] for r in rounds]
        los = [series[r][1] for r in rounds]
        his = [series[r][2] for r in rounds]
        ax_time.plot(rounds, ps, color=color, lw=2, marker="o", ms=4, label=label_str)
        ax_time.fill_between(rounds, los, his, color=color, alpha=0.15)

    ax_time.set_xlabel("Round")
    ax_time.set_ylabel("P(accept)")
    ax_time.set_ylim(0, 1.1)
    ax_time.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_time.set_title(panel_e_title)
    ax_time.legend(fontsize=8)
    ax_time.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5)

    fig.suptitle(f"{agent_label}: Trade Behaviour Analysis",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_broadcast_compare(bc_series: dict, nobc_series: dict,
                          baselines: dict[str, float],
                          agent_label: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for gi, g in enumerate(GOODS):
        ax = axes[gi]
        for series, color, lbl, ls in [
            (bc_series,   "#B71C1C", "With broadcast", "-"),
            (nobc_series, "#1565C0", "No broadcast",   "--"),
        ]:
            rounds, mn_vals, lo_vals, hi_vals = [], [], [], []
            for r in sorted(series):
                m, lo, hi = mean_ci(series[r][:, gi])
                rounds.append(r); mn_vals.append(m); lo_vals.append(lo); hi_vals.append(hi)
            ax.plot(rounds, mn_vals, color=color, lw=2.5, ls=ls,
                    marker="o", ms=5, label=lbl)
            ax.fill_between(rounds, lo_vals, hi_vals, color=color, alpha=0.12)
        ax.axhline(baselines[g], color=GCOL[g], lw=1, ls=":", alpha=0.65,
                   label=f"Baseline ({int(baselines[g])})")
        ax.set_title(f"Good {g}", fontweight="bold", color=GCOL[g], fontsize=12)
        ax.set_xlabel("Round")
        ax.set_xticks(PROBE_ROUNDS)
        ax.set_xlim(-0.5, 10.5)
        if gi == 0:
            ax.set_ylabel("Stated valuation (1–10)")
    axes[2].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        f"{agent_label}: Broadcast vs No-Broadcast — Probed Ratings\n"
        "(pooled across wording; mean ± 95% CI, n ≈ 20 runs per group)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_wording_compare(ext_series: dict, wa_series: dict,
                        baselines: dict[str, float],
                        agent_label: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for gi, g in enumerate(GOODS):
        ax = axes[gi]
        for series, color, lbl, ls in [
            (ext_series, "#6A1B9A", '"Extremely fair/unfair"', "-"),
            (wa_series,  "#E65100", '"Wonderful / Awful"',     "--"),
        ]:
            rounds, mn_vals, lo_vals, hi_vals = [], [], [], []
            for r in sorted(series):
                m, lo, hi = mean_ci(series[r][:, gi])
                rounds.append(r); mn_vals.append(m); lo_vals.append(lo); hi_vals.append(hi)
            ax.plot(rounds, mn_vals, color=color, lw=2.5, ls=ls,
                    marker="o", ms=5, label=lbl)
            ax.fill_between(rounds, lo_vals, hi_vals, color=color, alpha=0.12)
        ax.axhline(baselines[g], color=GCOL[g], lw=1, ls=":", alpha=0.65,
                   label=f"Baseline ({int(baselines[g])})")
        ax.set_title(f"Good {g}", fontweight="bold", color=GCOL[g], fontsize=12)
        ax.set_xlabel("Round")
        ax.set_xticks(PROBE_ROUNDS)
        ax.set_xlim(-0.5, 10.5)
        if gi == 0:
            ax.set_ylabel("Stated valuation (1–10)")
    axes[2].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        f"{agent_label}: Wording Comparison (broadcast conditions only)\n"
        "(mean ± 95% CI across 10 runs per wording)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_condition_compare(series_per_cond: dict, baselines: dict[str, float],
                          agent_label: str, save_path: Path,
                          cond_labels: list, colors: list, suptitle: str) -> None:
    """N-way per-good comparison (one line per condition), 3 panels (A/B/C)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for gi, g in enumerate(GOODS):
        ax = axes[gi]
        for lbl, color in zip(cond_labels, colors):
            series = series_per_cond[lbl]
            rounds, mn_vals, lo_vals, hi_vals = [], [], [], []
            for r in sorted(series):
                m, lo, hi = mean_ci(series[r][:, gi])
                rounds.append(r); mn_vals.append(m); lo_vals.append(lo); hi_vals.append(hi)
            ax.plot(rounds, mn_vals, color=color, lw=2.5, marker="o", ms=5,
                    label=lbl.replace("\n", " "))
            ax.fill_between(rounds, lo_vals, hi_vals, color=color, alpha=0.12)
        ax.axhline(baselines[g], color=GCOL[g], lw=1, ls=":", alpha=0.65,
                   label=f"Baseline ({int(baselines[g])})")
        ax.set_title(f"Good {g}", fontweight="bold", color=GCOL[g], fontsize=12)
        ax.set_xlabel("Round")
        ax.set_xticks(PROBE_ROUNDS)
        ax.set_xlim(-0.5, 10.5)
        if gi == 0:
            ax.set_ylabel("Stated valuation (1–10)")
    axes[1].legend(loc="lower center", fontsize=8, ncol=2,
                   bbox_to_anchor=(0.5, -0.42))
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — generate all figures for each agent type
# ══════════════════════════════════════════════════════════════════════════════

for atype, cfg in AGENT_TYPES.items():
    ids    = cfg["ids"]
    label  = cfg["label"]
    short  = cfg["short"]
    br     = cfg["baseline_ratings"]
    bb     = cfg["baseline_bundle"]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Pre-compute per-condition series
    ratings_series = {lbl: agent_probe_series(ALL_RUNS[lbl], ids, "ratings")
                      for lbl in COND_LABELS}
    bundle_series  = {lbl: agent_probe_series(ALL_RUNS[lbl], ids, "desired_bundle")
                      for lbl in COND_LABELS}
    net_series     = {lbl: net_acquisition_series(ALL_RUNS[lbl], ids)
                      for lbl in COND_LABELS}
    acc_records    = {lbl: acceptance_data(ALL_RUNS[lbl], ids) for lbl in COND_LABELS}
    proposals      = {lbl: proposal_targets(ALL_RUNS[lbl], ids) for lbl in COND_LABELS}
    wtp_all        = {lbl: wtp_data(ALL_RUNS[lbl], ids)         for lbl in COND_LABELS}

    # Fig 1 — ratings over rounds
    fig_probe(
        ratings_series, br,
        ylabel   = "Stated valuation (1–10)",
        suptitle = f"{label}: Probed Good Valuations Over Rounds\n"
                   f"(mean ± 95% CI across 10 runs; dotted = type baseline)",
        ylim     = (0, 11),
        yticks   = [0, 2, 4, 6, 8, 10],
        save_path = OUT_DIR / f"fig1_ratings_{short}.png",
    )

    # Fig 2 — desired bundle over rounds
    fig_probe(
        bundle_series, bb,
        ylabel   = "Desired units (out of 6)",
        suptitle = f"{label}: Desired Bundle Composition Over Rounds\n"
                   f"(mean ± 95% CI across 10 runs; dotted = type baseline)",
        ylim     = (-0.3, 6.3),
        yticks   = [0, 1, 2, 3, 4, 5, 6],
        save_path = OUT_DIR / f"fig2_bundle_{short}.png",
    )

    # Fig 3 — cumulative net acquisition
    fig_net(net_series, label, OUT_DIR / f"fig3_net_acquisition_{short}.png")

    # Fig 4 — trade behaviour
    fig_trade_behavior(
        acc_records, proposals, wtp_all,
        agent_label = label,
        save_path   = OUT_DIR / f"fig4_trade_behavior_{short}.png",
    )

    # Fig 5 — broadcast vs no-broadcast (pooled wording)
    bc_r   = pooled_probe_series(BC_LABELS,   ids, "ratings")
    nobc_r = pooled_probe_series(NOBC_LABELS, ids, "ratings")
    fig_broadcast_compare(bc_r, nobc_r, br, label,
                          OUT_DIR / f"fig5_broadcast_compare_{short}.png")

    # Fig 6 — wording comparison (broadcast conditions only)
    ext_r = agent_probe_series(ALL_RUNS[EXT_BC], ids, "ratings")
    wa_r  = agent_probe_series(ALL_RUNS[WA_BC],  ids, "ratings")
    fig_wording_compare(ext_r, wa_r, br, label,
                        OUT_DIR / f"fig6_wording_compare_{short}.png")

# ══════════════════════════════════════════════════════════════════════════════
# NUMERIC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("NUMERIC SUMMARY")
print(f"{'='*65}")

for atype, cfg in AGENT_TYPES.items():
    ids   = cfg["ids"]
    label = cfg["label"]
    br    = cfg["baseline_ratings"]
    print(f"\n{'─'*65}")
    print(f"{label}")
    print(f"{'─'*65}")
    for lbl in COND_LABELS:
        short_c = lbl.replace("\n", " ")
        runs    = ALL_RUNS[lbl]
        series  = agent_probe_series(runs, ids, "ratings")
        recs    = acceptance_data(runs, ids)
        tgts    = proposal_targets(runs, ids)
        total   = sum(tgts.values()) or 1
        print(f"\n  {short_c}  ({len(runs)} runs)")
        for r_lbl, r_idx in [("pre-run r=0", 0), ("round-10 r=10", 10)]:
            if r_idx in series:
                mn = np.nanmean(series[r_idx], axis=0)
                drift = [mn[i] - br[g] for i, g in enumerate(GOODS)]
                print(f"    Ratings @ {r_lbl}: "
                      + "  ".join(f"{g}={mn[i]:.2f}(Δ{drift[i]:+.2f})"
                                  for i, g in enumerate(GOODS)))
        for g in GOODS:
            p, lo, hi = acc_prob_ci(recs, g)
            n = sum(1 for r in recs if r["good_acquired"] == g)
            if not np.isnan(p):
                print(f"    P(accept|acq {g}): {p:.1%}  CI[{lo:.1%}–{hi:.1%}]  N={n}")
        for g in GOODS:
            pct = tgts.get(g, 0) / total * 100
            print(f"    Proposals seeking {g}: {pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — WAVE 2 (runs_7-7: control + scarcity/bandwagon/expert framings)
# ══════════════════════════════════════════════════════════════════════════════

for atype, cfg in AGENT_TYPES.items():
    ids    = cfg["ids"]
    label  = cfg["label"]
    short  = cfg["short"]
    br     = cfg["baseline_ratings"]
    bb     = cfg["baseline_bundle"]

    print(f"\n{'='*60}")
    print(f"  {label}  [wave 2 / runs_7-7]")
    print(f"{'='*60}")

    ratings_series2 = {lbl: agent_probe_series(ALL_RUNS2[lbl], ids, "ratings")
                       for lbl in COND_LABELS2}
    bundle_series2  = {lbl: agent_probe_series(ALL_RUNS2[lbl], ids, "desired_bundle")
                       for lbl in COND_LABELS2}
    net_series2     = {lbl: net_acquisition_series(ALL_RUNS2[lbl], ids)
                       for lbl in COND_LABELS2}
    acc_records2    = {lbl: acceptance_data(ALL_RUNS2[lbl], ids) for lbl in COND_LABELS2}
    proposals2      = {lbl: proposal_targets(ALL_RUNS2[lbl], ids) for lbl in COND_LABELS2}
    wtp_all2        = {lbl: wtp_data(ALL_RUNS2[lbl], ids)         for lbl in COND_LABELS2}

    # Fig 1 — ratings over rounds
    fig_probe(
        ratings_series2, br,
        ylabel   = "Stated valuation (1–10)",
        suptitle = f"{label}: Probed Good Valuations Over Rounds — Wave 2 (runs_7-7)\n"
                   f"(mean ± 95% CI across 10 runs; dotted = type baseline)",
        ylim     = (0, 11),
        yticks   = [0, 2, 4, 6, 8, 10],
        save_path = OUT_DIR / f"fig1_ratings_{short}_0707.png",
    )

    # Fig 2 — desired bundle over rounds
    fig_probe(
        bundle_series2, bb,
        ylabel   = "Desired units (out of 6)",
        suptitle = f"{label}: Desired Bundle Composition Over Rounds — Wave 2 (runs_7-7)\n"
                   f"(mean ± 95% CI across 10 runs; dotted = type baseline)",
        ylim     = (-0.3, 6.3),
        yticks   = [0, 1, 2, 3, 4, 5, 6],
        save_path = OUT_DIR / f"fig2_bundle_{short}_0707.png",
    )

    # Fig 3 — cumulative net acquisition
    fig_net(net_series2, f"{label} [Wave 2]", OUT_DIR / f"fig3_net_acquisition_{short}_0707.png")

    # Fig 4 — trade behaviour (control vs pooled treatment on panel E)
    fig_trade_behavior(
        acc_records2, proposals2, wtp_all2,
        agent_label   = f"{label} [Wave 2]",
        save_path     = OUT_DIR / f"fig4_trade_behavior_{short}_0707.png",
        cond_labels   = COND_LABELS2,
        short_labels  = SHORT_LABELS2,
        ccol          = CCOL2,
        panel_e_groups = [
            ("Control",             [CONTROL_LABEL2],    "#455A64"),
            ("Treatment (pooled)",  TREATMENT_LABELS2,    "#B71C1C"),
        ],
        panel_e_title = "E  Acceptance rate over rounds\n(control vs pooled B-favorable framings)",
    )

    # Fig 5 — condition comparison across all 4 wave-2 conditions
    fig_condition_compare(
        ratings_series2, br, f"{label} [Wave 2]",
        OUT_DIR / f"fig5_condition_compare_{short}_0707.png",
        cond_labels = COND_LABELS2, colors = CCOL2,
        suptitle = f"{label}: Control vs Scarcity/Bandwagon/Expert Framing — Probed Ratings\n"
                   f"(mean ± 95% CI across 10 runs per condition)",
    )

# ══════════════════════════════════════════════════════════════════════════════
# NUMERIC SUMMARY — WAVE 2
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("NUMERIC SUMMARY — WAVE 2 (runs_7-7)")
print(f"{'='*65}")

for atype, cfg in AGENT_TYPES.items():
    ids   = cfg["ids"]
    label = cfg["label"]
    br    = cfg["baseline_ratings"]
    print(f"\n{'─'*65}")
    print(f"{label}")
    print(f"{'─'*65}")
    for lbl in COND_LABELS2:
        short_c = lbl.replace("\n", " ")
        runs    = ALL_RUNS2[lbl]
        series  = agent_probe_series(runs, ids, "ratings")
        recs    = acceptance_data(runs, ids)
        tgts    = proposal_targets(runs, ids)
        total   = sum(tgts.values()) or 1
        print(f"\n  {short_c}  ({len(runs)} runs)")
        for r_lbl, r_idx in [("pre-run r=0", 0), ("round-10 r=10", 10)]:
            if r_idx in series:
                mn = np.nanmean(series[r_idx], axis=0)
                drift = [mn[i] - br[g] for i, g in enumerate(GOODS)]
                print(f"    Ratings @ {r_lbl}: "
                      + "  ".join(f"{g}={mn[i]:.2f}(Δ{drift[i]:+.2f})"
                                  for i, g in enumerate(GOODS)))
        for g in GOODS:
            p, lo, hi = acc_prob_ci(recs, g)
            n = sum(1 for r in recs if r["good_acquired"] == g)
            if not np.isnan(p):
                print(f"    P(accept|acq {g}): {p:.1%}  CI[{lo:.1%}–{hi:.1%}]  N={n}")
        for g in GOODS:
            pct = tgts.get(g, 0) / total * 100
            print(f"    Proposals seeking {g}: {pct:.1f}%")

print(f"\nAll figures saved to: {OUT_DIR}")
