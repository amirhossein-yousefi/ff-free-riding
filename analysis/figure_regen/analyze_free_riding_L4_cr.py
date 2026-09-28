#!/usr/bin/env python3
"""
Goodness-decomposition and depth-truncation figures for the 4-block (L4/D128) hardness-gated
ablation of Forward-Forward networks. Reads JSONL run logs; no GPU required.

[CR] Camera-ready copy of the project's analyze_free_riding.py, reduced to the code behind
Figs. 6 and 11 of the camera-ready paper (fig5_goodness_decomposition.pdf ->
fig_goodness_decomposition.pdf, fig4_depth_truncation.pdf -> fig_depth_truncation.pdf);
presentation-only changes are marked [CR2], and the data read and plotted are unchanged.
See analysis/figure_regen/README.md.

Inputs (read-only), in the latest run folder of each configuration in EXPERIMENT_ORDER:
    epoch_metrics.jsonl   all nine runs (Fig. 11)
    layer_metrics.jsonl   the four runs in FIG6_RUNS (Fig. 6)

Usage:
    FF_HOME=... python analyze_free_riding_L4_cr.py --output-dir OUT_DIR
    (--runs-dir defaults to the gated-ablation run folder under FF_BASE_DIR, whose default is the
    CIFAR-10 project folder inside the run archive FF_HOME)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# [CR] run archive location of the hardness-gated ablation (read-only): FF_BASE_DIR, else the
# CIFAR-10 project folder inside the run archive FF_HOME
_BASE = Path(os.environ.get("FF_BASE_DIR", Path(os.environ.get("FF_HOME", "")) / "amir-porjects" / "cifar_10_fair"
                            / "making_v1_v4_fair_comparison"))
_GATED = _BASE / "basic_fair_revised_valfix_cp_fair" / "from_colab" / "basic_fair_revised_valfix_cp_fair_hardness_gated"

# ──────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,  # [CR2] embedded TrueType, no Type 3 fonts
    "ps.fonttype": 42,
})

# ──────────────────────────────────────────────────────────────────────
# Experiment registry: short_name → (display_label, color, linestyle)
# ──────────────────────────────────────────────────────────────────────
EXPERIMENT_ORDER = [
    "baseline_constant",
    "adaptive_t1_k0",
    "adaptive_t3_k0",
    "adaptive_t1_k2",
    "adaptive_t1_k4",
    "adaptive_prev_only",
    "adaptive_prev_t1_k2",
    "lcff_prefix",
    "no_all_collab",
]

EXPERIMENT_META = {
    "baseline_constant":    (r"Constant $\gamma$=0.7 (baseline)", "#d62728", "-"),
    "adaptive_t1_k0":      (r"Adaptive $\tau$=1, $\kappa$=0",     "#2ca02c", "-"),
    "adaptive_t3_k0":      (r"Adaptive $\tau$=3, $\kappa$=0",     "#1f77b4", "-"),
    "adaptive_t1_k2":      (r"Adaptive $\tau$=1, $\kappa$=2",     "#ff7f0e", "--"),
    "adaptive_t1_k4":      (r"Adaptive $\tau$=1, $\kappa$=4",     "#9467bd", "--"),
    "adaptive_prev_only":  (r"Adaptive prev-only",                 "#8c564b", "-."),
    "adaptive_prev_t1_k2": (r"Adaptive prev $\tau$=1, $\kappa$=2","#e377c2", "-."),
    "lcff_prefix":         (r"LCFF prefix",                        "#7f7f7f", ":"),
    "no_all_collab":       (r"No collaboration",                   "#17becf", "-"),
}

# Runs split into own and inherited goodness in Fig. 6 (panel order)
FIG6_RUNS = ["baseline_constant", "adaptive_t1_k0", "no_all_collab", "lcff_prefix"]


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def find_run_dirs(runs_root: str) -> dict:
    """Map short experiment names to their run directories."""
    runs_root = Path(runs_root)
    mapping = {}

    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        for short_name in EXPERIMENT_ORDER:
            tag = f"ff_cifar10_{short_name}_L4"
            if tag in name:
                # If multiple runs, take the latest (by timestamp in dir name)
                if short_name not in mapping or name > mapping[short_name].name:
                    mapping[short_name] = d
                break

    return mapping


def load_layer_metrics(run_dir: Path) -> dict:
    """Load layer_metrics.jsonl → {(stage, epoch, layer): dict}."""
    path = run_dir / "layer_metrics.jsonl"
    data = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            key = (d["stage"], d["epoch"], d["layer"])
            data[key] = d
    return data


def load_epoch_metrics(run_dir: Path) -> list:
    """Load epoch_metrics.jsonl → list of dicts."""
    path = run_dir / "epoch_metrics.jsonl"
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_final_layer_metrics(layer_data: dict, stage: int = 1, num_layers: int = 4):
    """Get metrics from the last epoch of the given stage."""
    max_epoch = max(ep for (s, ep, l) in layer_data if s == stage)
    result = {}
    for l in range(num_layers):
        key = (stage, max_epoch, l)
        if key in layer_data:
            result[l] = layer_data[key]
    return result


def get_best_epoch_metrics(epoch_data: list, stage: int = 1):
    """Get epoch metrics with best eval_top1 for stage."""
    best = None
    for d in epoch_data:
        if d.get("stage") != stage:
            continue
        if d.get("eval_top1") is None:
            continue
        if best is None or d["eval_top1"] > best["eval_top1"]:
            best = d
    return best


# ──────────────────────────────────────────────────────────────────────
# Figure 4 = paper Fig. 11 (fig:depth_truncation): Depth-Truncation Analysis
# ──────────────────────────────────────────────────────────────────────
def plot_figure4(all_data: dict, output_dir: Path):
    """
    Grouped bar chart: eval_top1_d1 through d4 for each condition.
    Shows incremental accuracy gain from adding layers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))  # [CR2] larger type at print size

    experiments = [e for e in EXPERIMENT_ORDER if e in all_data]

    # ── Panel (a): Depth-truncation accuracy ──
    ax = axes[0]
    n_exp = len(experiments)
    bar_width = 0.8 / max(n_exp, 1)

    for i, exp in enumerate(experiments):
        epoch_data = all_data[exp]["epoch"]
        best = get_best_epoch_metrics(epoch_data, stage=1)
        if best is None:
            continue

        accs = []
        for d in range(1, 5):
            key = f"eval_top1_d{d}"
            v = best.get(key)
            accs.append(v * 100 if v is not None else 0)

        positions = [d + i * bar_width for d in range(4)]
        color = EXPERIMENT_META[exp][1]
        ax.bar(positions, accs, width=bar_width, color=color, alpha=0.8,
               label=EXPERIMENT_META[exp][0])

    center = bar_width * (n_exp - 1) / 2
    ax.set_xticks([d + center for d in range(4)])
    ax.set_xticklabels(["d=1", "d=2", "d=3", "d=4"])
    ax.set_xlabel("Depth (number of blocks used)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("(a) Depth-Truncation Accuracy", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)

    # Zoom y-axis for readability
    all_vals = []
    for exp in experiments:
        best = get_best_epoch_metrics(all_data[exp]["epoch"], stage=1)
        if best:
            for d in range(1, 5):
                v = best.get(f"eval_top1_d{d}")
                if v is not None:
                    all_vals.append(v * 100)
    if all_vals:
        ax.set_ylim(min(all_vals) - 0.5, max(all_vals) + 0.5)

    # ── Panel (b): Incremental gain per depth (relative to d=1) ──
    ax = axes[1]
    for i, exp in enumerate(experiments):
        epoch_data = all_data[exp]["epoch"]
        best = get_best_epoch_metrics(epoch_data, stage=1)
        if best is None:
            continue

        accs = []
        for d in range(1, 5):
            v = best.get(f"eval_top1_d{d}")
            accs.append(v * 100 if v is not None else 0)

        # Incremental gain relative to d=1 baseline
        gains = [accs[d] - accs[0] for d in range(len(accs))]

        label, color, ls = EXPERIMENT_META[exp]
        ax.plot([1, 2, 3, 4], gains, marker="o", color=color, linestyle=ls,
                linewidth=1.5, markersize=5, label=label, alpha=0.85)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Accuracy gain over d=1 (%)")
    ax.set_title("(b) Cumulative gain from adding blocks (vs. d=1)", fontweight="bold")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["d=1\n(baseline)", "d=2\n(+Block 1)", "d=3\n(+Block 2)", "d=4\n(+Block 3)"])
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    # [CR2] one shared legend below both panels (panel legends covered the gamma=0.7 line and bars);
    # handles from panel (b) carry both color and line style
    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False, fontsize=9)
    _save_fig(fig, output_dir / "fig4_depth_truncation")
    print("  Figure 4 saved.")


# ──────────────────────────────────────────────────────────────────────
# Figure 5 = paper Fig. 6 (fig:goodness_decomposition): goodness decomposition
# ──────────────────────────────────────────────────────────────────────
def plot_figure5(all_data: dict, output_dir: Path):
    """
    Stacked bar chart showing how total goodness (gpos) decomposes into
    current-block contribution (gpos_cur) vs inherited (gamma_pos * scale).
    """
    key_exps = [e for e in FIG6_RUNS if e in all_data]

    if len(key_exps) < 2:
        return

    # [CR2] smaller canvas (larger type at print size), paper names in the panel titles
    fig, axes = plt.subplots(1, len(key_exps), figsize=(2.1 * len(key_exps), 2.6), sharey=True)
    CR_TITLES = {"baseline_constant": r"Constant $\gamma$=0.7",
                 "adaptive_t1_k0": r"Adaptive $\tau$=1, $\kappa$=0",
                 "no_all_collab": r"$\gamma$=0 (history-free)",
                 "lcff_prefix": r"LCFF ($\gamma$=1)"}

    for ax_idx, exp in enumerate(key_exps):
        ax = axes[ax_idx]
        final_layer = get_final_layer_metrics(all_data[exp]["layer"])
        label = CR_TITLES[exp]

        gpos_cur = [final_layer[l].get("gpos_cur", 0) for l in range(4)]
        gpos_total = [final_layer[l].get("gpos", 0) for l in range(4)]
        inherited = [gpos_total[l] - gpos_cur[l] for l in range(4)]

        x = np.arange(4)
        ax.bar(x, gpos_cur, label=r"Own ($g^{+}_{\mathrm{cur}}$)", color="#2ca02c", alpha=0.85)
        ax.bar(x, [max(0, v) for v in inherited], bottom=gpos_cur,
               label="Inherited (history term)", color="#d62728", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in range(4)])
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Block")
        if ax_idx == 0:
            ax.set_ylabel(r"$g^{+}$ (own + inherited)")

        # Add percentage labels (own share)
        for l in range(4):
            total = gpos_total[l]
            if total > 0.01:
                pct = gpos_cur[l] / total * 100
                ax.text(l, total + 0.05, f"{pct:.0f}%", ha="center", fontsize=8)

        ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, fontsize=10)
    _save_fig(fig, output_dir / "fig5_goodness_decomposition")
    print("  Figure 5 (bonus) saved.")


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────
def _save_fig(fig, path_stem: Path):
    """Save figure as both PDF and PNG."""
    fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path_stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Free-riding analysis for FF networks with hardness-gated collaboration"
    )
    parser.add_argument(
        "--runs-dir",
        default=str(_GATED / "ff_gated_ablation" / "runs"),  # [CR] from FF_BASE_DIR / FF_HOME (README.md)
        help="Directory containing experiment run folders",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("FF_FIG_DIR", "figures"),  # [CR]
        help="Output directory for the figures",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Runs directory: {runs_dir}")
    print(f"Output directory: {output_dir}")

    # Discover experiments
    run_dirs = find_run_dirs(runs_dir)
    print(f"\nFound {len(run_dirs)} experiments:")
    for name, d in sorted(run_dirs.items()):
        print(f"  {name}: {d.name}")

    if not run_dirs:
        print("\nERROR: No experiment runs found. Check --runs-dir path.")
        sys.exit(1)

    # Load the logs the two figures use: epoch metrics of every run (Fig. 11),
    # layer metrics of the four Fig. 6 runs
    print("\nLoading metrics...")
    all_data = {}
    for exp_name, run_dir in run_dirs.items():
        try:
            entry = {"epoch": load_epoch_metrics(run_dir)}
            if exp_name in FIG6_RUNS:
                entry["layer"] = load_layer_metrics(run_dir)
            all_data[exp_name] = entry
            layer_str = f", {len(entry['layer'])} layer records" if "layer" in entry else ""
            print(f"  {exp_name}: {len(entry['epoch'])} epoch records{layer_str}")
        except Exception as e:
            print(f"  WARNING: Failed to load {exp_name}: {e}")

    print("\nGenerating figures...")
    plot_figure4(all_data, output_dir)  # paper Fig. 11
    plot_figure5(all_data, output_dir)  # paper Fig. 6


if __name__ == "__main__":
    main()
