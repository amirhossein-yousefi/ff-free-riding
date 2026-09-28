#!/usr/bin/env python3
"""
Goodness-decomposition and training-dynamics figures for the 8-block (L8/D128) hardness-gated
ablation of Forward-Forward networks. Reads JSONL run logs; no GPU required.

[CR] Camera-ready copy of the project's analyze_free_riding_L8.py, reduced to the code behind
Figs. 8 and 9 of the camera-ready paper (fig5_goodness_decomposition.pdf ->
fig_goodness_decomposition_L8.pdf, fig3_training_dynamics.pdf -> fig_training_dynamics_L8.pdf);
presentation-only changes are marked [CR2], and the data read and plotted are unchanged.
See analysis/figure_regen/README.md.

Inputs (read-only): layer_metrics.jsonl in the latest run folder of each of the four adaptive runs
in EXPERIMENT_LABELS (all four in Fig. 8, the two cumulative kappa=0 runs in Fig. 9).

Usage:
    FF_HOME=... python analyze_free_riding_L8_cr.py --output-dir OUT_DIR
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
import matplotlib.ticker as mticker
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
# Experiment registry: short_name → panel title
# ──────────────────────────────────────────────────────────────────────
NUM_LAYERS = 8  # 8-block experiments
DEEPEST = NUM_LAYERS - 1  # L7

# The four adaptive runs of Fig. 8, in panel order
EXPERIMENT_LABELS = {
    "adaptive_t1_k0_L8":      r"Adaptive $\tau$=1, $\kappa$=0",
    "adaptive_t3_k0_L8":      r"Adaptive $\tau$=3, $\kappa$=0",
    "adaptive_prev_only_L8":  r"Adaptive prev-only",
    "adaptive_prev_t1_k2_L8": r"Adaptive prev $\tau$=1, $\kappa$=2",
}
# The two cumulative kappa=0 runs of Fig. 9, in panel order
FIG9_RUNS = ["adaptive_t1_k0_L8", "adaptive_t3_k0_L8"]


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

        for short_name in EXPERIMENT_LABELS:
            tag = f"ff_cifar10_{short_name}_L{NUM_LAYERS}"
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


def extract_layer_trajectory(layer_data: dict, metric: str, stage: int = 1, num_layers: int = NUM_LAYERS):
    """Extract per-epoch, per-layer trajectory for a given metric.

    Returns: epochs (sorted list), values (dict layer → list of values)
    """
    epochs_set = set()
    for (s, ep, l) in layer_data:
        if s == stage:
            epochs_set.add(ep)

    epochs = sorted(epochs_set)
    values = {l: [] for l in range(num_layers)}

    for ep in epochs:
        for l in range(num_layers):
            key = (stage, ep, l)
            if key in layer_data:
                values[l].append(layer_data[key].get(metric, float("nan")))
            else:
                values[l].append(float("nan"))

    return epochs, values


def get_final_layer_metrics(layer_data: dict, stage: int = 1, num_layers: int = NUM_LAYERS):
    """Get metrics from the last epoch of the given stage."""
    max_epoch = max(ep for (s, ep, l) in layer_data if s == stage)
    result = {}
    for l in range(num_layers):
        key = (stage, max_epoch, l)
        if key in layer_data:
            result[l] = layer_data[key]
    return result


# ──────────────────────────────────────────────────────────────────────
# Figure 3 = paper Fig. 9 (fig:training_dynamics_L8): training dynamics of block goodness
# ──────────────────────────────────────────────────────────────────────
def plot_figure3(all_data: dict, output_dir: Path):
    """
    One panel per run in FIG9_RUNS: gpos_cur of Blocks 0, 1, 3, 5 and 7 over Stage-1 training.
    """
    key_exps = [e for e in FIG9_RUNS if e in all_data]

    n_plots = len(key_exps)
    ncols = min(n_plots, 2)
    nrows = (n_plots + ncols - 1) // ncols
    # [CR2] smaller canvas (larger type at print size)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 3.0 * nrows), sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Show representative layers for 8-block: L0, L1, L3, L5, L7
    show_layers = [0, 1, 3, 5, DEEPEST]
    layer_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(show_layers)))

    for ax_idx, exp in enumerate(key_exps):
        ax = axes_flat[ax_idx]
        label = EXPERIMENT_LABELS[exp]
        layer_data = all_data[exp]["layer"]
        epochs, values = extract_layer_trajectory(layer_data, "gpos_cur")

        for i, l in enumerate(show_layers):
            if l in values:
                ax.plot(epochs, values[l], color=layer_colors[i], linewidth=1.2,
                        label=f"Block {l}", alpha=0.85)

        ax.set_title(label)
        ax.grid(True, alpha=0.2)
        # [CR2] log axis so every block stays on the chart (the linear 0-3.5 axis cut off
        # Blocks 5 and 7, which reach ~14); x label on every panel
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 5, 10])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(0.5, 16)
        ax.set_xlabel("Epoch")
        if ax_idx % ncols == 0:
            ax.set_ylabel(r"$g^{+}_{\mathrm{cur}}$ (log scale)")

    # Hide unused subplots
    for i in range(len(key_exps), nrows * ncols):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               frameon=False, fontsize=9)
    _save_fig(fig, output_dir / "fig3_training_dynamics")
    print("  Figure 3 saved.")


# ──────────────────────────────────────────────────────────────────────
# Figure 5 = paper Fig. 8 (fig:goodness_decomposition_L8): goodness decomposition
# ──────────────────────────────────────────────────────────────────────
def plot_figure5(all_data: dict, output_dir: Path):
    """
    Stacked bar chart showing how total goodness (gpos) decomposes into
    current-block contribution (gpos_cur) vs inherited (gamma_pos * scale).
    """
    key_exps = [e for e in EXPERIMENT_LABELS if e in all_data]

    if len(key_exps) < 2:
        return

    fig, axes = plt.subplots(1, len(key_exps), figsize=(2.6 * len(key_exps), 2.8), sharey=True)  # [CR2]

    for ax_idx, exp in enumerate(key_exps):
        ax = axes[ax_idx]
        final_layer = get_final_layer_metrics(all_data[exp]["layer"])
        label = EXPERIMENT_LABELS[exp]

        gpos_cur = [final_layer.get(l, {}).get("gpos_cur", 0) for l in range(NUM_LAYERS)]
        gpos_total = [final_layer.get(l, {}).get("gpos", 0) for l in range(NUM_LAYERS)]
        inherited = [gpos_total[l] - gpos_cur[l] for l in range(NUM_LAYERS)]

        x = np.arange(NUM_LAYERS)
        # [CR2] same legend names as Fig. 6 (the gamma*M_{l-1} notation is not used in the paper)
        ax.bar(x, gpos_cur, label=r"Own ($g^{+}_{\mathrm{cur}}$)", color="#2ca02c", alpha=0.85)
        ax.bar(x, [max(0, v) for v in inherited], bottom=gpos_cur,
               label="Inherited (history term)", color="#d62728", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in range(NUM_LAYERS)], fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Block")
        if ax_idx == 0:
            ax.set_ylabel(r"$g^{+}$ (own + inherited)")

        # Add percentage labels (own share)
        for l in range(NUM_LAYERS):
            total = gpos_total[l]
            if total > 0.01:
                pct = gpos_cur[l] / total * 100
                ax.text(l, total + 0.05, f"{pct:.0f}%", ha="center", fontsize=7)

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
        default=str(_GATED / "ff_gated_ablation_L8" / "runs"),  # [CR] from FF_BASE_DIR / FF_HOME (README.md)
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

    # Load the logs the two figures use (layer metrics only)
    print("\nLoading metrics...")
    all_data = {}
    for exp_name, run_dir in run_dirs.items():
        try:
            layer_data = load_layer_metrics(run_dir)
            all_data[exp_name] = {"layer": layer_data}
            print(f"  {exp_name}: {len(layer_data)} layer records")
        except Exception as e:
            print(f"  WARNING: Failed to load {exp_name}: {e}")

    print("\nGenerating figures...")
    plot_figure3(all_data, output_dir)  # paper Fig. 9
    plot_figure5(all_data, output_dir)  # paper Fig. 8


if __name__ == "__main__":
    main()
