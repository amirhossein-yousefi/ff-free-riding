#!/usr/bin/env python3
"""
Generate dynamic training metric figures for the NeurIPS paper.

Produces 4 figures:
  1. Free-riding onset curve (sep_curr_nl per layer over epochs)
  2. Convergence speed (bar chart: epochs to 95%/99% of final accuracy)
  3. Depth contribution gap (eval_top1_d4 - eval_top1_d2 over epochs)
  4. Training stability (rolling std of eval_top1)

[CR] Camera-ready copy of analysis/scripts/plot_dynamics.py (the v1 supplement script) with
presentation-only changes (marked [CR]/[CR2]); the data read and plotted are unchanged. It renders
Figs. 5, 12, 15 and 16 of the camera-ready paper (fig8_freeriding_onset, fig9_convergence_speed,
fig10_depth_gap, fig11_training_stability); see analysis/figure_regen/README.md.

Usage:  FF_HOME=... FF_FIG_DIR=OUT_DIR python plot_dynamics_cr.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
# [CR] Paths come from the environment (analysis/README.md): FF_BASE_DIR, else the CIFAR-10 project
# folder inside the run archive FF_HOME; figures go to FF_FIG_DIR (default ./figures).
BASE = Path(os.environ.get('FF_BASE_DIR', Path(os.environ.get('FF_HOME', '')) / 'amir-porjects' / 'cifar_10_fair'
                           / 'making_v1_v4_fair_comparison'))
FIG_DIR = Path(os.environ.get('FF_FIG_DIR', 'figures'))

VARIANTS = {
    'CP-FAIR': {
        'layer_metrics': [
            BASE / 'basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260218_144731/layer_metrics.jsonl',
            BASE / 'basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035/layer_metrics.jsonl',
        ],
        'epoch_metrics': [
            BASE / 'basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260218_144731/epoch_metrics.jsonl',
            BASE / 'basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035/epoch_metrics.jsonl',
        ],
        'color': '#2171b5',
        'marker': 'o',
    },
    '$\\gamma{=}0$ (history-free)': {
        'layer_metrics': [
            BASE / 'ablations/ff_ablation/no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260311_120026/layer_metrics.jsonl',
            BASE / 'ablations/no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260312_132043/layer_metrics.jsonl',
        ],
        'epoch_metrics': [
            BASE / 'ablations/ff_ablation/no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260311_120026/epoch_metrics.jsonl',
            BASE / 'ablations/no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260312_132043/epoch_metrics.jsonl',
        ],
        'color': '#238b45',
        'marker': 's',
    },
    'LCFF-prefix ($\\gamma{=}1$)': {
        'layer_metrics': [
            BASE / 'ablations/ff_ablation/lcff_prefix/runs/ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260311_120438/layer_metrics.jsonl',
            BASE / 'ablations/lcff_prefix/runs/ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260312_132019/layer_metrics.jsonl',
        ],
        'epoch_metrics': [
            BASE / 'ablations/ff_ablation/lcff_prefix/runs/ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260311_120438/epoch_metrics.jsonl',
            BASE / 'ablations/lcff_prefix/runs/ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260312_132019/epoch_metrics.jsonl',
        ],
        'color': '#cb181d',
        'marker': '^',
    },
}

N_LAYERS = 4

# ---------- style ----------
plt.rcParams.update({
    # NeurIPS submission requires Type 1 or embedded TrueType fonts
    # (no Type 3 bitmap fonts). fonttype=42 forces matplotlib to embed
    # TrueType outlines in saved PDFs.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'dejavuserif',
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

LAYER_COLORS = ['#4292c6', '#08519c', '#fd8d3c', '#d94801']
LAYER_LABELS = ['Block 0', 'Block 1', 'Block 2', 'Block 3']


# ---------- data loading ----------
def load_jsonl_concat(paths):
    """Load and concatenate multiple JSONL files, dedup by (epoch, layer)."""
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f'WARNING: {p} not found, skipping')
            continue
        with open(p) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def load_layer_data(variant):
    """Return dict: {layer: {epoch: {metric: value}}}. Stage 1 only."""
    rows = load_jsonl_concat(variant['layer_metrics'])
    data = {l: {} for l in range(N_LAYERS)}
    for r in rows:
        if r.get('stage', 1) != 1:
            continue
        layer = r['layer']
        epoch = r['epoch']
        data[layer][epoch] = r
    return data


def load_epoch_data(variant):
    """Return dict: {epoch: {metric: value}}. Stage 1 only."""
    rows = load_jsonl_concat(variant['epoch_metrics'])
    data = {}
    for r in rows:
        if r.get('stage', 1) != 1:
            continue
        data[r['epoch']] = r
    return data


# ---------- Figure 1: Free-riding onset curve ----------
def plot_freeriding_onset():
    """sep_curr_nl per layer over epochs, one panel per variant."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

    for ax, (vname, vconf) in zip(axes, VARIANTS.items()):
        ldata = load_layer_data(vconf)
        for layer in range(N_LAYERS):
            epochs = sorted(ldata[layer].keys())
            vals = [ldata[layer][e]['sep_curr_nl'] for e in epochs]
            # smooth with rolling mean (window=5)
            if len(vals) > 5:
                vals_smooth = np.convolve(vals, np.ones(5)/5, mode='valid')
                epochs_smooth = epochs[2:-2]
            else:
                vals_smooth = vals
                epochs_smooth = epochs
            ax.plot(epochs_smooth, vals_smooth, color=LAYER_COLORS[layer],
                    label=LAYER_LABELS[layer], linewidth=1.2, alpha=0.85)

        ax.set_title(vname, fontsize=9)
        ax.set_xlabel('Epoch')
        if ax == axes[0]:
            ax.set_ylabel('$\\mathrm{sep}^{\\mathrm{cur}}_{\\mathrm{nl}}$')  # [CR2] symbol of Eq. 7
        ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    # [CR2] one-row legend above the panels (it covered CP-FAIR's Block-1 curve); the suptitle
    # repeated the caption and is dropped
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.98),
               frameon=False, fontsize=9)

    fig.savefig(FIG_DIR / 'fig8_freeriding_onset.pdf')
    fig.savefig(FIG_DIR / 'fig8_freeriding_onset.png')
    print('Saved fig8_freeriding_onset.pdf/png')
    plt.close(fig)


# ---------- Figure 2: Convergence speed ----------
def plot_convergence_speed():
    """Bar chart: epochs to reach 95% and 99% of final accuracy."""
    thresholds = [0.95, 0.99]
    results = {}

    for vname, vconf in VARIANTS.items():
        edata = load_epoch_data(vconf)
        eval_epochs = sorted([(e, d['eval_top1']) for e, d in edata.items()
                               if d.get('eval_top1') is not None], key=lambda x: x[0])
        if not eval_epochs:
            results[vname] = {t: None for t in thresholds}
            continue

        final_acc = eval_epochs[-1][1]
        results[vname] = {}
        for t in thresholds:
            target = t * final_acc
            reached = None
            for epoch, acc in eval_epochs:
                if acc >= target:
                    reached = epoch
                    break
            results[vname][t] = reached

    # Print table
    print('\n=== Convergence Speed ===')
    print(f'{"Variant":<35} {"Final Acc":>10} {"95% epoch":>10} {"99% epoch":>10}')
    for vname, vconf in VARIANTS.items():
        edata = load_epoch_data(vconf)
        eval_epochs = sorted([(e, d['eval_top1']) for e, d in edata.items()
                               if d.get('eval_top1') is not None], key=lambda x: x[0])
        final_acc = eval_epochs[-1][1] if eval_epochs else 0
        e95 = results[vname][0.95]
        e99 = results[vname][0.99]
        print(f'{vname:<35} {final_acc:>10.4f} {str(e95):>10} {str(e99):>10}')

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(VARIANTS))
    width = 0.35
    colors = [v['color'] for v in VARIANTS.values()]

    bars_95 = [results[v][0.95] or 0 for v in VARIANTS]
    bars_99 = [results[v][0.99] or 0 for v in VARIANTS]

    short_names = ['CP-FAIR\n($\\gamma{=}0.7$)', '$\\gamma{=}0$\n(history-free)', 'LCFF-prefix\n($\\gamma{=}1$)']  # [CR2] names of Figs. 15/16

    b1 = ax.bar(x - width/2, bars_95, width, label='95% of final', color=colors, alpha=0.4, edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + width/2, bars_99, width, label='99% of final', color=colors, alpha=1.0, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Epoch')
    ax.set_title('Convergence Speed')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylim(0, 165)  # [CR] headroom for the value labels
    # [CR2] neutral light/dark swatches (the bars are colored per variant; shade = threshold)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='0.72', edgecolor='black', linewidth=0.5, label='95% of final'),
                       Patch(facecolor='0.35', edgecolor='black', linewidth=0.5, label='99% of final')],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)  # [CR] legend outside the bars
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

    # Add value labels
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig9_convergence_speed.pdf')
    fig.savefig(FIG_DIR / 'fig9_convergence_speed.png')
    print('Saved fig9_convergence_speed.pdf/png')
    plt.close(fig)


# ---------- Figure 3: Depth contribution gap ----------
def plot_depth_gap():
    """acc(d=4) - acc(d=2) over epochs, one line per variant."""
    fig, ax = plt.subplots(figsize=(5, 3))

    for vname, vconf in VARIANTS.items():
        edata = load_epoch_data(vconf)
        eval_epochs = sorted([(e, d.get('eval_top1_d2'), d.get('eval_top1_d4'))
                               for e, d in edata.items()
                               if d.get('eval_top1_d2') is not None],
                              key=lambda x: x[0])
        if not eval_epochs:
            continue
        epochs = [e[0] for e in eval_epochs]
        gaps = [(e[2] - e[1]) * 100 for e in eval_epochs]  # percentage points
        ax.plot(epochs, gaps, color=vconf['color'], marker=vconf['marker'],
                markersize=4, linewidth=1.5, label=vname, alpha=0.85)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('acc(blocks 0–3) $-$ acc(0–1) (pp)')  # [CR2] block indices as in DS(d) of Eq. 7
    ax.set_title('Depth Contribution: 4-Block vs 2-Block Accuracy Gap')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig10_depth_gap.pdf')
    fig.savefig(FIG_DIR / 'fig10_depth_gap.png')
    print('Saved fig10_depth_gap.pdf/png')
    plt.close(fig)


# ---------- Figure 4: Training stability ----------
def plot_training_stability():
    """Rolling std of eval_top1, one line per variant.
    Since evals are sparse (~every 20 epochs), use all available eval points
    with a rolling window of 5 eval points."""
    fig, ax = plt.subplots(figsize=(5, 3))
    window = 5  # 5 eval points ≈ 100 epochs

    for vname, vconf in VARIANTS.items():
        edata = load_epoch_data(vconf)
        eval_epochs = sorted([(e, d['eval_top1']) for e, d in edata.items()
                               if d.get('eval_top1') is not None], key=lambda x: x[0])
        if len(eval_epochs) < window:
            continue
        epochs = [e[0] for e in eval_epochs]
        accs = np.array([e[1] for e in eval_epochs])

        # Rolling std
        stds = []
        epoch_centers = []
        for i in range(len(accs) - window + 1):
            stds.append(np.std(accs[i:i+window]) * 100)  # percentage points
            epoch_centers.append(epochs[i + window // 2])

        ax.plot(epoch_centers, stds, color=vconf['color'], marker=vconf['marker'],
                markersize=4, linewidth=1.5, label=vname, alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rolling std of val accuracy (pp)')
    ax.set_title(f'Training Stability (rolling SD over {window} consecutive evaluations)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig11_training_stability.pdf')
    fig.savefig(FIG_DIR / 'fig11_training_stability.png')
    print('Saved fig11_training_stability.pdf/png')
    plt.close(fig)


# ---------- main ----------
if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_convergence_speed()
    plot_freeriding_onset()
    plot_depth_gap()
    plot_training_stability()  # [CR] Fig. 16 (unchanged plotting code; population SD, ddof=0)
    print('\nAll figures generated.')
