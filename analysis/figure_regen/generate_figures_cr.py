#!/usr/bin/env python3
"""Generate all figures for the NeurIPS paper on Forward-Forward free-riding.

[CR] Camera-ready copy of analysis/generate_figures.py (the v1 supplement script) with presentation-only
changes (marked [CR]/[CR2]); the data read and plotted are unchanged. It renders Figs. 4, 7, 13 and 14 of
the camera-ready paper (fig2_freeriding_separation, fig_scaling_freeriding_L8D128, fig6_training_dynamics,
fig7_moe_routing); see analysis/figure_regen/README.md.

Usage:  FF_HOME=... python generate_figures_cr.py OUT_DIR
Paths:  FF_BASE_DIR (default $FF_HOME/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison) holds the
        variant folders named in RUN_REGISTRY; OUT_DIR (or FF_FIG_DIR, default ./figures) receives the PDFs.
"""

import json
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe

# ─── Constants ────────────────────────────────────────────────────────────────

import os as _os
# [CR] Paths come from the environment (analysis/README.md): FF_BASE_DIR, else the CIFAR-10 project
# folder inside the run archive FF_HOME.
BASE_DIR = _os.environ.get(
    "FF_BASE_DIR",
    _os.path.join(_os.environ.get("FF_HOME", ""), "amir-porjects", "cifar_10_fair",
                  "making_v1_v4_fair_comparison"),
)
OUT_DIR = _os.environ.get("FF_FIG_DIR", os.path.join(os.getcwd(), "figures"))  # output directory for the figures

# Variant → run directory (relative to BASE_DIR/<variant>/runs/)
RUN_REGISTRY = {
    "cp_fair": (
        "basic_fair_revised_valfix_cp_fair",
        "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035",
    ),
    "baseline": (
        "basic_fair_revised_valfix",
        "ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260108_214956",
    ),
    "do_this_first": (
        "basic_fair_revised_valfix_do_this_first",
        "ff_cifar10_cifar10_opt_do_this_first_ynorm_nomem_resid_L4_D256_bs512_seed42_20260128_084102",
    ),
    "v1_fair": (
        "v1_fair_split",
        "ff_cifar10_v1_fair_valsplit_workersfix_depthmetrics_L4_D256_bs128_seed42_20260121_111852",
    ),
    "atlas_ff": (
        "atlas_ff",
        "ff_cifar10_cifar10_atlas_ff_triangle_multineg_div_probe_L4_D256_bs512_seed42_20260301_191325",
    ),
    "dp_fair": (
        "basic_fair_revised_valfix_dp_fair",
        "ff_cifar10_cifar10_dp_fair_progressive_depth_L4_D256_bs512_seed42_20260213_145420",
    ),
    "cp_fair_v2": (
        "basic_fair_revised_valfix_cp_fair_v2",
        "ff_cifar10_cifar10_cp_fair_v2_low_contrastive_L4_D256_bs512_seed42_20260223_165508",
    ),
    "depthfix": (
        "basic_fair_depthfix_valfix",
        "ff_cifar10_cifar10_depthfix_multineg_div_valsplit_workersfix_L4_D256_bs512_seed42_20260111_141853",
    ),
    "topk_depth": (
        "basic_fair_revised_valfix_topk_depthsched",
        "ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_topkNL_depthsched_L4_D256_bs512_seed42_20260208_001416",
    ),
    "freez_v1": (
        "freez_layer_1",
        "ff_cifar10_cifar10_opt_triangle_greedy_gaspectA_workersfix_L4_D256_bs512_seed42_20260115_231345",
    ),
    "freez_v2": (
        "freez_layer_1_v2",
        "ff_cifar10_cifar10_opt_triangle_greedy_gaspectA_workersfix_v2_L4_D256_bs512_seed42_20260117_144151",
    ),
    "v2_fair": (
        "v2_fair_valsplit_workersfix",
        "ff_cifar10_v2_fair_valsplit_workersfix_L4_D256_bs128_seed42_20260202_171622",
    ),
}

# Display labels
VARIANT_LABELS = {
    "cp_fair": "CP-FAIR",  # [CR] no "(ours)" in the camera-ready figures
    "baseline": "Baseline",
    "do_this_first": "NoMem-L2",
    "v1_fair": "Fixed-NL",
    "atlas_ff": "Multi-Tech",
    "dp_fair": "DP-FAIR",
    "cp_fair_v2": "CP-FAIR-LC",
    "depthfix": "Gamma-Ramp",
    "topk_depth": "TopK-Depth",
    "freez_v1": "Greedy-Freeze",
    "freez_v2": "Prefix-Freeze",
    "v2_fair": "Block0-Label",
}

# Colorblind-friendly palette
VARIANT_COLORS = {
    "cp_fair": "#D55E00",      # vermillion (our best)
    "baseline": "#0072B2",     # blue
    "do_this_first": "#009E73", # bluish green
    "v1_fair": "#CC79A7",      # reddish purple
    "atlas_ff": "#E69F00",     # orange
    "dp_fair": "#56B4E9",      # sky blue
    "cp_fair_v2": "#F0E442",   # yellow
    "depthfix": "#999999",     # gray
    "topk_depth": "#882255",   # wine
    "freez_v1": "#DDCC77",     # sand
    "freez_v2": "#AA4499",     # purple
    "v2_fair": "#44AA99",      # teal
}

# Test accuracy (TTA) for each variant - from Table 1
TEST_ACC = {
    "cp_fair": 91.36, "baseline": 91.20, "do_this_first": 91.31,
    "v1_fair": 91.15, "atlas_ff": 91.25, "dp_fair": 91.12,
    "cp_fair_v2": 91.32, "depthfix": 90.82, "topk_depth": 90.69,
    "freez_v1": 86.23, "freez_v2": 90.95, "v2_fair": 90.82,
}


# ─── NeurIPS Style ────────────────────────────────────────────────────────────

def setup_style():
    plt.rcParams.update({
        # NeurIPS submission requires Type 1 or embedded TrueType fonts
        # (no Type 3 bitmap fonts). fonttype=42 forces matplotlib to
        # embed TrueType outlines in the saved PDFs.
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'dejavuserif',
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.5,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#555555',
        'ytick.color': '#555555',
    })

# ─── Shadow Effects ─────────────────────────────────────────────────────────

BAR_SHADOW = [pe.withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace='#888888', alpha=0.2)]
LINE_SHADOW = [pe.withStroke(linewidth=3, foreground='#cccccc', alpha=0.4)]
SCATTER_SHADOW = [pe.withSimplePatchShadow(offset=(1, -1), shadow_rgbFace='#888888', alpha=0.25)]

# NeurIPS widths in inches
FULL_WIDTH = 5.5
HALF_WIDTH = 2.65


# ─── Data Loading ─────────────────────────────────────────────────────────────

def run_dir(variant):
    vdir, rdir = RUN_REGISTRY[variant]
    return os.path.join(BASE_DIR, vdir, "runs", rdir)


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_layer_metrics(variant):
    path = os.path.join(run_dir(variant), "layer_metrics.jsonl")
    return load_jsonl(path)


def load_epoch_metrics(variant):
    path = os.path.join(run_dir(variant), "epoch_metrics.jsonl")
    return load_jsonl(path)


def get_last_epoch_layers(variant, stage=1):
    """Return dict: layer_idx -> metrics dict for the last epoch of given stage."""
    data = load_layer_metrics(variant)
    stage_data = [r for r in data if r.get('stage') == stage]
    if not stage_data:
        return {}
    last_ep = max(r['epoch'] for r in stage_data)
    result = {}
    for r in stage_data:
        if r['epoch'] == last_ep:
            result[r['layer']] = r
    return result


def get_layer_timeseries(variant, field, stage=1):
    """Return dict: layer_idx -> (epochs_array, values_array)."""
    data = load_layer_metrics(variant)
    stage_data = [r for r in data if r.get('stage') == stage]
    by_layer = {}
    for r in stage_data:
        layer = r['layer']
        if layer not in by_layer:
            by_layer[layer] = ([], [])
        by_layer[layer][0].append(r['epoch'])
        by_layer[layer][1].append(r.get(field, 0))
    return {k: (np.array(e), np.array(v)) for k, (e, v) in by_layer.items()}


# ─── Figure 2: Free-Riding Separation Across Depth ───────────────────────────

def fig2_freeriding_separation(save=True):
    """Two-panel line chart: sep_nl across depth for key variants."""
    variants = ["baseline", "cp_fair", "v1_fair", "do_this_first"]
    labels = ["Baseline", "CP-FAIR", "Fixed-NL", "NoMem-L2"]
    colors = [VARIANT_COLORS[v] for v in variants]
    markers = ['s', 'D', 'o', '^']
    n_layers = 4
    layers = list(range(n_layers))

    sep_nl_data = {}
    sep_curr_nl_data = {}
    for v in variants:
        lm = get_last_epoch_layers(v)
        sep_nl_data[v] = [lm[l].get('sep_nl', 0) for l in layers]
        sep_curr_nl_data[v] = [lm[l].get('sep_curr_nl', 0) for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.2))
    x = np.arange(n_layers)

    # Left panel: sep_nl (line plot, zoomed)
    for i, v in enumerate(variants):
        ax1.plot(x, sep_nl_data[v], color=colors[i], marker=markers[i],
                 markersize=5.5, label=labels[i], linewidth=1.4, zorder=5-i,
                 markeredgecolor='white', markeredgewidth=0.7,
                 path_effects=LINE_SHADOW)
    ax1.set_xlabel('Block depth')
    ax1.set_ylabel(r'$\mathrm{sep}_{\mathrm{nl}}$ (cumulative)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'B{i}' for i in range(n_layers)])
    ax1.set_title('(a) Wrong-label separation', fontsize=9)
    # Zoom to interesting range
    ax1.set_ylim(0.983, 0.997)
    # [CR2] ticks on exact 0.002 steps so the 3-decimal labels are not rounded; the unexplained
    # 'flat zone' band and its label are dropped
    from matplotlib.ticker import MultipleLocator
    ax1.yaxis.set_major_locator(MultipleLocator(0.002))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Right panel: sep_curr_nl
    for i, v in enumerate(variants):
        ax2.plot(x, sep_curr_nl_data[v], color=colors[i], marker=markers[i],
                 markersize=5.5, label=labels[i], linewidth=1.4, zorder=5-i,
                 markeredgecolor='white', markeredgewidth=0.7,
                 path_effects=LINE_SHADOW)
    ax2.set_xlabel('Block depth')
    ax2.set_ylabel(r'$\mathrm{sep}^{\mathrm{cur}}_{\mathrm{nl}}$')  # [CR2] symbol of Eq. 7; 'current block' is in the title
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'B{i}' for i in range(n_layers)])
    ax2.set_title('(b) Current-block separation', fontsize=9)

    # Shared legend
    handles, lbls = ax1.get_legend_handles_labels()
    plt.tight_layout()
    # [CR] one-row legend above both panels, clear of the panel (b) y-label
    fig.legend(handles, lbls, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=7.5,
               columnspacing=1.5, handletextpad=0.5)
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig2_freeriding_separation.{fmt}'),
                        bbox_inches='tight')
    print("  Figure 2 saved.")
    return fig


# ─── Figure 3+4: Loss Effect + BP vs FF (combined full-width) ────────────────

def fig3_4_loss_and_bpff(save=True):
    """Left: loss_block_curr across depth. Right: BP vs FF bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # ─── Left panel: loss_block_curr ───
    variants_loss = ["baseline", "cp_fair"]
    labels_loss = ["Baseline", "CP-FAIR"]
    colors_loss = [VARIANT_COLORS["baseline"], VARIANT_COLORS["cp_fair"]]
    n_layers = 4
    layers = list(range(n_layers))

    loss_data = {}
    for v in variants_loss:
        lm = get_last_epoch_layers(v)
        loss_data[v] = [lm[l].get('loss_block_curr', 0) for l in layers]

    x = np.arange(n_layers)
    width = 0.30
    for i, v in enumerate(variants_loss):
        offset = (i - 0.5) * width
        b = ax1.bar(x + offset, loss_data[v], width * 0.9,
                color=colors_loss[i], label=labels_loss[i],
                edgecolor='white', linewidth=0.5)
        for patch in b:
            patch.set_path_effects(BAR_SHADOW)

    # Annotate lambda weights for CP-FAIR vertically inside each bar
    lambdas = [0.25, 0.50, 0.75, 1.00]
    cp_fair_vals = loss_data["cp_fair"]
    for j in range(n_layers):
        bar_top = cp_fair_vals[j]
        if bar_top > 0.30:
            # rotate vertically so it fits inside the narrow bar
            ax1.text(x[j] + 0.5 * width, bar_top * 0.5,
                     f'$\\lambda{{=}}{lambdas[j]:.2f}$',
                     fontsize=6, ha='center', va='center', rotation=90,
                     color='white', fontweight='bold')
        else:
            ax1.text(x[j] + 0.5 * width, bar_top + 0.015,
                     f'$\\lambda{{=}}{lambdas[j]:.2f}$',
                     fontsize=5.5, ha='center', va='bottom',
                     color=VARIANT_COLORS["cp_fair"])

    cur_top = max(max(loss_data['baseline']), max(loss_data['cp_fair']))
    ax1.set_ylim(0, cur_top * 1.30)
    ax1.set_xlabel('Block depth')
    ax1.set_ylabel(r'$\mathcal{L}_{\mathrm{curr}}^{(d)}$')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'B{i}' for i in range(n_layers)])
    ax1.set_title('(a) Current-block loss across depth', fontsize=9)
    ax1.legend(fontsize=7, frameon=False, loc='upper left')

    # ─── Right panel: BP vs FF ───
    # All values are single-crop (no-TTA) for protocol consistency,
    # matching Table 2 columns "Test no-TTA". Round 8 fix: previously
    # 'Ours+FF+depth' showed 91.36 (TTA) which mixed protocols.
    conditions = [
        ('CNN\n+BP', 89.86, '#0072B2'),
        ('CNN\n+FF', 29.41, '#D55E00'),
        ('Ours\n+BP', 86.90, '#0072B2'),
        ('Ours\n+FF', 89.03, '#D55E00'),
        ('Ours+FF\n+depth', 90.45, '#D55E00'),
    ]
    names = [c[0] for c in conditions]
    accs = [c[1] for c in conditions]
    colors_bp = [c[2] for c in conditions]

    bars = ax2.bar(range(len(conditions)), accs, color=colors_bp,
                   edgecolor='white', linewidth=0.5, width=0.7)
    for patch in bars:
        patch.set_path_effects(BAR_SHADOW)

    # Add value labels on top
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=6.5,
                 fontweight='bold')

    # Delta annotations (small notes BELOW the bar value labels, centered above each bar)
    delta_map = {3: '+2.1%', 4: '+1.4%'}
    for idx, txt in delta_map.items():
        ax2.text(idx, accs[idx] - 5, txt, fontsize=6, color='#333333',
                 ha='center', va='top', fontweight='bold')

    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(names, fontsize=6.5)
    ax2.set_ylabel('Test accuracy (%)')
    ax2.set_title('(b) BP vs. FF controlled ablation', fontsize=9)
    ax2.set_ylim(0, 110)

    # BP/FF legend placed in the empty area above the short CNN+FF bar
    # (bar index 1, height ~29%); ylim=110 gives further headroom.
    bp_patch = mpatches.Patch(color='#0072B2', label='BP training')
    ff_patch = mpatches.Patch(color='#D55E00', label='FF training')
    ax2.legend(handles=[bp_patch, ff_patch], fontsize=7, frameon=False,
               loc='upper left', bbox_to_anchor=(0.18, 0.78))

    plt.tight_layout()
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig3_4_loss_bpff.{fmt}'),
                        bbox_inches='tight')
    print("  Figures 3+4 saved.")
    return fig


# ─── Figure 5: Goodness Magnitude vs Separation vs Accuracy ──────────────────

def fig5_magnitude_vs_separation(save=True):
    """Scatter: (a) g_pos vs accuracy, (b) sep_nl vs accuracy."""
    variants_to_plot = [
        "cp_fair", "baseline", "do_this_first", "v1_fair", "atlas_ff",
        "dp_fair", "cp_fair_v2", "depthfix", "topk_depth", "freez_v1",
        "freez_v2", "v2_fair",
    ]

    gpos_l3 = {}
    sep_nl_l3 = {}
    acc = {}
    for v in variants_to_plot:
        try:
            lm = get_last_epoch_layers(v)
            if 3 in lm:
                gpos_l3[v] = lm[3].get('gpos', 0)
                sep_nl_l3[v] = lm[3].get('sep_nl', 0)
                acc[v] = TEST_ACC.get(v, 0)
        except Exception as e:
            print(f"  Warning: skipping {v}: {e}")

    # Manual label offsets to avoid overlap (dx, dy in points)
    label_offsets = {
        "cp_fair": (5, 4), "baseline": (5, -8), "do_this_first": (5, -8),
        "v1_fair": (-35, -10), "atlas_ff": (5, 4), "dp_fair": (5, -8),
        "cp_fair_v2": (5, 4), "depthfix": (5, 4), "topk_depth": (5, -8),
        "freez_v1": (5, 4), "freez_v2": (-40, -8), "v2_fair": (-30, 6),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    # Panel (a): g_pos vs accuracy
    for v in gpos_l3:
        ax1.scatter(gpos_l3[v], acc[v], c=VARIANT_COLORS.get(v, '#333'),
                    s=45, zorder=5, edgecolors='white', linewidth=0.8,
                    path_effects=SCATTER_SHADOW)
        dx, dy = label_offsets.get(v, (4, 3))
        ax1.annotate(VARIANT_LABELS.get(v, v), (gpos_l3[v], acc[v]),
                     fontsize=5, textcoords="offset points", xytext=(dx, dy),
                     color='#333333')

    ax1.set_xlabel(r'Goodness magnitude $g_+$ (Block 3)')
    ax1.set_ylabel('Test accuracy (%, TTA)')
    ax1.set_title(r'(a) Magnitude $g_+$ vs. accuracy', fontsize=9)
    # Add "no correlation" text
    xs_a = np.array([gpos_l3[v] for v in gpos_l3])
    ys_a = np.array([acc[v] for v in gpos_l3])
    corr_a = np.corrcoef(xs_a, ys_a)[0, 1]
    ax1.text(0.95, 0.05, f'r = {corr_a:.2f}', transform=ax1.transAxes,
             fontsize=7, va='bottom', ha='right', color='#999999')

    # Panel (b): sep_nl vs accuracy
    # Use different offsets for panel b (sep_nl cluster is tighter)
    label_offsets_b = {
        "cp_fair": (4, 5), "baseline": (-45, -8), "do_this_first": (4, -9),
        "v1_fair": (4, -9), "atlas_ff": (4, 4), "dp_fair": (-35, -9),
        "cp_fair_v2": (4, 5), "depthfix": (4, -9), "topk_depth": (-50, 4),
        "freez_v1": (4, 4), "freez_v2": (4, -9), "v2_fair": (-30, 7),
    }
    for v in sep_nl_l3:
        ax2.scatter(sep_nl_l3[v], acc[v], c=VARIANT_COLORS.get(v, '#333'),
                    s=45, zorder=5, edgecolors='white', linewidth=0.8,
                    path_effects=SCATTER_SHADOW)
        dx, dy = label_offsets_b.get(v, (4, 3))
        ax2.annotate(VARIANT_LABELS.get(v, v), (sep_nl_l3[v], acc[v]),
                     fontsize=5, textcoords="offset points", xytext=(dx, dy),
                     color='#333333')

    xs = np.array([sep_nl_l3[v] for v in sep_nl_l3])
    ys = np.array([acc[v] for v in sep_nl_l3])
    if len(xs) > 2:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xs.min(), xs.max(), 50)
        ax2.plot(x_line, p(x_line), '--', color='#999999', linewidth=0.8, alpha=0.5)
        corr = np.corrcoef(xs, ys)[0, 1]
        ax2.text(0.95, 0.05, f'r = {corr:.2f}', transform=ax2.transAxes,
                 fontsize=7, va='bottom', ha='right', color='#666666')

    ax2.set_xlabel(r'$\mathrm{sep_{nl}}$ (Block 3)')
    ax2.set_ylabel('Test accuracy (%, TTA)')
    ax2.set_title(r'(b) Separation $\mathrm{sep_{nl}}$ vs. accuracy', fontsize=9)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig5_magnitude_vs_separation.{fmt}'),
                        bbox_inches='tight')
    print("  Figure 5 saved.")
    return fig


# ─── Figure 6 (Appendix): Training Dynamics ──────────────────────────────────

def fig6_training_dynamics(save=True):
    """2x2 grid: sep_nl over epochs per layer, zoomed to converged region."""
    variants = ["baseline", "cp_fair", "do_this_first", "v1_fair"]
    labels = ["Baseline", "CP-FAIR", "NoMem-L2", "Fixed-NL"]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 3.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for v_idx, v in enumerate(variants):
        ts = get_layer_timeseries(v, 'sep_nl')
        for layer_idx in range(4):
            if layer_idx in ts:
                epochs, vals = ts[layer_idx]
                # Filter to epoch > 50 for zoomed view
                mask = epochs > 50
                axes[layer_idx].plot(epochs[mask], vals[mask], color=colors[v_idx],
                                     label=labels[v_idx] if layer_idx == 0 else None,
                                     linewidth=1.0, alpha=0.85)

    for i in range(4):
        axes[i].set_title(f'Block {i}', fontsize=9)
        if i >= 2:
            axes[i].set_xlabel('Epoch')
        if i % 2 == 0:
            axes[i].set_ylabel(r'$\mathrm{sep_{nl}}$')
        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axes[i].set_ylim(0.975, 1.0)

    plt.tight_layout()
    # [CR2] one-row legend above the panels instead of over the Block-0 curves
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=7.5, columnspacing=1.5, handletextpad=0.5)
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig6_training_dynamics.{fmt}'),
                        bbox_inches='tight')
    print("  Figure 6 saved.")
    return fig


# ─── Figure 7 (Appendix): MoE Routing ────────────────────────────────────────

def fig7_moe_routing(save=True):
    """MoE effective experts over training for key variants."""
    variants = ["cp_fair", "baseline", "atlas_ff"]
    labels = ["CP-FAIR", "Baseline", "Multi-Tech"]
    colors = [VARIANT_COLORS[v] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.0))

    for v_idx, v in enumerate(variants):
        try:
            data = load_epoch_metrics(v)
            s1_data = [r for r in data if r.get('stage') == 1]
            epochs = [r['epoch'] for r in s1_data]
            eff = [r.get('moe_eff_experts_load_mean', 0) for r in s1_data]
            var = [r.get('moe_load_var_mean', 0) for r in s1_data]

            ax1.plot(epochs, eff, color=colors[v_idx], label=labels[v_idx],
                     linewidth=0.9, alpha=0.85)
            ax2.plot(epochs, var, color=colors[v_idx], label=labels[v_idx],
                     linewidth=0.9, alpha=0.85)
        except Exception as e:
            print(f"  Warning: skipping {v} for MoE: {e}")

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Effective experts')  # [CR2] shorter label (was clipped)
    ax1.set_title('(a) Effective expert count', fontsize=9)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Load variance')
    ax2.set_title('(b) Routing load variance', fontsize=9)

    plt.tight_layout()
    # [CR2] one-row legend above the panels
    handles, lbls = ax1.get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=7.5, columnspacing=1.5, handletextpad=0.5)
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig7_moe_routing.{fmt}'),
                        bbox_inches='tight')
    print("  Figure 7 saved.")
    return fig


# ─── Architecture Diagram Colors ─────────────────────────────────────────────

COL_BLUE      = '#0072B2'    # standard components
COL_VERMILLION= '#D55E00'    # novel / contribution
COL_GREEN     = '#009E73'    # positive streams
COL_PURPLE    = '#CC79A7'    # negative streams
COL_ORANGE    = '#E69F00'    # MoE / routing
COL_SKY       = '#56B4E9'    # light accent
COL_GRAY      = '#BBBBBB'    # frozen / inactive
COL_LIGHTGRAY = '#E8E8E8'    # background fills
COL_DARKGRAY  = '#555555'    # text / borders
COL_WHITE     = '#FFFFFF'


def _draw_box(ax, x, y, w, h, label, color=COL_BLUE, fontsize=7,
              alpha=0.15, ec=None, linestyle='-', fontcolor=None, bold=False):
    """Draw a rounded rectangle with centered label."""
    ec = ec or color
    fc = color if alpha == 1.0 else color
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor=ec,
                         alpha=alpha, linewidth=0.8,
                         linestyle=linestyle, zorder=2)
    ax.add_patch(box)
    # Draw border at full opacity
    border = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02",
                            facecolor='none', edgecolor=ec,
                            linewidth=0.8, linestyle=linestyle, zorder=3)
    ax.add_patch(border)
    weight = 'bold' if bold else 'normal'
    tc = fontcolor or COL_DARKGRAY
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, color=tc, zorder=4, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, color=COL_DARKGRAY, lw=0.8,
           style='->', headw=5, headl=4, linestyle='-'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=ArrowStyle('->', head_width=headw/72, head_length=headl/72),
        color=color, linewidth=lw, linestyle=linestyle,
        mutation_scale=10, zorder=3,
    )
    ax.add_patch(arrow)


def _arrow_curved(ax, x1, y1, x2, y2, color=COL_DARKGRAY, lw=0.8,
                  connectionstyle="arc3,rad=0.2", linestyle='-'):
    """Draw a curved arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=ArrowStyle('->', head_width=5/72, head_length=4/72),
        color=color, linewidth=lw, linestyle=linestyle,
        connectionstyle=connectionstyle,
        mutation_scale=10, zorder=3,
    )
    ax.add_patch(arrow)


# ─── Figure 1: Full System Overview ──────────────────────────────────────────

def fig1_architecture_overview(save=True):
    """Two-panel system overview: Stage 1 backbone + Stage 2 head."""
    fig, ax = plt.subplots(1, 1, figsize=(FULL_WIDTH, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Stage 1 panel (left side, x: 0.2 - 5.8) ──

    # Stage 1 title
    ax.text(3.0, 6.75, 'Stage 1: FF Backbone Training',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=COL_BLUE)

    # Input image
    _draw_box(ax, 0.3, 5.8, 1.2, 0.6, 'Input Image\n3x32x32',
              color=COL_SKY, fontsize=6)

    # ConvStem
    _arrow(ax, 1.5, 6.1, 1.8, 6.1, color=COL_DARKGRAY)
    _draw_box(ax, 1.8, 5.85, 1.1, 0.5, 'ConvStem\n32->16',
              color=COL_BLUE, fontsize=5.5)

    # Patch Embed
    _arrow(ax, 2.9, 6.1, 3.2, 6.1, color=COL_DARKGRAY)
    _draw_box(ax, 3.2, 5.85, 1.3, 0.5, 'PatchEmbed\n64 tokens, D=256',
              color=COL_BLUE, fontsize=5)

    # Arrow down to blocks
    _arrow(ax, 3.85, 5.85, 3.85, 5.55, color=COL_DARKGRAY)

    # Label y input (left side)
    ax.text(0.6, 3.8, 'Label\nhypothesis y',
            ha='center', va='center', fontsize=6, color=COL_VERMILLION,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='#FFF3E0', ec=COL_VERMILLION,
                      linewidth=0.6))

    # Four FF Hybrid Blocks
    block_y_positions = [4.85, 3.85, 2.85, 1.85]
    lambdas = [0.25, 0.50, 0.75, 1.00]

    for i, by in enumerate(block_y_positions):
        # Block box
        _draw_box(ax, 2.1, by, 3.5, 0.55, f'FFHybridBlock {i}',
                  color=COL_BLUE, fontsize=7, bold=True)

        # Label arrow into block
        _arrow(ax, 1.2, 3.8 + (0 if i != 0 else 0), 2.1, by + 0.275,
               color=COL_VERMILLION, lw=0.6, linestyle='--')

        # L2 norm diamond between blocks
        if i < 3:
            mid_y = by - 0.15
            diamond_size = 0.12
            diamond = plt.Polygon([
                [3.85, mid_y],
                [3.85 + diamond_size, mid_y - diamond_size],
                [3.85, mid_y - 2*diamond_size],
                [3.85 - diamond_size, mid_y - diamond_size],
            ], closed=True, fc=COL_LIGHTGRAY, ec=COL_DARKGRAY,
               linewidth=0.6, zorder=3)
            ax.add_patch(diamond)
            ax.text(3.85, mid_y - diamond_size, 'L2', ha='center', va='center',
                    fontsize=4, color=COL_DARKGRAY, zorder=4)
            # "detach" label
            ax.text(4.15, mid_y - diamond_size, 'detach',
                    ha='left', va='center', fontsize=4, color='#888888',
                    fontstyle='italic')
            # Arrows connecting blocks through diamond
            _arrow(ax, 3.85, by, 3.85, mid_y + 0.02, color=COL_DARKGRAY, lw=0.6)
            _arrow(ax, 3.85, mid_y - 2*diamond_size - 0.02, 3.85,
                   block_y_positions[i+1] + 0.55, color=COL_DARKGRAY, lw=0.6)

        # Goodness arrow to the right with lambda
        gx = 5.6 + 0.05
        _arrow(ax, 5.6, by + 0.275, gx + 0.15, by + 0.275,
               color=COL_VERMILLION, lw=0.7)
        ax.text(gx + 0.2, by + 0.275,
                f'g$_{{blk}}^{{({i})}}$',
                ha='left', va='center', fontsize=5.5, color=COL_VERMILLION)
        # Lambda weight
        ax.text(gx + 0.2, by + 0.05,
                f'$\\lambda_{{curr}}$={lambdas[i]:.2f}',
                ha='left', va='center', fontsize=4.5, color=COL_VERMILLION,
                fontstyle='italic')

    # Loss summary on the right
    loss_x = 6.65
    loss_items = [
        ('Aspect losses', COL_VERMILLION),
        ('Block SYMBA', COL_VERMILLION),
        ('Depth ordering', COL_VERMILLION),
        ('SupCon', COL_GREEN),
        ('MoE balance', COL_ORANGE),
        ('Reconstruction', COL_BLUE),
    ]
    ax.text(loss_x + 0.55, 5.55, 'Per-Block Losses',
            ha='center', va='center', fontsize=6.5, fontweight='bold',
            color=COL_DARKGRAY)
    for j, (name, color) in enumerate(loss_items):
        ly = 5.2 - j * 0.32
        ax.plot(loss_x, ly, 's', color=color, markersize=4, zorder=3)
        ax.text(loss_x + 0.1, ly, name, ha='left', va='center',
                fontsize=5, color=COL_DARKGRAY)

    # SAM optimizer note
    ax.text(loss_x + 0.55, 3.2, 'SAM optimizer\nper block',
            ha='center', va='center', fontsize=5, color=COL_DARKGRAY,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.1', fc='#F5F5F5', ec='#CCCCCC',
                      linewidth=0.5))

    # EMA teacher
    ax.text(loss_x + 0.55, 2.6, 'EMA teacher\n(hard neg mining)',
            ha='center', va='center', fontsize=5, color=COL_DARKGRAY,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.1', fc='#F5F5F5', ec='#CCCCCC',
                      linewidth=0.5))

    # ── Divider ──
    ax.plot([8.0, 8.0], [0.3, 6.65], color='#CCCCCC', linewidth=1.0,
            linestyle='--', zorder=1)

    # ── Stage 2 panel (right side, x: 8.2 - 9.8) ──
    ax.text(9.0, 6.75, 'Stage 2',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=COL_VERMILLION)

    # Frozen backbone
    _draw_box(ax, 8.15, 5.2, 1.7, 0.8,
              'Frozen\nBackbone',
              color=COL_GRAY, fontsize=7, bold=True)
    # Lock icon text
    ax.text(9.6, 5.85, '  ', ha='left', va='center', fontsize=7)

    # 10 class hypotheses
    _draw_box(ax, 8.15, 4.1, 1.7, 0.7,
              '10 label\nhypotheses\ny=0,...,9',
              color=COL_SKY, fontsize=5.5)
    _arrow(ax, 9.0, 5.2, 9.0, 4.8, color=COL_DARKGRAY)

    # Feature extraction
    _draw_box(ax, 8.15, 3.0, 1.7, 0.7,
              'Feature extraction\ng:[B,10,16]\na:[B,10,1024]',
              color=COL_BLUE, fontsize=4.5)
    _arrow(ax, 9.0, 4.1, 9.0, 3.7, color=COL_DARKGRAY)

    # Attentive head
    _draw_box(ax, 8.15, 1.9, 1.7, 0.7,
              'AttentiveHybridHead\n(cross-class attention)',
              color=COL_VERMILLION, fontsize=5, bold=True)
    _arrow(ax, 9.0, 3.0, 9.0, 2.6, color=COL_DARKGRAY)

    # Output logits
    _draw_box(ax, 8.4, 1.0, 1.2, 0.5,
              '10 class logits',
              color=COL_GREEN, fontsize=6)
    _arrow(ax, 9.0, 1.9, 9.0, 1.5, color=COL_DARKGRAY)

    # CE loss
    ax.text(9.0, 0.6, 'Cross-Entropy',
            ha='center', va='center', fontsize=5.5, color=COL_DARKGRAY,
            fontstyle='italic')

    # Input arrow to Stage 2
    _draw_box(ax, 8.4, 6.0, 1.2, 0.45, 'Test Image',
              color=COL_SKY, fontsize=6)
    _arrow(ax, 9.0, 6.0, 9.0, 6.0 - 0.02, color=COL_DARKGRAY)

    plt.tight_layout()
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig1_architecture_overview.{fmt}'),
                        bbox_inches='tight')
    print("  Figure 1 (architecture overview) saved.")
    return fig


# ─── Figure 1b: FFHybridBlock Internal Detail ────────────────────────────────

def fig1b_block_detail(save=True):
    """Detailed internal structure of one FFHybridBlock."""
    fig, ax = plt.subplots(1, 1, figsize=(FULL_WIDTH, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 5.2, 'FFHybridBlock (Block $d$)',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=COL_BLUE)

    # Outer container box
    outer = FancyBboxPatch((0.15, 0.3), 10.7, 4.6,
                           boxstyle="round,pad=0.05",
                           facecolor='none', edgecolor=COL_BLUE,
                           linewidth=1.2, linestyle='-', zorder=1)
    ax.add_patch(outer)

    # ── Main horizontal flow (y ~ 2.8) ──
    main_y = 2.8
    bh = 0.65  # box height
    gap = 0.15

    # 1. Input tokens
    _draw_box(ax, 0.3, main_y - bh/2, 1.0, bh,
              'Input\ntokens\n[B,N,D]',
              color=COL_SKY, fontsize=5)

    # Label embedding (above, merging in)
    _draw_box(ax, 0.5, main_y + bh/2 + 0.4, 0.9, 0.5,
              'Label emb\n[B,N,64]',
              color=COL_VERMILLION, fontsize=5)
    # Merge circle
    merge_x = 1.55
    merge_circle = plt.Circle((merge_x, main_y), 0.12, fc=COL_LIGHTGRAY,
                              ec=COL_DARKGRAY, linewidth=0.6, zorder=3)
    ax.add_patch(merge_circle)
    ax.text(merge_x, main_y, '+', ha='center', va='center', fontsize=8,
            color=COL_DARKGRAY, zorder=4, fontweight='bold')
    # Arrows to merge
    _arrow(ax, 1.3, main_y, merge_x - 0.12, main_y, color=COL_DARKGRAY, lw=0.6)
    _arrow(ax, 0.95, main_y + bh/2 + 0.4, merge_x, main_y + 0.12,
           color=COL_VERMILLION, lw=0.6)

    # 2. RMSNorm + Linear
    nx = merge_x + 0.3
    _draw_box(ax, nx, main_y - bh/2, 0.9, bh,
              'RMSNorm\n+Linear\n->[B,N,D]',
              color=COL_BLUE, fontsize=4.5)
    _arrow(ax, merge_x + 0.12, main_y, nx, main_y, color=COL_DARKGRAY, lw=0.6)

    # 3. Self-Attention + RoPE
    sa_x = nx + 0.9 + gap
    _draw_box(ax, sa_x, main_y - bh/2, 1.1, bh,
              'Self-Attn\n+ RoPE\n(8 heads)',
              color=COL_BLUE, fontsize=5)
    _arrow(ax, nx + 0.9, main_y, sa_x, main_y, color=COL_DARKGRAY, lw=0.6)

    # 4. Memory Cross-Attn (dashed = disabled)
    mem_x = sa_x + 1.1 + gap
    _draw_box(ax, mem_x, main_y - bh/2, 1.0, bh,
              'Memory\nCross-Attn',
              color=COL_GRAY, fontsize=5, linestyle='--')
    ax.text(mem_x + 0.5, main_y - bh/2 - 0.15, '(disabled by default)',
            ha='center', va='top', fontsize=4, color='#999999', fontstyle='italic')
    _arrow(ax, sa_x + 1.1, main_y, mem_x, main_y, color=COL_DARKGRAY, lw=0.6)

    # g_prev arrow into memory from above
    ax.text(mem_x + 0.5, main_y + bh/2 + 0.55, '$g_{prev}$',
            ha='center', va='center', fontsize=5.5, color=COL_GRAY)
    _arrow(ax, mem_x + 0.5, main_y + bh/2 + 0.4, mem_x + 0.5, main_y + bh/2,
           color=COL_GRAY, lw=0.5, linestyle='--')

    # 5. Top-k MoE FFN
    moe_x = mem_x + 1.0 + gap
    _draw_box(ax, moe_x, main_y - bh/2, 1.4, bh,
              'Top-k MoE FFN\n32 experts, k=4\nGEGLU activation',
              color=COL_ORANGE, fontsize=5, bold=True)
    _arrow(ax, mem_x + 1.0, main_y, moe_x, main_y, color=COL_DARKGRAY, lw=0.6)

    # Router detail (small box above MoE)
    _draw_box(ax, moe_x + 0.15, main_y + bh/2 + 0.15, 1.1, 0.35,
              'Router: D->32, top-4 softmax',
              color=COL_ORANGE, fontsize=4)
    _arrow(ax, moe_x + 0.7, main_y + bh/2 + 0.15, moe_x + 0.7, main_y + bh/2,
           color=COL_ORANGE, lw=0.5)

    # 6. ReLU
    relu_x = moe_x + 1.4 + gap
    _draw_box(ax, relu_x, main_y - bh/2 + 0.1, 0.55, bh - 0.2,
              'ReLU',
              color=COL_GREEN, fontsize=6)
    _arrow(ax, moe_x + 1.4, main_y, relu_x, main_y, color=COL_DARKGRAY, lw=0.6)

    # ── Split point after ReLU ──
    split_x = relu_x + 0.55 + 0.1
    split_y = main_y

    # Main output: L2 Norm -> Next Block
    out_x = split_x + 0.1
    _draw_box(ax, out_x, main_y - 0.2, 0.7, 0.4,
              'L2 Norm',
              color=COL_LIGHTGRAY, fontsize=5, ec=COL_DARKGRAY)
    _arrow(ax, relu_x + 0.55, main_y, out_x, main_y, color=COL_DARKGRAY, lw=0.6)

    # "Next Block" label
    ax.text(out_x + 0.7 + 0.15, main_y, 'Next\nBlock',
            ha='left', va='center', fontsize=5.5, color=COL_DARKGRAY,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.08', fc='#F0F0F0', ec=COL_DARKGRAY,
                      linewidth=0.5))
    _arrow(ax, out_x + 0.7, main_y, out_x + 0.7 + 0.12, main_y,
           color=COL_DARKGRAY, lw=0.6)

    # ── Branch UP: Attention Pool -> 4-Aspect Goodness ──
    branch_up_y = main_y + 1.6

    # Attention Pool
    ap_x = relu_x - 0.3
    _draw_box(ax, ap_x, branch_up_y - 0.2, 1.0, 0.45,
              'AttnPool\n[B,D]',
              color=COL_BLUE, fontsize=5)
    _arrow(ax, relu_x + 0.275, main_y + bh/2 - 0.02, ap_x + 0.5, branch_up_y - 0.2,
           color=COL_BLUE, lw=0.6)

    # 4-Aspect Goodness
    asp_x = ap_x + 1.2
    _draw_box(ax, asp_x, branch_up_y - 0.35, 1.5, 0.75,
              '4-Aspect Goodness\nalignment | energy\nattn_sharp | learned',
              color=COL_VERMILLION, fontsize=4.5, bold=True)
    _arrow(ax, ap_x + 1.0, branch_up_y, asp_x, branch_up_y,
           color=COL_VERMILLION, lw=0.6)

    # Prototype alignment detail
    _draw_box(ax, asp_x - 0.1, branch_up_y + 0.55, 0.9, 0.35,
              'Prototypes\n[C, D]',
              color=COL_VERMILLION, fontsize=4)
    _arrow(ax, asp_x + 0.35, branch_up_y + 0.55, asp_x + 0.5, branch_up_y + 0.4,
           color=COL_VERMILLION, lw=0.4)

    # Aggregation weights
    agg_x = asp_x + 1.7
    ax.text(agg_x, branch_up_y,
            '$g_{blk} = \\sum_a w_a \\cdot g_a$',
            ha='left', va='center', fontsize=5.5, color=COL_VERMILLION,
            fontweight='bold')
    _arrow(ax, asp_x + 1.5, branch_up_y, agg_x - 0.05, branch_up_y,
           color=COL_VERMILLION, lw=0.6)

    # ── Branch DOWN: Contrastive + Reconstruction ──
    branch_dn_y = main_y - 1.3

    # Contrastive projection
    con_x = relu_x - 0.1
    _draw_box(ax, con_x, branch_dn_y - 0.2, 1.1, 0.45,
              'Contrastive\nProjection',
              color=COL_GREEN, fontsize=5)
    _arrow(ax, relu_x + 0.1, main_y - bh/2 + 0.05, con_x + 0.55, branch_dn_y + 0.25,
           color=COL_GREEN, lw=0.6)

    # SupCon loss
    ax.text(con_x + 1.3, branch_dn_y, 'SupCon\nLoss',
            ha='left', va='center', fontsize=5, color=COL_GREEN,
            fontstyle='italic')
    _arrow(ax, con_x + 1.1, branch_dn_y, con_x + 1.25, branch_dn_y,
           color=COL_GREEN, lw=0.6)

    # Reconstruction branch (from MoE output area)
    recon_x = moe_x - 0.3
    _draw_box(ax, recon_x, branch_dn_y - 0.2, 1.0, 0.45,
              'Reconstructor\nMSE loss',
              color=COL_BLUE, fontsize=5)
    _arrow(ax, moe_x + 0.7, main_y - bh/2 + 0.05, recon_x + 0.5, branch_dn_y + 0.25,
           color=COL_BLUE, lw=0.6)

    # MoE aux losses
    moe_loss_x = moe_x - 1.7
    ax.text(moe_loss_x + 0.5, branch_dn_y,
            'MoE: balance\n+ z-loss',
            ha='center', va='center', fontsize=4.5, color=COL_ORANGE,
            fontstyle='italic')
    _arrow(ax, moe_x + 0.2, main_y - bh/2 + 0.05, moe_loss_x + 0.7, branch_dn_y + 0.25,
           color=COL_ORANGE, lw=0.5)

    # ── Residual connection annotations ──
    ax.text(sa_x + 0.55, main_y + bh/2 + 0.08, '+res', ha='center', va='bottom',
            fontsize=4, color='#999999', fontstyle='italic')
    ax.text(moe_x + 0.7, main_y + bh/2 + 0.08, '+res', ha='center', va='bottom',
            fontsize=4, color='#999999', fontstyle='italic')

    # ── DropPath annotation ──
    ax.text(moe_x + 1.4 + 0.05, main_y + bh/2 + 0.08, 'DropPath', ha='left', va='bottom',
            fontsize=3.5, color='#AAAAAA', fontstyle='italic')

    plt.tight_layout()
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig1b_block_detail.{fmt}'),
                        bbox_inches='tight')
    print("  Figure 1b (block detail) saved.")
    return fig


# ─── Figure 1c: Training Protocol & Inference ────────────────────────────────

def fig1c_training_inference(save=True):
    """Two-panel: (a) Stage 1 four-stream training, (b) 10-pass inference."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.4),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # ── Panel (a): Stage 1 Training ──
    ax1.set_xlim(0, 6.5)
    ax1.set_ylim(0, 6.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) Stage 1: Per-Block Local Training', fontsize=8,
                  fontweight='bold', pad=8)

    # Input image
    _draw_box(ax1, 0.3, 5.3, 1.0, 0.6, 'Image x',
              color=COL_SKY, fontsize=6)

    # Three augmented views
    aug_labels = [
        ('$x_1$ (aug)', COL_GREEN, 4.3),
        ('$x_2$ (aug)', '#66BB6A', 3.5),
        ('$x_3$ (aug)', COL_PURPLE, 2.7),
    ]
    for label, color, yy in aug_labels:
        _draw_box(ax1, 0.15, yy, 0.8, 0.5, label, color=color, fontsize=5)
        _arrow(ax1, 0.8, 5.3, 0.55, yy + 0.5, color=color, lw=0.5)

    # Wrong-image negative
    _draw_box(ax1, 0.15, 1.9, 0.8, 0.5, '$x_{ni}$\n(diff img)',
              color='#B71C1C', fontsize=5)
    ax1.text(0.55, 1.6, 'from another\nclass sample', ha='center', va='top',
             fontsize=3.5, color='#888888', fontstyle='italic')

    # Four streams into the block
    stream_info = [
        ('pos1: $x_1$ + $y$', COL_GREEN, 4.55),
        ('pos2: $x_2$ + $y$', '#66BB6A', 3.75),
        ('neg-label: $x_3$ + $y\'_{hard}$', COL_PURPLE, 2.95),
        ('neg-image: $x_{ni}$ + $y$', '#B71C1C', 2.15),
    ]

    # Block
    _draw_box(ax1, 2.6, 2.5, 1.8, 2.3, '',
              color=COL_BLUE, fontsize=7, alpha=0.08)
    ax1.text(3.5, 4.6, 'FFHybridBlock $d$',
             ha='center', va='center', fontsize=7, fontweight='bold',
             color=COL_BLUE)
    ax1.text(3.5, 4.3, '(shared weights)', ha='center', va='center',
             fontsize=5, color='#888888', fontstyle='italic')

    # Stream labels and arrows
    for i, (label, color, yy) in enumerate(stream_info):
        ax1.text(1.8, yy, label, ha='center', va='center',
                 fontsize=4.5, color=color, fontweight='bold')
        _arrow(ax1, 0.95, yy, 2.6, yy, color=color, lw=0.6)

    # Goodness outputs
    g_labels = [
        ('$g_{pos1}$', COL_GREEN, 4.2),
        ('$g_{pos2}$', '#66BB6A', 3.6),
        ('$g_{nl}$', COL_PURPLE, 3.0),
        ('$g_{ni}$', '#B71C1C', 2.4),
    ]

    for label, color, yy in g_labels:
        ax1.text(4.7, yy, label, ha='center', va='center',
                 fontsize=5, color=color, fontweight='bold')
        _arrow(ax1, 4.4, yy, 4.55, yy, color=color, lw=0.6)

    # Loss computation
    _draw_box(ax1, 5.0, 2.8, 1.3, 1.8, '',
              color=COL_VERMILLION, fontsize=5, alpha=0.08)
    ax1.text(5.65, 4.4, 'Local Losses', ha='center', va='center',
             fontsize=6, fontweight='bold', color=COL_VERMILLION)

    loss_list = [
        'Aspect margin/rank',
        'Block SYMBA',
        'Current-block (depth-scaled)',
        'Depth ordering',
        'SupCon',
        'Reconstruction',
        'MoE balance + z-loss',
    ]
    for j, lt in enumerate(loss_list):
        ax1.text(5.65, 4.05 - j * 0.22, lt, ha='center', va='center',
                 fontsize=3.8, color=COL_DARKGRAY)

    # Arrows from goodness to loss
    for _, color, yy in g_labels:
        _arrow(ax1, 4.85, yy, 5.0, 3.7, color=color, lw=0.4)

    # SAM optimizer
    ax1.text(5.65, 2.2, 'SAM 2-step\noptimizer',
             ha='center', va='center', fontsize=5, color=COL_DARKGRAY,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.1', fc='#F5F5F5', ec='#CCCCCC',
                       linewidth=0.5))
    _arrow(ax1, 5.65, 2.8, 5.65, 2.4, color=COL_DARKGRAY, lw=0.6)

    # Detach + pass to next block
    ax1.text(3.5, 1.4, 'L2 normalize + detach',
             ha='center', va='center', fontsize=5, color=COL_DARKGRAY,
             fontstyle='italic')
    _arrow(ax1, 3.5, 2.5, 3.5, 1.6, color=COL_DARKGRAY, lw=0.6)
    ax1.text(3.5, 1.0, 'Block $d+1$',
             ha='center', va='center', fontsize=6, color=COL_BLUE,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.1', fc=COL_LIGHTGRAY, ec=COL_BLUE,
                       linewidth=0.6))
    _arrow(ax1, 3.5, 1.25, 3.5, 1.15, color=COL_DARKGRAY, lw=0.6)

    # EMA teacher feedback arrow (dashed curved)
    ax1.text(1.5, 0.7, 'EMA teacher',
             ha='center', va='center', fontsize=4.5, color='#888888',
             fontstyle='italic')
    ax1.text(1.5, 0.4, '(hard neg mining)', ha='center', va='center',
             fontsize=3.5, color='#AAAAAA', fontstyle='italic')
    _arrow_curved(ax1, 3.5, 0.85, 1.8, 0.8, color='#888888', lw=0.5,
                  connectionstyle="arc3,rad=-0.3", linestyle='--')
    _arrow_curved(ax1, 1.2, 0.8, 0.55, 2.7, color='#888888', lw=0.5,
                  connectionstyle="arc3,rad=-0.4", linestyle='--')

    # ── Panel (b): Inference ──
    ax2.set_xlim(0, 5.5)
    ax2.set_ylim(0, 6.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('(b) Inference (10-pass)', fontsize=8,
                  fontweight='bold', pad=8)

    # Test image
    _draw_box(ax2, 1.6, 5.6, 1.0, 0.55, 'Test\nImage',
              color=COL_SKY, fontsize=6)

    # 10 copies
    ax2.text(2.1, 5.15, '10 label hypotheses', ha='center', va='center',
             fontsize=5.5, color=COL_VERMILLION, fontweight='bold')
    _arrow(ax2, 2.1, 5.6, 2.1, 5.25, color=COL_DARKGRAY, lw=0.6)

    # Backbone
    _draw_box(ax2, 0.8, 4.0, 2.6, 0.8,
              'Full Backbone\n(4 blocks, per hypothesis)',
              color=COL_BLUE, fontsize=5.5)
    _arrow(ax2, 2.1, 5.05, 2.1, 4.8, color=COL_DARKGRAY, lw=0.6)

    # Stage 1 path (left)
    ax2.text(0.7, 3.5, 'Stage 1', ha='center', va='center',
             fontsize=6, fontweight='bold', color=COL_BLUE)

    # Sum goodness
    _draw_box(ax2, 0.1, 2.5, 1.2, 0.6,
              'Sum block\ngoodness\nper class',
              color=COL_BLUE, fontsize=5)
    _arrow(ax2, 0.7, 4.0, 0.7, 3.1, color=COL_BLUE, lw=0.6)

    # 10 goodness bars (visual)
    bar_x = 0.2
    bar_w = 0.08
    bar_heights = [0.3, 0.25, 0.15, 0.35, 0.2, 0.12, 0.28, 0.45, 0.18, 0.22]
    for i, bh_val in enumerate(bar_heights):
        bx = bar_x + i * (bar_w + 0.02)
        color = COL_VERMILLION if i == 7 else COL_SKY
        rect = plt.Rectangle((bx, 1.5), bar_w, bh_val * 1.5,
                              fc=color, ec='white', linewidth=0.3, zorder=3)
        ax2.add_patch(rect)
    ax2.text(0.7, 1.35, 'argmax', ha='center', va='top',
             fontsize=5, color=COL_DARKGRAY, fontstyle='italic')
    _arrow(ax2, 0.7, 2.5, 0.7, 2.2, color=COL_BLUE, lw=0.6)

    # Predicted class
    _draw_box(ax2, 0.25, 0.8, 0.9, 0.4, 'Predicted\nclass',
              color=COL_GREEN, fontsize=5.5, bold=True)

    _arrow(ax2, 0.7, 1.35, 0.7, 1.2, color=COL_DARKGRAY, lw=0.6)

    # Stage 2 path (right)
    ax2.text(3.8, 3.5, 'Stage 2', ha='center', va='center',
             fontsize=6, fontweight='bold', color=COL_VERMILLION)

    # Feature extraction
    _draw_box(ax2, 3.0, 2.5, 1.6, 0.6,
              'Features for\nall 10 labels',
              color=COL_BLUE, fontsize=5)
    _arrow(ax2, 3.8, 4.0, 3.8, 3.1, color=COL_VERMILLION, lw=0.6)

    # Attentive head
    _draw_box(ax2, 2.9, 1.5, 1.8, 0.6,
              'AttentiveHybridHead\n(cross-class attn)',
              color=COL_VERMILLION, fontsize=5, bold=True)
    _arrow(ax2, 3.8, 2.5, 3.8, 2.1, color=COL_VERMILLION, lw=0.6)

    # Output
    _draw_box(ax2, 3.15, 0.8, 1.3, 0.4, 'Predicted\nclass',
              color=COL_GREEN, fontsize=5.5, bold=True)
    _arrow(ax2, 3.8, 1.5, 3.8, 1.2, color=COL_VERMILLION, lw=0.6)

    # "or" between paths
    ax2.text(2.1, 1.0, 'or', ha='center', va='center',
             fontsize=7, color='#AAAAAA', fontstyle='italic')

    plt.tight_layout()
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig1c_training_inference.{fmt}'),
                        bbox_inches='tight')
    print("  Figure 1c (training & inference) saved.")
    return fig


# ─── L8_D128 Scaling: Free-Riding Comparison ─────────────────────────────────

L8D128_RUNS_DIR = os.path.join(BASE_DIR, "basic_fair_revised_valfix_cp_fair", "scaling_L8D128_32e", "runs_scale_L8D128")
L8D128_REGISTRY = {
    "gamma0": (
        "ff_cifar10_scale_noall_G0_L8D128_L8_D128_bs512_seed42_20260314_015458",
        r"$\gamma$=0 (history-free)",
    ),
    "cpfair": (
        "ff_cifar10_scale_cpfair_G07_L8D128_L8_D128_bs512_seed42_20260315_190153",
        r"$\gamma$=0.7 (CP-FAIR)",
    ),
    "lcff": (
        "ff_cifar10_scale_lcff_G10_L8D128_L8_D128_bs512_seed42_20260316_140422",
        r"$\gamma$=1.0 (LCFF-prefix)",
    ),
}
L8D128_COLORS = {"gamma0": "#0072B2", "cpfair": "#D55E00", "lcff": "#009E73"}
L8D128_MARKERS = {"gamma0": "o", "cpfair": "D", "lcff": "^"}
L8D128_ORDER = ["gamma0", "cpfair", "lcff"]


def _load_L8D128_last_epoch(variant, stage=1):
    run_name = L8D128_REGISTRY[variant][0]
    path = os.path.join(L8D128_RUNS_DIR, run_name, "layer_metrics.jsonl")
    data = load_jsonl(path)
    stage_data = [r for r in data if r.get('stage') == stage]
    if not stage_data:
        return {}
    last_ep = max(r['epoch'] for r in stage_data)
    return {r['layer']: r for r in stage_data if r['epoch'] == last_ep}


def fig_scaling_freeriding(save=True):
    """3-panel figure: free-riding metrics across 8 blocks for gamma sweep."""
    n_layers = 8
    metrics = [
        # [CR2] short titles; panel (c) plots loss_block_curr, the current-block loss
        ("sep_curr_nl", r"$\mathrm{sep}^{\mathrm{cur}}_{\mathrm{nl}}$", "(a) Separation"),
        ("gpos_cur", r"$g^+_{\mathrm{cur}}$", "(b) Positive goodness"),
        ("loss_block_curr", "current-block loss", "(c) Current-block loss"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.2))
    x = np.arange(n_layers)

    for ax, (field, ylabel, title) in zip(axes, metrics):
        for v in L8D128_ORDER:
            lm = _load_L8D128_last_epoch(v)
            vals = [lm[l].get(field, 0) for l in range(n_layers)]
            ax.plot(x, vals, color=L8D128_COLORS[v], marker=L8D128_MARKERS[v],
                    markersize=4.5, label=L8D128_REGISTRY[v][1], linewidth=1.3)
        ax.set_xlabel("Block")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(n_layers)])  # [CR2] 0-7 (B0B1... ran together)
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=7.5,
               columnspacing=1.5, handletextpad=0.5)

    plt.tight_layout(w_pad=1.2)
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig_scaling_freeriding_L8D128.{fmt}'),
                        bbox_inches='tight')
    print("  Figure scaling_freeriding saved.")
    return fig


# ─── Deep-Dive Ablation Figures ───────────────────────────────────────────────

def fig_hnm_ksweep_ablation(save=True):
    """Horizontal bar chart: HNM k-sweep and NL/NI decomposition."""
    data = [
        (r"$k{=}4$ (mild hardness)", +0.19),
        (r"$k{=}16$ (hard, flat)", +0.09),
        ("NL only (wrong-label neg.)", +0.00),
        (r"$k{=}8$ (moderate, flat)", -0.25),
        (r"$k{=}1$ (random negatives)", -0.45),
        ("NI only (wrong-image neg.)", -0.87),
    ]
    labels = [d[0] for d in data]
    deltas = [d[1] for d in data]
    colors = ['#388e3c' if d >= 0 else '#d32f2f' for d in deltas]

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.5, 2.2))
    bars = ax.barh(range(len(labels)), deltas, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.65)
    for patch in bars:
        patch.set_path_effects(BAR_SHADOW)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(r'$\Delta$ S2 TTA (pp) vs. CP-FAIR baseline', fontsize=8)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.xaxis.grid(True, alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for bar, d in zip(bars, deltas):
        offset = 0.03 if d >= 0 else -0.03
        ha = 'left' if d >= 0 else 'right'
        ax.text(bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{d:+.2f}', va='center', ha=ha, fontsize=6.5)
    ax.set_xlim(-1.05, 0.35)
    ax.set_title('Hard-Negative Mining: $k$-Sweep & NL/NI', fontsize=9)
    plt.tight_layout()
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig_hnm_ksweep_ablation.{fmt}'),
                        bbox_inches='tight')
    print("  Figure hnm_ksweep_ablation saved.")
    return fig


def fig_aspect_ema_ablation(save=True):
    """Horizontal bar chart: aspect isolation and EMA ablation."""
    data = [
        ("Energy (rank) only", +0.27),
        ("EMA eval only (no HNM)", +0.24),
        ("Learned head only", +0.09),
        ("Attn sharpness only", +0.02),
        ("Proto alignment only", -0.27),
        ("No EMA teacher", -0.34),
    ]
    labels = [d[0] for d in data]
    deltas = [d[1] for d in data]
    colors = ['#388e3c' if d >= 0 else '#d32f2f' for d in deltas]

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.5, 2.2))
    bars = ax.barh(range(len(labels)), deltas, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.65)
    for patch in bars:
        patch.set_path_effects(BAR_SHADOW)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(r'$\Delta$ S2 TTA (pp) vs. CP-FAIR baseline', fontsize=8)
    ax.axvline(0, color='black', linewidth=0.6)
    ax.xaxis.grid(True, alpha=0.12, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for bar, d in zip(bars, deltas):
        offset = 0.02 if d >= 0 else -0.02
        ha = 'left' if d >= 0 else 'right'
        ax.text(bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{d:+.2f}', va='center', ha=ha, fontsize=6.5)
    ax.set_title('Aspect Isolation & EMA Ablation', fontsize=9)
    plt.tight_layout()
    if save:
        for fmt in ['pdf', 'png']:
            fig.savefig(os.path.join(OUT_DIR, f'fig_aspect_ema_ablation.{fmt}'),
                        bbox_inches='tight')
    print("  Figure aspect_ema_ablation saved.")
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS paper figures")
    parser.add_argument('--fig', default='all',
                        help='Figure to generate: 2, 3_4, 5, 6, 7, or all')
    parser.add_argument('--output-dir', default=OUT_DIR)
    args = parser.parse_args()

    import sys
    this_module = sys.modules[__name__]
    this_module.OUT_DIR = args.output_dir
    os.makedirs(this_module.OUT_DIR, exist_ok=True)
    setup_style()

    fig_map = {
        '1': fig1_architecture_overview,
        '1b': fig1b_block_detail,
        '1c': fig1c_training_inference,
        '2': fig2_freeriding_separation,
        '3_4': fig3_4_loss_and_bpff,
        '5': fig5_magnitude_vs_separation,
        '6': fig6_training_dynamics,
        '7': fig7_moe_routing,
        'scale': fig_scaling_freeriding,
        'hnm_ksweep': fig_hnm_ksweep_ablation,
        'aspect_ema': fig_aspect_ema_ablation,
    }

    if args.fig == 'all':
        for name, func in fig_map.items():
            print(f"Generating figure {name}...")
            func()
            plt.close('all')
    else:
        if args.fig in fig_map:
            print(f"Generating figure {args.fig}...")
            fig_map[args.fig]()
        else:
            print(f"Unknown figure: {args.fig}. Options: {list(fig_map.keys())} or 'all'")


if __name__ == '__main__':
    # [CR] only the four figures the camera-ready paper uses from this script
    setup_style()
    import sys as _sys
    OUT_DIR = _sys.argv[1] if len(_sys.argv) > 1 else OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    for _f in (fig2_freeriding_separation, fig6_training_dynamics, fig7_moe_routing, fig_scaling_freeriding):
        _f()
        plt.close('all')
