#!/usr/bin/env python3
"""Fig. 24 of the camera-ready paper (fig:magnitude_vs_sep, App. N): goodness magnitude and
wrong-label separation at Block 3 versus test accuracy for the 12 seed-42 CIFAR-10 variants.

[CR] The submitted fig5_magnitude_vs_separation.pdf (numbered badges, legend on the right) has no
surviving generator; analysis/generate_figures.py fig5_magnitude_vs_separation plots the same data
in a different layout. This script re-draws the submitted layout from the same inputs:
  * data: generate_figures_cr.py (RUN_REGISTRY, TEST_ACC, get_last_epoch_layers): gpos and sep_nl of
    Block 3 at the last Stage-1 epoch of each run's layer_metrics.jsonl, and the TEST_ACC table;
  * layout: the page size, axes boxes, axis limits and the badge positions (BADGE_A / BADGE_B, in data
    coordinates) were read from the submitted PDF; the data points it plots equal the values below to
    its drawing precision.
The only intended difference from the submitted figure is the legend label "CP-FAIR" (was
"CP-FAIR (ours)"); fonts and line styles follow the submitted PDF (DejaVu Sans, TrueType).

Usage:  FF_HOME=... python magnitude_vs_separation_cr.py OUT_DIR
        (data paths as in generate_figures_cr.py: FF_BASE_DIR or FF_HOME)
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the scripts

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "generate_figures_cr", os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_figures_cr.py"))
gf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gf)

VARIANTS = ["cp_fair", "baseline", "do_this_first", "v1_fair", "atlas_ff", "dp_fair",
            "cp_fair_v2", "depthfix", "topk_depth", "freez_v1", "freez_v2", "v2_fair"]

# badge centers (data coordinates) of the submitted figure, panel (a): (g+, accuracy)
BADGE_A = {
    "cp_fair": (6.3227, 92.2076), "baseline": (10.6026, 92.0163), "do_this_first": (3.7525, 90.8763),
    "v1_fair": (4.0775, 90.0268), "atlas_ff": (11.9873, 91.3432), "dp_fair": (11.4905, 89.6955),
    "cp_fair_v2": (4.5743, 91.6745), "depthfix": (8.5292, 92.3327), "topk_depth": (9.7421, 89.1624),
    "freez_v1": (7.5356, 89.0373), "freez_v2": (12.3123, 90.4937), "v2_fair": (5.4622, 89.3537),
}
# panel (b): (sep_nl, accuracy)
BADGE_B = {
    "cp_fair": (0.97821, 92.2034), "baseline": (1.00100, 91.3528), "do_this_first": (0.99540, 92.0226),
    "v1_fair": (0.96798, 90.8658), "atlas_ff": (0.97123, 91.6661), "dp_fair": (1.00236, 90.5042),
    "cp_fair_v2": (0.98706, 92.3339), "depthfix": (0.96934, 90.0172), "topk_depth": (0.98329, 89.0361),
    "freez_v1": (0.97494, 89.3474), "freez_v2": (0.99912, 89.7039), "v2_fair": (0.99213, 89.1666),
}

PT = 1.0 / 72.0
PAGE_W, PAGE_H = 542.25825, 219.31                      # points, as in the submitted PDF
AX_A = (37.10625, 34.95, 160.9411764706, 164.16)       # left, bottom, width, height (points)
AX_B = (241.8234264706, 34.95, 160.9411764706, 164.16)
LEG_BADGE_X, LEG_LABEL_X = 450.966485, 462.4737794118  # legend column (points)
LEG_Y0, LEG_DY, LEG_BASELINE = 192.5436, 13.729741, 2.203125


def badge(target, x, y, text, color, pad, **kw):
    return target.text(x, y, text, fontsize=7, fontweight="bold", color="black", ha="center", va="center",
                       bbox=dict(boxstyle=f"circle,pad={pad}", fc=color, ec="black", lw=0.5), **kw)


def main(out_dir):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                         "mathtext.fontset": "dejavusans", "axes.spines.top": False,
                         "axes.spines.right": False})
    g, s, a = {}, {}, {}
    for v in VARIANTS:
        lm = gf.get_last_epoch_layers(v)
        g[v], s[v], a[v] = lm[3]["gpos"], lm[3]["sep_nl"], gf.TEST_ACC[v]
    acc = np.array([a[v] for v in VARIANTS])

    fig = plt.figure(figsize=(PAGE_W * PT, PAGE_H * PT))
    rect = lambda r: (r[0] / PAGE_W, r[1] / PAGE_H, r[2] / PAGE_W, r[3] / PAGE_H)  # noqa: E731
    ax1, ax2 = fig.add_axes(rect(AX_A)), fig.add_axes(rect(AX_B))

    panels = [(ax1, g, BADGE_A, r"Goodness magnitude $g_+$ (Block 3)", r"(a) Magnitude $g_+$ vs. accuracy"),
              (ax2, s, BADGE_B, r"$\mathrm{sep}_{\mathrm{nl}}$ (Block 3)",
               r"(b) Separation $\mathrm{sep}_{\mathrm{nl}}$ vs. accuracy")]
    for ax, x, badges, xlabel, title in panels:
        xs = np.array([x[v] for v in VARIANTS])
        ax.grid(True, alpha=0.25, linewidth=0.5)
        for v in VARIANTS:  # leader lines from each point to its badge
            ax.plot([x[v], badges[v][0]], [a[v], badges[v][1]], color="#888888", lw=0.5, alpha=0.8, zorder=2)
        for v in VARIANTS:
            ax.scatter([x[v]], [a[v]], s=36, c=gf.VARIANT_COLORS[v], edgecolors="black", linewidths=0.5,
                       zorder=3)
        r = np.corrcoef(xs, acc)[0, 1]
        ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                color="#777777")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Test accuracy (%, TTA)", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.set_ylim(acc.min() - 1.7, acc.max() + 1.5)
        for i, v in enumerate(VARIANTS, start=1):
            badge(ax, badges[v][0], badges[v][1], str(i), gf.VARIANT_COLORS[v], 0.13, zorder=5, clip_on=False)
    ax1.set_xlim(min(g.values()) - 3, max(g.values()) + 3)
    ax2.set_xlim(min(s.values()) - 0.012, max(s.values()) + 0.010)
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    xs = np.array([s[v] for v in VARIANTS])
    fit = np.poly1d(np.polyfit(xs, acc, 1))
    x_line = np.linspace(xs.min(), xs.max(), 50)
    ax2.plot(x_line, fit(x_line), "--", color="#999999", linewidth=0.9, alpha=0.6, zorder=1)

    for i, v in enumerate(VARIANTS):  # legend: badge number and variant name
        y = LEG_Y0 - i * LEG_DY
        badge(fig, LEG_BADGE_X / PAGE_W, y / PAGE_H, str(i + 1), gf.VARIANT_COLORS[v], 0.16)
        fig.text(LEG_LABEL_X / PAGE_W, (y - LEG_BASELINE) / PAGE_H, gf.VARIANT_LABELS[v], fontsize=7.5,
                 ha="left", va="baseline")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig5_magnitude_vs_separation.pdf"))
    fig.savefig(os.path.join(out_dir, "fig5_magnitude_vs_separation.png"), dpi=300)
    plt.close(fig)
    print("r(g+, acc) = %.3f   r(sep_nl, acc) = %.3f" % (np.corrcoef([g[v] for v in VARIANTS], acc)[0, 1],
                                                          np.corrcoef(xs, acc)[0, 1]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FF_FIG_DIR", "figures"))
