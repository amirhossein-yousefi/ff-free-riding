#!/usr/bin/env python3
"""Tier-0 #1: Free-riding index F_d — the theorem's corollary quantity, never plotted.

F_d = E_x[1 - min(1, R_gamma(m_d, P_{d-1}))],  R = (1+e^{beta m})/(1+e^{beta(m+gamma P)}), beta=4.
Panel A: F_d vs depth per regime.  Panel B: batch-mean attenuation R_d (log scale).
Also: pointwise assumption coverage (m>=0 AND P>=0), upgrading the paper's batch-mean caveat.
CPU-only; input = per-example arrays from the published grad-attenuation snapshot.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/revised_training/grad_attenuation/grad_attenuation_arrays.npz"
OUTDIR = os.path.dirname(os.path.abspath(__file__))
BETA = 4.0

z = np.load(SRC, allow_pickle=True)
tags, gammas = list(z["tags"]), list(z["gammas"])
COLORS = {0: "#2a78d6", 1: "#008300", 2: "#e87ba4"}  # validated categorical slots 1-3
LABELS = {0: "γ=0 (block-local)", 1: "γ=0.7 (CP-FAIR-style)", 2: "γ=1.0 (LCFF)"}

def log_R(m, gP):
    # log R = log(1+e^{beta m}) - log(1+e^{beta(m+gP)}), stable via logaddexp
    return np.logaddexp(0, BETA * m) - np.logaddexp(0, BETA * (m + gP))

results = {"beta": BETA, "n_examples": 1024, "source": SRC, "variants": []}
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
fig.patch.set_facecolor("white")

for v in range(3):
    m, P = z[f"{v}_m"].astype(np.float64), z[f"{v}_P_prev"].astype(np.float64)
    g = float(gammas[v])
    R = np.exp(log_R(m, g * P))                      # (4, 1024)
    F = (1 - np.minimum(1, R)).mean(axis=1)          # F_d per block
    F_ci = 1.96 * (1 - np.minimum(1, R)).std(axis=1) / np.sqrt(R.shape[1])
    Rbar = R.mean(axis=1)
    Rmed = np.median(R, axis=1)
    Rp10, Rp90 = np.percentile(R, 10, axis=1), np.percentile(R, 90, axis=1)
    # the paper's tab:assumption_coverage statistic: R evaluated AT the batch-mean margins
    R_at_means = np.exp(log_R(m.mean(axis=1), g * P.mean(axis=1)))
    cover = ((m >= 0) & (P >= 0)).mean(axis=1)       # pointwise assumption coverage
    results["variants"].append({
        "tag": tags[v], "gamma": g,
        "F_d": [round(float(x), 4) for x in F],
        "F_d_ci95": [round(float(x), 4) for x in F_ci],
        "mean_R_d": [float(f"{x:.3e}") for x in Rbar],
        "median_R_d": [float(f"{x:.3e}") for x in Rmed],
        "R_at_batch_means (paper statistic)": [float(f"{x:.3e}") for x in R_at_means],
        "frac_fully_starved_R_le_1e-2_pct": [round(float((R[d] <= 1e-2).mean() * 100), 1) for d in range(4)],
        "assumption_coverage_pct": [round(float(x) * 100, 1) for x in cover],
    })
    blocks = np.arange(4)
    axes[0].errorbar(blocks, F, yerr=F_ci, color=COLORS[v], lw=2, marker="o", ms=5,
                     capsize=2, label=LABELS[v])
    floor = 1e-9
    axes[1].plot(blocks, np.maximum(Rmed, floor), color=COLORS[v], lw=2, marker="o", ms=5,
                 label=LABELS[v])
    axes[1].fill_between(blocks, np.maximum(Rp10, floor), np.maximum(Rp90, floor),
                         color=COLORS[v], alpha=0.13, lw=0)

axes[0].set_xlabel("Block d", fontsize=9)
axes[0].set_ylabel("Free-riding index  $F_d$", fontsize=9)
axes[0].set_title("A — $F_d = \\mathbb{E}[1-\\min(1,R_d)]$  (higher = starved)", fontsize=9, loc="left")
axes[0].set_ylim(-0.03, 1.05)
axes[0].annotate("γ=0: no free-riding by construction", xy=(1.5, 0.045), fontsize=7.5, color="#444444")
axes[1].set_yscale("log")
axes[1].set_ylim(1e-9, 3)
axes[1].set_xlabel("Block d", fontsize=9)
axes[1].set_ylabel("Per-example attenuation  $R_d$", fontsize=9)
axes[1].set_title("B — median $R_d$ with p10–p90 band (log scale)", fontsize=9, loc="left")
axes[1].axhline(1e-2, color="#999999", lw=1, ls=":")
axes[1].annotate("$10^{-2}$ — fully starved below this line\n(66–90% of examples at blocks 1–3, γ=0.7)",
                 xy=(0.05, 2.0e-2), fontsize=7, color="#666666")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    ax.set_xticks(range(4))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig_free_riding_index.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "fig_free_riding_index.pdf"), bbox_inches="tight")

with open(os.path.join(OUTDIR, "fd_results.json"), "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
