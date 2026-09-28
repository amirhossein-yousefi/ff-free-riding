"""
Plot per-block gradient attenuation from the snapshot data.

Two-panel figure:
  (a) per-block current-block margin m^(d) per variant (median + IQR), shows
      the free-riding collapse: under γ>0 deeper blocks abandon their own margin.
  (b) deepest-block scatter |gradient| vs upstream accumulated margin
      Σ_{j<d} m^(j), with the analytical envelope α·σ(-α(m+γ·Σm)) overlaid
      per variant — visualizes the proposition's exponential attenuation.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():
    here = Path(__file__).resolve().parent
    data = np.load(here / "grad_attenuation_arrays.npz", allow_pickle=True)
    with open(here / "grad_attenuation_data.json") as f:
        summary = json.load(f)

    label_map = {
        0.0: r"$\gamma=0$ (history-free)",
        0.7: r"$\gamma=0.7$ (CP-FAIR-style)",
        1.0: r"$\gamma=1.0$ (LCFF)",
    }
    colors = {0.0: "tab:blue", 0.7: "tab:orange", 1.0: "tab:red"}
    gammas = [s["gamma"] for s in summary]
    n_var = len(gammas)
    L = data["0_m"].shape[0]
    blocks = list(range(L))
    alpha = 4.0  # symba_alpha; matches all three configs

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))

    # ---- panel (a): per-block m^(d) per variant ----
    ax = axes[0]
    width = 0.22
    offsets = np.linspace(-width, width, n_var)
    for vi, g in enumerate(gammas):
        m = data[f"{vi}_m"]
        med = np.median(m, axis=1)
        p25 = np.quantile(m, 0.25, axis=1)
        p75 = np.quantile(m, 0.75, axis=1)
        x = np.array(blocks) + offsets[vi]
        yerr_lo = np.clip(med - p25, 0, None)
        yerr_hi = np.clip(p75 - med, 0, None)
        ax.errorbar(x, med, yerr=[yerr_lo, yerr_hi], fmt='o', capsize=3,
                    color=colors[g], label=label_map[g], ms=5, lw=1.4)
        # connect with line
        ax.plot(x, med, color=colors[g], lw=1.0, alpha=0.5)
    ax.set_xticks(blocks)
    ax.set_xticklabels([f"L{b}" for b in blocks])
    ax.set_xlabel(r"block depth $d$")
    ax.set_ylabel(r"current-block margin $m^{(d)}$")
    ax.set_title("(a)  Free-riding signature: per-block margin")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.25)

    # ---- panel (b): scatter of |gradient| vs P_prev at deepest block ----
    ax = axes[1]
    d_show = L - 1  # deepest block
    # analytical envelope evaluated at the variant's mean m^(d)
    P_grid = np.linspace(0.0, 8.0, 200)

    for vi, g in enumerate(gammas):
        m = data[f"{vi}_m"][d_show]      # [N]
        P_prev = data[f"{vi}_P_prev"][d_show]
        grad = data[f"{vi}_grad"][d_show]
        # subsample for readability
        idx = np.random.RandomState(0).choice(len(grad), size=min(400, len(grad)), replace=False)
        ax.scatter(P_prev[idx], grad[idx], color=colors[g], s=8, alpha=0.4, edgecolors="none")
        # analytical envelope at the variant's median m
        m_med = np.median(m)
        env = alpha * 1.0 / (1.0 + np.exp(alpha * (m_med + g * P_grid)))
        ax.plot(P_grid, env, color=colors[g], lw=1.6,
                label=label_map[g] + rf"  ($\bar m^{{({d_show})}}{{=}}{m_med:+.2f}$)")

    ax.set_xlabel(r"upstream accumulated margin $\sum_{j<d} m^{(j)}$")
    ax.set_ylabel(r"applied gradient $|\partial \ell_\alpha / \partial m^{(d)}|$")
    ax.set_title(rf"(b)  attenuation at deepest block $d{{=}}{d_show}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 5.0)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.25, which="both")

    fig.suptitle(
        rf"Empirical gradient attenuation predicted by Prop.~1  (CIFAR-10, L4/D128, $\alpha{{=}}{alpha}$, $N{{=}}1024$)",
        y=1.03,
    )
    fig.tight_layout()
    out_pdf = here / "fig_grad_attenuation.pdf"
    out_png = here / "fig_grad_attenuation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print("Saved", out_pdf)
    print("Saved", out_png)


if __name__ == "__main__":
    main()
