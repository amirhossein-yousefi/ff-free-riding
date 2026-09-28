#!/usr/bin/env python3
"""Camera-ready figures (NeurIPS 2026), built from raw sources only.

Figures (vector PDF, written to --output-dir, default CR/paper/figures):

  grad_fd   -> fig_grad_attenuation_fd.pdf   (item B2 / D8)
      Two panels that merge the submitted Fig. 1 (grad-attenuation snapshot) with the
      free-riding index F_d promised in the rebuttal:
        (a) F_d = E_x[1 - min(1, R_d)] per block and variant, with 95% normal CI;
        (b) per-example |d l_beta / d m^(d)| at the deepest block vs. the upstream margin
            P^(d-1), with the Theorem 3.1 curve beta*sigma(-beta(m~ + gamma P)).
      Sources: grad_attenuation_arrays.npz (the published snapshot pipeline) for both
      panels, cross-checked against ff-wiki/_build/analyses/fd_results.json and
      grad_attenuation_data.json.  The check is written to results/fd_figure_check.json.
      The per-example numbers are the published snapshot pipeline: raw (not EMA) Stage-1
      weights and an eval-normalization std that differs from training (H23 / E3).  Pass
      --fd-npz to rebuild from an E3 re-dump with the same array layout.

  loss_bpff -> fig3_4_loss_bpff_cr.pdf       (item C6 / J:21, G1)
      Port of fig3_4_loss_and_bpff() from
      TEX/supplement_code/analysis/generate_figures.py (copied, not run in place; its
      default OUT_DIR points into a deny-listed tree).  Changes: no "+2.1%" / "+1.4%"
      delta annotations; BP rows and plain-CNN FF at n=3 (mean +- sample SD) from the
      per-seed final_metrics.json files; the BP-strong bar is added; stripped-backbone FF
      and CP-FAIR stay single-seed (n=1) and carry no error bar; American English; no
      drop shadows.  Values are written to results/fig3_4_bpff_values.json.

  margins   -> fig_block_margin_cr.pdf       (optional appendix companion)
      The submitted Fig. 1(a) (per-block current margin, median + IQR) as a stand-alone
      panel, so its caption numbers keep a figure if Fig. 1(a) leaves the main text.

  dissociation_summary -> fig_dissociation_summary.pdf   (verifier fix; dropped from the camera-ready
      paper in the appendix trim, its values are printed in Tabs. 3 and 15)
      CIFAR-100 trio deepest-block sep vs. S2-TTA test top-1 (3-seed mean +- sample SD) from
      results/c100_trio.json; annotation uses the unrounded ratio / range (5.0x, 0.28 pp) instead
      of the submitted plot's printed-mean "4.97x ... 0.27 pp".  Values: results/
      fig_dissociation_summary_values.json.

  accuracy_vs_ratio -> fig_accuracy_vs_ratio.pdf        (App. H.4, Fig. 10; verifier fix)
      Port of plot_accuracy_vs_ratio() (analyze_additional_L8.py), same 15 L4+L8 D128 runs and
      quantities; the y-axis now says what is plotted (final-epoch Stage-2 validation accuracy,
      5k split) instead of "S2 Test Accuracy".  Values: results/fig_accuracy_vs_ratio_values.json.

  early_exit -> fig_early_exit_pareto.pdf       (Part I, item C14 / E2)
      Validation-selected early exit: test accuracy delta vs. FLOPs saved per run, CIFAR-10 L4/D128
      (9 arms, seed 42) and the CIFAR-100 trio (9 runs), one panel per selection rule (T, T', D);
      color + marker = objective family (cumulative vs. block-local / hardness-gated repairs).
      Source: results/e2_exit.json via analysis/early_exit.py.  Values (incl. the paired CIFAR-100
      repaired-minus-cumulative comparisons): results/fig_early_exit_pareto_values.json.

CPU only; no torch import.  Run:
  CUDA_VISIBLE_DEVICES='' python analysis/camera_ready/make_figures.py --fig all
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402  (shared path registry: CF, AN, BP_RUNS, RESULTS, PAPER)

# ----------------------------------------------------------------------------- paths
GRAD_ATT_DIR = C.CF / "revised_training" / "grad_attenuation"
GRAD_ATT_NPZ = GRAD_ATT_DIR / "grad_attenuation_arrays.npz"
GRAD_ATT_SUMMARY = GRAD_ATT_DIR / "grad_attenuation_data.json"
FD_RESULTS = C.AN / "fd_results.json"
# checkpoints/configs the snapshot pipeline used (snapshot_grad_attenuation.py VARIANTS)
GRAD_ATT_CONFIGS = {
    0.0: C.GA / "runs" / "ff_cifar10_no_all_collab_L4_D128_bs512_seed42_20260315_124117" / "config.json",
    0.7: C.GA / "runs" / "ff_cifar10_baseline_constant_L4_D128_bs512_seed42_20260313_232043" / "config.json",
    1.0: C.GA / "runs" / "ff_cifar10_lcff_prefix_L4_D128_bs512_seed42_20260315_084228" / "config.json",
}
SNAPSHOT_EVAL_STD = (0.2470, 0.2435, 0.2616)   # snapshot_grad_attenuation.py:133
TRAINER_EVAL_STD = (0.2023, 0.1994, 0.2010)    # hardness_gated trainer eval_tf

# fig3_4 (a): the two runs in generate_figures.py RUN_REGISTRY
LOSS_RUNS = {
    "baseline": C.CF / "basic_fair_revised_valfix" / "runs"
    / "ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260108_214956",
    "cp_fair": C.CF / "basic_fair_revised_valfix_cp_fair" / "runs"
    / "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035",
}
# the load-last continuation that finished the Baseline run (epoch 287 + Stage 2 + test eval)
BASELINE_FINISH_RUN = C.CF / "basic_fair_revised_valfix" / "runs" / \
    "ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260110_171401"
# fig3_4 (b): single-seed FF rows of Table 2
FF_STRIPPED_RUN = C.CF / "ff_backbone_stripped" / "runs_ff_backbone_stripped" / \
    "ff_backbone_stripped_cifar10_L4_D256_P2_seed42_20260306_153106"
CP_FAIR_RUN = LOSS_RUNS["cp_fair"]

BETA = 4.0          # softplus barrier sharpness (config symba_alpha = 4.0 in all three runs)
N_FD = 1024

# ----------------------------------------------------------------------------- style
# Okabe-Ito hues; the triple below passes the dataviz validator on a white surface with
# --pairs all (worst CVD dE 11.0, worst normal-vision dE 18.7, all >= 3:1 contrast).
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, INK2, GRID = "#222222", "#555555", "#e6e6e6"
VARIANT_STYLE = {   # gamma -> (color, marker, legend label)
    0.0: (BLUE, "o", r"$\gamma=0$ (history-free)"),
    0.7: (VERMILLION, "s", r"$\gamma=0.7$ (CP-FAIR-style)"),
    1.0: (GREEN, "^", r"$\gamma=1.0$ (LCFF)"),
}
FULL_WIDTH = 5.5    # NeurIPS \textwidth in inches


def setup_style():
    """House style of supplement_code/analysis/generate_figures.py (serif, TrueType)."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "patch.linewidth": 0.5,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "ytick.minor.size": 1.5, "ytick.minor.width": 0.4,
        "savefig.dpi": 300, "figure.dpi": 150,
    })


# ----------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("  wrote", path)


def save_pdf(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.savefig(out, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    w, h = fig.get_size_inches()
    print(f"  wrote {out}  ({w:.2f} x {h:.2f} in)")
    return out


def sample_mean_sd(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan")


# ============================================================================= F_d (B2)
def log_R(m, gP):
    """log R = log(1+e^{beta m}) - log(1+e^{beta(m+gamma P)}), stable (Theorem 3.1, Eq. 4)."""
    return np.logaddexp(0.0, BETA * m) - np.logaddexp(0.0, BETA * (m + gP))


def compute_fd(npz_path: Path):
    """Recompute every quantity of fd_results.json from the per-example arrays
    (same formulas as ff-wiki/_build/analyses/fd_figure.py, float64)."""
    z = np.load(npz_path, allow_pickle=True)
    tags, gammas = [str(t) for t in z["tags"]], [float(g) for g in z["gammas"]]
    out = []
    for v, g in enumerate(gammas):
        m = z[f"{v}_m"].astype(np.float64)
        P = z[f"{v}_P_prev"].astype(np.float64)
        R = np.exp(log_R(m, g * P))                     # (L, N)
        starved = 1.0 - np.minimum(1.0, R)
        L, N = R.shape
        out.append({
            "tag": tags[v], "gamma": g, "L": L, "N": N,
            "F_d": starved.mean(axis=1),
            "F_d_ci95": 1.96 * starved.std(axis=1) / np.sqrt(N),   # ddof=0, as fd_figure.py
            "mean_R_d": R.mean(axis=1),
            "median_R_d": np.median(R, axis=1),
            "R_at_batch_means": np.exp(log_R(m.mean(axis=1), g * P.mean(axis=1))),
            "frac_fully_starved_pct": (R <= 1e-2).mean(axis=1) * 100.0,
            "assumption_coverage_pct": ((m >= 0) & (P >= 0)).mean(axis=1) * 100.0,
            "m_mean": m.mean(axis=1), "m_median": np.median(m, axis=1),
            "m_p25": np.quantile(m, 0.25, axis=1), "m_p75": np.quantile(m, 0.75, axis=1),
            "P_mean": P.mean(axis=1),
            "_m": m, "_P": P, "_M": z[f"{v}_M"].astype(np.float64),
            "_grad": z[f"{v}_grad"].astype(np.float64),
        })
    return out


def check_fd(rec, npz_path: Path):
    """Assert the recomputed numbers match fd_results.json and grad_attenuation_data.json."""
    pub = json.load(open(FD_RESULTS))
    summ = json.load(open(GRAD_ATT_SUMMARY))
    assert abs(pub["beta"] - BETA) < 1e-12 and pub["n_examples"] == N_FD
    variants, all_ok = [], True
    # published rounding: F_d / CI to 4 dp; R stats to 4 significant digits; pct to 1 dp
    for r, p, s in zip(rec, pub["variants"], summ):
        assert r["tag"] == p["tag"] == s["tag"] and r["gamma"] == p["gamma"] == s["gamma"]
        assert s["alpha"] == BETA
        cmp = {}

        def absdiff(key_r, key_p, tol):
            d = float(np.max(np.abs(np.asarray(r[key_r]) - np.asarray(p[key_p]))))
            cmp[key_p] = {"recomputed": [float(x) for x in r[key_r]], "published": p[key_p],
                          "max_abs_diff": d, "tol": tol, "ok": d <= tol}
            return d <= tol

        def reldiff(key_r, key_p, rtol):
            a, b = np.asarray(r[key_r], float), np.asarray(p[key_p], float)
            d = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
            cmp[key_p] = {"recomputed": [float(f"{x:.6e}") for x in a], "published": p[key_p],
                          "max_rel_diff": d, "rtol": rtol, "ok": d <= rtol}
            return d <= rtol

        ok = all([
            absdiff("F_d", "F_d", 5.0e-5 + 1e-9),
            absdiff("F_d_ci95", "F_d_ci95", 5.0e-5 + 1e-9),
            reldiff("mean_R_d", "mean_R_d", 1e-3),
            reldiff("median_R_d", "median_R_d", 1e-3),
            reldiff("R_at_batch_means", "R_at_batch_means (paper statistic)", 1e-3),
            absdiff("frac_fully_starved_pct", "frac_fully_starved_R_le_1e-2_pct", 0.05 + 1e-9),
            absdiff("assumption_coverage_pct", "assumption_coverage_pct", 0.05 + 1e-9),
        ])
        # grad_attenuation_data.json per-block means (float32 torch vs numpy float64)
        sm = np.array([b["m_mean"] for b in s["summary"]])
        sP = np.array([b["P_prev_mean"] for b in s["summary"]])
        d_m = float(np.max(np.abs(sm - r["m_mean"])))
        d_P = float(np.max(np.abs(sP - r["P_mean"])))
        ok_sum = d_m < 1e-4 and d_P < 1e-4
        # npz internal consistency: M = m + gamma P, grad = beta sigma(-beta M)
        d_M = float(np.max(np.abs(r["_M"] - (r["_m"] + r["gamma"] * r["_P"]))))
        d_g = float(np.max(np.abs(r["_grad"] - BETA / (1.0 + np.exp(BETA * r["_M"])))))
        ok_int = d_M < 1e-5 and d_g < 1e-5
        all_ok &= ok and ok_sum and ok_int
        variants.append({
            "tag": r["tag"], "gamma": r["gamma"], "ok": bool(ok and ok_sum and ok_int),
            "vs_fd_results_json": cmp,
            "vs_grad_attenuation_data_json": {"m_mean_max_abs_diff": d_m, "P_prev_mean_max_abs_diff": d_P,
                                              "ok": ok_sum},
            "npz_internal": {"M_eq_m_plus_gammaP_max_abs_err": d_M,
                             "grad_eq_beta_sigmoid_max_abs_err": d_g, "ok": ok_int},
        })
    # headline assertions requested for the caption
    g0 = next(r for r in rec if r["gamma"] == 0.0)
    g07 = next(r for r in rec if r["gamma"] == 0.7)
    head = {
        "F_d_gamma0_all_blocks_zero": bool(np.all(np.abs(g0["F_d"]) < 1e-12)),
        "F_d_gamma07_blocks1to3": [round(float(x), 4) for x in g07["F_d"][1:]],
        "F_d_gamma07_blocks1to3_expected": [0.9299, 0.9523, 0.955],
    }
    head["F_d_gamma07_blocks1to3_ok"] = bool(np.allclose(head["F_d_gamma07_blocks1to3"],
                                                         head["F_d_gamma07_blocks1to3_expected"], atol=5e-5))
    all_ok &= head["F_d_gamma0_all_blocks_zero"] and head["F_d_gamma07_blocks1to3_ok"]
    return all_ok, variants, head


def fig_grad_fd(out_dir: Path, results_dir: Path, npz_path: Path = GRAD_ATT_NPZ):
    rec = compute_fd(npz_path)
    ok, var_checks, head = check_fd(rec, npz_path)
    if not ok:
        write_json(results_dir / "fd_figure_check.json", {"all_checks_pass": False, "variants": var_checks,
                                                           "headline": head})
        raise SystemExit("F_d check FAILED: recomputed values differ from fd_results.json")

    L = rec[0]["L"]
    d_show = L - 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]},
                                   layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.06)

    # ---- (a) free-riding index per block ------------------------------------------------
    dodge = {0.0: 0.0, 0.7: -0.07, 1.0: 0.07}
    blocks = np.arange(L)
    for r in rec:
        col, mk, lab = VARIANT_STYLE[r["gamma"]]
        x = blocks + dodge[r["gamma"]]
        ax1.errorbar(x, r["F_d"], yerr=r["F_d_ci95"], color=col, marker=mk, ms=4.2, lw=1.3,
                     capsize=1.8, elinewidth=0.8, label=lab, zorder=3,
                     mec="white", mew=0.5)
    g07 = next(r for r in rec if r["gamma"] == 0.7)
    for d in range(1, L):   # label the gamma=0.7 values (3 dp, so 0.955 is not rounded either way)
        ax1.annotate(f"{g07['F_d'][d]:.3f}", xy=(d + dodge[0.7], g07["F_d"][d]),
                     xytext=(3 if d == 1 else 0, -7.5), textcoords="offset points",
                     ha="left" if d == 1 else "center", va="top", fontsize=6.0, color=INK2)
    ax1.set_xticks(blocks)
    ax1.set_xlim(-0.35, L - 0.65)
    ax1.set_ylim(-0.04, 1.06)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xlabel(r"Block $d$")
    ax1.set_ylabel(r"$\mathcal{F}_d=\mathbb{E}\,[1-\min(1,R_d)]$")
    ax1.set_title("(a) Free-riding index per block", loc="left")
    ax1.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), frameon=False,
               handlelength=1.6, borderaxespad=0.2, labelspacing=0.35)

    # ---- (b) deepest-block gradient vs upstream margin ----------------------------------
    P_all = np.concatenate([r["_P"][d_show] for r in rec])
    x_lo, x_hi = np.floor(P_all.min()) - 0.3, np.ceil(P_all.max()) + 0.3
    P_grid = np.linspace(x_lo, x_hi, 400)
    ax2.axvspan(x_lo, 0.0, color="#f2f2f2", lw=0, zorder=0)
    order = [1.0, 0.7, 0.0]   # draw the lowest cloud first
    envelope_m = {}
    for g in order:
        r = next(q for q in rec if q["gamma"] == g)
        col, mk, _ = VARIANT_STYLE[g]
        P = r["_P"][d_show]
        grad = r["_grad"][d_show]
        ax2.scatter(P, grad, s=2.2, marker=mk, color=col, alpha=0.28, lw=0, zorder=2,
                    rasterized=False)
    for g in order:
        r = next(q for q in rec if q["gamma"] == g)
        col, _, _ = VARIANT_STYLE[g]
        m_med = float(np.median(r["_m"][d_show]))
        envelope_m[g] = m_med
        env = BETA / (1.0 + np.exp(BETA * (m_med + g * P_grid)))
        ax2.plot(P_grid, env, color="white", lw=2.6, zorder=4, solid_capstyle="round")
        ax2.plot(P_grid, env, color=col, lw=1.4, zorder=5, solid_capstyle="round")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 30)
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1])
    ax2.set_xlabel(r"Upstream margin $P^{(d-1)}=\sum_{j<d} m^{(j)}$")
    ax2.set_ylabel(r"$|\partial\ell_\beta/\partial m^{(d)}|$")
    ax2.set_title(f"(b) Gradient at the deepest block, $d={d_show}$", loc="left", fontsize=8.2)
    ax2.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    theory = Line2D([], [], color=INK, lw=1.4)
    pts = Line2D([], [], color=INK2, marker="o", ls="none", ms=2.5, alpha=0.6)
    outside = mpatches.Patch(facecolor="#f2f2f2", edgecolor="#cccccc", lw=0.5)
    ax2.legend([theory, pts, outside],
               [r"Theorem 3.1: $\beta\,\sigma(-\beta(\tilde m+\gamma P))$",
                r"Examples ($N=1024$ per $\gamma$)",
                r"$P^{(d-1)}<0$: outside the theorem"],
               loc="lower left", frameon=False, fontsize=6.0, handlelength=1.4,
               borderaxespad=0.15, labelspacing=0.3)
    save_pdf(fig, out_dir, "fig_grad_attenuation_fd.pdf")
    plt.close(fig)

    # ---- check JSON + caption facts ------------------------------------------------------
    cfgs = {g: json.load(open(p)) for g, p in GRAD_ATT_CONFIGS.items()}
    aux = {str(g): {k: cfgs[g].get(k) for k in ("gamma_scale", "block_curr_lambda", "block_curr_depth_slope",
                                                 "depth_order_lambda", "symba_alpha", "use_sam", "epochs_stage1",
                                                 "d_model", "num_blocks")}
           for g in cfgs}
    old_clip = {}
    for r in rec:
        gr = r["_grad"][d_show]
        old_clip[str(r["gamma"])] = {
            "frac_below_1e-4_pct": round(float((gr < 1e-4).mean() * 100), 1),
            "min_grad": float(f"{gr.min():.3e}"),
        }
    facts = {
        "setting": "CIFAR-10 L4/D128, seed 42, three Stage-1 checkpoints (stage1_best, raw training weights, not EMA); "
                   "first 1024 CIFAR-10 test images; one uniformly random wrong label per example as the negative "
                   "(torch seed 42); beta = 4.",
        "variants_differ_in_more_than_gamma": aux,
        "variants_note": "gamma=0.7 (baseline_constant) also trains with the depth-scaled current-block loss "
                         "(block_curr_lambda 0.25, slope 3 -> 0.25(1+d)) and depth-order 0.2; gamma=0 (no_all_collab) "
                         "and gamma=1.0 (lcff_prefix) have both set to 0. SAM is off in all three. So 'only gamma "
                         "differs' (submitted caption, D8) is false.",
        "normalization_caveat": {"snapshot_eval_std": SNAPSHOT_EVAL_STD, "trainer_eval_std": TRAINER_EVAL_STD,
                                 "note": "snapshot_grad_attenuation.py normalizes with a different std than training "
                                         "(H23); E3 recomputes on EMA weights with the trainer std."},
        "panel_a_F_d": {str(r["gamma"]): [round(float(x), 4) for x in r["F_d"]] for r in rec},
        "panel_a_F_d_ci95_halfwidth": {str(r["gamma"]): [round(float(x), 4) for x in r["F_d_ci95"]] for r in rec},
        "panel_a_ci_note": "normal-approximation 95% CI over the 1024 examples, 1.96*SD(ddof=0)/sqrt(N), "
                           "as in fd_figure.py",
        "pointwise_coverage_pct_m_ge0_and_P_ge0": {str(r["gamma"]): [round(float(x), 1)
                                                                      for x in r["assumption_coverage_pct"]]
                                                   for r in rec},
        "pointwise_coverage_range_pct": [round(float(min(x for r in rec for x in r["assumption_coverage_pct"])), 1),
                                         round(float(max(x for r in rec for x in r["assumption_coverage_pct"])), 1)],
        "median_R_d": {str(r["gamma"]): [float(f"{x:.3g}") for x in r["median_R_d"]] for r in rec},
        "frac_R_le_1e-2_pct": {str(r["gamma"]): [round(float(x), 1) for x in r["frac_fully_starved_pct"]]
                               for r in rec},
        "panel_b_block": d_show,
        "panel_b_envelope_median_m": {str(g): round(v, 4) for g, v in envelope_m.items()},
        "panel_b_envelope_note": "curve = beta*sigma(-beta*(m~ + gamma*P)) with m~ the per-variant MEDIAN "
                                 "m^(3) (numpy median); the submitted caption's \\bar m is a median, not a mean.",
        "panel_b_all_points_shown": "all 1024 examples per variant (the submitted Fig. 1(b) showed a random "
                                    "subsample of 400 and clipped the y-axis at 1e-4)",
        "submitted_fig1b_hidden_below_1e-4_at_d3": old_clip,
        "panel_b_P_range": [round(float(P_all.min()), 2), round(float(P_all.max()), 2)],
        "panel_b_frac_P_lt_0_pct": {str(r["gamma"]): round(float((r["_P"][d_show] < 0).mean() * 100), 1)
                                    for r in rec},
        "per_block_median_margin_m": {str(r["gamma"]): [round(float(x), 2) for x in r["m_median"]] for r in rec},
    }
    check = {
        "generated_by": "analysis/make_figures.py --fig grad_fd",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "figure": str(out_dir / "fig_grad_attenuation_fd.pdf"),
        "inputs": {
            "npz": {"path": str(npz_path), "sha256": sha256(npz_path)},
            "fd_results_json": {"path": str(FD_RESULTS), "sha256": sha256(FD_RESULTS)},
            "grad_attenuation_data_json": {"path": str(GRAD_ATT_SUMMARY), "sha256": sha256(GRAD_ATT_SUMMARY)},
            "configs": {str(g): str(p) for g, p in GRAD_ATT_CONFIGS.items()},
        },
        "beta": BETA, "n_examples": N_FD,
        "all_checks_pass": True,
        "headline": head,
        "variants": var_checks,
        "caption_facts": facts,
    }
    write_json(results_dir / "fd_figure_check.json", check)
    return check


# ============================================================================= margins (optional)
def fig_margins(out_dir: Path, npz_path: Path = GRAD_ATT_NPZ):
    rec = compute_fd(npz_path)
    L = rec[0]["L"]
    fig, ax = plt.subplots(figsize=(2.7, 2.0), layout="constrained")
    dodge = {0.0: -0.12, 0.7: 0.0, 1.0: 0.12}
    for r in rec:
        col, mk, lab = VARIANT_STYLE[r["gamma"]]
        x = np.arange(L) + dodge[r["gamma"]]
        med, lo, hi = r["m_median"], r["m_p25"], r["m_p75"]
        ax.errorbar(x, med, yerr=[med - lo, hi - med], color=col, marker=mk, ms=4, lw=1.2,
                    capsize=1.8, elinewidth=0.8, label=lab, mec="white", mew=0.5)
    ax.axhline(0, color="#999999", lw=0.6, ls=":")
    ax.set_xticks(range(L))
    ax.set_xlabel(r"Block $d$")
    ax.set_ylabel(r"Current-block margin $m^{(d)}$")
    ax.set_title("Median (IQR) margin per block", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_ylim(-1.45, 3.75)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
              fontsize=6.2, handlelength=1.5, borderaxespad=0.15, labelspacing=0.3)
    save_pdf(fig, out_dir, "fig_block_margin_cr.pdf")
    plt.close(fig)


# ============================================================================= dissociation summary (F)
def fig_dissociation_summary(out_dir: Path, results_dir: Path):
    """CIFAR-100 trio: deepest-block sep (x) vs. S2-TTA test top-1 (y), mean +- sample SD.

    Replaces the submitted fig_dissociation_summary.pdf, whose annotation read "4.97x ... 0.27 pp"
    (ratio and range of the 2-dp printed means).  Here both come from the unrounded 3-seed means in
    results/c100_trio.json (aggregate.py, raw events.jsonl / layer_metrics.jsonl): 5.0x and 0.28 pp.
    """
    trio = json.load(open(results_dir / "c100_trio.json"))
    V = trio["variants"]
    order = [  # key, direct label, color, marker; labels sit centered above each error bar
        ("cumulative", r"Cumulative ($\gamma{>}0$)", VERMILLION, "s"),
        ("gated_k0", r"Adaptive $\kappa{=}0$", GREEN, "^"),
        ("gamma0", r"$\gamma{=}0$ (history-free)", BLUE, "o"),
    ]
    pts = {}
    for key, *_ in order:
        a = V[key]["agg"]
        pts[key] = {"sep_mean": a["sep_curr_nl_deepest"]["mean"], "sep_sd": a["sep_curr_nl_deepest"]["sd"],
                    "acc_mean": a["S2-TTA"]["mean"], "acc_sd": a["S2-TTA"]["sd"],
                    "acc_values": a["S2-TTA"]["values"], "sep_values": a["sep_curr_nl_deepest"]["values"]}
    seps = [p["sep_mean"] for p in pts.values()]
    accs = [p["acc_mean"] for p in pts.values()]
    ratio, rng = max(seps) / min(seps), max(accs) - min(accs)
    assert abs(ratio - trio["deepest_sep_ratio"]["gamma0_over_cumulative"]) < 1e-6
    assert abs(rng - trio["range_across_trio_pp"]["S2-TTA"]) < 1e-6

    fig, ax = plt.subplots(figsize=(4.2, 2.1), layout="constrained")
    xs = sorted(pts, key=lambda k: pts[k]["sep_mean"])
    ax.plot([pts[k]["sep_mean"] for k in xs], [pts[k]["acc_mean"] for k in xs],
            color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    for key, lab, col, mk in order:
        p = pts[key]
        ax.errorbar(p["sep_mean"], p["acc_mean"], xerr=p["sep_sd"], yerr=p["acc_sd"], fmt=mk,
                    color=col, ms=6, capsize=2.5, elinewidth=1.0, mec="white", mew=0.8, zorder=3)
        ax.annotate(lab, (p["sep_mean"], p["acc_mean"] + p["acc_sd"]), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(0.02, 0.04, f"{ratio:.1f}$\\times$ range in block health $\\Rightarrow$ "
                        f"{rng:.2f} pp range in accuracy",
            transform=ax.transAxes, fontsize=7, color=INK2, ha="left", va="bottom")
    ax.set_xlim(0, 5.6)
    ax.set_ylim(68.2, 70.05)
    ax.set_xlabel(r"deepest-block $\mathrm{sep}^{\mathrm{cur}}_{\mathrm{nl}}$ (L3)"
                  r"  $\longrightarrow$ healthier blocks")
    ax.set_ylabel("S2 TTA test top-1 (%)")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    save_pdf(fig, out_dir, "fig_dissociation_summary.pdf")
    plt.close(fig)
    out = {
        "generated_by": "analysis/make_figures.py --fig dissociation_summary",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "figure": str(out_dir / "fig_dissociation_summary.pdf"),
        "source": str(results_dir / "c100_trio.json"),
        "protocol": "CIFAR-100 dense L4/D256, seeds 42/123/456; x = last-Stage-1-epoch deepest-block "
                    "sep_curr_nl, y = S2-TTA test top-1; mean +- sample SD (ddof=1)",
        "points": pts,
        "sep_ratio_max_over_min_unrounded": ratio,
        "s2_tta_range_pp_unrounded": rng,
        "replaces_annotation": "4.97x change in block health => 0.27 pp accuracy (2-dp printed means)",
    }
    write_json(results_dir / "fig_dissociation_summary_values.json", out)
    return out


# ============================================================================= accuracy vs ratio (C)
AVR_L4_RUNS = C.GA / "runs"
AVR_L8_RUNS = C.CF / "basic_fair_revised_valfix_cp_fair" / "ff_gated_ablation_L8" / "runs"
AVR_RUNS = {  # (depth, condition) -> run dir; the latest dir per condition, as in the submitted
    # analyze_additional_L8.py::find_run_dirs (prev_only L4: 20260315_042603; t1_k4 L8: 212730)
    (4, "baseline_constant"): "ff_cifar10_baseline_constant_L4_D128_bs512_seed42_20260313_232043",
    (4, "adaptive_t1_k0"): "ff_cifar10_adaptive_t1_k0_L4_D128_bs512_seed42_20260314_032410",
    (4, "adaptive_t3_k0"): "ff_cifar10_adaptive_t3_k0_L4_D128_bs512_seed42_20260314_071928",
    (4, "adaptive_t1_k2"): "ff_cifar10_adaptive_t1_k2_L4_D128_bs512_seed42_20260314_112422",
    (4, "adaptive_t1_k4"): "ff_cifar10_adaptive_t1_k4_L4_D128_bs512_seed42_20260314_152252",
    (4, "adaptive_prev_only"): "ff_cifar10_adaptive_prev_only_L4_D128_bs512_seed42_20260315_042603",
    (4, "adaptive_prev_t1_k2"): "ff_cifar10_adaptive_prev_t1_k2_L4_D128_bs512_seed42_20260315_044213",
    (4, "lcff_prefix"): "ff_cifar10_lcff_prefix_L4_D128_bs512_seed42_20260315_084228",
    (4, "no_all_collab"): "ff_cifar10_no_all_collab_L4_D128_bs512_seed42_20260315_124117",
    (8, "adaptive_t1_k0"): "ff_cifar10_adaptive_t1_k0_L8_L8_D128_bs512_seed42_20260315_201449",
    (8, "adaptive_t3_k0"): "ff_cifar10_adaptive_t3_k0_L8_L8_D128_bs512_seed42_20260316_033950",
    (8, "adaptive_t1_k2"): "ff_cifar10_adaptive_t1_k2_L8_L8_D128_bs512_seed42_20260316_105222",
    (8, "adaptive_t1_k4"): "ff_cifar10_adaptive_t1_k4_L8_L8_D128_bs512_seed42_20260316_212730",
    (8, "adaptive_prev_only"): "ff_cifar10_adaptive_prev_only_L8_L8_D128_bs512_seed42_20260317_031838",
    (8, "adaptive_prev_t1_k2"): "ff_cifar10_adaptive_prev_t1_k2_L8_L8_D128_bs512_seed42_20260317_105615",
}
AVR_LABELS = {
    "baseline_constant": r"const. $\gamma{=}0.7$", "adaptive_t1_k0": r"$\tau1,\kappa0$",
    "adaptive_t3_k0": r"$\tau3,\kappa0$", "adaptive_t1_k2": r"$\tau1,\kappa2$",
    "adaptive_t1_k4": r"$\tau1,\kappa4$", "adaptive_prev_only": r"prev $\kappa0$",
    "adaptive_prev_t1_k2": r"prev $\kappa2$", "lcff_prefix": "LCFF", "no_all_collab": r"$\gamma{=}0$",
}
AVR_OFFSETS = {  # label offsets in points (dx, dy, ha, va), set by eye to avoid collisions
    (4, "adaptive_prev_t1_k2"): (6, 0, "left", "center"),
    (4, "adaptive_t1_k4"): (6, 2, "left", "bottom"),
    (4, "baseline_constant"): (0, -6, "center", "top"),
    (4, "lcff_prefix"): (6, 0, "left", "center"),
    (4, "adaptive_t1_k2"): (6, 0, "left", "center"),
    (4, "no_all_collab"): (-6, 0, "right", "center"),
    (4, "adaptive_t3_k0"): (6, 0, "left", "center"),
    (4, "adaptive_prev_only"): (6, 1, "left", "bottom"),
    (4, "adaptive_t1_k0"): (6, -1, "left", "top"),
    (8, "adaptive_prev_t1_k2"): (6, 0, "left", "center"),
    (8, "adaptive_prev_only"): (0, -6, "center", "top"),
    (8, "adaptive_t3_k0"): (-6, 0, "right", "center"),
    (8, "adaptive_t1_k4"): (6, 0, "left", "center"),
    (8, "adaptive_t1_k0"): (-6, 0, "right", "center"),
    (8, "adaptive_t1_k2"): (-6, 0, "right", "center"),
}


def fig_accuracy_vs_ratio(out_dir: Path, results_dir: Path):
    """Final-epoch S2 validation accuracy vs. deepest/Block-0 g+_cur ratio, 15 L4+L8 D128 runs.

    Port of plot_accuracy_vs_ratio() in basic_fair_revised_valfix_cp_fair/from_colab/
    basic_fair_revised_valfix_cp_fair_hardness_gated/analyze_additional_L8.py (same runs, same
    quantities: last-Stage-1-epoch gpos_cur ratio; last Stage-2 epoch eval_top1 from
    epoch_metrics.jsonl, i.e. the 5k validation split).  The submitted y-axis label read
    "S2 Test Accuracy"; the values are validation accuracy, and the label now says so.
    """
    pts = []
    for (L, cond), name in AVR_RUNS.items():
        run = (AVR_L4_RUNS if L == 4 else AVR_L8_RUNS) / name
        ep, lay = last_stage1_layers(run)
        g0, gd = lay[0]["gpos_cur"], lay[L - 1]["gpos_cur"]
        s2 = [e for e in load_jsonl(run / "epoch_metrics.jsonl") if e.get("stage") == 2 and e.get("eval_top1")]
        best_s2 = [e.get("best_s2") for e in load_jsonl(run / "events.jsonl") if e.get("best_s2") is not None] \
            if (run / "events.jsonl").exists() else []
        pts.append({"L": L, "condition": cond, "run": str(run), "stage1_epoch": ep,
                    "gpos_cur_L0": g0, "gpos_cur_deepest": gd, "ratio": gd / max(g0, 0.01),
                    "s2_val_final_epoch_pct": s2[-1]["eval_top1"] * 100, "s2_final_epoch": s2[-1].get("epoch"),
                    "s2_val_max_over_epochs_pct": max(e["eval_top1"] for e in s2) * 100,
                    "best_s2_logged_pct": best_s2[-1] * 100 if best_s2 else None})
    x = np.array([p["ratio"] for p in pts])
    y = np.array([p["s2_val_final_epoch_pct"] for p in pts])
    r = float(np.corrcoef(x, y)[0, 1])
    slope, icpt = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(5.5, 3.1), layout="constrained")
    ax.axvline(1.0, color="#999999", lw=0.6, ls=":", zorder=1)
    xl = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
    ax.plot(xl, slope * xl + icpt, color="#aaaaaa", lw=0.9, ls="--", zorder=1)
    for L, col, mk, lab in [(4, BLUE, "o", r"$L{=}4$"), (8, VERMILLION, "^", r"$L{=}8$")]:
        sel = [p for p in pts if p["L"] == L]
        ax.scatter([p["ratio"] for p in sel], [p["s2_val_final_epoch_pct"] for p in sel], s=34, marker=mk,
                   color=col, edgecolors="white", linewidths=0.7, zorder=3, label=lab)
        for p in sel:
            dx, dy, ha, va = AVR_OFFSETS[(L, p["condition"])]
            ax.annotate(AVR_LABELS[p["condition"]], (p["ratio"], p["s2_val_final_epoch_pct"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=6, color=INK2)
    ax.text(0.015, 0.03, f"Pearson $r = {r:.2f}$ ($n={len(pts)}$)", transform=ax.transAxes,
            fontsize=7, color=INK, ha="left", va="bottom")
    ax.set_xlabel(r"deepest-block / Block-0 ratio of $g^{+}_{\mathrm{cur}}$ (last Stage-1 epoch)")
    ax.set_ylabel("S2 validation accuracy (%)\n(final epoch, 5k split)")
    ax.set_xlim(-0.05, 4.1)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
              fontsize=6.8, handletextpad=0.3, borderaxespad=0.3)
    save_pdf(fig, out_dir, "fig_accuracy_vs_ratio.pdf")
    plt.close(fig)
    out = {
        "generated_by": "analysis/make_figures.py --fig accuracy_vs_ratio",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "figure": str(out_dir / "fig_accuracy_vs_ratio.pdf"),
        "ported_from": str(C.CF / "basic_fair_revised_valfix_cp_fair" / "from_colab"
                           / "basic_fair_revised_valfix_cp_fair_hardness_gated" / "analyze_additional_L8.py")
                       + " :: plot_accuracy_vs_ratio",
        "y": "final-epoch Stage-2 eval_top1 (5k validation split) from epoch_metrics.jsonl",
        "x": "gpos_cur(deepest block) / max(gpos_cur(block 0), 0.01) at the last Stage-1 epoch",
        "pearson_r": r, "n": len(pts), "ols_slope": float(slope), "ols_intercept": float(icpt),
        "points": pts,
    }
    write_json(results_dir / "fig_accuracy_vs_ratio_values.json", out)
    return out


# ============================================================================= fig3_4 (C6)
def last_stage1_layers(run_dir: Path):
    rows = [r for r in load_jsonl(run_dir / "layer_metrics.jsonl") if r.get("stage") == 1]
    last = max(r["epoch"] for r in rows)
    return last, {r["layer"]: r for r in rows if r["epoch"] == last}


def final_test_eval(run_dir: Path):
    ev = [e for e in load_jsonl(run_dir / "events.jsonl") if e.get("event") == "final_test_eval"]
    assert len(ev) == 1, f"expected one final_test_eval in {run_dir}"
    return ev[0]


def collect_bpff_values():
    """Every number of fig3_4, read from the raw per-run files."""
    vals = {"panel_a": {}, "panel_b": []}
    # ---- (a) current-block loss at the last logged Stage-1 epoch ----
    for key, rd in LOSS_RUNS.items():
        ep, layers = last_stage1_layers(rd)
        cfg = json.load(open(rd / "config.json"))
        lam, slope, L = cfg["block_curr_lambda"], cfg.get("block_curr_depth_slope", 0.0), cfg["num_blocks"]
        vals["panel_a"][key] = {
            "run_dir": str(rd),
            "last_logged_stage1_epoch": ep,
            "epochs_stage1_config": cfg["epochs_stage1"],
            "loss_block_curr": [float(layers[d]["loss_block_curr"]) for d in range(L)],
            "lambda_curr_per_block": [lam * (1.0 + slope * d / max(1, L - 1)) for d in range(L)],
            "gamma_scale": cfg["gamma_scale"], "depth_order_lambda": cfg["depth_order_lambda"],
            "residual_weighted_loss_logged": key == "cp_fair",
        }
    fin = final_test_eval(BASELINE_FINISH_RUN)
    ep_fin, _ = last_stage1_layers(BASELINE_FINISH_RUN)
    vals["panel_a"]["baseline"]["finished_by"] = {
        "run_dir": str(BASELINE_FINISH_RUN), "stage1_epoch": ep_fin,
        "stage1_test_top1_tta": fin["stage1_test_top1_tta"],
        "note": "the plotted run's log ends at epoch 286 (no final_test_eval); a load-last continuation "
                "ran epoch 287 and the test evaluation (config epochs_stage1 = 287)",
    }

    # ---- (b) single-crop (no-TTA) test top-1 ----
    def bp_row(key, label_arch, label_train, aug, color):
        per = {}
        for s, rd in C.BP_RUNS[key].items():
            fm = json.load(open(rd / "final_metrics.json"))
            assert fm.get("eval_tta") in (None, False), rd
            per[s] = round(100.0 * fm["test_top1"], 2)
        mu, sd = sample_mean_sd(list(per.values()))
        return {"key": key, "arch": label_arch, "training": label_train, "aug": aug, "color": color,
                "n": len(per), "per_seed": per, "mean": mu, "sd_ddof1": sd,
                "run_dirs": {s: str(p) for s, p in C.BP_RUNS[key].items()}}

    fm_ff = json.load(open(FF_STRIPPED_RUN / "final_metrics.json"))
    cfg_ff = json.load(open(FF_STRIPPED_RUN / "config.json"))
    assert fm_ff["eval_tta"] is False
    ev_cp = final_test_eval(CP_FAIR_RUN)
    vals["panel_b"] = [
        bp_row("plain_cnn_bp", "Plain CNN", "BP", "strong", "BP"),
        bp_row("plain_cnn_ff", "Plain CNN", "FF", "FF-pipeline", "FF"),
        bp_row("bp_weak", "Stripped FF backbone", "BP", "weak", "BP"),
        bp_row("bp_strong", "Stripped FF backbone", "BP", "BP-strong", "BP"),
        {"key": "ff_stripped", "arch": "Stripped FF backbone", "training": "FF", "aug": "FF-pipeline",
         "color": "FF", "n": 1, "per_seed": {42: round(100.0 * fm_ff["test_top1"], 2)},
         "mean": round(100.0 * fm_ff["test_top1"], 2), "sd_ddof1": None,
         "run_dirs": {42: str(FF_STRIPPED_RUN)},
         "note": f"gamma_scale {cfg_ff['gamma_scale']}, block_curr_lambda {cfg_ff['block_curr_lambda']} with "
                 f"depth slope {cfg_ff['block_curr_depth_slope']} (already depth-scaled); no MoE, memory, "
                 f"contrastive or reconstruction; random wrong-label negatives"},
        {"key": "cp_fair", "arch": "Full FF backbone", "training": "FF (CP-FAIR)", "aug": "FF-pipeline",
         "color": "FF", "n": 1, "per_seed": {42: round(100.0 * ev_cp["stage1_test_top1_no_tta"], 2)},
         "mean": round(100.0 * ev_cp["stage1_test_top1_no_tta"], 2), "sd_ddof1": None,
         "run_dirs": {42: str(CP_FAIR_RUN)},
         "note": "stage1_test_top1_no_tta from events.jsonl final_test_eval (S1, single crop)"},
    ]
    return vals


def fig_loss_bpff(out_dir: Path, results_dir: Path):
    vals = collect_bpff_values()
    # guard the numbers the paper prints (Table 2 / C6); fail loudly if a source moved
    expect = {"plain_cnn_bp": (90.08, 0.28), "plain_cnn_ff": (29.85, 0.50), "bp_weak": (87.43, 0.62),
              "bp_strong": (93.85, 0.18), "ff_stripped": (89.03, None), "cp_fair": (90.45, None)}
    for row in vals["panel_b"]:
        mu, sd = expect[row["key"]]
        assert round(row["mean"], 2) == mu, (row["key"], row["mean"])
        if sd is not None:
            assert round(row["sd_ddof1"], 2) == sd, (row["key"], row["sd_ddof1"])

    col_a = {"baseline": GREEN, "cp_fair": VERMILLION}
    col_b = {"BP": BLUE, "FF": VERMILLION}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3),
                                   gridspec_kw={"width_ratios": [1.0, 1.55]}, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.04)

    # ---- (a) current-block loss across depth ----
    pa = vals["panel_a"]
    L = len(pa["cp_fair"]["loss_block_curr"])
    x = np.arange(L)
    w = 0.36
    labels_a = {
        "baseline": r"Baseline: $\lambda_{\mathrm{curr}}=0.5$",
        "cp_fair": r"CP-FAIR: $\lambda^{(d)}_{\mathrm{curr}}=0.25\,(1+d)$",
    }
    for i, key in enumerate(("baseline", "cp_fair")):
        ax1.bar(x + (i - 0.5) * w, pa[key]["loss_block_curr"], w * 0.92, color=col_a[key],
                label=labels_a[key], lw=0, zorder=3)
    top = max(max(pa[k]["loss_block_curr"]) for k in pa)
    ax1.set_ylim(0, top * 1.38)
    ax1.set_xticks(x)
    ax1.set_xlabel(r"Block $d$")
    ax1.set_ylabel(r"Current-block loss $\mathcal{L}^{(d)}_{\mathrm{curr}}$")
    ax1.set_title("(a) Current-block loss across depth", loc="left")
    ax1.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax1.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.2, borderaxespad=0.1,
               labelspacing=0.3)

    # ---- (b) BP vs FF calibration (single-crop test top-1), horizontal bars ----
    rows = vals["panel_b"]
    group_of = {"plain_cnn_bp": 0, "plain_cnn_ff": 0, "bp_weak": 1, "bp_strong": 1, "ff_stripped": 1,
                "cp_fair": 2}
    headers = ["Plain CNN", "Stripped FF backbone", "Full FF backbone"]
    row_label = {"plain_cnn_bp": "BP", "plain_cnn_ff": "FF", "bp_weak": "BP, weak aug.",
                 "bp_strong": "BP, strong aug.", "ff_stripped": "FF", "cp_fair": "FF (CP-FAIR)"}
    ypos, header_y, y, cur = [], {}, 0.0, None
    for r in rows:
        gidx = group_of[r["key"]]
        if gidx != cur:
            header_y[gidx] = y
            y += 1.0
            cur = gidx
        ypos.append(y)
        y += 1.0
    bar_h = 0.66
    for yi, r in zip(ypos, rows):
        ax2.barh(yi, r["mean"], bar_h, color=col_b[r["color"]], lw=0, zorder=3)
        if r["n"] > 1:
            ax2.errorbar(r["mean"], yi, xerr=r["sd_ddof1"], color=INK, capsize=1.8, elinewidth=0.7,
                         capthick=0.7, lw=0, zorder=4)
            txt = f"{r['mean']:.2f} $\\pm$ {r['sd_ddof1']:.2f}"
        else:
            txt = f"{r['mean']:.2f}  ($n=1$)"
        ax2.text(r["mean"] + (r["sd_ddof1"] or 0.0) + 1.6, yi, txt, ha="left", va="center",
                 fontsize=6.0, color=INK, zorder=5)
    for gidx, hy in header_y.items():
        ax2.text(0.8, hy, headers[gidx], ha="left", va="center", fontsize=6.3, color=INK2,
                 style="italic")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([row_label[r["key"]] for r in rows], fontsize=6.4)
    ax2.tick_params(axis="y", length=0, pad=2)
    ax2.set_ylim(y - 0.4, -0.6)
    ax2.set_xlim(0, 132)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.spines["bottom"].set_bounds(0, 100)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("Test top-1, single crop (%)")
    ax2.set_title("(b) BP vs. FF calibration (CIFAR-10)", loc="left")
    ax2.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    handles = [mpatches.Patch(color=BLUE, label="BP training"),
               mpatches.Patch(color=VERMILLION, label="FF training")]
    ax2.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 1.0 - (header_y[0] + 0.6) / (y + 0.2)),
               ncol=2, frameon=False, fontsize=6.0, handlelength=1.0, columnspacing=0.8,
               borderaxespad=0.0, handletextpad=0.4)
    save_pdf(fig, out_dir, "fig3_4_loss_bpff_cr.pdf")
    plt.close(fig)

    out = {
        "generated_by": "analysis/make_figures.py --fig loss_bpff",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "figure": str(out_dir / "fig3_4_loss_bpff_cr.pdf"),
        "ported_from": str(C.TEX / "supplement_code" / "analysis" / "generate_figures.py")
                       + " :: fig3_4_loss_and_bpff (lines 288-394)",
        "protocol_panel_b": "single-crop (no TTA) test top-1 on the 10k CIFAR-10 test set; mean +- sample SD "
                            "(ddof=1) over seeds 42/123/456 where n=3",
        "panel_a": vals["panel_a"],
        "panel_b": [{k: v for k, v in r.items() if k != "color"} | {"training_color": r["color"]}
                    for r in vals["panel_b"]],
        "removed_annotations": ["+2.1% (Ours+FF 89.03 vs weak-aug BP 86.90)",
                                "+1.4% (Ours+FF+depth 90.45 vs Ours+FF 89.03)"],
    }
    write_json(results_dir / "fig3_4_bpff_values.json", out)
    return out


# ============================================================================= E2 early exit (C14)
EE_FAMILY_STYLE = {  # objective family -> (color, marker, legend label); same hues as the trio figures
    "cumulative": (VERMILLION, "s", r"cumulative ($\gamma{=}0.7$; C10 also LCFF $\gamma{=}1.0$)"),
    "gamma0": (BLUE, "o", r"repaired: history-free $\gamma{=}0$"),
    "gated": (GREEN, "^", r"repaired: hardness-gated"),
}
EE_RULES = (("T_iso_val", "T: shallowest depth with\nval. acc. $\\geq$ full depth"),
            ("T_argmax_val", "T$'$: val.-best depth"),
            ("D_dynamic_exit", "D: dynamic exit\n(val.-chosen margin)"))


def _dodge(points, xr, yr, wx=0.046, wy=0.075, step=0.018):
    """Horizontal offsets (data units) that keep markers from overlapping. A marker spans about wx of the
    axis width and wy of its height (xr, yr = axis ranges); points are placed in family order (cumulative,
    gamma0, gated), then by y, and each moves only if it would overlap an already placed marker, by the
    smallest multiple of step*xr that clears it (cumulative tries left first, gated right first)."""
    order = {"cumulative": 0, "gamma0": 1, "gated": 2}
    first = {"cumulative": -1, "gamma0": 1, "gated": 1}
    off, placed = [0.0] * len(points), []
    for i in sorted(range(len(points)), key=lambda i: (order[points[i]["family"]], points[i]["y"])):
        x, y, sgn = points[i]["x"], points[i]["y"], first[points[i]["family"]]
        for k in [0] + [m * s for m in range(1, 60) for s in (sgn, -sgn)]:
            xo = x + k * step * xr
            if all(abs(xo - px) >= wx * xr or abs(y - py) >= wy * yr for px, py in placed):
                off[i] = k * step * xr
                placed.append((xo, y))
                break
    return off


def fig_early_exit_pareto(out_dir: Path, results_dir: Path):
    """Validation-selected early exit (E2): test delta vs FLOPs saved, per run, both families.

    Rows: CIFAR-10 L4/D128 (9 arms, seed 42) and the CIFAR-100 L4/D256 trio (3 variants x 3 seeds);
    columns: the three validation-selection rules. Color + marker = objective family, so the reader can
    see whether repaired arms exit earlier than cumulative ones. All values come from results/e2_exit.json
    via analysis/early_exit.py; the derived summaries (incl. the paired CIFAR-100 comparisons) are written
    to results/fig_early_exit_pareto_values.json.
    """
    import early_exit as E
    e2 = E.load(results_dir)
    S = E.summarize(e2)
    R = e2["runs"]
    rows = (("c10", "CIFAR-10 (L4/D128)", S["c10_all"]["members"], 0.2),
            ("c100", "CIFAR-100 (L4/D256)", S["c100_all"]["members"], 0.5))
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 3.2), layout="constrained", sharey="row")
    plotted = {}
    for i, (ds, row_label, names, ystep) in enumerate(rows):
        dep = S["flops_saved_pct_by_depth"][ds]            # depth 1..4 -> % saved
        ys = [E.op(R[n], rule)["delta_pp"] for n in names for rule, _ in EE_RULES]
        pad = 0.12 * (max(ys) - min(ys))
        for j, (rule, title) in enumerate(EE_RULES):
            ax = axes[i, j]
            pts = [dict(name=n, family=E.family_of(n), x=E.op(R[n], rule)["saved_pct"], y=E.op(R[n], rule)["delta_pp"])
                   for n in names]
            # Truncation picks one of four depths, so exact ties are dodged into family slots. Dynamic exit
            # saves 64-75% in every run, so its column is zoomed (x from 61 to 79) and dodged at that scale.
            zoom = rule == "D_dynamic_exit"
            xlim = (61.0, 79.0) if zoom else (-8.0, 84.0)
            off = _dodge(pts, xlim[1] - xlim[0], (max(ys) - min(ys)) + 2 * pad)
            for d, x in zip((1, 2, 3, 4), dep):
                if not xlim[0] < x < xlim[1]:
                    continue
                ax.axvline(x, color=GRID, lw=0.6, zorder=0)
                ax.text(x, 1.0, f"$d{{=}}{d}$", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=5.8, color=INK2)
            ax.axhline(0.0, color="#9a9a9a", lw=0.7, zorder=1)
            for fam in ("cumulative", "gamma0", "gated"):
                sel = [k for k, p in enumerate(pts) if p["family"] == fam]
                col, mk, _ = EE_FAMILY_STYLE[fam]
                ax.scatter([pts[k]["x"] + off[k] for k in sel], [pts[k]["y"] for k in sel], s=24, marker=mk,
                           color=col, edgecolors="white", linewidths=0.6, zorder=3 if fam == "cumulative" else 2)
            ax.set_xlim(*xlim)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ystep))
            if zoom:
                ax.set_xticks([65, 70, 75])
                ax.text(0.03, 0.03, "x-axis zoomed", transform=ax.transAxes, fontsize=5.8, color=INK2,
                        ha="left", va="bottom")
            else:
                ax.set_xticks([dep[3], dep[2], dep[1], dep[0]])
                ax.set_xticklabels([f"{x:.0f}" for x in (dep[3], dep[2], dep[1], dep[0])])
            if i == 0:
                ax.set_title(title, fontsize=7.2, pad=10)
            if j == 0:
                ax.set_ylabel(f"{row_label}\ntest $\\Delta$ (pp)")
            plotted[f"{ds}/{rule}"] = [dict(p, x_plotted=p["x"] + o) for p, o in zip(pts, off)]
    fig.supxlabel("FLOPs saved vs. full depth (%)", fontsize=8)
    handles = [Line2D([], [], ls="", marker=mk, color=col, mec="white", mew=0.6, ms=5.8, label=lab)
               for col, mk, lab in EE_FAMILY_STYLE.values()]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=6.6,
               handletextpad=0.3, columnspacing=1.2)
    save_pdf(fig, out_dir, "fig_early_exit_pareto.pdf")
    plt.close(fig)
    out = {
        "generated_by": "analysis/make_figures.py --fig early_exit (summaries: analysis/early_exit.py)",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "figure": str(out_dir / "fig_early_exit_pareto.pdf"),
        "source": str(results_dir / "e2_exit.json"),
        "source_sha256": sha256(results_dir / "e2_exit.json"),
        "protocol": e2["_header"]["description"] + " Selection on each run's own 5k validation split; one test "
                    "evaluation; delta = selected-point minus full-depth test accuracy of the same checkpoint.",
        "units": "saved_pct / exit_acc_pct / full_acc_pct in %, delta_pp and paired differences in pp; "
                 "SD = sample SD (ddof=1)",
        "families": {"cumulative": "constant-gamma cumulative goodness (C10 gamma=0.7 and LCFF gamma=1.0; C100 gamma=0.7)",
                     "gamma0": "block-local repair (gamma=0)", "gated": "hardness-gated repair",
                     "repaired": "gamma0 + gated"},
        "plot_note": "overlapping markers are offset horizontally (x_plotted - x) for visibility only",
        "summary": S,
        "points": plotted,
    }
    write_json(results_dir / "fig_early_exit_pareto_values.json", out)
    return out


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fig", default="all",
                    choices=["all", "grad_fd", "loss_bpff", "margins", "dissociation_summary",
                             "accuracy_vs_ratio", "early_exit"])
    ap.add_argument("--output-dir", type=Path, default=C.CR / "figures")
    ap.add_argument("--results-dir", type=Path, default=C.RESULTS)
    ap.add_argument("--fd-npz", type=Path, default=GRAD_ATT_NPZ,
                    help="per-example arrays (default: the published snapshot npz)")
    a = ap.parse_args()
    out_dir = a.output_dir.resolve()
    # never write into the (read-only) run-log archive
    if str(C.HOME) not in ("", ".") and str(out_dir).startswith(str(C.HOME.resolve())):
        raise SystemExit(f"refusing to write inside FF_HOME: {out_dir}")
    setup_style()
    if a.fig in ("all", "grad_fd"):
        print("Figure: fig_grad_attenuation_fd")
        if a.fd_npz != GRAD_ATT_NPZ:
            raise SystemExit("custom --fd-npz: re-point the published-value checks first (check_fd)")
        fig_grad_fd(out_dir, a.results_dir, a.fd_npz)
    if a.fig in ("all", "loss_bpff"):
        print("Figure: fig3_4_loss_bpff_cr")
        fig_loss_bpff(out_dir, a.results_dir)
    if a.fig in ("all", "margins"):
        print("Figure: fig_block_margin_cr")
        fig_margins(out_dir, a.fd_npz)
    if a.fig in ("all", "dissociation_summary"):
        print("Figure: fig_dissociation_summary")
        fig_dissociation_summary(out_dir, a.results_dir)
    if a.fig in ("all", "accuracy_vs_ratio"):
        print("Figure: fig_accuracy_vs_ratio")
        fig_accuracy_vs_ratio(out_dir, a.results_dir)
    if a.fig in ("all", "early_exit"):
        print("Figure: fig_early_exit_pareto")
        fig_early_exit_pareto(out_dir, a.results_dir)


if __name__ == "__main__":
    main()
