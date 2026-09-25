#!/usr/bin/env python3
"""E3: F_d, pointwise coverage and attenuation statistics, plus comparison with the published numbers.

The statistics are the same as ff-wiki/_build/analyses/fd_figure.py (the source of fd_results.json):
  R = (1 + e^{beta m}) / (1 + e^{beta (m + gamma P)}),  beta = 4,
  F_d = mean(1 - min(1, R)), CI95 = 1.96 * sd(ddof=0) / sqrt(N)   [the published convention],
  pointwise coverage = mean(m >= 0 and P >= 0), median R, R at batch means, frac(R <= 1e-2).

Gates:
  G1: this code applied to the published arrays reproduces fd_results.json.
  G2: the raw-weights, legacy-std run with the recovered published wrong labels reproduces the published
      F_d to within 0.01 per block.
  G3: the EMA accuracy gate passed for all three checkpoints; the strict net_ema load had 0 missing and 0 unexpected keys.
Output: CR/results/e3_fd_ema.json
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HOME = FF_HOME
CF = f"{HOME}/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
CR = FF_OUT  # [CR] was the camera-ready repository; outputs go to <FF_OUT>/results
HERE = Path(__file__).resolve().parent
PUB_NPZ = f"{CF}/revised_training/grad_attenuation/grad_attenuation_arrays.npz"
PUB_JSON = f"{HOME}/amir-porjects/ff-wiki/_build/analyses/fd_results.json"
OUT = f"{CR}/results/e3_fd_ema.json"
BETA = 4.0
MODES = ["ema_fixed_published", "ema_fixed_seed", "raw_legacy_published", "raw_legacy_seed",
         "raw_fixed_published", "ema_legacy_published"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def log_R(m, gP):
    return np.logaddexp(0, BETA * m) - np.logaddexp(0, BETA * (m + gP))


def stats(m, P, g, rounded=False):
    m, P = m.astype(np.float64), P.astype(np.float64)
    R = np.exp(log_R(m, g * P))
    one_minus = 1 - np.minimum(1, R)
    F = one_minus.mean(axis=1)
    F_ci = 1.96 * one_minus.std(axis=1) / np.sqrt(R.shape[1])
    R_at_means = np.exp(log_R(m.mean(axis=1), g * P.mean(axis=1)))
    cover = ((m >= 0) & (P >= 0)).mean(axis=1)
    out = {
        "F_d": F.tolist(), "F_d_ci95": F_ci.tolist(),
        "mean_R_d": R.mean(axis=1).tolist(), "median_R_d": np.median(R, axis=1).tolist(),
        "R_at_batch_means (paper statistic)": R_at_means.tolist(),
        "frac_fully_starved_R_le_1e-2_pct": [float((R[d] <= 1e-2).mean() * 100) for d in range(R.shape[0])],
        "assumption_coverage_pct": (cover * 100).tolist(),
        "m_mean": m.mean(axis=1).tolist(), "P_prev_mean": P.mean(axis=1).tolist(),
        "n_examples": int(m.shape[1]),
    }
    if rounded:  # the published rounding, for gate G1
        out = {
            "F_d": [round(float(x), 4) for x in F], "F_d_ci95": [round(float(x), 4) for x in F_ci],
            "mean_R_d": [float(f"{x:.3e}") for x in R.mean(axis=1)],
            "median_R_d": [float(f"{x:.3e}") for x in np.median(R, axis=1)],
            "R_at_batch_means (paper statistic)": [float(f"{x:.3e}") for x in R_at_means],
            "frac_fully_starved_R_le_1e-2_pct": [round(float((R[d] <= 1e-2).mean() * 100), 1) for d in range(4)],
            "assumption_coverage_pct": [round(float(x) * 100, 1) for x in cover],
        }
    return out, R


def load_mode(mode):
    d = HERE / "out" / mode
    res = []
    for i in range(3):
        z = np.load(d / f"grad_attenuation_arrays_v{i}.npz")
        j = json.load(open(d / f"grad_attenuation_data_v{i}.json"))
        res.append({"m": z["m"], "P_prev": z["P_prev"], "gamma": float(z["gamma"]), "tag": str(z["tag"]),
                    "meta": j["meta"], "ckpt_meta": j["variant"]["ckpt_meta"], "acc_gate": j["variant"]["acc_gate"],
                    "npz_sha256": sha256(d / f"grad_attenuation_arrays_v{i}.npz")})
    return res


def main():
    pub = np.load(PUB_NPZ, allow_pickle=True)
    pub_json = json.load(open(PUB_JSON))
    gates = []
    # G1: reproduce fd_results.json from the published arrays
    pub_stats, pub_R = [], []
    for v in range(3):
        g = float(pub["gammas"][v])
        s_r, _ = stats(pub[f"{v}_m"], pub[f"{v}_P_prev"], g, rounded=True)
        s, R = stats(pub[f"{v}_m"], pub[f"{v}_P_prev"], g)
        pub_stats.append(s)
        pub_R.append(R)
        want = pub_json["variants"][v]
        ok = all(s_r[k] == want[k] for k in s_r)
        gates.append({"check": f"G1 fd_results.json reproduced from published arrays, variant {v}", "ok": ok})

    modes = {m: load_mode(m) for m in MODES if (HERE / "out" / m).exists()}
    mode_stats = {}
    for mode, res in modes.items():
        mode_stats[mode] = []
        for v, r in enumerate(res):
            s, R = stats(r["m"], r["P_prev"], r["gamma"])
            mode_stats[mode].append({"tag": r["tag"], "gamma": r["gamma"], **s, "_R": R, "_m": r["m"], "_P": r["P_prev"]})

    # G2: raw/legacy/published-labels reproduces the published F_d within 0.01 per block
    for v in range(3):
        a = np.array(mode_stats["raw_legacy_published"][v]["F_d"])
        b = np.array(pub_stats[v]["F_d"])
        gates.append({"check": f"G2 raw/legacy rerun vs published F_d, variant {v}", "max_abs_diff": float(np.abs(a - b).max()),
                      "ok": bool(np.abs(a - b).max() < 0.01)})
    # G3: EMA accuracy gate and strict load
    for v, r in enumerate(modes["ema_fixed_published"]):
        ag = r["acc_gate"]
        gates.append({"check": f"G3 EMA S1 no-TTA test accuracy, variant {v}", "got": ag["acc_s1_no_tta"],
                      "logged": ag["logged_stage1_test_top1_no_tta"], "task_anchor": ag["task_anchor"],
                      "n_disagree_with_logged_preds": ag["n_disagree_with_logged_preds_npz"],
                      "ok": bool(ag["passed"] and not r["ckpt_meta"]["missing"] and not r["ckpt_meta"]["unexpected"]
                                 and r["ckpt_meta"]["state_dict_key"] == "net_ema")})
    rec = json.load(open(HERE / "out" / "recover_published_neg_labels.json"))
    gates.append({"check": "G0 published wrong labels recovered (relaxed gate; strict per-example 2e-3 match failed for 4.5-6.7% of examples)",
                  "ok": bool(rec["passed_relaxed"]), "strict_passed": rec["passed_strict_all_within_tol"],
                  "frac_all_blocks_within_2e-3": [x["frac_all_blocks_within_tol"] for x in rec["variants"]],
                  "per_block_median_err": [x["per_block_median_err"] for x in rec["variants"]]})
    all_ok = all(g["ok"] for g in gates)

    # paired per-example comparison: EMA/fixed vs published raw/legacy (same images, same wrong labels)
    paired = []
    for v in range(3):
        e = mode_stats["ema_fixed_published"][v]
        Re, Rp = e["_R"], pub_R[v]
        Fe, Fp = 1 - np.minimum(1, Re), 1 - np.minimum(1, Rp)
        dF = Fe - Fp
        rec_v = {"tag": e["tag"], "gamma": e["gamma"],
                 "delta_F_d_ema_minus_published": dF.mean(axis=1).tolist(),
                 "delta_F_d_ci95_paired": (1.96 * dF.std(axis=1, ddof=1) / np.sqrt(dF.shape[1])).tolist(),
                 "delta_coverage_pct": (np.array(e["assumption_coverage_pct"]) - np.array(pub_stats[v]["assumption_coverage_pct"])).tolist(),
                 "starved_status_agreement_pct": [float(((Re[d] <= 1e-2) == (Rp[d] <= 1e-2)).mean() * 100) for d in range(4)],
                 "spearman_m_ema_vs_published": []}
        from scipy.stats import spearmanr
        for d in range(4):
            rec_v["spearman_m_ema_vs_published"].append(float(spearmanr(e["_m"][d], pub[f"{v}_m"][d]).statistic))
        paired.append(rec_v)

    # attribution: which change moves F_d (weights vs normalization), at the published labels
    attribution = {}
    for mode in ["raw_legacy_published", "raw_fixed_published", "ema_legacy_published", "ema_fixed_published"]:
        if mode in mode_stats:
            attribution[mode] = [{"tag": s["tag"], "F_d": s["F_d"], "assumption_coverage_pct": s["assumption_coverage_pct"],
                                  "median_R_d": s["median_R_d"]} for s in mode_stats[mode]]

    def clean(s):
        return {k: v for k, v in s.items() if not k.startswith("_")}

    headline = []
    for v in range(3):
        e = clean(mode_stats["ema_fixed_published"][v])
        headline.append({
            "tag": e["tag"], "gamma": e["gamma"],
            "F_d_ema": e["F_d"], "F_d_ema_ci95": e["F_d_ci95"],
            "F_d_published_raw": pub_stats[v]["F_d"],
            "coverage_pct_ema": e["assumption_coverage_pct"], "coverage_pct_published_raw": pub_stats[v]["assumption_coverage_pct"],
            "median_R_ema": e["median_R_d"], "median_R_published_raw": pub_stats[v]["median_R_d"],
            "R_at_batch_means_ema": e["R_at_batch_means (paper statistic)"],
            "R_at_batch_means_published_raw": pub_stats[v]["R_at_batch_means (paper statistic)"],
            "frac_starved_R_le_1e-2_pct_ema": e["frac_fully_starved_R_le_1e-2_pct"],
            "frac_starved_R_le_1e-2_pct_published_raw": pub_stats[v]["frac_fully_starved_R_le_1e-2_pct"],
            "F_d_ema_seed42_cpu_labels": clean(mode_stats["ema_fixed_seed"][v])["F_d"] if "ema_fixed_seed" in mode_stats else None,
            "coverage_pct_ema_seed42_cpu_labels": clean(mode_stats["ema_fixed_seed"][v])["assumption_coverage_pct"] if "ema_fixed_seed" in mode_stats else None,
        })
    ema = modes["ema_fixed_published"]
    out = {
        "title": "E3: F_d / gradient attenuation on EMA weights with the trainer's CIFAR-10 normalization",
        "generated": "2026-09-25",
        "scripts": ["experiments/e3_fd/snapshot_grad_attenuation_ema.py", "experiments/e3_fd/recover_published_neg_labels.py",
                    "experiments/e3_fd/fd_stats.py"],
        "setup": {
            "checkpoints": [r["ckpt_meta"]["ckpt"] for r in ema],
            "ckpt_epoch_stage1": [r["ckpt_meta"]["ckpt_epoch_stage1"] for r in ema],
            "ckpt_best_s1_val": [r["ckpt_meta"]["ckpt_best_s1"] for r in ema],
            "use_sam": [r["ckpt_meta"]["ckpt_cfg_use_sam"] for r in ema],
            "arch": "L4/D128, CIFAR-10, seed 42 (gated-ablation family); SAM off in all three",
            "weights": "ckpt['net_ema'], strict=True (0 missing / 0 unexpected)",
            "normalization": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010],
                              "published_std_was": [0.2470, 0.2435, 0.2616]},
            "examples": "first 1024 CIFAR-10 test images (4 x 256, shuffle=False), as published",
            "wrong_labels": "the published snapshot's wrong labels, recovered exactly (see G0); robustness: CPU seed-42 draw",
            "beta": BETA, "margins": "current-block aggregated goodness, true label minus wrong label",
            "F_d": "mean over examples of 1 - min(1, R), R = (1+e^{beta m})/(1+e^{beta(m+gamma P)}); CI95 = 1.96*sd/sqrt(N) (published convention)",
            "coverage": "pointwise fraction of examples with m >= 0 and P >= 0 (block 0: P = 0)",
            "device": "CPU (CUDA_VISIBLE_DEVICES='')",
        },
        "gates": {"all_passed": all_ok, "checks": gates},
        "ema_accuracy_gate": [r["acc_gate"] for r in ema],
        "headline_ema_vs_published": headline,
        "paired_ema_minus_published": paired,
        "attribution_weights_vs_normalization": attribution,
        "all_modes": {mode: [clean(s) for s in ss] for mode, ss in mode_stats.items()},
        "published_reference": {"fd_results_json": PUB_JSON, "arrays": PUB_NPZ, "arrays_sha256": sha256(PUB_NPZ),
                                "stats_full_precision": pub_stats},
        "arrays_sha256": {mode: [r["npz_sha256"] for r in res] for mode, res in modes.items()},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("gates all passed:", all_ok)
    for g in gates:
        print(("  ok  " if g["ok"] else "  FAIL") + " " + g["check"])
    for h in headline:
        print(f"\n{h['tag']}")
        print("  F_d EMA      ", [f"{x:.4f}" for x in h["F_d_ema"]], " +/-", [f"{x:.4f}" for x in h["F_d_ema_ci95"]])
        print("  F_d published", [f"{x:.4f}" for x in h["F_d_published_raw"]])
        print("  F_d EMA (cpu-seed labels)", [f"{x:.4f}" for x in h["F_d_ema_seed42_cpu_labels"]])
        print("  coverage EMA ", [f"{x:.1f}" for x in h["coverage_pct_ema"]], " published", [f"{x:.1f}" for x in h["coverage_pct_published_raw"]])
        print("  median R EMA ", [f"{x:.3e}" for x in h["median_R_ema"]], " published", [f"{x:.3e}" for x in h["median_R_published_raw"]])
        print("  starved% EMA ", [f"{x:.1f}" for x in h["frac_starved_R_le_1e-2_pct_ema"]], " published", [f"{x:.1f}" for x in h["frac_starved_R_le_1e-2_pct_published_raw"]])
    for mode, rows in attribution.items():
        print(mode, [[round(x, 4) for x in r["F_d"]] for r in rows])


if __name__ == "__main__":
    main()
