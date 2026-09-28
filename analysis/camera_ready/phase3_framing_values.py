#!/usr/bin/env python3
"""Phase 3 (framing, A7/A11) derived values -> results/phase3_framing_values.json.

(1) A11 / Fig. 10 (fig:accuracy_vs_ratio; Fig. 7 when submitted): the pooled Pearson r = -0.49 (n=15) over the L4+L8 D128
    conditions, split by depth. Input: results/fig_accuracy_vs_ratio_values.json (the exact 15 points
    plotted). Reports within-depth Pearson/Spearman, the depth-demeaned Pearson, and leave-one-out r,
    so the caption can say what carries the pooled association.
(2) A7: the L8/D128/32-expert seed-42 rows (gamma=0, CP-FAIR gamma=0.7, LCFF gamma=1.0) in all four
    protocols, read from each run's events.jsonl::final_test_eval (the tab:dissociation_full rows print
    S2-TTA only).

CPU only; reads files, writes one JSON under results/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

import config as C

OUT = C.RESULTS / "phase3_framing_values.json"


def fig7_within_depth() -> dict:
    src = C.RESULTS / "fig_accuracy_vs_ratio_values.json"
    d = json.loads(src.read_text())
    pts = d["points"]
    x = np.array([p["ratio"] for p in pts], float)
    y = np.array([p["s2_val_final_epoch_pct"] for p in pts], float)
    depth = np.array([p["L"] for p in pts], int)
    out: dict = {
        "source": str(src.relative_to(C.CR)),
        "x": d["x"],
        "y": d["y"],
        "pooled": {"n": int(len(x)), "pearson_r": float(stats.pearsonr(x, y)[0]),
                   "pearson_p": float(stats.pearsonr(x, y)[1]),
                   "spearman_rho": float(stats.spearmanr(x, y)[0])},
        "within_depth": {},
    }
    xd, yd = x.copy(), y.copy()
    for L in (4, 8):
        m = depth == L
        pr = stats.pearsonr(x[m], y[m])
        sr = stats.spearmanr(x[m], y[m])
        out["within_depth"][f"L{L}"] = {
            "n": int(m.sum()), "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
            "ratio_mean": float(x[m].mean()), "acc_mean_pct": float(y[m].mean()),
        }
        xd[m] -= x[m].mean()
        yd[m] -= y[m].mean()
    out["depth_demeaned_pearson_r"] = float(stats.pearsonr(xd, yd)[0])
    loo = {}
    for i, p in enumerate(pts):
        m = np.ones(len(pts), bool)
        m[i] = False
        loo[f"L{p['L']}_{p['condition']}"] = float(stats.pearsonr(x[m], y[m])[0])
    out["leave_one_out_pearson_r"] = loo
    out["reading"] = ("the pooled r is carried by the L4-vs-L8 contrast (L8 runs: higher ratio, lower "
                      "180-epoch validation accuracy); within each depth |r| is about 0.1, and no single "
                      "point's removal brings the pooled r above -0.35")
    return out


L8_32E_RUNS = {
    "gamma0": "ff_cifar10_scale_noall_G0_L8D128_L8_D128_bs512_seed42_20260314_015458",
    "cp_fair_gamma0.7": "ff_cifar10_scale_cpfair_G07_L8D128_L8_D128_bs512_seed42_20260315_190153",
    "lcff_gamma1.0": "ff_cifar10_scale_lcff_G10_L8D128_L8_D128_bs512_seed42_20260316_140422",
}


def l8_32e_rows() -> dict:
    base = C.CF / "basic_fair_revised_valfix_cp_fair" / "scaling_L8D128_32e" / "runs_scale_L8D128"
    rows = {}
    for name, run in L8_32E_RUNS.items():
        ev = [json.loads(line) for line in (base / run / "events.jsonl").read_text().splitlines() if line.strip()]
        fte = [e for e in ev if e.get("event") == "final_test_eval"]
        assert len(fte) == 1, (run, len(fte))
        cfg = json.loads((base / run / "config.json").read_text()) if (base / run / "config.json").exists() else {}
        rows[name] = {
            "run_dir": str((base / run).relative_to(C.PR.parent)),
            **{k: round(100 * fte[0][v], 2) for k, v in C.PROTOCOLS.items()},
            "n_experts": cfg.get("n_experts"), "moe_top_k": cfg.get("moe_top_k"), "use_sam": cfg.get("use_sam"),
        }
    s2t = {k: v["S2-TTA"] for k, v in rows.items()}
    return {
        "note": "seed 42 only (single seed); the gamma=0 sibling run ..._20260315_185311 has identical "
                "final_test_eval and is a duplicate; the CP-FAIR sibling ..._20260315_071624 has no final_test_eval.",
        "rows": rows,
        "cp_fair_minus_gamma0_S2-TTA_pp": round(s2t["cp_fair_gamma0.7"] - s2t["gamma0"], 2),
        "cp_fair_minus_lcff_S2-TTA_pp": round(s2t["cp_fair_gamma0.7"] - s2t["lcff_gamma1.0"], 2),
    }


def main() -> None:
    out = {
        "_meta": {
            "description": "Phase 3 framing values: Fig. 10 (fig:accuracy_vs_ratio) within-depth correlation split (A11) and the "
                           "L8/D128/32-expert four-protocol rows (A7).",
            "generated_by": "analysis/phase3_framing_values.py",
            "generated": str(date.today()),
        },
        "fig7_accuracy_vs_ratio": fig7_within_depth(),
        "l8_d128_32e_seed42": l8_32e_rows(),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
