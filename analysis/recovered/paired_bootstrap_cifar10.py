#!/usr/bin/env python3
"""Tier-0 #4: CIFAR-10 example-level statistics for the flagship gamma=0 vs CP-FAIR equivalence.

Paired bootstrap (5000 folds) + exact McNemar per matched seed, S1-TTA and S2-TTA.
Provenance gate: every file's top-1 must match aggregated_results.json before pairing.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json
import os

import numpy as np

BASE = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
AGG = json.load(open(f"{BASE}/NeurIPS_FF_Net_Last_submitted/supplement_code/metric_summaries/aggregated_results.json"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paired_bootstrap_cifar10.json")

RUNS = {
    ("gamma0", 42): f"{BASE}/ablations/no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260312_132043",
    ("gamma0", 123): f"{BASE}/revised_training/out/gamma0_seed123/runs/ff_cifar10_cifar10_gamma0_seed123_L4_D256_bs512_seed123_20260417_235110",
    ("gamma0", 456): f"{BASE}/revised_training/out/gamma0_seed456/runs/ff_cifar10_cifar10_gamma0_seed456_L4_D256_bs512_seed456_20260422_135437",
    ("cp_fair", 42): f"{BASE}/basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035",
    ("cp_fair", 123): f"{BASE}/basic_fair_revised_valfix_cp_fair/runs_seed123/ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_bs512_seed123_20260320_001527",
    ("cp_fair", 456): f"{BASE}/basic_fair_revised_valfix_cp_fair/runs_seed456/ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_bs512_seed456_20260327_080127",
}
PROTOCOLS = {"s1_tta": "preds_stage1_test_tta.npz", "s2_tta": "preds_stage2_test_tta.npz"}
N_BOOT, SEED = 5000, 0


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value via binomial tail."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


rng = np.random.default_rng(SEED)
results = {}
for proto, fname in PROTOCOLS.items():
    per_seed, pooled_a, pooled_b, pooled_true = [], [], [], []
    for seed in (42, 123, 456):
        za = np.load(os.path.join(RUNS[("cp_fair", seed)], fname))
        zb = np.load(os.path.join(RUNS[("gamma0", seed)], fname))
        assert np.array_equal(za["y_true"], zb["y_true"]), f"y_true mismatch seed {seed}"
        # provenance gate against canonical JSON (structures differ per variant)
        for tag, z in (("cp_fair", za), ("gamma0", zb)):
            acc = float((z["y_pred"] == z["y_true"]).mean())
            if tag == "gamma0":
                expect = AGG["c10_gamma0"][str(seed)][proto]["top1"]
            else:
                row = next(r for r in AGG["c10_cp_fair"]["per_seed"] if r["seed"] == seed)
                expect = row[proto]
            assert abs(acc - expect) < 5e-4, f"{tag} seed{seed} {proto}: {acc:.4f} != JSON {expect}"
        ca, cb = za["y_pred"] == za["y_true"], zb["y_pred"] == zb["y_true"]
        delta = ca.astype(int) - cb.astype(int)
        n = len(delta)
        boot = rng.choice(delta, size=(N_BOOT, n), replace=True).mean(axis=1) * 100
        flips_ab = int(((ca) & (~cb)).sum())   # cp_fair right, gamma0 wrong
        flips_ba = int(((~ca) & (cb)).sum())
        per_seed.append({
            "seed": seed,
            "acc_cp_fair": round(float(ca.mean()) * 100, 2),
            "acc_gamma0": round(float(cb.mean()) * 100, 2),
            "delta_pp": round(float(delta.mean()) * 100, 3),
            "ci95_pp": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            "disagreement_pct": round(float((za["y_pred"] != zb["y_pred"]).mean()) * 100, 2),
            "flips_cpfair_only_right": flips_ab,
            "flips_gamma0_only_right": flips_ba,
            "mcnemar_p": round(mcnemar_exact(flips_ab, flips_ba), 4),
        })
        pooled_a.append(ca), pooled_b.append(cb), pooled_true.append(za["y_true"])
    ca, cb = np.concatenate(pooled_a), np.concatenate(pooled_b)
    delta = ca.astype(int) - cb.astype(int)
    boot = rng.choice(delta, size=(N_BOOT, len(delta)), replace=True).mean(axis=1) * 100
    results[proto] = {
        "per_seed": per_seed,
        "pooled": {
            "n": len(delta),
            "delta_pp": round(float(delta.mean()) * 100, 3),
            "ci95_pp": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            "inside_pm1pp": bool(np.percentile(boot, 2.5) > -1 and np.percentile(boot, 97.5) < 1),
        },
    }

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
