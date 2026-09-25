#!/usr/bin/env python3
"""Tier-0 #3-weak: empirical demonstration of prop:redistribution on real logits.

Take the cumulative (gamma=0.7) CIFAR-100 model's real S1 goodness-sum scores S(x,c).
Construct several per-block decompositions g'_d(x,c) = w_d * S(x,c) whose deepest-block
"own share" spans the empirically observed range, verify Sum_d g'_d == S exactly, and
show ZERO of 10k predictions change while per-block diagnostics swing arbitrarily.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import glob
import json
import os

import numpy as np

BASE = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "redistribution_demo.json")

run = sorted(glob.glob(f"{BASE}/revised_training/extracted/ff_cifar100_multiseed/runs_cumulative_seed42/*/preds_stage1_test_no_tta.npz"))[0]
z = np.load(run)
S, y_pred, y_true = z["scores"].astype(np.float64), z["y_pred"], z["y_true"]
base_pred = S.argmax(axis=1)
assert np.array_equal(base_pred, y_pred), "scores argmax != saved y_pred (protocol mismatch)"
acc = float((base_pred == y_true).mean())

# Depth profiles w (L=4, sum=1): own-share of block 3 spans the observed range.
PROFILES = {
    "measured-cumulative-like": [0.50, 0.20, 0.16, 0.14],   # free-riding: L3 own-share 14%
    "uniform": [0.25, 0.25, 0.25, 0.25],
    "reversed": [0.14, 0.16, 0.20, 0.50],
    "adversarial-healthy": [0.005, 0.005, 0.01, 0.98],      # L3 own-share 98% (looks 'repaired')
}

results = {"run": os.path.relpath(run, BASE), "n": int(len(y_true)),
           "baseline_acc_s1_no_tta": round(acc * 100, 2), "decompositions": []}
for name, w in PROFILES.items():
    w = np.asarray(w, dtype=np.float64)
    assert abs(w.sum() - 1) < 1e-12
    g = w[None, :, None] * S[:, None, :]          # [N, L, C] decomposition
    S_re = g.sum(axis=1)                          # reconstruct
    max_err = float(np.abs(S_re - S).max())
    pred_re = S_re.argmax(axis=1)
    flips = int((pred_re != base_pred).sum())
    own_share_L3 = float(w[3])
    results["decompositions"].append({
        "profile": name, "weights": w.tolist(), "L3_own_share_pct": round(own_share_L3 * 100, 1),
        "reconstruction_max_abs_err": max_err, "prediction_flips": flips,
    })

# Strong-form teaser on the same arrays: explicit zero-sum transfer q(x,y) between blocks 0 and 3
# applied to the uniform decomposition — shifts diagnostics, predictions provably unchanged.
w = np.full(4, 0.25)
g = w[None, :, None] * S[:, None, :]
rng = np.random.default_rng(0)
q = rng.uniform(0, 0.2, size=S.shape) * np.abs(S)  # arbitrary class-dependent transfer
g2 = g.copy()
g2[:, 0, :] -= q
g2[:, 3, :] += q
S2 = g2.sum(axis=1)
results["zero_sum_transfer"] = {
    "max_abs_score_change": float(np.abs(S2 - S).max()),
    "prediction_flips": int((S2.argmax(axis=1) != base_pred).sum()),
    "L3_mean_goodness_change_pct": round(float((g2[:, 3, :].mean() / g[:, 3, :].mean() - 1) * 100), 1),
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))
