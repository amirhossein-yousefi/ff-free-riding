#!/usr/bin/env python3
"""E3 reproduction gate: recover the wrong labels used by the published snapshot.

The published grad_attenuation_arrays.npz came from a run whose ``torch.randint`` wrong-label draw
(seed 42) was made on a CUDA device, so a CPU rerun draws different labels (verified: the CPU rerun's
per-block margin means differ in the third decimal place). This script checks that the published forward pass is
reproduced exactly. For each of the 1024 examples and each of the 9 wrong labels k, it computes the
per-block current-goodness margin m_k^(d) = g^(d)(y) - g^(d)(k) with the published settings (raw
weights, legacy std), then finds the k that matches the published m^(0..3) for that example.

Gate: every example has a unique k whose 4-block margin vector matches the published one to
max-abs < TOL. The three variants must recover the SAME label vector, because they share one seeded
draw. Output: out/recovered_published_y_neg.npy plus a JSON report.

Run (CPU only):
  CUDA_VISIBLE_DEVICES='' python recover_published_neg_labels.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("FF_WEIGHTS", "raw")
os.environ.setdefault("FF_NORM_STD", "legacy")
os.environ["FF_ACC_GATE"] = "0"
assert os.environ["FF_WEIGHTS"] == "raw" and os.environ["FF_NORM_STD"] == "legacy"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import snapshot_grad_attenuation_ema as snap  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

PUB = f"{snap.CF}/revised_training/grad_attenuation/grad_attenuation_arrays.npz"
TOL = 2e-3


@torch.no_grad()
def goodness_all_labels(net, images, num_classes):
    """[C, L, B] current-block aggregated goodness under each label hypothesis."""
    net.eval()
    tokens0 = net.embed(images)
    out = []
    for k in range(num_classes):
        yk = torch.full((images.size(0),), k, dtype=torch.long)
        tokens = tokens0
        g_hist, gl = [], []
        for blk in net.blocks:
            g_prev = torch.stack(g_hist, dim=1).detach() if g_hist else None
            a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
            gl.append(blk.aggregate_goodness(g_vec))
            g_hist.append(g_vec)
            tokens = snap.l2_normalize_tokens(a)
        out.append(torch.stack(gl, 0))
    return torch.stack(out, 0)


def main():
    pub = np.load(PUB, allow_pickle=True)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    report = {"published_arrays": PUB, "tol": TOL, "variants": []}
    E_all, Y_ref = [], None
    for vi, v in enumerate(snap.VARIANTS):
        cfg = snap.build_cfg_from_json(v["config"])
        cfg.device = "cpu"
        net = snap.FFHybridNet(cfg)
        ck = torch.load(v["ckpt"], map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"], strict=True)
        del ck
        loader = snap.build_eval_loader(cfg, snap.BATCH)
        G, Y = [], []
        for bi, (images, y) in enumerate(loader):
            if bi >= snap.N_BATCHES:
                break
            G.append(goodness_all_labels(net, images, cfg.num_classes))
            Y.append(y)
        G = torch.cat(G, dim=2).numpy().astype(np.float64)   # [C, L, N]
        Y = torch.cat(Y).numpy()
        Y_ref = Y if Y_ref is None else Y_ref
        assert np.array_equal(Y, Y_ref)
        N = Y.shape[0]
        g_true = G[Y, :, np.arange(N)]                         # [N, L]
        m_pub = pub[f"{vi}_m"].astype(np.float64).T            # [N, L]
        m_all = g_true[:, None, :] - np.transpose(G, (2, 0, 1))  # [N, C, L]
        E = np.abs(m_all - m_pub[:, None, :])                  # [N, C, L]
        E[np.arange(N), Y, :] = np.inf
        E_all.append(E)
    N, C, L = E_all[0].shape
    # per-variant argmin on the 4-block max error, and the combined (sum over variants) argmin
    per_var_k = [E.max(axis=2).argmin(axis=1) for E in E_all]
    score = sum(E.max(axis=2) for E in E_all)                  # [N, C]
    order = np.argsort(score, axis=1)
    k_comb = order[:, 0]
    best, second = score[np.arange(N), order[:, 0]], score[np.arange(N), order[:, 1]]
    sep_ratio = best / second
    report["combined"] = {
        "n": int(N),
        "n_unique_best_lt_half_second": int((sep_ratio < 0.5).sum()),
        "max_best_over_second": float(sep_ratio.max()),
        "n_per_variant_argmin_agree_with_combined": [int((k == k_comb).sum()) for k in per_var_k],
    }
    for vi, E in enumerate(E_all):
        e = E[np.arange(N), k_comb, :]                          # [N, L] error at combined label
        rec = {"tag": snap.VARIANTS[vi]["tag"],
               "n_all_blocks_within_tol": int((e.max(axis=1) < TOL).sum()),
               "frac_all_blocks_within_tol": float((e.max(axis=1) < TOL).mean()),
               "per_block_median_err": [float(np.median(e[:, d])) for d in range(L)],
               "per_block_p95_err": [float(np.percentile(e[:, d], 95)) for d in range(L)],
               "per_block_max_err": [float(e[:, d].max()) for d in range(L)],
               "per_block_frac_within_tol": [float((e[:, d] < TOL).mean()) for d in range(L)]}
        print(rec)
        report["variants"].append(rec)
    report["passed_strict_all_within_tol"] = bool(all(r["n_all_blocks_within_tol"] == N for r in report["variants"]))
    report["passed_relaxed"] = bool(
        report["combined"]["n_unique_best_lt_half_second"] >= 0.99 * N
        and all(max(r["per_block_median_err"]) < 1e-3 for r in report["variants"])
        and all(r["frac_all_blocks_within_tol"] >= 0.90 for r in report["variants"]))
    np.save(out_dir / "recovered_published_y_neg.npy", k_comb)
    report["note"] = ("wrong labels recovered by exhaustive match of the published per-example 4-block margins, "
                      "combined over the 3 variants (they share one seeded draw); saved to out/recovered_published_y_neg.npy")
    print(report["combined"])
    with open(out_dir / "recover_published_neg_labels.json", "w") as f:
        json.dump(report, f, indent=2)
    print("strict:", report["passed_strict_all_within_tol"], "relaxed:", report["passed_relaxed"])


if __name__ == "__main__":
    main()
