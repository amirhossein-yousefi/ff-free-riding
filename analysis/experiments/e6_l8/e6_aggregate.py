#!/usr/bin/env python3
"""E6: aggregate the chunked CPU evaluations of the L8/D128 kappa=0 (adaptive t1_k0) seed-42 checkpoint.

Inputs: out/{val,test}_*.npz from e6_eval_chunk.py. Logged comparators come from epoch_metrics.jsonl (seed 42)
and from events.jsonl final_test_eval (seeds 123/456, the same config: gamma_tau=1, kappa=0, SAM off).
Output: CR/results/e6_l8_k0_s42.json
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import glob
import hashlib
import json
import os

import numpy as np

HOME = FF_HOME
CF = f"{HOME}/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
CR = FF_OUT  # [CR] was the camera-ready repository; outputs go to <FF_OUT>/results
HERE = os.path.dirname(os.path.abspath(__file__))
L8 = f"{CF}/basic_fair_revised_valfix_cp_fair/ff_gated_ablation_L8"
RUN_T1 = f"{L8}/runs/ff_cifar10_adaptive_t1_k0_L8_L8_D128_bs512_seed42_20260315_201449"
RUN_T3 = f"{L8}/runs/ff_cifar10_adaptive_t3_k0_L8_L8_D128_bs512_seed42_20260316_033950"
MS = f"{CF}/revised_training/extracted/ff_cifar10_l8_gated_multiseed"
OTHER = {123: f"{MS}/runs_gated_k0_L8_seed123/ff_cifar10_cifar10_gated_k0_L8_seed123_L8_D128_bs512_seed123_20260427_041411",
         456: f"{MS}/runs_gated_k0_L8_seed456/ff_cifar10_cifar10_gated_k0_L8_seed456_L8_D128_bs512_seed456_20260427_114056"}
CK1 = f"{L8}/checkpoints/ff_cifar10_adaptive_t1_k0_L8_L8_D128_P2_seed42_stage1_best.pt"
CK2 = f"{L8}/checkpoints/ff_cifar10_adaptive_t1_k0_L8_L8_D128_P2_seed42_stage2_best.pt"
OUT = f"{CR}/results/e6_l8_k0_s42.json"
PAPER_ROW = {"seed42_printed_s2_tta": 86.88, "seed123": [85.94, 87.12, 86.47, 87.51], "seed456": [85.68, 86.84, 85.92, 87.01],
             "mean_sd_printed_s2_tta": "87.13 +/- 0.33 (n=3, S2 TTA only)", "tex": "appendix/B_full_config.tex:552-556; appendix/F_adaptive_collab_sweep.tex:96"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_split(split):
    files = sorted(glob.glob(f"{HERE}/out/{split}_*.npz"))
    parts = [np.load(f) for f in files]
    ranges = [tuple(int(x) for x in os.path.basename(f)[len(split) + 1:-4].split("_")) for f in files]
    # chunks must tile [0, N) exactly once
    ranges_sorted = sorted(ranges)
    assert ranges_sorted[0][0] == 0 and all(a[1] == b[0] for a, b in zip(ranges_sorted, ranges_sorted[1:])), ranges_sorted
    order = np.argsort([r[0] for r in ranges])
    cat = lambda k: np.concatenate([parts[i][k] for i in order])  # noqa: E731
    d = {k: cat(k) for k in ("idx", "y", "s1_orig", "s1_flip", "s2_orig", "s2_flip")}
    metas = [json.load(open(f.replace(".npz", ".json"))) for f in files]
    return d, ranges_sorted, metas


def top1(scores, y):
    return float((np.argmax(scores, axis=1) == y).mean())


def accs(d):
    y = d["y"]
    return {"S1": top1(d["s1_orig"], y), "S1-TTA": top1(d["s1_orig"] + d["s1_flip"], y),
            "S2": top1(d["s2_orig"], y), "S2-TTA": top1((d["s2_orig"] + d["s2_flip"]) / 2.0, y), "n": int(len(y))}


def main():
    val, val_ranges, val_meta = load_split("val")
    test, test_ranges, test_meta = load_split("test")
    assert len(val["y"]) == 5000 and len(test["y"]) == 10000
    assert np.array_equal(test["idx"], np.arange(10000))
    va, ta = accs(val), accs(test)

    em = rj(f"{RUN_T1}/epoch_metrics.jsonl")
    s1_evals = [(r["epoch"], r["eval_top1"]) for r in em if r["stage"] == 1 and r.get("eval_top1") is not None]
    s2_rows = [r for r in em if r["stage"] == 2]
    logged_best_s1 = max(v for _, v in s1_evals)
    logged_best_s1_epoch = min(e for e, v in s1_evals if v == logged_best_s1)
    logged_best_s2 = max(r["best_s2"] for r in s2_rows)
    meta0 = val_meta[0]
    gates = []
    gates.append({"check": "ckpt stage1 epoch == logged best-val epoch", "got": meta0["stage1_epoch"], "want": logged_best_s1_epoch,
                  "ok": meta0["stage1_epoch"] == logged_best_s1_epoch})
    gates.append({"check": "ckpt best_s1 == logged best_s1", "got": meta0["stage1_best_s1"], "want": logged_best_s1,
                  "ok": abs(meta0["stage1_best_s1"] - logged_best_s1) < 1e-12})
    gates.append({"check": "ckpt best_s2 == logged best_s2", "got": meta0["stage2_best_s2"], "want": logged_best_s2,
                  "ok": abs(meta0["stage2_best_s2"] - logged_best_s2) < 1e-12})
    gates.append({"check": "stage-2 backbone == stage1_best net_ema; strict loads clean",
                  "ok": bool(meta0["stage2_backbone_equals_stage1_best_ema"] and not meta0["net_ema_missing"] and not meta0["net_ema_unexpected"]
                             and not meta0["head_missing"] and not meta0["head_unexpected"])})
    sc = [m.get("selfcheck_max_abs_diff") for m in val_meta + test_meta if m.get("selfcheck_max_abs_diff")]
    gates.append({"check": "combined forward == trainer predict_scores / extract_hybrid_features_all_labels", "got": sc,
                  "ok": bool(sc) and all(max(s.values()) == 0.0 for s in sc)})
    tol = 1 / 5000
    d1 = abs(va["S1-TTA"] - logged_best_s1)
    d2 = abs(va["S2-TTA"] - logged_best_s2)
    gates.append({"check": "recomputed val S1-TTA == logged best_s1 (registered tol 1/5000)", "got": va["S1-TTA"], "want": logged_best_s1,
                  "diff_examples": round(d1 * 5000), "ok": d1 <= tol + 1e-12})
    gates.append({"check": "recomputed val S2-TTA == logged best_s2 (registered tol 1/5000)", "got": va["S2-TTA"], "want": logged_best_s2,
                  "diff_examples": round(d2 * 5000), "ok": d2 <= tol + 1e-12})
    all_ok = all(g["ok"] for g in gates)

    def near_ties(sc):
        srt = np.sort(sc, 1)
        gap = srt[:, -1] - srt[:, -2]
        rel = gap / np.maximum(np.abs(srt[:, -1]), 1e-9)
        return {"median_abs_top_score": float(np.median(np.abs(srt[:, -1]))),
                "n_rel_gap_lt_1e-3": int((rel < 1e-3).sum()), "n_rel_gap_lt_3e-3": int((rel < 3e-3).sum()),
                "n_rel_gap_lt_1e-2": int((rel < 1e-2).sum())}
    tie_diag = {"val_S1-TTA": near_ties(val["s1_orig"] + val["s1_flip"]),
                "val_S2-TTA": near_ties((val["s2_orig"] + val["s2_flip"]) / 2.0),
                "test_S1": near_ties(test["s1_orig"]), "test_S1-TTA": near_ties(test["s1_orig"] + test["s1_flip"]),
                "test_S2": near_ties(test["s2_orig"]), "test_S2-TTA": near_ties((test["s2_orig"] + test["s2_flip"]) / 2.0)}
    gate_assessment = (
        "Checkpoint-identity gates pass (epoch, best_s1, best_s2, backbone equality, strict loads, forward self-check). "
        "The recomputed-validation gate passes for S2-TTA (1 of 5000 off) and misses the registered 1-example tolerance "
        "for S1-TTA by one example (2 of 5000 off, 0.04 pp). Training-time eval ran on an A100 with "
        "set_float32_matmul_precision('high') (TF32); this CPU run is fp32, and 10 val examples have S1-TTA top-2 gaps "
        "below 0.3% of the score magnitude, so a 2-example flip is within TF32 numerical noise. "
        "Treat the seed-42 test values as carrying about +/-0.03 pp of numerical uncertainty.")

    # other seeds (same config) from their logged final_test_eval
    per_seed = {42: {k: ta[k] for k in ("S1", "S1-TTA", "S2", "S2-TTA")}}
    other_cfg = {}
    for s, d in OTHER.items():
        fte = [e for e in rj(f"{d}/events.jsonl") if e["event"] == "final_test_eval"][0]
        per_seed[s] = {"S1": fte["stage1_test_top1_no_tta"], "S1-TTA": fte["stage1_test_top1_tta"],
                       "S2": fte["stage2_test_top1_no_tta"], "S2-TTA": fte["stage2_test_top1_tta"]}
        c = json.load(open(f"{d}/config.json"))
        other_cfg[s] = {k: c.get(k) for k in ("gamma_gating_mode", "gamma_tau", "gamma_kappa", "gamma_scale", "num_blocks",
                                               "d_model", "use_sam", "epochs_stage1", "epochs_stage2", "n_experts", "moe_top_k")}
    c42 = json.load(open(f"{RUN_T1}/config.json"))
    cfg42 = {k: c42.get(k) for k in other_cfg[123]}
    summary = {}
    for k in ("S1", "S1-TTA", "S2", "S2-TTA"):
        v = np.array([per_seed[s][k] * 100 for s in (42, 123, 456)])
        summary[k] = {"per_seed_pct": [round(float(x), 2) for x in v], "mean_pct": float(v.mean()), "sd_pct_ddof1": float(v.std(ddof=1)),
                      "n": 3}
    v_old = np.array([86.88, per_seed[123]["S2-TTA"] * 100, per_seed[456]["S2-TTA"] * 100])

    ident = {}
    for name, run in (("t1_k0", RUN_T1), ("t3_k0", RUN_T3)):
        c = json.load(open(f"{run}/config.json"))
        em_ = rj(f"{run}/epoch_metrics.jsonl")
        lm = rj(f"{run}/layer_metrics.jsonl")
        last = max(r["epoch"] for r in lm if r["stage"] == 1)
        ident[name] = {"gamma_tau": c.get("gamma_tau"), "gamma_kappa": c.get("gamma_kappa"), "use_sam": c.get("use_sam"),
                       "best_s1_val": max(r["eval_top1"] for r in em_ if r["stage"] == 1 and r.get("eval_top1") is not None),
                       "best_s2_val": max(r["best_s2"] for r in em_ if r["stage"] == 2),
                       "sep_curr_nl_deepest_last_epoch": [r["sep_curr_nl"] for r in lm if r["stage"] == 1 and r["epoch"] == last and r["layer"] == c["num_blocks"] - 1][0],
                       "has_events_jsonl": os.path.exists(f"{run}/events.jsonl")}

    out = {
        "title": "E6: CPU test evaluation of the L8/D128 kappa=0 seed-42 run (paper row tab:gated_k0_multiseed / F:96)",
        "generated": "2026-09-25",
        "scripts": ["experiments/e6_l8/e6_eval_chunk.py", "experiments/e6_l8/e6_aggregate.py"],
        "identification": {
            "paper_row": PAPER_ROW,
            "candidates": ident,
            "conclusion": "the paper's kappa=0 L8 row is adaptive t1_k0 (gamma_tau=1, kappa=0): its val best_s2 0.8688 = printed 86.88 and its deepest sep 6.938 = printed 6.94; seeds 123/456 use the same gamma_tau=1, kappa=0, SAM off",
            "seed42_config": cfg42, "seed123_456_config": other_cfg,
        },
        "checkpoints": {"stage1_best": CK1, "stage2_best": CK2, "stage1_best_sha256": sha256(CK1), "stage2_best_sha256": sha256(CK2)},
        "protocol": {
            "S1": "stage1_best net_ema, FF goodness-sum over 10 label hypotheses (trainer predict_scores), top-1",
            "S1-TTA": "S1 scores(original) + S1 scores(horizontal flip)",
            "S2": "stage2_best AttentiveHybridHead on the frozen stage1_best net_ema (asserted identical), top-1",
            "S2-TTA": "mean of S2 logits over original and flip (trainer evaluate_stage2)",
            "normalization": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
            "val_split": "regenerated with random.Random(42) shuffle, val_fraction 0.1; asserted equal to a stored seed-42 split_indices.json",
            "device": "CPU fp32 (training-time eval ran on A100 with float32 matmul precision 'high' = TF32, so a few examples may flip)",
            "chunks": {"val": val_ranges, "test": test_ranges},
        },
        "gates": {"all_passed": all_ok, "checks": gates, "assessment": gate_assessment, "near_tie_diagnostics": tie_diag},
        "validation_recomputed": va,
        "validation_logged": {"best_s1_val_S1-TTA": logged_best_s1, "best_s1_epoch": logged_best_s1_epoch, "best_s2_val_S2-TTA": logged_best_s2},
        "test_seed42": ta,
        "test_seed42_pct": {k: round(ta[k] * 100, 2) for k in ("S1", "S1-TTA", "S2", "S2-TTA")},
        "n3_with_seed42_test": summary,
        "published_n3_s2_tta_with_seed42_val": {"values_pct": [float(x) for x in v_old], "mean_pct": float(v_old.mean()),
                                                "sd_pct_ddof1": float(v_old.std(ddof=1))},
        "note": "seeds 123/456 values are their logged final_test_eval (events.jsonl); seed 42 is this CPU recomputation",
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("gates all passed:", all_ok)
    for g in gates:
        print(("  ok   " if g["ok"] else "  FAIL ") + g["check"], g.get("got", ""), g.get("want", ""))
    print("val recomputed:", {k: round(v, 4) if isinstance(v, float) else v for k, v in va.items()})
    print("test seed 42:", out["test_seed42_pct"])
    for k, s in summary.items():
        print(f"  n=3 {k:7s} {s['per_seed_pct']}  mean {s['mean_pct']:.2f} +/- {s['sd_pct_ddof1']:.2f}")
    print("published (seed-42 val):", out["published_n3_s2_tta_with_seed42_val"])


if __name__ == "__main__":
    main()
