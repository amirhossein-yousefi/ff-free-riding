#!/usr/bin/env python3
"""Aggregate every camera-ready result family from raw sources into results/*.json.

Port and extension of CR/recompute_rebuttal.py and CR/mgc_c100_paired_churn.py
(CAMERA_READY_PLAN.md items E4/E5, feeding B1 and C1-C13).

  * CPU only, numpy only (no torch import). Reads raw run dirs / analysis JSONs listed in
    config.py (all read-only) and writes one JSON per result family into results/.
  * Accuracies are read as fractions from events.jsonl::final_test_eval (or
    final_metrics.json for the BP rows) and stored in percentage points.
  * Every spread is the SAMPLE SD (ddof=1). Where a source JSON stored a population SD,
    the population value is kept only under a key that says so.
  * Paired bootstraps: 5000 resamples, numpy default_rng(0), percentile CIs.

Usage:  python analysis/aggregate.py            (from CR; ~1-2 min on CPU)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

PROT = list(C.PROTOCOLS)                 # ["S1", "S1-TTA", "S2", "S2-TTA"]
SEEDS = list(C.SEEDS)
NOW = _dt.date.today().isoformat()
WRITTEN: list = []          # files written by this run (other scripts write other results/*.json)


# ----------------------------------------------------------------------------- helpers
def r8(x):
    """Store floats at 8 dp (avoids float noise but keeps full precision for any later rounding)."""
    if x is None:
        return None
    return round(float(x), 8)


def pp(frac):
    """fraction (4 dp in events.jsonl) -> percentage points, exact at 2 dp."""
    return round(100.0 * float(frac), 2)


def summ(values):
    """per-seed list -> {values, mean, sd (sample, ddof=1), n}."""
    vals = [float(v) for v in values]
    return {
        "values": [r8(v) for v in vals],
        "mean": r8(st.mean(vals)),
        "sd": r8(st.stdev(vals)) if len(vals) > 1 else None,
        "n": len(vals),
    }


def rel(p: Path) -> str:
    """path relative to the home prefix, for compact provenance."""
    try:
        return str(Path(p).relative_to(C.HOME))
    except ValueError:
        return str(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def final_eval(d: Path) -> dict:
    evs = [json.loads(l) for l in open(d / "events.jsonl")]
    fe = [e for e in evs if e.get("event") == "final_test_eval"]
    assert len(fe) == 1, (d, len(fe))
    return fe[0]


def layer_rows(d: Path, stage: int = 1):
    rows = [json.loads(l) for l in open(d / "layer_metrics.jsonl")]
    return [r for r in rows if r.get("stage", 1) == stage]


def last_epoch_layers(d: Path):
    rows = layer_rows(d)
    ep = max(r["epoch"] for r in rows)
    last = {r["layer"]: r for r in rows if r["epoch"] == ep}
    return ep, [last[k] for k in sorted(last)], rows


def accs(d: Path) -> dict:
    fe = final_eval(d)
    return {p: pp(fe[k]) for p, k in C.PROTOCOLS.items()}


def load_preds(d: Path, prot: str):
    z = np.load(d / C.PRED_FILES[prot])
    return z["y_true"], z["y_pred"]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_means(delta: np.ndarray, rng, n_boot: int = C.N_BOOT) -> np.ndarray:
    """Loop form used by paired_dissociation_stats.py and mgc_c100_paired_churn.py."""
    n = len(delta)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = delta[rng.integers(0, n, size=n)].mean()
    return out


def write(name: str, obj: dict):
    obj.setdefault("_meta", {})
    obj["_meta"].setdefault("generated_by", "analysis/aggregate.py")
    obj["_meta"].setdefault("generated", NOW)
    obj["_meta"].setdefault("sd_convention", "sample SD (ddof=1) over seeds")
    p = C.RESULTS / f"{name}.json"
    WRITTEN.append(p)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    print(f"wrote results/{name}.json")


def check_close(label, got, expect, tol):
    ok = abs(float(got) - float(expect)) <= tol
    return {"check": label, "recomputed": r8(got), "reference": r8(expect), "tol": tol, "pass": bool(ok)}


# ----------------------------------------------------------------------------- run-family readers
def run_family(runs: dict, lam: bool = False) -> dict:
    """Per-seed four-protocol accuracies + last-Stage-1-epoch per-block sep (and MGC lambda)."""
    per_seed, src = {}, {}
    for s, d in runs.items():
        a = accs(d)
        ep, last, rows = last_epoch_layers(d)
        rec = dict(a)
        rec["final_stage1_epoch"] = ep
        rec["first_stage1_epoch_logged"] = min(r["epoch"] for r in rows)
        rec["sep_curr_nl"] = [r8(r["sep_curr_nl"]) for r in last]
        rec["gpos_cur"] = [r8(r["gpos_cur"]) for r in last]
        if lam:
            rec["lam_mean_final"] = [r8(r.get("mgc_lam_mean")) for r in last]
            rec["lam_p95_final"] = [r8(r.get("mgc_lam_p95")) for r in last]
            nl = len(last)
            rec["lam_mean_whole_logged"] = [
                r8(np.mean([x["mgc_lam_mean"] for x in rows if x["layer"] == l])) for l in range(nl)]
        per_seed[s] = rec
        src[s] = rel(d)
    agg = {p: summ([per_seed[s][p] for s in runs]) for p in PROT}
    nblk = len(next(iter(per_seed.values()))["sep_curr_nl"])
    agg["sep_curr_nl_per_block"] = [summ([per_seed[s]["sep_curr_nl"][b] for s in runs]) for b in range(nblk)]
    agg["sep_curr_nl_deepest"] = agg["sep_curr_nl_per_block"][-1]
    if lam:
        agg["lam_mean_final_per_block"] = [summ([per_seed[s]["lam_mean_final"][b] for s in runs]) for b in range(nblk)]
    return {"per_seed": {str(k): v for k, v in per_seed.items()}, "agg": agg, "run_dirs": {str(k): v for k, v in src.items()}}


def config_diff(a: Path, b: Path, ignore=("seed", "version_tag", "run_name", "run_dir", "out_dir", "ckpt_dir", "timestamp")):
    ca, cb = json.load(open(a / "config.json")), json.load(open(b / "config.json"))
    keys = sorted(set(ca) | set(cb))
    return {k: [ca.get(k), cb.get(k)] for k in keys
            if ca.get(k) != cb.get(k) and not any(t in k for t in ignore)}


# ============================================================================= 1. CIFAR-100 trio
def build_trio():
    fam = {v: run_family(C.TRIO_RUNS[v]) for v in ("gamma0", "gated_k0", "cumulative")}
    means = {v: {p: fam[v]["agg"][p]["mean"] for p in PROT} for v in fam}
    rng_pp = {p: r8(max(means[v][p] for v in fam) - min(means[v][p] for v in fam)) for p in PROT}
    rng_pp_2dp = {p: r8(max(round(means[v][p], 2) for v in fam) - min(round(means[v][p], 2) for v in fam)) for p in PROT}
    sep = {v: fam[v]["agg"]["sep_curr_nl_deepest"]["mean"] for v in fam}
    # gate against the submitted supplement's aggregated_results.json
    agg = json.load(open(C.SUPP_AGG))["c100"]
    gates = []
    keymap = {"S1": "s1_no_tta", "S1-TTA": "s1_tta", "S2": "s2_no_tta", "S2-TTA": "s2_tta"}
    for v in fam:
        for row in agg[v]["per_seed"]:
            s = str(row["seed"])
            for p, k in keymap.items():
                gates.append(check_close(f"{v} s{s} {p} vs supplement aggregated_results.json",
                                         fam[v]["per_seed"][s][p], 100 * row[k], 0.005))
            gates.append(check_close(f"{v} s{s} deepest sep vs supplement sep_curr_nl_deep",
                                     fam[v]["per_seed"][s]["sep_curr_nl"][-1], row["sep_curr_nl_deep"], 1e-6))
    obj = {
        "_meta": {
            "description": "CIFAR-100 dense L4/D256 dissociation trio (gamma=0, adaptive kappa=0, cumulative gamma=0.7), 3 seeds; four protocols from events.jsonl::final_test_eval; sep = last-Stage-1-epoch training sep_curr_nl from layer_metrics.jsonl (not recomputed on the selected checkpoint).",
            "protocols": C.PROTOCOL_LABELS,
            "sep_label": "last-Stage-1-epoch training sep^cur_nl (layer_metrics.jsonl, final Stage-1 epoch 362)",
            "excluded_aborted_duplicates": {k: [rel(p) for p in v] for k, v in C.TRIO_EXCLUDED.items()},
            "all_gates_pass": all(g["pass"] for g in gates),
        },
        "variants": fam,
        "range_across_trio_pp": rng_pp,
        "range_across_trio_pp_of_2dp_means": rng_pp_2dp,
        "range_note": "the submitted paper computed ranges from the printed (2-dp) means (0.43/0.48/0.10/0.27); from unrounded means S2 and S2-TTA give 0.11/0.28.",
        "deepest_sep_ratio": {
            "gamma0_over_cumulative": r8(sep["gamma0"] / sep["cumulative"]),
            "gated_k0_over_cumulative": r8(sep["gated_k0"] / sep["cumulative"]),
            "gamma0_over_gated_k0": r8(sep["gamma0"] / sep["gated_k0"]),
        },
        "gates": gates,
    }
    write("c100_trio", obj)
    return obj


# ============================================================================= 2. MGC CIFAR-100
def build_mgc_c100(trio):
    mgc = run_family(C.MGC_C100_RUNS, lam=True)
    cum = trio["variants"]["cumulative"]
    deltas = {}
    for p in PROT:
        per = [r8(mgc["per_seed"][str(s)][p] - cum["per_seed"][str(s)][p]) for s in SEEDS]
        deltas[p] = summ(per)
    sep_m = mgc["agg"]["sep_curr_nl_deepest"]
    sep_c = cum["agg"]["sep_curr_nl_deepest"]
    lam_final = [summ([mgc["per_seed"][str(s)]["lam_mean_final"][b] for s in SEEDS]) for b in range(4)]
    # range across the four C100 variants (trio + MGC)
    allv = {v: trio["variants"][v]["agg"] for v in ("gamma0", "gated_k0", "cumulative")}
    allv["mgc"] = mgc["agg"]
    rng4 = {p: r8(max(a[p]["mean"] for a in allv.values()) - min(a[p]["mean"] for a in allv.values())) for p in PROT}
    rng4_2dp = {p: r8(max(round(a[p]["mean"], 2) for a in allv.values()) - min(round(a[p]["mean"], 2) for a in allv.values())) for p in PROT}
    seps = {v: a["sep_curr_nl_deepest"]["mean"] for v, a in allv.items()}
    cfg = {str(s): config_diff(C.TRIO_RUNS["cumulative"][s], C.MGC_C100_RUNS[s]) for s in SEEDS}
    s456 = mgc["per_seed"]["456"]
    obj = {
        "_meta": {
            "description": "MGC (missing-gradient compensation, c0=1) on CIFAR-100 dense L4/D256, 3 seeds (Colab runs), versus the cumulative (gamma=0.7) arm on matched seeds.",
            "protocols": C.PROTOCOL_LABELS,
            "implementation_note": "faithful implementation, not exact recovery: eps=1e-6, positive-part clamp, only lambda mean and p95 logged per block. The MGC arm REPLACES the baseline's depth-scaled current-block term; block 0 therefore loses its 0.25 auxiliary and its compensator is zero.",
            "resumed_run_note": f"seed 456 was resumed: its layer_metrics.jsonl and epoch_metrics.jsonl start at epoch {s456['first_stage1_epoch_logged']} (final_test_eval is complete); lam_mean_whole_logged for seed 456 therefore averages epochs {s456['first_stage1_epoch_logged']}-{s456['final_stage1_epoch']} only.",
            "cumulative_runs_used": "ONLY seed42_20260418_032149, seed123_20260419_010632, seed456_20260420_051234",
            "sep_label": "last-Stage-1-epoch training sep^cur_nl (layer_metrics.jsonl, final Stage-1 epoch)",
            "trainer": rel(C.MGC_TRAINER),
        },
        "mgc": mgc,
        "cumulative": {"per_seed": cum["per_seed"], "agg": cum["agg"], "run_dirs": cum["run_dirs"]},
        "delta_mgc_minus_cumulative_pp": deltas,
        "deepest_block_sep": {
            "mgc": sep_m, "cumulative": sep_c,
            "ratio_mgc_over_cumulative_unrounded": r8(sep_m["mean"] / sep_c["mean"]),
            "ratio_of_2dp_rounded_means": r8(round(sep_m["mean"], 2) / round(sep_c["mean"], 2)),
        },
        "lambda_telemetry": {
            "final_epoch_mean_lambda_per_block_mean_over_seeds": lam_final,
            "seed42_whole_training_mean_lambda_per_block": mgc["per_seed"]["42"]["lam_mean_whole_logged"],
            "note": "lambda at block 0 is identically 0 (no compensator at block 0).",
        },
        "four_variant_c100": {
            "range_pp_per_protocol": rng4,
            "range_pp_per_protocol_of_2dp_means": rng4_2dp,
            "note": "table convention = ranges and ratios of the printed (2-dp) means, so a reader can verify them from the table; unrounded values kept alongside.",
            "deepest_sep_mean": {k: r8(v) for k, v in seps.items()},
            "deepest_sep_max_over_min_unrounded": r8(max(seps.values()) / min(seps.values())),
            "deepest_sep_max_over_min_2dp": r8(round(max(seps.values()), 2) / round(min(seps.values()), 2)),
        },
        "config_diff_cumulative_vs_mgc": cfg,
    }
    write("mgc_c100", obj)
    return obj


# ============================================================================= 3. MGC L4/D128 (CIFAR-10)
def build_mgc_d128():
    mgc = run_family(C.MGC_D128_RUNS, lam=True)
    use_sam = {str(s): json.load(open(d / "config.json")).get("use_sam") for s, d in C.MGC_D128_RUNS.items()}
    anchors = {}
    for k, d in C.GA_RUNS.items():
        a = accs(d)
        a["use_sam"] = json.load(open(d / "config.json")).get("use_sam")
        a["run_dir"] = rel(d)
        anchors[k] = a
    s1_mean = mgc["agg"]["S1"]["mean"]
    an = json.load(open(C.AN_MGC_D128))
    gates = [check_close(f"S1/S2-TTA means vs AN mgc_3seed_aggregate ({p})", mgc["agg"][p]["mean"], an[k]["mean"], 0.006)
             for p, k in C.PROTOCOLS.items()]
    obj = {
        "_meta": {
            "description": "MGC (c0=1) on CIFAR-10 L4/D128, 180+10 epochs, 3 seeds. Seed 42 is the local DGX run (MR/runs, checkpoint lineage MR/checkpoints); seeds 123/456 are Colab runs.",
            "protocols": C.PROTOCOL_LABELS,
            "sam_note": "all three MGC-D128 runs used SAM (use_sam=true); the seed-42 gated-ablation anchors (gamma=0.7, gamma=0, kappa=0, LCFF) used SAM off. Every MGC-D128 vs anchor comparison is cross-SAM (D13).",
            "anchor_note": "anchors are seed-42 events.jsonl::final_test_eval values; the submitted tab:gated prints 86.17/85.76/85.87 for gamma=0.7/gamma=0/kappa=0 S1, which differ by 0.01 pp from these final_test_eval values (86.18/85.75/85.86).",
        },
        "mgc": mgc,
        "use_sam": use_sam,
        "seed42_sam_off_anchors": anchors,
        "stage1_cost_check": {
            "mgc_S1_mean": s1_mean,
            "below_all_three_anchor_S1": bool(all(s1_mean < anchors[k]["S1"] for k in ("gamma07_constant", "gamma0_no_all_collab", "gated_k0_t1"))),
            "gamma07_seed42_S2TTA": anchors["gamma07_constant"]["S2-TTA"],
            "mgc_S2TTA_mean": mgc["agg"]["S2-TTA"]["mean"],
        },
        "gates": gates,
    }
    write("mgc_d128", obj)
    return obj


# ============================================================================= 4. squared hinge
def build_hinge():
    arms = {a: run_family(C.HINGE_RUNS[a]) for a in ("g07", "g0")}
    cfgs = {}
    for a, runs in C.HINGE_RUNS.items():
        for s, d in runs.items():
            c = json.load(open(d / "config.json"))
            cfgs[f"{a}_s{s}"] = {k: c.get(k) for k in ("barrier_type", "barrier_margin", "gamma_scale", "use_sam", "use_mgc", "d_model", "epochs_stage1")}
    aborted = {}
    for d in C.HINGE_ABORTED_DGX:
        em = [json.loads(l) for l in open(d / "epoch_metrics.jsonl")]
        has = any(json.loads(l).get("event") == "final_test_eval" for l in open(d / "events.jsonl"))
        aborted[rel(d)] = {"last_epoch": max(x.get("epoch", 0) for x in em), "has_final_test_eval": has}
    d07, d0 = arms["g07"]["agg"]["sep_curr_nl_deepest"], arms["g0"]["agg"]["sep_curr_nl_deepest"]
    obj = {
        "_meta": {
            "description": "Squared-hinge barrier max(0, theta - u)^2, theta=2, CIFAR-10 L4/D128, 180+10 epochs, SAM on, 3 seeds per arm (gamma=0.7 cumulative vs gamma=0).",
            "protocols": C.PROTOCOL_LABELS,
            "lineage": "ALL seeds (incl. seed 42) from MR/from_colab/runs_from_colab. The DGX seed-42 dirs in MR/runs are aborted (see aborted_dgx_seed42) and are never aggregated.",
            "resumed_run_note": f"gamma=0.7 seed 456 was resumed: its layer log starts at epoch {arms['g07']['per_seed']['456']['first_stage1_epoch_logged']} (final_test_eval complete).",
            "sep_label": "last-Stage-1-epoch training sep^cur_nl (layer_metrics.jsonl, epoch 180)",
        },
        "arms": {"gamma0.7": arms["g07"], "gamma0": arms["g0"]},
        "delta_gamma07_minus_gamma0_pp": {p: r8(arms["g07"]["agg"][p]["mean"] - arms["g0"]["agg"][p]["mean"]) for p in PROT},
        "deepest_sep_ratio_gamma0_over_gamma07": r8(d0["mean"] / d07["mean"]),
        "block0_sep": {"gamma0.7": arms["g07"]["agg"]["sep_curr_nl_per_block"][0], "gamma0": arms["g0"]["agg"]["sep_curr_nl_per_block"][0]},
        "configs": cfgs,
        "aborted_dgx_seed42": aborted,
    }
    write("hinge", obj)
    return obj


# ============================================================================= 5. BP / FF calibration
def build_bp():
    labels = {
        "bp_strong": "FF backbone (stripped) + BP, strong augmentation",
        "bp_weak": "FF backbone (stripped) + BP, weak (simple) augmentation",
        "plain_cnn_bp": "Plain CNN + BP",
        "plain_cnn_ff": "Plain CNN + FF (co-design control)",
    }
    rows = {}
    for key, runs in C.BP_RUNS.items():
        ps, src = {}, {}
        for s, d in runs.items():
            fm = json.load(open(d / "final_metrics.json"))
            val = fm.get("val_top1", fm.get("best_val_top1"))
            ps[str(s)] = {
                "test_top1": pp(fm["test_top1"]),
                "val_top1": pp(val),
                "n_params": fm.get("num_parameters", fm.get("n_params")),
                "n_params_millions": r8(fm.get("num_parameters", fm.get("n_params")) / 1e6),
                "train_aug": fm.get("train_aug"),
                "eval_tta": fm.get("eval_tta", False),
            }
            src[str(s)] = rel(d / "final_metrics.json")
        rows[key] = {
            "label": labels[key],
            "per_seed": ps,
            "test_top1": summ([ps[str(s)]["test_top1"] for s in SEEDS]),
            "val_top1": summ([ps[str(s)]["val_top1"] for s in SEEDS]),
            "sources": src,
        }
    # determinism check: July rerun of strong-aug seed 42 vs the March original
    a, b = C.BP_RUNS["bp_strong"][42], C.BP_STRONG_S42_RERUN
    rerun = {
        "original": rel(a), "rerun": rel(b),
        "test_top1_equal": json.load(open(a / "final_metrics.json"))["test_top1"] == json.load(open(b / "final_metrics.json"))["test_top1"],
        "preds_test_sha256_equal": sha256(a / "preds_test.npz") == sha256(b / "preds_test.npz"),
    }
    an = json.load(open(C.AN_BP))
    anmap = {"bp_strong": "FF-backbone + BP, strong aug", "bp_weak": "FF-backbone + BP, weak aug",
             "plain_cnn_bp": "Plain CNN + BP", "plain_cnn_ff": "Plain CNN + FF (co-design)"}
    gates = []
    for k, ak in anmap.items():
        gates.append(check_close(f"{k} mean vs AN", rows[k]["test_top1"]["mean"], an[ak]["mean"], 0.006))
        gates.append(check_close(f"{k} sd vs AN", rows[k]["test_top1"]["sd"], an[ak]["std"], 0.006))
    obj = {
        "_meta": {
            "description": "CIFAR-10 BP/FF calibration rows of Table 2 at n=3 (seeds 42/123/456). Seed 42 = the submitted single-seed runs in CF/ablations; seeds 123/456 = rebuttal runs in MR/bp_anchors. Single-crop test top-1 (no TTA) from final_metrics.json.",
            "param_note": "5,539,530 parameters applies to the stripped-backbone BP rows only.",
        },
        "rows": rows,
        "co_design_gap_pp": r8(rows["plain_cnn_bp"]["test_top1"]["mean"] - rows["plain_cnn_ff"]["test_top1"]["mean"]),
        "augmentation_effect_pp": r8(rows["bp_strong"]["test_top1"]["mean"] - rows["bp_weak"]["test_top1"]["mean"]),
        "strong_aug_seed42_rerun_determinism": rerun,
        "gates": gates,
    }
    write("bp_calibration", obj)
    return obj


# ============================================================================= 6. churn + paired bootstraps (C100)
def paired_stats(runs_a: dict, runs_b: dict, prot: str):
    """Per-seed flips/disagreement and the pooled per-example correctness delta (a - b)."""
    per_seed, deltas = {}, []
    for s in SEEDS:
        ya, pa = load_preds(runs_a[s], prot)
        yb, pb = load_preds(runs_b[s], prot)
        assert np.array_equal(ya, yb), ("y_true mismatch", s, prot)
        ca, cb = (pa == ya), (pb == yb)
        d = ca.astype(np.int32) - cb.astype(np.int32)
        deltas.append(d)
        b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
        per_seed[str(s)] = {
            "acc_a": r8(100 * ca.mean()), "acc_b": r8(100 * cb.mean()),
            "delta_pp": r8(100 * d.mean()),
            "disagreement_pct": r8(100 * (pa != pb).mean()),
            "a_right_b_wrong": b_, "a_wrong_b_right": c_,
            "mcnemar_exact_p": r8(mcnemar_exact(b_, c_)),
        }
    return per_seed, np.concatenate(deltas)


def ci_block(delta, means):
    lo, hi = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {
        "n_examples": int(len(delta)),
        "delta_pp": r8(100 * delta.mean()),
        "ci95_pp": [r8(100 * lo), r8(100 * hi)],
        "within_pm1pp": bool(100 * lo > -1 and 100 * hi < 1),
    }


def build_c100_paired(trio):
    # (i) trio pairs: the published procedure (paired_dissociation_stats.py): ONE default_rng(0)
    #     consumed sequentially over the pairs in this order, per protocol.
    pairs = [("gamma0", "gated_k0"), ("gamma0", "cumulative"), ("gated_k0", "cumulative")]
    trio_out = {}
    for p in PROT:
        rng = np.random.default_rng(C.BOOT_SEED)
        trio_out[p] = {}
        for a, b in pairs:
            per_seed, delta = paired_stats(C.TRIO_RUNS[a], C.TRIO_RUNS[b], p)
            means = boot_means(delta, rng)
            trio_out[p][f"{a}_minus_{b}"] = {"per_seed": per_seed, "pooled": ci_block(delta, means)}
    pub = json.load(open(C.C100_PAIRED_AUDIT))["pairwise_paired_bootstrap"]
    gates = []
    for k, v in pub.items():
        got = trio_out["S2-TTA"][k]["pooled"]
        gates.append(check_close(f"S2-TTA {k} ci_lo vs published audit", got["ci95_pp"][0], 100 * v["ci_lo_95"], 1e-6))
        gates.append(check_close(f"S2-TTA {k} ci_hi vs published audit", got["ci95_pp"][1], 100 * v["ci_hi_95"], 1e-6))
    write("c100_trio_paired", {
        "_meta": {
            "description": "Paired bootstrap on the matched CIFAR-100 test set for the trio pairs, all four protocols (pooled over matched seeds 42/123/456, n=30,000 per pair). Procedure = TEX/supplement_code/analysis/paired_dissociation_stats.py (5000 resamples, one default_rng(0) consumed over the three pairs in order), applied per protocol. S2-TTA reproduces the published audit JSON exactly.",
            "ci_conditioning": "CIs quantify paired test-set uncertainty conditional on the trained checkpoints; seed variation is reported separately as sample SD.",
            "n_boot": C.N_BOOT, "rng": "numpy default_rng(0), fresh per protocol",
            "all_gates_pass": all(g["pass"] for g in gates),
        },
        "pairs": trio_out,
        "gates": gates,
    })

    # (ii) MGC vs cumulative (C4/E5): template CR/mgc_c100_paired_churn.py at 5000 resamples,
    #      fresh default_rng(0) per protocol (its first 2000 draws equal the 2000-resample prototype).
    mgc_out, churn_mgc = {}, {}
    for p in PROT:
        rng = np.random.default_rng(C.BOOT_SEED)
        per_seed, delta = paired_stats(C.MGC_C100_RUNS, C.TRIO_RUNS["cumulative"], p)
        means = boot_means(delta, rng)
        mgc_out[p] = {"per_seed": per_seed, "pooled": ci_block(delta, means)}
        churn_mgc[p] = {s: per_seed[s]["disagreement_pct"] for s in per_seed}
    write("mgc_paired_bootstrap", {
        "_meta": {
            "description": "MGC minus cumulative, CIFAR-100 L4/D256, matched seeds 42/123/456 on the identical test order; pooled per-example correctness deltas (n=30,000), 5000-resample paired bootstrap, percentile 95% CI; exact McNemar per seed.",
            "protocols": C.PROTOCOL_LABELS,
            "n_boot": C.N_BOOT, "rng": "numpy default_rng(0), fresh per protocol (template CR/mgc_c100_paired_churn.py)",
            "within_pm1pp_rule": "CI entirely inside (-1, +1) pp",
            "ci_conditioning": "CIs are conditional on the trained checkpoints (test-set resampling only).",
            "cumulative_runs_used": "ONLY seed42_20260418_032149, seed123_20260419_010632, seed456_20260420_051234",
        },
        "mgc_minus_cumulative": mgc_out,
    })

    # (iii) churn (prediction disagreement) per seed per protocol
    trio_churn = {p: {k: {s: v["disagreement_pct"] for s, v in trio_out[p][k]["per_seed"].items()}
                      for k in trio_out[p]} for p in PROT}
    churn = {
        "_meta": {
            "description": "Prediction churn = % of the 10,000 CIFAR-100 test images whose predicted label differs between two variants on the same seed (preds_*.npz). MGC vs cumulative and the trio pairs, four protocols.",
            "protocols": C.PROTOCOL_LABELS,
        },
        "mgc_vs_cumulative": {p: {"per_seed": churn_mgc[p], **{k: v for k, v in summ([churn_mgc[p][str(s)] for s in SEEDS]).items() if k != "values"}} for p in PROT},
        "trio_pairs": {p: {k: {"per_seed": v, **{kk: vv for kk, vv in summ([v[str(s)] for s in SEEDS]).items() if kk != "values"}}
                           for k, v in trio_churn[p].items()} for p in PROT},
    }
    churn["trio_mean_disagreement_S2-TTA_pct"] = r8(st.mean([churn["trio_pairs"]["S2-TTA"][k]["mean"] for k in trio_churn["S2-TTA"]]))
    write("churn", churn)


# ============================================================================= 7. CIFAR-10 flagship pair
def build_c10_paired():
    agg = json.load(open(C.SUPP_AGG))
    R = C.C10_PAIR_RUNS
    keymap = {"S1": "s1_no_tta", "S1-TTA": "s1_tta", "S2": "s2_no_tta", "S2-TTA": "s2_tta"}

    def gate_acc(tag, s, prot, acc):
        if tag == "gamma0":
            expect = agg["c10_gamma0"][str(s)][keymap[prot]]["top1"]
        else:
            expect = next(r for r in agg["c10_cp_fair"]["per_seed"] if r["seed"] == s)[keymap[prot]]
        assert abs(acc - expect) < 5e-4, (tag, s, prot, acc, expect)

    def run(protos, rng):
        out = {}
        for prot in protos:
            per_seed, pa_, pb_ = [], [], []
            for s in SEEDS:
                ya, pa = load_preds(R["cp_fair"][s], prot)
                yb, pb = load_preds(R["gamma0"][s], prot)
                assert np.array_equal(ya, yb)
                gate_acc("cp_fair", s, prot, float((pa == ya).mean()))
                gate_acc("gamma0", s, prot, float((pb == yb).mean()))
                ca, cb = pa == ya, pb == yb
                delta = ca.astype(int) - cb.astype(int)
                boot = rng.choice(delta, size=(C.N_BOOT, len(delta)), replace=True).mean(axis=1) * 100
                b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
                per_seed.append({
                    "seed": s, "acc_cp_fair": r8(100 * ca.mean()), "acc_gamma0": r8(100 * cb.mean()),
                    "delta_pp": r8(100 * delta.mean()),
                    "ci95_pp": [r8(np.percentile(boot, 2.5)), r8(np.percentile(boot, 97.5))],
                    "disagreement_pct": r8(100 * (pa != pb).mean()),
                    "flips_cpfair_only_right": b_, "flips_gamma0_only_right": c_,
                    "mcnemar_p": r8(mcnemar_exact(b_, c_)),
                })
                pa_.append(ca), pb_.append(cb)
            ca, cb = np.concatenate(pa_), np.concatenate(pb_)
            delta = ca.astype(int) - cb.astype(int)
            boot = rng.choice(delta, size=(C.N_BOOT, len(delta)), replace=True).mean(axis=1) * 100
            lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            out[prot] = {"per_seed": per_seed, "pooled": {
                "n": int(len(delta)), "delta_pp": r8(100 * delta.mean()), "ci95_pp": [r8(lo), r8(hi)],
                "within_pm1pp": bool(lo > -1 and hi < 1)}}
            del boot
        return out

    # exact replay of AN/paired_bootstrap_cifar10.py: one default_rng(0) over S1-TTA then S2-TTA
    main = run(["S1-TTA", "S2-TTA"], np.random.default_rng(C.BOOT_SEED))
    extra = run(["S1", "S2"], np.random.default_rng(C.BOOT_SEED))
    an = json.load(open(C.AN_C10_PAIRED))
    gates = []
    for prot, ak in (("S1-TTA", "s1_tta"), ("S2-TTA", "s2_tta")):
        gates.append(check_close(f"{prot} pooled delta vs AN", main[prot]["pooled"]["delta_pp"], an[ak]["pooled"]["delta_pp"], 0.0006))
        for i in (0, 1):
            gates.append(check_close(f"{prot} pooled CI[{i}] vs AN", main[prot]["pooled"]["ci95_pp"][i], an[ak]["pooled"]["ci95_pp"][i], 0.0006))
    min_p = min(r["mcnemar_p"] for prot in main for r in main[prot]["per_seed"])
    write("c10_paired", {
        "_meta": {
            "description": "CIFAR-10 flagship pair, CP-FAIR minus gamma=0 (L4/D256), matched seeds 42/123/456, 10k test each; paired bootstrap (5000) + exact McNemar per seed. Provenance-gated against supplement aggregated_results.json. CIs conditional on the checkpoints.",
            "procedure": "exact replay of ff-wiki/_build/analyses/paired_bootstrap_cifar10.py (rng.choice, one default_rng(0) over S1-TTA then S2-TTA); the no-TTA protocols use a separate fresh default_rng(0) and were not posted.",
            "config_note": "gamma=0 seed 42 used a different configuration from seeds 123/456 (block_curr_lambda 0 / depth_order_lambda 0 vs 0.25 / 0.2; D7/X1).",
            "run_dirs": {v: {str(s): rel(d) for s, d in R[v].items()} for v in R},
            "all_gates_pass": all(g["pass"] for g in gates),
        },
        "cp_fair_minus_gamma0": {**main, **extra},
        "min_mcnemar_p_tta_protocols": r8(min_p),
        "min_mcnemar_p_tta_protocols_floor_2dp": math.floor(min_p * 100) / 100,
        "min_mcnemar_p_all_protocols": r8(min(r["mcnemar_p"] for prot in {**main, **extra} for r in {**main, **extra}[prot]["per_seed"])),
        "gates": gates,
    })


# ============================================================================= 8. S2-FF readouts (C8/C9)
def build_s2ff(trio):
    tri = {}
    for v in ("gamma0", "gated_k0", "cumulative"):
        a, b = json.load(open(C.S2FF_TRIO[v])), json.load(open(C.S2FFCE_TRIO[v]))
        assert a["s1_gate_ok"] and b["s1_gate_ok"]
        tri[v] = {
            "goodness_sum_S1": trio["variants"][v]["per_seed"]["42"]["S1"],
            "goodness_sum_S1_recomputed_by_readout_run": a["s1_sum_no_tta"],
            "strict_ff_block_alone": a["s2ff_alone_no_tta"],
            "strict_ff_plus_sum": a["s2ff_sum_no_tta"],
            "ce_block_alone": b["s2ffce_alone_no_tta"],
            "ce_plus_sum": b["s2ffce_sum_no_tta"],
            "s2_attentive_probe": trio["variants"][v]["per_seed"]["42"]["S2"],
            "sources": [rel(C.S2FF_TRIO[v]), rel(C.S2FFCE_TRIO[v]), rel(C.TRIO_RUNS[v][42] / "events.jsonl")],
        }
    mg = {}
    for s in SEEDS:
        a, b = json.load(open(C.S2FF_MGC[s])), json.load(open(C.S2FFCE_MGC[s]))
        assert a["s1_gate_ok"] and b["s1_gate_ok"]
        mg[str(s)] = {"goodness_sum_S1": accs(C.MGC_C100_RUNS[s])["S1"],
                      "goodness_sum_S1_recomputed_by_readout_run": a["s1_sum_no_tta"],
                      "strict_ff_block_alone": a["s2ff_alone_no_tta"],
                      "strict_ff_plus_sum": a["s2ff_sum_no_tta"], "ce_block_alone": b["s2ffce_alone_no_tta"],
                      "ce_plus_sum": b["s2ffce_sum_no_tta"],
                      "s2_attentive_probe": accs(C.MGC_C100_RUNS[s])["S2"],
                      "sources": [rel(C.S2FF_MGC[s]), rel(C.S2FFCE_MGC[s])]}
    keys = ["goodness_sum_S1", "strict_ff_block_alone", "strict_ff_plus_sum", "ce_block_alone", "ce_plus_sum", "s2_attentive_probe"]
    mg_agg = {k: summ([mg[str(s)][k] for s in SEEDS]) for k in keys}
    gate_max = max([abs(tri[v]["goodness_sum_S1"] - tri[v]["goodness_sum_S1_recomputed_by_readout_run"]) for v in tri]
                   + [abs(mg[s]["goodness_sum_S1"] - mg[s]["goodness_sum_S1_recomputed_by_readout_run"]) for s in mg])
    m42 = mg["42"]
    matched = {v: r8(m42["strict_ff_block_alone"] - tri[v]["strict_ff_block_alone"]) for v in tri}
    plus_sum_42 = {**{v: tri[v]["strict_ff_plus_sum"] for v in tri}, "mgc": m42["strict_ff_plus_sum"]}
    ce_shift = {v: r8(tri[v]["ce_block_alone"] - tri[v]["strict_ff_block_alone"]) for v in tri}
    ce_shift["mgc_mean"] = r8(mg_agg["ce_block_alone"]["mean"] - mg_agg["strict_ff_block_alone"]["mean"])
    an = json.load(open(C.AN_S2FF_FULL))["colab_mgc_seeds"]
    gates = [check_close(f"MGC s{s} s2ff_alone vs AN s2ff_full", mg[str(s)]["strict_ff_block_alone"], an[str(s)]["s2ff_alone_no_tta"], 1e-9) for s in SEEDS]
    write("s2ff_mgc", {
        "_meta": {
            "description": "Frozen-backbone readouts on CIFAR-100 (single crop, full 10k test). Strict-FF block = FF-shaped readout block trained only with the local FF goodness objective; CE block = same architecture trained with sampled-softmax CE; '+ sum' adds the frozen blocks' goodness. The Stage-2 attentive head is a probe, not part of the FF method.",
            "protocol": "no TTA, all-100-hypothesis scoring, EMA stage1_best backbones; S1 gate within 0.01 pp for every run",
            "goodness_sum_note": "goodness_sum_S1 = logged S1 (events.jsonl final_test_eval) so that it matches the other tables; the readout runs recompute it (goodness_sum_S1_recomputed_by_readout_run).",
            "comparison_rule": "compare like with like: matched seed 42 (MGC s42 vs trio s42); never mix the MGC 3-seed mean with seed-42 trio values.",
        },
        "trio_seed42": tri,
        "mgc_per_seed": mg,
        "mgc_agg": mg_agg,
        "matched_seed42_mgc_minus_trio_strict_ff_alone_pp": matched,
        "seed42_strict_ff_plus_sum": plus_sum_42,
        "seed42_strict_ff_plus_sum_span_pp": r8(max(plus_sum_42.values()) - min(plus_sum_42.values())),
        "ce_minus_strict_ff_block_alone_pp": ce_shift,
        "S1_logged_vs_readout_recompute_max_abs_diff_pp": r8(gate_max),
        "trio_seed42_strict_ff_alone_span_pp": r8(max(tri[v]["strict_ff_block_alone"] for v in tri) - min(tri[v]["strict_ff_block_alone"] for v in tri)),
        "gates": gates,
    })


# ============================================================================= 9. readout ladder (C11)
def build_ladder():
    d = json.load(open(C.AN_VAL_REFIT))
    pc = d["per_checkpoint"]
    rungs = ["R0", "R1", "R2", "R3"]
    per_var = {}
    for v in ("gamma0", "gated_k0", "cumulative"):
        ck = [f"c100_{v}_seed{s}" for s in SEEDS]
        assert all(pc[k]["gate_pass"] for k in ck)
        acc = {R: summ([100 * pc[k]["readouts"][R]["eval_acc"] for k in ck]) for R in rungs}
        gain = {R: summ([100 * (pc[k]["readouts"][R]["eval_acc"] - pc[k]["readouts"]["R0"]["eval_acc"]) for k in ck]) for R in rungs[1:]}
        pop = {R: d["aggregates"][f"c100/{v}"]["readouts"][R]["delta_vs_R0_pp_std"] for R in rungs[1:]}
        per_var[v] = {"eval_acc_pp": acc, "gain_vs_R0_pp": gain, "posted_population_sd_gain_DO_NOT_PRINT": pop,
                      "best_by_fit_readouts": d["aggregates"][f"c100/{v}"]["best_by_fit_readouts"]}
    span = {R: r8(max(per_var[v]["eval_acc_pp"][R]["mean"] for v in per_var) - min(per_var[v]["eval_acc_pp"][R]["mean"] for v in per_var)) for R in rungs}
    r0_vs_logged = max(abs(100 * (pc[k]["readouts"]["R0"]["eval_acc"] - pc[k]["anchor_s1_top1_no_tta"])) for k in pc)
    inter = d["interaction"]
    gates = [check_close(f"span {R} vs AN", span[R], d["c100_trio_span_per_readout"][R]["span_pp"], 0.006) for R in rungs]
    write("readout_ladder", {
        "_meta": {
            "description": "Validation-fitted readout ladder R0-R3 on the frozen CIFAR-100 trio (fit on each run's own 5k validation split, evaluated once on the full 10k test; S1 no-TTA, EMA weights). Exploratory and checkpoint-conditional.",
            "rungs": d["_header"]["readout_ladder"],
            "R2_note": "R2 (per-depth temperature log-opinion pool) is exactly R1 constrained to positive weights; when the R1 weights are positive, R2 is identical to R1.",
            "sd_note": "sample SDs (ddof=1) recomputed from per-checkpoint eval_acc; the source JSON's delta_vs_R0_pp_std are population SDs (kept under posted_population_sd_gain_DO_NOT_PRINT).",
            "interaction_ci_note": "differential-gain CIs are copied from the source (5000-fold paired bootstrap over the full 10k test); the per-example val/test dumps behind them were in a deleted temporary scratchpad and are not recomputable here.",
            "do_not_use": "readout_spectrum_results.json (fitted on the test split)",
            "source": rel(C.AN_VAL_REFIT),
        },
        "per_variant": per_var,
        "trio_span_pp_per_rung": span,
        "R0_vs_logged_S1_max_abs_diff_pp": r8(r0_vs_logged),
        "differential_gain_R1": {
            "gamma0_minus_cumulative": {"diff_pp": inter["R1"]["gamma0_minus_cumulative"]["diff_pp"], "ci95_pp": inter["R1"]["gamma0_minus_cumulative"]["ci95_pp"]},
            "gated_k0_minus_cumulative": {"diff_pp": inter["R1"]["gated_k0_minus_cumulative"]["diff_pp"], "ci95_pp": inter["R1"]["gated_k0_minus_cumulative"]["ci95_pp"]},
        },
        "differential_gain_all": inter,
        "gates": gates,
    })


# ============================================================================= 10. probes (C10)
PROBE_PANELS = {
    # key in source                      : (label, family)
    "ridge_goodness_blocksoftmax":        ("ridge, 400-D goodness (block-softmax normalized)", "goodness"),
    "logistic_goodness_blocksoftmax":     ("logistic, 400-D goodness (block-softmax normalized)", "goodness"),
    "logistic_goodness_flat":             ("logistic, 400-D goodness (raw)", "goodness"),
    "ridge_goodness_flat":                ("ridge, 400-D goodness (raw)", "goodness"),
    "knn10cos_goodness_blocksoftmax":     ("k-NN-10 (cosine), 400-D goodness (block-softmax normalized)", "goodness"),
    "knn10cos_goodness_flat":             ("k-NN-10 (cosine), 400-D goodness (raw)", "goodness"),
    "logistic_goodness_block3":           ("logistic, block-3 goodness (100-D)", "block3"),
    "knn10cos_goodness_block3":           ("k-NN-10 (cosine), block-3 goodness (100-D)", "block3"),
    "logistic_pooled_meanemb_concat":     ("logistic, pooled embedding, all blocks (mean label token)", "pooled"),
    "knn10cos_pooled_meanemb_concat":     ("k-NN-10 (cosine), pooled embedding, all blocks (mean label token)", "pooled"),
    "logistic_pooled_zerotok_concat":     ("logistic, pooled embedding, all blocks (zero label token)", "pooled"),
    "knn10cos_pooled_zerotok_concat":     ("k-NN-10 (cosine), pooled embedding, all blocks (zero label token)", "pooled"),
    "logistic_pooled_meanemb_block3":     ("logistic, pooled embedding, block 3 (mean label token)", "pooled_block3"),
    "knn10cos_pooled_meanemb_block3":     ("k-NN-10 (cosine), pooled embedding, block 3 (mean label token)", "pooled_block3"),
    "logistic_pooled_zerotok_block3":     ("logistic, pooled embedding, block 3 (zero label token)", "pooled_block3"),
    "knn10cos_pooled_zerotok_block3":     ("k-NN-10 (cosine), pooled embedding, block 3 (zero label token)", "pooled_block3"),
}


def build_probes():
    d = json.load(open(C.AN_CLEAN_PROBES))
    by = {}
    for lst in (d["per_checkpoint"], d["per_checkpoint_addendum"]):
        for row in lst:
            by.setdefault((row["variant"], row["seed"]), {}).update(row["probes"])
    panels = {}
    gates = []
    for key, (label, fam) in PROBE_PANELS.items():
        pv = {}
        for v in ("gamma0", "gated_k0", "cumulative"):
            vals = [100 * by[(v, s)][key]["acc_test10k"] for s in SEEDS]
            pv[v] = summ(vals)
            gates.append(check_close(f"{key} {v} mean vs source per_variant", pv[v]["mean"],
                                     100 * d["per_variant"][v][key]["mean"], 0.0006))
        span = r8(max(pv[v]["mean"] for v in pv) - min(pv[v]["mean"] for v in pv))
        gates.append(check_close(f"{key} span vs source trio_spans", span, d["trio_spans"][key]["span_pp"], 0.0015))
        panels[key] = {"label": label, "family": fam, "per_variant": pv, "trio_span_pp": span,
                       "span_lt_1pp": bool(span < 1), "posted": key in {
                           "ridge_goodness_blocksoftmax", "knn10cos_goodness_blocksoftmax",
                           "logistic_pooled_meanemb_concat", "knn10cos_pooled_meanemb_concat"}}
    anchors = {}
    for k in ("anchor_s1_no_tta", "anchor_s2_no_tta"):
        anchors[k] = {"per_variant_mean_pp": {v: r8(100 * x) for v, x in d["trio_spans"][k]["per_variant_mean"].items()},
                      "span_pp": d["trio_spans"][k]["span_pp"]}
    pooled_b3 = [panels[k]["trio_span_pp"] for k in panels if panels[k]["family"] == "pooled_block3"]
    write("probes", {
        "_meta": {
            "description": "Label-free probes on frozen CIFAR-100 trio features (fit on each run's own 5k validation split, evaluated on the full 10k test). Source: clean_probes_c100.json ONLY (simple_probes_c100.json feats_true panels are label-leaked, H24).",
            "fit_eval_protocol": d["_header"]["fit_eval_protocol"],
            "feature_definitions": d["_header"]["feature_definitions"],
            "sd_note": "sample SDs (ddof=1) recomputed from per-checkpoint acc_test10k; the source per_variant 'std' fields are population SDs and are not used.",
            "knn_normalization_note": "k-NN stability depends on per-block normalization: the 10-NN trio span is 1.47 pp on block-softmax goodness but 8.58 pp on raw goodness.",
            "source": rel(C.AN_CLEAN_PROBES),
            "all_gates_pass": all(g["pass"] for g in gates),
        },
        "panels": panels,
        "anchors": anchors,
        "pooled_block3_span_range_pp": [r8(min(pooled_b3)), r8(max(pooled_b3))],
        "gates": gates,
    })


# ============================================================================= 11. gradient share (C7)
def build_gradshare():
    d = json.load(open(C.AN_GRADSHARE))
    names = {"c10_d128_gamma07_baseline_constant_seed42": "gamma0.7",
             "c10_d128_gamma0_no_all_collab_seed42": "gamma0",
             "c10_d128_mgc_c1_seed42": "mgc"}
    arms = {}
    lin0, lin13, cos_disc = [], [], []
    for k, short in names.items():
        a = d["arms"][k]
        blocks = []
        for b in a["per_block"]:
            terms = {t: {"share": r8(v["share_mean"]), "cos": r8(v["cos_mean"])} for t, v in b["terms"].items() if v["present"]}
            blocks.append({"block": b["block"], "disc_share_block_cum": r8(b["terms"]["block_cum"]["share_mean"]),
                           "disc_cos_block_cum": r8(b["terms"]["block_cum"]["cos_mean"]),
                           "linearity_rel_err_max": b["linearity_rel_err_max"], "terms": terms})
            (lin0 if b["block"] == 0 else lin13).append(b["linearity_rel_err_max"])
            cos_disc.append(b["terms"]["block_cum"]["cos_mean"])
        arms[short] = {"source_arm": k, "gate": a["gates"], "use_sam": a["config_summary"]["use_sam"],
                       "config_summary": a["config_summary"], "n_batches": a["n_batches"], "batch_size": a["batch_size"],
                       "epoch_stage1": a["epoch_stage1"], "weights": a["weights_differentiated"], "per_block": blocks}
    write("gradient_share", {
        "_meta": {
            "description": "Per-term gradient share ||grad_t|| / ||grad_total|| of the Stage-1 per-block loss (seed 42, CIFAR-10 L4/D128, gated-ablation arms + MGC), 20 training-transform batches, raw training weights (ckpt['model']). 'disc' = the discrimination term on cumulative goodness (block_cum).",
            "labels": "gated-ablation D128, raw training weights, seed 42; MGC arm trained with SAM, the two gated arms without SAM",
            "linearity_note": "gradients of the terms sum to the total gradient up to float error: max relative error < 3e-4 at block 0 and about 1e-6 at blocks 1-3.",
            "source": rel(C.AN_GRADSHARE),
        },
        "arms": arms,
        "linearity_rel_err_max_block0": r8(max(lin0)),
        "linearity_rel_err_max_blocks1to3": r8(max(lin13)),
        "disc_cos_range": [r8(min(cos_disc)), r8(max(cos_disc))],
        "mgc_block0_disc_share": arms["mgc"]["per_block"][0]["disc_share_block_cum"],
    })


# ============================================================================= 12. redistribution (C12 / A4)
def build_redistribution():
    run = C.TRIO_RUNS["cumulative"][42] / C.PRED_FILES["S1"]
    z = np.load(run)
    S, y_pred, y_true = z["scores"].astype(np.float64), z["y_pred"], z["y_true"]
    base = S.argmax(axis=1)
    assert np.array_equal(base, y_pred)
    profiles = {"measured-cumulative-like": [0.50, 0.20, 0.16, 0.14], "uniform": [0.25] * 4,
                "reversed": [0.14, 0.16, 0.20, 0.50], "adversarial-healthy": [0.005, 0.005, 0.01, 0.98]}
    dec = []
    for name, w in profiles.items():
        w = np.asarray(w, dtype=np.float64)
        g = w[None, :, None] * S[:, None, :]
        Sr = g.sum(axis=1)
        dec.append({"profile": name, "weights": w.tolist(), "L3_own_share_pct": round(float(w[3]) * 100, 1),
                    "reconstruction_max_abs_err": float(np.abs(Sr - S).max()),
                    "prediction_flips": int((Sr.argmax(axis=1) != base).sum())})
    w = np.full(4, 0.25)
    g = w[None, :, None] * S[:, None, :]
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 0.2, size=S.shape) * np.abs(S)
    g2 = g.copy(); g2[:, 0, :] -= q; g2[:, 3, :] += q
    S2 = g2.sum(axis=1)
    zs = {"max_abs_score_change": float(np.abs(S2 - S).max()),
          "prediction_flips": int((S2.argmax(axis=1) != base).sum()),
          "L3_mean_goodness_change_pct": round(float((g2[:, 3, :].mean() / g[:, 3, :].mean() - 1) * 100), 1)}
    an = json.load(open(C.AN_REDIST))
    gates = [check_close("baseline acc vs AN", 100 * (base == y_true).mean(), an["baseline_acc_s1_no_tta"], 0.005)]
    gates += [{"check": f"{a['profile']} flips/share vs AN", "pass": a["prediction_flips"] == b["prediction_flips"] and a["L3_own_share_pct"] == b["L3_own_share_pct"]}
              for a, b in zip(dec, an["decompositions"])]
    gates.append({"check": "zero-sum transfer vs AN", "pass": zs["prediction_flips"] == an["zero_sum_transfer"]["prediction_flips"]
                  and zs["L3_mean_goodness_change_pct"] == an["zero_sum_transfer"]["L3_mean_goodness_change_pct"]})
    write("redistribution", {
        "_meta": {
            "description": "Arithmetic redistribution demo (Prop. redistribution): re-split the cumulative CIFAR-100 seed-42 model's real S1 goodness-sum scores S(x,c) into per-block parts with arbitrary depth profiles; the deepest block's own share moves 14% -> 98% while 0 of 10,000 predictions change. Covers exactly-redistributing changes only, not objective changes such as MGC.",
            "source_preds": rel(run),
            "empirical_counterpart": "prediction churn between trained variants: see churn.json",
        },
        "n": int(len(y_true)),
        "baseline_acc_s1_no_tta": round(float((base == y_true).mean()) * 100, 2),
        "decompositions": dec,
        "zero_sum_transfer": zs,
        "gates": gates,
    })


# ============================================================================= main
def main():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    trio = build_trio()
    build_mgc_c100(trio)
    build_mgc_d128()
    build_hinge()
    build_bp()
    build_c100_paired(trio)
    build_c10_paired()
    build_s2ff(trio)
    build_ladder()
    build_probes()
    build_gradshare()
    build_redistribution()
    # summary of every gate across files
    bad = []
    for p in WRITTEN:
        o = json.load(open(p))
        for g in o.get("gates", []):
            if not g.get("pass", False):
                bad.append((p.name, g))
    if bad:
        print("GATE FAILURES:")
        for b in bad:
            print("  ", b)
    else:
        print("all gates pass")


if __name__ == "__main__":
    main()
