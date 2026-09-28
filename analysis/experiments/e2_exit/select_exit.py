#!/usr/bin/env python3
"""E2 (CR-19) selector: VALIDATION-selected truncation depth and dynamic-exit threshold, evaluated
ONCE on test. Replaces the rebuttal-era ff-wiki/_build/analyses/pareto_truncation.py, which chose
depths and thresholds on the test split.

Inputs: experiments/e2_exit/dumps/{<name>.json, block_dump_valtest_<name>.npz} written by
dump_val_test_block_data.py (cumulative per-depth scores S_d(x,c) of net_ema; every dump must have
passed its identity gates). Output: results/e2_exit.json.

Protocol (fixed before looking at any test number):
  readout   S1 goodness-sum, single crop (no TTA), EMA weights of the validation-selected
            stage1_best checkpoint -- the protocol of final_test_eval stage1_test_top1_no_tta.
  selection data: the run's own 5k validation split (split_indices.json val_idx), never test.
  T  (truncation, primary)   d* = smallest depth d with acc_val(d) >= acc_val(L).
  T' (truncation, secondary) d* = argmax_d acc_val(d) (ties -> smallest d).
  D  (dynamic exit)  after block d < L, exit if top1 - top2 of the running score S_d > tau; forced
            exit at L. Threshold grid = {-inf, +inf} U quantiles (101 levels, 0..1) of the pooled
            VALIDATION margins at depths 1..L-1. tau* = the grid value with the fewest mean blocks on
            validation subject to acc_val(tau) >= acc_val(L) (ties: higher val acc, then larger tau).
  test      each selected operating point is applied once to the 10k test split; reported against
            the full-depth test accuracy of the same checkpoint (paired, same images).
  FLOPs     torch FlopCounterMode on CPU (2*MAC; matmul/conv/SDPA counted, elementwise excluded),
            per image = stem + C * sum_{j<=d} block_j (the C label hypotheses all run blocks 1..d);
            dynamic exit = per-image cost averaged over the test images.
Also reported, clearly labeled as TEST-SELECTED (optimistic, for comparison with the rebuttal
numbers only): best test depth and test iso-accuracy depth.

The paper's "87.40" early-exit claim (appendix L and the fig:depth_truncation caption) is
recomputed from the same dumps: it is the VALIDATION flip-TTA accuracy at depth 1 of the
adaptive tau=3, kappa=0 arm at its best-val epoch (epoch_metrics eval_top1_d1); the matching test
numbers (TTA and no-TTA) are reported next to it.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ffcr_common as C  # noqa: E402

torch = C.setup_env(threads=6)
import numpy as np  # noqa: E402

DUMPS = C.EXP / "e2_exit" / "dumps"
OUT_JSON = C.RESULTS / "e2_exit.json"
OLD_PARETO = C.PR / "ff-wiki" / "_build" / "analyses" / "pareto_truncation_results.json"
N_TAU_LEVELS = 101

FAMILIES = {
    "c10_l4d128_gated_seed42": list(C.E2_C10),
    "c100_l4d256_trio": list(C.E2_C100),
    "c100_gamma0": [f"c100_gamma0_seed{s}" for s in (42, 123, 456)],
    "c100_gated_k0": [f"c100_gated_k0_seed{s}" for s in (42, 123, 456)],
    "c100_cumulative": [f"c100_cumulative_seed{s}" for s in (42, 123, 456)],
}
OLD_NAME = {  # names used by the rebuttal-era test-selected analysis
    **{f"c10_gated_{a}_seed42": f"c10_gated_{a}_seed42" for a in C.GA_ARMS},
    **{n: n for n in C.E2_C100},
}


# ------------------------------------------------------------------ FLOPs
def count_flops(name):
    """Executed forward FLOPs of the stem and of each block for ONE label hypothesis, B=1."""
    from torch.utils.flop_counter import FlopCounterMode
    e = C.REG[name]
    M = C.load_trainer(e["trainer"])
    ck = torch.load(e["ckpt"], map_location="cpu", weights_only=True) if e.get("cfg_from_ckpt") else None
    persisted, _ = C.persisted_config(e, ck)
    cfg = C.build_cfg(M, persisted)
    net = M.FFHybridNet(cfg).eval()
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    with FlopCounterMode(display=False) as fc:
        tokens = net.embed(x)
    stem = fc.get_total_flops()
    y = torch.zeros(1, dtype=torch.long)
    blocks, g_hist = [], []
    for blk in net.blocks:
        g_prev = torch.stack(g_hist, dim=1) if g_hist else None
        with FlopCounterMode(display=False) as fc:
            a, g_vec, _, _, _ = blk(tokens, y, g_prev)
        blocks.append(int(fc.get_total_flops()))
        g_hist.append(g_vec)
        tokens = M.l2_normalize_tokens(a)
    arch = dict(d_model=int(cfg.d_model), num_blocks=int(cfg.num_blocks), n_experts=int(cfg.n_experts),
                moe_top_k=int(cfg.moe_top_k), patch_size=int(cfg.patch_size), use_mem=bool(cfg.use_mem),
                num_classes=int(cfg.num_classes), num_tokens=int(tokens.shape[1]))
    return dict(stem_flops=int(stem), block_flops=blocks, arch=arch, reference_run=str(e["run_dir"]),
                convention="FLOPs = 2*MACs (torch FlopCounterMode); matmul/conv/SDPA counted, elementwise excluded")


def gflops_at_depth(fl, d):
    return (fl["stem_flops"] + fl["arch"]["num_classes"] * sum(fl["block_flops"][:d])) / 1e9


# ------------------------------------------------------------ operating points
def accs_by_depth(S, y):
    return [float((S[:, d].argmax(1) == y).mean()) for d in range(S.shape[1])]


def exit_depths(S, tau):
    """0-indexed exit block per example for margin threshold tau (forced exit at L-1)."""
    L = S.shape[1]
    Ss = np.sort(S[:, : L - 1, :], axis=2)
    marg = Ss[:, :, -1] - Ss[:, :, -2]                     # [N, L-1]
    ex = marg > tau
    return np.where(ex.any(1), ex.argmax(1), L - 1), marg


def dyn_eval(S, y, tau, fl):
    first, _ = exit_depths(S, tau)
    N = S.shape[0]
    preds = S.argmax(2)[np.arange(N), first]
    L = S.shape[1]
    cost = np.array([gflops_at_depth(fl, d + 1) for d in range(L)])[first]
    return dict(acc=float((preds == y).mean()), mean_blocks=float((first + 1).mean()),
                mean_gflops=float(cost.mean()), exit_hist=np.bincount(first, minlength=L).tolist())


def tau_grid(S_val):
    L = S_val.shape[1]
    _, marg = exit_depths(S_val, np.inf)
    qs = np.unique(np.quantile(marg[:, : L - 1].ravel(), np.linspace(0.0, 1.0, N_TAU_LEVELS)))
    return np.concatenate([[-np.inf], qs, [np.inf]])


def select_run(name, fl):
    side = json.load(open(DUMPS / f"{name}.json"))
    assert side["status"] in ("PASS", "PASS_NUMERICS"), f"{name}: dump gates did not pass"
    z = np.load(DUMPS / f"block_dump_valtest_{name}.npz")
    Sv, Sfv, St = z["scores_val"].astype(np.float32), z["scores_val_flip"], z["scores_test"].astype(np.float32)
    yv, yt = z["y_val"].astype(np.int64), z["y_test"].astype(np.int64)
    L = St.shape[1]
    g_full = gflops_at_depth(fl, L)

    av, at = accs_by_depth(Sv, yv), accs_by_depth(St, yt)
    assert abs(at[-1] - side["r0_test_recomputed"]) < 1e-6        # pairing check with the sidecar
    trunc = {}
    for rule in ("T_iso_val", "T_argmax_val"):
        if rule == "T_iso_val":
            d = next(i for i in range(L) if av[i] >= av[-1])
        else:
            d = int(np.argmax(av))
        trunc[rule] = dict(depth=d + 1, val_acc=av[d], val_acc_full=av[-1],
                           test_acc=at[d], test_acc_full=at[-1], delta_test_pp=round(100.0 * (at[d] - at[-1]), 6),
                           gflops=gflops_at_depth(fl, d + 1), flops_saved=1.0 - gflops_at_depth(fl, d + 1) / g_full)

    # dynamic exit: threshold chosen on validation
    grid = tau_grid(Sv)
    val_curve = []
    for tau in grid:
        r = dyn_eval(Sv, yv, tau, fl)
        r["tau"] = float(tau)
        val_curve.append(r)
    feas = [r for r in val_curve if r["acc"] >= av[-1]]
    best = min(feas, key=lambda r: (r["mean_blocks"], -r["acc"], -r["tau"]))
    te = dyn_eval(St, yt, best["tau"], fl)
    dyn = dict(tau=best["tau"], val_acc=best["acc"], val_mean_blocks=best["mean_blocks"], val_acc_full=av[-1],
               test_acc=te["acc"], test_acc_full=at[-1], delta_test_pp=round(100.0 * (te["acc"] - at[-1]), 6),
               test_mean_blocks=te["mean_blocks"], test_mean_gflops=te["mean_gflops"],
               flops_saved=1.0 - te["mean_gflops"] / g_full, test_exit_hist=te["exit_hist"])
    test_curve = [dict(tau=r["tau"], **{k: v for k, v in dyn_eval(St, yt, r["tau"], fl).items() if k != "exit_hist"})
                  for r in val_curve]

    # TEST-selected reference (optimistic; the rebuttal-era protocol)
    d_tb = int(np.argmax(at))
    d_ti = next(i for i in range(L) if at[i] >= at[-1])
    test_selected = dict(best_test_depth=d_tb + 1, acc_at_best_test_depth=at[d_tb],
                         iso_test_depth=d_ti + 1, flops_saved_at_iso_test=1.0 - gflops_at_depth(fl, d_ti + 1) / g_full)

    tta = None
    if "scores_test_flip" in z.files:
        Stf = z["scores_test_flip"]
        tta = dict(val_tta_by_depth=accs_by_depth(Sv + Sfv, yv), test_tta_by_depth=accs_by_depth(St + Stf, yt))
    return dict(
        name=name, dataset=side["dataset"], config=side["config"], seed=side["seed"],
        ckpt=side["ckpt"], ckpt_sha256=side["ckpt_sha256"], run_dir=side["run_dir"],
        gates=dict(status=side["status"], test_strict=side["gate_test"]["strict_pass"],
                   test_delta_examples=side["gate_test"]["delta_examples"], ckpt=side["gate_ckpt"],
                   consistency=side["consistency"]),
        val_no_tta_by_depth=av, test_no_tta_by_depth=at,
        gflops_by_depth=[gflops_at_depth(fl, d) for d in range(1, L + 1)],
        truncation=trunc, dynamic_exit=dyn,
        dynamic_exit_curves=dict(val=val_curve, test=test_curve),
        test_selected_reference=test_selected, tta=tta,
    )


def summarize(rows, rule_key, sub):
    vals = np.array([r[rule_key][sub]["delta_test_pp"] if sub else r[rule_key]["delta_test_pp"] for r in rows])
    fs = np.array([r[rule_key][sub]["flops_saved"] if sub else r[rule_key]["flops_saved"] for r in rows])
    n = len(vals)
    return dict(n=n,
                delta_test_pp_mean=float(vals.mean()), delta_test_pp_sd=float(vals.std(ddof=1)) if n > 1 else None,
                delta_test_pp_min=float(vals.min()), delta_test_pp_max=float(vals.max()),
                n_delta_ge_0=int((vals >= 0).sum()), n_delta_gt_0=int((vals > 0).sum()),
                flops_saved_mean=float(fs.mean()), flops_saved_sd=float(fs.std(ddof=1)) if n > 1 else None,
                flops_saved_min=float(fs.min()), flops_saved_max=float(fs.max()))


def main():
    t0 = time.time()
    names = [n for n in list(C.E2_C10) + list(C.E2_C100) if (DUMPS / f"{n}.json").exists()]
    missing = [n for n in list(C.E2_C10) + list(C.E2_C100) if n not in names]
    fl = {"c10": count_flops("c10_gated_baseline_constant_seed42") if any(n.startswith("c10") for n in names) else None,
          "c100": count_flops("c100_gamma0_seed42") if any(n.startswith("c100") for n in names) else None}
    runs = {}
    for n in names:
        side = json.load(open(DUMPS / f"{n}.json"))
        if side["status"] not in ("PASS", "PASS_NUMERICS"):
            print(f"[skip] {n}: dump status {side['status']}")
            missing.append(n)
            continue
        # Fix 2026-09-26: the key used to be `"c10" if n.startswith("c10") else "c100"`, which is also true
        # for "c100_*" names, so every CIFAR-100 run was costed with the CIFAR-10 FLOP model (10 classes,
        # D128). Only gflops / flops_saved of the C100 runs changed; depths, taus and deltas did not.
        fl_run = fl["c100" if n.startswith("c100") else "c10"]
        n_cls = {"cifar10": 10, "cifar100": 100}[side["dataset"]]
        assert fl_run["arch"]["num_classes"] == n_cls, (n, side["dataset"], fl_run["arch"])
        runs[n] = select_run(n, fl_run)
        r = runs[n]
        print(f"{n:40s} val {['%.4f' % a for a in r['val_no_tta_by_depth']]} test {['%.4f' % a for a in r['test_no_tta_by_depth']]} "
              f"| T d*={r['truncation']['T_iso_val']['depth']} dTest={r['truncation']['T_iso_val']['delta_test_pp']:+.2f}pp "
              f"| D mb={r['dynamic_exit']['test_mean_blocks']:.2f} dTest={r['dynamic_exit']['delta_test_pp']:+.2f}pp", flush=True)

    fams = {}
    for fam, members in FAMILIES.items():
        rows = [runs[m] for m in members if m in runs]
        if len(rows) < 1:
            continue
        fams[fam] = dict(
            members=[r["name"] for r in rows],
            T_iso_val=summarize(rows, "truncation", "T_iso_val"),
            T_argmax_val=summarize(rows, "truncation", "T_argmax_val"),
            D_dynamic_exit=summarize(rows, "dynamic_exit", None),
            T_iso_val_depths=[r["truncation"]["T_iso_val"]["depth"] for r in rows],
            T_argmax_val_depths=[r["truncation"]["T_argmax_val"]["depth"] for r in rows],
            D_test_mean_blocks=[r["dynamic_exit"]["test_mean_blocks"] for r in rows],
        )

    # consistency with the rebuttal-era (GPU) test dumps: per-depth test accuracy
    old_cmp = None
    if OLD_PARETO.exists():
        old = json.load(open(OLD_PARETO))["conditions"]
        old_cmp = {n: dict(old=old[OLD_NAME[n]]["acc_by_depth"], new=runs[n]["test_no_tta_by_depth"],
                           max_abs_diff=float(np.max(np.abs(np.array(old[OLD_NAME[n]]["acc_by_depth"]) -
                                                            np.array(runs[n]["test_no_tta_by_depth"])))))
                   for n in runs if OLD_NAME.get(n) in old}

    # the paper's 87.40 claim (t3_k0 at depth 1) and 87.06 / 87.48 companions
    claim = None
    key = "c10_gated_adaptive_t3_k0_seed42"
    if key in runs and runs[key]["tta"] is not None:
        side = json.load(open(DUMPS / f"{key}.json"))
        base = runs.get("c10_gated_baseline_constant_seed42")
        claim = dict(
            paper_text="adaptive tau=3, kappa=0 achieves 87.40% using only Block 0, exceeding its full 4-block "
                       "accuracy of 87.06%; the constant-gamma baseline peaks at 87.48% at d=2 "
                       "(appendix L 'Adaptive gating enables early exit'; fig:depth_truncation caption)",
            protocol_found=("VALIDATION split (5k, the run's own val_idx), flip-TTA, EMA weights, at the best-val "
                            "epoch = epoch_metrics eval_top1_d<d>; i.e. the numbers are validation accuracies "
                            "presented as if they were test accuracies, and d is 1-indexed (Block 0 = d=1)"),
            t3_k0=dict(val_tta_by_depth_recomputed=runs[key]["tta"]["val_tta_by_depth"],
                       val_tta_by_depth_logged=side["logged_val_tta_by_depth"],
                       test_tta_by_depth=runs[key]["tta"]["test_tta_by_depth"],
                       test_no_tta_by_depth=runs[key]["test_no_tta_by_depth"]),
            baseline_constant=(dict(val_tta_by_depth_recomputed=base["tta"]["val_tta_by_depth"],
                                    test_tta_by_depth=base["tta"]["test_tta_by_depth"],
                                    test_no_tta_by_depth=base["test_no_tta_by_depth"]) if base and base["tta"] else None),
        )

    out = dict(
        _header=dict(
            description="E2 (CR-19): validation-selected truncation depth and dynamic-exit threshold, "
                        "evaluated once on test. S1 goodness-sum readout, no TTA, EMA weights of stage1_best.",
            created=time.strftime("%Y-%m-%d"), script=str(Path(__file__).resolve()),
            dumps_dir=str(DUMPS), n_runs=len(runs), missing_or_failed=missing,
            rules=dict(
                T_iso_val="smallest depth d with acc_val(d) >= acc_val(L); applied once to test",
                T_argmax_val="argmax_d acc_val(d), ties -> smallest d; applied once to test",
                D_dynamic_exit=f"exit after block d<L when top1-top2 of running score > tau; tau grid = +-inf and "
                               f"{N_TAU_LEVELS} quantile levels of pooled val margins (depths 1..L-1); tau* = fewest val "
                               "mean blocks with acc_val(tau) >= acc_val(L); applied once to test"),
            delta_definition="delta_test_pp = 100*(test acc at the selected operating point - test acc at full depth L), same checkpoint, same 10k test images",
            flops=fl,
            sd="sample SD (ddof=1) across runs of a family",
            caveats=[
                "val and test are both single splits: 5k val -> 1 example = 0.02 pp; 10k test -> 0.01 pp",
                "C10 arms: 9 different training configurations at seed 42 (one checkpoint each), SAM off",
                "C100 trio: 3 variants x 3 seeds, L4/D256, single expert",
            ]),
        runs=runs, families=fams, consistency_with_rebuttal_test_dumps=old_cmp, claim_87_40=claim,
        seconds=round(time.time() - t0, 1),
    )
    C.write_json_atomic(OUT_JSON, out)
    for fam, s in fams.items():
        print(f"{fam}: T_iso_val dTest {s['T_iso_val']['delta_test_pp_mean']:+.3f}pp "
              f"(n>=0 {s['T_iso_val']['n_delta_ge_0']}/{s['T_iso_val']['n']}), saved {s['T_iso_val']['flops_saved_mean']:.3f} | "
              f"D dTest {s['D_dynamic_exit']['delta_test_pp_mean']:+.3f}pp saved {s['D_dynamic_exit']['flops_saved_mean']:.3f}")
    print(f"wrote {OUT_JSON} ({len(runs)} runs; missing/failed: {missing})")


if __name__ == "__main__":
    main()
