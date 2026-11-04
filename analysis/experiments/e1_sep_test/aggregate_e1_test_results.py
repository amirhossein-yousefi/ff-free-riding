#!/usr/bin/env python3
"""Aggregate the test-image recomputation of the deepest-block sep_curr_nl (paper App. Tab. e1_aligned, column
"Test images") into results/e1_test_sep.json, with the sep-vs-accuracy analysis of E1 tier 2.

Inputs : experiments/e1_sep_test/out/<name>__test.json, written by measure_sep_split.py --splits test (official
         test set, eval transform, one clean view; the validation-selected EMA checkpoint in eval mode; hard
         negatives mined by the EMA teacher through the trainer's own diagnostic path), for the 18 points of
         results/e1_sep_tier2.json, and the control <name>__train_clean.json (--splits train_clean: as many clean
         images of the run's own Stage-1 training split, same protocol), which separates clean vs augmented images
         from test vs training images. Accuracies are re-read from each run's events.jsonl (final_test_eval), as
         in aggregate_e1_tier2.py.
Checks : every point has a test output with status OK, the tier-2 point's checkpoint SHA-256, Stage-1 epoch and
         accuracies, and one protocol per dataset (data seed, order seed, batch size, batch count).
Analysis: the statistics of aggregate_e1_tier2.py, imported unchanged and called in the same order with the same
         seed (0) and permutation count, applied to the test-image separation.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "e1_sep"))
import ffcr_common as C  # noqa: E402
import aggregate_e1_tier2 as A  # noqa: E402  (E1 tier-2 statistics, reused unchanged)

OUT_DIR = C.EXP / "e1_sep_test" / "out"
TIER2_JSON = C.RESULTS / "e1_sep_tier2.json"
OUT_JSON = C.RESULTS / "e1_test_sep.json"


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(Path(C.HOME).resolve()).as_posix()
    except ValueError:
        return p.name


def device_label(dev):
    """'cuda (NVIDIA GB10; FP32, TF32 off)' -> 'NVIDIA GB10 GPU, FP32 (TF32 off)'; 'cpu' -> 'CPU, FP32'."""
    if dev == "cpu":
        return "CPU, FP32"
    assert dev.startswith("cuda (") and "FP32, TF32 off" in dev, dev
    return dev[len("cuda ("):].split(";")[0] + " GPU, FP32 (TF32 off)"


def analyse(points):
    """aggregate_e1_tier2.main()'s analysis block, verbatim in its statistics and call order."""
    rng = np.random.default_rng(0)
    sep = np.array([p["sep_test"] for p in points])
    groups = [p["config"] for p in points]
    analysis = {}
    for lab in A.ACC_KEYS:
        acc = np.array([100.0 * p["acc"][lab] for p in points])
        rho, p_rho = A.perm_p_global(sep, acc, A.spearman, rng)
        w_r, w_p = A.within_perm(sep, acc, groups, rng)
        cfgs = sorted(set(groups))
        ms = [float(np.mean([s for s, g in zip(sep, groups) if g == c])) for c in cfgs]
        ma = [float(np.mean([a for a, g in zip(acc, groups) if g == c])) for c in cfgs]
        b_rho, b_p = A.between_exact(ms, ma)
        per_cfg = {}
        for c in cfgs:
            ii = [i for i, g in enumerate(groups) if g == c]
            per_cfg[c] = dict(n=len(ii), sep=[float(sep[i]) for i in ii], acc_pct=[float(acc[i]) for i in ii],
                              spearman=(A.spearman(sep[ii], acc[ii]) if len(ii) >= 3 else None))
        fam = {}
        for ds in ("cifar10", "cifar100"):
            ii = [i for i, p in enumerate(points) if p["dataset"] == ds]
            if len(ii) >= 4:
                fr, fp = A.perm_p_global(sep[ii], acc[ii], A.spearman, rng)
                fam[ds] = dict(n=len(ii), spearman=fr, perm_p=fp)
        analysis[lab] = dict(
            accuracy_protocol=A.ACC_KEYS[lab],
            across_points=dict(n=len(points), spearman=rho, perm_p_two_sided=p_rho, n_perm=A.N_PERM),
            within_config=dict(statistic="Pearson r of config-centered sep vs config-centered acc",
                               r=w_r, stratified_perm_p_two_sided=w_p, n_perm=A.N_PERM, n_configs=len(cfgs)),
            across_configs=dict(n_configs=len(cfgs), configs=cfgs, sep_means=ms, acc_means_pct=ma,
                                spearman=b_rho, exact_perm_p_two_sided=b_p),
            per_config=per_cfg, per_dataset=fam)
    return analysis


def main():
    t2 = json.load(open(TIER2_JSON))
    t2pts = {p["name"]: p for p in t2["points"]}
    names = [n for n in C.E1_POINTS if n in t2pts]
    assert names == [p["name"] for p in t2["points"]], "tier-2 point order differs from E1_POINTS"
    points, proto, devices, shared = [], {}, set(), False
    for name in names:
        q = t2pts[name]
        r = json.load(open(OUT_DIR / f"{name}__test.json"))
        assert r["status"] == "OK" and r["split"] == "test", name
        assert r["ckpt"]["sha256"] == q["ckpt_sha256"], f"{name}: not the tier-2 checkpoint"
        assert r["epoch_stage1"] == q["epoch_stage1"], name
        assert r["gate"]["pass_"], f"{name}: identity gate"
        e = C.REG[name]
        fte, _ = C.read_events(e["run_dir"])
        acc = {k: fte[v] for k, v in A.ACC_KEYS.items()}
        assert all(abs(acc[k] - q["acc"][k]) < 1e-12 for k in acc), f"{name}: accuracies differ from tier 2"
        pr = r["protocol"]
        key = (pr["n_batches"], pr["batch_size"], pr["data_seed"], pr["order_seed"])
        assert proto.setdefault(e["dataset"], key) == key, f"{name}: protocol differs within {e['dataset']}"
        devices.add(device_label(r["device"]))
        shared |= "shared" in r["device"]           # the run's own device note (GPU shared with another job)
        # control: the same measurement on as many clean images of the run's own training split
        c = json.load(open(OUT_DIR / f"{name}__train_clean.json"))
        assert c["status"] == "OK" and c["split"] == "train_clean", name
        assert c["ckpt"]["sha256"] == q["ckpt_sha256"] and c["epoch_stage1"] == q["epoch_stage1"], name
        pc = c["protocol"]
        assert (pc["n_batches"], pc["batch_size"], pc["data_seed"], pc["order_seed"]) == key, f"{name}: control protocol"
        d = r["deepest_block"]
        points.append(dict(
            name=name, config=q["config"], dataset=e["dataset"], seed=e["seed"], ckpt_sha256=q["ckpt_sha256"],
            epoch_stage1=q["epoch_stage1"], sep_test=d["sep_curr_nl"], sep_test_se=d["se"],
            sep_train_clean=c["deepest_block"]["sep_curr_nl"], sep_train_clean_se=c["deepest_block"]["se"],
            sep_tier2=q["sep_tier2"], acc=acc,
            per_block_test=[b["measured"]["margin_curr_nl"]["mean"] for b in r["per_block"]],
            k_neg_per_block=[b["k_neg"] for b in r["per_block"]],
            n_batches=pr["n_batches"], batch_size=pr["batch_size"], n_images=pr["n_examples"],
            data_seed=pr["data_seed"], seconds=r["seconds"], seconds_train_clean=c["seconds"]))
    assert len(points) == t2["_header"]["n_points"] == 18 and len(devices) == 1, (len(points), devices)
    secs = [p["seconds"] for p in points]
    secs_c = [p["seconds_train_clean"] for p in points]
    out = dict(
        _header=dict(
            description="Deepest-block sep_curr_nl of the 18 E1 tier-2 points recomputed on the official test set "
                        "(the validation-selected stage1_best checkpoints, net_ema in eval mode, through the trainer's "
                        "diagnostic path with EMA-teacher hard negatives), vs accuracy of the same checkpoints "
                        "(App. Tab. e1_aligned, column Test images). It differs from tier 2 in the images (clean test "
                        "images, one view, instead of augmented training batches, three views), in their number and in the "
                        "device (a GPU, so the hard-negative draws differ).",
            created=time.strftime("%Y-%m-%d"), script=rel(__file__), measure_script=rel(HERE / "measure_sep_split.py"),
            n_points=len(points), missing_or_failed=[],
            protocol=dict(
                split="official test set, eval transform (ToTensor + Normalize), x1 = x2 = x3",
                per_dataset={ds: dict(n_batches=k[0], batch_size=k[1], n_images=k[0] * k[1], data_seed=k[2],
                                      order_seed=k[3]) for ds, k in sorted(proto.items())},
                weights="net_ema, eval mode", hnm="trainer _choose_hard_negatives with the EMA block",
                device=devices.pop(), gpu_shared_with_another_job=shared,
                train_clean_control="sep_train_clean: the same measurement (same run, checkpoint, protocol and image "
                                    "count) on clean images (eval transform, x1 = x2 = x3) drawn from the run's own "
                                    "Stage-1 training split; it separates clean vs augmented images from test vs "
                                    "training images"),
            counts=dict(test_above_tier2=sum(p["sep_test"] > p["sep_tier2"] for p in points),
                        train_clean_above_test=sum(p["sep_train_clean"] > p["sep_test"] for p in points),
                        n=len(points)),
            timing=dict(seconds_per_point_min_max=[min(secs), max(secs)], total_seconds=round(sum(secs), 1),
                        total_h=round(sum(secs) / 3600, 2),
                        train_clean_seconds_per_point_min_max=[min(secs_c), max(secs_c)],
                        train_clean_total_seconds=round(sum(secs_c), 1),
                        train_clean_total_h=round(sum(secs_c) / 3600, 2),
                        total_h_both=round((sum(secs) + sum(secs_c)) / 3600, 2),
                        note="wall seconds of each point's measure_sep_split.py run per split (checkpoint load, gate "
                             "lookup, measurement); test and train_clean ran on the same GPU"),
            disclosure="C10 gamma=0 seed 42 is excluded, as in tier 2: its published checkpoint is unrecoverable.",
            sd="per-point SE = per-batch SD (ddof=1)/sqrt(N batches)",
            exploratory="all correlations are exploratory (small n, configurations differ in many respects)"),
        points=points, analysis=analyse(points))
    C.write_json_atomic(OUT_JSON, out)
    cnt = out["_header"]["counts"]
    print(f"wrote {rel(OUT_JSON)}: {len(points)} points; test > tier 2 at {cnt['test_above_tier2']} of {cnt['n']}; "
          f"clean training images > test at {cnt['train_clean_above_test']} of {cnt['n']}")
    for lab, a in out["analysis"].items():
        print(f"  {lab}: across rho={a['across_points']['spearman']:.3f} p={a['across_points']['perm_p_two_sided']:.3f} | "
              f"within r={a['within_config']['r']:.3f} p={a['within_config']['stratified_perm_p_two_sided']:.3f} | "
              f"configs rho={a['across_configs']['spearman']:.3f} p={a['across_configs']['exact_perm_p_two_sided']:.3f}")


if __name__ == "__main__":
    main()
