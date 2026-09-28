#!/usr/bin/env python3
"""E2 (CR-19) dumps: per-depth cumulative class scores on each run's OWN validation split and on
the official test split, for validation-selected early exit / truncation.

Port of analysis/recovered/dump_val_test_block_data.py (rebuttal-era CIFAR-100 trio dumper):
  * paths repointed (experiments/ffcr_common.py; the old :25-33 hard-coded temporary and home
    paths are gone; the trio checkpoints are read from the FLAT ff-next/checkpoints/trio/ layout instead
    of checkpoints_<variant>_seed<seed>/ at the old :136-139);
  * a CIFAR-10 mode: the 9 gated L4/D128 seed-42 arms, loaded with the trainer that trained them
    (hardness_gated), CIFAR-10 normalization mean (0.4914,0.4822,0.4465) std (0.2023,0.1994,0.2010)
    (asserted on the trainer's own eval transform); t1_k4 builds its config from ckpt['cfg']
    (its config.json is an HTML 404 page); prev_only uses the completed run _042603;
  * the loaders are the trainer's own make_loaders_stage1(): VAL = the run's 5k split (CIFAR train
    images at split_indices.json val_idx, asserted identical, eval transform, sorted order),
    TEST = official test split;
  * we store the CUMULATIVE scores S_d(x,c) = sum_{j<=d} g_j(x,c) exactly as the trainer's
    predict_scores(return_all_depths=True) returns them (float32), so every per-depth accuracy is
    bit-identical to the trainer's evaluate_ff_by_depth; per-block goodness = first difference.

Arrays in dumps/block_dump_valtest_<name>.npz (large; not distributed):
  scores_val [5000,L,C], scores_val_flip [5000,L,C], scores_test [10000,L,C],
  scores_test_flip [10000,L,C] (CIFAR-10 only, for the TTA version of the 87.40 claim),
  y_val, y_test, val_idx.
Sidecar dumps/<name>.json: gates, provenance (SHA-256 of checkpoint and trainer), timings.

Gates.
  HARD  G_test : acc(argmax S_test[:,L]) vs final_test_eval stage1_test_top1_no_tta within 1e-4
                 (0.01 pp, one image) -- the identity gate on the split the log reports. Pre-declared
                 numerics fallback (ffcr_common.test_gate): |delta| <= 3 images AND >= 99.9% argmax
                 agreement with the run's stored preds_stage1_test_no_tta.npz (the trainers evaluated
                 with TF32 on A100s; we recompute in exact FP32 on CPU) -> status PASS_NUMERICS.
  HARD  G_ckpt : ckpt best_s1 == run_end best_s1 and ckpt epoch_stage1 == best-val epoch.
  CONSISTENCY (<= 5 images; near-tie flips of TTA sums are expected under TF32 vs FP32):
        val flip-TTA at full depth vs run_end best_s1; val flip-TTA at EVERY depth vs epoch_metrics
        eval_top1_d<d> at the best-val epoch (also validates the val indexing: a wrong split would
        be off by whole percentage points); (CIFAR-10) test flip-TTA vs stage1_test_top1_tta.
  Status: PASS (strict gate) | PASS_NUMERICS (fallback) | FAIL. --regate re-scores an existing dump.
Recorded, not gated: batch-0 predict_scores vs cumulative-last max-abs-diff, score-level comparison
with the stored GPU predictions (median / 99.9th pct / max |diff|).

CPU only: CUDA_VISIBLE_DEVICES=''; torch threads = 6; resumable (skips finished runs).
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ffcr_common as C  # noqa: E402

torch = C.setup_env(threads=int(os.environ.get("FFCR_THREADS", "6")))
import numpy as np  # noqa: E402

OUT_DIR = C.EXP / "e2_exit" / "dumps"


@torch.no_grad()
def dump_split(net, loader, also_flip, check_batch0=True, tag=""):
    """Cumulative per-depth class scores [N,L,C] (float32) from the trainer's predict_scores."""
    S, Sf, ys = [], [], []
    b0 = None
    t0 = time.time()
    n = 0
    for bi, batch in enumerate(loader):
        images, labels = batch[0], batch[-1]
        sd = net.predict_scores(images, return_all_depths=True)          # [B,L,C] cumulative
        S.append(sd.numpy().astype(np.float32))
        if check_batch0 and bi == 0:
            plain = net.predict_scores(images)                           # [B,C]
            b0 = float((plain - sd[:, -1, :]).abs().max().item())
        if also_flip:
            sdf = net.predict_scores(torch.flip(images, dims=[3]), return_all_depths=True)
            Sf.append(sdf.numpy().astype(np.float32))
        ys.append(labels.numpy())
        n += images.size(0)
        if bi % 10 == 0:
            print(f"    {tag} batch {bi}: {n} images, {time.time() - t0:.0f}s", flush=True)
    out = dict(S=np.concatenate(S), y=np.concatenate(ys).astype(np.int16), b0_diff=b0)
    if also_flip:
        out["Sf"] = np.concatenate(Sf)
    return out


def acc_at(S, y, d):
    return float((S[:, d].argmax(1) == y).mean())


SOFT_MAX_EXAMPLES = 5   # consistency checks (TTA sums, per-depth val): <= 5 images (val 0.1 pp, test 0.05 pp)


def run_one(name, args):
    t0 = time.time()
    e = C.REG[name]
    M = C.load_trainer(e["trainer"])
    ckpt_id = C.file_id(e["ckpt"])
    npz_path = OUT_DIR / f"block_dump_valtest_{name}.npz"
    ckpt = torch.load(e["ckpt"], map_location="cpu", weights_only=True, mmap=True)
    persisted, cfg_src = C.persisted_config(e, ckpt)
    cfg = C.build_cfg(M, persisted)
    cfg_diff = None if e.get("cfg_from_ckpt") else C.cfg_vs_ckpt_diff(persisted, ckpt)
    E = int(ckpt["epoch_stage1"])
    ckpt_best_s1 = float(ckpt["best_s1"])
    L = int(cfg.num_blocks)
    is_c10 = int(cfg.num_classes) == 10
    test_flip = {"auto": is_c10, "on": True, "off": False}[args.test_flip]

    if args.regate:   # re-evaluate the gates of an existing dump without recomputing it
        z = np.load(npz_path)
        arrays = {k: z[k] for k in z.files}
        b0 = dict(val=None, test=None)
        prev = json.load(open(OUT_DIR / f"{name}.json")) if (OUT_DIR / f"{name}.json").exists() else {}
        b0 = dict(val=prev.get("b0_diff_val"), test=prev.get("b0_diff_test"))
        dump_seconds = prev.get("seconds_dump", prev.get("seconds"))
        _, _, _, _, run_split = C.loaders(M, cfg, e["run_dir"], workers=0)
        assert np.array_equal(arrays["val_idx"], np.asarray(run_split["val_idx"], dtype=np.int32))
        del ckpt
    else:
        net = C.load_nets(M, cfg, ckpt, which=("net_ema",))["net_ema"]
        del ckpt
        _, val_loader, test_loader, split_info, run_split = C.loaders(M, cfg, e["run_dir"], workers=0)
        val_idx = run_split["val_idx"]
        assert len(val_idx) == 5000 and run_split["n_total"] == 50000
        td = time.time()
        print(f"[{name}] VAL (5000, + flipped)...", flush=True)
        val = dump_split(net, val_loader, also_flip=True, tag="val")
        print(f"[{name}] TEST (10000{', + flipped' if test_flip else ''})...", flush=True)
        test = dump_split(net, test_loader, also_flip=test_flip, tag="test")
        dump_seconds = round(time.time() - td, 1)
        arrays = dict(scores_val=val["S"], scores_val_flip=val["Sf"], scores_test=test["S"],
                      y_val=val["y"], y_test=test["y"], val_idx=np.asarray(val_idx, dtype=np.int32))
        if test_flip:
            arrays["scores_test_flip"] = test["Sf"]
        b0 = dict(val=val["b0_diff"], test=test["b0_diff"])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, npz_path)

    fte, rend = C.read_events(e["run_dir"])
    best_row = C.best_epoch_row(e)
    Sv, Sfv, St = arrays["scores_val"], arrays["scores_val_flip"], arrays["scores_test"]
    yv, yt = arrays["y_val"].astype(np.int64), arrays["y_test"].astype(np.int64)
    nv, nt = len(yv), len(yt)

    # ---- hard gates: strict identity on the logged S1 no-TTA test accuracy (+ numerics fallback) ----
    gate = C.test_gate(St[:, -1].argmax(1), yt, fte["stage1_test_top1_no_tta"], e["run_dir"], scores=St[:, -1])
    g_ckpt = abs(ckpt_best_s1 - float(rend["best_s1"])) < 1e-9 and best_row is not None and int(best_row["epoch"]) == E

    # ---- consistency checks (TTA sums and per-depth val are near-tie sensitive; <= 5 images) ----
    val_tta_depth = [acc_at(Sv + Sfv, yv, d) for d in range(L)]
    val_no_tta_depth = [acc_at(Sv, yv, d) for d in range(L)]
    test_no_tta_depth = [acc_at(St, yt, d) for d in range(L)]
    logged_val_depth = [best_row.get(f"eval_top1_d{d + 1}") if best_row else None for d in range(L)]
    diff_val_depth = [None if v is None else int(round(abs(a - float(v)) * nv)) for a, v in zip(val_tta_depth, logged_val_depth)]
    diff_val_full = int(round(abs(val_tta_depth[-1] - float(rend["best_s1"])) * nv))
    test_tta_depth, diff_test_tta = None, None
    if "scores_test_flip" in arrays:
        test_tta_depth = [acc_at(St + arrays["scores_test_flip"], yt, d) for d in range(L)]
        diff_test_tta = int(round(abs(test_tta_depth[-1] - float(fte["stage1_test_top1_tta"])) * nt))
    soft = dict(
        val_tta_full_vs_best_s1_examples=diff_val_full,
        val_tta_by_depth_vs_logged_examples=diff_val_depth,
        test_tta_vs_events_examples=diff_test_tta,
        max_examples_allowed=SOFT_MAX_EXAMPLES,
    )
    soft_ok = (diff_val_full <= SOFT_MAX_EXAMPLES
               and all(x is not None and x <= SOFT_MAX_EXAMPLES for x in diff_val_depth)
               and (diff_test_tta is None or diff_test_tta <= SOFT_MAX_EXAMPLES))
    soft_exact = (diff_val_full == 0 and all(x == 0 for x in diff_val_depth) and diff_test_tta in (None, 0))
    soft["all_exact"] = bool(soft_exact)
    soft["pass_"] = bool(soft_ok)

    if gate["pass_"] and g_ckpt and soft_ok:
        status = "PASS" if gate["strict_pass"] else "PASS_NUMERICS"
    else:
        status = "FAIL"
    entry = dict(
        name=name, status=status, dataset=e["dataset"], config=e["config"], seed=e["seed"],
        run_dir=str(e["run_dir"]), ckpt=ckpt_id["path"], ckpt_size=ckpt_id["size"], ckpt_sha256=ckpt_id["sha256"],
        trainer=str(C.TRAINERS[e["trainer"]]), trainer_sha256=C.sha256(C.TRAINERS[e["trainer"]]),
        config_source=cfg_src, config_json_vs_ckpt_cfg_diff=cfg_diff,
        weights_used="net_ema (eval mode)", normalization=C.NORM[int(cfg.num_classes)],
        epoch_stage1=E, best_val_epoch=(int(best_row["epoch"]) if best_row else None),
        ckpt_best_s1=ckpt_best_s1, run_end_best_s1=float(rend["best_s1"]),
        anchor_test_s1_top1_no_tta=float(fte["stage1_test_top1_no_tta"]), r0_test_recomputed=round(gate["recomputed"], 6),
        anchor_test_s1_top1_tta=float(fte["stage1_test_top1_tta"]),
        test_tta_recomputed=(round(test_tta_depth[-1], 6) if test_tta_depth else None),
        val_tta_by_depth=[round(a, 6) for a in val_tta_depth], logged_val_tta_by_depth=logged_val_depth,
        val_no_tta_by_depth=[round(a, 6) for a in val_no_tta_depth],
        test_no_tta_by_depth=[round(a, 6) for a in test_no_tta_depth],
        test_tta_by_depth=([round(a, 6) for a in test_tta_depth] if test_tta_depth else None),
        gate_test=gate, gate_test_pass=bool(gate["pass_"]), gate_ckpt=bool(g_ckpt), consistency=soft,
        b0_diff_val=b0["val"], b0_diff_test=b0["test"],
        shapes={k: list(v.shape) for k, v in arrays.items()},
        npz=str(npz_path), seconds_dump=dump_seconds, seconds=round(time.time() - t0, 1),
        torch=torch.__version__, threads=torch.get_num_threads(),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
        regated=bool(args.regate),
    )
    C.write_json_atomic(OUT_DIR / f"{name}.json", entry)
    print(f"[{status}] {name}: S1 test {gate['recomputed']:.4f} (events {fte['stage1_test_top1_no_tta']:.4f}, "
          f"{gate['delta_examples']} img; stored-preds agreement "
          f"{(gate['stored_comparison'] or {}).get('argmax_agreement')}) | val TTA by depth "
          f"{['%.4f' % a for a in val_tta_depth]} (logged {logged_val_depth}; diff imgs {diff_val_depth}) | "
          + (f"test TTA {test_tta_depth[-1]:.4f} (events {fte['stage1_test_top1_tta']:.4f}) | " if test_tta_depth else "")
          + f"ckpt gate {g_ckpt} | dump {dump_seconds}s", flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=list(C.E2_C10) + list(C.E2_C100))
    ap.add_argument("--test-flip", choices=["auto", "on", "off"], default="auto",
                    help="dump flipped test images (auto: CIFAR-10 only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--regate", action="store_true",
                    help="re-evaluate the gates of existing dumps (no recomputation)")
    args = ap.parse_args()
    n_fail = 0
    for name in args.names:
        side = OUT_DIR / f"{name}.json"
        npz = OUT_DIR / f"block_dump_valtest_{name}.npz"
        if args.regate and not npz.exists():
            print(f"[{name}] --regate: no dump yet; skipping", flush=True)
            continue
        if side.exists() and npz.exists() and not args.force:
            prev = json.load(open(side))
            if prev.get("status") in ("PASS", "PASS_NUMERICS") and not args.regate:
                print(f"[{name}] done already; skipping", flush=True)
                continue
        print(f"\n===== {name} =====", flush=True)
        entry = run_one(name, args)
        n_fail += entry["status"] not in ("PASS", "PASS_NUMERICS")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
