#!/usr/bin/env python3
"""Dump per-block per-class goodness for the CIFAR-100 trio (9 checkpoints):
  - VAL:  the run's own 5000-example validation split (CIFAR-100 TRAIN images at
          split_indices.json val_idx, eval transform, no augmentation), plus the
          horizontally flipped copies (needed only for the best_s1 TTA anchor).
  - TEST: full official 10k test set.

Weights: net_ema from stage1_best.pt (the checkpoint whose EMA weights produced both
events.jsonl anchors). Per-block goodness g_d(x,c) [N,L,C] obtained from the training
module's own predict_scores(return_all_depths=True) (cumulative) by first-differencing
along depth, so R0 = sum_d g_d reproduces predict_scores exactly by construction.

Gates per checkpoint (tol 5e-4):
  G1: acc(argmax sum_d goodness_test) == events.jsonl final_test_eval stage1_test_top1_no_tta
  G2: acc(argmax [sum_d goodness_val + sum_d goodness_val_flip]) == events.jsonl run_end best_s1
Extra: ckpt['best_s1'] == run_end best_s1 (checkpoint identity), batch-0 plain
predict_scores vs cumulative-last max-abs-diff, strict state_dict load.

Single process, sequential checkpoints, torch.no_grad. GPU-exclusive machine.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import argparse, json, os, sys, time
import numpy as np
import torch

SCRATCH = FF_WORK
TRAIN_DIR = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/revised_training/extracted/ff_cifar100_multiseed"
DATA_ROOT = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/mgc_rebuttal/data_c100"
CKPT_ROOT = os.path.join(SCRATCH, "c100_ckpts", "ff_cifar100_multiseed")
OUT_DIR = os.path.join(SCRATCH, "dumps_val")
MANIFEST = os.path.join(OUT_DIR, "dump_val_manifest.json")

sys.path.insert(0, TRAIN_DIR)
import basic_fair_revised_valfix_cp_fair_cifar100_fixed as M  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)  # CIFAR-100 lineage std (block_dumps_manifest normalization_note)

RUNS = {  # name -> (variant, seed, run_dir basename)
    "c100_gamma0_seed42":      ("gamma0", 42,  "runs_gamma0_seed42/ff_cifar100_cifar100_gamma0_seed42_L4_D256_bs256_seed42_20260420_101044"),
    "c100_gamma0_seed123":     ("gamma0", 123, "runs_gamma0_seed123/ff_cifar100_cifar100_gamma0_seed123_L4_D256_bs256_seed123_20260421_124316"),
    "c100_gamma0_seed456":     ("gamma0", 456, "runs_gamma0_seed456/ff_cifar100_cifar100_gamma0_seed456_L4_D256_bs256_seed456_20260422_142245"),
    "c100_gated_k0_seed42":    ("gated_k0", 42,  "runs_gated_k0_seed42/ff_cifar100_cifar100_gated_k0_seed42_L4_D256_bs256_seed42_20260422_185437"),
    "c100_gated_k0_seed123":   ("gated_k0", 123, "runs_gated_k0_seed123/ff_cifar100_cifar100_gated_k0_seed123_L4_D256_bs256_seed123_20260423_150626"),
    "c100_gated_k0_seed456":   ("gated_k0", 456, "runs_gated_k0_seed456/ff_cifar100_cifar100_gated_k0_seed456_L4_D256_bs256_seed456_20260424_153513"),
    "c100_cumulative_seed42":  ("cumulative", 42,  "runs_cumulative_seed42/ff_cifar100_cifar100_cumulative_seed42_L4_D256_bs256_seed42_20260418_032149"),
    "c100_cumulative_seed123": ("cumulative", 123, "runs_cumulative_seed123/ff_cifar100_cifar100_cumulative_seed123_L4_D256_bs256_seed123_20260419_010632"),
    "c100_cumulative_seed456": ("cumulative", 456, "runs_cumulative_seed456/ff_cifar100_cifar100_cumulative_seed456_L4_D256_bs256_seed456_20260420_051234"),
}


def read_anchors(run_dir):
    fte, rend = None, None
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("event") == "final_test_eval":
                fte = ev
            elif ev.get("event") == "run_end":
                rend = ev
    return fte["stage1_test_top1_no_tta"], rend["best_s1"]


def build_net(config_path, device):
    cfg = M.FFConfig()
    persisted = json.load(open(config_path))
    for k, v in persisted.items():
        if k == "device":
            continue
        setattr(cfg, k, v)
    net = M.FFHybridNet(cfg).to(device)
    net.eval()
    return net, cfg


@torch.no_grad()
def dump_split(net, loader, device, also_flip=False, check_batch0=True):
    """Returns per-block goodness increments [N,L,C] (float32 numpy), labels [N],
    optionally the flipped-image goodness, and the batch-0 cross-check diff."""
    inc_all, inc_flip_all, ys = [], [], []
    b0_diff = None
    for bi, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        sd = net.predict_scores(images, return_all_depths=True)  # [B,L,C] cumulative
        inc = sd.clone()
        inc[:, 1:, :] -= sd[:, :-1, :]
        inc_all.append(inc.cpu().numpy().astype(np.float32))
        if check_batch0 and bi == 0:
            plain = net.predict_scores(images)  # [B,C]
            b0_diff = float((plain - sd[:, -1, :]).abs().max().item())
        if also_flip:
            sdf = net.predict_scores(torch.flip(images, dims=[3]), return_all_depths=True)
            incf = sdf.clone()
            incf[:, 1:, :] -= sdf[:, :-1, :]
            inc_flip_all.append(incf.cpu().numpy().astype(np.float32))
        ys.append(labels.numpy())
    out = dict(inc=np.concatenate(inc_all), y=np.concatenate(ys).astype(np.int16), b0_diff=b0_diff)
    if also_flip:
        out["inc_flip"] = np.concatenate(inc_flip_all)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=sorted(RUNS.keys()))
    ap.add_argument("--matmul", default="highest", choices=["high", "highest"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    torch.set_float32_matmul_precision(args.matmul)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR100(DATA_ROOT, train=True, download=False, transform=eval_tf)
    test_ds = datasets.CIFAR100(DATA_ROOT, train=False, download=False, transform=eval_tf)

    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"_header": {
        "description": "VAL(5k, own split, eval tf, + flipped copies for the TTA anchor) and TEST(10k) per-block per-class goodness dumps, net_ema of stage1_best.pt, CIFAR-100 trio.",
        "created": time.strftime("%Y-%m-%d"),
        "script": os.path.abspath(__file__),
        "matmul_precision": args.matmul,
    }, "dumps": {}}

    for name in args.names:
        variant, seed, run_rel = RUNS[name]
        run_dir = os.path.join(TRAIN_DIR, run_rel)
        t0 = time.time()
        anchor_test, anchor_val_tta = read_anchors(run_dir)
        split = json.load(open(os.path.join(run_dir, "split_indices.json")))
        val_idx = split["val_idx"]
        assert len(val_idx) == 5000 and split["n_total"] == 50000

        ckpt_dir = os.path.join(CKPT_ROOT, f"checkpoints_{variant}_seed{seed}")
        ckpt_path = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith("stage1_best.pt")]
        assert len(ckpt_path) == 1, ckpt_path
        ckpt_path = ckpt_path[0]

        net, cfg = build_net(os.path.join(run_dir, "config.json"), device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt.get("net_ema") is not None
        net.load_state_dict(ckpt["net_ema"], strict=True)
        ckpt_best_s1 = float(ckpt.get("best_s1", float("nan")))
        del ckpt

        val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=args.batch, shuffle=False,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=2, pin_memory=True)

        val = dump_split(net, val_loader, device, also_flip=True)
        test = dump_split(net, test_loader, device, also_flip=False)

        # Gates
        r0_test = float((test["inc"].sum(1).argmax(1) == test["y"]).mean())
        val_sum = val["inc"].sum(1)
        val_tta = float(((val_sum + val["inc_flip"].sum(1)).argmax(1) == val["y"]).mean())
        r0_val_no_tta = float((val_sum.argmax(1) == val["y"]).mean())
        g1 = abs(r0_test - anchor_test) <= args.tol
        g2 = abs(val_tta - anchor_val_tta) <= args.tol
        g3 = abs(ckpt_best_s1 - anchor_val_tta) <= 1e-9

        npz_path = os.path.join(OUT_DIR, f"block_dump_valtest_{name}.npz")
        np.savez_compressed(npz_path,
                            goodness_val=val["inc"], goodness_val_flip=val["inc_flip"],
                            goodness_test=test["inc"],
                            y_val=val["y"], y_test=test["y"],
                            val_idx=np.asarray(val_idx, dtype=np.int32))
        entry = dict(name=name, variant=variant, seed=seed, run_dir=run_dir, ckpt=ckpt_path,
                     weights_used="net_ema", ckpt_best_s1=ckpt_best_s1,
                     anchor_test_s1_top1_no_tta=anchor_test, r0_test_recomputed=round(r0_test, 6),
                     anchor_val_best_s1_tta=anchor_val_tta, val_tta_recomputed=round(val_tta, 6),
                     r0_val_no_tta=round(r0_val_no_tta, 6),
                     gate_test=bool(g1), gate_val_tta=bool(g2), gate_ckpt_best_s1=bool(g3),
                     b0_diff_val=val["b0_diff"], b0_diff_test=test["b0_diff"],
                     shapes=dict(goodness_val=list(val["inc"].shape), goodness_test=list(test["inc"].shape)),
                     npz=npz_path, seconds=round(time.time() - t0, 1))
        manifest["dumps"][name] = entry
        json.dump(manifest, open(MANIFEST, "w"), indent=1)
        status = "PASS" if (g1 and g2) else "FAIL"
        print(f"[{status}] {name}: R0_test={r0_test:.4f} (anchor {anchor_test:.4f}) "
              f"val_tta={val_tta:.4f} (anchor {anchor_val_tta:.4f}) ckpt_best_s1={ckpt_best_s1:.4f} "
              f"b0diff={test['b0_diff']:.2e} {entry['seconds']}s", flush=True)

        del net
        torch.cuda.empty_cache()

    n_pass = sum(1 for e in manifest["dumps"].values() if e["gate_test"] and e["gate_val_tta"])
    print(f"DONE: {n_pass}/{len(manifest['dumps'])} dumps pass both gates")


if __name__ == "__main__":
    main()
