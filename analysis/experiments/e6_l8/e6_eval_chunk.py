#!/usr/bin/env python3
"""E6: CPU test evaluation of the L8/D128 kappa=0 seed-42 run (gated ablation, adaptive t1_k0).

The paper's row prints seed 42 = 86.88. That value is the run's *validation* best_s2; the run never logged a
final_test_eval. This script evaluates, on one chunk of a split at a time (each call stays under 9 minutes on CPU):
  * S1  = stage1_best EMA backbone, FF goodness-sum over the 10 label hypotheses (trainer: predict_scores)
  * S2  = stage2_best attentive head on the frozen stage-2 backbone (trainer: evaluate_stage2)
  * both on the original image and on its horizontal flip (TTA = sum of S1 scores / mean of S2 logits)
In one forward pass per (image, flip, label) it collects both the aggregated goodness (S1) and the hybrid features (S2).
The stage-2 checkpoint's net_ema is asserted identical to the stage1_best net_ema, so one backbone serves both.

Usage (CPU only):
  CUDA_VISIBLE_DEVICES='' python e6_eval_chunk.py --split val  --start 0 --end 2000
  ... then e6_aggregate.py
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HOME = FF_HOME
CF = f"{HOME}/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
L8 = f"{CF}/basic_fair_revised_valfix_cp_fair/ff_gated_ablation_L8"
RUN = f"{L8}/runs/ff_cifar10_adaptive_t1_k0_L8_L8_D128_bs512_seed42_20260315_201449"
CK1 = f"{L8}/checkpoints/ff_cifar10_adaptive_t1_k0_L8_L8_D128_P2_seed42_stage1_best.pt"
CK2 = f"{L8}/checkpoints/ff_cifar10_adaptive_t1_k0_L8_L8_D128_P2_seed42_stage2_best.pt"
REF_SPLIT = (f"{CF}/basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/runs/"
             "ff_cifar10_adaptive_t1_k0_L4_D128_bs512_seed42_20260314_032410/split_indices.json")
DATA_DIR = f"{CF}/revised_training/data_c10"
TRAINER_DIR = f"{CF}/basic_fair_revised_valfix_cp_fair_hardness_gated"   # the trainer that produced this run
TRAINER_MODULE = "basic_fair_revised_valfix_cp_fair_hardness_gated"
OUT = Path(__file__).resolve().parent / "out"
os.environ["FF_CKPT_DIR"] = (FF_WORK + "/trainer_ckpt_dir_unused")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(6)
sys.path.insert(0, TRAINER_DIR)
try:
    import tqdm.auto  # noqa: F401
except ImportError:  # the trainer imports tqdm for progress bars only
    import importlib.machinery
    import types
    import torchvision  # noqa: F401
    _tq, _tqa = types.ModuleType("tqdm"), types.ModuleType("tqdm.auto")
    _tq.__spec__ = importlib.machinery.ModuleSpec("tqdm", None)
    _tqa.__spec__ = importlib.machinery.ModuleSpec("tqdm.auto", None)

    class _NoTqdm:
        def __init__(self, it=None, *a, **k):
            self.it = it

        def __iter__(self):
            return iter(self.it)

        def __getattr__(self, name):
            return lambda *a, **k: None

    _tq.tqdm = _tqa.tqdm = _NoTqdm
    _tq.auto = _tqa
    sys.modules["tqdm"], sys.modules["tqdm.auto"] = _tq, _tqa
trainer = __import__(TRAINER_MODULE)

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)   # trainer make_loaders_stage1 eval_tf


def build_cfg():
    c = json.load(open(f"{RUN}/config.json"))
    cfg = trainer.FFConfig()
    cfg.__post_init__()
    for k, v in c.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.aspect_loss_types = list(cfg.aspect_loss_types[: cfg.n_aspects])
    cfg.aspect_lambdas = list(cfg.aspect_lambdas[: cfg.n_aspects])
    cfg.aspect_theta_inits = list(cfg.aspect_theta_inits[: cfg.n_aspects])
    cfg.device = "cpu"
    return cfg, c


def load_models(cfg):
    c1 = torch.load(CK1, map_location="cpu", weights_only=False)
    c2 = torch.load(CK2, map_location="cpu", weights_only=False)
    meta = {"stage1_epoch": c1["epoch_stage1"], "stage1_best_s1": c1["best_s1"],
            "stage2_epoch_stage1": c2["epoch_stage1"], "stage2_epoch_stage2": c2["epoch_stage2"],
            "stage2_best_s2": c2["best_s2"], "stage2_best_s1": c2["best_s1"]}
    same = all(torch.equal(c1["net_ema"][k], c2["net_ema"][k]) for k in c1["net_ema"]) and set(c1["net_ema"]) == set(c2["net_ema"])
    meta["stage2_backbone_equals_stage1_best_ema"] = bool(same)
    assert same
    net = trainer.FFHybridNet(cfg)
    r = net.load_state_dict(c1["net_ema"], strict=True)
    meta["net_ema_missing"], meta["net_ema_unexpected"] = list(r.missing_keys), list(r.unexpected_keys)
    head = trainer.AttentiveHybridHead(in_dim_g=cfg.num_blocks * cfg.n_aspects, in_dim_a=cfg.d_model * cfg.num_blocks,
                                       hidden=cfg.head_hidden, num_classes=cfg.num_classes,
                                       num_heads=cfg.head_num_heads, dropout=cfg.head_dropout)
    r2 = head.load_state_dict(c2["head_state"], strict=True)
    meta["head_missing"], meta["head_unexpected"] = list(r2.missing_keys), list(r2.unexpected_keys)
    ck_cfg = c1["cfg"]
    for k in ("num_blocks", "d_model", "gamma_gating_mode", "gamma_tau", "gamma_kappa", "use_sam", "seed", "eval_tta",
              "head_hidden", "head_num_heads", "n_aspects", "n_experts", "moe_top_k"):
        assert ck_cfg.get(k) == getattr(cfg, k), (k, ck_cfg.get(k), getattr(cfg, k))
    del c1, c2
    net.eval()
    head.eval()
    return net, head, meta


@torch.no_grad()
def forward_all(net, images):
    """One pass per label: S1 scores (predict_scores order) + hybrid features (extract_hybrid_features_all_labels)."""
    net.eval()
    B = images.size(0)
    tokens0 = net.embed(images)
    scores, feats_g, feats_a = [], [], []
    for k in range(net.num_classes):
        yk = torch.full((B,), k, dtype=torch.long)
        tokens = tokens0
        g_hist, a_hist = [], []
        total_g = torch.zeros(B)
        for blk in net.blocks:
            g_prev = torch.stack(g_hist, dim=1) if g_hist else None
            if g_prev is not None:
                g_prev = g_prev.detach()
            a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
            total_g = total_g + blk.aggregate_goodness(g_vec)
            g_hist.append(g_vec)
            a_hist.append(blk.get_pooled_rep(a))
            tokens = trainer.l2_normalize_tokens(a)
        scores.append(total_g.unsqueeze(1))
        feats_g.append(torch.cat(g_hist, dim=1).unsqueeze(1))
        feats_a.append(torch.cat(a_hist, dim=1).unsqueeze(1))
    return torch.cat(scores, 1), torch.cat(feats_g, 1), torch.cat(feats_a, 1)


@torch.no_grad()
def head_logits(head, fg, fa):
    B, C = fg.shape[:2]
    return head(fg.reshape(B * C, -1), fa.reshape(B * C, -1))


def dataset(split):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if split == "test":
        ds = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=tf)
        return ds, list(range(len(ds)))
    ds = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=tf)
    rng = random.Random(42)                              # trainer make_loaders_stage1, seed 42, val_fraction 0.1
    perm = list(range(len(ds)))
    rng.shuffle(perm)
    val_idx = sorted(int(i) for i in perm[:int(round(len(ds) * 0.1))])
    ref = json.load(open(REF_SPLIT))["val_idx"]
    assert val_idx == ref, "regenerated val split differs from stored seed-42 split_indices.json"
    return ds, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--selfcheck", action="store_true", help="compare forward_all with trainer methods on 8 images")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    cfg, _ = build_cfg()
    net, head, meta = load_models(cfg)
    ds, idx = dataset(args.split)
    sel = idx[args.start:args.end]
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(ds, sel), batch_size=256, shuffle=False, num_workers=2)
    if args.selfcheck:
        x, _ = next(iter(DataLoader(Subset(ds, sel[:8]), batch_size=8)))
        s, fg, fa = forward_all(net, x)
        s_ref = net.predict_scores(x)
        fg_ref, fa_ref = net.extract_hybrid_features_all_labels(x)
        meta["selfcheck_max_abs_diff"] = {"scores": float((s - s_ref).abs().max()), "feats_g": float((fg - fg_ref).abs().max()),
                                          "feats_a": float((fa - fa_ref).abs().max())}
        print("selfcheck", meta["selfcheck_max_abs_diff"])
    ys, s1o, s1f, s2o, s2f = [], [], [], [], []
    for images, y in loader:
        s, fg, fa = forward_all(net, images)
        s1o.append(s)
        s2o.append(head_logits(head, fg, fa))
        flipped = torch.flip(images, dims=[3])
        s, fg, fa = forward_all(net, flipped)
        s1f.append(s)
        s2f.append(head_logits(head, fg, fa))
        ys.append(y)
    out = OUT / f"{args.split}_{args.start:05d}_{args.end:05d}.npz"
    np.savez_compressed(out, idx=np.array(sel), y=torch.cat(ys).numpy(),
                        s1_orig=torch.cat(s1o).numpy(), s1_flip=torch.cat(s1f).numpy(),
                        s2_orig=torch.cat(s2o).numpy(), s2_flip=torch.cat(s2f).numpy())
    meta.update({"split": args.split, "start": args.start, "end": args.end, "n": len(sel), "seconds": time.time() - t0,
                 "torch": torch.__version__, "threads": torch.get_num_threads(), "trainer": f"{TRAINER_DIR}/{TRAINER_MODULE}.py"})
    json.dump(meta, open(str(out).replace(".npz", ".json"), "w"), indent=1)
    y = torch.cat(ys).numpy()
    acc = lambda sc: float((np.asarray(sc).argmax(1) == y).mean())  # noqa: E731
    s1o_, s1f_, s2o_, s2f_ = (torch.cat(v).numpy() for v in (s1o, s1f, s2o, s2f))
    print(f"{args.split}[{args.start}:{args.end}] S1={acc(s1o_):.4f} S1-TTA={acc(s1o_ + s1f_):.4f} "
          f"S2={acc(s2o_):.4f} S2-TTA={acc((s2o_ + s2f_) / 2):.4f}  ({meta['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
