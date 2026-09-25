"""Verify that no gradient from a deeper block's loss reaches prior blocks' parameters.

Two checks, both CPU-only:

1. Cumulative-goodness path (v1 check): the patched _sum_prev_goodness /
   _prev_only_goodness do not leak gradients from block-l's loss into blocks[0..l-1]
   (a mock of the accumulation path, one backward).
2. Depth-order path (--depth-order, added for the camera-ready release): runs the
   trainer's REAL train_epoch_stage1 on a tiny configuration with synthetic data and
   gradient hooks on every parameter. A gradient that arrives on block j while block
   l != j is being trained is a leak. In the v1 trainers the depth-order loss called
   blocks[l-1].aggregate_goodness(...) with grad enabled, so block l's backward wrote a
   gradient into blocks[l-1].alpha_raw.grad; it was discarded by
   opt_{l-1}.zero_grad(set_to_none=True) before block l-1's next step. The camera-ready
   trainers wrap those calls in torch.no_grad(). With --v1-dir (or FF_V1_TRAINERS) the
   check also runs the v1 trainer on the same data and seeds, reports its (discarded)
   leak, and asserts that parameters and EMA parameters after N steps are bit-identical
   between v1 and the camera-ready trainer, i.e. the fix changes no published result.

Usage:
    python verify_locality.py                       # cifar-10 trainer, check 1
    python verify_locality.py --module cifar100     # cifar-100 trainer
    python verify_locality.py --module tiny         # tiny-imagenet trainer
    python verify_locality.py --module gated        # hardness-gated trainer
    python verify_locality.py --module text         # shared FF block library (cp_fair_core)
    python verify_locality.py --module gated --depth-order [--v1-dir DIR] [--json OUT]

Exit code 0 means the locality assertion held. Exit code 1 means a leak was
detected (or, with --v1-dir, that parameters differ from v1) — use this as a
pre-flight check before any full training run.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

# Locate the trainers/ directory relative to this script (override via FF_TRAINER_DIR).
HERE = Path(__file__).resolve().parent
DEFAULT_TRAINER_DIR = HERE.parent / "trainers"
TRAINER_DIR = Path(os.environ.get("FF_TRAINER_DIR", str(DEFAULT_TRAINER_DIR))).resolve()
sys.path.insert(0, str(TRAINER_DIR))


MODULE_MAP = {
    "cifar10": "cp_fair_cifar10",
    "cifar100": "cp_fair_cifar100",
    "tiny": "cp_fair_tiny_imagenet",
    "gated": "cp_fair_hardness_gated",
    "text": "cp_fair_core",
}


def _build_tiny_cfg(mod):
    cfg = mod.FFConfig()
    cfg.num_blocks = 3
    cfg.d_model = 32
    cfg.num_heads = 4
    cfg.mlp_mult = 2
    cfg.n_experts = 2
    cfg.moe_top_k = 1
    cfg.num_classes = 4
    cfg.patch_size = 8
    cfg.img_size = 32
    cfg.n_aspects = 2
    cfg.use_sam = False
    cfg.label_embed_dim = 8
    cfg.proj_dim = 16
    cfg.use_mem = False
    cfg.stem_channels = 16
    cfg.use_conv_stem = True
    cfg.use_rope = False
    cfg.__post_init__()
    return cfg


def _mock_forward_with_accumulation(net, mod, cfg, device):
    """Mimic the accumulation path in train_epoch_stage1 for a single synthetic
    batch: collect per-block goodness, then compute a block-`li` loss that
    depends on `gamma = _sum_prev_goodness(net, g_hist[:li])`. Finally backward
    and inspect .grad on blocks[0..li-1] aggregator params."""
    torch.manual_seed(0)
    B = 2
    images = torch.randn(B, 3, cfg.img_size, cfg.img_size, device=device)
    labels = torch.zeros(B, dtype=torch.long, device=device)

    tokens = net.embed(images)

    g_hist_pos: List[torch.Tensor] = []
    g_hist_neg: List[torch.Tensor] = []

    # Walk the first (num_blocks - 1) blocks to populate history (values
    # detached at append-time, exactly as the real trainer does).
    for li, block in enumerate(net.blocks[:-1]):
        a_pos, gpos, _, _, _ = block(tokens, labels, None if li == 0 else torch.stack(g_hist_pos, dim=1).detach())
        a_neg, gneg, _, _, _ = block(tokens, (labels + 1) % cfg.num_classes, None if li == 0 else torch.stack(g_hist_neg, dim=1).detach())
        g_hist_pos.append(gpos.detach())
        g_hist_neg.append(gneg.detach())
        # feed next block with detached activations, as the real loop does
        tokens = F.normalize(a_pos.detach(), dim=-1)

    # Now compute a "block-li" style loss at the final block that depends on
    # cumulative prior-block goodness via _sum_prev_goodness.
    li = cfg.num_blocks - 1
    block = net.blocks[li]
    g_prev = torch.stack(g_hist_pos, dim=1).detach()
    a_pos, gpos, _, _, _ = block(tokens, labels, g_prev)
    _, gneg, _, _, _ = block(tokens, (labels + 1) % cfg.num_classes, g_prev)

    gamma_pos = mod._sum_prev_goodness(net, g_hist_pos)
    gamma_neg = mod._sum_prev_goodness(net, g_hist_neg)

    gpos_curr = block.aggregate_goodness(gpos)
    gneg_curr = block.aggregate_goodness(gneg)

    gamma_scale = 0.7
    gpos_blk = gpos_curr + gamma_scale * gamma_pos
    gneg_blk = gneg_curr + gamma_scale * gamma_neg

    loss = mod.symba_loss(gpos_blk, gneg_blk, alpha=4.0)
    return loss, li


def _zero_all_grads(net):
    for p in net.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _nonzero_grads_on_prior_blocks(net, li_current: int):
    offenders = []
    for j in range(li_current):
        blk = net.blocks[j]
        for name, p in blk.named_parameters():
            if p.grad is not None and torch.any(p.grad.abs() > 0).item():
                offenders.append((j, name, float(p.grad.abs().max().item())))
    return offenders


# ------------------------------------------------------------------ depth-order path (camera-ready)
DO_BATCHES = 3
DO_BATCH = 8


def _depth_order_cfg(mod, use_sam: bool):
    cfg = _build_tiny_cfg(mod)
    cfg.n_aspects = 4
    cfg.head_hidden = 32
    cfg.queue_size = 64
    cfg.queue_neg_samples = 16
    cfg.hard_negative_k_first = 2
    cfg.hard_negative_k_last = 3
    cfg.use_sam = use_sam
    cfg.use_ema = True
    cfg.use_ema_hnm = True
    cfg.device = "cpu"
    cfg.workers = 0
    cfg.__post_init__()
    cfg.depth_order_lambda = 0.2
    cfg.depth_margin_pos = 0.05
    cfg.depth_margin_neg = 0.1
    return cfg


def _depth_order_run(mod, cfg):
    """N real Stage-1 steps; returns (params, ema params, leak events, discarded-grad events)."""
    import random
    from copy import deepcopy
    import numpy as np
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    net = mod.FFHybridNet(cfg)
    net_ema = deepcopy(net)
    for p in net_ema.parameters():
        p.requires_grad_(False)
    opts = []
    for li, block in enumerate(net.blocks):
        params = list(block.parameters()) + (list(net.embed.parameters()) if li == 0 else [])
        opts.append(mod.SAM(params, torch.optim.AdamW, rho=cfg.sam_rho, adaptive=cfg.sam_adaptive,
                            lr=cfg.lr, weight_decay=cfg.weight_decay))
    queues = [mod.ContrastiveQueue(cfg.queue_size, cfg.proj_dim, device="cpu") for _ in range(cfg.num_blocks)]
    state = {"cur": None}
    leaks, discarded = [], []
    for li, opt in enumerate(opts):
        orig = opt.zero_grad

        def zg(set_to_none=True, _li=li, _orig=orig):
            if state["cur"] != _li:
                gr = net.blocks[_li].alpha_raw.grad
                if gr is not None and float(gr.abs().max()) > 0:
                    discarded.append((_li, float(gr.abs().max())))
            state["cur"] = _li
            return _orig(set_to_none=set_to_none)
        opt.zero_grad = zg
    for j, blk in enumerate(net.blocks):
        for name, p in blk.named_parameters():
            def hook(grad, _j=j, _name=name):
                if state["cur"] is not None and state["cur"] != _j and float(grad.abs().max()) > 0:
                    leaks.append((state["cur"], _j, _name, float(grad.abs().max())))
                return grad
            p.register_hook(hook)
    g = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(DO_BATCHES):
        xs = [torch.randn(DO_BATCH, 3, cfg.img_size, cfg.img_size, generator=g) for _ in range(3)]
        batches.append((*xs, torch.randint(0, cfg.num_classes, (DO_BATCH,), generator=g)))
    mod.train_epoch_stage1(net, net_ema, opts, batches, cfg, queues, epoch=1)
    params = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ema = {k: v.detach().clone() for k, v in net_ema.state_dict().items()}
    return params, ema, leaks, discarded


def _load_module_from(dir_path, mod_name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, os.path.join(dir_path, mod_name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


def depth_order_check(module_key, v1_dir=None):
    mod_name = MODULE_MAP[module_key]
    mod = importlib.import_module(mod_name)
    v1 = _load_module_from(v1_dir, mod_name, "v1_" + mod_name) if v1_dir else None
    out = {}
    ok = True
    for use_sam in (False, True):
        cfg = _depth_order_cfg(mod, use_sam)
        p_new, e_new, leaks, disc = _depth_order_run(mod, cfg)
        rec = {"leak_events": len(leaks),
               "leaked_params": sorted({f"blocks[{j}].{n} (during block {c})" for c, j, n, _ in leaks}),
               "discard_events": len(disc)}
        ok &= (len(leaks) == 0)
        if v1 is not None:
            p_v1, e_v1, leaks_v1, disc_v1 = _depth_order_run(v1, _depth_order_cfg(v1, use_sam))
            same = all(torch.equal(p_new[k], p_v1[k]) for k in p_v1) and set(p_new) == set(p_v1)
            same_ema = all(torch.equal(e_new[k], e_v1[k]) for k in e_v1) and set(e_new) == set(e_v1)
            rec.update({
                "v1_leak_events": len(leaks_v1),
                "v1_leaked_params": sorted({f"blocks[{j}].{n} (during block {c})" for c, j, n, _ in leaks_v1}),
                "v1_max_leaked_grad": max((m for *_, m in leaks_v1), default=0.0),
                "v1_discard_events": len(disc_v1),
                "params_bit_identical_to_v1": bool(same),
                "ema_params_bit_identical_to_v1": bool(same_ema),
            })
            ok &= same and same_ema
        out[f"sam={use_sam}"] = rec
        print(f"[verify_locality] depth-order {mod_name} sam={use_sam}: {rec}")
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="cifar10", choices=list(MODULE_MAP.keys()))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth-order", action="store_true",
                    help="run the depth-order-path check on the real train_epoch_stage1 (CPU)")
    ap.add_argument("--v1-dir", default=os.environ.get("FF_V1_TRAINERS"),
                    help="v1 trainers directory; adds the bit-identical-parameters comparison")
    ap.add_argument("--json", default=None, help="append the depth-order result to this JSON file")
    args = ap.parse_args()

    if args.depth_order:
        torch.set_num_threads(int(os.environ.get("FF_TEST_THREADS", "6")))
        ok, res = depth_order_check(args.module, args.v1_dir)
        if args.json:
            import json
            try:
                with open(args.json) as f:
                    allres = json.load(f)
            except (FileNotFoundError, ValueError):
                allres = {}
            allres[MODULE_MAP[args.module]] = {"pass": bool(ok), "v1_compared": bool(args.v1_dir), **res}
            with open(args.json, "w") as f:
                json.dump(allres, f, indent=2)
        print(f"[verify_locality] depth-order {'PASS' if ok else 'FAIL'} ({MODULE_MAP[args.module]})")
        sys.exit(0 if ok else 1)

    sys.path.insert(0, ".")
    mod_name = MODULE_MAP[args.module]
    print(f"[verify_locality] loading module: {mod_name}")
    mod = importlib.import_module(mod_name)

    cfg = _build_tiny_cfg(mod)
    net = mod.FFHybridNet(cfg).to(args.device)
    net.train()

    _zero_all_grads(net)
    loss, li = _mock_forward_with_accumulation(net, mod, cfg, args.device)
    print(f"[verify_locality] block-{li} loss = {loss.item():.6f}")
    loss.backward()

    offenders = _nonzero_grads_on_prior_blocks(net, li)
    if offenders:
        print(f"[verify_locality] FAIL: {len(offenders)} parameters in blocks[0..{li-1}] received non-zero gradient from block-{li} loss:")
        for j, name, gmax in offenders[:10]:
            print(f"  blocks[{j}].{name}: max|grad| = {gmax:.3e}")
        if len(offenders) > 10:
            print(f"  ... and {len(offenders) - 10} more")
        sys.exit(1)

    print(f"[verify_locality] PASS: all parameters in blocks[0..{li-1}] have zero gradient.")
    sys.exit(0)


if __name__ == "__main__":
    main()
