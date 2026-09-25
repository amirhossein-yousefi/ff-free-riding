#!/usr/bin/env python3
"""LABEL-FREE pooled feature extraction for the CIFAR-100 trio (9 checkpoints).

Motivation: the prior simple_probes_c100.json feats_true panel was contaminated —
features were per-block attn-pooled activations under the TRUE-label hypothesis
(the FF net embeds the hypothesized label into its input via _append_label, so
probes decode the injected label). This script extracts genuinely label-free
pooled features: the per-block label token is replaced by a fixed, input- and
label-independent vector.

Two constructions (both [N, L=4, D=256], net_ema weights, eval transform):
  meanemb: label slot = mean over the block's 100 label embeddings
           (blk.label_emb.weight.mean(0)) — the uniform-prior label token.
  zerotok: label slot = zero vector.

Why this is exact: all 9 run configs have use_mem=False, so _cross_attend_mem is
an identity (returns h unchanged, needs no goodness history), and the label enters
the forward ONLY through _append_label. The custom forward below therefore mirrors
FFHybridBlock.forward step by step (in_proj∘ln_in ∘ cat-label-token, _self_attend,
_cross_attend_mem(h, None), ln_moe→moe→+dp3, a=relu(h)); the pooled feature is
blk.get_pooled_rep(a) = attn_pool(a), exactly the quantity the contaminated panel
pooled (extract_hybrid_features_all_labels' pooled_a), and tokens propagate as
l2_normalize_tokens(a) exactly as in predict_scores.

Equivalence gate per checkpoint (batch 0 of test, k=0): the same custom forward,
with the label slot set to blk.label_emb.weight[k], must reproduce the module's
own blk.forward(tokens, y=k, g_prev) pooled reps to 0 (assert < 1e-5).

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
import torch.nn.functional as F

SCRATCH = FF_WORK
TRAIN_DIR = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/revised_training/extracted/ff_cifar100_multiseed"
DATA_ROOT = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/mgc_rebuttal/data_c100"
CKPT_ROOT = os.path.join(SCRATCH, "c100_ckpts", "ff_cifar100_multiseed")
OUT_DIR = os.path.join(SCRATCH, "dumps_pooled_labelfree")
MANIFEST = os.path.join(OUT_DIR, "pooled_labelfree_manifest.json")

sys.path.insert(0, TRAIN_DIR)
import basic_fair_revised_valfix_cp_fair_cifar100_fixed as M  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

RUNS = {  # name -> (variant, seed, run_dir relpath) — identical to dump_val_test_block_data.py
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
def custom_pooled(net, images, label_slot):
    """Label-free (or fixed-class, for the gate) pooled reps [B, L, D].

    label_slot: "meanemb" | "zerotok" | int class id (gate mode).
    Mirrors FFHybridBlock.forward exactly for use_mem=False nets; the pooled
    feature is blk.get_pooled_rep(relu(h)) as in extract_hybrid_features_all_labels.
    """
    tokens = net.embed(images)
    B, N, _ = tokens.shape
    pooled_all = []
    for blk in net.blocks:
        assert not blk.use_mem
        if label_slot == "meanemb":
            yv = blk.label_emb.weight.mean(0)
        elif label_slot == "zerotok":
            yv = torch.zeros_like(blk.label_emb.weight[0])
        else:
            yv = blk.label_emb.weight[int(label_slot)]
        ytok = yv.view(1, 1, -1).expand(B, N, yv.numel())
        x = torch.cat([tokens, ytok], dim=-1)          # == _append_label with yv in the slot
        h = blk.in_proj(blk.ln_in(x))
        h = blk._self_attend(h)
        h, _ = blk._cross_attend_mem(h, None)          # use_mem=False -> identity
        m = blk.ln_moe(h)
        moe_out, _, _ = blk.moe(m)
        h = h + blk.dp3(moe_out)
        a = F.relu(h)
        pooled_all.append(blk.get_pooled_rep(a))
        tokens = M.l2_normalize_tokens(a)
    return torch.stack(pooled_all, dim=1)              # [B, L, D]


@torch.no_grad()
def reference_pooled_classk(net, images, k):
    """Ground-truth path through the module's own blk.forward under hypothesis k."""
    tokens = net.embed(images)
    B = images.size(0)
    yk = torch.full((B,), k, dtype=torch.long, device=images.device)
    pooled_all, g_hist = [], []
    for blk in net.blocks:
        g_prev = torch.stack(g_hist, dim=1) if g_hist else None
        a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
        g_hist.append(g_vec)
        pooled_all.append(blk.get_pooled_rep(a))
        tokens = M.l2_normalize_tokens(a)
    return torch.stack(pooled_all, dim=1)


@torch.no_grad()
def run_split(net, loader, device, modes):
    outs = {m: [] for m in modes}
    ys = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        for m in modes:
            outs[m].append(custom_pooled(net, images, m).cpu().numpy().astype(np.float32))
        ys.append(labels.numpy())
    return {m: np.concatenate(v) for m, v in outs.items()}, np.concatenate(ys).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=sorted(RUNS.keys()))
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("highest")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR100(DATA_ROOT, train=True, download=False, transform=eval_tf)
    test_ds = datasets.CIFAR100(DATA_ROOT, train=False, download=False, transform=eval_tf)

    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"_header": {
        "description": "Label-free per-block attn-pooled features [N,L=4,D=256] for VAL(5k own split) and TEST(10k), net_ema of stage1_best.pt, CIFAR-100 trio. Label slot replaced by meanemb (mean of the block's label embeddings) or zerotok (zeros); exact because use_mem=False in all runs (label enters only via _append_label).",
        "created": time.strftime("%Y-%m-%d"),
        "script": os.path.abspath(__file__),
        "matmul_precision": "highest",
    }, "dumps": {}}

    modes = ["meanemb", "zerotok"]
    for name in args.names:
        variant, seed, run_rel = RUNS[name]
        run_dir = os.path.join(TRAIN_DIR, run_rel)
        t0 = time.time()
        split = json.load(open(os.path.join(run_dir, "split_indices.json")))
        val_idx = split["val_idx"]
        assert len(val_idx) == 5000 and split["n_total"] == 50000

        ckpt_dir = os.path.join(CKPT_ROOT, f"checkpoints_{variant}_seed{seed}")
        ckpt_path = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith("stage1_best.pt")]
        assert len(ckpt_path) == 1, ckpt_path
        ckpt_path = ckpt_path[0]

        net, cfg = build_net(os.path.join(run_dir, "config.json"), device)
        assert cfg.use_mem is False, "label-free forward requires use_mem=False"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt.get("net_ema") is not None
        net.load_state_dict(ckpt["net_ema"], strict=True)
        ckpt_best_s1 = float(ckpt.get("best_s1", float("nan")))
        del ckpt

        val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=args.batch, shuffle=False,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=2, pin_memory=True)

        # Equivalence gate: custom forward with label slot = emb[k] must equal blk.forward path.
        images0, _ = next(iter(test_loader))
        images0 = images0.to(device)
        gate_diff = float((custom_pooled(net, images0, 0) - reference_pooled_classk(net, images0, 0)).abs().max().item())
        assert gate_diff < 1e-5, f"equivalence gate failed: {gate_diff}"

        val_feats, y_val = run_split(net, val_loader, device, modes)
        test_feats, y_test = run_split(net, test_loader, device, modes)

        # Consistency vs the goodness dump of the same checkpoint (labels + split identity).
        gnpz = np.load(os.path.join(SCRATCH, "dumps_val", f"block_dump_valtest_{name}.npz"))
        assert np.array_equal(gnpz["y_val"], y_val) and np.array_equal(gnpz["y_test"], y_test)
        assert np.array_equal(gnpz["val_idx"], np.asarray(val_idx, dtype=np.int32))

        npz_path = os.path.join(OUT_DIR, f"pooled_labelfree_{name}.npz")
        np.savez_compressed(npz_path,
                            pooled_val_meanemb=val_feats["meanemb"], pooled_test_meanemb=test_feats["meanemb"],
                            pooled_val_zerotok=val_feats["zerotok"], pooled_test_zerotok=test_feats["zerotok"],
                            y_val=y_val, y_test=y_test, val_idx=np.asarray(val_idx, dtype=np.int32))
        entry = dict(name=name, variant=variant, seed=seed, ckpt=ckpt_path, weights_used="net_ema",
                     ckpt_best_s1=ckpt_best_s1, equivalence_gate_maxabsdiff=gate_diff,
                     shapes={f"pooled_val_{m}": list(val_feats[m].shape) for m in modes} |
                            {f"pooled_test_{m}": list(test_feats[m].shape) for m in modes},
                     labels_and_split_match_goodness_dump=True,
                     npz=npz_path, seconds=round(time.time() - t0, 1))
        manifest["dumps"][name] = entry
        json.dump(manifest, open(MANIFEST, "w"), indent=1)
        print(f"[OK] {name}: gate_diff={gate_diff:.2e} val{val_feats['meanemb'].shape} "
              f"test{test_feats['meanemb'].shape} {entry['seconds']}s", flush=True)

        del net
        torch.cuda.empty_cache()

    print(f"DONE: {len(manifest['dumps'])} pooled label-free dumps")


if __name__ == "__main__":
    main()
