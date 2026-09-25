"""
Label-free probe feature extraction for the CIFAR-100 dissociation trio.

Per checkpoint (9 = {gamma0, gated_k0, cumulative} x seeds {42,123,456}):
  1. VAL split (true train/val split from the run's split_indices.json, 5000 examples,
     eval transform):
       goodness_val [5000, L=4, C=100]  per-block per-class goodness, no TTA
                                        (label-free: computed under every hypothesis)
       feats_lf_val [5000, L, D=256]    label-free pooled features (mean-label-token
                                        forward, see below)
  2. TEST split (official 10k):
       feats_lf_test [10000, L, D]      same label-free forward
       (goodness_test reused from existing dumps/block_dump_c100_*.npz)

Label-free forward ("mean-label-token"): in every block, label_emb.weight rows are
replaced by their class-average (so _append_label injects the average label embedding
regardless of y) AND prototypes rows are replaced by their class-average (so the
prototype-alignment aspect of g_vec, which feeds the g_prev memory of later blocks,
is also label-independent). After the patch a forward with any y is identical; we
assert this on a batch (y=0 vs y=57) before trusting it.

Asserts per checkpoint (hard-fail):
  A1 test S1 no-TTA recomputed from the existing dump goodness == aggregated_results
     per-seed s1_no_tta (tol 3.1e-4)
  A2 recomputed val_idx from random.Random(seed) == split_indices.json val_idx
  A3 val goodness decomposition sums to net.predict_scores on batch 0 (tol 5e-2 abs,
     chunked fp order)
  A4 val TTA top-1 (goodness-sum + flipped predict_scores) == ckpt best_s1 (tol 2e-3,
     validation during training evaluated net_ema with eval_tta=True)
  A5 patched label-free forward invariant to y (max abs diff < 1e-4)
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(FF_WORK)
DUMPS = SCRATCH / "dumps"
OUT = SCRATCH / "dumps_val"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"

AGG = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/NeurIPS_FF_Net_Last_submitted/supplement_code/metric_summaries/aggregated_results.json"

spec = importlib.util.spec_from_file_location("dbd", str(SCRATCH / "dump_block_data.py"))
dbd = importlib.util.module_from_spec(spec)
sys.modules["dbd"] = dbd
spec.loader.exec_module(dbd)

manifest = json.loads((DUMPS / "dump_manifest.json").read_text())["dumps"]
agg = json.loads(Path(AGG).read_text())["c100"]

VARIANTS = ["gamma0", "gated_k0", "cumulative"]
SEEDS = [42, 123, 456]
CLASS_CHUNK = 2


def build_val_loader(val_idx, batch_size=500):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    fam = dbd.FAMILIES["c100"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(fam["mean"], fam["std"])])
    ds = datasets.CIFAR100(fam["data_dir"], train=True, download=False, transform=tf)
    return DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


@torch.no_grad()
def goodness_sweep(net, loader, l2norm, class_chunk=CLASS_CHUNK, check_ps=True):
    """Per-block per-class goodness under every label hypothesis. Returns G [N,L,C], Y [N], ps_diff."""
    L, C = net.num_blocks, net.num_classes
    G_all, Y_all = [], []
    ps_diff = None
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        B = images.size(0)
        tokens0 = net.embed(images)
        G = torch.zeros(B, L, C, device=DEVICE)
        for k0 in range(0, C, class_chunk):
            ks = list(range(k0, min(k0 + class_chunk, C)))
            H = len(ks)
            tokens = tokens0.unsqueeze(0).expand(H, B, tokens0.size(1), tokens0.size(2)).reshape(H * B, tokens0.size(1), tokens0.size(2))
            yk = torch.tensor(ks, dtype=torch.long, device=DEVICE).repeat_interleave(B)
            g_hist = []
            for bi, blk in enumerate(net.blocks):
                g_prev = torch.stack(g_hist, dim=1) if g_hist else None
                a, g_vec = blk(tokens, yk, g_prev)[:2]
                G[:, bi, ks] = blk.aggregate_goodness(g_vec).view(H, B).transpose(0, 1)
                g_hist.append(g_vec)
                tokens = l2norm(a)
        if check_ps and ps_diff is None:
            ps = net.predict_scores(images)
            ps_diff = float((G.sum(dim=1) - ps).abs().max())
            assert ps_diff < 5e-2, f"A3 FAIL: decomposition vs predict_scores diff {ps_diff}"
        G_all.append(G.cpu())
        Y_all.append(labels)
    return torch.cat(G_all).numpy().astype(np.float32), torch.cat(Y_all).numpy().astype(np.int16), ps_diff


@torch.no_grad()
def flip_scores(net, loader):
    """Summed goodness scores of horizontally flipped images via net.predict_scores."""
    S = []
    for images, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        S.append(net.predict_scores(torch.flip(images, dims=[3])).cpu())
    return torch.cat(S).numpy().astype(np.float32)


@torch.no_grad()
def labelfree_feats(net, loader, l2norm, y_const=0):
    """Pooled per-block features under the mean-label-token patch. Returns F [N,L,D]."""
    F_all = []
    for images, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        B = images.size(0)
        y = torch.full((B,), y_const, dtype=torch.long, device=DEVICE)
        tokens = net.embed(images)
        g_hist, feats = [], []
        for bi, blk in enumerate(net.blocks):
            g_prev = torch.stack(g_hist, dim=1) if g_hist else None
            a, g_vec = blk(tokens, y, g_prev)[:2]
            feats.append(blk.get_pooled_rep(a))
            g_hist.append(g_vec)
            tokens = l2norm(a)
        F_all.append(torch.stack(feats, dim=1).cpu())
    return torch.cat(F_all).numpy().astype(np.float32)


def main():
    trainer = dbd.load_trainer("c100")
    l2norm = trainer.l2_normalize_tokens
    results = {}
    for variant in VARIANTS:
        anchors = {p["seed"]: p for p in agg[variant]["per_seed"]}
        for seed in SEEDS:
            name = f"c100_{variant}_seed{seed}"
            out_npz = OUT / f"probe_feats_{name}.npz"
            if out_npz.exists():
                print(f"=== {name} already done, skip ===", flush=True)
                continue
            t0 = time.time()
            print(f"=== {name} ===", flush=True)
            meta = manifest[name]

            # A1: test S1 from existing dump goodness vs aggregated anchor
            d = np.load(meta["npz"])
            G_test, y_test = d["goodness"], d["y_true"]
            s1_test = float((G_test.sum(axis=1).argmax(axis=1) == y_test).mean())
            anchor_s1 = anchors[seed]["s1_no_tta"]
            assert abs(s1_test - anchor_s1) <= 3.1e-4, f"A1 FAIL {name}: {s1_test} vs {anchor_s1}"

            # A2: val split
            run_dir = Path(meta["config"]).parent
            split = json.loads((run_dir / "split_indices.json").read_text())
            val_idx = [int(i) for i in split["val_idx"]]
            rng = random.Random(int(seed))
            perm = list(range(50000)); rng.shuffle(perm)
            expect = sorted(perm[:5000])
            assert expect == val_idx, f"A2 FAIL {name}: recomputed val_idx mismatch"

            # load net_ema
            ckpt = torch.load(meta["ckpt"], map_location="cpu", weights_only=False)
            cfg = dbd.build_cfg(trainer, meta["config"], ckpt=ckpt)
            net = trainer.FFHybridNet(cfg).to(DEVICE)
            missing, unexpected = net.load_state_dict(ckpt["net_ema"], strict=False)
            assert not missing and not unexpected, f"load mismatch {name}"
            net.eval()
            best_s1_val_tta = float(ckpt["best_s1"])

            val_loader = build_val_loader(val_idx)

            # val goodness sweep (no TTA) + A3
            G_val, y_val, ps_diff = goodness_sweep(net, val_loader, l2norm)
            val_s1_no_tta = float((G_val.sum(axis=1).argmax(axis=1) == y_val).mean())

            # A4: TTA val top-1 vs ckpt best_s1
            S_flip = flip_scores(net, val_loader)
            val_s1_tta = float(((G_val.sum(axis=1) + S_flip).argmax(axis=1) == y_val).mean())
            d_tta = val_s1_tta - best_s1_val_tta
            assert abs(d_tta) <= 2e-3, f"A4 FAIL {name}: val TTA {val_s1_tta} vs ckpt best_s1 {best_s1_val_tta}"

            # mean-label-token patch
            with torch.no_grad():
                for blk in net.blocks:
                    blk.label_emb.weight.data[:] = blk.label_emb.weight.data.mean(dim=0, keepdim=True)
                    blk.prototypes.data[:] = blk.prototypes.data.mean(dim=0, keepdim=True)

            # A5: y-invariance of patched forward
            images0, _ = next(iter(val_loader))
            images0 = images0[:64].to(DEVICE)
            with torch.no_grad():
                fa = _feats_one(net, images0, l2norm, 0)
                fb = _feats_one(net, images0, l2norm, 57)
            inv = float(np.abs(fa - fb).max())
            assert inv < 1e-4, f"A5 FAIL {name}: patched forward varies with y ({inv})"

            F_lf_val = labelfree_feats(net, val_loader, l2norm)
            test_loader = dbd.build_loader("c100", 500)
            F_lf_test = labelfree_feats(net, test_loader, l2norm)

            info = {
                "name": name, "variant": variant, "seed": seed,
                "ckpt": meta["ckpt"], "run_dir": str(run_dir),
                "weights_used": "net_ema",
                "n_val": len(val_idx), "n_test": int(len(y_test)),
                "A1_test_s1_no_tta": s1_test, "A1_anchor": anchor_s1,
                "A3_ps_diff_batch0": ps_diff,
                "A4_val_s1_tta_recomputed": val_s1_tta,
                "A4_ckpt_best_s1": best_s1_val_tta, "A4_delta": d_tta,
                "A5_y_invariance_maxabs": inv,
                "val_s1_no_tta": val_s1_no_tta,
                "labelfree_forward": "mean-label-token: label_emb.weight and prototypes rows replaced by class-average in every block; pooled = attn_pool per block; token chain l2-normalized as in training",
                "seconds": round(time.time() - t0, 1),
            }
            np.savez_compressed(out_npz, goodness_val=G_val, feats_lf_val=F_lf_val,
                                feats_lf_test=F_lf_test, y_val=y_val, y_test=y_test,
                                val_idx=np.array(val_idx, dtype=np.int32),
                                meta=json.dumps(info))
            results[name] = info
            print(json.dumps(info), flush=True)
            del net, G_val, F_lf_val, F_lf_test
            torch.cuda.empty_cache()

    man_path = OUT / "probe_feats_manifest.json"
    old = json.loads(man_path.read_text()) if man_path.exists() else {}
    old.update(results)
    man_path.write_text(json.dumps(old, indent=2))
    print("ALL DONE", flush=True)


@torch.no_grad()
def _feats_one(net, images, l2norm, y_const):
    B = images.size(0)
    y = torch.full((B,), y_const, dtype=torch.long, device=DEVICE)
    tokens = net.embed(images)
    g_hist, feats = [], []
    for blk in net.blocks:
        g_prev = torch.stack(g_hist, dim=1) if g_hist else None
        a, g_vec = blk(tokens, y, g_prev)[:2]
        feats.append(blk.get_pooled_rep(a))
        g_hist.append(g_vec)
        tokens = l2norm(a)
    return torch.stack(feats, dim=1).cpu().numpy()


if __name__ == "__main__":
    main()
