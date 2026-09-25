"""
Shared checkpoint dumps for FF/CP-FAIR analyses.

Per checkpoint: one pass over the standard test set (no TTA, no_grad, EMA weights
as used by every trainer's final_test_eval), saving to
  dumps/block_dump_<name>.npz
    goodness   : float32 [N, L, C]  per-block per-class TOTAL goodness g_d(x,c)
                 BEFORE summation across blocks (aggregate_goodness of each block
                 under each label hypothesis, exactly the terms summed by
                 predict_scores)
    feats_true : float32 [N, L, D]  per-block pooled feature (attn_pool) under the
                 TRUE-label hypothesis
    y_true     : int16   [N]
    meta       : json string (ckpt, config, family, recomputed acc, anchor,
                 assert status, agreement with stored preds npz where available)

Loader pattern adapted from revised_training/grad_attenuation/snapshot_grad_attenuation.py:
reconstruct FFConfig from the run's config.json, instantiate the trainer module's
FFHybridNet, load the checkpoint dict. IMPORTANT DIFFERENCE: final_test_eval in all
trainers evaluates the EMA weights (`net_ema` entry), so we load net_ema.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison")
SCRATCH = Path(FF_WORK)
DUMP_DIR = SCRATCH / "dumps"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAMILIES = {
    "c100": {
        "trainer": ROOT / "revised_training/extracted/ff_cifar100_multiseed/basic_fair_revised_valfix_cp_fair_cifar100_fixed.py",
        "dataset": "cifar100",
        "data_dir": FF_HOME + "/amir-porjects/cifar_100_fair/data_c100",
        "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761),
        "l2_tokens": True, "inject_kw": False,
    },
    "gated_c10": {
        "trainer": ROOT / "basic_fair_revised_valfix_cp_fair_hardness_gated/basic_fair_revised_valfix_cp_fair_hardness_gated.py",
        "dataset": "cifar10",
        "data_dir": str(ROOT / "revised_training/data_c10"),
        # NOTE: this trainer lineage uses the 0.2023-convention std (verified at
        # basic_fair_revised_valfix_cp_fair_hardness_gated.py:1425-1426); only v2 uses 0.2470.
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010),
        "l2_tokens": True, "inject_kw": False,
    },
    "v2": {
        "trainer": ROOT / "v2_fair_valsplit_workersfix/v2_fair_valsplit_workersfix.py",
        "dataset": "cifar10",
        "data_dir": str(ROOT / "revised_training/data_c10"),
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616),
        "l2_tokens": False, "inject_kw": True,
    },
    "c10_fixed": {  # revised_training D256 trainer (gamma0 out/ runs)
        "trainer": ROOT / "revised_training/basic_fair_revised_valfix_cp_fair_fixed.py",
        "dataset": "cifar10",
        "data_dir": str(ROOT / "revised_training/data_c10"),
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010),
        "l2_tokens": True, "inject_kw": False,
    },
    "cp_fair": {  # basic_fair_revised_valfix_cp_fair (pre-fix trainer dir)
        "trainer": ROOT / "basic_fair_revised_valfix_cp_fair/basic_fair_revised_valfix_cp_fair.py",
        "dataset": "cifar10",
        "data_dir": str(ROOT / "revised_training/data_c10"),
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010),
        "l2_tokens": True, "inject_kw": False,
    },
}

_MODULE_CACHE = {}


def load_trainer(family):
    path = FAMILIES[family]["trainer"]
    key = str(path)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    modname = f"trainer_{family}"
    # trainer scripts sys.path-import nothing project-local; load standalone
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    _MODULE_CACHE[key] = mod
    return mod


def build_cfg(trainer, config_json, ckpt=None):
    if config_json == "__from_ckpt__":
        # raw run config.json unusable (e.g. Drive 404 HTML); use the cfg object
        # stored inside the checkpoint itself (verified to match run config.json
        # exactly on sibling conditions)
        stored = ckpt["cfg"]
        c = dict(vars(stored)) if not isinstance(stored, dict) else dict(stored)
        c = {k: v for k, v in c.items() if not k.startswith("_")}
    else:
        with open(config_json) as f:
            c = json.load(f)
    cfg = trainer.FFConfig()
    cfg.__post_init__()
    for k, v in c.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.aspect_loss_types = list(cfg.aspect_loss_types[: cfg.n_aspects])
    cfg.aspect_lambdas = list(cfg.aspect_lambdas[: cfg.n_aspects])
    cfg.aspect_theta_inits = list(cfg.aspect_theta_inits[: cfg.n_aspects])
    cfg.device = DEVICE
    return cfg


def build_loader(family, batch_size=500):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    fam = FAMILIES[family]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(fam["mean"], fam["std"]),
    ])
    if fam["dataset"] == "cifar100":
        ds = datasets.CIFAR100(fam["data_dir"], train=False, download=False, transform=tf)
    else:
        ds = datasets.CIFAR10(fam["data_dir"], train=False, download=False, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def call_block(blk, tokens, y, g_prev, inject_kw, inject):
    if inject_kw:
        out = blk(tokens, y, g_prev, inject_label=inject)
    else:
        out = blk(tokens, y, g_prev)
    a, g_vec = out[0], out[1]
    return a, g_vec


@torch.no_grad()
def dump_one(family, ckpt_path, config_json, anchor_s1_no_tta, name,
             preds_npz=None, batch_size=500, tol=2e-3, class_chunk=1):
    t0 = time.time()
    trainer = load_trainer(family)
    fam = FAMILIES[family]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = build_cfg(trainer, config_json, ckpt=ckpt)
    net = trainer.FFHybridNet(cfg).to(DEVICE)
    weights_used = None
    sd = None
    if isinstance(ckpt, dict):
        if ckpt.get("net_ema") is not None:
            sd = ckpt["net_ema"]
            weights_used = "net_ema"
        elif ckpt.get("model") is not None:
            sd = ckpt["model"]
            weights_used = "model (no net_ema in ckpt!)"
    if sd is None:
        raise RuntimeError(f"unrecognized ckpt structure: {list(ckpt.keys())[:10]}")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.eval()

    lifb = bool(getattr(cfg, "label_inject_only_first_block", False))
    l2norm = trainer.l2_normalize_tokens if fam["l2_tokens"] else None

    loader = build_loader(family, batch_size)
    L = net.num_blocks
    C = net.num_classes

    G_all, F_all, Y_all = [], [], []
    checked_predict_scores = False
    ps_max_diff = None
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        y_true = labels.to(DEVICE, non_blocking=True)
        B = images.size(0)
        tokens0 = net.embed(images)
        G = torch.zeros(B, L, C, device=DEVICE)
        for k0 in range(0, C, class_chunk):
            ks = list(range(k0, min(k0 + class_chunk, C)))
            H = len(ks)
            # replicate the batch for H label hypotheses; every op in the block
            # (RMSNorm/LayerNorm, per-sample attention, per-token MoE routing,
            # attn_pool) is per-sample, so this is mathematically identical to
            # H separate forwards.
            tokens = tokens0.unsqueeze(0).expand(H, B, tokens0.size(1), tokens0.size(2)).reshape(H * B, tokens0.size(1), tokens0.size(2))
            yk = torch.tensor(ks, dtype=torch.long, device=DEVICE).repeat_interleave(B)
            g_hist = []
            for bi, blk in enumerate(net.blocks):
                g_prev = torch.stack(g_hist, dim=1) if g_hist else None
                inject = (not lifb) or (bi == 0)
                a, g_vec = call_block(blk, tokens, yk, g_prev, fam["inject_kw"], inject)
                G[:, bi, ks] = blk.aggregate_goodness(g_vec).view(H, B).transpose(0, 1)
                g_hist.append(g_vec)
                tokens = l2norm(a) if l2norm is not None else a
        # true-label pass for pooled features
        tokens = tokens0
        g_hist = []
        feats = []
        for bi, blk in enumerate(net.blocks):
            g_prev = torch.stack(g_hist, dim=1) if g_hist else None
            inject = (not lifb) or (bi == 0)
            a, g_vec = call_block(blk, tokens, y_true, g_prev, fam["inject_kw"], inject)
            feats.append(blk.get_pooled_rep(a))
            g_hist.append(g_vec)
            tokens = l2norm(a) if l2norm is not None else a
        feats = torch.stack(feats, dim=1)  # [B, L, D]

        if not checked_predict_scores:
            # sanity: our per-block decomposition must sum to the model's own scores
            ps = net.predict_scores(images)
            ps_max_diff = float((G.sum(dim=1) - ps).abs().max())
            # chunked label-hypothesis batching changes fp reduction order; goodness
            # magnitudes can be O(1e2-1e4), so allow small absolute drift and hard-fail
            # only on semantic-level mismatch
            assert ps_max_diff < 5e-2, f"decomposition mismatch vs predict_scores: {ps_max_diff}"
            checked_predict_scores = True

        G_all.append(G.cpu())
        F_all.append(feats.cpu())
        Y_all.append(y_true.cpu())

    G = torch.cat(G_all).numpy().astype(np.float32)      # [N, L, C]
    F = torch.cat(F_all).numpy().astype(np.float32)      # [N, L, D]
    Y = torch.cat(Y_all).numpy().astype(np.int16)        # [N]

    scores = G.sum(axis=1)                               # [N, C]
    y_pred = scores.argmax(axis=1)
    acc = float((y_pred == Y).mean())
    delta = None if anchor_s1_no_tta is None else acc - float(anchor_s1_no_tta)
    ok = None if anchor_s1_no_tta is None else abs(delta) <= tol

    preds_agree = None
    if preds_npz and Path(preds_npz).exists():
        d = np.load(preds_npz)
        preds_agree = float((y_pred == d["y_pred"]).mean())

    meta = {
        "name": name,
        "family": family,
        "ckpt": str(ckpt_path),
        "config": str(config_json),
        "weights_used": weights_used,
        "load_missing": len(missing), "load_unexpected": len(unexpected),
        "N": int(G.shape[0]), "L": int(L), "C": int(C), "D": int(F.shape[2]),
        "recomputed_s1_top1_no_tta": acc,
        "anchor_s1_top1_no_tta": anchor_s1_no_tta,
        "delta": delta,
        "assert_ok": ok,
        "predict_scores_max_abs_diff_batch0": ps_max_diff,
        "stored_preds_agreement": preds_agree,
        "preds_npz": str(preds_npz) if preds_npz else None,
        "eval_protocol": "S1 no-TTA, EMA weights, official test split, eval transform (ToTensor+Normalize)",
        "seconds": round(time.time() - t0, 1),
    }
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    out = DUMP_DIR / f"block_dump_{name}.npz"
    np.savez_compressed(out, goodness=G, feats_true=F, y_true=Y, meta=json.dumps(meta))
    del G_all, F_all, G, F
    torch.cuda.empty_cache()
    print(json.dumps(meta))
    # append to manifest
    man_path = DUMP_DIR / "dump_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    meta["npz"] = str(out)
    man[name] = meta
    man_path.write_text(json.dumps(man, indent=2))
    if ok is False:
        print(f"WARNING: ASSERT FAILED for {name}: recomputed {acc:.4f} vs anchor {anchor_s1_no_tta:.4f}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="path to a JSON list of job dicts")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--class-chunk", type=int, default=1)
    args = ap.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    for j in jobs:
        print(f"=== {j['name']} ===", flush=True)
        try:
            dump_one(j["family"], j["ckpt"], j["config"], j.get("anchor"),
                     j["name"], preds_npz=j.get("preds"), batch_size=args.batch_size,
                     class_chunk=args.class_chunk)
        except Exception as e:
            print(f"ERROR {j['name']}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
