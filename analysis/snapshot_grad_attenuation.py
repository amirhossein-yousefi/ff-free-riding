"""
Compute per-block analytical FF cumulative-loss gradient signal from existing
trained checkpoints (no retraining).

For each variant (gamma in {0, 0.7, 1.0}) we:
 1. Load the stage-1 checkpoint of FFHybridNet.
 2. Run forward on a batch of CIFAR-10 test images with two label
    hypotheses (positive = true label, negative = uniformly random
    wrong label).
 3. Read off per-block, per-example current-block goodness margin
        m^(d) = aggregate(g_pos_cur^(d)) - aggregate(g_neg_cur^(d)).
 4. Compute the cumulative margin M^(d) = m^(d) + gamma * Sum_{j<d} m^(j).
 5. The analytical gradient supplied to block d's m^(d) by the cumulative
    softplus loss l_alpha(M^(d)) is
        d l_alpha / d m^(d) = -alpha * sigmoid(-alpha * M^(d))
    so |grad| = alpha * sigmoid(-alpha * M^(d)).
 6. Save per-example arrays (npz) and a JSON summary; the companion
    plot_grad_attenuation.py renders the figure.

Configure paths via environment variables:
  FF_TRAINER_DIR : directory containing the hardness-gated trainer
                   (default: ../trainers relative to this file).
  FF_CHECKPOINTS_DIR : directory containing the three stage-1 best
                       checkpoints listed below (must contain
                       ff_cifar10_no_all_collab_*, ff_cifar10_baseline_constant_*,
                       ff_cifar10_lcff_prefix_* .pt files).
  FF_RUNS_DIR : directory containing the matching config.json files
                (one run subdirectory per variant; the script picks the
                first matching subdir per name pattern).
  FF_DATA_DIR : CIFAR-10 data dir for torchvision (will download if
                missing). Default ./data_c10.

Run: python snapshot_grad_attenuation.py
"""
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

# ---- locate trainer module ----
HERE = Path(__file__).resolve().parent
DEFAULT_TRAINER_DIR = HERE.parent / "trainers"
TRAINER_DIR = Path(os.environ.get("FF_TRAINER_DIR", str(DEFAULT_TRAINER_DIR))).resolve()
sys.path.insert(0, str(TRAINER_DIR))

# Trainer module is named cp_fair_hardness_gated.py in the supplement.
import cp_fair_hardness_gated as trainer  # noqa: E402

FFConfig = trainer.FFConfig
FFHybridNet = trainer.FFHybridNet
l2_normalize_tokens = trainer.l2_normalize_tokens

# ---- locate checkpoints + runs ----
CKPT_DIR = Path(os.environ.get("FF_CHECKPOINTS_DIR", "checkpoints")).resolve()
RUNS_DIR = Path(os.environ.get("FF_RUNS_DIR", "runs")).resolve()


def _find_first(dirpath: Path, pattern: str) -> Path:
    matches = sorted(dirpath.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find any file matching {pattern!r} under {dirpath}"
        )
    return matches[0]


def _find_first_dir(dirpath: Path, pattern: str) -> Path:
    matches = sorted(p for p in dirpath.glob(pattern) if p.is_dir())
    if not matches:
        raise FileNotFoundError(
            f"Could not find any directory matching {pattern!r} under {dirpath}"
        )
    return matches[0]


VARIANTS = [
    {
        "tag": "gamma=0 (purely local)",
        "gamma": 0.0,
        "ckpt_pattern": "ff_cifar10_no_all_collab_*_stage1_best.pt",
        "run_pattern": "ff_cifar10_no_all_collab_*",
    },
    {
        "tag": "gamma=0.7 (CP-FAIR-style)",
        "gamma": 0.7,
        "ckpt_pattern": "ff_cifar10_baseline_constant_*_stage1_best.pt",
        "run_pattern": "ff_cifar10_baseline_constant_*",
    },
    {
        "tag": "gamma=1.0 (LCFF)",
        "gamma": 1.0,
        "ckpt_pattern": "ff_cifar10_lcff_prefix_*_stage1_best.pt",
        "run_pattern": "ff_cifar10_lcff_prefix_*",
    },
]

# Resolve concrete paths
for v in VARIANTS:
    v["ckpt"] = _find_first(CKPT_DIR, v["ckpt_pattern"])
    v["config"] = _find_first_dir(RUNS_DIR, v["run_pattern"]) / "config.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BATCHES = 4   # 4 x 256 = 1024 examples per variant
BATCH = 256
DATA_DIR = str(Path(os.environ.get("FF_DATA_DIR", "./data_c10")).resolve())


def build_cfg_from_json(path: Path) -> "FFConfig":
    with open(path) as f:
        c = json.load(f)
    cfg = FFConfig()
    # post_init populates aspect_*; we apply config.json on top
    cfg.__post_init__()
    for k, v in c.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # safety: some checkpoints saved aspect_* under their own keys; ensure lengths
    cfg.aspect_loss_types = list(cfg.aspect_loss_types[: cfg.n_aspects])
    cfg.aspect_lambdas = list(cfg.aspect_lambdas[: cfg.n_aspects])
    cfg.aspect_theta_inits = list(cfg.aspect_theta_inits[: cfg.n_aspects])
    return cfg


def build_eval_loader(cfg, batch_size: int):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


@torch.no_grad()
def per_block_margins(net: "FFHybridNet", images: torch.Tensor, y_pos: torch.Tensor, y_neg: torch.Tensor):
    """Run TWO forward passes (positive label, negative label) and return
    per-block scalar goodness for each. Uses the same aggregation as in the
    trainer's training loop.

    Returns:
      gpos_curr : [L, B]  current-block goodness with y=true label
      gnl_curr  : [L, B]  current-block goodness with y=random wrong label
    """
    net.eval()
    L = net.num_blocks

    # -- positive forward (true label hypothesis) --
    tokens0 = net.embed(images)
    tokens = tokens0
    g_hist_pos = []
    gpos_curr_list = []
    for blk in net.blocks:
        g_prev = torch.stack(g_hist_pos, dim=1) if g_hist_pos else None
        if g_prev is not None:
            g_prev = g_prev.detach()
        a, g_vec, _, _, _ = blk(tokens, y_pos, g_prev)
        gpos_curr_list.append(blk.aggregate_goodness(g_vec))
        g_hist_pos.append(g_vec)
        tokens = l2_normalize_tokens(a)
    gpos_curr = torch.stack(gpos_curr_list, dim=0)  # [L, B]

    # -- negative forward (random wrong label hypothesis) --
    tokens = tokens0
    g_hist_nl = []
    gnl_curr_list = []
    for blk in net.blocks:
        g_prev = torch.stack(g_hist_nl, dim=1) if g_hist_nl else None
        if g_prev is not None:
            g_prev = g_prev.detach()
        a, g_vec, _, _, _ = blk(tokens, y_neg, g_prev)
        gnl_curr_list.append(blk.aggregate_goodness(g_vec))
        g_hist_nl.append(g_vec)
        tokens = l2_normalize_tokens(a)
    gnl_curr = torch.stack(gnl_curr_list, dim=0)  # [L, B]

    return gpos_curr, gnl_curr


def random_wrong(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    r = torch.randint(0, num_classes, y.shape, device=y.device)
    return (r + (r == y).long()) % num_classes


def run_variant(variant):
    print(f"\n=== {variant['tag']} (gamma={variant['gamma']}) ===")
    cfg = build_cfg_from_json(variant["config"])
    cfg.device = DEVICE
    print(f"  L={cfg.num_blocks} D={cfg.d_model} alpha={cfg.symba_alpha} gamma_scale={cfg.gamma_scale}")

    net = FFHybridNet(cfg).to(DEVICE)
    sd = torch.load(variant["ckpt"], map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"  loaded ckpt: {len(sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")

    loader = build_eval_loader(cfg, BATCH)

    L = cfg.num_blocks
    alpha = float(cfg.symba_alpha)
    gamma = float(variant["gamma"])

    all_m = []  # list of [L, B] tensors over batches
    torch.manual_seed(42)
    for bi, (images, y) in enumerate(loader):
        if bi >= N_BATCHES:
            break
        images = images.to(DEVICE, non_blocking=True)
        y_pos = y.to(DEVICE, non_blocking=True)
        y_neg = random_wrong(y_pos, cfg.num_classes)

        gpos_curr, gnl_curr = per_block_margins(net, images, y_pos, y_neg)
        m = gpos_curr - gnl_curr  # [L, B]
        all_m.append(m.detach().cpu())

    m_LB = torch.cat(all_m, dim=1)  # [L, N]
    L_, N = m_LB.shape
    assert L_ == L

    # cumulative previous margin: P^(d-1) = sum_{j<d} m^(j)
    P_prev = torch.zeros_like(m_LB)
    for d in range(1, L):
        P_prev[d] = m_LB[:d].sum(dim=0)

    # cumulative loss-input margin during training: M^(d) = m^(d) + gamma * P^(d-1)
    M = m_LB + gamma * P_prev  # [L, N]
    # analytical applied gradient on m^(d) from the cumulative loss:
    #   |grad| = alpha * sigmoid(-alpha * M^(d))
    grad = alpha * torch.sigmoid(-alpha * M)  # [L, N]

    # also compute the "purely-local" reference grad (gamma=0) for sanity
    grad_local = alpha * torch.sigmoid(-alpha * m_LB)

    # per-block summary
    summary = []
    for d in range(L):
        rec = {
            "block": d,
            "m_mean": float(m_LB[d].mean()),
            "m_median": float(m_LB[d].median()),
            "P_prev_mean": float(P_prev[d].mean()),
            "P_prev_median": float(P_prev[d].median()),
            "M_mean": float(M[d].mean()),
            "grad_mean": float(grad[d].mean()),
            "grad_median": float(grad[d].median()),
            "grad_p10": float(torch.quantile(grad[d], 0.10)),
            "grad_p90": float(torch.quantile(grad[d], 0.90)),
            "grad_local_mean": float(grad_local[d].mean()),
        }
        summary.append(rec)
    return {
        "tag": variant["tag"],
        "gamma": gamma,
        "alpha": alpha,
        "L": L,
        "N": N,
        "summary": summary,
        "m": m_LB.numpy(),       # [L, N]
        "P_prev": P_prev.numpy(), # [L, N]
        "M": M.numpy(),
        "grad": grad.numpy(),
    }


def main():
    out_dir = Path(__file__).resolve().parent
    results = [run_variant(v) for v in VARIANTS]

    # summary JSON (numbers only, no per-example arrays)
    light = []
    for r in results:
        light.append({"tag": r["tag"], "gamma": r["gamma"], "alpha": r["alpha"], "summary": r["summary"]})
    with open(out_dir / "grad_attenuation_data.json", "w") as f:
        json.dump(light, f, indent=2)
    print("\nSaved summary →", out_dir / "grad_attenuation_data.json")

    # full dump (per-example) for plotting
    np.savez_compressed(
        out_dir / "grad_attenuation_arrays.npz",
        **{
            f"{i}_{k}": r[k]
            for i, r in enumerate(results)
            for k in ("m", "P_prev", "M", "grad")
        },
        tags=np.array([r["tag"] for r in results]),
        gammas=np.array([r["gamma"] for r in results]),
    )
    print("Saved arrays →", out_dir / "grad_attenuation_arrays.npz")

    # quick textual summary
    for r in results:
        print(f"\n{r['tag']}:")
        for s in r["summary"]:
            print(f"  L{s['block']}  m̄={s['m_mean']:+.3f}  P_prev̄={s['P_prev_mean']:+.3f}  "
                  f"M̄={s['M_mean']:+.3f}  |grad|̄={s['grad_mean']:.4f}  "
                  f"|grad_local|̄={s['grad_local_mean']:.4f}")


if __name__ == "__main__":
    main()
