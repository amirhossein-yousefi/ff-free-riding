import math
import random
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from torchvision import datasets, transforms
except ImportError:
    print("Error: torchvision not found. Please install it (pip install torchvision).")
    exit()
import numpy as np
from tqdm.auto import tqdm  # <-- TQDM

# -------------------- token normalization helper --------------------
def l2_normalize_tokens(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize token vectors along the feature dim (last dim)."""
    return t / (t.norm(p=2, dim=-1, keepdim=True) + eps)



# ==================== FAIR-COMPARE UTILITIES & LOGGING ====================
import json
import time
import platform
from datetime import datetime

# NOTE: _workersfix tag to avoid confusing runs with the earlier valsplit version
# that could hang with num_workers>0 on some systems.
VERSION_TAG = os.environ.get("FF_VERSION_TAG", "cifar10_cp_fair_clean_progressive")

# Fixed per-layer key schema so all variants can be compared fairly.
FAIR_LAYER_KEYS = [
    "loss_total",
    "loss_aspects",
    "loss_block",
    "loss_block_curr",
    "loss_inc",
    "loss_depth",
    "loss_con",
    "loss_recon",
    "loss_div",
    "loss_bal",
    "loss_z",
    "sep_nl",
    "sep_ni",
    "sep_curr_nl",
    "sep_curr_ni",
    "gpos",
    "gnl",
    "gni",
    "gpos_cur",
    "gnl_cur",
    "gni_cur",
    "gpos_base",
    "gnl_base",
    "gni_base",
    "gamma_pos",
    "gamma_nl",
    "gamma_ni",
    "k_neg",
]


def _as_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        if isinstance(x, (float, int)):
            return float(x)
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float(default)


def _to_jsonable(x):
    # Make tensors/arrays JSON-serializable (for raw logs).
    try:
        if x is None:
            return None
        if isinstance(x, (bool, int, float, str)):
            return x
        if hasattr(x, "item"):
            return float(x.item())
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        return str(x)
    except Exception:
        return str(x)


def pad_layer_metrics(d: dict) -> dict:
    """Ensure each layer metrics dict has the same keys for fair comparison."""
    out = {}
    for k in FAIR_LAYER_KEYS:
        out[k] = _as_float(d.get(k, 0.0), 0.0)
    return out


def collect_env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device0": torch.cuda.get_device_name(
            0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,
    }
    return info


class JSONLLogger:
    """
    Standardized logging:
      - epoch_metrics.jsonl: per-epoch scalars
      - layer_metrics.jsonl: per-epoch per-layer scalars with fixed FAIR_LAYER_KEYS
      - layer_metrics_raw.jsonl: per-epoch per-layer full dict (version-specific extras)
      - events.jsonl: lifecycle events
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self._f_epoch = open(os.path.join(self.run_dir, "epoch_metrics.jsonl"), "a", encoding="utf-8")
        self._f_layer = open(os.path.join(self.run_dir, "layer_metrics.jsonl"), "a", encoding="utf-8")
        self._f_layer_raw = open(os.path.join(self.run_dir, "layer_metrics_raw.jsonl"), "a", encoding="utf-8")
        self._f_event = open(os.path.join(self.run_dir, "events.jsonl"), "a", encoding="utf-8")

    def log_config(self, cfg_dict: dict, env: dict):
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2, sort_keys=True)
        with open(os.path.join(self.run_dir, "env.json"), "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2, sort_keys=True)
        with open(os.path.join(self.run_dir, "layer_metric_keys.json"), "w", encoding="utf-8") as f:
            json.dump(FAIR_LAYER_KEYS, f, indent=2)

    def log_epoch(self, stage: int, epoch: int, metrics: dict):
        rec = {"ts": time.time(), "stage": int(stage), "epoch": int(epoch)}
        rec.update({k: _as_float(v, None) if v is not None else None for k, v in metrics.items()})
        self._f_epoch.write(json.dumps(rec) + "\n")
        self._f_epoch.flush()

    def log_layer(self, stage: int, epoch: int, layer_idx: int, metrics: dict):
        rec = {"ts": time.time(), "stage": int(stage), "epoch": int(epoch), "layer": int(layer_idx)}
        rec.update(pad_layer_metrics(metrics))
        self._f_layer.write(json.dumps(rec) + "\n")
        self._f_layer.flush()

    def log_layer_raw(self, stage: int, epoch: int, layer_idx: int, metrics: dict):
        rec = {"ts": time.time(), "stage": int(stage), "epoch": int(epoch), "layer": int(layer_idx)}
        if metrics:
            rec.update({str(k): _to_jsonable(v) for k, v in metrics.items()})
        self._f_layer_raw.write(json.dumps(rec) + "\n")
        self._f_layer_raw.flush()

    def log_event(self, name: str, payload=None):
        rec = {"ts": time.time(), "event": str(name)}
        if payload:
            rec.update(payload)
        self._f_event.write(json.dumps(rec) + "\n")
        self._f_event.flush()

    def close(self):
        for f in (self._f_epoch, self._f_layer, self._f_layer_raw, self._f_event):
            try:
                f.close()
            except Exception:
                pass


def make_run_dir(cfg) -> str:
    runs_root = os.environ.get("FF_RUNS_DIR", "./runs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ff_cifar{getattr(cfg, 'num_classes', 10)}_{VERSION_TAG}_L{cfg.num_blocks}_D{cfg.d_model}_bs{cfg.batch_size}_seed{cfg.seed}_{stamp}"
    return os.path.join(runs_root, run_name)


# ==================== CHECKPOINT CONFIG ====================

# You can override this with:
#   export FF_CKPT_DIR=/path/to/checkpoints
# (Falls back to CKPT_DIR if you already use that env var.)
CKPT_DIR = os.environ.get("FF_CKPT_DIR", os.environ.get("CKPT_DIR", "./checkpoints"))
os.makedirs(CKPT_DIR, exist_ok=True)

# Stage-1: keep TWO files (last + best)
# Stage-2: keep TWO files (last + best) - DIFFERENT NAMES so it won't overwrite stage-1
# NOTE: These filenames are placeholders and will be overwritten inside `main()`
# based on the run configuration so multiple experiments won't collide.
STAGE1_LAST_PATH = os.path.join(CKPT_DIR, "ff_cifar10_stage1_last.pt")
STAGE1_BEST_PATH = os.path.join(CKPT_DIR, "ff_cifar10_stage1_best.pt")
STAGE2_LAST_PATH = os.path.join(CKPT_DIR, "ff_cifar10_stage2_last.pt")
STAGE2_BEST_PATH = os.path.join(CKPT_DIR, "ff_cifar10_stage2_best.pt")

# Behavior toggles
PREFER_STAGE2_RESUME = True  # if Stage-2 ckpt exists, resume it and skip Stage-1 automatically
RESUME_STAGE1_FROM_BEST = False  # Stage-1 training loads stage1_best if it exists, else stage1_last
USE_BEST_STAGE1_FOR_STAGE2 = True  # Stage-2 starts from the best Stage-1 backbone (unless resuming Stage-2)


# ==================== [CR] RNG STATE SAVE / RESTORE ====================
# Every checkpoint stores "rng_state"; resuming restores it. Only plain python containers
# and CPU ByteTensors are stored, so torch.load(weights_only=True) (the default since
# torch 2.6) accepts the checkpoint.
# Scope: the main process. The Stage-1 loader uses persistent DataLoader workers; after a
# resume they are re-created and re-seeded from the restored main-process torch RNG. A
# resumed run is therefore reproducible from its checkpoint, but it is not bitwise
# identical to an uninterrupted run (the workers' augmentation streams differ).
# Checkpoints written by the v1 trainers (no "rng_state") still load; the RNG is then left
# as seeded, which is the v1 behavior.
RNG_STATE_KEY = "rng_state"
LAST_RNG_RESTORE: Dict[str, object] = {}


def _capture_rng_state() -> dict:
    """Snapshot of every RNG the training loop draws from. Draws no random numbers."""
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(np_state[0]),
            "keys": [int(v) for v in np_state[1]],
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [t.cpu() for t in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Optional[dict], where: str) -> bool:
    """Restore a snapshot written by _capture_rng_state. Returns True if restored."""
    global LAST_RNG_RESTORE
    if not state:
        print(f"[Checkpoint] {where}: checkpoint has no RNG state (legacy); RNG left as seeded.")
        LAST_RNG_RESTORE = {"where": where, "restored": False, "reason": "absent"}
        return False
    py = state["python"]
    random.setstate((int(py[0]), tuple(int(v) for v in py[1]), py[2]))
    ns = state["numpy"]
    np.random.set_state((str(ns["bit_generator"]), np.asarray(ns["keys"], dtype=np.uint32),
                         int(ns["pos"]), int(ns["has_gauss"]), float(ns["cached_gaussian"])))
    torch.set_rng_state(state["torch_cpu"].detach().to("cpu", dtype=torch.uint8))
    cuda_restored = False
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        cuda_states = [t.detach().to("cpu", dtype=torch.uint8) for t in cuda_states]
        n_dev = torch.cuda.device_count()
        if len(cuda_states) == n_dev:
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            for i in range(min(n_dev, len(cuda_states))):
                torch.cuda.set_rng_state(cuda_states[i], device=i)
        cuda_restored = True
    elif cuda_states is not None:
        print(f"[Checkpoint] {where}: CUDA RNG state present but CUDA is unavailable; skipped.")
    print(f"[Checkpoint] {where}: RNG state restored (python, numpy, torch CPU"
          f"{', torch CUDA' if cuda_restored else ''}).")
    LAST_RNG_RESTORE = {"where": where, "restored": True, "cuda": cuda_restored}
    return True


def _load_backbone_only(cfg, net: nn.Module, net_ema: Optional[nn.Module], path: str):
    if not os.path.exists(path):
        print(f"[Checkpoint] Backbone-only load: not found -> {path}")
        return None
    ckpt = torch.load(path, map_location=cfg.device)
    net.load_state_dict(ckpt["model"])
    if net_ema is not None and ckpt.get("net_ema") is not None:
        net_ema.load_state_dict(ckpt["net_ema"])
    print(f"[Checkpoint] Loaded backbone weights from {path}")
    return ckpt


def _save_checkpoint(
        stage: int,
        cfg,
        epoch_stage1: int,
        best_s1: float,
        net: nn.Module,
        net_ema: Optional[nn.Module],
        opts: List["SAM"],
        scheds: List[Optional[torch.optim.lr_scheduler._LRScheduler]],
        queues: List["ContrastiveQueue"],
        epoch_stage2: Optional[int] = None,
        best_s2: Optional[float] = None,
        head: Optional[nn.Module] = None,
        opt_head: Optional[torch.optim.Optimizer] = None,
        sched_head: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        path: str = STAGE1_LAST_PATH,
):
    """
    Generic checkpoint saver.

    Only saves torch state_dicts and raw tensors/ints/floats to avoid
    serialization problems with custom classes / lambda functions.
    """
    # Save queues as plain tensors + ints
    queue_states = []
    for q in queues:
        queue_states.append({
            "z": q.z.detach().cpu(),
            "y": q.y.detach().cpu(),
            "ptr": int(q.ptr),
            "full": bool(q.full),
            "size": int(q.size),
            "dim": int(q.dim),
        })

    state = {
        "stage": int(stage),
        "cfg": asdict(cfg),

        # Stage 1
        "epoch_stage1": int(epoch_stage1),
        "best_s1": float(best_s1),
        "model": net.state_dict(),
        "net_ema": net_ema.state_dict() if net_ema is not None else None,
        "opt_states": [
            opt.base_optimizer.state_dict() if opt.base_optimizer is not None else None
            for opt in opts
        ],
        "sched_states": [
            sch.state_dict() if sch is not None else None
            for sch in scheds
        ],
        "queues": queue_states,
        # [CR] RNG snapshot (restored on resume)
        RNG_STATE_KEY: _capture_rng_state(),
    }

    # Stage 2 (optional)
    if epoch_stage2 is not None:
        state["epoch_stage2"] = int(epoch_stage2)
        state["best_s2"] = float(best_s2 if best_s2 is not None else 0.0)
    if head is not None:
        state["head_state"] = head.state_dict()
    if opt_head is not None:
        state["opt_head_state"] = opt_head.state_dict()
    if sched_head is not None:
        state["sched_head_state"] = sched_head.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved (stage={stage}, s1_epoch={epoch_stage1}, s2_epoch={epoch_stage2}) -> {path}")


def _load_checkpoint_for_stage1(
        cfg,
        net: nn.Module,
        net_ema: Optional[nn.Module],
        opts: List["SAM"],
        scheds: List[Optional[torch.optim.lr_scheduler._LRScheduler]],
        queues: List["ContrastiveQueue"],
        path: str = STAGE1_LAST_PATH,
):
    """
    Load checkpoint (if exists) into Stage‑1 objects.
    Returns:
        ckpt (dict or None),
        start_epoch_stage1 (int),
        best_s1 (float)
    """
    if not os.path.exists(path):
        print("[Checkpoint] No checkpoint found. Starting Stage‑1 from scratch.")
        return None, 1, 0.0

    ckpt = torch.load(path, map_location=cfg.device)
    print("[Checkpoint] Loaded from", path)
    print(f"           stage={ckpt.get('stage', 1)}, "
          f"s1_epoch={ckpt.get('epoch_stage1', 0)}, "
          f"s2_epoch={ckpt.get('epoch_stage2', 0)}")

    # Load backbone
    net.load_state_dict(ckpt["model"])
    if net_ema is not None and ckpt.get("net_ema") is not None:
        net_ema.load_state_dict(ckpt["net_ema"])

    # Load optimizers
    opt_states = ckpt.get("opt_states", None)
    if opt_states is not None:
        for opt, st in zip(opts, opt_states):
            if opt.base_optimizer is not None and st is not None:
                opt.base_optimizer.load_state_dict(st)

    # Load schedulers
    sched_states = ckpt.get("sched_states", None)
    if sched_states is not None:
        for sch, st in zip(scheds, sched_states):
            if sch is not None and st is not None:
                sch.load_state_dict(st)

    # Load queues
    queue_states = ckpt.get("queues", None)
    if queue_states is not None:
        for q, qs in zip(queues, queue_states):
            if qs is None:
                continue
            if q.z.shape != qs["z"].shape or q.y.shape != qs["y"].shape:
                raise ValueError("Mismatch between saved queue shape and current queue shape.")
            q.z.copy_(qs["z"].to(cfg.device))
            q.y.copy_(qs["y"].to(cfg.device))
            q.ptr = int(qs["ptr"])
            q.full = bool(qs["full"])

    # [CR] Restore RNG for a Stage-1 checkpoint. A Stage-2 checkpoint restores it in
    # _load_checkpoint_for_stage2, after the Stage-2 head (whose init draws RNG) is built.
    if int(ckpt.get("stage", 1)) < 2:
        _restore_rng_state(ckpt.get(RNG_STATE_KEY), where=f"Stage-1 resume ({os.path.basename(path)})")

    epoch_stage1 = int(ckpt.get("epoch_stage1", 0))
    best_s1 = float(ckpt.get("best_s1", 0.0))
    start_epoch = epoch_stage1 + 1
    # If already finished Stage‑1, skip it
    if start_epoch > cfg.epochs_stage1:
        print(f"[Checkpoint] Stage‑1 already finished at epoch {epoch_stage1}.")
    else:
        print(f"[Checkpoint] Resuming Stage‑1 from epoch {start_epoch}.")

    return ckpt, start_epoch, best_s1


def _load_checkpoint_for_stage2(
        cfg,
        ckpt: Optional[dict],
        head: nn.Module,
        opt_head: torch.optim.Optimizer,
        sched_head: torch.optim.lr_scheduler._LRScheduler,
):
    """
    Load Stage‑2 state from checkpoint (if it exists and stage >= 2).
    Returns:
        start_epoch_stage2 (int),
        best_s2 (float)
    """
    if ckpt is None:
        print("[Checkpoint] No checkpoint for Stage‑2. Starting Stage‑2 from scratch.")
        return 1, 0.0

    if ckpt.get("stage", 1) < 2:
        print("[Checkpoint] Checkpoint has no Stage‑2 state yet. Starting Stage‑2 from scratch.")
        return 1, 0.0

    # Load head and its optimizer / scheduler
    if "head_state" in ckpt:
        head.load_state_dict(ckpt["head_state"])
    if "opt_head_state" in ckpt:
        opt_head.load_state_dict(ckpt["opt_head_state"])
    if "sched_head_state" in ckpt:
        sched_head.load_state_dict(ckpt["sched_head_state"])

    epoch_stage2 = int(ckpt.get("epoch_stage2", 0))
    best_s2 = float(ckpt.get("best_s2", 0.0))
    start_epoch = epoch_stage2 + 1
    if start_epoch > cfg.epochs_stage2:
        print(f"[Checkpoint] Stage‑2 already finished at epoch {epoch_stage2}.")
        start_epoch = cfg.epochs_stage2 + 1
    else:
        print(f"[Checkpoint] Resuming Stage‑2 from epoch {start_epoch} "
              f"(best acc={best_s2 * 100:.2f}%).")
    # [CR] Restore RNG last, so the next draw is the first draw of the resumed Stage-2 epoch.
    _restore_rng_state(ckpt.get(RNG_STATE_KEY), where="Stage-2 resume")
    return start_epoch, best_s2


# -------------------- small utils --------------------

def incorrect_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    r = torch.randint(0, num_classes, y.shape, device=y.device)
    return (r + (r == y).long()) % num_classes


def ff_margin_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return F.softplus(theta - g_pos).mean() + F.softplus(g_neg - theta).mean()


def pairwise_rank_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return F.softplus(-(g_pos - g_neg) / gamma).mean()


def sinusoid_encoding_table(n_pos: int, d: int, device=None) -> torch.Tensor:
    pe = torch.zeros(n_pos, d, device=device if device is not None else "cpu")
    position = torch.arange(0, n_pos, device=pe.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=pe.device) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def inv_softplus_init(target: float) -> float:
    y = max(1e-6, float(target))
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


# -------------------- RMSNorm, GEGLU, Losses, DropPath --------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def supcon_generic(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    # Supervised Contrastive Loss
    z = F.normalize(z, dim=1)
    N = z.size(0)
    sim = torch.matmul(z, z.T) / max(1e-6, temperature)
    logits_mask = torch.ones_like(sim) - torch.eye(N, device=z.device)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * logits_mask
    exp_logits = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_counts + 1e-12)
    valid_mask = pos_counts > 0
    if valid_mask.any():
        return -mean_log_prob_pos[valid_mask].mean()
    else:
        return torch.tensor(0.0, device=z.device)


class LabelSmoothingCE(nn.Module):
    # Updated default num_classes to 10 for CIFAR-10
    def __init__(self, eps: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        if target.ndim == 1:
            nll = F.nll_loss(log_probs, target, reduction='mean')
            smooth = -log_probs.mean(dim=1).mean()
            return (1 - self.eps) * nll + self.eps * smooth
        else:
            # Handle soft targets (Mixup/CutMix)
            return -(target * log_probs).sum(dim=1).mean()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -------------------- hybrid ViT stem, MoE, Memory Encoder, RoPE, Attention --------------------

class ConvStem(nn.Module):
    def __init__(self, in_chans: int = 3, out_chans: int = 128):
        super().__init__()
        mid = max(64, out_chans // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_chans, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
        )
        for m in self.stem:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256, dropout=0.0,
                 use_conv_stem: bool = True, stem_channels: int = 128, use_abs_pos_embed: bool = True):
        super().__init__()
        self.use_conv_stem = use_conv_stem
        self.use_abs_pos_embed = use_abs_pos_embed
        self.stem = ConvStem(in_chans, stem_channels) if use_conv_stem else None
        reduced = img_size // (2 if use_conv_stem else 1)
        assert reduced % patch_size == 0
        self.grid = reduced // patch_size
        self.num_patches = self.grid * self.grid
        in_c = stem_channels if use_conv_stem else in_chans
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if self.use_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv_stem:
            x = self.stem(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        return self.drop(x)


class TopKMoEMLP(nn.Module):
    """
    Token-choice top-k MoE MLP with explicit (Switch-style) load-balancing loss and
    ST-MoE router z-loss, plus routing diagnostics to detect expert collapse.

    Stored (detached) after every forward:
        - last_mean_p: [E] mean router probability per expert (importance distribution)
        - last_load:   [E] fraction of tokens whose top-1 expert is i (load distribution)
        - last_counts: [E] raw counts of top-1 routed tokens
        - last_importance: [E] sum of router probabilities across tokens (sums to #tokens)
        - last_balance_switch: scalar  (Eq.4 in Switch Transformers)
        - last_balance_entropy: scalar (KL(mean_p || uniform))
        - last_z_loss: scalar (Eq.5 in ST-MoE, logsumexp^2)

    Returns:
        y: [B, L, D]
        balance: scalar (depends on self.balance_type)
        z_loss: scalar (depends on self.z_loss_type)
    """

    def __init__(
            self,
            d_model: int,
            n_experts: int = 4,
            mult: int = 4,
            dropout: float = 0.0,
            gate_tau: float = 1.0,
            top_k: int = 2,
            balance_type: str = "switch",  # {"switch","entropy"}
            z_loss_type: str = "stmoe",  # {"stmoe","l2"}
    ):
        super().__init__()
        self.n_experts = n_experts
        self.tau = gate_tau
        self.top_k = top_k
        self.balance_type = balance_type
        self.z_loss_type = z_loss_type

        # Routing diagnostics (populated each forward; detached)
        self.last_mean_p: Optional[torch.Tensor] = None
        self.last_load: Optional[torch.Tensor] = None
        self.last_counts: Optional[torch.Tensor] = None
        self.last_importance: Optional[torch.Tensor] = None
        self.last_balance_switch: Optional[torch.Tensor] = None
        self.last_balance_entropy: Optional[torch.Tensor] = None
        self.last_z_loss: Optional[torch.Tensor] = None

        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    GEGLU(d_model, mult * d_model),
                    nn.Linear(mult * d_model, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(n_experts)
            ]
        )
        for e in self.experts:
            nn.init.xavier_normal_(e[1].weight)
            if e[1].bias is not None:
                nn.init.zeros_(e[1].bias)
        nn.init.xavier_normal_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D]
        returns:
            y: [B, L, D]
            balance: scalar
            z_loss: scalar
        """
        B, L, D = x.shape
        E = self.n_experts
        K = int(self.top_k) if (self.top_k is not None) else E
        K = max(1, min(E, K))

        # Router logits (temperature-scaled)
        logits = self.gate(x) / max(1e-6, float(self.tau))  # [B, L, E]

        # Top-k masking (token-choice)
        if K < E:
            _, topi = logits.topk(K, dim=-1)  # [B, L, K]
            mask = torch.zeros_like(logits).scatter_(-1, topi, 1.0)
            masked = logits.masked_fill(mask == 0, float("-inf"))
            probs_full = masked.softmax(dim=-1)  # [B, L, E]
        else:
            probs_full = logits.softmax(dim=-1)  # [B, L, E]
            topi = torch.arange(E, device=x.device).view(1, 1, E).expand(B, L, E)

        # ---------------- Router diagnostics ----------------
        tokens = float(B * L)

        mean_p = probs_full.mean(dim=(0, 1)) + 1e-9  # [E] (importance distribution, sums to 1)
        top1 = probs_full.argmax(dim=-1).reshape(-1)  # [T]
        counts = torch.bincount(top1, minlength=E).float()  # [E] (top-1 load counts)
        load = counts / max(1.0, tokens)  # [E] (load distribution, sums to 1)
        importance = probs_full.sum(dim=(0, 1)).float()  # [E] (sums to #tokens)

        # Save detached summaries so the training loop can aggregate across an epoch
        self.last_mean_p = mean_p.detach()
        self.last_counts = counts.detach()
        self.last_load = load.detach()
        self.last_importance = importance.detach()

        # Switch Transformers auxiliary load-balancing loss:
        # loss = α * N * sum_i f_i * P_i
        # Here we return the unweighted "N * sum f_i P_i" (α applied outside).
        balance_switch = float(E) * (load * mean_p).sum()
        # Entropy-style KL(mean_p || uniform) (>=0), sometimes used as a smoother proxy
        balance_entropy = (mean_p * mean_p.log()).sum() + math.log(E)

        self.last_balance_switch = balance_switch.detach()
        self.last_balance_entropy = balance_entropy.detach()

        bt = str(self.balance_type).lower().strip()
        if bt in {"entropy", "kl", "kld"}:
            balance = balance_entropy
        else:
            # default: Switch-style
            balance = balance_switch

        # Router z-loss (ST-MoE):
        # Lz = mean_t (logsumexp(router_logits_t))^2
        zt = str(self.z_loss_type).lower().strip()
        if zt in {"l2", "mse"}:
            z_loss = (logits ** 2).mean()
        else:
            z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        self.last_z_loss = z_loss.detach()

        # ---------------- Expert computation ----------------
        probs_top = torch.gather(probs_full, dim=-1, index=topi)  # [B, L, K]

        x_flat = x.reshape(B * L, D)  # [T, D]
        topi_flat = topi.reshape(-1, K)  # [T, K]
        probs_flat = probs_top.reshape(-1, K)  # [T, K]

        y_flat = x_flat.new_zeros(x_flat.size(0), D)  # [T, D]

        for expert_id, expert in enumerate(self.experts):
            mask_e = (topi_flat == expert_id)  # [T, K]
            if not mask_e.any():
                continue

            token_idx, which_k = mask_e.nonzero(as_tuple=True)  # [N_e], [N_e]
            x_e = x_flat[token_idx]  # [N_e, D]
            out_e = expert(x_e)  # [N_e, D]
            w_e = probs_flat[token_idx, which_k].unsqueeze(-1)  # [N_e, 1]

            y_flat[token_idx] += out_e * w_e

        y = y_flat.view(B, L, D)
        return y, balance, z_loss


class ScalarMemEncoder(nn.Module):
    def __init__(self, d_model: int, max_depth: int, in_dims: int = 1, norm_layer=RMSNorm):
        super().__init__()
        self.d_model = d_model
        self.to_tok = nn.Sequential(nn.Linear(in_dims, d_model), nn.GELU(), norm_layer(d_model))
        pe = sinusoid_encoding_table(max_depth, d_model)
        self.register_buffer("depth_pe", pe, persistent=False)

    def forward(self, g_vec: torch.Tensor) -> torch.Tensor:
        if g_vec is None or g_vec.numel() == 0:
            B = g_vec.size(0) if g_vec is not None else 0
            device = g_vec.device if g_vec is not None else self.depth_pe.device
            return torch.zeros(B, 0, self.d_model, device=device)
        B, Lm, _ = g_vec.shape
        x = self.to_tok(g_vec)
        pe = self.depth_pe[:Lm].unsqueeze(0).expand(B, Lm, -1)
        return x + pe


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def _get_cache(self, seq_len: int, device):
        key = (seq_len, device)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cache[key] = (emb.cos().to(device), emb.sin().to(device))
        return self._cache[key]

    def forward(self, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get_cache(seq_len, device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, use_rope: bool = False,
                 rotary_emb: Optional[RotaryEmbedding] = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.rotary_emb = rotary_emb if use_rope else None
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, need_weights: bool = False):
        B, N, D = x.shape
        is_cross_attn = context is not None
        q = self.Wq(x)
        if is_cross_attn:
            k = self.Wk(context)
            v = self.Wv(context)
            N_ctx = context.shape[1]
        else:
            k = self.Wk(x)
            v = self.Wv(x)
            N_ctx = N
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and self.rotary_emb is not None and not is_cross_attn:
            cos, sin = self.rotary_emb(N, device=x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use efficient SDPA if available
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )
            # NOTE: PyTorch's SDPA fast-path does not expose attention weights.
            # For "attention sharpness" diagnostics/aspects we optionally compute
            # the probabilities explicitly from q,k (cheap for short sequences and
            # memory cross-attn).
            if need_weights:
                scale = 1.0 / math.sqrt(self.head_dim)
                attn_logits = (q @ k.transpose(-2, -1)) * scale
                attn_w = F.softmax(attn_logits, dim=-1)
            else:
                attn_w = None
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn_w = F.softmax(attn, dim=-1)
            attn_drop = F.dropout(attn_w, p=self.dropout_p, training=self.training)
            y = attn_drop @ v

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        out = self.resid_dropout(self.Wo(y))
        return out, attn_w


class AttentionPool(nn.Module):
    """Pools sequence features using a learnable query vector."""

    def __init__(self, d_model: int, n_heads: int, norm_layer=RMSNorm):
        super().__init__()
        self.ln = norm_layer(d_model)
        self.mha = Attention(d_model, n_heads, use_rope=False)
        self.q_vec = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.q_vec, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln(x)
        B = x.size(0)
        q = self.q_vec.expand(B, -1, -1)
        pooled_out, _ = self.mha(q, context=x_norm)
        return pooled_out.squeeze(1)


# -------------------- FF block (Synthesized Design) --------------------

class FFHybridBlock(nn.Module):
    def __init__(self, d_model: int, num_classes: int, num_heads: int,
                 n_experts: int, mlp_mult: int, dropout: float,
                 gate_tau: float, max_depth: int,
                 n_aspects: int, aspect_theta_inits: Optional[List[float]],
                 block_theta_init: float, use_softmax_weights: bool,
                 proj_dim: int, drop_path: float, moe_top_k: int,
                 use_rope: bool, rotary_emb: Optional[RotaryEmbedding],
                 recon_mult: int, proto_temp: float,
                 label_embed_dim: int, use_mem: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_aspects = n_aspects
        self.use_softmax_weights = use_softmax_weights
        self.proto_temp = proto_temp
        self.use_mem = bool(use_mem)

        # Label embedding
        self.label_embed_dim = label_embed_dim
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        nn.init.normal_(self.label_emb.weight, std=0.02)

        norm_layer = RMSNorm

        # Input processing
        self.ln_in = norm_layer(d_model + label_embed_dim)
        self.in_proj = nn.Linear(d_model + label_embed_dim, d_model)
        nn.init.xavier_normal_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        # Self-Attention (with RoPE)
        self.ln_attn = norm_layer(d_model)
        self.self_attn = Attention(d_model, num_heads, dropout=dropout, use_rope=use_rope, rotary_emb=rotary_emb)
        self.dp1 = DropPath(drop_path)

        # Memory Cross-Attention
        self.mem_enc = ScalarMemEncoder(d_model, max_depth, in_dims=n_aspects, norm_layer=norm_layer)
        self.ln_gattn = norm_layer(d_model)
        self.g_attn = Attention(d_model, num_heads, dropout=dropout, use_rope=False)
        self.dp2 = DropPath(drop_path)

        # MoE
        self.ln_moe = norm_layer(d_model)
        self.moe = TopKMoEMLP(d_model, n_experts=n_experts, mult=mlp_mult, dropout=dropout,
                              gate_tau=gate_tau, top_k=moe_top_k)
        self.dp3 = DropPath(drop_path)

        # Attention Pooling
        self.attn_pool = AttentionPool(d_model, n_heads=num_heads, norm_layer=norm_layer)

        # Prototypes
        self.prototypes = nn.Parameter(torch.empty(num_classes, d_model))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

        # Learned Aspect Head
        self.ln_head = norm_layer(d_model)
        head_mid = max(64, d_model // 4)
        self.lin_head = nn.Sequential(
            nn.Linear(d_model, head_mid),
            nn.GELU(),
            nn.Linear(head_mid, 1)
        )

        # Contrastive projection
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        for m in self.contrastive_proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Reconstruction decoder
        self.reconstructor = nn.Sequential(
            norm_layer(d_model),
            nn.Linear(d_model, d_model * recon_mult),
            nn.GELU(),
            nn.Linear(d_model * recon_mult, d_model),
        )

        # Theta init
        if aspect_theta_inits is None or len(aspect_theta_inits) < n_aspects:
            raise ValueError("aspect_theta_inits must be correctly configured.")

        self.theta_aspect_raw = nn.Parameter(
            torch.tensor([inv_softplus_init(x) for x in aspect_theta_inits], dtype=torch.float)
        )
        self.alpha_raw = nn.Parameter(torch.zeros(n_aspects))
        self.block_theta_raw = nn.Parameter(
            torch.tensor(inv_softplus_init(block_theta_init), dtype=torch.float)
        )

    def get_pooled_rep(self, a: torch.Tensor) -> torch.Tensor:
        # representation used for downstream tasks
        return self.attn_pool(a)

    def _append_label(self, tokens: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, N, _ = tokens.shape
        y_emb = self.label_emb(y)  # [B, label_embed_dim]
        ytok = y_emb.unsqueeze(1).expand(B, N, self.label_embed_dim)
        return torch.cat([tokens, ytok], dim=-1)

    def _self_attend(self, h: torch.Tensor) -> torch.Tensor:
        q = self.ln_attn(h)
        out, _ = self.self_attn(q)
        return h + self.dp1(out)

    def _cross_attend_mem(self, h: torch.Tensor, g_prev: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h.size(0)
        if not self.use_mem:
            return h, h.new_zeros(B)
        if g_prev is None or g_prev.numel() == 0:
            return h, h.new_zeros(B)
        q = self.ln_gattn(h)
        mem = self.mem_enc(g_prev)
        out, attn_w = self.g_attn(q, context=mem, need_weights=True)

        if attn_w is not None:
            sharp = attn_w.max(dim=-1).values.mean(dim=(1, 2))
        else:
            sharp = h.new_zeros(B)

        return h + self.dp2(out), sharp

    def _compute_aspects(self, h: torch.Tensor, attn_sharp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pooled = self.attn_pool(h)  # [B, D]

        # Prototype alignment
        z = F.normalize(pooled, p=2, dim=1)
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=1)
        target_prototypes = F.embedding(y, normalized_prototypes)
        similarity = (z * target_prototypes).sum(dim=1)
        g_alignment = similarity / self.proto_temp

        # Other aspects
        g_energy = (h * h).mean(dim=(1, 2))
        g_attn = attn_sharp.clamp(min=0.0)
        g_learned = F.softplus(self.lin_head(self.ln_head(pooled))).squeeze(1)

        g_list = [g_alignment, g_energy, g_attn, g_learned]
        return torch.stack(g_list[:self.n_aspects], dim=1)

    def aggregate_goodness(self, g_vec: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.alpha_raw, dim=0) if self.use_softmax_weights else F.softplus(self.alpha_raw)
        return (g_vec * w.unsqueeze(0)).sum(dim=1)

    def aspect_thetas(self) -> torch.Tensor:
        return F.softplus(self.theta_aspect_raw)

    def block_theta(self) -> torch.Tensor:
        return F.softplus(self.block_theta_raw)

    def contrastive_embed(self, a: torch.Tensor) -> torch.Tensor:
        # representation used for contrastive loss only
        h = self.get_pooled_rep(a)
        z = self.contrastive_proj(h)
        return F.normalize(z, dim=1)

    def forward(self, tokens: torch.Tensor, y: torch.Tensor, g_prev: Optional[torch.Tensor]):
        x = self._append_label(tokens, y)
        h = self.in_proj(self.ln_in(x))

        h = self._self_attend(h)
        h, attn_sharp = self._cross_attend_mem(h, g_prev)
        m = self.ln_moe(h)
        moe_out, balance, zloss = self.moe(m)
        h = h + self.dp3(moe_out)

        recon = self.reconstructor(h) if self.training else None

        g_aspects = self._compute_aspects(h, attn_sharp, y)
        a = F.relu(h)

        return a, g_aspects, balance, zloss, recon


# -------------------- full model --------------------

class FFHybridNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.num_blocks = cfg.num_blocks
        self.n_aspects = cfg.n_aspects
        self.use_rope = cfg.use_rope

        use_abs_pos_embed = not cfg.use_rope

        self.embed = PatchEmbed(img_size=cfg.img_size, patch_size=cfg.patch_size, in_chans=3, embed_dim=cfg.d_model,
                                dropout=cfg.dropout, use_conv_stem=cfg.use_conv_stem, stem_channels=cfg.stem_channels,
                                use_abs_pos_embed=use_abs_pos_embed)

        if cfg.use_rope:
            rope_dim = cfg.d_model // cfg.num_heads
            self.rotary_emb = RotaryEmbedding(dim=rope_dim, theta=cfg.rope_theta)
        else:
            self.rotary_emb = None

        dprs = torch.linspace(0, cfg.drop_path_rate, steps=cfg.num_blocks).tolist()

        self.blocks = nn.ModuleList([
            FFHybridBlock(
                d_model=cfg.d_model,
                num_classes=cfg.num_classes,
                num_heads=cfg.num_heads,
                n_experts=cfg.n_experts,
                mlp_mult=cfg.mlp_mult,
                dropout=cfg.dropout,
                gate_tau=cfg.gate_tau_start,
                max_depth=cfg.num_blocks,
                n_aspects=cfg.n_aspects,
                aspect_theta_inits=cfg.aspect_theta_inits,
                block_theta_init=cfg.block_theta_init,
                use_softmax_weights=cfg.use_softmax_weights,
                proj_dim=cfg.proj_dim,
                drop_path=dprs[i],
                moe_top_k=cfg.moe_top_k,
                use_rope=cfg.use_rope,
                rotary_emb=self.rotary_emb,
                recon_mult=cfg.recon_mult,
                proto_temp=cfg.proto_temp,
                label_embed_dim=cfg.label_embed_dim,
                use_mem=cfg.use_mem,
            )
            for i in range(cfg.num_blocks)
        ])

        # Configure MoE auxiliary-loss style (aligns with Switch + ST-MoE by default)
        bal_type = getattr(cfg, "moe_balance_type", "switch")
        z_type = getattr(cfg, "moe_zloss_type", "stmoe")
        for blk in self.blocks:
            try:
                blk.moe.balance_type = bal_type
                blk.moe.z_loss_type = z_type
            except Exception:
                pass

    @torch.no_grad()
    def predict_scores(
            self,
            images: torch.Tensor,
            max_blocks: Optional[int] = None,
            return_all_depths: bool = False,
    ) -> torch.Tensor:
        """
        Forward-Forward inference by summing per-block goodness for each class hypothesis.

        Args:
            images: [B, 3, H, W]
            max_blocks: if set, only use the first `max_blocks` blocks.
            return_all_depths: if True, return cumulative class scores after each block,
                shape [B, L_use, C]. Otherwise return final scores [B, C].

        Returns:
            scores: [B, C] or [B, L_use, C]
        """
        self.eval()
        B = images.size(0)
        tokens0 = self.embed(images)

        blocks = self.blocks
        if max_blocks is not None:
            blocks = blocks[: int(max_blocks)]
        L_use = len(blocks)

        if return_all_depths:
            scores_depth = torch.zeros(B, L_use, self.num_classes, device=images.device)

        all_scores = []

        for k in range(self.num_classes):
            yk = torch.full((B,), k, dtype=torch.long, device=images.device)
            tokens = tokens0
            g_hist = []
            total_g = torch.zeros(B, device=images.device)
            for bi, blk in enumerate(blocks):
                g_prev = torch.stack(g_hist, dim=1) if len(g_hist) > 0 else None
                if g_prev is not None:
                    g_prev = g_prev.detach()
                a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
                total_g = total_g + blk.aggregate_goodness(g_vec)
                g_hist.append(g_vec)
                tokens = l2_normalize_tokens(a)
                if return_all_depths:
                    scores_depth[:, bi, k] = total_g
            if not return_all_depths:
                all_scores.append(total_g.unsqueeze(1))

        if return_all_depths:
            return scores_depth
        return torch.cat(all_scores, dim=1)

    @torch.no_grad()
    def extract_hybrid_features_all_labels(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts features for all class hypotheses using embeddings from ALL layers.
        Returns:
            feats_g (torch.Tensor): [B, C, D_g]
            feats_a (torch.Tensor): [B, C, D_a * N_blocks]
        """
        self.eval()
        B = images.size(0)
        tokens0 = self.embed(images)
        feats_g = []
        feats_a = []

        for k in range(self.num_classes):
            yk = torch.full((B,), k, dtype=torch.long, device=images.device)
            tokens = tokens0
            g_hist = []
            a_embed_hist = []

            for blk in self.blocks:
                g_prev = torch.stack(g_hist, dim=1) if g_hist else None
                if g_prev is not None:
                    g_prev = g_prev.detach()
                a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
                g_hist.append(g_vec)
                # pooled_a = blk.contrastive_embed(a)
                pooled_a = blk.get_pooled_rep(a)
                a_embed_hist.append(pooled_a)
                tokens = l2_normalize_tokens(a)

            feats_g.append(torch.cat(g_hist, dim=1).unsqueeze(1))
            feats_a.append(torch.cat(a_embed_hist, dim=1).unsqueeze(1))

        return torch.cat(feats_g, dim=1), torch.cat(feats_a, dim=1)

    def set_router_tau(self, tau_base: float, depth_slope: float = 0.0):
        L = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            scale = (1.0 + depth_slope * (i / max(1, L - 1)))
            blk.moe.tau = max(1e-6, tau_base * scale)


# -------------------- Stage 2 Head, Queue, Mixup, Data, EMA, Eval, Hard Negatives, SAM --------------------


class AttentiveHybridHead(nn.Module):
    def __init__(self, in_dim_g: int, in_dim_a: int, hidden: int, num_classes: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        norm_layer = RMSNorm
        self.in_dim = in_dim_g + in_dim_a
        self.num_classes = num_classes
        self.hidden = hidden

        self.proj = nn.Sequential(
            norm_layer(self.in_dim),
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm_attn = norm_layer(hidden)
        self.attn = Attention(hidden, num_heads, dropout=dropout, use_rope=False)
        self.dropout_attn = nn.Dropout(dropout)

        self.norm_ffn = norm_layer(hidden)
        self.ffn = nn.Sequential(
            GEGLU(hidden, hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, feats_g: torch.Tensor, feats_a: torch.Tensor) -> torch.Tensor:
        B_C = feats_g.size(0)
        C = self.num_classes
        B = B_C // C

        h_fused = torch.cat([feats_g, feats_a], dim=1)
        h = self.proj(h_fused)
        h = h.view(B, C, self.hidden)

        h_norm = self.norm_attn(h)
        attn_out, _ = self.attn(h_norm)
        h = h + self.dropout_attn(attn_out)

        h_norm = self.norm_ffn(h)
        ffn_out = self.ffn(h_norm)
        h = h + ffn_out

        return self.classifier(h).squeeze(-1)


class ContrastiveQueue:
    def __init__(self, size: int, dim: int, device: str):
        self.size = size
        self.dim = dim
        self.device = device
        self.ptr = 0
        self.full = False
        self.z = torch.zeros(size, dim, device=device)
        self.y = torch.zeros(size, dtype=torch.long, device=device)

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        B = feats.size(0)
        if B >= self.size:
            self.z.copy_(feats[-self.size:])
            self.y.copy_(labels[-self.size:])
            self.ptr = 0
            self.full = True
            return
        end = self.ptr + B
        if end <= self.size:
            self.z[self.ptr:end] = feats
            self.y[self.ptr:end] = labels
        else:
            first = self.size - self.ptr
            self.z[self.ptr:] = feats[:first]
            self.y[self.ptr:] = labels[:first]
            self.z[:end - self.size] = feats[first:]
            self.y[:end - self.size] = labels[first:]
            self.full = True
        self.ptr = (self.ptr + B) % self.size

    @torch.no_grad()
    def sample(self, k: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        n = self.size if self.full else self.ptr
        if n == 0 or k <= 0:
            return None, None
        k = min(k, n)
        idx = torch.randint(0, n, (k,), device=self.device)
        return self.z[idx], self.y[idx]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_w // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class Mixup:
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, num_classes=10):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes

    def __call__(self, x, target):
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).float()
        if np.random.rand() > self.prob:
            return x, target
        use_cutmix = np.random.rand() < self.switch_prob
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        if alpha <= 0:
            return x, target
        lam = np.random.beta(alpha, alpha)
        indices = torch.randperm(x.size(0), device=x.device)
        if use_cutmix:
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                       (x.size()[-1] * x.size()[-2]))
        else:
            x = x * lam + x[indices] * (1. - lam)
        target_shuffled = target[indices]
        target = target * lam + target_shuffled * (1. - lam)
        return x, target


class CIFAR10Triple(datasets.CIFAR10):
    def __init__(self, root, train, transform1, transform2, transform3, download=False):
        super().__init__(root=root, train=train, transform=None, download=download)
        self.t1 = transform1
        self.t2 = transform2
        self.t3 = transform3

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return self.t1(img), self.t2(img), self.t3(img), target


class CIFAR10SingleFromBase(torch.utils.data.Dataset):
    """Lightweight wrapper to reuse an existing CIFAR-10 dataset instance.

    Why this exists:
      - Creating *another* datasets.CIFAR10(train=True) instance duplicates the full 50k
        images array in memory in the parent process.
      - With num_workers>0, DataLoader workers can end up consuming memory proportional
        to the parent process size, which can trigger worker OOM / hangs on some systems.

    This wrapper reuses the *same* underlying CIFAR-10 train split already loaded by
    `CIFAR10Triple`, but returns a single (optionally transformed) view: (x, y).
    """

    def __init__(self, base_cifar: datasets.CIFAR10, transform=None):
        super().__init__()
        self.base = base_cifar
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Use the base CIFAR-10 __getitem__ to get raw PIL image + label.
        # CIFAR10Triple sets `transform=None`, so this yields the un-augmented image.
        img, target = datasets.CIFAR10.__getitem__(self.base, int(idx))
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_loaders_stage1(
        batch_size: int,
        test_batch_size: int,
        seed: int,
        val_fraction: float = 0.1,
        workers: int = 4,
):
    """
    Stage-1 loaders with a deterministic train/val split from the CIFAR-10 *training* split.

    Returns:
        train_loader_s1: CIFAR-10 train subset with triple-view augmentations
        val_loader:      held-out subset (from CIFAR-10 train split), no augmentation
        test_loader:     official CIFAR-10 test split, no augmentation
        split_info:      dict with train_idx / val_idx (original CIFAR-10 train indices)
    """
    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)

    def create_aug(magnitude: int):
        return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandAugment(num_ops=2, magnitude=magnitude),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Three stochastic views for Stage-1 local losses
    aug1 = create_aug(10)
    aug2 = create_aug(8)
    aug3 = create_aug(10)

    # No augmentation for val/test (avoid test peeking + keep protocol clean)
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_dir = "./data_c10"

    # Full CIFAR-10 train split (50k)
    train_full_s1 = CIFAR10Triple(
        data_dir,
        train=True,
        transform1=aug1,
        transform2=aug2,
        transform3=aug3,
        download=True,
    )
    n_total = len(train_full_s1)

    # Deterministic train/val split (seeded)
    # IMPORTANT: avoid torch.randperm/torch.Generator here.
    # On some Unix setups, calling OpenMP/torch ops before DataLoader forks workers
    # can trigger rare deadlocks/hangs. Using a local Python RNG keeps this split
    # deterministic without touching torch's thread pools.
    vf = float(val_fraction)
    vf = max(0.0, min(0.5, vf))
    n_val = int(round(n_total * vf))
    n_val = max(1, min(n_total - 1, n_val))

    rng = random.Random(int(seed))
    perm = list(range(n_total))
    rng.shuffle(perm)
    val_idx = sorted(int(i) for i in perm[:n_val])
    train_idx = sorted(int(i) for i in perm[n_val:])

    train_ds_s1 = Subset(train_full_s1, train_idx)

    # Val split: CIFAR-10 train images but no augmentation.
    # Reuse the already-loaded CIFAR-10 train split to avoid duplicating the full
    # 50k images array in memory.
    val_full = CIFAR10SingleFromBase(train_full_s1, transform=eval_tf)
    val_ds = Subset(val_full, val_idx)

    # Official CIFAR-10 test split: no augmentation
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    persistent = workers > 0
    train_loader_s1 = DataLoader(
        train_ds_s1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    split_info = {
        "seed": int(seed),
        "val_fraction": float(vf),
        "n_total": int(n_total),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    return train_loader_s1, val_loader, test_loader, split_info


def make_loader_stage2(
        batch_size: int,
        train_indices: List[int],
        workers: int = 4,
):
    """
    Stage-2 loader: trains the head ONLY on the Stage-1 train subset (excludes val indices).
    """
    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data_dir = "./data_c10"
    ds_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    if train_indices is not None:
        ds = Subset(ds_full, list(train_indices))
    else:
        ds = ds_full

    persistent = workers > 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
    )


@torch.no_grad()
def update_ema(model_src: nn.Module, model_tgt: nn.Module, decay: float = 0.999):
    for p_tgt, p_src in zip(model_tgt.parameters(), model_src.parameters()):
        if p_src.requires_grad:
            p_tgt.data.mul_(decay).add_(p_src.data, alpha=1 - decay)
    for b_tgt, b_src in zip(model_tgt.buffers(), model_src.buffers()):
        b_tgt.copy_(b_src)


@torch.no_grad()
def topk_correct_counts(
        logits: torch.Tensor,
        targets: torch.Tensor,
        topk: Tuple[int, ...] = (1,)
) -> Dict[int, int]:
    """
    Returns counts (not percentages) of correct predictions for each k in topk.
    logits: [B, C]
    targets: [B] (or one-hot / soft labels [B, C] -> argmax will be used)
    """
    if targets.ndim > 1:
        targets = targets.argmax(dim=1)

    B, C = logits.shape
    ks = tuple(int(k) for k in topk)
    maxk = min(max(ks), C)

    # topk indices: [B, maxk]
    pred = logits.topk(maxk, dim=1, largest=True, sorted=True).indices  # :contentReference[oaicite:1]{index=1}
    correct = pred.eq(targets.view(-1, 1))  # [B, maxk]

    out: Dict[int, int] = {}
    for k in ks:
        kk = min(k, maxk)
        out[k] = int(correct[:, :kk].any(dim=1).sum().item())
    return out


@torch.no_grad()
def evaluate_ff(
        net: FFHybridNet,
        loader: DataLoader,
        device: str,
        tta: bool = True,
        topk: Tuple[int, ...] = tuple(range(1, 11)),
) -> Dict[int, float]:
    net.eval()
    total = 0
    correct_counts = {k: 0 for k in topk}

    for batch in loader:
        if len(batch) == 4:
            images, _, _, labels = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        if tta:
            scores = net.predict_scores(images) + net.predict_scores(torch.flip(images, dims=[3]))
        else:
            scores = net.predict_scores(images)

        batch_counts = topk_correct_counts(scores, labels, topk=topk)
        for k in topk:
            correct_counts[k] += batch_counts[k]
        total += labels.numel()

    denom = max(1, total)
    return {k: correct_counts[k] / denom for k in topk}


@torch.no_grad()
def evaluate_ff_by_depth(
        net: FFHybridNet,
        loader: DataLoader,
        device: str,
        tta: bool = True,
        depths: Optional[Tuple[int, ...]] = None,
        topk: Tuple[int, ...] = (1,),
) -> Dict[int, Dict[int, float]]:
    """
    Evaluate Forward-Forward inference at multiple truncation depths (early-exit diagnostic).

    Returns:
        A nested dict: depth -> {k -> accuracy}
    """
    net.eval()
    if depths is None:
        depths = tuple(range(1, net.num_blocks + 1))
    depths = tuple(int(d) for d in depths)
    max_depth = int(max(depths)) if len(depths) > 0 else net.num_blocks

    total = 0
    correct_counts = {d: {k: 0 for k in topk} for d in depths}

    for batch in loader:
        if len(batch) == 4:
            images, _, _, labels = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        if tta:
            scores_depth = net.predict_scores(images, max_blocks=max_depth, return_all_depths=True)
            scores_depth = scores_depth + net.predict_scores(
                torch.flip(images, dims=[3]), max_blocks=max_depth, return_all_depths=True
            )
        else:
            scores_depth = net.predict_scores(images, max_blocks=max_depth, return_all_depths=True)

        for d in depths:
            logits_d = scores_depth[:, d - 1, :]
            batch_counts = topk_correct_counts(logits_d, labels, topk=topk)
            for k in topk:
                correct_counts[d][k] += batch_counts[k]

        total += labels.numel()

    denom = max(1, total)
    return {d: {k: correct_counts[d][k] / denom for k in topk} for d in depths}


@torch.no_grad()
def evaluate_stage2(
        net: FFHybridNet,
        head: AttentiveHybridHead,
        loader: DataLoader,
        device: str,
        tta: bool = True,
        topk: Tuple[int, ...] = tuple(range(1, 11)),
) -> Dict[int, float]:
    net.eval()
    head.eval()
    total = 0
    correct_counts = {k: 0 for k in topk}
    C = net.num_classes

    for batch in loader:
        if len(batch) == 4:
            images, _, _, labels = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        B = images.size(0)

        feats_g, feats_a = net.extract_hybrid_features_all_labels(images)
        feats_g_flat = feats_g.view(B * C, -1)
        feats_a_flat = feats_a.view(B * C, -1)
        logits = head(feats_g_flat, feats_a_flat)

        if tta:
            feats_g2, feats_a2 = net.extract_hybrid_features_all_labels(torch.flip(images, dims=[3]))
            feats_g2_flat = feats_g2.view(B * C, -1)
            feats_a2_flat = feats_a2.view(B * C, -1)
            logits = (logits + head(feats_g2_flat, feats_a2_flat)) / 2.0

        batch_counts = topk_correct_counts(logits, labels, topk=topk)
        for k in topk:
            correct_counts[k] += batch_counts[k]
        total += labels.numel()

    denom = max(1, total)
    return {k: correct_counts[k] / denom for k in topk}


@torch.no_grad()
def collect_ff_predictions(
        net: FFHybridNet,
        loader: DataLoader,
        device: str,
        tta: bool = False,
        save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect per-example predictions for Forward-Forward inference.

    Saves (optionally) an .npz with:
        y_true: [N]
        y_pred: [N]
        scores: [N, C]
    """
    net.eval()
    y_true_list = []
    y_pred_list = []
    scores_list = []

    for batch in loader:
        if len(batch) == 4:
            images, _, _, labels = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        if tta:
            scores = net.predict_scores(images) + net.predict_scores(torch.flip(images, dims=[3]))
        else:
            scores = net.predict_scores(images)

        preds = scores.argmax(dim=1)

        y_true_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(preds.detach().cpu().numpy())
        scores_list.append(scores.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    scores_np = np.concatenate(scores_list, axis=0)

    out = {"y_true": y_true, "y_pred": y_pred, "scores": scores_np}
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **out)
    return out


@torch.no_grad()
def collect_stage2_predictions(
        net: FFHybridNet,
        head: AttentiveHybridHead,
        loader: DataLoader,
        device: str,
        tta: bool = False,
        save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect per-example predictions for Stage-2 head inference.

    Saves (optionally) an .npz with:
        y_true: [N]
        y_pred: [N]
        scores: [N, C]
    """
    net.eval()
    head.eval()

    y_true_list = []
    y_pred_list = []
    scores_list = []

    C = net.num_classes

    for batch in loader:
        if len(batch) == 4:
            images, _, _, labels = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        B = images.size(0)

        feats_g, feats_a = net.extract_hybrid_features_all_labels(images)
        feats_g_flat = feats_g.view(B * C, -1)
        feats_a_flat = feats_a.view(B * C, -1)
        logits = head(feats_g_flat, feats_a_flat)

        if tta:
            feats_g2, feats_a2 = net.extract_hybrid_features_all_labels(torch.flip(images, dims=[3]))
            feats_g2_flat = feats_g2.view(B * C, -1)
            feats_a2_flat = feats_a2.view(B * C, -1)
            logits = (logits + head(feats_g2_flat, feats_a2_flat)) / 2.0

        preds = logits.argmax(dim=1)

        y_true_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(preds.detach().cpu().numpy())
        scores_list.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    scores_np = np.concatenate(scores_list, axis=0)

    out = {"y_true": y_true, "y_pred": y_pred, "scores": scores_np}
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **out)
    return out


def _choose_hard_negatives(
        block: FFHybridBlock,
        x_tokens: torch.Tensor,  # [B, N, D]
        y_true: torch.Tensor,  # [B]
        g_prev: Optional[torch.Tensor],
        num_classes: int,
        k: int
) -> torch.Tensor:
    if k <= 1:
        return incorrect_labels(y_true, num_classes)

    device = y_true.device
    B = y_true.size(0)

    # sample candidate labels [B, K], ensuring y != y_true
    cands = torch.randint(0, num_classes, (B, k), device=device)
    same = cands.eq(y_true.unsqueeze(1))
    while same.any():
        # resample where equal
        cands[same] = torch.randint(0, num_classes, (same.sum().item(),), device=device)
        same = cands.eq(y_true.unsqueeze(1))

    # flatten candidates and replicate inputs
    y_flat = cands.reshape(-1)  # [B*K]

    x_rep = x_tokens.unsqueeze(1).expand(B, k, *x_tokens.shape[1:])
    x_flat = x_rep.reshape(B * k, *x_tokens.shape[1:])  # [B*K, N, D]

    if g_prev is not None:
        g_rep = g_prev.unsqueeze(1).expand(B, k, *g_prev.shape[1:])
        g_flat = g_rep.reshape(B * k, *g_prev.shape[1:])  # [B*K, Lm, A]
    else:
        g_flat = None

    with torch.no_grad():
        # one batched forward instead of K looped forwards
        _, gv_all, _, _, _ = block(x_flat, y_flat, g_flat)  # gv_all: [B*K, A]
        scores = block.aggregate_goodness(gv_all)  # [B*K]

    scores = scores.view(B, k)  # [B, K]
    best_idx = scores.argmax(dim=1)  # [B]
    y_hard = cands[torch.arange(B, device=device), best_idx]
    return y_hard


class SAM:
    def __init__(self, params, base_optim_cls, rho=0.05, adaptive=False, **kwargs):
        params = list(filter(lambda p: p.requires_grad, params))
        if not params:
            self.base_optimizer = None
            return
        self.base_optimizer = base_optim_cls(params, **kwargs)
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.param_groups = self.base_optimizer.param_groups
        self._e = {}

    def zero_grad(self, set_to_none=True):
        if self.base_optimizer:
            self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        if not self.base_optimizer:
            return
        grad_norm = self._grad_norm()
        if grad_norm < 1e-12:
            return
        for group in self.param_groups:
            scale = self.rho / grad_norm
            for p in group["params"]:
                if p.grad is None:
                    continue
                T_w = torch.abs(p) if self.adaptive else 1.0
                e_w = T_w * p.grad * scale
                p.add_(e_w)
                self._e[p] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        if not self.base_optimizer:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p in self._e:
                    p.sub_(self._e[p])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
        self._e.clear()

    def _grad_norm(self):
        if not self.base_optimizer:
            return 0.0
        try:
            shared_device = self.param_groups[0]["params"][0].device
        except IndexError:
            return torch.tensor(0.0)
        norms = torch.stack([
            ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ])
        return torch.norm(norms, p=2)


# -------------------- Stage-1 (SFF + CFF + EMA-HNM) --------------------

def _sum_prev_goodness(net: FFHybridNet, g_hist: List[torch.Tensor]) -> torch.Tensor:
    # Locality fix: every input gv is already .detach()'d at append time, but
    # aggregate_goodness() re-enters prior blocks' learnable alpha_raw and builds
    # a fresh graph. Wrap the whole aggregation in no_grad to guarantee no
    # gradient flows through prior blocks' parameters when this sum is later
    # added into gpos_blk/gnl_blk/gni_blk (see SYMBA loss on gpos_blk).
    if not g_hist:
        return None
    with torch.no_grad():
        s = None
        for j, gv in enumerate(g_hist):
            gj = net.blocks[j].aggregate_goodness(gv)
            s = gj if s is None else (s + gj)
    return s.detach() if s is not None else None


def symba_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, alpha: float = 4.0):
    # g_pos, g_neg: [B]
    sep = g_pos - g_neg  # separation
    # scale separation and push it to be large and positive
    return F.softplus(-alpha * sep).mean()


def train_epoch_stage1(
        net: "FFHybridNet",
        net_ema: Optional["FFHybridNet"],
        opts: List["SAM"],
        loader: DataLoader,
        cfg,
        queues: List["ContrastiveQueue"],
        epoch: int,
):
    """
    Stage-1 training (Forward-Forward style) with standardized, comparable per-layer metrics.

    Returns:
        layer_avgs: List[dict] of length=num_blocks. Each dict contains:
            - fixed FAIR_LAYER_KEYS (epoch-averaged)
            - additional version-specific scalar metrics (epoch-averaged) for layer_metrics_raw.jsonl
    """
    net.train()
    if net_ema:
        net_ema.eval()

    device = cfg.device

    # ---- Fair-compare layer metric accumulators ----
    layer_sums: List[dict] = []
    for _ in range(cfg.num_blocks):
        layer_sums.append({"n": 0.0, **{k: 0.0 for k in FAIR_LAYER_KEYS}})

    # ---- MoE routing histogram accumulators (per-layer, per-epoch) ----
    # Used for reviewer-grade diagnostics (routing histograms, effective #experts, load variance).
    route_accums: List[dict] = []
    try:
        n_exp = int(getattr(cfg, "n_experts", 0))
    except Exception:
        n_exp = 0
    for _ in range(cfg.num_blocks):
        route_accums.append(
            {
                "tokens": 0.0,
                "counts": torch.zeros(n_exp, device=device) if n_exp > 0 else None,
                "importance": torch.zeros(n_exp, device=device) if n_exp > 0 else None,
            }
        )

    pbar = tqdm(loader, desc=f"Stage-1 Epoch {epoch}", leave=False, dynamic_ncols=True)

    for x1, x2, x3, labels in pbar:
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        labels = labels.to(device)
        B = x1.size(0)

        # negative-image same label (NI): pick a different image whose true label != current label
        if B > 1:
            base = torch.arange(B, device=device)
            indices_ni = torch.randint(0, B, (B,), device=device)
            bad = (indices_ni == base) | (labels[indices_ni] == labels)
            # resample a few times; with CIFAR-10 and large batches this resolves quickly
            for _ in range(10):
                if not bad.any():
                    break
                indices_ni[bad] = torch.randint(0, B, (int(bad.sum().item()),), device=device)
                bad = (indices_ni == base) | (labels[indices_ni] == labels)
            # fallback: simple shift (may include a small amount of label collisions)
            if bad.any():
                shift = torch.randint(1, B, (1,), device=device).item()
                indices_ni = (base + shift) % B
        else:
            indices_ni = torch.arange(B, device=device)
        x_ni = x1[indices_ni]

        # cached tokens and histories
        t_pos1 = t_pos2 = t_nl = t_ni = None
        g_hist_pos1: List[torch.Tensor] = []
        g_hist_pos2: List[torch.Tensor] = []
        g_hist_nl: List[torch.Tensor] = []
        g_hist_ni: List[torch.Tensor] = []

        for li, block in enumerate(net.blocks):
            opt = opts[li]
            if opt.base_optimizer is None:
                continue

            # Progressive hard-negative candidate K (per-layer)
            k_neg = int(round(
                cfg.hard_negative_k_first +
                (cfg.hard_negative_k_last - cfg.hard_negative_k_first) * (li / max(1, cfg.num_blocks - 1))
            ))

            # Use EMA teacher for HNM if requested
            hnm_block = block
            if net_ema is not None and getattr(cfg, "use_ema_hnm", False):
                hnm_block = net_ema.blocks[li]

            def forward_block(recompute_embed: bool, accum_route: bool = False):
                nonlocal t_pos1, t_pos2, t_nl, t_ni

                # get tokens for each stream
                if li == 0:
                    if recompute_embed or t_pos1 is None:
                        t_pos1_local = net.embed(x1)
                        t_pos2_local = net.embed(x2)
                        t_ni_local = net.embed(x_ni)
                        t_nl_local = net.embed(x3)
                    else:
                        t_pos1_local = t_pos1
                        t_pos2_local = t_pos2
                        t_ni_local = t_ni
                        t_nl_local = t_nl
                else:
                    t_pos1_local = t_pos1
                    t_pos2_local = t_pos2
                    t_nl_local = t_nl
                    t_ni_local = t_ni

                def get_g_prev(hist: List[torch.Tensor]):
                    g_prev = torch.stack(hist, dim=1) if hist else None
                    return g_prev.detach() if g_prev is not None else None

                g_prev_pos1 = get_g_prev(g_hist_pos1)
                g_prev_pos2 = get_g_prev(g_hist_pos2)
                g_prev_nl = get_g_prev(g_hist_nl)
                g_prev_ni = get_g_prev(g_hist_ni)

                # Optional per-epoch MoE routing histogram accumulation (top-1 counts + importance)
                def _accum_route(counts_t: Optional[torch.Tensor], imp_t: Optional[torch.Tensor]):
                    if not accum_route:
                        return
                    try:
                        ra = route_accums[li]
                        if ra.get("counts", None) is None or ra.get("importance", None) is None:
                            return
                        if counts_t is None or imp_t is None:
                            return
                        ra["counts"] += counts_t.detach().float()
                        ra["importance"] += imp_t.detach().float()
                        ra["tokens"] += float(counts_t.sum().item())
                    except Exception:
                        return

                # positives
                a_pos1, gpos1, lb_pos1, zloss_pos1, recon1 = block(t_pos1_local, labels, g_prev_pos1)
                mp_pos1 = getattr(block.moe, 'last_mean_p', None)
                ml_pos1 = getattr(block.moe, 'last_load', None)
                mc_pos1 = getattr(block.moe, 'last_counts', None)
                mi_pos1 = getattr(block.moe, 'last_importance', None)
                _accum_route(mc_pos1, mi_pos1)
                a_pos2, gpos2, lb_pos2, zloss_pos2, recon2 = block(t_pos2_local, labels, g_prev_pos2)
                mp_pos2 = getattr(block.moe, 'last_mean_p', None)
                ml_pos2 = getattr(block.moe, 'last_load', None)
                mc_pos2 = getattr(block.moe, 'last_counts', None)
                mi_pos2 = getattr(block.moe, 'last_importance', None)
                _accum_route(mc_pos2, mi_pos2)

                # wrong-label negatives (NL)
                y_nl = _choose_hard_negatives(hnm_block, t_nl_local, labels, g_prev_nl, cfg.num_classes, k_neg)
                a_nl, gnl, lb_nl, zloss_nl, _ = block(t_nl_local, y_nl, g_prev_nl)
                mp_nl = getattr(block.moe, 'last_mean_p', None)
                ml_nl = getattr(block.moe, 'last_load', None)
                mc_nl = getattr(block.moe, 'last_counts', None)
                mi_nl = getattr(block.moe, 'last_importance', None)
                _accum_route(mc_nl, mi_nl)

                # wrong-image negatives (NI)
                a_ni, gni, lb_ni, zloss_ni, _ = block(t_ni_local, labels, g_prev_ni)
                mp_ni = getattr(block.moe, 'last_mean_p', None)
                ml_ni = getattr(block.moe, 'last_load', None)
                mc_ni = getattr(block.moe, 'last_counts', None)
                mi_ni = getattr(block.moe, 'last_importance', None)
                _accum_route(mc_ni, mi_ni)

                # prev goodness (gamma) for the gated accumulation used in this script
                gamma_pos1 = _sum_prev_goodness(net, g_hist_pos1)
                gamma_nl = _sum_prev_goodness(net, g_hist_nl)
                gamma_ni = _sum_prev_goodness(net, g_hist_ni)

                zero = a_pos1.new_zeros(B)
                if gamma_pos1 is None:
                    gamma_pos1 = zero
                if gamma_nl is None:
                    gamma_nl = zero
                if gamma_ni is None:
                    gamma_ni = zero

                # ---- aspect losses ----
                thetas = block.aspect_thetas()
                loss_aspects = 0.0
                for k in range(cfg.n_aspects):
                    lp = gpos1[:, k]
                    ln_nl = gnl[:, k]
                    ln_ni = gni[:, k]

                    if cfg.aspect_loss_types[k] == "margin":
                        loss_k_nl = ff_margin_loss(lp, ln_nl, thetas[k])
                        loss_k_ni = ff_margin_loss(lp, ln_ni, thetas[k])
                    else:
                        loss_k_nl = pairwise_rank_loss(lp, ln_nl)
                        loss_k_ni = pairwise_rank_loss(lp, ln_ni)

                    loss_k = (1.0 - cfg.sff_ni_weight) * loss_k_nl + cfg.sff_ni_weight * loss_k_ni
                    loss_aspects += cfg.aspect_lambdas[k] * loss_k

                # ---- block goodness (+ gamma) ----
                gpos_curr = block.aggregate_goodness(gpos1)
                gnl_curr = block.aggregate_goodness(gnl)
                gni_curr = block.aggregate_goodness(gni)

                # ---- current-block discrimination (prevents depth free-riding) ----
                if li > 0:
                    # Residual weighting (Change 3):
                    # focus this block on examples that previous blocks have NOT yet separated well.
                    with torch.no_grad():
                        prev_margin_nl = (gamma_pos1 - gamma_nl).detach()
                        prev_margin_ni = (gamma_pos1 - gamma_ni).detach()
                        w_nl = torch.sigmoid(-cfg.symba_alpha * prev_margin_nl)
                        w_ni = torch.sigmoid(-cfg.symba_alpha * prev_margin_ni)
                        # keep mean weight ~1 so the overall scale of the loss is stable
                        w_nl = w_nl / (w_nl.mean() + 1e-6)
                        w_ni = w_ni / (w_ni.mean() + 1e-6)

                    loss_block_curr_nl_vec = F.softplus(-cfg.symba_alpha * (gpos_curr - gnl_curr))
                    loss_block_curr_ni_vec = F.softplus(-cfg.symba_alpha * (gpos_curr - gni_curr))
                    loss_block_curr_nl = (loss_block_curr_nl_vec * w_nl).mean()
                    loss_block_curr_ni = (loss_block_curr_ni_vec * w_ni).mean()
                else:
                    loss_block_curr_nl = symba_loss(gpos_curr, gnl_curr, alpha=cfg.symba_alpha)
                    loss_block_curr_ni = symba_loss(gpos_curr, gni_curr, alpha=cfg.symba_alpha)

                loss_block_curr = (
                    (1.0 - cfg.sff_ni_weight) * loss_block_curr_nl
                    + cfg.sff_ni_weight * loss_block_curr_ni
                )

                gpos_blk = gpos_curr + cfg.gamma_scale * gamma_pos1
                gnl_blk = gnl_curr + cfg.gamma_scale * gamma_nl
                gni_blk = gni_curr + cfg.gamma_scale * gamma_ni

                # ---- block discrimination (SYMBA) ----
                loss_block_nl = symba_loss(gpos_blk, gnl_blk, alpha=cfg.symba_alpha)
                loss_block_ni = symba_loss(gpos_blk, gni_blk, alpha=cfg.symba_alpha)
                loss_block = (1.0 - cfg.sff_ni_weight) * loss_block_nl + cfg.sff_ni_weight * loss_block_ni

                # ---- depth ordering ----
                loss_depth = 0.0
                if li > 0 and getattr(cfg, "depth_order_lambda", 0.0) > 0.0:
                    # [CR] P3 locality: aggregate_goodness re-enters block li-1's learnable
                    # alpha_raw. In v1 these calls ran with grad enabled, so block li's backward
                    # wrote a gradient into blocks[li-1].alpha_raw.grad; it was discarded by
                    # opt_{li-1}.zero_grad(set_to_none=True) before block li-1's next step, so
                    # v1 parameters are unaffected (analysis/verify_locality.py --depth-order
                    # checks bit-identical parameters with and without this no_grad).
                    with torch.no_grad():  # [CR] P3 locality, see note above
                        prev_pos_blk = net.blocks[li - 1].aggregate_goodness(g_hist_pos1[-1])
                    loss_depth += F.softplus(cfg.depth_margin_pos - (gpos_blk - prev_pos_blk)).mean()

                    with torch.no_grad():  # [CR] P3 locality, see note above
                        prev_nl_blk = net.blocks[li - 1].aggregate_goodness(g_hist_nl[-1])
                    loss_depth += F.softplus(cfg.depth_margin_neg - (prev_nl_blk - gnl_blk)).mean()

                    with torch.no_grad():  # [CR] P3 locality, see note above
                        prev_ni_blk = net.blocks[li - 1].aggregate_goodness(g_hist_ni[-1])
                    loss_depth += F.softplus(cfg.depth_margin_neg - (prev_ni_blk - gni_blk)).mean()

                    loss_depth *= cfg.depth_order_lambda

                # ---- MoE aux losses ----
                loss_bal = (lb_pos1 + lb_pos2 + lb_nl + lb_ni) / 4.0
                loss_z = (zloss_pos1 + zloss_pos2 + zloss_nl + zloss_ni) / 4.0

                # ---- MoE routing diagnostics (reviewer-grade collapse checks) ----
                moe_eff_experts_imp = None
                moe_eff_experts_load = None
                moe_entropy_imp = None
                moe_entropy_load = None
                moe_load_var = None
                moe_imp_var = None
                moe_top1prob = None
                moe_top1load = None
                try:
                    # Importance distribution P_i = fraction of router probability mass allocated to expert i.
                    moe_ps = [p for p in (mp_pos1, mp_pos2, mp_nl, mp_ni) if p is not None]
                    if len(moe_ps) > 0:
                        mean_p = torch.stack([p.to(device) for p in moe_ps], dim=0).mean(dim=0)  # [E]
                        mean_p = mean_p / (mean_p.sum() + 1e-12)
                        moe_eff_experts_imp = 1.0 / (mean_p.pow(2).sum() + 1e-12)
                        moe_entropy_imp = -(mean_p * (mean_p + 1e-12).log()).sum()
                        moe_imp_var = mean_p.var(unbiased=False)
                        moe_top1prob = mean_p.max()

                    # Load distribution f_i = fraction of tokens dispatched to expert i (top-1).
                    moe_ls = [l for l in (ml_pos1, ml_pos2, ml_nl, ml_ni) if l is not None]
                    if len(moe_ls) > 0:
                        mean_l = torch.stack([l.to(device) for l in moe_ls], dim=0).mean(dim=0)  # [E]
                        mean_l = mean_l / (mean_l.sum() + 1e-12)
                        moe_eff_experts_load = 1.0 / (mean_l.pow(2).sum() + 1e-12)
                        moe_entropy_load = -(mean_l * (mean_l + 1e-12).log()).sum()
                        moe_load_var = mean_l.var(unbiased=False)
                        moe_top1load = mean_l.max()
                except Exception:
                    pass

                # ---- reconstruction ----
                loss_recon = 0.0
                if getattr(cfg, "recon_lambda", 0.0) > 0.0 and recon1 is not None and recon2 is not None:
                    loss_recon += F.mse_loss(recon1, t_pos1_local.detach())
                    loss_recon += F.mse_loss(recon2, t_pos2_local.detach())
                    loss_recon *= cfg.recon_lambda

                # ---- contrastive ----
                z1 = block.contrastive_embed(a_pos1)
                z2 = block.contrastive_embed(a_pos2)
                z_all = torch.cat([z1, z2], dim=0)
                y_all = torch.cat([labels, labels], dim=0)

                if getattr(cfg, "queue_neg_samples", 0) > 0:
                    zq, yq = queues[li].sample(cfg.queue_neg_samples)
                    if zq is not None:
                        z_all = torch.cat([z_all, zq.detach()], dim=0)
                        y_all = torch.cat([y_all, yq.detach()], dim=0)

                loss_con = supcon_generic(z_all, y_all, temperature=cfg.contrastive_temp)

                # ---- total ----
                # MoE aux-loss weighting (optional ramp-up + optional depth scaling)
                ramp = 1.0
                try:
                    ramp_epochs = int(getattr(cfg, "moe_aux_ramp_epochs", 0))
                    if ramp_epochs > 0:
                        ramp = min(1.0, float(epoch) / float(max(1, ramp_epochs)))
                except Exception:
                    ramp = 1.0

                depth_scale = 1.0
                try:
                    depth_k = float(getattr(cfg, "moe_aux_depth_scale", 0.0))
                    if depth_k != 0.0:
                        depth_scale = 1.0 + depth_k * (li / max(1, cfg.num_blocks - 1))
                except Exception:
                    depth_scale = 1.0

                moe_bal_w = float(getattr(cfg, "moe_balance_coef", 0.0)) * ramp * depth_scale
                moe_z_w = float(getattr(cfg, "moe_zloss_coef", 0.0)) * ramp * depth_scale

                # CP-FAIR: Depth-scheduled block_curr_lambda (stronger pressure at deeper blocks).
                # eff = λ * (1 + slope * li/(L-1))  →  [0.25, 0.5, 0.75, 1.0] for slope=3.0, λ=0.25
                block_curr_lambda_eff = float(cfg.block_curr_lambda)
                try:
                    slope = float(getattr(cfg, "block_curr_depth_slope", 0.0))
                    if slope != 0.0:
                        block_curr_lambda_eff = float(cfg.block_curr_lambda) * (
                            1.0 + slope * (li / max(1, cfg.num_blocks - 1))
                        )
                except Exception:
                    block_curr_lambda_eff = float(cfg.block_curr_lambda)

                loss = (
                        loss_aspects
                        + cfg.block_lambda * loss_block
                        + block_curr_lambda_eff * loss_block_curr
                        + loss_depth
                        + moe_bal_w * loss_bal
                        + moe_z_w * loss_z
                        + cfg.contrastive_lambda * loss_con
                        + loss_recon
                )

                cached_outputs = (a_pos1, a_pos2, a_nl, a_ni,
                                  gpos1, gpos2, gnl, gni,
                                  z1, z2)

                # ---- standardized + raw metrics ----
                metrics = {
                    "loss_total": loss.detach(),
                    "loss_aspects": loss_aspects.detach() if hasattr(loss_aspects, "detach") else loss.new_tensor(
                        float(loss_aspects)),
                    "loss_block": loss_block.detach() if hasattr(loss_block, "detach") else loss.new_tensor(
                        float(loss_block)),
                    "loss_block_curr": loss_block_curr.detach() if hasattr(loss_block_curr,
                                                                           "detach") else loss.new_tensor(
                        float(loss_block_curr)),
                    "loss_inc": loss_block_curr.detach() if hasattr(loss_block_curr, "detach") else loss.new_tensor(
                        float(loss_block_curr)),
                    "loss_depth": loss_depth.detach() if hasattr(loss_depth, "detach") else loss.new_tensor(0.0),
                    "loss_con": loss_con.detach() if hasattr(loss_con, "detach") else loss.new_tensor(0.0),
                    "loss_recon": loss_recon.detach() if hasattr(loss_recon, "detach") else loss.new_tensor(0.0),
                    "loss_div": loss.new_tensor(0.0),
                    "loss_bal": loss_bal.detach() if hasattr(loss_bal, "detach") else loss.new_tensor(0.0),
                    "loss_z": loss_z.detach() if hasattr(loss_z, "detach") else loss.new_tensor(0.0),

                    "moe_bal_w": loss.new_tensor(float(moe_bal_w)),
                    "moe_z_w": loss.new_tensor(float(moe_z_w)),

                    # separation scores
                    "sep_nl": (gpos_blk > gnl_blk).float().mean(),
                    "sep_ni": (gpos_blk > gni_blk).float().mean(),
                    "sep_curr_nl": (gpos_curr - gnl_curr).mean(),
                    "sep_curr_ni": (gpos_curr - gni_curr).mean(),
                    # incremental (per-block) margin contributions (means + stds)
                    "inc_margin_nl": (gpos_curr - gnl_curr).mean(),
                    "inc_margin_ni": (gpos_curr - gni_curr).mean(),
                    "inc_margin_nl_std": (gpos_curr - gnl_curr).std(unbiased=False),
                    "inc_margin_ni_std": (gpos_curr - gni_curr).std(unbiased=False),
                    # cumulative margin at this block
                    "margin_blk_nl": (gpos_blk - gnl_blk).mean(),
                    "margin_blk_ni": (gpos_blk - gni_blk).mean(),

                    # goodness values (means)
                    "gpos": gpos_blk.mean(),
                    "gnl": gnl_blk.mean(),
                    "gni": gni_blk.mean(),

                    "gpos_cur": gpos_curr.mean(),
                    "gnl_cur": gnl_curr.mean(),
                    "gni_cur": gni_curr.mean(),

                    "gpos_base": gpos_curr.mean(),
                    "gnl_base": gnl_curr.mean(),
                    "gni_base": gni_curr.mean(),

                    "gamma_pos": gamma_pos1.mean(),
                    "gamma_nl": gamma_nl.mean(),
                    "gamma_ni": gamma_ni.mean(),

                    "k_neg": loss.new_tensor(float(k_neg)),
                }

                # ---- Extra (version-specific) scalar metrics for layer_metrics_raw.jsonl ----
                try:
                    metrics["theta_block"] = block.block_theta().detach()
                except Exception:
                    pass

                # Router temperature (may be float)
                try:
                    metrics["moe_tau"] = loss.new_tensor(float(getattr(block.moe, "tau", 0.0)))
                except Exception:
                    pass

                # Per-aspect goodness means + thetas (helps compare 'what each layer used')
                for k in range(cfg.n_aspects):
                    metrics[f"gpos_a{k}"] = gpos1[:, k].mean()
                    metrics[f"gnl_a{k}"] = gnl[:, k].mean()
                    metrics[f"gni_a{k}"] = gni[:, k].mean()
                    try:
                        metrics[f"theta_a{k}"] = thetas[k].detach()
                    except Exception:
                        metrics[f"theta_a{k}"] = thetas[k]

                # Aggregation weights (alpha) as scalars per aspect
                try:
                    w = F.softmax(block.alpha_raw, dim=0) if block.use_softmax_weights else F.softplus(block.alpha_raw)
                    for k in range(cfg.n_aspects):
                        metrics[f"alpha_w{k}"] = w[k].detach()
                except Exception:
                    pass

                # MoE diagnostics (routing utilization)
                if moe_eff_experts_imp is not None:
                    metrics["moe_eff_experts_imp"] = moe_eff_experts_imp.detach() if hasattr(moe_eff_experts_imp,
                                                                                             "detach") else float(
                        moe_eff_experts_imp)
                    # Backwards-compatible alias
                    metrics["moe_eff_experts"] = metrics["moe_eff_experts_imp"]
                if moe_eff_experts_load is not None:
                    metrics["moe_eff_experts_load"] = moe_eff_experts_load.detach() if hasattr(moe_eff_experts_load,
                                                                                               "detach") else float(
                        moe_eff_experts_load)

                if moe_entropy_imp is not None:
                    metrics["moe_entropy_imp"] = moe_entropy_imp.detach() if hasattr(moe_entropy_imp,
                                                                                     "detach") else float(
                        moe_entropy_imp)
                    metrics["moe_entropy"] = metrics["moe_entropy_imp"]
                if moe_entropy_load is not None:
                    metrics["moe_entropy_load"] = moe_entropy_load.detach() if hasattr(moe_entropy_load,
                                                                                       "detach") else float(
                        moe_entropy_load)

                if moe_imp_var is not None:
                    metrics["moe_importance_var"] = moe_imp_var.detach() if hasattr(moe_imp_var, "detach") else float(
                        moe_imp_var)
                if moe_load_var is not None:
                    metrics["moe_load_var"] = moe_load_var.detach() if hasattr(moe_load_var, "detach") else float(
                        moe_load_var)

                if moe_top1prob is not None:
                    metrics["moe_top1prob"] = moe_top1prob.detach() if hasattr(moe_top1prob, "detach") else float(
                        moe_top1prob)
                if moe_top1load is not None:
                    metrics["moe_top1load"] = moe_top1load.detach() if hasattr(moe_top1load, "detach") else float(
                        moe_top1load)

                return loss, cached_outputs, metrics

            opt.zero_grad(set_to_none=True)
            recompute_flag = (li == 0)

            loss1, cached, metrics1 = forward_block(recompute_embed=recompute_flag, accum_route=True)
            loss1.backward()

            if cfg.use_sam:
                opt.first_step(zero_grad=True)

                # second SAM step (recompute forward)
                loss2, _, _ = forward_block(recompute_embed=recompute_flag, accum_route=False)
                loss2.backward()
                opt.second_step(zero_grad=True)
            else:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in opt.param_groups for p in g["params"]], cfg.grad_clip
                    )
                opt.base_optimizer.step()
                opt.zero_grad(set_to_none=True)
                loss2 = loss1  # for logging

            if li == (cfg.num_blocks - 1):
                pbar.set_postfix(L1=f"{loss1.item():.3f}", L2=f"{loss2.item():.3f}")

            if net_ema is not None:
                update_ema(block, net_ema.blocks[li], cfg.ema_decay)
                if li == 0:
                    update_ema(net.embed, net_ema.embed, cfg.ema_decay)

            # accumulate epoch stats (use first SAM pass metrics)
            s = layer_sums[li]
            s["n"] += float(B)
            m1 = pad_layer_metrics(metrics1)
            for k in FAIR_LAYER_KEYS:
                s[k] += m1[k] * float(B)

            # also accumulate any extra scalar keys (for raw logs)
            for k, v in metrics1.items():
                if k in FAIR_LAYER_KEYS or k == "n":
                    continue
                s[k] = s.get(k, 0.0) + _as_float(v, 0.0) * float(B)

            a_pos1, a_pos2, a_nl, a_ni, gpos1, gpos2, gnl, gni, z1, z2 = cached

            # cache tokens for next block
            t_pos1 = l2_normalize_tokens(a_pos1.detach())
            t_pos2 = l2_normalize_tokens(a_pos2.detach())
            t_nl = l2_normalize_tokens(a_nl.detach())
            t_ni = l2_normalize_tokens(a_ni.detach())

            # record goodness history
            g_hist_pos1.append(gpos1.detach())
            g_hist_pos2.append(gpos2.detach())
            g_hist_nl.append(gnl.detach())
            g_hist_ni.append(gni.detach())

            # update contrastive queues
            with torch.no_grad():
                queues[li].enqueue(z1.detach(), labels.detach())
                queues[li].enqueue(z2.detach(), labels.detach())

    # return epoch averages per layer
    layer_avgs: List[dict] = []
    for li in range(cfg.num_blocks):
        s = layer_sums[li]
        n = max(1.0, float(s.get("n", 0.0)))
        out = {k: (s.get(k, 0.0) / n) for k in FAIR_LAYER_KEYS}
        for k, v in s.items():
            if k in FAIR_LAYER_KEYS or k == "n":
                continue
            out[k] = v / n

        # ---- Per-epoch MoE routing histograms (top-1) + derived collapse stats ----
        # These are logged only in layer_metrics_raw.jsonl (lists are not part of FAIR_LAYER_KEYS).
        try:
            ra = route_accums[li] if ("route_accums" in locals() or "route_accums" in globals()) else None
            if ra is not None and ra.get("tokens", 0.0) > 0 and ra.get("counts", None) is not None and ra.get(
                    "importance", None) is not None:
                tok = float(ra["tokens"])
                counts = ra["counts"].detach().float()
                imp = ra["importance"].detach().float()

                load = (counts / max(1.0, tok)).clamp_min(0.0)
                imp_d = (imp / max(1.0, tok)).clamp_min(0.0)
                load = load / (load.sum() + 1e-12)
                imp_d = imp_d / (imp_d.sum() + 1e-12)

                out["moe_tokens"] = tok
                out["moe_load_counts"] = [float(x) for x in counts.cpu().tolist()]
                out["moe_load_hist"] = [float(x) for x in load.cpu().tolist()]
                out["moe_importance_hist"] = [float(x) for x in imp_d.cpu().tolist()]

                # Effective number of experts used (Simpson inverse): 1 / sum p_i^2
                out["moe_eff_experts_load_epoch"] = float((1.0 / (load.pow(2).sum() + 1e-12)).item())
                out["moe_eff_experts_imp_epoch"] = float((1.0 / (imp_d.pow(2).sum() + 1e-12)).item())

                out["moe_load_var_epoch"] = float(load.var(unbiased=False).item())
                out["moe_importance_var_epoch"] = float(imp_d.var(unbiased=False).item())

                out["moe_load_max_epoch"] = float(load.max().item())
                out["moe_load_min_epoch"] = float(load.min().item())
                out["moe_importance_max_epoch"] = float(imp_d.max().item())
                out["moe_importance_min_epoch"] = float(imp_d.min().item())

                # Coefficient of variation (std/mean). Mean is 1/E under perfect balance.
                out["moe_load_cv_epoch"] = float((load.std(unbiased=False) / (load.mean() + 1e-12)).item())
                out["moe_importance_cv_epoch"] = float((imp_d.std(unbiased=False) / (imp_d.mean() + 1e-12)).item())
        except Exception:
            pass

        layer_avgs.append(out)
    return layer_avgs


# -------------------- scheduler --------------------

def cosine_warmup_lambda(epoch: int, total_epochs: int, warmup: int) -> float:
    if epoch < warmup:
        return float(epoch + 1) / float(max(1, warmup))
    progress = (epoch - warmup) / float(max(1, total_epochs - warmup))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# -------------------- config --------------------

@dataclass
class FFConfig:
    batch_size: int = 512
    test_batch_size: int = 256
    epochs_stage1: int = 362
    epochs_stage2: int = 20  # CP-FAIR: consistent with revised_valfix, topk_depthsched, freez variants
    seed: int = 42
    val_fraction: float = 0.1
    save_predictions: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 4
    symba_alpha: float = 4.0
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    use_sam: bool = True
    sam_rho: float = 0.1
    sam_adaptive: bool = True
    use_mem: bool = False  # (do-this-first) disable memory cross-attention
    warmup_epochs: int = 15

    num_classes: int = 10  # CIFAR-10
    img_size: int = 32
    patch_size: int = 2
    d_model: int = 256
    num_blocks: int = 4
    num_heads: int = 8
    n_experts: int = 32
    mlp_mult: int = 4
    dropout: float = 0.15
    proj_dim: int = 256
    label_embed_dim: int = 64
    moe_top_k: int = 4

    use_rope: bool = True
    rope_theta: float = 10000.0

    use_conv_stem: bool = True
    stem_channels: int = 384
    drop_path_rate: float = 0.25

    n_aspects: int = 4
    aspect_loss_types: List[str] = field(default_factory=list)
    aspect_lambdas: List[float] = field(default_factory=list)
    aspect_theta_inits: List[float] = field(default_factory=list)
    block_lambda: float = 1.0
    block_curr_lambda: float = 0.25  # CP-FAIR: base at L0; schedule ramps to 1.0 at L3
    block_curr_depth_slope: float = 3.0  # CP-FAIR: eff = λ*(1 + slope*li/(L-1)) → [0.25, 0.5, 0.75, 1.0]
    block_theta_init: float = 5.0
    use_softmax_weights: float = True
    moe_balance_coef: float = 0.01
    moe_zloss_coef: float = 1e-3

    # MoE aux-loss variants (reviewer-friendly, aligns with Switch / ST-MoE)
    moe_balance_type: str = "switch"  # {"switch","entropy"}
    moe_zloss_type: str = "stmoe"  # {"stmoe","l2"}
    # Optional ramping / depth-scaling for aux losses (set 0 to disable)
    moe_aux_ramp_epochs: int = 0
    moe_aux_depth_scale: float = 0.0

    proto_temp: float = 0.1
    sff_ni_weight: float = 0.5

    # For 10 classes we don't really need 40-60 candidates, but
    # I keep your schedule; you can shrink to e.g. 8->16 if desired
    hard_negative_k_first: int = 8
    hard_negative_k_last: int = 16
    queue_size: int = 16384
    queue_neg_samples: int = 4096

    gamma_scale: float = 0.7
    depth_order_lambda: float = 0.2   # CP-FAIR: doubled to enforce monotonic g_nl (was 0.1)
    depth_margin_pos: float = 0.05
    depth_margin_neg: float = 0.1    # CP-FAIR: doubled — DP-FAIR showed 0.05 too weak for g_nl ordering

    recon_lambda: float = 0.05
    recon_mult: int = 2

    gate_tau_start: float = 2.0
    gate_tau_end: float = 0.5
    gate_tau_depth_slope: float = 0.2

    use_ema: bool = True
    ema_decay: float = 0.999
    use_ema_hnm: bool = True
    eval_tta: bool = True
    head_hidden: int = 1536
    head_dropout: float = 0.1
    stage2_lr: float = 1e-3
    stage2_wd: float = 0
    label_smoothing: float = 0
    head_num_heads: int = 4

    use_mixup_cutmix: bool = False
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.6
    mixup_switch_prob: float = 0.5

    contrastive_temp: float = 0.2
    contrastive_lambda: float = 0.5  # (set something sensible if not in your original)

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.aspect_loss_types = ["margin", "rank", "rank", "margin"]
        self.aspect_lambdas = [1.5, 0.2, 0.3, 0.3]
        self.aspect_theta_inits = [5.0, 2.5, 0.5, 1.0]
        self.block_theta_init = 6.0

        self.aspect_loss_types = self.aspect_loss_types[:self.n_aspects]
        self.aspect_lambdas = self.aspect_lambdas[:self.n_aspects]
        self.aspect_theta_inits = self.aspect_theta_inits[:self.n_aspects]


# -------------------- build & main --------------------


def main():
    cfg = FFConfig()
    cfg.__post_init__()

    # Optional override from env var so you can do:
    #   FF_NUM_BLOCKS=8 FF_D_MODEL=128 FF_USE_SAM=0 python script.py
    try:
        cfg.num_blocks = int(os.environ.get("FF_NUM_BLOCKS", cfg.num_blocks))
    except Exception:
        pass
    try:
        cfg.d_model = int(os.environ.get("FF_D_MODEL", cfg.d_model))
    except Exception:
        pass
    try:
        cfg.num_heads = int(os.environ.get("FF_NUM_HEADS", cfg.num_heads))
    except Exception:
        pass
    _sam_env = os.environ.get("FF_USE_SAM", "").lower()
    if _sam_env in ("0", "false", "no"):
        cfg.use_sam = False
    elif _sam_env in ("1", "true", "yes"):
        cfg.use_sam = True

    try:
        cfg.epochs_stage1 = int(os.environ.get("FF_EPOCHS_STAGE1", cfg.epochs_stage1))
    except Exception:
        pass
    try:
        cfg.seed = int(os.environ.get("FF_SEED", cfg.seed))
    except Exception:
        pass

    assert cfg.d_model % cfg.num_heads == 0, \
        f"d_model={cfg.d_model} not divisible by num_heads={cfg.num_heads}"

    # ---- Factorial ablation env-var overrides ----
    # Usage: FF_GAMMA_SCALE=0 FF_BLOCK_CURR_LAMBDA=0 FF_DEPTH_ORDER_LAMBDA=0 python script.py
    for env_key, attr in [
        ("FF_GAMMA_SCALE", "gamma_scale"),
        ("FF_BLOCK_CURR_LAMBDA", "block_curr_lambda"),
        ("FF_DEPTH_ORDER_LAMBDA", "depth_order_lambda"),
        ("FF_CONTRASTIVE_LAMBDA", "contrastive_lambda"),
    ]:
        if env_key in os.environ:
            try:
                setattr(cfg, attr, float(os.environ[env_key]))
            except Exception:
                pass

    # Integer ablation overrides (HNM, MoE, aspects)
    for env_key, attr in [
        ("FF_K_FIRST", "hard_negative_k_first"),
        ("FF_K_LAST", "hard_negative_k_last"),
        ("FF_N_EXPERTS", "n_experts"),
        ("FF_N_ASPECTS", "n_aspects"),
    ]:
        if env_key in os.environ:
            try:
                setattr(cfg, attr, int(os.environ[env_key]))
            except Exception:
                pass

    # Determinism for fair comparison
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = cfg.device
    print(f"Using device: {device}")
    print(f"CIFAR-10 Hybrid Scaled Config: D={cfg.d_model}, L={cfg.num_blocks}, "
          f"H={cfg.num_heads}, P={cfg.patch_size}, RoPE={cfg.use_rope}, ASAM={cfg.sam_adaptive}")
    print(f"Key Features: SYMBA, DepthOrder, SupCon, EMA={cfg.use_ema}, EMA-HNM={cfg.use_ema_hnm}, Attentive Head")
    print(f"HNM K schedule (per-layer): {cfg.hard_negative_k_first} -> {cfg.hard_negative_k_last}")
    print(f"Version tag: {VERSION_TAG}")

    # ---- Per-run checkpoint file names (avoid overwriting across versions/runs) ----
    global STAGE1_LAST_PATH, STAGE1_BEST_PATH, STAGE2_LAST_PATH, STAGE2_BEST_PATH
    run_name = f"ff_cifar10_{VERSION_TAG}_L{cfg.num_blocks}_D{cfg.d_model}_P{cfg.patch_size}_seed{cfg.seed}"
    STAGE1_LAST_PATH = os.path.join(CKPT_DIR, f"{run_name}_stage1_last.pt")
    STAGE1_BEST_PATH = os.path.join(CKPT_DIR, f"{run_name}_stage1_best.pt")
    STAGE2_LAST_PATH = os.path.join(CKPT_DIR, f"{run_name}_stage2_last.pt")
    STAGE2_BEST_PATH = os.path.join(CKPT_DIR, f"{run_name}_stage2_best.pt")

    print("Checkpoint paths:")
    print(f"  Stage-1 last: {STAGE1_LAST_PATH}")
    print(f"  Stage-1 best: {STAGE1_BEST_PATH}")
    print(f"  Stage-2 last: {STAGE2_LAST_PATH}")
    print(f"  Stage-2 best: {STAGE2_BEST_PATH}")

    # ---- Logger ----
    run_dir = make_run_dir(cfg)
    logger = JSONLLogger(run_dir)
    logger.log_config(asdict(cfg), collect_env_info())
    logger.log_event("run_start", {"version": VERSION_TAG, "ckpt_dir": CKPT_DIR})
    print(f"[FairCompare] Logging to: {run_dir}")

    try:
        # Data (deterministic train/val split from CIFAR-10 train split; no test peeking)
        train_loader_s1, val_loader, test_loader, split_info = make_loaders_stage1(
            cfg.batch_size,
            cfg.test_batch_size,
            seed=cfg.seed,
            val_fraction=cfg.val_fraction,
            workers=cfg.workers,
        )
        train_indices = split_info["train_idx"]
        val_indices = split_info["val_idx"]

        # Save split indices for reproducibility (written into the run directory)
        split_path = os.path.join(run_dir, "split_indices.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, sort_keys=True)
        logger.log_event(
            "split_indices_saved",
            {"path": split_path, "n_train": int(len(train_indices)), "n_val": int(len(val_indices))},
        )

        # Model
        net = FFHybridNet(cfg).to(device)
        net_ema = deepcopy(net).to(device) if cfg.use_ema else None
        if net_ema is not None:
            for p in net_ema.parameters():
                p.requires_grad_(False)

        def router_tau_at(epoch: int) -> float:
            t = min(1.0, max(0.0, epoch / max(1, cfg.epochs_stage1 - 1)))
            return cfg.gate_tau_start + t * (cfg.gate_tau_end - cfg.gate_tau_start)

        # Optimizers and schedulers
        opts: List[SAM] = []
        scheds: List[Optional[torch.optim.lr_scheduler._LRScheduler]] = []
        for li, block in enumerate(net.blocks):
            params = list(block.parameters())
            if li == 0:
                params += list(net.embed.parameters())

            opt = SAM(params, torch.optim.AdamW, rho=cfg.sam_rho, adaptive=cfg.sam_adaptive,
                      lr=cfg.lr, weight_decay=cfg.weight_decay)

            opts.append(opt)
            if opt.base_optimizer is None:
                scheds.append(None)
            else:
                sched = torch.optim.lr_scheduler.LambdaLR(
                    opt.base_optimizer,
                    lr_lambda=lambda e, total=cfg.epochs_stage1, w=cfg.warmup_epochs:
                    cosine_warmup_lambda(e, total, w)
                )
                scheds.append(sched)

        queues = [ContrastiveQueue(cfg.queue_size, cfg.proj_dim, device=device) for _ in range(cfg.num_blocks)]

        # ========== LOAD CHECKPOINT (Stage-2 > Stage-1 best > Stage-1 last) ==========
        ckpt = None
        best_s1 = 0.0
        start_epoch_stage1 = 1

        if PREFER_STAGE2_RESUME and os.path.exists(STAGE2_LAST_PATH):
            ckpt, start_epoch_stage1, best_s1 = _load_checkpoint_for_stage1(
                cfg, net, net_ema, opts, scheds, queues, path=STAGE2_LAST_PATH
            )
        else:
            stage1_load_path = (
                STAGE1_BEST_PATH
                if (RESUME_STAGE1_FROM_BEST and os.path.exists(STAGE1_BEST_PATH))
                else STAGE1_LAST_PATH
            )
            ckpt, start_epoch_stage1, best_s1 = _load_checkpoint_for_stage1(
                cfg, net, net_ema, opts, scheds, queues, path=stage1_load_path
            )

        # [CR] record resumes in events.jsonl (a resumed run writes a new run directory)
        if ckpt is not None:
            _ck_stage = int(ckpt.get("stage", 1))
            logger.log_event("resume", {
                "ckpt_stage": _ck_stage,
                "epoch_stage1": int(ckpt.get("epoch_stage1", 0)),
                "epoch_stage2": int(ckpt.get("epoch_stage2", 0)),
                "best_s1": float(best_s1),
                "rng_restored": (bool(LAST_RNG_RESTORE.get("restored", False))
                                 if _ck_stage < 2 else "deferred_to_stage2"),
            })

        # ==================== STAGE 1 TRAINING ====================
        print("Starting Stage-1 Training...")

        EVAL_FREQ = 20
        sep_log_every = int(getattr(cfg, "sep_log_every", 20))

        if start_epoch_stage1 <= cfg.epochs_stage1:
            pbar_s1_epochs = tqdm(range(start_epoch_stage1, cfg.epochs_stage1 + 1),
                                  desc="Stage-1 Epochs", dynamic_ncols=True)
            for epoch in pbar_s1_epochs:
                eval_accs = None  # filled when evaluation runs
                depth_accs = None  # filled when evaluation runs

                current_tau = router_tau_at(epoch - 1)
                net.set_router_tau(current_tau, depth_slope=cfg.gate_tau_depth_slope)

                layer_stats = train_epoch_stage1(net, net_ema, opts, train_loader_s1, cfg, queues, epoch=epoch)

                # Log per-layer metrics (both fixed-key and raw)
                for li, lm in enumerate(layer_stats):
                    logger.log_layer(stage=1, epoch=epoch, layer_idx=li, metrics=lm)
                    logger.log_layer_raw(stage=1, epoch=epoch, layer_idx=li, metrics=lm)

                # Epoch-level MoE routing summary (mean across layers) for quick plotting
                moe_summary = {}
                try:
                    v_eff_load = [lm.get("moe_eff_experts_load_epoch", None) for lm in layer_stats]
                    v_eff_load = [float(v) for v in v_eff_load if v is not None]
                    v_eff_imp = [lm.get("moe_eff_experts_imp_epoch", None) for lm in layer_stats]
                    v_eff_imp = [float(v) for v in v_eff_imp if v is not None]

                    v_load_var = [lm.get("moe_load_var_epoch", None) for lm in layer_stats]
                    v_load_var = [float(v) for v in v_load_var if v is not None]
                    v_imp_var = [lm.get("moe_importance_var_epoch", None) for lm in layer_stats]
                    v_imp_var = [float(v) for v in v_imp_var if v is not None]

                    v_load_max = [lm.get("moe_load_max_epoch", None) for lm in layer_stats]
                    v_load_max = [float(v) for v in v_load_max if v is not None]

                    if len(v_eff_load) > 0:
                        moe_summary["moe_eff_experts_load_mean"] = float(np.mean(v_eff_load))
                    if len(v_eff_imp) > 0:
                        moe_summary["moe_eff_experts_imp_mean"] = float(np.mean(v_eff_imp))
                    if len(v_load_var) > 0:
                        moe_summary["moe_load_var_mean"] = float(np.mean(v_load_var))
                    if len(v_imp_var) > 0:
                        moe_summary["moe_importance_var_mean"] = float(np.mean(v_imp_var))
                    if len(v_load_max) > 0:
                        moe_summary["moe_load_max_mean"] = float(np.mean(v_load_max))
                except Exception:
                    moe_summary = {}

                # Scheduler step
                current_lr = cfg.lr
                for sch in scheds:
                    if sch:
                        sch.step()
                        current_lr = sch.get_last_lr()[0]

                pbar_s1_epochs.set_postfix(lr=f"{current_lr:.2e}", tau=f"{current_tau:.2f}")

                # Optional per-block separation print (cheap, no eval)
                if (epoch % sep_log_every == 0) or (epoch == cfg.epochs_stage1):
                    sep_str = " | ".join(
                        [
                            f"L{i} sepNL={layer_stats[i].get('sep_nl', 0.0) * 100:.1f}% sepNI={layer_stats[i].get('sep_ni', 0.0) * 100:.1f}%"
                            for i in range(cfg.num_blocks)]
                    )
                    pbar_s1_epochs.write(f"[Stage-1] Epoch {epoch:03d} sep: {sep_str}")

                eval_model = net_ema if (net_ema is not None) else net

                is_new_best_s1 = False
                TOPKS = tuple(range(1, 11))

                if epoch % EVAL_FREQ == 0 or epoch == cfg.epochs_stage1:
                    print(f"\n[Epoch {epoch}] Starting Stage-1 evaluation on VAL (FF agg)...")

                    # Single-pass evaluation: get Top-K accuracies at every truncation depth.
                    depth_accs = evaluate_ff_by_depth(
                        eval_model,
                        val_loader,
                        device,
                        tta=cfg.eval_tta,
                        depths=tuple(range(1, cfg.num_blocks + 1)),
                        topk=TOPKS,
                    )
                    # Full-depth accuracy is just the last depth entry.
                    eval_accs = depth_accs[cfg.num_blocks]

                    if eval_accs[1] > best_s1:
                        best_s1 = eval_accs[1]
                        is_new_best_s1 = True

                    topk_str = " | ".join([f"Top{k}: {eval_accs[k] * 100:.2f}%" for k in TOPKS])
                    pbar_s1_epochs.write(
                        f"[Stage-1] Epoch {epoch:03d}/{cfg.epochs_stage1} | {topk_str} | Best@1: {best_s1 * 100:.2f}%"
                    )
                    if depth_accs is not None:
                        depth_str = " | ".join(
                            [f"D{d}: {depth_accs[d][1] * 100:.2f}%" for d in range(1, cfg.num_blocks + 1)]
                        )
                        pbar_s1_epochs.write(f"[Stage-1] Depth Top1 | {depth_str}")

                    pbar_s1_epochs.set_description(f"Stage-1 Epochs (Best@1: {best_s1 * 100:.2f}%)")

                # Per-epoch logging
                logger.log_epoch(stage=1, epoch=epoch, metrics={
                    "lr": current_lr,
                    "tau": current_tau,
                    "best_s1": best_s1,
                    "eval_top1": (eval_accs[1] if eval_accs is not None else None),
                    "eval_top5": (eval_accs[5] if (eval_accs is not None and 5 in eval_accs) else None),
                    **{f"eval_top1_d{d}": (depth_accs[d][1] if depth_accs is not None else None) for d in
                       range(1, cfg.num_blocks + 1)},
                    **moe_summary,
                })

                # Always save Stage-1 LAST checkpoint
                _save_checkpoint(
                    stage=1,
                    cfg=cfg,
                    epoch_stage1=epoch,
                    best_s1=best_s1,
                    net=net,
                    net_ema=net_ema,
                    opts=opts,
                    scheds=scheds,
                    queues=queues,
                    path=STAGE1_LAST_PATH,
                )

                # Save Stage-1 BEST checkpoint only when improved
                if is_new_best_s1:
                    _save_checkpoint(
                        stage=1,
                        cfg=cfg,
                        epoch_stage1=epoch,
                        best_s1=best_s1,
                        net=net,
                        net_ema=net_ema,
                        opts=opts,
                        scheds=scheds,
                        queues=queues,
                        path=STAGE1_BEST_PATH,
                    )
        else:
            print("Skipping Stage-1 training (already completed in checkpoint).")

        # From here on we assume Stage‑1 done
        epoch_stage1_final = cfg.epochs_stage1

        # ==================== STAGE 2 TRAINING ====================
        print("\nStarting Stage-2 Training (Attentive Hybrid Head)...")

        s2_batch_size = cfg.batch_size * 2
        train_loader_s2 = make_loader_stage2(s2_batch_size, train_indices=train_indices, workers=cfg.workers)

        # If we're NOT resuming an existing Stage-2 run, start Stage-2 from BEST Stage-1 backbone
        resuming_stage2 = (ckpt is not None and int(ckpt.get("stage", 1)) >= 2)
        if USE_BEST_STAGE1_FOR_STAGE2 and (not resuming_stage2) and os.path.exists(STAGE1_BEST_PATH):
            _load_backbone_only(cfg, net, net_ema, STAGE1_BEST_PATH)

        backbone = net_ema if (net_ema is not None) else net
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()

        head_in_dim_g = cfg.num_blocks * cfg.n_aspects
        head_in_dim_a = cfg.d_model * cfg.num_blocks

        head = AttentiveHybridHead(
            in_dim_g=head_in_dim_g,
            in_dim_a=head_in_dim_a,
            hidden=cfg.head_hidden,
            num_classes=cfg.num_classes,
            num_heads=cfg.head_num_heads,
            dropout=cfg.head_dropout
        ).to(device)

        opt_head = torch.optim.AdamW(head.parameters(), lr=cfg.stage2_lr, weight_decay=cfg.stage2_wd)
        sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=cfg.epochs_stage2)

        criterion = LabelSmoothingCE(eps=cfg.label_smoothing, num_classes=cfg.num_classes)

        mixup_fn = None
        if cfg.use_mixup_cutmix:
            mixup_fn = Mixup(
                mixup_alpha=cfg.mixup_alpha,
                cutmix_alpha=cfg.cutmix_alpha,
                prob=cfg.mixup_prob,
                switch_prob=cfg.mixup_switch_prob,
                num_classes=cfg.num_classes
            )

        # ========== LOAD CHECKPOINT FOR STAGE-2 (IF ANY) ==========
        start_epoch_stage2, best_s2 = _load_checkpoint_for_stage2(cfg, ckpt, head, opt_head, sched_head)
        if ckpt is not None and int(ckpt.get("stage", 1)) >= 2:  # [CR]
            logger.log_event("resume_stage2", {
                "epoch_stage2": int(ckpt.get("epoch_stage2", 0)),
                "rng_restored": bool(LAST_RNG_RESTORE.get("restored", False)),
            })

        EVAL_FREQ_S2 = 20

        if start_epoch_stage2 <= cfg.epochs_stage2:
            pbar_s2_epochs = tqdm(range(start_epoch_stage2, cfg.epochs_stage2 + 1),
                                  desc="Stage-2 Epochs", dynamic_ncols=True)
            C = cfg.num_classes
            for epoch in pbar_s2_epochs:
                head.train()
                epoch_loss = 0.0

                pbar_s2_batches = tqdm(train_loader_s2, desc=f"Stage-2 Epoch {epoch}",
                                       leave=False, dynamic_ncols=True)
                for images, labels in pbar_s2_batches:
                    images = images.to(device)
                    labels = labels.to(device)
                    B = images.size(0)

                    if mixup_fn is not None:
                        images, labels = mixup_fn(images, labels)

                    with torch.no_grad():
                        feats_g, feats_a = backbone.extract_hybrid_features_all_labels(images)

                    feats_g_flat = feats_g.view(B * C, -1)
                    feats_a_flat = feats_a.view(B * C, -1)

                    logits = head(feats_g_flat, feats_a_flat)

                    loss = criterion(logits, labels)

                    opt_head.zero_grad(set_to_none=True)
                    loss.backward()
                    if cfg.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
                    opt_head.step()

                    epoch_loss += float(loss.item())
                    pbar_s2_batches.set_postfix(loss=f"{loss.item():.4f}")

                sched_head.step()
                avg_loss = epoch_loss / max(1, len(train_loader_s2))
                current_lr_s2 = sched_head.get_last_lr()[0]
                pbar_s2_epochs.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr_s2:.2e}")

                is_new_best_s2 = False
                TOPKS = tuple(range(1, 11))
                eval_accs2 = None

                if epoch % EVAL_FREQ_S2 == 0 or epoch == cfg.epochs_stage2:
                    print(f"\n[Epoch {epoch}] Starting Stage-2 evaluation on VAL...")
                    eval_accs2 = evaluate_stage2(backbone, head, val_loader, device, tta=cfg.eval_tta, topk=TOPKS)

                    if eval_accs2[1] > best_s2:
                        best_s2 = eval_accs2[1]
                        is_new_best_s2 = True

                    topk_str2 = " | ".join([f"Top{k}: {eval_accs2[k] * 100:.2f}%" for k in TOPKS])
                    pbar_s2_epochs.write(
                        f"[Stage-2] Epoch {epoch:03d}/{cfg.epochs_stage2} | {topk_str2} | Best@1: {best_s2 * 100:.2f}%"
                    )
                    pbar_s2_epochs.set_description(f"Stage-2 Epochs (Best@1: {best_s2 * 100:.2f}%)")

                # Per-epoch logging (Stage-2)
                logger.log_epoch(stage=2, epoch=epoch, metrics={
                    "lr": current_lr_s2,
                    "loss": avg_loss,
                    "best_s2": best_s2,
                    "eval_top1": (eval_accs2[1] if eval_accs2 is not None else None),
                    "eval_top5": (eval_accs2[5] if (eval_accs2 is not None and 5 in eval_accs2) else None),
                })

                # Always save Stage-2 LAST checkpoint (different filename, won't touch Stage-1)
                _save_checkpoint(
                    stage=2,
                    cfg=cfg,
                    epoch_stage1=epoch_stage1_final,
                    best_s1=best_s1,
                    net=net,
                    net_ema=net_ema,
                    opts=opts,
                    scheds=scheds,
                    queues=queues,
                    epoch_stage2=epoch,
                    best_s2=best_s2,
                    head=head,
                    opt_head=opt_head,
                    sched_head=sched_head,
                    path=STAGE2_LAST_PATH,
                )

                # Save Stage-2 BEST checkpoint only when improved
                if is_new_best_s2:
                    _save_checkpoint(
                        stage=2,
                        cfg=cfg,
                        epoch_stage1=epoch_stage1_final,
                        best_s1=best_s1,
                        net=net,
                        net_ema=net_ema,
                        opts=opts,
                        scheds=scheds,
                        queues=queues,
                        epoch_stage2=epoch,
                        best_s2=best_s2,
                        head=head,
                        opt_head=opt_head,
                        sched_head=sched_head,
                        path=STAGE2_BEST_PATH,
                    )
        else:
            print("Skipping Stage-2 training (already completed in checkpoint).")

        print("\nTraining Finished.")
        print(f"Best Stage-1 VAL (FF agg) acc: {best_s1 * 100:.2f}%")
        print(f"Best Stage-2 VAL (Hybrid head) acc: {best_s2 * 100:.2f}%")

        # ==================== FINAL HELD-OUT TEST EVALUATION (run once) ====================
        final_test_metrics = {}
        TOPKS_TEST = (1, 5)

        print("\n========== FINAL HELD-OUT TEST EVALUATION (OFFICIAL CIFAR-10 TEST SET) ==========")

        # ---- Stage-1: load BEST (selected on VAL) and evaluate once on TEST ----
        stage1_eval_path = STAGE1_BEST_PATH if os.path.exists(STAGE1_BEST_PATH) else STAGE1_LAST_PATH
        if os.path.exists(stage1_eval_path):
            _load_backbone_only(cfg, net, net_ema, stage1_eval_path)
            eval_model_s1 = net_ema if (net_ema is not None) else net

            acc_s1_no = evaluate_ff(eval_model_s1, test_loader, device, tta=False, topk=TOPKS_TEST)
            acc_s1_tta = evaluate_ff(eval_model_s1, test_loader, device, tta=True, topk=TOPKS_TEST)

            print(
                f"[Final Test][Stage-1 FF] no-TTA: Top1={acc_s1_no[1] * 100:.2f}% | Top5={acc_s1_no[5] * 100:.2f}%"
            )
            print(
                f"[Final Test][Stage-1 FF]   TTA: Top1={acc_s1_tta[1] * 100:.2f}% | Top5={acc_s1_tta[5] * 100:.2f}%"
            )

            final_test_metrics.update({
                "stage1_ckpt": str(stage1_eval_path),
                "stage1_test_top1_no_tta": float(acc_s1_no[1]),
                "stage1_test_top5_no_tta": float(acc_s1_no[5]),
                "stage1_test_top1_tta": float(acc_s1_tta[1]),
                "stage1_test_top5_tta": float(acc_s1_tta[5]),
            })

            if getattr(cfg, "save_predictions", False):
                collect_ff_predictions(
                    eval_model_s1,
                    test_loader,
                    device,
                    tta=False,
                    save_path=os.path.join(run_dir, "preds_stage1_test_no_tta.npz"),
                )
                collect_ff_predictions(
                    eval_model_s1,
                    test_loader,
                    device,
                    tta=True,
                    save_path=os.path.join(run_dir, "preds_stage1_test_tta.npz"),
                )
        else:
            print("[Final Test][Stage-1 FF] WARNING: No Stage-1 checkpoint found; skipping Stage-1 test eval.")

        # ---- Stage-2: load BEST (selected on VAL) and evaluate once on TEST ----
        stage2_eval_path = STAGE2_BEST_PATH if os.path.exists(STAGE2_BEST_PATH) else STAGE2_LAST_PATH
        if os.path.exists(stage2_eval_path):
            ckpt2 = torch.load(stage2_eval_path, map_location=device)
            try:
                net.load_state_dict(ckpt2["model"])
            except Exception:
                # allow fallback if key mismatch
                if "model" in ckpt2:
                    net.load_state_dict(ckpt2["model"], strict=False)

            if net_ema is not None and ckpt2.get("net_ema") is not None:
                try:
                    net_ema.load_state_dict(ckpt2["net_ema"])
                except Exception:
                    net_ema.load_state_dict(ckpt2["net_ema"], strict=False)

            if "head_state" in ckpt2 and ckpt2["head_state"] is not None:
                head.load_state_dict(ckpt2["head_state"])
            else:
                print(
                    "[Final Test][Stage-2 Head] WARNING: head_state missing in checkpoint; using current head weights.")

            backbone_eval = net_ema if (net_ema is not None and ckpt2.get("net_ema") is not None) else net
            for p in backbone_eval.parameters():
                p.requires_grad_(False)
            backbone_eval.eval()
            head.eval()

            acc_s2_no = evaluate_stage2(backbone_eval, head, test_loader, device, tta=False, topk=TOPKS_TEST)
            acc_s2_tta = evaluate_stage2(backbone_eval, head, test_loader, device, tta=True, topk=TOPKS_TEST)

            print(
                f"[Final Test][Stage-2 Head] no-TTA: Top1={acc_s2_no[1] * 100:.2f}% | Top5={acc_s2_no[5] * 100:.2f}%"
            )
            print(
                f"[Final Test][Stage-2 Head]   TTA: Top1={acc_s2_tta[1] * 100:.2f}% | Top5={acc_s2_tta[5] * 100:.2f}%"
            )

            final_test_metrics.update({
                "stage2_ckpt": str(stage2_eval_path),
                "stage2_test_top1_no_tta": float(acc_s2_no[1]),
                "stage2_test_top5_no_tta": float(acc_s2_no[5]),
                "stage2_test_top1_tta": float(acc_s2_tta[1]),
                "stage2_test_top5_tta": float(acc_s2_tta[5]),
            })

            if getattr(cfg, "save_predictions", False):
                collect_stage2_predictions(
                    backbone_eval,
                    head,
                    test_loader,
                    device,
                    tta=False,
                    save_path=os.path.join(run_dir, "preds_stage2_test_no_tta.npz"),
                )
                collect_stage2_predictions(
                    backbone_eval,
                    head,
                    test_loader,
                    device,
                    tta=True,
                    save_path=os.path.join(run_dir, "preds_stage2_test_tta.npz"),
                )
        else:
            print("[Final Test][Stage-2 Head] WARNING: No Stage-2 checkpoint found; skipping Stage-2 test eval.")

        # Log final test metrics once (no peeking during training)
        logger.log_event("final_test_eval", final_test_metrics)

        logger.log_event("run_end", {"best_s1": float(best_s1), "best_s2": float(best_s2), **final_test_metrics})

    except Exception as e:
        try:
            logger.log_event("error", {"type": type(e).__name__, "msg": str(e)})
        except Exception:
            pass
        raise
    finally:
        try:
            logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 8:
            try:
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('high')
            except Exception as e:
                print(f"Could not set float32 matmul precision: {e}")

    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("\n*** CUDA OUT OF MEMORY ERROR ***")
            try:
                bs = FFConfig.batch_size
            except:
                bs = 192
            print(f"Consider reducing 'batch_size' (currently {bs}) in FFConfig.")
            print("If the batch size is already very low, consider reducing model dimensions "
                  "(d_model, num_blocks) or increasing patch_size (e.g., 4).")
        else:
            raise e


"""
Training Finished.
Best Stage-1 VAL (FF agg) acc: 91.90%
Best Stage-2 VAL (Hybrid head) acc: 91.80%

========== FINAL HELD-OUT TEST EVALUATION (OFFICIAL CIFAR-10 TEST SET) ==========
[Checkpoint] Loaded backbone weights from ./checkpoints/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_P2_seed42_stage1_best.pt
[Final Test][Stage-1 FF] no-TTA: Top1=90.45% | Top5=99.62%
[Final Test][Stage-1 FF]   TTA: Top1=91.36% | Top5=99.66%
[Final Test][Stage-2 Head] no-TTA: Top1=90.42% | Top5=99.47%
[Final Test][Stage-2 Head]   TTA: Top1=91.27% | Top5=99.59%"""