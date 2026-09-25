
import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Some torchvision builds fail at import time while registering a fake NMS op.
# This script only needs datasets/transforms, so we safely bypass that specific
# registration error to keep the baseline runnable in mismatched environments.
if hasattr(torch, "library") and hasattr(torch.library, "register_fake"):
    _orig_register_fake = torch.library.register_fake

    def _safe_register_fake(op_name, *args, **kwargs):
        decorator = _orig_register_fake(op_name, *args, **kwargs)

        def wrapper(fn):
            try:
                return decorator(fn)
            except RuntimeError as e:
                if "torchvision::nms does not exist" in str(e):
                    return fn
                raise

        return wrapper

    torch.library.register_fake = _safe_register_fake

try:
    from torchvision import datasets, transforms
except ImportError as e:
    raise SystemExit("torchvision is required. Install it with `pip install torchvision`.") from e


VERSION_TAG = "ff_backbone_stripped_bp_cifar10_v1"


# -------------------- utils --------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_warmup_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def topk_correct_counts(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Dict[int, int]:
    if targets.ndim > 1:
        targets = targets.argmax(dim=1)
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out: Dict[int, int] = {}
    for k in topk:
        kk = min(k, logits.shape[1])
        out[k] = int(correct[:kk].any(dim=0).sum().item())
    return out


def make_run_dir(cfg) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = (
        f"bp_stripped_backbone_cifar10_"
        f"L{cfg.num_blocks}_D{cfg.d_model}_P{cfg.patch_size}_"
        f"seed{cfg.seed}_{stamp}"
    )
    return os.path.join(cfg.output_root, name)


# -------------------- config --------------------

@dataclass
class StrippedBPConfig:
    data_dir: str = "./data_c10"
    output_root: str = "./runs_ff_backbone_bp"
    split_path: Optional[str] = None
    resume_path: Optional[str] = None

    seed: int = int(os.environ.get("FF_SEED", 42))
    val_fraction: float = 0.10
    batch_size: int = 256
    test_batch_size: int = 256
    epochs: int = 362
    warmup_epochs: int = 15
    workers: int = 4
    eval_every: int = 20

    optimizer: str = "adamw"  # {"adamw", "sgd"}
    lr: float = 1e-3
    weight_decay: float = 0.1
    momentum: float = 0.9
    label_smoothing: float = 0.0
    grad_clip: float = 1.0

    num_classes: int = 10
    img_size: int = 32
    patch_size: int = 2
    d_model: int = 256
    num_blocks: int = 4
    num_heads: int = 8
    mlp_mult: int = 4
    dropout: float = 0.15
    head_dropout: float = 0.0
    use_rope: bool = True
    rope_theta: float = 10000.0
    use_conv_stem: bool = True
    stem_channels: int = 384
    drop_path_rate: float = 0.25

    train_aug: str = "simple"  # {"simple", "strong"}
    eval_tta: bool = False
    save_predictions: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> None:
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be one of {'adamw', 'sgd'}")
        if self.train_aug not in {"simple", "strong"}:
            raise ValueError("train_aug must be one of {'simple', 'strong'}")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")


# -------------------- data --------------------

class CIFAR10SingleFromBase(Dataset):
    """Lightweight wrapper to reuse an existing CIFAR-10 dataset instance."""

    def __init__(self, base_cifar: datasets.CIFAR10, indices: Sequence[int], transform=None):
        super().__init__()
        self.base = base_cifar
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx])
        img, target = datasets.CIFAR10.__getitem__(self.base, base_idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def deterministic_train_val_split(n_total: int, seed: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    vf = float(val_fraction)
    vf = max(0.0, min(0.5, vf))
    n_val = int(round(n_total * vf))
    n_val = max(1, min(n_total - 1, n_val))

    rng = random.Random(int(seed))
    perm = list(range(n_total))
    rng.shuffle(perm)
    val_idx = sorted(int(i) for i in perm[:n_val])
    train_idx = sorted(int(i) for i in perm[n_val:])
    return train_idx, val_idx


def load_or_make_split(cfg: StrippedBPConfig, n_total: int) -> Dict[str, object]:
    if cfg.split_path:
        with open(cfg.split_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        if "train_idx" not in split_info or "val_idx" not in split_info:
            raise ValueError(f"split_path={cfg.split_path!r} does not contain train_idx/val_idx")
        train_idx = [int(i) for i in split_info["train_idx"]]
        val_idx = [int(i) for i in split_info["val_idx"]]
        return {
            "seed": int(split_info.get("seed", cfg.seed)),
            "val_fraction": float(split_info.get("val_fraction", cfg.val_fraction)),
            "n_total": int(split_info.get("n_total", n_total)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "source": "loaded",
            "split_path": cfg.split_path,
        }

    train_idx, val_idx = deterministic_train_val_split(n_total=n_total, seed=cfg.seed, val_fraction=cfg.val_fraction)
    return {
        "seed": int(cfg.seed),
        "val_fraction": float(cfg.val_fraction),
        "n_total": int(n_total),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "source": "generated",
        "split_path": None,
    }


def make_loaders(cfg: StrippedBPConfig):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if cfg.train_aug == "simple":
        train_tf = transforms.Compose([
            transforms.RandomCrop(cfg.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_full = datasets.CIFAR10(cfg.data_dir, train=True, transform=None, download=True)
    n_total = len(train_full)
    split_info = load_or_make_split(cfg, n_total)

    train_ds = CIFAR10SingleFromBase(train_full, indices=split_info["train_idx"], transform=train_tf)
    val_ds = CIFAR10SingleFromBase(train_full, indices=split_info["val_idx"], transform=eval_tf)
    test_ds = datasets.CIFAR10(cfg.data_dir, train=False, transform=eval_tf, download=True)

    persistent = cfg.workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader, split_info


# -------------------- model pieces --------------------

class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.0):
        super().__init__()
        self.eps = float(max(0.0, eps))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.eps <= 0.0:
            return F.cross_entropy(logits, target)
        log_probs = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(log_probs, target, reduction="mean")
        smooth = -log_probs.mean(dim=1).mean()
        return (1.0 - self.eps) * nll + self.eps * smooth


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


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
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 256,
        dropout: float = 0.0,
        use_conv_stem: bool = True,
        stem_channels: int = 128,
        use_abs_pos_embed: bool = True,
    ):
        super().__init__()
        self.use_conv_stem = bool(use_conv_stem)
        self.use_abs_pos_embed = bool(use_abs_pos_embed)
        self.stem = ConvStem(in_chans, stem_channels) if self.use_conv_stem else None

        reduced = img_size // (2 if self.use_conv_stem else 1)
        if reduced % patch_size != 0:
            raise ValueError("reduced image size must be divisible by patch_size")
        self.grid = reduced // patch_size
        self.num_patches = self.grid * self.grid

        in_c = stem_channels if self.use_conv_stem else in_chans
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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def _get_cache(self, seq_len: int, device):
        key = (int(seq_len), str(device))
        if key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cache[key] = (emb.cos().to(device), emb.sin().to(device))
        return self._cache[key]

    def forward(self, seq_len: int, device):
        return self._get_cache(seq_len, device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.use_rope = bool(use_rope)
        self.rotary_emb = rotary_emb if self.use_rope else None
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        for layer in (self.Wq, self.Wk, self.Wv, self.Wo):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        bsz, n_tok, d_model = x.shape
        is_cross_attn = context is not None

        q = self.Wq(x)
        if is_cross_attn:
            k = self.Wk(context)
            v = self.Wv(context)
            n_ctx = context.shape[1]
        else:
            k = self.Wk(x)
            v = self.Wv(x)
            n_ctx = n_tok

        q = q.view(bsz, n_tok, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n_ctx, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n_ctx, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and self.rotary_emb is not None and not is_cross_attn:
            cos, sin = self.rotary_emb(n_tok, device=x.device)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
            if need_weights:
                scale = 1.0 / math.sqrt(self.head_dim)
                attn_logits = (q @ k.transpose(-2, -1)) * scale
                attn_w = F.softmax(attn_logits, dim=-1)
            else:
                attn_w = None
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_logits = (q @ k.transpose(-2, -1)) * scale
            attn_w = F.softmax(attn_logits, dim=-1)
            attn_drop = F.dropout(attn_w, p=self.dropout_p, training=self.training)
            y = attn_drop @ v
            if not need_weights:
                attn_w = None

        y = y.transpose(1, 2).contiguous().view(bsz, n_tok, d_model)
        out = self.resid_dropout(self.Wo(y))
        return out, attn_w


class AttentionPool(nn.Module):
    def __init__(self, d_model: int, n_heads: int, norm_layer=RMSNorm):
        super().__init__()
        self.ln = norm_layer(d_model)
        self.mha = Attention(d_model, n_heads, use_rope=False)
        self.q_vec = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.q_vec, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln(x)
        bsz = x.size(0)
        q = self.q_vec.expand(bsz, -1, -1)
        pooled_out, _ = self.mha(q, context=x_norm)
        return pooled_out.squeeze(1)


class PlainTokenMLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ff = nn.Sequential(
            GEGLU(d_model, mult * d_model),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        nn.init.xavier_normal_(self.ff[1].weight)
        if self.ff[1].bias is not None:
            nn.init.zeros_(self.ff[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class BPTokenBlock(nn.Module):
    """Architecture-matched supervised block: self-attention + GEGLU MLP + drop path."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_mult: int,
        dropout: float,
        drop_path: float,
        use_rope: bool,
        rotary_emb: Optional[RotaryEmbedding],
    ):
        super().__init__()
        self.ln_attn = RMSNorm(d_model)
        self.self_attn = Attention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            rotary_emb=rotary_emb,
        )
        self.dp1 = DropPath(drop_path)

        self.ln_ffn = RMSNorm(d_model)
        self.ffn = PlainTokenMLP(d_model, mult=mlp_mult, dropout=dropout)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(self.ln_attn(x))
        x = x + self.dp1(attn_out)
        x = x + self.dp2(self.ffn(self.ln_ffn(x)))
        return x


class StrippedBackboneBP(nn.Module):
    """Patch embed + self-attention token blocks + attention pooling classifier."""

    def __init__(self, cfg: StrippedBPConfig):
        super().__init__()
        use_abs_pos_embed = not cfg.use_rope

        self.embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=3,
            embed_dim=cfg.d_model,
            dropout=cfg.dropout,
            use_conv_stem=cfg.use_conv_stem,
            stem_channels=cfg.stem_channels,
            use_abs_pos_embed=use_abs_pos_embed,
        )

        if cfg.use_rope:
            rope_dim = cfg.d_model // cfg.num_heads
            self.rotary_emb = RotaryEmbedding(dim=rope_dim, theta=cfg.rope_theta)
        else:
            self.rotary_emb = None

        dprs = torch.linspace(0, cfg.drop_path_rate, steps=cfg.num_blocks).tolist()
        self.blocks = nn.ModuleList([
            BPTokenBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                mlp_mult=cfg.mlp_mult,
                dropout=cfg.dropout,
                drop_path=dprs[i],
                use_rope=cfg.use_rope,
                rotary_emb=self.rotary_emb,
            )
            for i in range(cfg.num_blocks)
        ])

        self.pool = AttentionPool(cfg.d_model, n_heads=cfg.num_heads, norm_layer=RMSNorm)
        self.head_norm = RMSNorm(cfg.d_model)
        self.head_drop = nn.Dropout(cfg.head_dropout) if cfg.head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x)
        for blk in self.blocks:
            tokens = blk(tokens)
        pooled = self.pool(tokens)
        pooled = self.head_norm(pooled)
        pooled = self.head_drop(pooled)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)


# -------------------- optimization / eval --------------------

def build_criterion(cfg: StrippedBPConfig) -> nn.Module:
    try:
        return nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
    except TypeError:
        return LabelSmoothingCE(eps=cfg.label_smoothing)


def build_optimizer(cfg: StrippedBPConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_one_epoch(
    model: StrippedBackboneBP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: StrippedBPConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    device = cfg.device

    total = 0
    running_loss = 0.0
    correct_counts = {1: 0, 5: 0}

    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)

        optimizer.step()

        bsz = labels.size(0)
        total += bsz
        running_loss += float(loss.item()) * bsz
        counts = topk_correct_counts(logits.detach(), labels, topk=(1, 5))
        correct_counts[1] += counts[1]
        correct_counts[5] += counts[5]

        avg_loss = running_loss / max(1, total)
        avg_top1 = 100.0 * correct_counts[1] / max(1, total)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.2f}")

    denom = max(1, total)
    return {
        "loss": running_loss / denom,
        "top1": correct_counts[1] / denom,
        "top5": correct_counts[5] / denom,
    }


@torch.no_grad()
def evaluate(
    model: StrippedBackboneBP,
    loader: DataLoader,
    device: str,
    tta: bool = False,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[int, float]:
    model.eval()
    total = 0
    correct_counts = {k: 0 for k in topk}
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if tta:
            logits = logits + model(torch.flip(images, dims=[3]))

        loss = criterion(logits, labels)
        bsz = labels.size(0)
        total += bsz
        loss_sum += float(loss.item()) * bsz

        counts = topk_correct_counts(logits, labels, topk=topk)
        for k in topk:
            correct_counts[k] += counts[k]

    denom = max(1, total)
    out = {k: correct_counts[k] / denom for k in topk}
    out["loss"] = loss_sum / denom
    return out


@torch.no_grad()
def collect_predictions(
    model: StrippedBackboneBP,
    loader: DataLoader,
    device: str,
    tta: bool = False,
):
    model.eval()
    y_true_list = []
    y_pred_list = []
    logits_list = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if tta:
            logits = logits + model(torch.flip(images, dims=[3]))

        preds = logits.argmax(dim=1)
        y_true_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(preds.detach().cpu().numpy())
        logits_list.append(logits.detach().cpu().numpy())

    return {
        "y_true": np.concatenate(y_true_list, axis=0),
        "y_pred": np.concatenate(y_pred_list, axis=0),
        "logits": np.concatenate(logits_list, axis=0),
    }


# -------------------- checkpointing --------------------

def save_checkpoint(
    path: str,
    model: StrippedBackboneBP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_top1: float,
    best_epoch: int,
    cfg: StrippedBPConfig,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_val_top1": float(best_val_top1),
        "best_epoch": int(best_epoch),
        "config": asdict(cfg),
        "version_tag": VERSION_TAG,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: StrippedBackboneBP,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


# -------------------- CLI --------------------

def build_config_from_args() -> StrippedBPConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Architecture-matched supervised BP baseline for the stripped FF backbone on CIFAR-10. "
            "Keeps patch embedding, self-attention token blocks, RoPE option, drop path, and attention pooling; "
            "removes FF label conditioning/goodness training and trains end-to-end with cross-entropy."
        )
    )
    parser.add_argument("--data_dir", type=str, default="./data_c10")
    parser.add_argument("--output_root", type=str, default="./runs_ff_backbone_bp")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=362)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=20)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--stem_channels", type=int, default=384)
    parser.add_argument("--drop_path_rate", type=float, default=0.25)

    parser.add_argument("--train_aug", type=str, default="simple", choices=["simple", "strong"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--no_rope", action="store_false", dest="use_rope")
    parser.add_argument("--no_conv_stem", action="store_false", dest="use_conv_stem")
    parser.add_argument("--eval_tta", action="store_true")
    parser.add_argument("--no_save_predictions", action="store_false", dest="save_predictions")
    parser.set_defaults(
        use_rope=True,
        use_conv_stem=True,
        eval_tta=False,
        save_predictions=True,
    )

    args = parser.parse_args()
    cfg = StrippedBPConfig(
        data_dir=args.data_dir,
        output_root=args.output_root,
        split_path=args.split_path,
        resume_path=args.resume_path,
        seed=args.seed,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        workers=args.workers,
        eval_every=args.eval_every,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        num_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        use_conv_stem=args.use_conv_stem,
        stem_channels=args.stem_channels,
        drop_path_rate=args.drop_path_rate,
        train_aug=args.train_aug,
        eval_tta=args.eval_tta,
        save_predictions=args.save_predictions,
        device=args.device,
    )
    cfg.validate()
    return cfg


# -------------------- main --------------------

def main() -> None:
    cfg = build_config_from_args()
    set_seed(cfg.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = cfg.device
    if device.startswith("cpu"):
        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    print(f"Using device: {device}")
    print(
        f"BP stripped-backbone config: D={cfg.d_model}, L={cfg.num_blocks}, "
        f"H={cfg.num_heads}, P={cfg.patch_size}, RoPE={cfg.use_rope}, "
        f"aug={cfg.train_aug}, opt={cfg.optimizer}"
    )
    print(
        "Kept backbone pieces: patch embedding, self-attention token blocks, "
        "GEGLU MLP, drop path, attention pooling."
    )
    print(
        "Removed FF-specific machinery: label-conditioned passes, local goodness "
        "objectives, negative sampling, multi-aspect scoring, and stage-2 head."
    )

    if cfg.resume_path:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(cfg.resume_path)))
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg)
        os.makedirs(run_dir, exist_ok=True)

    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_json(os.path.join(run_dir, "config.json"), asdict(cfg))

    train_loader, val_loader, test_loader, split_info = make_loaders(cfg)
    save_json(os.path.join(run_dir, "split_indices.json"), split_info)

    model = StrippedBackboneBP(cfg).to(device)
    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e, total=cfg.epochs, warmup=cfg.warmup_epochs: cosine_warmup_lambda(e, total, warmup),
    )

    ckpt_last = os.path.join(checkpoints_dir, "last.pt")
    ckpt_best = os.path.join(checkpoints_dir, "best.pt")

    start_epoch = 1
    best_val_top1 = 0.0
    best_epoch = 0
    best_checkpoint_source = ckpt_best

    if cfg.resume_path:
        print(f"Resuming from: {cfg.resume_path}")
        ckpt = load_checkpoint(cfg.resume_path, model, optimizer=optimizer, scheduler=scheduler, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_top1 = float(ckpt.get("best_val_top1", 0.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        print(f"Resumed at epoch {start_epoch} with best val@1 = {best_val_top1 * 100.0:.2f}%")

    history_path = os.path.join(run_dir, "history.jsonl")

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg, epoch)
        scheduler.step()

        record = {
            "epoch": int(epoch),
            "time_sec": float(time.time() - epoch_start),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_top1": float(train_metrics["top1"]),
            "train_top5": float(train_metrics["top5"]),
        }

        should_eval = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
        if should_eval:
            val_metrics = evaluate(model, val_loader, device=device, tta=cfg.eval_tta, topk=(1, 5))
            record["val_loss"] = float(val_metrics["loss"])
            record["val_top1"] = float(val_metrics[1])
            record["val_top5"] = float(val_metrics[5])

            print(
                f"[Epoch {epoch:03d}/{cfg.epochs}] "
                f"train@1={train_metrics['top1'] * 100.0:.2f}% "
                f"val@1={val_metrics[1] * 100.0:.2f}% "
                f"val@5={val_metrics[5] * 100.0:.2f}%"
            )

            if val_metrics[1] >= best_val_top1:
                best_val_top1 = float(val_metrics[1])
                best_epoch = int(epoch)
                save_checkpoint(ckpt_best, model, optimizer, scheduler, epoch, best_val_top1, best_epoch, cfg)
                best_checkpoint_source = ckpt_best
                print(f"  New best checkpoint saved to: {ckpt_best}")

        append_jsonl(history_path, record)
        save_checkpoint(ckpt_last, model, optimizer, scheduler, epoch, best_val_top1, best_epoch, cfg)

    if os.path.exists(ckpt_best):
        load_checkpoint(ckpt_best, model, device=device)
    elif os.path.exists(ckpt_last):
        load_checkpoint(ckpt_last, model, device=device)
        best_checkpoint_source = ckpt_last

    val_final = evaluate(model, val_loader, device=device, tta=cfg.eval_tta, topk=(1, 5))
    test_final = evaluate(model, test_loader, device=device, tta=cfg.eval_tta, topk=(1, 5))

    summary = {
        "version_tag": VERSION_TAG,
        "run_dir": run_dir,
        "device": device,
        "num_parameters": int(params),
        "best_checkpoint": best_checkpoint_source,
        "best_val_epoch": int(best_epoch),
        "eval_tta": bool(cfg.eval_tta),
        "train_aug": cfg.train_aug,
        "optimizer": cfg.optimizer,
        "val_top1": float(val_final[1]),
        "val_top5": float(val_final[5]),
        "test_top1": float(test_final[1]),
        "test_top5": float(test_final[5]),
    }
    save_json(os.path.join(run_dir, "final_metrics.json"), summary)

    if cfg.save_predictions:
        preds_val = collect_predictions(model, val_loader, device=device, tta=cfg.eval_tta)
        np.savez_compressed(os.path.join(run_dir, "preds_val.npz"), **preds_val)

        preds_test = collect_predictions(model, test_loader, device=device, tta=cfg.eval_tta)
        np.savez_compressed(os.path.join(run_dir, "preds_test.npz"), **preds_test)

    print("Training finished.")
    print(f"Best VAL Top-1:  {best_val_top1 * 100.0:.2f}% @ epoch {best_epoch}")
    print(f"Final TEST Top-1: {test_final[1] * 100.0:.2f}%")
    print(f"Final TEST Top-5: {test_final[5] * 100.0:.2f}%")
    print(f"Best checkpoint:  {best_checkpoint_source}")
    print(f"Run directory:    {run_dir}")


if __name__ == "__main__":
    main()


"""
Best VAL Top-1:  87.34% @ epoch 360
Final TEST Top-1: 86.90%
Final TEST Top-5: 98.66%
Best checkpoint:  ./runs_ff_backbone_bp/bp_stripped_backbone_cifar10_L4_D256_P2_seed42_20260307_035223/checkpoints/best.pt
Run directory:    ./runs_ff_backbone_bp/bp_stripped_backbone_cifar10_L4_D256_P2_seed42_20260307_035223"""