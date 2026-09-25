import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
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
if hasattr(torch, 'library') and hasattr(torch.library, 'register_fake'):
    _orig_register_fake = torch.library.register_fake

    def _safe_register_fake(op_name, *args, **kwargs):
        decorator = _orig_register_fake(op_name, *args, **kwargs)

        def wrapper(fn):
            try:
                return decorator(fn)
            except RuntimeError as e:
                if 'torchvision::nms does not exist' in str(e):
                    return fn
                raise

        return wrapper

    torch.library.register_fake = _safe_register_fake

try:
    from torchvision import datasets, transforms
except ImportError as e:
    raise SystemExit("torchvision is required. Install it with `pip install torchvision`.") from e


VERSION_TAG = "plain_cnn_ff_cifar10_v1"


# -------------------- utils --------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def cosine_warmup_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))



def topk_correct_counts(scores: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Dict[int, int]:
    if targets.ndim > 1:
        targets = targets.argmax(dim=1)
    maxk = min(max(topk), scores.shape[1])
    _, pred = scores.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out: Dict[int, int] = {}
    for k in topk:
        kk = min(k, scores.shape[1])
        out[k] = int(correct[:kk].any(dim=0).sum().item())
    return out



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)



def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")



def sample_incorrect_labels(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 1:
        raise ValueError("num_classes must be >= 2 for negative labels")
    offsets = torch.randint(1, num_classes, size=targets.shape, device=targets.device)
    return (targets + offsets) % num_classes



def rms_norm_no_mean(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Per-sample RMS normalization without mean subtraction.

    This matches the spirit of Hinton's FF normalization step: the goodness is
    encoded in the length of the activation vector, while only the orientation
    is passed to deeper layers.
    """
    denom = torch.sqrt(torch.mean(x.pow(2), dim=(1, 2, 3), keepdim=True) + eps)
    return x / denom


# -------------------- config --------------------

@dataclass
class FFConfig:
    data_dir: str = "./data_c10"
    output_root: str = "./runs_plain_cnn_ff"
    split_path: Optional[str] = None

    seed: int = int(os.environ.get("FF_SEED", 42))
    val_fraction: float = 0.10
    batch_size: int = 256
    test_batch_size: int = 256
    epochs: int = 362
    warmup_epochs: int = 15
    workers: int = 4

    optimizer: str = "adamw"  # {"adamw", "sgd"}
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # used only for SGD
    grad_clip: float = 0.0

    num_classes: int = 10
    img_size: int = 32
    stage_channels: Tuple[int, ...] = (64, 64, 128, 128, 256, 256)
    pool_after: Tuple[bool, ...] = (False, True, False, True, False, False)
    kernel_size: int = 3
    label_scale: float = 1.0
    goodness_threshold: float = 1.0
    norm_eps: float = 1e-4
    score_start_layer: int = 0
    eval_label_chunk: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- model --------------------

class FFLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, pool: bool = False) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = F.relu(z, inplace=False)
        z = self.pool(z)
        return z

    def goodness(self, z: torch.Tensor) -> torch.Tensor:
        # Mean squared activity keeps the scale comparable across layers.
        return z.pow(2).flatten(1).mean(dim=1)


class PlainFFCNN(nn.Module):
    """A plain convolutional network trained with layer-local Forward-Forward updates.

    The architecture mirrors a simple VGG-style CIFAR-10 CNN at the convolutional
    level (6 conv layers with max-pooling after layers 2 and 4). Labels are injected
    as spatially broadcast one-hot planes concatenated to the image channels.
    """

    def __init__(
        self,
        num_classes: int = 10,
        stage_channels: Sequence[int] = (64, 64, 128, 128, 256, 256),
        pool_after: Sequence[bool] = (False, True, False, True, False, False),
        kernel_size: int = 3,
        label_scale: float = 1.0,
        norm_eps: float = 1e-4,
        score_start_layer: int = 0,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(pool_after):
            raise ValueError("stage_channels and pool_after must have the same length")
        if not 0 <= int(score_start_layer) < len(stage_channels):
            raise ValueError("score_start_layer must be between 0 and n_layers-1")

        self.num_classes = int(num_classes)
        self.label_scale = float(label_scale)
        self.norm_eps = float(norm_eps)
        self.score_start_layer = int(score_start_layer)

        layers: List[FFLayer] = []
        in_ch = 3 + self.num_classes
        for out_ch, pool in zip(stage_channels, pool_after):
            layers.append(FFLayer(in_ch=in_ch, out_ch=int(out_ch), kernel_size=kernel_size, pool=bool(pool)))
            in_ch = int(out_ch)
        self.layers = nn.ModuleList(layers)

    def make_label_planes(self, labels: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=dtype)
        planes = one_hot.view(labels.size(0), self.num_classes, 1, 1).expand(-1, -1, height, width)
        if self.label_scale != 1.0:
            planes = planes * self.label_scale
        return planes

    def inject_labels(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        planes = self.make_label_planes(labels, images.shape[2], images.shape[3], images.dtype)
        return torch.cat([images, planes], dim=1)

    def forward_goodness(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.inject_labels(images, labels)
        total = images.new_zeros(images.size(0))
        per_layer: List[torch.Tensor] = []
        for li, layer in enumerate(self.layers):
            z = layer.forward_raw(x)
            g = layer.goodness(z)
            per_layer.append(g)
            if li >= self.score_start_layer:
                total = total + g
            x = rms_norm_no_mean(z, eps=self.norm_eps)
        return total, per_layer


# -------------------- data --------------------

class TransformedCIFARSubset(Dataset):
    def __init__(self, base_ds: datasets.CIFAR10, indices: Sequence[int], transform=None):
        super().__init__()
        self.base_ds = base_ds
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, target = self.base_ds[int(self.indices[idx])]
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



def load_or_make_split(cfg: FFConfig, n_total: int) -> Dict[str, object]:
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



def make_loaders(cfg: FFConfig):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(cfg.img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    base_train = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=None)
    split_info = load_or_make_split(cfg, n_total=len(base_train))
    train_idx = split_info["train_idx"]
    val_idx = split_info["val_idx"]

    train_ds = TransformedCIFARSubset(base_train, train_idx, transform=train_tf)
    val_ds = TransformedCIFARSubset(base_train, val_idx, transform=eval_tf)
    test_ds = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=eval_tf)

    persistent = cfg.workers > 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader, split_info


# -------------------- optimization --------------------


def make_optimizer(cfg: FFConfig, module: nn.Module) -> torch.optim.Optimizer:
    opt_name = str(cfg.optimizer).lower().strip()
    if opt_name == "adamw":
        return torch.optim.AdamW(module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(
            module.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r}")



def make_scheduler(cfg: FFConfig, optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e, total=cfg.epochs, warm=cfg.warmup_epochs: cosine_warmup_lambda(e, total, warm),
    )


# -------------------- train / eval --------------------


def ff_goodness_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, threshold: float) -> torch.Tensor:
    pos_term = F.softplus(threshold - g_pos).mean()
    neg_term = F.softplus(g_neg - threshold).mean()
    return pos_term + neg_term



def train_one_epoch(
    model: PlainFFCNN,
    loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    device: torch.device,
    cfg: FFConfig,
) -> Dict[str, float]:
    model.train()
    n_layers = len(model.layers)

    layer_loss_sums = [0.0 for _ in range(n_layers)]
    layer_pos_sums = [0.0 for _ in range(n_layers)]
    layer_neg_sums = [0.0 for _ in range(n_layers)]
    layer_sep_sums = [0.0 for _ in range(n_layers)]
    layer_disc_sums = [0.0 for _ in range(n_layers)]
    total_examples = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        neg_targets = sample_incorrect_labels(targets, cfg.num_classes)

        pos_x = model.inject_labels(images, targets)
        neg_x = model.inject_labels(images, neg_targets)
        batch_size = images.size(0)

        batch_loss_mean = 0.0
        batch_disc_mean = 0.0

        for li, (layer, optimizer) in enumerate(zip(model.layers, optimizers)):
            optimizer.zero_grad(set_to_none=True)

            pos_z = layer.forward_raw(pos_x)
            neg_z = layer.forward_raw(neg_x)
            g_pos = layer.goodness(pos_z)
            g_neg = layer.goodness(neg_z)

            loss = ff_goodness_loss(g_pos, g_neg, threshold=cfg.goodness_threshold)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(layer.parameters(), cfg.grad_clip)

            optimizer.step()

            with torch.no_grad():
                pos_z_fresh = layer.forward_raw(pos_x)
                neg_z_fresh = layer.forward_raw(neg_x)
                pos_x = rms_norm_no_mean(pos_z_fresh, eps=cfg.norm_eps)
                neg_x = rms_norm_no_mean(neg_z_fresh, eps=cfg.norm_eps)

            pos_acc = (g_pos.detach() > cfg.goodness_threshold).float().mean()
            neg_acc = (g_neg.detach() < cfg.goodness_threshold).float().mean()
            disc_acc = 0.5 * (pos_acc + neg_acc)
            sep = (g_pos.detach() - g_neg.detach()).mean()

            layer_loss_sums[li] += float(loss.item()) * batch_size
            layer_pos_sums[li] += float(g_pos.detach().mean().item()) * batch_size
            layer_neg_sums[li] += float(g_neg.detach().mean().item()) * batch_size
            layer_sep_sums[li] += float(sep.item()) * batch_size
            layer_disc_sums[li] += float(disc_acc.item()) * batch_size

            batch_loss_mean += float(loss.item())
            batch_disc_mean += float(disc_acc.item())

        total_examples += batch_size
        pbar.set_postfix(
            loss=f"{batch_loss_mean / max(1, n_layers):.4f}",
            local_acc=f"{100.0 * batch_disc_mean / max(1, n_layers):.2f}%",
        )

    out: Dict[str, float] = {
        "loss": sum(layer_loss_sums) / max(1, total_examples * n_layers),
        "local_acc": sum(layer_disc_sums) / max(1, total_examples * n_layers),
        "pos_goodness": sum(layer_pos_sums) / max(1, total_examples * n_layers),
        "neg_goodness": sum(layer_neg_sums) / max(1, total_examples * n_layers),
        "sep": sum(layer_sep_sums) / max(1, total_examples * n_layers),
    }
    for li in range(n_layers):
        out[f"layer{li + 1}_loss"] = layer_loss_sums[li] / max(1, total_examples)
        out[f"layer{li + 1}_local_acc"] = layer_disc_sums[li] / max(1, total_examples)
        out[f"layer{li + 1}_pos_goodness"] = layer_pos_sums[li] / max(1, total_examples)
        out[f"layer{li + 1}_neg_goodness"] = layer_neg_sums[li] / max(1, total_examples)
        out[f"layer{li + 1}_sep"] = layer_sep_sums[li] / max(1, total_examples)
    return out


@torch.no_grad()
def predict_scores(model: PlainFFCNN, images: torch.Tensor, label_chunk: int = 5) -> torch.Tensor:
    """Compute FF goodness scores for all candidate labels.

    The evaluation is chunked over candidate labels to trade memory for speed.
    For CIFAR-10, chunk sizes between 2 and 10 are usually reasonable.
    """
    n_classes = model.num_classes
    label_chunk = max(1, min(int(label_chunk), n_classes))
    batch = images.size(0)
    pieces: List[torch.Tensor] = []

    for start in range(0, n_classes, label_chunk):
        stop = min(n_classes, start + label_chunk)
        chunk_labels = torch.arange(start, stop, device=images.device, dtype=torch.long)
        chunk = chunk_labels.numel()

        img_rep = images.unsqueeze(1).expand(batch, chunk, *images.shape[1:]).reshape(batch * chunk, *images.shape[1:])
        y_rep = chunk_labels.unsqueeze(0).expand(batch, chunk).reshape(-1)

        chunk_scores, _ = model.forward_goodness(img_rep, y_rep)
        pieces.append(chunk_scores.view(batch, chunk))

    return torch.cat(pieces, dim=1)


@torch.no_grad()
def evaluate(
    model: PlainFFCNN,
    loader: DataLoader,
    device: torch.device,
    topk: Tuple[int, ...] = (1, 5),
    eval_label_chunk: int = 5,
) -> Dict[str, float]:
    model.eval()
    running_count = 0
    correct = {k: 0 for k in topk}
    margin_sum = 0.0
    true_score_sum = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        scores = predict_scores(model, images, label_chunk=eval_label_chunk)
        counts = topk_correct_counts(scores, targets, topk=topk)
        for k in topk:
            correct[k] += counts[k]

        true_scores = scores.gather(1, targets.view(-1, 1)).squeeze(1)
        wrong_scores = scores.clone()
        wrong_scores.scatter_(1, targets.view(-1, 1), float("-inf"))
        best_wrong = wrong_scores.max(dim=1).values

        true_score_sum += float(true_scores.sum().item())
        margin_sum += float((true_scores - best_wrong).sum().item())
        running_count += images.size(0)

    out: Dict[str, float] = {}
    for k in topk:
        out[f"top{k}"] = correct[k] / max(1, running_count)
    out["avg_true_score"] = true_score_sum / max(1, running_count)
    out["avg_margin"] = margin_sum / max(1, running_count)
    return out



def save_checkpoint(
    path: str,
    model: PlainFFCNN,
    optimizers: Sequence[torch.optim.Optimizer],
    schedulers: Sequence[Optional[torch.optim.lr_scheduler._LRScheduler]],
    epoch: int,
    best_val_top1: float,
    cfg: FFConfig,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_top1": float(best_val_top1),
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "schedulers": [sch.state_dict() if sch is not None else None for sch in schedulers],
            "cfg": asdict(cfg),
            "version_tag": VERSION_TAG,
        },
        path,
    )


# -------------------- main --------------------


def parse_args() -> FFConfig:
    parser = argparse.ArgumentParser(description="Plain CNN + Forward-Forward baseline for CIFAR-10")
    parser.add_argument("--data_dir", type=str, default=FFConfig.data_dir)
    parser.add_argument("--output_root", type=str, default=FFConfig.output_root)
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Optional path to split_indices.json from your FF/BP run for exact split matching.",
    )
    parser.add_argument("--seed", type=int, default=FFConfig.seed)
    parser.add_argument("--val_fraction", type=float, default=FFConfig.val_fraction)
    parser.add_argument("--batch_size", type=int, default=FFConfig.batch_size)
    parser.add_argument("--test_batch_size", type=int, default=FFConfig.test_batch_size)
    parser.add_argument("--epochs", type=int, default=FFConfig.epochs)
    parser.add_argument("--warmup_epochs", type=int, default=FFConfig.warmup_epochs)
    parser.add_argument("--workers", type=int, default=FFConfig.workers)
    parser.add_argument("--optimizer", type=str, default=FFConfig.optimizer, choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=FFConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=FFConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=FFConfig.momentum)
    parser.add_argument("--grad_clip", type=float, default=FFConfig.grad_clip)
    parser.add_argument("--label_scale", type=float, default=FFConfig.label_scale)
    parser.add_argument("--goodness_threshold", type=float, default=FFConfig.goodness_threshold)
    parser.add_argument("--score_start_layer", type=int, default=FFConfig.score_start_layer,
                        help="Sum goodness starting from this layer index (0-based).")
    parser.add_argument("--eval_label_chunk", type=int, default=FFConfig.eval_label_chunk,
                        help="How many candidate labels to score at once during validation/test.")
    parser.add_argument(
        "--stage_channels", type=int, nargs="+", default=list(FFConfig.stage_channels),
        help="Channels per FF layer (any number of ints). Default matches the small "
             "baseline (64 64 128 128 256 256, ~0.7M params); use e.g. "
             "`--stage_channels 160 160 320 320 640 640` to scale to ~7M params for "
             "parameter-matched FF-vs-BP comparisons. Length must match --pool_after.",
    )
    parser.add_argument(
        "--pool_after", type=int, nargs="+",
        default=[int(b) for b in FFConfig.pool_after],
        help="0/1 flags per layer indicating whether to apply 2x2 max-pool after that "
             "layer. Length must match --stage_channels.",
    )
    parser.add_argument("--device", type=str, default=FFConfig.device)
    args = parser.parse_args()

    if len(args.stage_channels) != len(args.pool_after):
        raise SystemExit(
            f"--stage_channels (len={len(args.stage_channels)}) and "
            f"--pool_after (len={len(args.pool_after)}) must have the same length"
        )

    return FFConfig(
        data_dir=args.data_dir,
        output_root=args.output_root,
        split_path=args.split_path,
        seed=args.seed,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        workers=args.workers,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        grad_clip=args.grad_clip,
        label_scale=args.label_scale,
        goodness_threshold=args.goodness_threshold,
        score_start_layer=args.score_start_layer,
        eval_label_chunk=args.eval_label_chunk,
        stage_channels=tuple(args.stage_channels),
        pool_after=tuple(bool(b) for b in args.pool_after),
        device=args.device,
    )



def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        try:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"ff_plaincnn_{VERSION_TAG}_seed{cfg.seed}_{timestamp}"
    run_dir = os.path.join(cfg.output_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    print(f"Run dir: {run_dir}")

    train_loader, val_loader, test_loader, split_info = make_loaders(cfg)
    save_json(os.path.join(run_dir, "config.json"), asdict(cfg))
    save_json(os.path.join(run_dir, "split_indices.json"), split_info)

    print(
        f"Split source: {split_info['source']} | "
        f"n_train={split_info['n_train']} | n_val={split_info['n_val']} | "
        f"n_test=10000"
    )

    model = PlainFFCNN(
        num_classes=cfg.num_classes,
        stage_channels=cfg.stage_channels,
        pool_after=cfg.pool_after,
        kernel_size=cfg.kernel_size,
        label_scale=cfg.label_scale,
        norm_eps=cfg.norm_eps,
        score_start_layer=cfg.score_start_layer,
    ).to(device)
    n_params = count_parameters(model)
    print(f"Model: PlainFFCNN | trainable params: {n_params:,}")

    optimizers = [make_optimizer(cfg, layer) for layer in model.layers]
    schedulers = [make_scheduler(cfg, opt) for opt in optimizers]

    best_val_top1 = 0.0
    best_epoch = 0
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")
    history_path = os.path.join(run_dir, "history.jsonl")

    epoch_pbar = tqdm(range(1, cfg.epochs + 1), desc="epochs")
    for epoch in epoch_pbar:
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizers=optimizers,
            device=device,
            cfg=cfg,
        )
        val_metrics = evaluate(model, val_loader, device=device, topk=(1, 5), eval_label_chunk=cfg.eval_label_chunk)
        current_lr = float(optimizers[0].param_groups[0]["lr"])
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()

        is_best = val_metrics["top1"] > best_val_top1
        if is_best:
            best_val_top1 = float(val_metrics["top1"])
            best_epoch = int(epoch)
            save_checkpoint(best_path, model, optimizers, schedulers, epoch, best_val_top1, cfg)

        save_checkpoint(last_path, model, optimizers, schedulers, epoch, best_val_top1, cfg)

        epoch_log = {
            "epoch": int(epoch),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "train_local_acc": float(train_metrics["local_acc"]),
            "train_pos_goodness": float(train_metrics["pos_goodness"]),
            "train_neg_goodness": float(train_metrics["neg_goodness"]),
            "train_sep": float(train_metrics["sep"]),
            "val_top1": float(val_metrics["top1"]),
            "val_top5": float(val_metrics["top5"]),
            "val_avg_true_score": float(val_metrics["avg_true_score"]),
            "val_avg_margin": float(val_metrics["avg_margin"]),
            "best_val_top1": float(best_val_top1),
            "best_epoch": int(best_epoch),
        }
        for key, value in train_metrics.items():
            if key.startswith("layer"):
                epoch_log[key] = float(value)
        append_jsonl(history_path, epoch_log)

        epoch_pbar.set_postfix(
            train_local_acc=f"{train_metrics['local_acc'] * 100:.2f}%",
            val_top1=f"{val_metrics['top1'] * 100:.2f}%",
            best=f"{best_val_top1 * 100:.2f}%",
        )
        tqdm.write(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"lr={current_lr:.5g} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_local_acc={train_metrics['local_acc'] * 100:.2f}% | "
            f"train_sep={train_metrics['sep']:.4f} | "
            f"val_top1={val_metrics['top1'] * 100:.2f}% | "
            f"val_top5={val_metrics['top5'] * 100:.2f}% | "
            f"val_margin={val_metrics['avg_margin']:.4f} | "
            f"best={best_val_top1 * 100:.2f}%"
        )

    if not os.path.exists(best_path):
        raise RuntimeError("Best checkpoint was not saved; training appears to have failed.")

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model"])

    final_test = evaluate(model, test_loader, device=device, topk=(1, 5), eval_label_chunk=cfg.eval_label_chunk)
    summary = {
        "version_tag": VERSION_TAG,
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val_top1),
        "test_top1": float(final_test["top1"]),
        "test_top5": float(final_test["top5"]),
        "test_avg_true_score": float(final_test["avg_true_score"]),
        "test_avg_margin": float(final_test["avg_margin"]),
        "n_params": int(n_params),
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "run_dir": run_dir,
    }
    save_json(os.path.join(run_dir, "final_metrics.json"), summary)

    print("\nTraining finished.")
    print(f"Best VAL Top-1:  {best_val_top1 * 100:.2f}% @ epoch {best_epoch}")
    print(f"Final TEST Top-1: {final_test['top1'] * 100:.2f}%")
    print(f"Final TEST Top-5: {final_test['top5'] * 100:.2f}%")
    print(f"Best checkpoint:  {best_path}")
    print(f"Run directory:    {run_dir}")


if __name__ == "__main__":
    main()
"""
Training finished.
Best VAL Top-1:  32.20% @ epoch 58
Final TEST Top-1: 29.41%
Final TEST Top-5: 83.12%
Best checkpoint:  ./runs_plain_cnn_ff/ff_plaincnn_plain_cnn_ff_cifar10_v1_seed42_20260306_121504/checkpoints/best.pt
Run directory:    ./runs_plain_cnn_ff/ff_plaincnn_plain_cnn_ff_cifar10_v1_seed42_20260306_121504"""