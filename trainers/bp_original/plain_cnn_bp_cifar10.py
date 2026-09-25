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


VERSION_TAG = "plain_cnn_bp_cifar10_v1"


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


# -------------------- config --------------------

@dataclass
class BPConfig:
    data_dir: str = "./data_c10"
    output_root: str = "./runs_plain_cnn_bp"
    split_path: Optional[str] = None

    seed: int = int(os.environ.get("FF_SEED", 42))
    val_fraction: float = 0.10
    batch_size: int = 512
    test_batch_size: int = 256
    epochs: int = 362
    warmup_epochs: int = 15
    workers: int = 4

    optimizer: str = "adamw"  # {"adamw", "sgd"}
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # used only for SGD
    label_smoothing: float = 0.0
    grad_clip: float = 0.0

    num_classes: int = 10
    img_size: int = 32
    channels: Tuple[int, int, int] = (64, 128, 256)
    dropout: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- model --------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlainCNN(nn.Module):
    """A simple VGG-style CNN baseline for CIFAR-10."""

    def __init__(self, num_classes: int = 10, channels: Sequence[int] = (64, 128, 256), dropout: float = 0.0):
        super().__init__()
        c1, c2, c3 = [int(c) for c in channels]
        self.features = nn.Sequential(
            ConvBlock(3, c1, pool=True),
            ConvBlock(c1, c2, pool=True),
            ConvBlock(c2, c3, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(c3, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


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


def load_or_make_split(cfg: BPConfig, n_total: int) -> Dict[str, object]:
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


def make_loaders(cfg: BPConfig):
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

def make_optimizer(cfg: BPConfig, model: nn.Module) -> torch.optim.Optimizer:
    opt_name = str(cfg.optimizer).lower().strip()
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r}")


def make_scheduler(cfg: BPConfig, optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e, total=cfg.epochs, warm=cfg.warmup_epochs: cosine_warmup_lambda(e, total, warm),
    )


# -------------------- train / eval --------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_top1 = 0
    running_count = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        running_top1 += topk_correct_counts(logits.detach(), targets, topk=(1,))[1]
        running_count += batch_size

        avg_loss = running_loss / max(1, running_count)
        avg_top1 = running_top1 / max(1, running_count)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1 * 100:.2f}%")

    return {
        "loss": running_loss / max(1, running_count),
        "top1": running_top1 / max(1, running_count),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_count = 0
    correct = {k: 0 for k in topk}

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, targets)
            running_loss += float(loss.item()) * images.size(0)

        counts = topk_correct_counts(logits, targets, topk=topk)
        for k in topk:
            correct[k] += counts[k]
        running_count += images.size(0)

    out: Dict[str, float] = {}
    if criterion is not None:
        out["loss"] = running_loss / max(1, running_count)
    for k in topk:
        out[f"top{k}"] = correct[k] / max(1, running_count)
    return out


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, epoch: int, best_val_top1: float, cfg: BPConfig) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_top1": float(best_val_top1),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": asdict(cfg),
            "version_tag": VERSION_TAG,
        },
        path,
    )


# -------------------- main --------------------

def parse_args() -> BPConfig:
    parser = argparse.ArgumentParser(description="Plain CNN + Backprop baseline for CIFAR-10")
    parser.add_argument("--data_dir", type=str, default=BPConfig.data_dir)
    parser.add_argument("--output_root", type=str, default=BPConfig.output_root)
    parser.add_argument("--split_path", type=str, default=None,
                        help="Optional path to split_indices.json from your FF run for exact split matching.")
    parser.add_argument("--seed", type=int, default=BPConfig.seed)
    parser.add_argument("--val_fraction", type=float, default=BPConfig.val_fraction)
    parser.add_argument("--batch_size", type=int, default=BPConfig.batch_size)
    parser.add_argument("--test_batch_size", type=int, default=BPConfig.test_batch_size)
    parser.add_argument("--epochs", type=int, default=BPConfig.epochs)
    parser.add_argument("--warmup_epochs", type=int, default=BPConfig.warmup_epochs)
    parser.add_argument("--workers", type=int, default=BPConfig.workers)
    parser.add_argument("--optimizer", type=str, default=BPConfig.optimizer, choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=BPConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=BPConfig.weight_decay)
    parser.add_argument("--momentum", type=float, default=BPConfig.momentum)
    parser.add_argument("--label_smoothing", type=float, default=BPConfig.label_smoothing)
    parser.add_argument("--grad_clip", type=float, default=BPConfig.grad_clip)
    parser.add_argument("--dropout", type=float, default=BPConfig.dropout)
    parser.add_argument(
        "--channels", type=int, nargs=3, default=list(BPConfig.channels),
        metavar=("C1", "C2", "C3"),
        help="Channels per stage (3 ints). Default matches the small baseline "
             "(64 128 256, ~1.2M params); use e.g. `--channels 160 320 640` to "
             "scale to ~7M params for parameter-matched FF-vs-BP comparisons.",
    )
    parser.add_argument("--device", type=str, default=BPConfig.device)
    args = parser.parse_args()

    return BPConfig(
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
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        channels=tuple(args.channels),
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
    run_name = f"bp_plaincnn_{VERSION_TAG}_seed{cfg.seed}_{timestamp}"
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

    model = PlainCNN(num_classes=cfg.num_classes, channels=cfg.channels, dropout=cfg.dropout).to(device)
    n_params = count_parameters(model)
    print(f"Model: PlainCNN | trainable params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

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
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=cfg.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device=device, criterion=criterion, topk=(1, 5))
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        is_best = val_metrics["top1"] > best_val_top1
        if is_best:
            best_val_top1 = float(val_metrics["top1"])
            best_epoch = int(epoch)
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_top1, cfg)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_top1, cfg)

        epoch_log = {
            "epoch": int(epoch),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "train_top1": float(train_metrics["top1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_top1": float(val_metrics["top1"]),
            "val_top5": float(val_metrics["top5"]),
            "best_val_top1": float(best_val_top1),
            "best_epoch": int(best_epoch),
        }
        append_jsonl(history_path, epoch_log)

        epoch_pbar.set_postfix(
            train_top1=f"{train_metrics['top1'] * 100:.2f}%",
            val_top1=f"{val_metrics['top1'] * 100:.2f}%",
            best=f"{best_val_top1 * 100:.2f}%",
        )
        tqdm.write(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"lr={current_lr:.5g} | "
            f"train_loss={train_metrics['loss']:.4f} | train_top1={train_metrics['top1'] * 100:.2f}% | "
            f"val_loss={val_metrics['loss']:.4f} | val_top1={val_metrics['top1'] * 100:.2f}% | "
            f"val_top5={val_metrics['top5'] * 100:.2f}% | best={best_val_top1 * 100:.2f}%"
        )

    if not os.path.exists(best_path):
        raise RuntimeError("Best checkpoint was not saved; training appears to have failed.")

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model"])

    final_test = evaluate(model, test_loader, device=device, criterion=criterion, topk=(1, 5))
    summary = {
        "version_tag": VERSION_TAG,
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val_top1),
        "test_loss": float(final_test["loss"]),
        "test_top1": float(final_test["top1"]),
        "test_top5": float(final_test["top5"]),
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
Best VAL Top-1:  90.78% @ epoch 281
Final TEST Top-1: 89.86%
Final TEST Top-5: 99.61%
Best checkpoint:  ./runs_plain_cnn_bp/bp_plaincnn_plain_cnn_bp_cifar10_v1_seed42_20260306_104748/checkpoints/best.pt
Run directory:    ./runs_plain_cnn_bp/bp_plaincnn_plain_cnn_bp_cifar10_v1_seed42_20260306_104748"""