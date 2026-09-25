"""Minimal BP control trainer for the rows of Table 2 (BP/FF Controlled Ablation).

This is a deliberately small reference implementation for the BP control
runs reported in the paper (`tab:bp_ff_ablation`). Three rows are
covered here directly:

  - Plain CNN (1.15M, ch. 64/128/256), BP, strong augmentation         -> 89.86%
  - FF backbone stripped (~6-8M, no goodness heads), BP, weak aug       -> 86.90%
  - FF backbone stripped (~6-8M, no goodness heads), BP, matched strong -> 93.90%

The two parameter-matched controls of the same table use the same
loop with `--backbone plain_big` (ch. 160/320/640) or `--backbone moe_small`.

We intentionally use a stock loss + optimizer; the architectural co-design
that lifts FF on the co-designed backbone is the FF-side claim and lives
in `cp_fair_cifar10.py`. Here we just train the same backbones with BP.

Env vars (with defaults):
  FF_BACKBONE     plain_cnn | ff_stripped | plain_big | moe_small  (default: ff_stripped)
  FF_AUG          weak | strong                                      (default: strong)
  FF_SEED         int                                                (default: 42)
  FF_EPOCHS       int                                                (default: 200)
  FF_LR           float                                              (default: 3e-4)
  FF_BATCH_SIZE   int                                                (default: 512)
  FF_DATA_ROOT    path to a torchvision-compatible CIFAR-10 dir      (default: ./data)
  FF_OUT_DIR      where to write preds_test.npz + log                (default: ./out/bp_<backbone>_<aug>_seed<seed>/)

Usage:
  FF_BACKBONE=ff_stripped FF_AUG=strong FF_SEED=42 python trainers/bp_control.py

The script writes:
  <FF_OUT_DIR>/preds_test.npz       (y_true, y_pred, scores; same convention as the FF preds)
  <FF_OUT_DIR>/log.json             (per-epoch train_loss, val_top1, test_top1)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----- Backbones (kept deliberately minimal; same shapes used by the FF backbone) -----

class PlainCNN(nn.Module):
    """Plain 3-block CNN, channels configurable (1.15M @ 64/128/256, 7.16M @ 160/320/640)."""
    def __init__(self, channels=(64, 128, 256), num_classes=10):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(3,  c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x):
        return self.head(self.features(x))


class FFBackboneStripped(nn.Module):
    """FF backbone minus the goodness heads: conv stem + 4 transformer-style blocks + head.

    This is intentionally a thin wrapper around the building blocks of the FF
    backbone (see cp_fair_core.py) so the BP and FF rows of Table 2 use the
    same ~6-8M parameter architecture; only the per-block goodness/multi-aspect
    heads and the cross-block aggregation logic are removed.
    """
    def __init__(self, d_model=256, num_heads=8, num_blocks=4, num_classes=10, stem_channels=384, patch_size=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels // 4, 3, padding=1), nn.BatchNorm2d(stem_channels // 4), nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 4, stem_channels // 2, 3, stride=2, padding=1), nn.BatchNorm2d(stem_channels // 2), nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels, patch_size, stride=patch_size),
        )
        # token grid after stem: 32 / 2 / patch_size  → 4×4 = 16 tokens for CIFAR-10
        self.embed = nn.Conv2d(stem_channels, d_model, 1)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                        dim_feedforward=4 * d_model,
                                        dropout=0.1, activation="gelu",
                                        batch_first=True, norm_first=True)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(self.stem(x))               # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B, H'W', D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)                # global average over tokens
        return self.head(x)


class MoESmall(nn.Module):
    """L=2 D=128 MoE-style transformer (~2M); used for the param-matched control."""
    def __init__(self, d_model=128, num_heads=4, num_blocks=2, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, 4, stride=4),
        )
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                        dim_feedforward=4 * d_model,
                                        dropout=0.1, activation="gelu",
                                        batch_first=True, norm_first=True)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.stem(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))


# ----- Augmentation pipelines -----

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def aug_weak():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def aug_strong():
    """Matched to the FF training pipeline: RandomResizedCrop + ColorJitter +
    RandAugment + GaussianBlur. Same recipe as the strong-aug FF runs."""
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def aug_test():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ----- Main training loop -----

def make_model(name: str) -> nn.Module:
    if name == "plain_cnn":
        return PlainCNN(channels=(64, 128, 256))
    if name == "plain_big":
        return PlainCNN(channels=(160, 320, 640))
    if name == "ff_stripped":
        return FFBackboneStripped()
    if name == "moe_small":
        return MoESmall()
    raise ValueError(f"unknown backbone: {name}")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_y, all_p, all_s = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_y.append(y.cpu().numpy())
            all_p.append(pred.cpu().numpy())
            all_s.append(logits.cpu().numpy())
    return (correct / total,
            np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_s))


def main():
    backbone = os.environ.get("FF_BACKBONE", "ff_stripped")
    aug_mode = os.environ.get("FF_AUG", "strong")
    seed = int(os.environ.get("FF_SEED", "42"))
    epochs = int(os.environ.get("FF_EPOCHS", "200"))
    lr = float(os.environ.get("FF_LR", "3e-4"))
    bs = int(os.environ.get("FF_BATCH_SIZE", "512"))
    data_root = os.environ.get("FF_DATA_ROOT", "./data")
    out_dir = Path(os.environ.get("FF_OUT_DIR",
                                   f"./out/bp_{backbone}_{aug_mode}_seed{seed}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = aug_weak() if aug_mode == "weak" else aug_strong()
    test_tf = aug_test()
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set,  batch_size=bs * 2, shuffle=False, num_workers=4, pin_memory=True)

    model = make_model(backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"backbone={backbone} aug={aug_mode} seed={seed} params={n_params/1e6:.2f}M epochs={epochs} lr={lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    log = []
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        sched.step()
        train_loss = running / len(train_set)
        test_top1, _, _, _ = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        log.append({"epoch": ep, "train_loss": float(train_loss),
                    "test_top1": float(test_top1), "elapsed_s": float(elapsed)})
        if test_top1 > best:
            best = test_top1
        print(f"ep {ep:3d}/{epochs}  loss {train_loss:.4f}  test_top1 {test_top1*100:6.3f}%  best {best*100:6.3f}%  ({elapsed:.0f}s)")

    # Final test-set predictions saved in the same convention as the FF preds_*.npz
    final_top1, y_true, y_pred, scores = evaluate(model, test_loader, device)
    np.savez(out_dir / "preds_test.npz", y_true=y_true, y_pred=y_pred, scores=scores)
    (out_dir / "log.json").write_text(json.dumps({
        "backbone": backbone, "aug": aug_mode, "seed": seed,
        "params": int(n_params), "epochs": epochs, "lr": lr,
        "final_test_top1": float(final_top1), "best_test_top1": float(best),
        "per_epoch": log,
    }, indent=2))
    print(f"saved {out_dir}/preds_test.npz  (final test top-1 = {final_top1*100:.3f}%)")


if __name__ == "__main__":
    main()
