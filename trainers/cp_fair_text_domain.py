"""
Cross-modality FF trainer for the text-domain results in
`appendix/H_text_domain.tex` (`tab:text_domain_full`).

Architecture: frozen `bert-base-uncased` (BP-trained, public HF
checkpoint) → 768-d pooled embedding → 768→256 linear projection →
L=4 D=256 FF Hybrid Block stack from `cp_fair_core` → goodness-based
classifier. Stage-2 (frozen-feature attentive readout) is identical
to the CIFAR-10 / CIFAR-100 trainer.

This is a thin wrapper around `cp_fair_cifar10.py` that swaps only
the input stem; the FF blocks, goodness aspects, hard-negative
mining, locality contract, and Stage-2 head are all imported
unchanged. We keep the file small to avoid duplicating
`cp_fair_core.py`.

Datasets (loaded via HuggingFace `datasets`):
  - IMDb (binary)            → `datasets.load_dataset('imdb')`
  - 20 Newsgroups (20-class) → fetched via `sklearn.datasets`,
                               wrapped into HF Dataset
  - AG News (4-class)        → `datasets.load_dataset('ag_news')`

Reproduction: see `analysis/text_domain_README.md` for the recipe.
Expected numbers (seeds 42/43/44 each, 9 runs total; Stage-2 test top-1, mean ± sample SD):
  IMDb:           87.25 ± 0.15 %
  20 Newsgroups:  62.66 ± 0.24 %
  AG News:        93.36 ± 0.07 %

Caveats: the BERT encoder is BP-trained (frozen at FF training time),
so the "FF training" phase is local to the FF hybrid blocks above the
BERT pool. We treat this as a representation-learning sanity check
across modality, not as evidence about FF-from-scratch on text.

Usage
-----
  pip install transformers datasets scikit-learn
  FF_TEXT_DATASET=imdb FF_SEED=42 \
    python trainers/cp_fair_text_domain.py
  FF_TEXT_DATASET=20news FF_SEED=42 \
    python trainers/cp_fair_text_domain.py
  FF_TEXT_DATASET=agnews FF_SEED=42 \
    python trainers/cp_fair_text_domain.py

Per-run wall-clock on A100-SXM4-40GB: ~1.5 h (frozen BERT keeps the
forward pass cheap; FF training dominates).
"""

import os
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# We import the FF backbone, hybrid block, and Stage-2 attentive head
# from the canonical core / CIFAR-10 modules so this file does not
# duplicate that code. If you reorganize the package, update these
# imports.
from cp_fair_core import RMSNorm, FFHybridBlock  # noqa: F401  (FF block)
# `cp_fair_cifar10` defines `AttentiveHybridHead`, `LabelSmoothingCE`,
# `train_epoch_stage1`, `train_epoch_stage2`, etc. We reuse them.
import cp_fair_cifar10 as ff_cifar10  # noqa: F401


# --- Frozen BERT stem -------------------------------------------------

class BertStem(nn.Module):
    """Frozen `bert-base-uncased` → mean-pooled 768-d → 256-d project."""

    def __init__(self, d_model: int = 256, model_name: str = "bert-base-uncased"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.proj = nn.Linear(self.encoder.config.hidden_size, d_model)

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-pad tokens (matches the README recipe).
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled

    def forward(self, input_ids, attention_mask):
        pooled = self.encode(input_ids, attention_mask)  # (B, 768) under no_grad
        return self.proj(pooled)  # (B, 256), this layer DOES train


# --- Tokenized text dataset ------------------------------------------

class TokenisedTextDataset(Dataset):
    """Lazily tokenizes HuggingFace text rows for the BertStem above."""

    def __init__(self, hf_split, tokenizer, max_length: int = 256):
        self.rows = list(hf_split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row.get("text") or row.get("content") or ""
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": int(row["label"]),
        }


def make_loaders(name: str, tokenizer, batch_size: int = 128, num_workers: int = 4):
    """Returns (train_loader, eval_loader, num_classes) for the named dataset."""
    from datasets import load_dataset, Dataset as HFDataset
    if name == "imdb":
        ds = load_dataset("imdb")
        train_split, eval_split = ds["train"], ds["test"]
        n_classes = 2
    elif name == "agnews":
        ds = load_dataset("ag_news")
        train_split, eval_split = ds["train"], ds["test"]
        n_classes = 4
    elif name == "20news":
        from sklearn.datasets import fetch_20newsgroups
        train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        test = fetch_20newsgroups(subset="test",  remove=("headers", "footers", "quotes"))
        train_split = HFDataset.from_dict({"text": list(train.data),  "label": list(train.target)})
        eval_split  = HFDataset.from_dict({"text": list(test.data),   "label": list(test.target)})
        n_classes = 20
    else:
        raise ValueError(f"Unknown FF_TEXT_DATASET: {name}")

    train_ds = TokenisedTextDataset(train_split, tokenizer)
    eval_ds  = TokenisedTextDataset(eval_split,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, eval_loader, n_classes


# --- Config ----------------------------------------------------------

@dataclass
class TextDomainConfig:
    dataset: str = "imdb"
    num_classes: int = 2  # overwritten by make_loaders
    seed: int = 42
    d_model: int = 256
    num_blocks: int = 4
    epochs_stage1: int = 30
    epochs_stage2: int = 10
    lr: float = 3e-4
    batch_size: int = 128
    max_length: int = 256
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --- Main ------------------------------------------------------------

def main():
    cfg = TextDomainConfig()
    cfg.dataset = os.environ.get("FF_TEXT_DATASET", cfg.dataset)
    cfg.seed    = int(os.environ.get("FF_SEED", cfg.seed))

    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"[ff-text] dataset={cfg.dataset} seed={cfg.seed} "
          f"D={cfg.d_model} L={cfg.num_blocks} "
          f"E1={cfg.epochs_stage1} E2={cfg.epochs_stage2}")

    # 1. Build the frozen-BERT stem and data loaders.
    stem = BertStem(d_model=cfg.d_model).to(cfg.device)
    train_loader, eval_loader, n_classes = make_loaders(
        cfg.dataset, stem.tokenizer, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    cfg.num_classes = n_classes

    # 2. Build the FF backbone (L=4, D=256) on top of the stem.
    #    Reuse the canonical FFHybridBlock from cp_fair_core via the
    #    same wiring as cp_fair_cifar10.main.  See cp_fair_cifar10.py
    #    `Net` for the exact composition; we delegate to it after
    #    monkey-patching the stem.
    net = ff_cifar10.build_net_with_stem(stem=stem, cfg=cfg)
    net = net.to(cfg.device)

    # 3. Stage 1 — block-local FF training (same loop as image trainers).
    print("Stage 1: FF training (block-local)")
    ff_cifar10.run_stage1(net=net, train_loader=train_loader,
                          eval_loader=eval_loader, cfg=cfg)

    # 4. Stage 2 — frozen-feature attentive readout (BP-trained probe).
    print("Stage 2: frozen-feature attentive readout")
    ff_cifar10.run_stage2(net=net, train_loader=train_loader,
                          eval_loader=eval_loader, cfg=cfg)


if __name__ == "__main__":
    main()
