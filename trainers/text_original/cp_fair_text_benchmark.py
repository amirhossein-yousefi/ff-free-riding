
#!/usr/bin/env python3
"""
CP-FAIR text benchmark adapted from the author's Tiny-ImageNet CP-FAIR notebook.

Design choice:
- Keep the CP-FAIR FF blocks, losses, queues, and Stage-2 head intact.
- Move text into the same pipeline through frozen sentence embeddings.
- Convert each sentence embedding into a short sequence of virtual tokens so the
  hybrid attention/MoE blocks still operate on tokens instead of a flat vector.

This mirrors the setup used in prior FF-on-text papers, which relied on frozen
Word2Vec/BERT features rather than training a tokenizer+encoder from scratch.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Missing dependency: datasets\n"
        "Install with: pip install datasets"
    ) from e

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as e:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install with: pip install transformers sentencepiece"
    ) from e

# Released copy: the reported runs imported cp_fair_core_from_user.py, the pre-camera-ready copy of
# ../cp_fair_core.py. The released core adds only no-grad locality guards (the applied updates are
# unchanged), RNG save/restore and resume logging, so this copy imports it instead.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cp_fair_core as core  # noqa: E402


# ----------------------------- helpers ----------------------------- #

STRICT_FF_TARGETS = {
    "imdb": {
        "best_acc": 84.86,
        "name": "IMDb",
        "source": "Extending the Forward Forward Algorithm (2023)",
        "note": "strict FF baseline with Word2Vec features",
    },
    "20newsgroups": {
        "best_acc": 61.634,
        "name": "20 Newsgroups",
        "source": "Cyclic Neural Network / GOMLP FF-Complete (2024)",
        "note": "strict FF baseline with frozen BERT features",
    },
}
NEARBY_FORWARD_ONLY_TARGETS = {
    "ag_news": {
        "best_acc": 91.29,
        "name": "AG News",
        "source": "Forward layer-wise learning through SI maximization (2024)",
        "note": "forward-only / local-learning target, not strict FF",
    },
    "dbpedia_14": {
        "best_acc": 98.77,
        "name": "DBPedia-14",
        "source": "Forward layer-wise learning through SI maximization (2024)",
        "note": "forward-only / local-learning target, not strict FF",
    },
}
ALL_TARGETS = {**STRICT_FF_TARGETS, **NEARBY_FORWARD_ONLY_TARGETS}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def maybe_topk_tuple(num_classes: int, max_k: int = 10) -> Tuple[int, ...]:
    return tuple(range(1, min(num_classes, max_k) + 1))


def infer_max_length(dataset_name: str) -> int:
    if dataset_name == "imdb":
        return 256
    if dataset_name in {"20newsgroups", "20_newsgroups"}:
        return 256
    return 128


# ----------------------------- data loading ----------------------------- #

def load_text_classification_data(dataset_name: str) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str]]:
    """Return train_texts, train_labels, test_texts, test_labels, label_names."""
    dataset_name = dataset_name.lower()

    if dataset_name == "ag_news":
        ds = load_dataset("ag_news")
        train_texts = [clean_text(t) for t in ds["train"]["text"]]
        test_texts = [clean_text(t) for t in ds["test"]["text"]]
        train_labels = np.asarray(ds["train"]["label"], dtype=np.int64)
        test_labels = np.asarray(ds["test"]["label"], dtype=np.int64)
        label_names = list(ds["train"].features["label"].names)
        return train_texts, train_labels, test_texts, test_labels, label_names

    if dataset_name == "imdb":
        ds = load_dataset("imdb")
        train_texts = [clean_text(t) for t in ds["train"]["text"]]
        test_texts = [clean_text(t) for t in ds["test"]["text"]]
        train_labels = np.asarray(ds["train"]["label"], dtype=np.int64)
        test_labels = np.asarray(ds["test"]["label"], dtype=np.int64)
        label_names = list(ds["train"].features["label"].names)
        return train_texts, train_labels, test_texts, test_labels, label_names

    if dataset_name == "dbpedia_14":
        ds = load_dataset("dbpedia_14")
        train_texts = [
            clean_text(f"{title}. {content}")
            for title, content in zip(ds["train"]["title"], ds["train"]["content"])
        ]
        test_texts = [
            clean_text(f"{title}. {content}")
            for title, content in zip(ds["test"]["title"], ds["test"]["content"])
        ]
        train_labels = np.asarray(ds["train"]["label"], dtype=np.int64)
        test_labels = np.asarray(ds["test"]["label"], dtype=np.int64)
        label_names = list(ds["train"].features["label"].names)
        return train_texts, train_labels, test_texts, test_labels, label_names

    if dataset_name in {"20newsgroups", "20_newsgroups"}:
        train = fetch_20newsgroups(
            subset="train",
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=42,
        )
        test = fetch_20newsgroups(
            subset="test",
            remove=("headers", "footers", "quotes"),
            shuffle=False,
        )
        train_texts = [clean_text(t) for t in train.data]
        test_texts = [clean_text(t) for t in test.data]
        train_labels = np.asarray(train.target, dtype=np.int64)
        test_labels = np.asarray(test.target, dtype=np.int64)
        label_names = list(train.target_names)
        return train_texts, train_labels, test_texts, test_labels, label_names

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        f"Choose from imdb, ag_news, dbpedia_14, 20newsgroups."
    )


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    encoder_model: str,
    max_length: int,
    batch_size: int,
    device: str,
    pool: str = "mean",
) -> torch.Tensor:
    """
    Encode texts with a frozen HF model into one vector per example.

    pool:
      - mean: masked mean of last hidden state
      - cls : first token
    """
    tokenizer = AutoTokenizer.from_pretrained(encoder_model, use_fast=True)
    model = AutoModel.from_pretrained(encoder_model).to(device)
    model.eval()

    out_batches: List[torch.Tensor] = []
    iterator = range(0, len(texts), batch_size)
    for start in tqdm(iterator, desc=f"Encoding with {encoder_model}", dynamic_ncols=True):
        batch_texts = list(texts[start : start + batch_size])
        tok = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        outputs = model(**tok)
        hidden = outputs.last_hidden_state  # [B, L, D]
        if pool == "cls":
            emb = hidden[:, 0]
        else:
            mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        emb = F.normalize(emb, dim=-1)
        out_batches.append(emb.cpu())

    return torch.cat(out_batches, dim=0)


def maybe_load_or_build_features(
    cache_path: Path,
    texts: Sequence[str],
    labels: np.ndarray,
    encoder_model: str,
    max_length: int,
    encode_batch_size: int,
    device: str,
    pool: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        x = payload["x"].float()
        y = payload["y"].long()
        return x, y

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    x = encode_texts(
        texts=texts,
        encoder_model=encoder_model,
        max_length=max_length,
        batch_size=encode_batch_size,
        device=device,
        pool=pool,
    ).float()
    y = torch.as_tensor(labels, dtype=torch.long)
    torch.save({"x": x.half(), "y": y}, cache_path)
    return x, y


# ----------------------------- datasets ----------------------------- #

class EmbeddingAugment:
    def __init__(self, dropout_p: float = 0.0, noise_std: float = 0.0, renorm: bool = True):
        self.dropout_p = float(dropout_p)
        self.noise_std = float(noise_std)
        self.renorm = bool(renorm)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        if self.dropout_p > 0.0:
            keep = (torch.rand_like(out) > self.dropout_p).float()
            denom = max(1e-6, 1.0 - self.dropout_p)
            out = out * keep / denom
        if self.noise_std > 0.0:
            out = out + torch.randn_like(out) * self.noise_std
        if self.renorm:
            out = F.normalize(out, dim=-1)
        return out


class EmbeddingTripleDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        aug1: EmbeddingAugment,
        aug2: EmbeddingAugment,
        aug3: EmbeddingAugment,
    ):
        self.x = x.float().contiguous()
        self.y = y.long().contiguous()
        self.aug1 = aug1
        self.aug2 = aug2
        self.aug3 = aug3

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        feat = self.x[idx]
        return self.aug1(feat), self.aug2(feat), self.aug3(feat), int(self.y[idx].item())


class EmbeddingSingleDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x.float().contiguous()
        self.y = y.long().contiguous()

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx].item())


# ----------------------------- model ----------------------------- #

class TextVectorEmbed(nn.Module):
    """
    Project one frozen sentence embedding into a short sequence of virtual tokens.

    This preserves the token-based CP-FAIR blocks while keeping the text
    encoder frozen and cheap to benchmark.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_virtual_tokens: int = 8,
        dropout: float = 0.1,
        hidden_mult: int = 2,
        use_abs_pos_embed: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.num_virtual_tokens = int(num_virtual_tokens)
        hidden = max(self.input_dim, self.d_model * hidden_mult)

        self.norm = nn.LayerNorm(self.input_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_virtual_tokens * self.d_model),
        )
        self.dropout = nn.Dropout(dropout)
        if use_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_virtual_tokens, self.d_model))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)

        for mod in self.proj:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [B, D] embeddings, got shape={tuple(x.shape)}")
        h = self.proj(self.norm(x))
        h = h.view(x.size(0), self.num_virtual_tokens, self.d_model)
        if self.pos_embed is not None:
            h = h + self.pos_embed
        h = self.dropout(h)
        return h


class TextFFHybridNet(nn.Module):
    def __init__(self, cfg: "TextFFConfig"):
        super().__init__()
        self.num_classes = int(cfg.num_classes)
        self.num_blocks = int(cfg.num_blocks)
        self.n_aspects = int(cfg.n_aspects)
        self.use_rope = bool(cfg.use_rope)

        use_abs_pos_embed = not cfg.use_rope
        self.embed = TextVectorEmbed(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            num_virtual_tokens=cfg.num_virtual_tokens,
            dropout=cfg.dropout,
            hidden_mult=cfg.virtual_token_hidden_mult,
            use_abs_pos_embed=use_abs_pos_embed,
        )

        if cfg.use_rope:
            rope_dim = cfg.d_model // cfg.num_heads
            self.rotary_emb = core.RotaryEmbedding(dim=rope_dim, theta=cfg.rope_theta)
        else:
            self.rotary_emb = None

        dprs = torch.linspace(0, cfg.drop_path_rate, steps=cfg.num_blocks).tolist()
        self.blocks = nn.ModuleList(
            [
                core.FFHybridBlock(
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
            ]
        )

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
        vectors: torch.Tensor,
        max_blocks: Optional[int] = None,
        return_all_depths: bool = False,
    ) -> torch.Tensor:
        self.eval()
        bsz = vectors.size(0)
        tokens0 = self.embed(vectors)

        blocks = self.blocks[: int(max_blocks)] if max_blocks is not None else self.blocks
        l_use = len(blocks)

        if return_all_depths:
            scores_depth = torch.zeros(bsz, l_use, self.num_classes, device=vectors.device)

        all_scores: List[torch.Tensor] = []
        for k in range(self.num_classes):
            yk = torch.full((bsz,), k, dtype=torch.long, device=vectors.device)
            tokens = tokens0
            g_hist: List[torch.Tensor] = []
            total_g = torch.zeros(bsz, device=vectors.device)

            for bi, blk in enumerate(blocks):
                g_prev = torch.stack(g_hist, dim=1) if g_hist else None
                if g_prev is not None:
                    g_prev = g_prev.detach()
                a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
                total_g = total_g + blk.aggregate_goodness(g_vec)
                g_hist.append(g_vec)
                tokens = core.l2_normalize_tokens(a)
                if return_all_depths:
                    scores_depth[:, bi, k] = total_g

            if not return_all_depths:
                all_scores.append(total_g.unsqueeze(1))

        if return_all_depths:
            return scores_depth
        return torch.cat(all_scores, dim=1)

    @torch.no_grad()
    def extract_hybrid_features_all_labels(self, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        bsz = vectors.size(0)
        tokens0 = self.embed(vectors)
        feats_g: List[torch.Tensor] = []
        feats_a: List[torch.Tensor] = []

        for k in range(self.num_classes):
            yk = torch.full((bsz,), k, dtype=torch.long, device=vectors.device)
            tokens = tokens0
            g_hist: List[torch.Tensor] = []
            a_embed_hist: List[torch.Tensor] = []

            for blk in self.blocks:
                g_prev = torch.stack(g_hist, dim=1) if g_hist else None
                if g_prev is not None:
                    g_prev = g_prev.detach()
                a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
                g_hist.append(g_vec)
                pooled_a = blk.get_pooled_rep(a)
                a_embed_hist.append(pooled_a)
                tokens = core.l2_normalize_tokens(a)

            feats_g.append(torch.cat(g_hist, dim=1).unsqueeze(1))
            feats_a.append(torch.cat(a_embed_hist, dim=1).unsqueeze(1))

        return torch.cat(feats_g, dim=1), torch.cat(feats_a, dim=1)

    def set_router_tau(self, tau_base: float, depth_slope: float = 0.0) -> None:
        l_tot = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            scale = 1.0 + depth_slope * (i / max(1, l_tot - 1))
            blk.moe.tau = max(1e-6, tau_base * scale)


# ----------------------------- config ----------------------------- #

@dataclass
class TextFFConfig:
    # data / io
    dataset: str = "imdb"
    run_dir: str = "./runs_text"
    cache_dir: str = "./cache_text"
    encoder_model: str = "bert-base-uncased"
    encoder_pool: str = "mean"
    max_length: int = 256
    encode_batch_size: int = 64

    # optimization
    batch_size: int = 256
    test_batch_size: int = 512
    epochs_stage1: int = 40
    epochs_stage2: int = 10
    seed: int = 42
    val_fraction: float = 0.1
    save_predictions: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 2
    symba_alpha: float = 4.0
    lr: float = 1e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = True
    use_mem: bool = False
    warmup_epochs: int = 5

    # model
    num_classes: int = 2
    input_dim: int = 768
    d_model: int = 256
    num_virtual_tokens: int = 8
    virtual_token_hidden_mult: int = 2
    num_blocks: int = 4
    num_heads: int = 8
    n_experts: int = 16
    mlp_mult: int = 4
    dropout: float = 0.10
    proj_dim: int = 256
    label_embed_dim: int = 64
    moe_top_k: int = 4
    use_rope: bool = True
    rope_theta: float = 10000.0
    drop_path_rate: float = 0.10

    # losses / FF specifics
    n_aspects: int = 4
    aspect_loss_types: List[str] = field(default_factory=list)
    aspect_lambdas: List[float] = field(default_factory=list)
    aspect_theta_inits: List[float] = field(default_factory=list)
    block_lambda: float = 1.0
    block_curr_lambda: float = 0.25
    block_curr_depth_slope: float = 2.0
    block_theta_init: float = 6.0
    use_softmax_weights: bool = True
    moe_balance_coef: float = 0.01
    moe_zloss_coef: float = 1e-3
    moe_balance_type: str = "switch"
    moe_zloss_type: str = "stmoe"
    moe_aux_ramp_epochs: int = 3
    moe_aux_depth_scale: float = 0.0
    proto_temp: float = 0.1
    sff_ni_weight: float = 0.5
    hard_negative_k_first: int = 3
    hard_negative_k_last: int = 8
    queue_size: int = 8192
    queue_neg_samples: int = 1024
    gamma_scale: float = 0.0
    depth_order_lambda: float = 0.0
    depth_margin_pos: float = 0.05
    depth_margin_neg: float = 0.05
    recon_lambda: float = 0.02
    recon_mult: int = 2
    gate_tau_start: float = 2.0
    gate_tau_end: float = 0.75
    gate_tau_depth_slope: float = 0.2
    use_ema: bool = True
    ema_decay: float = 0.999
    use_ema_hnm: bool = True
    eval_tta: bool = False

    contrastive_temp: float = 0.2
    contrastive_lambda: float = 0.25

    # stage-2
    head_hidden: int = 1024
    head_dropout: float = 0.1
    stage2_lr: float = 1e-3
    stage2_wd: float = 0.0
    label_smoothing: float = 0.0
    head_num_heads: int = 4

    # embedding augmentations
    aug_dropout_1: float = 0.05
    aug_dropout_2: float = 0.10
    aug_dropout_3: float = 0.15
    aug_noise_1: float = 0.01
    aug_noise_2: float = 0.015
    aug_noise_3: float = 0.02

    # eval
    eval_freq_s1: int = 1
    eval_freq_s2: int = 1

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.aspect_loss_types = ["margin", "rank", "rank", "margin"][: self.n_aspects]
        self.aspect_lambdas = [1.5, 0.2, 0.3, 0.3][: self.n_aspects]
        self.aspect_theta_inits = [5.0, 2.5, 0.5, 1.0][: self.n_aspects]


def dataset_default_overrides(dataset_name: str) -> Dict[str, object]:
    name = dataset_name.lower()
    if name == "imdb":
        return {
            "epochs_stage1": 40,
            "epochs_stage2": 10,
            "max_length": 256,
            "batch_size": 256,
            "test_batch_size": 512,
            "hard_negative_k_first": 1,
            "hard_negative_k_last": 1,
            "dropout": 0.10,
            "head_hidden": 1024,
        }
    if name == "ag_news":
        return {
            "epochs_stage1": 25,
            "epochs_stage2": 8,
            "max_length": 128,
            "batch_size": 384,
            "test_batch_size": 768,
            "hard_negative_k_first": 2,
            "hard_negative_k_last": 3,
            "dropout": 0.08,
            "head_hidden": 768,
        }
    if name == "dbpedia_14":
        return {
            "epochs_stage1": 20,
            "epochs_stage2": 6,
            "max_length": 128,
            "batch_size": 512,
            "test_batch_size": 1024,
            "hard_negative_k_first": 4,
            "hard_negative_k_last": 8,
            "dropout": 0.08,
            "head_hidden": 768,
        }
    if name in {"20newsgroups", "20_newsgroups"}:
        return {
            "epochs_stage1": 35,
            "epochs_stage2": 10,
            "max_length": 256,
            "batch_size": 256,
            "test_batch_size": 512,
            "hard_negative_k_first": 5,
            "hard_negative_k_last": 10,
            "dropout": 0.10,
            "head_hidden": 1024,
        }
    return {}


# ----------------------------- training ----------------------------- #

def router_tau_at(epoch: int, total_epochs: int, tau_start: float, tau_end: float) -> float:
    t = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
    return tau_start + t * (tau_end - tau_start)


def build_stage1_loaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    cfg: TextFFConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    aug1 = EmbeddingAugment(cfg.aug_dropout_1, cfg.aug_noise_1, renorm=True)
    aug2 = EmbeddingAugment(cfg.aug_dropout_2, cfg.aug_noise_2, renorm=True)
    aug3 = EmbeddingAugment(cfg.aug_dropout_3, cfg.aug_noise_3, renorm=True)

    train_ds = EmbeddingTripleDataset(x_train, y_train, aug1, aug2, aug3)
    val_ds = EmbeddingSingleDataset(x_val, y_val)
    test_ds = EmbeddingSingleDataset(x_test, y_test)

    persistent = cfg.workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader


def build_stage2_loader(x: torch.Tensor, y: torch.Tensor, cfg: TextFFConfig, shuffle: bool = True) -> DataLoader:
    persistent = cfg.workers > 0
    ds = EmbeddingSingleDataset(x, y)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size * 2,
        shuffle=shuffle,
        num_workers=cfg.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )


def build_backbone_and_optim(cfg: TextFFConfig):
    device = cfg.device
    net = TextFFHybridNet(cfg).to(device)
    net_ema = deepcopy(net).to(device) if cfg.use_ema else None
    if net_ema is not None:
        for p in net_ema.parameters():
            p.requires_grad_(False)

    opts: List[core.SAM] = []
    scheds: List[Optional[torch.optim.lr_scheduler._LRScheduler]] = []
    for li, block in enumerate(net.blocks):
        params = list(block.parameters())
        if li == 0:
            params += list(net.embed.parameters())

        opt = core.SAM(
            params,
            torch.optim.AdamW,
            rho=cfg.sam_rho,
            adaptive=cfg.sam_adaptive,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        opts.append(opt)

        if opt.base_optimizer is None:
            scheds.append(None)
        else:
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt.base_optimizer,
                lr_lambda=lambda e, total=cfg.epochs_stage1, w=cfg.warmup_epochs: core.cosine_warmup_lambda(e, total, w),
            )
            scheds.append(sched)

    queues = [core.ContrastiveQueue(cfg.queue_size, cfg.proj_dim, device=device) for _ in range(cfg.num_blocks)]
    return net, net_ema, opts, scheds, queues


def train_stage2(
    backbone: TextFFHybridNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TextFFConfig,
    run_dir: Path,
) -> Tuple[core.AttentiveHybridHead, Dict[str, float]]:
    device = cfg.device
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    head_in_dim_g = cfg.num_blocks * cfg.n_aspects
    head_in_dim_a = cfg.d_model * cfg.num_blocks
    head = core.AttentiveHybridHead(
        in_dim_g=head_in_dim_g,
        in_dim_a=head_in_dim_a,
        hidden=cfg.head_hidden,
        num_classes=cfg.num_classes,
        num_heads=cfg.head_num_heads,
        dropout=cfg.head_dropout,
    ).to(device)

    criterion = core.LabelSmoothingCE(eps=cfg.label_smoothing, num_classes=cfg.num_classes)
    opt_head = torch.optim.AdamW(head.parameters(), lr=cfg.stage2_lr, weight_decay=cfg.stage2_wd)
    sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=cfg.epochs_stage2)

    best_state = None
    best_val = 0.0
    history: List[Dict[str, float]] = []
    topks = maybe_topk_tuple(cfg.num_classes)

    pbar_epochs = tqdm(range(1, cfg.epochs_stage2 + 1), desc="Stage-2 epochs", dynamic_ncols=True)
    for epoch in pbar_epochs:
        head.train()
        epoch_loss = 0.0

        pbar_batches = tqdm(train_loader, desc=f"Stage-2 {epoch}", leave=False, dynamic_ncols=True)
        for vectors, labels in pbar_batches:
            vectors = vectors.to(device)
            labels = labels.to(device)
            bsz = vectors.size(0)
            c = cfg.num_classes

            with torch.no_grad():
                feats_g, feats_a = backbone.extract_hybrid_features_all_labels(vectors)

            logits = head(feats_g.view(bsz * c, -1), feats_a.view(bsz * c, -1))
            loss = criterion(logits, labels)

            opt_head.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
            opt_head.step()

            epoch_loss += float(loss.item())
            pbar_batches.set_postfix(loss=f"{loss.item():.4f}")

        sched_head.step()
        avg_loss = epoch_loss / max(1, len(train_loader))

        rec: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(avg_loss),
            "lr": float(sched_head.get_last_lr()[0]),
        }
        if epoch % cfg.eval_freq_s2 == 0 or epoch == cfg.epochs_stage2:
            val_accs = core.evaluate_stage2(backbone, head, val_loader, device, tta=False, topk=topks)
            rec.update({f"val_top{k}": float(v) for k, v in val_accs.items()})
            if val_accs[1] > best_val:
                best_val = float(val_accs[1])
                best_state = deepcopy(head.state_dict())
                torch.save(
                    {"head": best_state, "best_val_top1": best_val, "cfg": asdict(cfg)},
                    run_dir / "stage2_best.pt",
                )
        history.append(rec)
        pbar_epochs.set_postfix(loss=f"{avg_loss:.4f}", best=f"{best_val * 100:.2f}%")

    if best_state is not None:
        head.load_state_dict(best_state)

    test_accs = core.evaluate_stage2(backbone, head, test_loader, device, tta=False, topk=topks)
    metrics = {
        "stage2_best_val_top1": float(best_val),
        **{f"stage2_test_top{k}": float(v) for k, v in test_accs.items()},
    }

    with open(run_dir / "stage2_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return head, metrics


# ----------------------------- main ----------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CP-FAIR on text classification")
    p.add_argument("--dataset", type=str, default="imdb", choices=["imdb", "ag_news", "dbpedia_14", "20newsgroups"])
    p.add_argument("--encoder_model", type=str, default="bert-base-uncased")
    p.add_argument("--encoder_pool", type=str, default="mean", choices=["mean", "cls"])
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--test_batch_size", type=int, default=None)
    p.add_argument("--epochs_stage1", type=int, default=None)
    p.add_argument("--epochs_stage2", type=int, default=None)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--n_experts", type=int, default=16)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--num_virtual_tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--run_root", type=str, default="./runs_text")
    p.add_argument("--cache_dir", type=str, default="./cache_text")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stage2_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--gamma_scale", type=float, default=0.0)
    p.add_argument("--depth_order_lambda", type=float, default=0.0)
    p.add_argument("--use_sam", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train_limit", type=int, default=None, help="debug/smoke-test only")
    p.add_argument("--test_limit", type=int, default=None, help="debug/smoke-test only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    ds_defaults = dataset_default_overrides(args.dataset)
    cfg = TextFFConfig(
        dataset=args.dataset,
        encoder_model=args.encoder_model,
        encoder_pool=args.encoder_pool,
        max_length=(args.max_length if args.max_length is not None else int(ds_defaults.get("max_length", infer_max_length(args.dataset)))),
        encode_batch_size=args.encode_batch_size,
        batch_size=(args.batch_size if args.batch_size is not None else int(ds_defaults.get("batch_size", 256))),
        test_batch_size=(args.test_batch_size if args.test_batch_size is not None else int(ds_defaults.get("test_batch_size", 512))),
        epochs_stage1=(args.epochs_stage1 if args.epochs_stage1 is not None else int(ds_defaults.get("epochs_stage1", 30))),
        epochs_stage2=(args.epochs_stage2 if args.epochs_stage2 is not None else int(ds_defaults.get("epochs_stage2", 10))),
        seed=args.seed,
        workers=args.workers,
        lr=args.lr,
        stage2_lr=args.stage2_lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        n_experts=args.n_experts,
        proj_dim=args.proj_dim,
        num_virtual_tokens=args.num_virtual_tokens,
        gamma_scale=args.gamma_scale,
        depth_order_lambda=args.depth_order_lambda,
        use_sam=args.use_sam,
        device=args.device,
        cache_dir=args.cache_dir,
        run_dir=args.run_root,
        val_fraction=args.val_fraction,
        hard_negative_k_first=int(ds_defaults.get("hard_negative_k_first", 3)),
        hard_negative_k_last=int(ds_defaults.get("hard_negative_k_last", 8)),
        dropout=float(ds_defaults.get("dropout", 0.10)),
        head_hidden=int(ds_defaults.get("head_hidden", 1024)),
    )

    run_root = Path(cfg.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{slugify(cfg.dataset)}_{slugify(cfg.encoder_model)}_"
        f"L{cfg.num_blocks}_D{cfg.d_model}_V{cfg.num_virtual_tokens}_seed{cfg.seed}"
    )
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Device: {cfg.device}")
    if cfg.dataset in ALL_TARGETS:
        target = ALL_TARGETS[cfg.dataset]
        print(
            f"Published target for {target['name']}: {target['best_acc']:.3f}% "
            f"({target['source']}; {target['note']})"
        )

    train_texts, train_labels_np, test_texts, test_labels_np, label_names = load_text_classification_data(cfg.dataset)

    if args.train_limit is not None:
        n = min(len(train_texts), int(args.train_limit))
        train_texts = train_texts[:n]
        train_labels_np = train_labels_np[:n]
    if args.test_limit is not None:
        n = min(len(test_texts), int(args.test_limit))
        test_texts = test_texts[:n]
        test_labels_np = test_labels_np[:n]

    cfg.num_classes = int(len(set(train_labels_np.tolist())))
    cfg.hard_negative_k_first = max(1, min(cfg.hard_negative_k_first, cfg.num_classes - 1))
    cfg.hard_negative_k_last = max(1, min(cfg.hard_negative_k_last, cfg.num_classes - 1))

    train_idx, val_idx = train_test_split(
        np.arange(len(train_texts)),
        test_size=cfg.val_fraction,
        random_state=cfg.seed,
        stratify=train_labels_np,
    )

    cache_root = Path(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    enc_slug = slugify(cfg.encoder_model)
    limit_stub = ""
    if args.train_limit is not None:
        limit_stub += f"_trlim{int(args.train_limit)}"
    if args.test_limit is not None:
        limit_stub += f"_telim{int(args.test_limit)}"
    base_stub = f"{slugify(cfg.dataset)}_{enc_slug}_max{cfg.max_length}_{cfg.encoder_pool}{limit_stub}"

    # Cache full train/test embeddings once, then derive seed-specific train/val splits
    # by indexing into the full training tensor. This avoids split leakage across seeds
    # and makes repeated multi-seed runs much faster.
    x_train_all, y_train_all = maybe_load_or_build_features(
        cache_root / f"{base_stub}_train_all.pt",
        train_texts,
        train_labels_np,
        cfg.encoder_model,
        cfg.max_length,
        cfg.encode_batch_size,
        cfg.device,
        cfg.encoder_pool,
    )
    x_test, y_test = maybe_load_or_build_features(
        cache_root / f"{base_stub}_test.pt",
        test_texts,
        test_labels_np,
        cfg.encoder_model,
        cfg.max_length,
        cfg.encode_batch_size,
        cfg.device,
        cfg.encoder_pool,
    )

    x_train = x_train_all[train_idx].clone()
    y_train = y_train_all[train_idx].clone()
    x_val = x_train_all[val_idx].clone()
    y_val = y_train_all[val_idx].clone()

    cfg.input_dim = int(x_train_all.size(1))

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(run_dir / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2, ensure_ascii=False)
    with open(run_dir / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_texts),
            },
            f,
            indent=2,
        )

    train_loader_s1, val_loader, test_loader = build_stage1_loaders(x_train, y_train, x_val, y_val, x_test, y_test, cfg)
    train_loader_s2 = build_stage2_loader(x_train, y_train, cfg, shuffle=True)

    net, net_ema, opts, scheds, queues = build_backbone_and_optim(cfg)
    device = cfg.device

    best_s1 = 0.0
    best_state = None
    topks = maybe_topk_tuple(cfg.num_classes)
    s1_history: List[Dict[str, float]] = []

    epoch_iter = tqdm(range(1, cfg.epochs_stage1 + 1), desc="Stage-1 epochs", dynamic_ncols=True)
    for epoch in epoch_iter:
        tau = router_tau_at(epoch - 1, cfg.epochs_stage1, cfg.gate_tau_start, cfg.gate_tau_end)
        net.set_router_tau(tau, cfg.gate_tau_depth_slope)
        if net_ema is not None:
            net_ema.set_router_tau(tau, cfg.gate_tau_depth_slope)

        layer_avgs = core.train_epoch_stage1(
            net=net,
            net_ema=net_ema,
            opts=opts,
            loader=train_loader_s1,
            cfg=cfg,
            queues=queues,
            epoch=epoch,
        )
        for sched in scheds:
            if sched is not None:
                sched.step()

        rec: Dict[str, float] = {
            "epoch": float(epoch),
            "tau": float(tau),
        }
        if epoch % cfg.eval_freq_s1 == 0 or epoch == cfg.epochs_stage1:
            backbone_eval = net_ema if net_ema is not None else net
            val_accs = core.evaluate_ff(backbone_eval, val_loader, device, tta=False, topk=topks)
            rec.update({f"val_top{k}": float(v) for k, v in val_accs.items()})
            if val_accs[1] > best_s1:
                best_s1 = float(val_accs[1])
                best_state = {
                    "model": deepcopy(net.state_dict()),
                    "ema": deepcopy(net_ema.state_dict()) if net_ema is not None else None,
                    "best_val_top1": best_s1,
                    "cfg": asdict(cfg),
                }
                torch.save(best_state, run_dir / "stage1_best.pt")

        # Optional lightweight per-layer summary
        for li, met in enumerate(layer_avgs):
            # save only scalar-ish fields
            layer_rec = {"epoch": epoch, "layer": li}
            for k, v in met.items():
                if isinstance(v, (float, int)):
                    layer_rec[k] = float(v)
            with open(run_dir / f"stage1_layer_{li}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(layer_rec) + "\n")

        s1_history.append(rec)
        epoch_iter.set_postfix(best=f"{best_s1 * 100:.2f}%")

    with open(run_dir / "stage1_history.json", "w", encoding="utf-8") as f:
        json.dump(s1_history, f, indent=2)

    if best_state is not None:
        net.load_state_dict(best_state["model"])
        if net_ema is not None and best_state["ema"] is not None:
            net_ema.load_state_dict(best_state["ema"])

    backbone = net_ema if net_ema is not None else net
    val_ff = core.evaluate_ff(backbone, val_loader, device, tta=False, topk=topks)
    test_ff = core.evaluate_ff(backbone, test_loader, device, tta=False, topk=topks)

    ff_metrics = {
        "stage1_best_val_top1": float(best_s1),
        **{f"stage1_val_top{k}": float(v) for k, v in val_ff.items()},
        **{f"stage1_test_top{k}": float(v) for k, v in test_ff.items()},
    }

    _, stage2_metrics = train_stage2(backbone, train_loader_s2, val_loader, test_loader, cfg, run_dir)

    metrics = {
        "dataset": cfg.dataset,
        "encoder_model": cfg.encoder_model,
        "encoder_pool": cfg.encoder_pool,
        "num_classes": cfg.num_classes,
        "label_names": label_names,
        **ff_metrics,
        **stage2_metrics,
    }

    if cfg.dataset in ALL_TARGETS:
        target_acc = ALL_TARGETS[cfg.dataset]["best_acc"] / 100.0
        metrics["published_target_top1"] = float(target_acc)
        metrics["stage1_delta_vs_target_pp"] = float((metrics["stage1_test_top1"] - target_acc) * 100.0)
        if "stage2_test_top1" in metrics:
            metrics["stage2_delta_vs_target_pp"] = float((metrics["stage2_test_top1"] - target_acc) * 100.0)

    with open(run_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 88)
    print(f"Dataset:       {cfg.dataset}")
    print(f"Encoder:       {cfg.encoder_model} ({cfg.encoder_pool} pooling)")
    print(f"Classes:       {cfg.num_classes}")
    print("-" * 88)
    print(f"Stage-1 FF     Top-1 test: {metrics['stage1_test_top1'] * 100:.2f}%")
    if "stage2_test_top1" in metrics:
        print(f"Stage-2 Head   Top-1 test: {metrics['stage2_test_top1'] * 100:.2f}%")
    if cfg.dataset in ALL_TARGETS:
        target = ALL_TARGETS[cfg.dataset]
        print(f"Published tgt  Top-1 test: {target['best_acc']:.2f}%   [{target['source']}]")
        if "stage2_delta_vs_target_pp" in metrics:
            print(f"Delta vs tgt   Stage-2:     {metrics['stage2_delta_vs_target_pp']:+.2f} pp")
        print(f"Delta vs tgt   Stage-1:     {metrics['stage1_delta_vs_target_pp']:+.2f} pp")
    print("=" * 88)
    print(f"Saved metrics: {run_dir / 'final_metrics.json'}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    main()
