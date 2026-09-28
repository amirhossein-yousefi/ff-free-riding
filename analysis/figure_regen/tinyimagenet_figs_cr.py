#!/usr/bin/env python3
"""Regenerate the three Tiny ImageNet figures of App. H.7 (CPU only, matplotlib).

Same plotting code and style as the project's tiny_imagenet_fair/analyze_and_plot.py
(plot_dynamics, plot_perseed, plot_crossdata), with these changes:
  * the Stage-1 convergence curve concatenates every logged segment of each resumed run
    (the original script read only the last segment, so seeds 42/123 showed from epoch
    ~332 and seed 456 from epoch 209);
  * the per-seed chart is titled "validation accuracy" (Tiny ImageNet reports the
    labeled validation split; the trainer's field names say "test");
  * [CR2] the convergence curve starts at the first evaluation (epoch 20), the per-seed plot is a
    dot plot with mean +- sample std, and plot_crossdata recomputes the Tiny ImageNet points from
    the epoch-362 layer_metrics rows of each seed's final segment.
Figs. 17, 18 and 19 of the camera-ready paper; see analysis/figure_regen/README.md.

Inputs (read-only; FF_TINY_DIR, default $FF_HOME/amir-porjects/tiny_imagenet_fair):
  final segments   <FF_TINY_DIR>/{from_drive_1/extracted/...,extracted_seeds/...}
  earlier segments <FF_TINY_DIR>/from_drive_2/ff_tiny_imagenet_multiseed-20260415T151058Z-3-002.zip
Usage: FF_HOME=... python tinyimagenet_figs_cr.py OUT_DIR
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# [CR] paths come from the environment (analysis/README.md)
ROOT = Path(os.environ.get("FF_TINY_DIR", Path(os.environ.get("FF_HOME", "")) / "amir-porjects" / "tiny_imagenet_fair"))
ZIP = ROOT / "from_drive_2/ff_tiny_imagenet_multiseed-20260415T151058Z-3-002.zip"
RUN = "ff_tiny_imagenet_tiny_imagenet_cp_fair_seed{s}_L4_D256_bs256_seed{s}_{ts}"

# (source, run timestamp) per segment, in training order; the last segment holds final_test_eval
SEGMENTS = {
    42: [("disk:from_drive_1/extracted/ff_tiny_imagenet_training/runs_seed42", "20260411_081714"),
         ("disk:from_drive_1/extracted/ff_tiny_imagenet_training/runs_seed42", "20260412_104050")],
    123: [("zip:ff_tiny_imagenet_multiseed/runs_seed123", "20260412_200131"),
          ("disk:extracted_seeds/ff_tiny_imagenet_multiseed/runs_seed123", "20260413_205048")],
    456: [("zip:ff_tiny_imagenet_multiseed/runs_seed456", "20260414_060008"),
          ("disk:extracted_seeds/ff_tiny_imagenet_multiseed/runs_seed456", "20260414_212958")],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "mathtext.fontset": "dejavuserif",  # [CR2] serif math like the text
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})
COLOR_SEEDS = ["#1B9E77", "#D95F02", "#7570B3"]


def _read(seed: int, src: str, ts: str, name: str) -> list[dict]:
    kind, base = src.split(":", 1)
    rel = f"{base}/{RUN.format(s=seed, ts=ts)}/{name}"
    if kind == "zip":
        with zipfile.ZipFile(ZIP) as z:
            text = z.read(rel).decode()
    else:
        text = (ROOT / rel).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_curve(seed: int):
    """Best-so-far Stage-1 top-1 per epoch over all segments (later segment wins).

    best_s1 is the flip-TTA full-depth accuracy on the trainer's 10k held-out subset of the
    100k training images (split_indices.json), evaluated every 20 epochs and at the last epoch.
    [CR2] Epochs before the first evaluation (best_s1 = 0.0 placeholder) are dropped.
    """
    by_epoch: dict[int, float] = {}
    first_eval = None
    for src, ts in SEGMENTS[seed]:
        for d in _read(seed, src, ts, "epoch_metrics.jsonl"):
            if d.get("stage") == 1 and d.get("best_s1") is not None:
                by_epoch[int(d["epoch"])] = float(d["best_s1"])
            if d.get("stage") == 1 and d.get("eval_top1") is not None:
                e = int(d["epoch"])
                first_eval = e if first_eval is None else min(first_eval, e)
    ep = np.array(sorted(e for e in by_epoch if e >= first_eval))
    return ep, np.array([by_epoch[e] for e in ep])


def load_final_eval(seed: int) -> dict:
    src, ts = SEGMENTS[seed][-1]
    for ev in _read(seed, src, ts, "events.jsonl"):
        if ev.get("event") == "final_test_eval":
            return ev
    raise RuntimeError(f"no final_test_eval for seed {seed}")


def plot_dynamics(out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)  # [CR2] larger type at print size
    curves, max_ep, info = [], 0, {}
    for i, seed in enumerate(SEGMENTS):
        ep, acc = load_curve(seed)
        curves.append((ep, acc))
        max_ep = max(max_ep, int(ep.max()))
        info[seed] = {"first_epoch": int(ep.min()), "last_epoch": int(ep.max()), "n_epochs": int(len(ep)),
                      "final_best_s1": float(acc[-1])}
        ax.plot(ep, acc * 100.0, color=COLOR_SEEDS[i], alpha=0.55, linewidth=1.2, label=f"seed {seed}")
    common = np.arange(min(int(e.min()) for e, _ in curves), max_ep + 1)
    arr = np.vstack([np.interp(common, ep, acc, left=np.nan, right=np.nan) for ep, acc in curves]) * 100.0
    mask = ~np.isnan(arr).any(axis=0)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    ax.plot(common[mask], mean[mask], color="black", linewidth=2.0, label="mean (3 seeds)")
    ax.fill_between(common[mask], mean[mask] - std[mask], mean[mask] + std[mask], color="black", alpha=0.12)
    ax.set_xlabel("Stage-1 epoch")
    ax.set_ylabel("Best held-out top-1 (%)")  # [CR2] held-out part of the training images
    ax.set_title("Tiny ImageNet Stage-1 convergence")
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=8)
    ax.set_xlim(0, max_ep)
    ax.set_ylim(bottom=0)
    fig.savefig(out / "fig_tinyimagenet_dynamics.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out / "fig_tinyimagenet_dynamics.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    info["mean_band_epochs"] = [int(common[mask].min()), int(common[mask].max())]
    info["max_std_pp_after_epoch_100"] = float(np.nanmax(std[mask & (common >= 100)]))
    return info


def plot_perseed(out: Path) -> dict:
    metrics = [("stage1_test_top1_no_tta", "S1 no-TTA"), ("stage1_test_top1_tta", "S1 TTA"),
               ("stage2_test_top1_no_tta", "S2 no-TTA"), ("stage2_test_top1_tta", "S2 TTA")]
    per_seed = {s: load_final_eval(s) for s in SEGMENTS}
    # [CR2] dot plot (no truncated bars); seeds side by side, mean +- sample std in its own slot
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    x = np.arange(len(metrics))
    offs = [-0.27, -0.12, 0.03]
    for i, seed in enumerate(per_seed):
        vals = [per_seed[seed][m] * 100.0 for m, _ in metrics]
        ax.plot(x + offs[i], vals, "o", color=COLOR_SEEDS[i], markersize=5, label=f"seed {seed}",
                markeredgecolor="white", markeredgewidth=0.6, linestyle="none")
    agg = {}
    for j, (m, _) in enumerate(metrics):
        vals = np.array([per_seed[s][m] for s in per_seed]) * 100.0
        agg[m] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
        ax.errorbar(j + 0.22, vals.mean(), yerr=vals.std(ddof=1), fmt="D", color="black", markersize=4,
                    capsize=3, zorder=3, label="mean ± std" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metrics], fontsize=8)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Tiny ImageNet per-seed validation accuracy", fontsize=10)
    ax.set_ylim(48, 53.5)
    ax.legend(loc="upper left", frameon=False, ncol=1, fontsize=8)
    fig.savefig(out / "fig_tinyimagenet_perseed.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out / "fig_tinyimagenet_perseed.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return {"per_seed": {str(s): {m: per_seed[s][m] for m, _ in metrics} for s in per_seed}, "agg": agg}


# CIFAR-10 gamma=0 reference (seed 42, epoch 362; paper Tab. 30), as in analyze_and_plot.py
CIFAR10_SEP_CURR_NL = {0: 2.302, 1: 2.886, 2: 3.175, 3: 3.276}
CIFAR10_GPOS_CUR = {0: 2.58, 1: 2.23, 2: 2.29, 3: 2.33}
COLOR_CIFAR = "#2E5BBA"
COLOR_TIN = "#D95F02"


def plot_crossdata(out: Path) -> dict:
    """[CR2] analyze_and_plot.py plot_crossdata: Tiny ImageNet per-block mean +- sample std over the
    three seeds at the last Stage-1 epoch (362) of each run's final segment, vs. the CIFAR-10 reference.
    The submitted figure's Block-0 Tiny ImageNet points did not match these logs."""
    per = {m: {i: [] for i in range(4)} for m in ("sep_curr_nl", "gpos_cur")}
    for seed in SEGMENTS:
        src, ts = SEGMENTS[seed][-1]
        rows = [d for d in _read(seed, src, ts, "layer_metrics.jsonl") if d.get("stage") == 1 and d.get("epoch") == 362]
        assert len(rows) == 4, (seed, len(rows))
        for d in rows:
            for m in per:
                per[m][d["layer"]].append(float(d[m]))
    summ = {m: {"mean": [float(np.mean(per[m][i])) for i in range(4)],
                "std": [float(np.std(per[m][i], ddof=1)) for i in range(4)]} for m in per}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    layers = np.arange(4)
    panels = [("sep_curr_nl", CIFAR10_SEP_CURR_NL, r"$\mathrm{sep}^{\mathrm{cur}}_{\mathrm{nl}}$", "(a) Current-block separation"),
              ("gpos_cur", CIFAR10_GPOS_CUR, r"$g^{+}_{\mathrm{cur}}$", "(b) Current-block positive goodness")]
    for ax, (m, ref, ylab, title) in zip(axes, panels):
        ax.plot(layers, [ref[i] for i in range(4)], marker="o", color=COLOR_CIFAR,
                label="CIFAR-10 ($\\gamma{=}0$, seed 42)", linewidth=2, markersize=6)
        mean, std = np.asarray(summ[m]["mean"]), np.asarray(summ[m]["std"])
        ax.errorbar(layers, mean, yerr=std, marker="s", color=COLOR_TIN,
                    label="Tiny ImageNet ($\\gamma{=}0$, mean of 3 seeds)", linewidth=2, markersize=6, capsize=3)
        ax.fill_between(layers, mean - std, mean + std, color=COLOR_TIN, alpha=0.15)
        ax.set_xlabel("Block")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.set_ylim(bottom=0)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.savefig(out / "fig_tinyimagenet_crossdata.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out / "fig_tinyimagenet_crossdata.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return summ


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    summary = {"dynamics": plot_dynamics(out), "perseed": plot_perseed(out), "crossdata": plot_crossdata(out)}
    (out / "regen_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
