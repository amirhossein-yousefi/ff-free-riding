#!/usr/bin/env python3
"""
Launch multi-seed CP-FAIR text benchmarks and optionally export artifacts to Google Drive.

Default grid:
  datasets = imdb,20newsgroups,ag_news
  seeds    = 42,43,44

Recommended Colab usage:
  1) Mount Google Drive in a notebook cell.
  2) Run this launcher with --drive_dir /content/drive/MyDrive/<your-folder>
  3) Keep --drive_mode archive to reduce small-file I/O on mounted Drive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence


STRICT_FF_TARGETS = {
    "imdb": {
        "best_acc": 84.86,
        "name": "IMDb",
        "source": "Extending the Forward Forward Algorithm (2023)",
        "note": "strict FF baseline with Word2Vec features",
    },
    # FF-Complete reports 38.36% error on NewsGroup, i.e. about 61.64% accuracy.
    "20newsgroups": {
        "best_acc": 61.64,
        "name": "20 Newsgroups",
        "source": "Cyclic Neural Network / FF-Complete (2024)",
        "note": "strict FF baseline inferred from reported 38.36% error",
    },
}

NEARBY_FORWARD_ONLY_TARGETS = {
    # Best AG News error reported is 8.71% on VD-CNN depth 17, i.e. 91.29% accuracy.
    "ag_news": {
        "best_acc": 91.29,
        "name": "AG News",
        "source": "Forward layer-wise learning through SI maximization (2024)",
        "note": "forward-only / local-learning target, not strict FF",
    },
    # Best DBPedia error reported is 1.23% on VD-CNN depth 29, i.e. 98.77% accuracy.
    "dbpedia_14": {
        "best_acc": 98.77,
        "name": "DBPedia-14",
        "source": "Forward layer-wise learning through SI maximization (2024)",
        "note": "forward-only / local-learning target, not strict FF",
    },
}

ALL_TARGETS = {**STRICT_FF_TARGETS, **NEARBY_FORWARD_ONLY_TARGETS}

DEFAULT_DATASETS = ["imdb", "20newsgroups", "ag_news"]
DEFAULT_SEEDS = [42, 43, 44]


def slugify(text: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in parse_csv_list(raw):
        values.append(int(item))
    return values


def expected_run_name(
    dataset: str,
    encoder_model: str,
    num_blocks: int,
    d_model: int,
    num_virtual_tokens: int,
    seed: int,
) -> str:
    return (
        f"{slugify(dataset)}_{slugify(encoder_model)}_"
        f"L{int(num_blocks)}_D{int(d_model)}_V{int(num_virtual_tokens)}_seed{int(seed)}"
    )


def maybe_mount_drive(mount_point: str = "/content/drive") -> bool:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return False

    mount_path = Path(mount_point)
    mount_path.mkdir(parents=True, exist_ok=True)
    if (mount_path / "MyDrive").exists():
        return True

    drive.mount(str(mount_path))
    return (mount_path / "MyDrive").exists()


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_directory(src_dir: Path, archive_dir: Path) -> Path:
    mkdir(archive_dir)
    archive_path = archive_dir / f"{src_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    return archive_path


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    mkdir(dst.parent)
    shutil.copy2(src, dst)


def sync_run_artifacts(
    run_dir: Path,
    drive_dir: Path,
    drive_mode: str,
    local_archive_dir: Path,
) -> Dict[str, str]:
    synced: Dict[str, str] = {}
    mkdir(drive_dir)

    if drive_mode in {"archive", "both"}:
        archive_path = archive_directory(run_dir, local_archive_dir)
        dst_archive = drive_dir / "archives" / archive_path.name
        copy_file_if_exists(archive_path, dst_archive)
        synced["drive_archive"] = str(dst_archive)

    if drive_mode in {"mirror", "both"}:
        dst_run_dir = drive_dir / "runs" / run_dir.name
        mkdir(dst_run_dir.parent)
        shutil.copytree(run_dir, dst_run_dir, dirs_exist_ok=True)
        synced["drive_run_dir"] = str(dst_run_dir)

    metric_file = run_dir / "final_metrics.json"
    if metric_file.exists():
        dst_metric = drive_dir / "metrics" / f"{run_dir.name}_final_metrics.json"
        copy_file_if_exists(metric_file, dst_metric)
        synced["drive_metric"] = str(dst_metric)

    return synced


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def pct_from_unit(x: Any) -> Optional[float]:
    val = safe_float(x)
    if val is None:
        return None
    return val * 100.0


def target_pct_for_dataset(dataset: str, metrics: Optional[Dict[str, Any]] = None) -> Optional[float]:
    if metrics is not None and "published_target_top1" in metrics:
        return pct_from_unit(metrics["published_target_top1"])
    if dataset in ALL_TARGETS:
        return float(ALL_TARGETS[dataset]["best_acc"])
    return None


def build_run_row(
    dataset: str,
    seed: int,
    status: str,
    run_dir: Path,
    metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    command: Optional[List[str]] = None,
    synced: Optional[Dict[str, str]] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": int(seed),
        "status": status,
        "run_dir": str(run_dir),
        "command": " ".join(command) if command else None,
        "error": error,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": elapsed_sec,
    }
    if synced:
        row.update(synced)

    if metrics:
        target_pct = target_pct_for_dataset(dataset, metrics)
        stage1_pct = pct_from_unit(metrics.get("stage1_test_top1"))
        stage2_pct = pct_from_unit(metrics.get("stage2_test_top1"))

        row.update(
            {
                "encoder_model": metrics.get("encoder_model"),
                "encoder_pool": metrics.get("encoder_pool"),
                "num_classes": metrics.get("num_classes"),
                "stage1_test_top1_pct": stage1_pct,
                "stage2_test_top1_pct": stage2_pct,
                "stage1_best_val_top1_pct": pct_from_unit(metrics.get("stage1_best_val_top1")),
                "stage2_best_val_top1_pct": pct_from_unit(metrics.get("stage2_best_val_top1")),
                "target_top1_pct": target_pct,
                "stage1_delta_vs_target_pp": (stage1_pct - target_pct) if (stage1_pct is not None and target_pct is not None) else None,
                "stage2_delta_vs_target_pp": (stage2_pct - target_pct) if (stage2_pct is not None and target_pct is not None) else None,
                "beat_target_stage1": bool(stage1_pct is not None and target_pct is not None and stage1_pct > target_pct),
                "beat_target_stage2": bool(stage2_pct is not None and target_pct is not None and stage2_pct > target_pct),
            }
        )
    return row


def sample_std(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    return float(stdev(values))


def build_dataset_aggregates(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") in {"completed", "reused"}:
            grouped[str(row["dataset"])].append(row)

    out: List[Dict[str, Any]] = []
    for dataset, items in sorted(grouped.items()):
        stage1 = [float(x["stage1_test_top1_pct"]) for x in items if x.get("stage1_test_top1_pct") is not None]
        stage2 = [float(x["stage2_test_top1_pct"]) for x in items if x.get("stage2_test_top1_pct") is not None]
        target = next((x.get("target_top1_pct") for x in items if x.get("target_top1_pct") is not None), None)
        best_stage2_row = max(
            (x for x in items if x.get("stage2_test_top1_pct") is not None),
            key=lambda x: float(x["stage2_test_top1_pct"]),
            default=None,
        )

        out.append(
            {
                "dataset": dataset,
                "n_runs": len(items),
                "seeds": ",".join(str(x["seed"]) for x in sorted(items, key=lambda z: int(z["seed"]))),
                "target_top1_pct": target,
                "stage1_mean_pct": float(mean(stage1)) if stage1 else None,
                "stage1_std_pct": sample_std(stage1),
                "stage1_best_pct": max(stage1) if stage1 else None,
                "stage2_mean_pct": float(mean(stage2)) if stage2 else None,
                "stage2_std_pct": sample_std(stage2),
                "stage2_best_pct": max(stage2) if stage2 else None,
                "stage2_best_seed": best_stage2_row.get("seed") if best_stage2_row else None,
                "stage2_mean_delta_vs_target_pp": (float(mean(stage2)) - float(target)) if (stage2 and target is not None) else None,
                "stage2_best_delta_vs_target_pp": (float(max(stage2)) - float(target)) if (stage2 and target is not None) else None,
                "any_stage2_beats_target": any(bool(x.get("beat_target_stage2")) for x in items),
            }
        )
    return out


def save_json(path: Path, payload: Any) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    mkdir(path.parent)
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_markdown_summary(path: Path, rows: Sequence[Dict[str, Any]], aggs: Sequence[Dict[str, Any]]) -> None:
    mkdir(path.parent)
    lines: List[str] = []
    lines.append("# CP-FAIR Text Multi-Run Summary\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Dataset aggregates\n")
    if aggs:
        lines.append("| Dataset | Runs | Seeds | Stage-2 mean (%) | Stage-2 std | Best Stage-2 (%) | Target (%) | Best delta (pp) |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for row in aggs:
            lines.append(
                "| {dataset} | {n_runs} | {seeds} | {stage2_mean_pct} | {stage2_std_pct} | {stage2_best_pct} | {target_top1_pct} | {stage2_best_delta_vs_target_pp} |".format(
                    dataset=row.get("dataset", ""),
                    n_runs=row.get("n_runs", ""),
                    seeds=row.get("seeds", ""),
                    stage2_mean_pct=_fmt(row.get("stage2_mean_pct")),
                    stage2_std_pct=_fmt(row.get("stage2_std_pct")),
                    stage2_best_pct=_fmt(row.get("stage2_best_pct")),
                    target_top1_pct=_fmt(row.get("target_top1_pct")),
                    stage2_best_delta_vs_target_pp=_fmt(row.get("stage2_best_delta_vs_target_pp")),
                )
            )
    else:
        lines.append("No completed runs yet.\n")

    lines.append("\n## Per-run results\n")
    if rows:
        lines.append("| Dataset | Seed | Status | Stage-1 (%) | Stage-2 (%) | Target (%) | Stage-2 delta (pp) |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for row in sorted(rows, key=lambda x: (str(x.get("dataset")), int(x.get("seed", 0)))):
            lines.append(
                "| {dataset} | {seed} | {status} | {s1} | {s2} | {tgt} | {d2} |".format(
                    dataset=row.get("dataset", ""),
                    seed=row.get("seed", ""),
                    status=row.get("status", ""),
                    s1=_fmt(row.get("stage1_test_top1_pct")),
                    s2=_fmt(row.get("stage2_test_top1_pct")),
                    tgt=_fmt(row.get("target_top1_pct")),
                    d2=_fmt(row.get("stage2_delta_vs_target_pp")),
                )
            )
    else:
        lines.append("No runs recorded.\n")

    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(val: Any) -> str:
    if val is None:
        return ""
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def save_summaries(summary_dir: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Path]:
    aggs = build_dataset_aggregates(rows)
    files = {
        "runs_json": summary_dir / "all_runs_summary.json",
        "runs_csv": summary_dir / "all_runs_summary.csv",
        "aggs_json": summary_dir / "dataset_aggregates.json",
        "aggs_csv": summary_dir / "dataset_aggregates.csv",
        "report_md": summary_dir / "summary_report.md",
    }
    save_json(files["runs_json"], list(rows))
    save_csv(files["runs_csv"], list(rows))
    save_json(files["aggs_json"], list(aggs))
    save_csv(files["aggs_csv"], list(aggs))
    save_markdown_summary(files["report_md"], list(rows), list(aggs))
    return files


def sync_summary_files(summary_files: Dict[str, Path], drive_dir: Path) -> None:
    for name, src in summary_files.items():
        dst = drive_dir / "summary" / src.name
        copy_file_if_exists(src, dst)


def build_command(args: argparse.Namespace, run_root: Path, cache_dir: Path, dataset: str, seed: int) -> List[str]:
    bench_script = Path(args.benchmark_script).resolve()
    cmd: List[str] = [
        sys.executable,
        str(bench_script),
        "--dataset",
        dataset,
        "--encoder_model",
        args.encoder_model,
        "--encoder_pool",
        args.encoder_pool,
        "--encode_batch_size",
        str(args.encode_batch_size),
        "--d_model",
        str(args.d_model),
        "--num_blocks",
        str(args.num_blocks),
        "--num_heads",
        str(args.num_heads),
        "--n_experts",
        str(args.n_experts),
        "--proj_dim",
        str(args.proj_dim),
        "--num_virtual_tokens",
        str(args.num_virtual_tokens),
        "--seed",
        str(seed),
        "--workers",
        str(args.workers),
        "--run_root",
        str(run_root),
        "--cache_dir",
        str(cache_dir),
        "--device",
        args.device,
    ]

    opt_ints = {
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "epochs_stage1": args.epochs_stage1,
        "epochs_stage2": args.epochs_stage2,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
    }
    opt_floats = {
        "val_fraction": args.val_fraction,
        "lr": args.lr,
        "stage2_lr": args.stage2_lr,
        "weight_decay": args.weight_decay,
        "gamma_scale": args.gamma_scale,
        "depth_order_lambda": args.depth_order_lambda,
    }

    for key, value in opt_ints.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    for key, value in opt_floats.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    if args.use_sam:
        cmd.append("--use_sam")
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 3-seed CP-FAIR text benchmarks and export artifacts")
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS), help="Comma-separated datasets")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS), help="Comma-separated seeds")
    p.add_argument("--benchmark_script", type=str, default=None)
    p.add_argument("--encoder_model", type=str, default="bert-base-uncased")
    p.add_argument("--encoder_pool", type=str, default="mean", choices=["mean", "cls"])
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--n_experts", type=int, default=16)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--num_virtual_tokens", type=int, default=8)

    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--test_batch_size", type=int, default=None)
    p.add_argument("--epochs_stage1", type=int, default=None)
    p.add_argument("--epochs_stage2", type=int, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stage2_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--gamma_scale", type=float, default=0.0)
    p.add_argument("--depth_order_lambda", type=float, default=0.0)
    p.add_argument("--use_sam", action="store_true")
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--test_limit", type=int, default=None)

    p.add_argument("--local_root", type=str, default="./cp_fair_text_grid_outputs")
    p.add_argument("--experiment_name", type=str, default="imdb_20ng_agnews_3seeds")
    p.add_argument("--cache_dir", type=str, default=None, help="Shared embedding cache directory")
    p.add_argument("--drive_dir", type=str, default=None, help="Mounted Google Drive folder to receive artifacts")
    p.add_argument("--drive_mode", type=str, default="archive", choices=["archive", "mirror", "both", "none"])
    p.add_argument("--mount_drive", action="store_true", help="Attempt to mount /content/drive when running in Colab")
    p.add_argument("--skip_existing", action="store_true", help="Reuse completed runs if final_metrics.json already exists")
    p.add_argument("--fail_fast", action="store_true", help="Stop immediately if a run fails")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_csv_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    if not datasets:
        raise SystemExit("No datasets provided")
    if not seeds:
        raise SystemExit("No seeds provided")

    benchmark_script = Path(args.benchmark_script).resolve() if args.benchmark_script else Path(__file__).resolve().with_name("cp_fair_text_benchmark.py")
    if not benchmark_script.exists():
        raise SystemExit(f"Benchmark script not found: {benchmark_script}")
    args.benchmark_script = str(benchmark_script)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cuda"

    session_root = mkdir(Path(args.local_root).resolve() / args.experiment_name)
    run_root = mkdir(session_root / "runs_text")
    summary_dir = mkdir(session_root / "summary")
    local_archive_dir = mkdir(session_root / "archives_local")
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else mkdir(Path(args.local_root).resolve() / "cache_text")
    mkdir(cache_dir)

    drive_dir: Optional[Path] = None
    if args.drive_dir:
        if args.mount_drive:
            maybe_mount_drive()
        drive_dir = Path(args.drive_dir).resolve()
        if str(drive_dir).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
            raise SystemExit(
                "Google Drive does not appear to be mounted. Mount it in a notebook cell first, "
                "or pass --mount_drive when running in Colab."
            )
        mkdir(drive_dir)

    print("=" * 100)
    print("CP-FAIR text multirun launcher")
    print(f"Datasets:        {datasets}")
    print(f"Seeds:           {seeds}")
    print(f"Benchmark script:{benchmark_script.resolve()}")
    print(f"Session root:    {session_root}")
    print(f"Run root:        {run_root}")
    print(f"Cache dir:       {cache_dir}")
    print(f"Drive dir:       {drive_dir if drive_dir else '(disabled)'}")
    print(f"Drive mode:      {args.drive_mode}")
    print("=" * 100)

    all_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        for seed in seeds:
            run_name = expected_run_name(
                dataset=dataset,
                encoder_model=args.encoder_model,
                num_blocks=args.num_blocks,
                d_model=args.d_model,
                num_virtual_tokens=args.num_virtual_tokens,
                seed=seed,
            )
            run_dir = run_root / run_name
            final_metrics_file = run_dir / "final_metrics.json"
            command = build_command(args, run_root=run_root, cache_dir=cache_dir, dataset=dataset, seed=seed)

            print("\n" + "-" * 100)
            print(f"Dataset={dataset} | Seed={seed}")
            print("Command:")
            print(" ".join(command))
            print("-" * 100)

            started_at = datetime.now().isoformat(timespec="seconds")
            t0 = time.time()

            try:
                if args.skip_existing and final_metrics_file.exists():
                    metrics = json.loads(final_metrics_file.read_text())
                    synced = {}
                    if drive_dir is not None and args.drive_mode != "none":
                        synced = sync_run_artifacts(run_dir, drive_dir, args.drive_mode, local_archive_dir)
                    row = build_run_row(
                        dataset=dataset,
                        seed=seed,
                        status="reused",
                        run_dir=run_dir,
                        metrics=metrics,
                        command=command,
                        synced=synced,
                        started_at=started_at,
                        finished_at=datetime.now().isoformat(timespec="seconds"),
                        elapsed_sec=time.time() - t0,
                    )
                    all_rows.append(row)
                else:
                    env = os.environ.copy()
                    env.setdefault("TOKENIZERS_PARALLELISM", "false")
                    subprocess.run(command, check=True, env=env)

                    if not final_metrics_file.exists():
                        raise FileNotFoundError(f"Expected metrics not found: {final_metrics_file}")

                    metrics = json.loads(final_metrics_file.read_text())
                    synced = {}
                    if drive_dir is not None and args.drive_mode != "none":
                        synced = sync_run_artifacts(run_dir, drive_dir, args.drive_mode, local_archive_dir)
                    row = build_run_row(
                        dataset=dataset,
                        seed=seed,
                        status="completed",
                        run_dir=run_dir,
                        metrics=metrics,
                        command=command,
                        synced=synced,
                        started_at=started_at,
                        finished_at=datetime.now().isoformat(timespec="seconds"),
                        elapsed_sec=time.time() - t0,
                    )
                    all_rows.append(row)

            except Exception as exc:
                row = build_run_row(
                    dataset=dataset,
                    seed=seed,
                    status="failed",
                    run_dir=run_dir,
                    error=f"{type(exc).__name__}: {exc}",
                    command=command,
                    started_at=started_at,
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    elapsed_sec=time.time() - t0,
                )
                all_rows.append(row)
                print(f"Run failed: {row['error']}")
                if args.fail_fast:
                    summary_files = save_summaries(summary_dir, all_rows)
                    if drive_dir is not None:
                        sync_summary_files(summary_files, drive_dir)
                    raise

            summary_files = save_summaries(summary_dir, all_rows)
            if drive_dir is not None:
                sync_summary_files(summary_files, drive_dir)

    aggs = build_dataset_aggregates(all_rows)
    print("\n" + "=" * 100)
    print("Finished grid")
    print(f"Saved per-run summary:     {summary_dir / 'all_runs_summary.csv'}")
    print(f"Saved dataset aggregates:  {summary_dir / 'dataset_aggregates.csv'}")
    if drive_dir is not None:
        print(f"Drive summary directory:   {drive_dir / 'summary'}")
        if args.drive_mode != "none":
            print(f"Drive artifacts mode:      {args.drive_mode}")
    print("\nDataset aggregates:")
    for row in aggs:
        print(
            f"  - {row['dataset']}: stage2_mean={_fmt(row.get('stage2_mean_pct'))}% "
            f"stage2_std={_fmt(row.get('stage2_std_pct'))} best={_fmt(row.get('stage2_best_pct'))}% "
            f"target={_fmt(row.get('target_top1_pct'))}%"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
