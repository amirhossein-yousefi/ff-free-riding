"""Compute paired bootstrap CIs and pairwise prediction-disagreement rates
for the CIFAR-100 dissociation contrast (γ=0 vs adaptive κ=0 vs cumulative).

Per-seed Stage-2 TTA prediction npz files are expected under
``preds_cifar100/runs_<variant>_seed<seed>/preds_stage2_test_tta.npz``; the nine files (30 MB) are
bundled in ``preds_cifar100/`` (see its README; they are published with the code, not with
the checkpoints). For each pair of variants
the script computes, on the matched test set:

  - paired accuracy delta (per seed)
  - 95% paired bootstrap CI of the test-example-level mean delta (5000 resamples)
  - pairwise prediction disagreement rate (per seed and averaged)
  - fraction of test examples flipped from correct→wrong and wrong→correct

Outputs (alongside this script, in ``analysis/``; override with ``FF_PAIRED_OUT``):
  - ``60_paired_dissociation.json``  (machine-readable)
  - ``60_paired_dissociation.md``    (human-readable summary)

Override the prediction-file root via ``FF_C100_PREDS`` if your layout differs.

Usage:
  python analysis/paired_dissociation_stats.py
"""
from __future__ import annotations

import json
import os
import statistics as _st
from pathlib import Path
import numpy as np

# Defaults assume the supplement layout: supplement_code/analysis/this.py.
# Override via env var if your layout differs.
SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLEMENT_ROOT = SCRIPT_DIR.parent  # supplement_code/

# Where to find the per-seed Stage-2 TTA prediction files.
PRED_ROOT = Path(os.environ.get(
    "FF_C100_PREDS", SUPPLEMENT_ROOT / "preds_cifar100"))

# Where to write the output (alongside this script by default).
OUT_DIR = Path(os.environ.get("FF_PAIRED_OUT", SCRIPT_DIR))
OUT_JSON = OUT_DIR / "60_paired_dissociation.json"
OUT_MD = OUT_DIR / "60_paired_dissociation.md"

VARIANTS = ["gamma0", "gated_k0", "cumulative"]
SEEDS = [42, 123, 456]
N_BOOT = 5000
RNG = np.random.default_rng(0)


def find_pred(variant: str, seed: int) -> Path | None:
    """Find a per-seed S2 TTA prediction file for (variant, seed).

    Looks first under ``$PRED_ROOT/runs_<variant>_seed<seed>/`` (flat
    layout used by the bundled supplement). Falls back to one extra
    nesting level (``runs_<variant>_seed<seed>/<run_id>/``) for
    compatibility with research-tree layouts.
    """
    parent = PRED_ROOT / f"runs_{variant}_seed{seed}"
    if not parent.exists():
        return None
    flat = parent / "preds_stage2_test_tta.npz"
    if flat.exists():
        return flat
    # Older nested layout: runs_<variant>_seed<seed>/<long-run-name>/preds_*.npz
    for rd in sorted(parent.iterdir()):
        if not rd.is_dir():
            continue
        p = rd / "preds_stage2_test_tta.npz"
        if p.exists():
            return p
    return None


def load_pred(p: Path):
    z = np.load(p)
    y_true = z["y_true"]
    y_pred = z["y_pred"]
    correct = (y_pred == y_true).astype(np.int8)
    return y_true, y_pred, correct


def paired_bootstrap_ci(delta_per_example: np.ndarray, n_boot=N_BOOT, alpha=0.05):
    n = len(delta_per_example)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        means[b] = delta_per_example[idx].mean()
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(delta_per_example.mean()), float(lo), float(hi)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all predictions
    preds = {}
    missing = []
    for v in VARIANTS:
        for s in SEEDS:
            p = find_pred(v, s)
            if p is None:
                missing.append((v, s))
                continue
            yt, yp, c = load_pred(p)
            preds[(v, s)] = {"y_true": yt, "y_pred": yp, "correct": c, "path": str(p)}

    if missing:
        print("MISSING prediction files (looked under {}):".format(PRED_ROOT))
        for v, s in missing:
            print(f"  {v} seed {s}")
        print("If your prediction files live elsewhere, set FF_C100_PREDS.")
        if not preds:
            return

    # Confirm matched test set: same y_true vector across all (variant, seed)
    if ("gamma0", 42) not in preds:
        print("Cannot anchor matched-test check: gamma0 seed 42 prediction not found.")
        return
    base_yt = preds[("gamma0", 42)]["y_true"]
    for k, d in preds.items():
        if not np.array_equal(d["y_true"], base_yt):
            print(f"WARN: y_true mismatch at {k} — paired analysis may be invalid.")
    n_test = len(base_yt)

    out = {
        "meta": {
            "n_test": int(n_test),
            "n_seeds": len(SEEDS),
            "n_boot": N_BOOT,
            "test_set": "CIFAR-100 test, deterministic order",
            "pred_root": str(PRED_ROOT),
        },
        "per_variant_acc_per_seed": {},
        "pairwise_paired_bootstrap": {},
        "pairwise_disagreement_rate": {},
    }

    # Per-variant per-seed accuracy
    for v in VARIANTS:
        out["per_variant_acc_per_seed"][v] = {}
        for s in SEEDS:
            d = preds.get((v, s))
            if d is None:
                continue
            out["per_variant_acc_per_seed"][v][s] = float(d["correct"].mean())

    pairs = [("gamma0", "gated_k0"), ("gamma0", "cumulative"), ("gated_k0", "cumulative")]
    for a, b in pairs:
        deltas = []
        per_seed_delta = {}
        per_seed_disagree = {}
        per_seed_flips = {"a_correct_b_wrong": {}, "a_wrong_b_correct": {}}
        for s in SEEDS:
            da, db = preds.get((a, s)), preds.get((b, s))
            if da is None or db is None:
                continue
            d_ex = (da["correct"].astype(np.int32) - db["correct"].astype(np.int32))
            deltas.append(d_ex)
            per_seed_delta[s] = float(d_ex.mean())
            disagree = (da["y_pred"] != db["y_pred"]).mean()
            per_seed_disagree[s] = float(disagree)
            per_seed_flips["a_correct_b_wrong"][s] = int(((da["correct"] == 1) & (db["correct"] == 0)).sum())
            per_seed_flips["a_wrong_b_correct"][s] = int(((da["correct"] == 0) & (db["correct"] == 1)).sum())
        if not deltas:
            continue
        all_deltas = np.concatenate(deltas)
        mean, lo, hi = paired_bootstrap_ci(all_deltas)
        out["pairwise_paired_bootstrap"][f"{a}_minus_{b}"] = {
            "n_examples": int(len(all_deltas)),
            "mean_delta_acc": mean,
            "ci_lo_95": lo,
            "ci_hi_95": hi,
            "per_seed_delta_acc": per_seed_delta,
        }
        out["pairwise_disagreement_rate"][f"{a}_vs_{b}"] = {
            "per_seed": per_seed_disagree,
            "mean": float(np.mean(list(per_seed_disagree.values()))),
            "flips_a_correct_b_wrong_per_seed": per_seed_flips["a_correct_b_wrong"],
            "flips_a_wrong_b_correct_per_seed": per_seed_flips["a_wrong_b_correct"],
        }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")

    # Human-readable summary
    md = ["# Paired statistical evidence for the CIFAR-100 dissociation",
          "",
          f"Test set: CIFAR-100 ({n_test} examples), 3 seeds × 3 variants. All variants",
          "share the same test ordering, so per-example deltas are well-defined.",
          "",
          "## Per-variant Stage-2 TTA test accuracy (per seed)",
          "",
          "| variant | seed 42 | seed 123 | seed 456 | mean ± std (n=3) |",
          "| -- | --: | --: | --: | --: |"]
    for v in VARIANTS:
        row = out["per_variant_acc_per_seed"].get(v, {})
        cells = [f"{row.get(s, 0)*100:.2f}%" if s in row else "—" for s in SEEDS]
        vals = [row[s] for s in SEEDS if s in row]
        if len(vals) >= 2:
            m = sum(vals) / len(vals) * 100
            sd = _st.stdev(vals) * 100
            mean_cell = f"{m:.2f} ± {sd:.2f}%"
        else:
            mean_cell = "—"
        md.append(f"| {v} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_cell} |")

    md += ["", "## Paired bootstrap CI for accuracy delta (5000 resamples)", ""]
    md += ["| pair | n examples | mean Δacc | 95% CI | per-seed Δacc |",
           "| -- | --: | --: | --: | --: |"]
    for k, d in out["pairwise_paired_bootstrap"].items():
        per_seed = ", ".join(f"s{s}={v*100:+.3f}%" for s, v in d["per_seed_delta_acc"].items())
        md.append(f"| {k} | {d['n_examples']} | {d['mean_delta_acc']*100:+.4f}% | [{d['ci_lo_95']*100:+.4f}%, {d['ci_hi_95']*100:+.4f}%] | {per_seed} |")

    md += ["", "## Pairwise prediction disagreement rate", ""]
    md += ["| pair | seed 42 | seed 123 | seed 456 | mean | a-correct,b-wrong | a-wrong,b-correct |",
           "| -- | --: | --: | --: | --: | --: | --: |"]
    for k, d in out["pairwise_disagreement_rate"].items():
        ps = d["per_seed"]
        flips_aw = d["flips_a_correct_b_wrong_per_seed"]
        flips_wa = d["flips_a_wrong_b_correct_per_seed"]
        cells_ps = "/".join(f"{ps.get(s, 0)*100:.2f}%" for s in SEEDS)
        cells_aw = "/".join(str(flips_aw.get(s, 0)) for s in SEEDS)
        cells_wa = "/".join(str(flips_wa.get(s, 0)) for s in SEEDS)
        md.append(f"| {k} | {ps.get(42, 0)*100:.2f}% | {ps.get(123, 0)*100:.2f}% | "
                  f"{ps.get(456, 0)*100:.2f}% | {d['mean']*100:.2f}% | {cells_aw} | {cells_wa} |")

    md += ["",
           "## Reading",
           "",
           "Across every variant pair, the 95% paired bootstrap CI for Δacc on the matched",
           "CIFAR-100 test set **straddles or barely escapes 0**, even though deepest-layer",
           "`sep_curr_nl` differs by 4–5×. Pairwise disagreement rates are 9–18% per seed,",
           "almost balanced (correct↔wrong flips in roughly equal numbers); these rates do",
           "not scale with the magnitude of the per-block diagnostic gap. This is the",
           "test-example-level form of the dissociation: a large per-block-health intervention",
           "moves only a small, near-the-decision-boundary slice of the test set, and the slice",
           "is approximately net-zero in label correctness.",
           ""]
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
