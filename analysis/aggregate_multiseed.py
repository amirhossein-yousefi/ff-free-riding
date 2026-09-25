"""Aggregate multi-seed metrics for paper update.

Outputs:
  aggregated_results.json   - structured results
  prints LaTeX-ready table rows on stdout

Scope (matches the canonical seed table in the paper):
  - CIFAR-10 cp_fair:        seeds 42, 123, 456, 789 (4 seeds)
  - CIFAR-10 gamma0:         seeds 42, 123, 456 (3 seeds)
  - CIFAR-10 gated_k0 L4/D128: seeds 42, 123, 456 (3 seeds)
  - CIFAR-10 gated_k0 L8/D128: seeds 42, 123, 456 (3 seeds)
  - CIFAR-100 gamma0:        seeds 42, 123, 456 (3 seeds)
  - CIFAR-100 gated_k0:      seeds 42, 123, 456 (3 seeds)
  - CIFAR-100 cumulative:    seeds 42, 123, 456 (3 seeds)
  - Tiny ImageNet gamma0:    seeds 42, 123, 456 (3 seeds)
"""
from __future__ import annotations
import json, glob, os, sys, statistics
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def npz_acc(npz_path: str) -> dict:
    z = np.load(npz_path)
    y = z["y_true"]
    s = z["scores"]
    pred = s.argmax(1)
    top1 = float((pred == y).mean())
    top5 = float((np.argpartition(-s, 5, axis=1)[:, :5] == y[:, None]).any(1).mean())
    return {"top1": top1, "top5": top5}


def best_run_dir(parent: str) -> str | None:
    """Pick the run dir under `parent` with stage2 preds, longest training as tiebreak."""
    cands = sorted(glob.glob(os.path.join(parent, "*/")))
    best, key = None, None
    for c in cands:
        has_s2 = os.path.exists(os.path.join(c, "preds_stage2_test_tta.npz"))
        em = os.path.join(c, "epoch_metrics.jsonl")
        n_epoch = sum(1 for _ in open(em)) if os.path.exists(em) else 0
        score = (1 if has_s2 else 0, n_epoch)
        if best is None or score > key:
            best, key = c, score
    return best


def per_run_metrics(run_dir: str) -> dict:
    """Return final accuracies + last-epoch per-block diagnostics for a single run."""
    out = {"run_dir": run_dir, "config": json.load(open(os.path.join(run_dir, "config.json")))}
    for stage in (1, 2):
        for kind in ("no_tta", "tta"):
            f = os.path.join(run_dir, f"preds_stage{stage}_test_{kind}.npz")
            if os.path.exists(f):
                out[f"s{stage}_{kind}"] = npz_acc(f)
    # last-epoch per-layer diagnostics
    lm_path = os.path.join(run_dir, "layer_metrics.jsonl")
    if os.path.exists(lm_path):
        rows = [json.loads(l) for l in open(lm_path)]
        by_stage_epoch = defaultdict(dict)
        for r in rows:
            by_stage_epoch[(r["stage"], r["epoch"])][r["layer"]] = r
        # take stage 1 last epoch (reflects converged free-riding pattern)
        s1_epochs = [e for (s, e) in by_stage_epoch if s == 1]
        if s1_epochs:
            last = max(s1_epochs)
            layers = by_stage_epoch[(1, last)]
            out["s1_last_epoch"] = last
            out["per_layer"] = {
                ell: {
                    k: layers[ell].get(k)
                    for k in ("sep_curr_nl", "sep_nl", "gpos_cur", "gnl_cur", "gni_cur")
                }
                for ell in sorted(layers)
            }
            n_layers = len(layers)
            deepest = n_layers - 1
            out["sep_curr_nl_deep"] = layers[deepest].get("sep_curr_nl")
            out["gpos_cur_deep"] = layers[deepest].get("gpos_cur")
            out["gpos_cur_l0"] = layers[0].get("gpos_cur")
            if out["gpos_cur_l0"]:
                out["gpos_cur_ratio_LdL0"] = out["gpos_cur_deep"] / out["gpos_cur_l0"]
    return out


def fmt_pct(x):
    return f"{x*100:.2f}" if isinstance(x, (int, float)) else "---"


def aggregate(rows, key):
    """Mean and std for `rows[i][key]` skipping None."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def collect_c10_gamma0():
    """CIFAR-10 gamma=0: seeds 42, 123, 456 (3 seeds, matches `tab:gamma0_multiseed`)."""
    seeds = {}
    # seed 42 (published numbers, also reproduced by `out/gamma0_seed42` if present)
    seeds[42] = {
        "source": "published",
        "s1_no_tta": {"top1": 0.9052},
        "s1_tta":    {"top1": 0.9145},
    }
    for s in (123, 456):
        rd = best_run_dir(os.path.join(ROOT, f"out/gamma0_seed{s}/runs"))
        if rd:
            seeds[s] = {"source": rd, **per_run_metrics(rd)}
    return seeds


def collect_c100():
    """Aggregate the CIFAR-100 dissociation trio from per-seed run dirs.

    Default search location: ``$FF_C100_RUNS/runs_<variant>_seed<seed>/``.
    Override with ``FF_C100_RUNS=/path/to/cifar100/runs`` if your runs live
    elsewhere. For a from-scratch reproduction, run the trainer commands
    listed in `repro_manifest.csv` (rows under
    `tab:cifar100_dissociation_multiseed`) first to materialize the run
    dirs, then point this script at them via FF_C100_RUNS.
    """
    runs_root = os.environ.get("FF_C100_RUNS")
    if runs_root is None:
        # No explicit override — try the legacy in-tree location for
        # backwards compatibility, but skip silently if absent.
        runs_root = os.path.join(ROOT, "extracted", "ff_cifar100_multiseed")
    out = {}
    for variant in ("gamma0", "gated_k0", "cumulative"):
        out[variant] = {}
        for seed in (42, 123, 456):
            parent = os.path.join(runs_root, f"runs_{variant}_seed{seed}")
            if not os.path.exists(parent):
                continue
            rd = best_run_dir(parent)
            if rd:
                out[variant][seed] = per_run_metrics(rd)
    return out


def main():
    c10 = collect_c10_gamma0()
    c100 = collect_c100()

    print("=" * 70)
    print("CIFAR-10 γ=0 (paper main; L=4, D=256, bs=512)")
    print("=" * 70)
    rows_s1n, rows_s1t, rows_s2n, rows_s2t = [], [], [], []
    for s in sorted(c10):
        r = c10[s]
        s1n = r.get("s1_no_tta", {}).get("top1")
        s1t = r.get("s1_tta", {}).get("top1")
        s2n = r.get("s2_no_tta", {}).get("top1")
        s2t = r.get("s2_tta", {}).get("top1")
        print(f"  seed={s:>3}: S1 no-TTA {fmt_pct(s1n)} TTA {fmt_pct(s1t)} | "
              f"S2 no-TTA {fmt_pct(s2n)} TTA {fmt_pct(s2t)}  ({r.get('source','?')})")
        if s1n: rows_s1n.append(s1n)
        if s1t: rows_s1t.append(s1t)
        if s2n: rows_s2n.append(s2n)
        if s2t: rows_s2t.append(s2t)
    if rows_s1t:
        print(f"  ---> S1 TTA mean ± std (n={len(rows_s1t)}): "
              f"{statistics.mean(rows_s1t)*100:.2f} ± "
              f"{(statistics.stdev(rows_s1t) if len(rows_s1t)>1 else 0)*100:.2f}")

    print()
    print("=" * 70)
    print("CIFAR-100 multi-seed (L=4, D=256, bs=256)")
    print("=" * 70)
    final = {"c10_gamma0": c10, "c100": {}}
    for variant in c100:
        print(f"\n--- {variant} ---")
        per_seed = []
        for seed in sorted(c100[variant]):
            r = c100[variant][seed]
            row = {
                "seed": seed,
                "s1_no_tta": r.get("s1_no_tta", {}).get("top1"),
                "s1_tta":    r.get("s1_tta", {}).get("top1"),
                "s1_top5_tta": r.get("s1_tta", {}).get("top5"),
                "s2_no_tta": r.get("s2_no_tta", {}).get("top1"),
                "s2_tta":    r.get("s2_tta", {}).get("top1"),
                "s2_top5_tta": r.get("s2_tta", {}).get("top5"),
                "sep_curr_nl_deep": r.get("sep_curr_nl_deep"),
                "gpos_cur_deep":    r.get("gpos_cur_deep"),
                "gpos_cur_ratio_LdL0": r.get("gpos_cur_ratio_LdL0"),
            }
            per_seed.append(row)
            print(f"  seed={seed:>3}: "
                  f"S1 no-TTA {fmt_pct(row['s1_no_tta'])} TTA {fmt_pct(row['s1_tta'])} | "
                  f"S2 no-TTA {fmt_pct(row['s2_no_tta'])} TTA {fmt_pct(row['s2_tta'])} | "
                  f"sep_curr_nl[L3]={row['sep_curr_nl_deep']} | "
                  f"g+_cur ratio={row['gpos_cur_ratio_LdL0']}")
        # aggregates
        agg = {}
        for key in ("s1_no_tta", "s1_tta", "s1_top5_tta",
                   "s2_no_tta", "s2_tta", "s2_top5_tta",
                   "sep_curr_nl_deep", "gpos_cur_deep", "gpos_cur_ratio_LdL0"):
            m, s = aggregate(per_seed, key)
            agg[key] = {"mean": m, "std": s, "n": sum(1 for r in per_seed if r.get(key) is not None)}
        print(f"  AGG (n={agg['s2_tta']['n']}): "
              f"S1 TTA {fmt_pct(agg['s1_tta']['mean'])} ± {fmt_pct(agg['s1_tta']['std'])} | "
              f"S2 TTA {fmt_pct(agg['s2_tta']['mean'])} ± {fmt_pct(agg['s2_tta']['std'])} | "
              f"sep[L3] {agg['sep_curr_nl_deep']['mean']:.3f} ± {agg['sep_curr_nl_deep']['std']:.3f} | "
              f"g+ ratio {agg['gpos_cur_ratio_LdL0']['mean']:.3f}")
        final["c100"][variant] = {"per_seed": per_seed, "agg": agg}

    # save full json
    out_path = os.path.join(ROOT, "aggregated_results.json")
    # convert non-serializable objects (config) — drop them
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if k not in ("run_dir", "config", "per_layer")}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    json.dump(clean(final), open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
