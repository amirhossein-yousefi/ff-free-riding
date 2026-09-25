#!/usr/bin/env python3
"""Aggregate E1 tier 2 (recomputed deepest-block sep_curr_nl on the validation-selected EMA
checkpoints) into results/e1_sep_tier2.json, with the sep-vs-accuracy analysis.

Inputs : experiments/e1_sep/out/<name>__ema_eval.json (18 points) and
         experiments/e1_sep/out/c10_mgc_c1_seed42__{raw_train,ema_eval}.json (code validation).
Raw sources re-read here (not taken from the per-point JSONs): events.jsonl final_test_eval
(accuracies) and layer_metrics.jsonl at the checkpoint epoch (tier-1 logged sep_curr_nl).

Statistics (all exploratory; n is small):
  * across points : Spearman rho(sep, acc) over the n points; two-sided permutation p
                    (100k random permutations of acc, seed 0);
  * within config : Pearson r of config-centered values (sep - config mean vs acc - config mean),
                    stratified permutation p (acc permuted within each configuration, 100k);
                    per-configuration Spearman listed for transparency (n = 1..4 per config);
  * across configs: Spearman rho of configuration means; exact permutation p (all orderings).
Configurations: the kappa=0 D128 seed 42 (GA run, SAM off) and seeds 123/456 (SAM on) are separate
configurations (D13); C10 gamma=0 has seeds 123/456 only (seed 42's checkpoint is unrecoverable,
and that run used a different config -- D7), so tier 2 has n=18 points in 7 configurations.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ffcr_common as C  # noqa: E402

OUT_DIR = C.EXP / "e1_sep" / "out"
OUT_JSON = C.RESULTS / "e1_sep_tier2.json"
N_PERM = 100_000
ACC_KEYS = {"S2-TTA": "stage2_test_top1_tta", "S1": "stage1_test_top1_no_tta"}


def spearman(x, y):
    return float(stats.spearmanr(x, y).statistic)


def perm_p_global(x, y, stat, rng, n=N_PERM):
    """Two-sided permutation p for Spearman rho (vectorized: permuting y permutes its ranks)."""
    obs = stat(x, y)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    ryc = ry - ry.mean()
    ryc = ryc / np.linalg.norm(ryc)
    cnt = 0
    for chunk in range(0, n, 10_000):
        m = min(10_000, n - chunk)
        perms = np.argsort(rng.random((m, len(y))), axis=1)
        r = ryc[perms] @ rx
        cnt += int((np.abs(r) >= abs(obs) - 1e-12).sum())
    return obs, (cnt + 1) / (n + 1)


def centred(v, groups):
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    for g in set(groups):
        m = np.array([gg == g for gg in groups])
        out[m] = v[m] - v[m].mean()
    return out


def within_stat(sep, acc, groups):
    cs, ca = centred(sep, groups), centred(acc, groups)
    if np.allclose(cs, 0) or np.allclose(ca, 0):
        return float("nan")
    return float(np.corrcoef(cs, ca)[0, 1])


def within_perm(sep, acc, groups, rng, n=N_PERM):
    obs = within_stat(sep, acc, groups)
    acc = np.asarray(acc, dtype=np.float64)
    idx = {g: np.where(np.array(groups) == g)[0] for g in set(groups)}
    cnt = 0
    for _ in range(n):
        a = acc.copy()
        for g, ii in idx.items():
            a[ii] = acc[rng.permutation(ii)]
        if abs(within_stat(sep, a, groups)) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def between_exact(x, y):
    obs = spearman(x, y)
    perms = list(itertools.permutations(range(len(y))))
    y = np.asarray(y)
    cnt = sum(1 for p in perms if abs(spearman(x, y[list(p)])) >= abs(obs) - 1e-12)
    return obs, cnt / len(perms)


def main():
    rng = np.random.default_rng(0)
    points, missing = [], []
    for name in C.E1_POINTS:
        p = OUT_DIR / f"{name}__ema_eval.json"
        if not p.exists():
            missing.append(name)
            continue
        r = json.load(open(p))
        if r.get("status") != "OK" or not r["gates"]["gate_pass"]:
            missing.append(name)
            continue
        e = C.REG[name]
        fte, rend = C.read_events(e["run_dir"])
        E = r["epoch_stage1"]
        lm = C.layer_metrics_at_epoch(e, E)
        Ld = r["deepest_block"]["block"]
        points.append(dict(
            name=name, config=C.E1_CONFIG[name], dataset=e["dataset"], seed=e["seed"],
            ckpt_sha256=r["ckpt"]["sha256"], epoch_stage1=E,
            sep_tier2=r["deepest_block"]["sep_curr_nl"], sep_tier2_se=r["deepest_block"]["sep_curr_nl_se"],
            sep_tier1_logged_at_ckpt_epoch=lm[Ld].get("sep_curr_nl"),
            acc={k: fte[v] for k, v in ACC_KEYS.items()},
            gate=dict(recomputed=r["gates"]["recomputed_ema_test_top1_no_tta"],
                      anchor=r["gates"]["anchor_final_test_s1_top1_no_tta"], strict=r["gates"].get("gate_strict"),
                      source=r["gates"]["gate_source"]),
            per_block_tier2=[b["measured"]["margin_curr_nl"]["mean"] for b in r["per_block"]],
            n_batches=r["protocol"]["n_batches"], batch_size=r["protocol"]["batch_size"],
            data_seed=r["protocol"]["data_seed"],
        ))

    analysis = {}
    if len(points) >= 4:
        sep = np.array([p["sep_tier2"] for p in points])
        groups = [p["config"] for p in points]
        for lab in ACC_KEYS:
            acc = np.array([100.0 * p["acc"][lab] for p in points])
            rho, p_rho = perm_p_global(sep, acc, spearman, rng)
            w_r, w_p = within_perm(sep, acc, groups, rng)
            cfgs = sorted(set(groups))
            ms = [float(np.mean([s for s, g in zip(sep, groups) if g == c])) for c in cfgs]
            ma = [float(np.mean([a for a, g in zip(acc, groups) if g == c])) for c in cfgs]
            b_rho, b_p = between_exact(ms, ma)
            per_cfg = {}
            for c in cfgs:
                ii = [i for i, g in enumerate(groups) if g == c]
                per_cfg[c] = dict(n=len(ii), sep=[float(sep[i]) for i in ii], acc_pct=[float(acc[i]) for i in ii],
                                  spearman=(spearman(sep[ii], acc[ii]) if len(ii) >= 3 else None))
            # same analysis within each dataset family (C10 vs C100)
            fam = {}
            for ds in ("cifar10", "cifar100"):
                ii = [i for i, p in enumerate(points) if p["dataset"] == ds]
                if len(ii) >= 4:
                    fr, fp = perm_p_global(sep[ii], acc[ii], spearman, rng)
                    fam[ds] = dict(n=len(ii), spearman=fr, perm_p=fp)
            analysis[lab] = dict(
                accuracy_protocol=ACC_KEYS[lab],
                across_points=dict(n=len(points), spearman=rho, perm_p_two_sided=p_rho, n_perm=N_PERM),
                within_config=dict(statistic="Pearson r of config-centred sep vs config-centred acc",
                                   r=w_r, stratified_perm_p_two_sided=w_p, n_perm=N_PERM, n_configs=len(cfgs)),
                across_configs=dict(n_configs=len(cfgs), configs=cfgs, sep_means=ms, acc_means_pct=ma,
                                    spearman=b_rho, exact_perm_p_two_sided=b_p),
                per_config=per_cfg, per_dataset=fam)

    val = {}
    for mode in ("raw_train", "ema_eval"):
        p = OUT_DIR / f"{C.E1_VALIDATION}__{mode}.json"
        if p.exists():
            r = json.load(open(p))
            val[mode] = dict(deepest_sep_curr_nl=r["deepest_block"]["sep_curr_nl"],
                             se=r["deepest_block"]["sep_curr_nl_se"],
                             per_block=[b["measured"]["margin_curr_nl"]["mean"] for b in r["per_block"]],
                             logged_at_ckpt_epoch=r["deepest_block"]["logged_sep_curr_nl_at_ckpt_epoch"],
                             vs_gradshare=r.get("validation_vs_gradshare"),
                             gate=r["gates"])

    out = dict(
        _header=dict(
            description="E1 tier 2 (CR-18): deepest-block sep_curr_nl recomputed on the validation-selected "
                        "stage1_best checkpoints (net_ema, eval mode) through the trainer's diagnostic path "
                        "(forward_block + EMA-teacher hard negatives), vs accuracy of the same checkpoints.",
            created=time.strftime("%Y-%m-%d"), script=str(Path(__file__).resolve()),
            measure_script=str(C.EXP / "e1_sep" / "measure_sep_tier2.py"),
            n_points=len(points), missing_or_failed=missing,
            disclosure="C10 gamma=0 seed 42 is excluded from tier 2: its published checkpoint is unrecoverable "
                       "(n=18 instead of 19; tier 1 covers all 19). Seed 42 also used a different config (D7).",
            sd="per-point SE = per-batch SD (ddof=1)/sqrt(N batches)",
            exploratory="all correlations are exploratory (small n, configurations differ in many respects)"),
        points=points, analysis=analysis, code_validation_mgc_d128_seed42=val)
    C.write_json_atomic(OUT_JSON, out)
    print(f"wrote {OUT_JSON}: {len(points)} points; missing {missing}")
    for lab, a in analysis.items():
        print(f"  {lab}: across rho={a['across_points']['spearman']:.3f} p={a['across_points']['perm_p_two_sided']:.3f} | "
              f"within r={a['within_config']['r']:.3f} p={a['within_config']['stratified_perm_p_two_sided']:.3f} | "
              f"configs rho={a['across_configs']['spearman']:.3f} p={a['across_configs']['exact_perm_p_two_sided']:.3f}")


if __name__ == "__main__":
    main()
