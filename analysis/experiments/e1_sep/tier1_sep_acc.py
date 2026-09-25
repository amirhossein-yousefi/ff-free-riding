#!/usr/bin/env python3
"""E1 tier 1: epoch-aligned deepest-block separation vs validation-selected accuracy.

No model compute. For each of the 19 points of
ff-wiki/_build/analyses/per_seed_sep_vs_acc.json:
  * rebuild the full Stage-1 log across resumed segments (epoch_metrics / layer_metrics),
  * find the stage1_best epoch (first Stage-1 val eval whose EMA top-1 equals run_end.best_s1;
    the trainers update best with a strict ``>``),
  * read the deepest block's ``sep_curr_nl`` from layer_metrics.jsonl at that epoch,
  * pair it with final_test_eval accuracies (S1 / S1-TTA from stage1_best, S2 / S2-TTA from
    stage2_best, whose backbone is the frozen stage1_best EMA network).
Then Spearman correlations (pooled; within-config demeaned; across config means) with
permutation p-values (10,000 perms, fixed seed; p = (count+1)/(n_perm+1), two-sided on |rho|).

Run:  python3 tier1_sep_acc.py   (numpy + scipy only; no torch)
Output: CR/results/e1_tier1_sep_acc.json
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import hashlib
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

HOME = FF_HOME
CF = f"{HOME}/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
CR = FF_OUT  # [CR] was the camera-ready repository; outputs go to <FF_OUT>/results
WIKI_JSON = f"{HOME}/amir-porjects/ff-wiki/_build/analyses/per_seed_sep_vs_acc.json"
OUT = f"{CR}/results/e1_tier1_sep_acc.json"
N_PERM = 10_000
PERM_SEED = 20260925

C100 = f"{CF}/revised_training/extracted/ff_cifar100_multiseed"
GATED_MS = f"{CF}/revised_training/extracted/ff_cifar10_gated_multiseed"

# Each point: ordered list of run-dir segments (oldest first). The LAST segment carries
# final_test_eval/run_end. Earlier segments are included only when the last one resumed them
# (its Stage-1 log does not start at epoch 1). Segments that were restarted from scratch are
# listed under "superseded" for provenance only.
POINTS = [
    # --- CIFAR-10 D256, gamma = 0 ---------------------------------------------------------
    dict(config="c10_gamma0", family="c10_d256", seed=42,
         segments=[f"{CF}/ablations/no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260311_120026",
                   f"{CF}/ablations/no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260312_132043"]),
    dict(config="c10_gamma0", family="c10_d256", seed=123,
         segments=[f"{CF}/revised_training/out/gamma0_seed123/runs/ff_cifar10_cifar10_gamma0_seed123_L4_D256_bs512_seed123_20260417_235110"]),
    dict(config="c10_gamma0", family="c10_d256", seed=456,
         segments=[f"{CF}/revised_training/out/gamma0_seed456/runs/ff_cifar10_cifar10_gamma0_seed456_L4_D256_bs512_seed456_20260422_135437"]),
    # --- CIFAR-10 D256, CP-FAIR (cumulative gamma = 0.7) ---------------------------------------
    dict(config="c10_cp_fair", family="c10_d256", seed=42,
         segments=[f"{CF}/basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260218_144731",
                   f"{CF}/basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035"]),
    dict(config="c10_cp_fair", family="c10_d256", seed=123,
         segments=[f"{CF}/basic_fair_revised_valfix_cp_fair/runs_seed123/ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_bs512_seed123_20260318_001952",
                   f"{CF}/basic_fair_revised_valfix_cp_fair/runs_seed123/ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_bs512_seed123_20260320_001527"]),
    dict(config="c10_cp_fair", family="c10_d256", seed=456,
         segments=[f"{CF}/basic_fair_revised_valfix_cp_fair/runs_seed456/ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_bs512_seed456_20260324_135535",
                   f"{CF}/basic_fair_revised_valfix_cp_fair/runs_seed456/ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_bs512_seed456_20260327_080127"]),
    dict(config="c10_cp_fair", family="c10_d256", seed=789,
         segments=[f"{CF}/basic_fair_revised_valfix_cp_fair/runs_seed789/ff_cifar10_cifar10_cp_fair_clean_progressive_seed789_L4_D256_bs512_seed789_20260330_064018"]),
    # --- CIFAR-10 L4/D128 gated kappa = 0 ------------------------------------------------------
    dict(config="gated_l4d128_k0", family="gated_l4d128", seed=42,
         segments=[f"{CF}/basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/runs/ff_cifar10_adaptive_t1_k0_L4_D128_bs512_seed42_20260314_032410"]),
    dict(config="gated_l4d128_k0", family="gated_l4d128", seed=123,
         segments=[f"{GATED_MS}/runs_gated_k0_seed123/ff_cifar10_cifar10_gated_k0_seed123_L4_D128_bs512_seed123_20260426_071309"]),
    dict(config="gated_l4d128_k0", family="gated_l4d128", seed=456,
         segments=[f"{GATED_MS}/runs_gated_k0_seed456/ff_cifar10_cifar10_gated_k0_seed456_L4_D128_bs512_seed456_20260426_150234"]),
    # --- CIFAR-100 L4/D256 trio (dense, no MoE, no SAM) ------------------------------------------
    dict(config="c100_gamma0", family="c100_d256", seed=42,
         segments=[f"{C100}/runs_gamma0_seed42/ff_cifar100_cifar100_gamma0_seed42_L4_D256_bs256_seed42_20260420_101044"],
         superseded=[f"{C100}/runs_gamma0_seed42/ff_cifar100_cifar100_gamma0_seed42_L4_D256_bs256_seed42_20260417_215649"]),
    dict(config="c100_gamma0", family="c100_d256", seed=123,
         segments=[f"{C100}/runs_gamma0_seed123/ff_cifar100_cifar100_gamma0_seed123_L4_D256_bs256_seed123_20260421_015141",
                   f"{C100}/runs_gamma0_seed123/ff_cifar100_cifar100_gamma0_seed123_L4_D256_bs256_seed123_20260421_124316"]),
    dict(config="c100_gamma0", family="c100_d256", seed=456,
         segments=[f"{C100}/runs_gamma0_seed456/ff_cifar100_cifar100_gamma0_seed456_L4_D256_bs256_seed456_20260422_012047",
                   f"{C100}/runs_gamma0_seed456/ff_cifar100_cifar100_gamma0_seed456_L4_D256_bs256_seed456_20260422_142245"]),
    dict(config="c100_gated_k0", family="c100_d256", seed=42,
         segments=[f"{C100}/runs_gated_k0_seed42/ff_cifar100_cifar100_gated_k0_seed42_L4_D256_bs256_seed42_20260422_185437"]),
    dict(config="c100_gated_k0", family="c100_d256", seed=123,
         segments=[f"{C100}/runs_gated_k0_seed123/ff_cifar100_cifar100_gated_k0_seed123_L4_D256_bs256_seed123_20260423_150626"],
         superseded=[f"{C100}/runs_gated_k0_seed123/ff_cifar100_cifar100_gated_k0_seed123_L4_D256_bs256_seed123_20260423_102837"]),
    dict(config="c100_gated_k0", family="c100_d256", seed=456,
         segments=[f"{C100}/runs_gated_k0_seed456/ff_cifar100_cifar100_gated_k0_seed456_L4_D256_bs256_seed456_20260424_065206",
                   f"{C100}/runs_gated_k0_seed456/ff_cifar100_cifar100_gated_k0_seed456_L4_D256_bs256_seed456_20260424_153513"]),
    dict(config="c100_cumulative", family="c100_d256", seed=42,
         segments=[f"{C100}/runs_cumulative_seed42/ff_cifar100_cifar100_cumulative_seed42_L4_D256_bs256_seed42_20260418_032149"]),
    dict(config="c100_cumulative", family="c100_d256", seed=123,
         segments=[f"{C100}/runs_cumulative_seed123/ff_cifar100_cifar100_cumulative_seed123_L4_D256_bs256_seed123_20260418_190627",
                   f"{C100}/runs_cumulative_seed123/ff_cifar100_cifar100_cumulative_seed123_L4_D256_bs256_seed123_20260419_010632"]),
    dict(config="c100_cumulative", family="c100_d256", seed=456,
         segments=[f"{C100}/runs_cumulative_seed456/ff_cifar100_cifar100_cumulative_seed456_L4_D256_bs256_seed456_20260419_141800",
                   f"{C100}/runs_cumulative_seed456/ff_cifar100_cifar100_cumulative_seed456_L4_D256_bs256_seed456_20260420_051234"]),
]

# Config keys that define "the same configuration" for the split grouping (D7 / D13).
CFG_KEYS = ["use_sam", "gamma_scale", "block_curr_lambda", "depth_order_lambda", "d_model",
            "num_blocks", "n_experts", "moe_top_k", "batch_size", "epochs_stage1", "epochs_stage2",
            "num_classes", "hardness_gated", "hardness_kappa", "hardness_tau", "residual_mode",
            "cumulative_mode", "lr", "weight_decay", "ema_decay"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_point(p):
    segs = p["segments"]
    info = {"segments": [os.path.relpath(s, CF) for s in segs],
            "superseded_restarted_from_scratch": [os.path.relpath(s, CF) for s in p.get("superseded", [])]}
    em_by_epoch, lm_by_epoch, seg_of_epoch = {}, defaultdict(dict), {}
    seg_ranges = []
    for si, s in enumerate(segs):
        em = [r for r in read_jsonl(f"{s}/epoch_metrics.jsonl") if r.get("stage") == 1]
        lm = [r for r in read_jsonl(f"{s}/layer_metrics.jsonl") if r.get("stage") == 1]
        seg_ranges.append((em[0]["epoch"], em[-1]["epoch"]))
        for r in em:
            em_by_epoch[r["epoch"]] = r
            seg_of_epoch[r["epoch"]] = si
        for r in lm:
            lm_by_epoch[r["epoch"]][r["layer"]] = r
    info["segment_stage1_epoch_ranges"] = seg_ranges
    # Lineage checks: first segment starts at 1; each later segment starts after the previous ends.
    assert seg_ranges[0][0] == 1, (p["config"], p["seed"], seg_ranges)
    for (a0, a1), (b0, b1) in zip(seg_ranges, seg_ranges[1:]):
        assert b0 > a1, (p["config"], p["seed"], seg_ranges)
    missing = sorted(set(range(1, seg_ranges[-1][1] + 1)) - set(em_by_epoch))
    info["stage1_epochs_missing_from_logs"] = missing
    # Configs must match across segments.
    cfgs = [json.load(open(f"{s}/config.json")) for s in segs]
    for c in cfgs[1:]:
        diff = {k for k in set(c) | set(cfgs[0]) if c.get(k) != cfgs[0].get(k)}
        assert not diff, (p["config"], p["seed"], diff)
    cfg = cfgs[-1]
    last = segs[-1]
    ev = read_jsonl(f"{last}/events.jsonl")
    fte = [e for e in ev if e.get("event") == "final_test_eval"]
    rend = [e for e in ev if e.get("event") == "run_end"]
    assert len(fte) == 1 and len(rend) == 1, (p["config"], p["seed"])
    fte, rend = fte[0], rend[0]
    best_s1 = rend["best_s1"]
    # Best epoch, derivation A: eval_top1 series, first epoch attaining the max (strict '>' update).
    evals = sorted((e, r["eval_top1"]) for e, r in em_by_epoch.items() if r.get("eval_top1") is not None)
    vmax = max(v for _, v in evals)
    assert abs(vmax - best_s1) < 1e-12, (p["config"], p["seed"], vmax, best_s1)
    best_epoch_a = min(e for e, v in evals if v == vmax)
    # Derivation B: the logged running best_s1 field first equals its final value.
    best_epoch_b = min(e for e, r in em_by_epoch.items() if r.get("best_s1") is not None and abs(r["best_s1"] - best_s1) < 1e-12)
    assert best_epoch_a == best_epoch_b, (p["config"], p["seed"], best_epoch_a, best_epoch_b)
    # Resume consistency: first row of each later segment carries best_s1 == max eval so far.
    for si in range(1, len(segs)):
        e0 = seg_ranges[si][0]
        prior = [v for e, v in evals if e < e0]
        if prior:
            assert abs(em_by_epoch[e0]["best_s1"] - max(prior)) < 1e-12, (p["config"], p["seed"], si)
    L = int(cfg["num_blocks"])
    deep = L - 1
    last_epoch = max(lm_by_epoch)
    assert last_epoch == int(cfg["epochs_stage1"]), (p["config"], p["seed"], last_epoch)
    sep_best = lm_by_epoch[best_epoch_a][deep]["sep_curr_nl"]
    sep_last = lm_by_epoch[last_epoch][deep]["sep_curr_nl"]
    # Full per-block profile at both epochs (for the paper's per-block tables).
    sep_best_blocks = [lm_by_epoch[best_epoch_a][l]["sep_curr_nl"] for l in range(L)]
    sep_last_blocks = [lm_by_epoch[last_epoch][l]["sep_curr_nl"] for l in range(L)]
    # sep trajectory at eval epochs (deepest block), useful to see the late-training drift
    traj = {int(e): lm_by_epoch[e][deep]["sep_curr_nl"] for e, _ in evals if e in lm_by_epoch}
    acc = {"S1": fte["stage1_test_top1_no_tta"], "S1-TTA": fte["stage1_test_top1_tta"],
           "S2": fte["stage2_test_top1_no_tta"], "S2-TTA": fte["stage2_test_top1_tta"]}
    # Cross-check against saved prediction arrays when present.
    npz_check = {}
    for key, fn in [("S1", "preds_stage1_test_no_tta.npz"), ("S1-TTA", "preds_stage1_test_tta.npz"),
                    ("S2", "preds_stage2_test_no_tta.npz"), ("S2-TTA", "preds_stage2_test_tta.npz")]:
        f = f"{last}/{fn}"
        if os.path.exists(f):
            z = np.load(f)
            a = float((z["y_pred"] == z["y_true"]).mean())
            npz_check[key] = a
            assert abs(a - acc[key]) < 5e-5, (p["config"], p["seed"], key, a, acc[key])
    info.update({
        "config": p["config"], "family": p["family"], "seed": p["seed"],
        "dataset": "cifar100" if cfg["num_classes"] == 100 else "cifar10",
        "run_cfg": {k: cfg.get(k) for k in CFG_KEYS if k in cfg},
        "use_sam": bool(cfg.get("use_sam")),
        "best_s1_val": best_s1, "best_s2_val": rend["best_s2"],
        "stage1_best_epoch": best_epoch_a, "stage1_last_epoch": last_epoch,
        "stage1_best_epoch_segment": os.path.relpath(segs[seg_of_epoch[best_epoch_a]], CF),
        "sep_deep_at_best_epoch": sep_best, "sep_deep_at_last_epoch": sep_last,
        "sep_blocks_at_best_epoch": sep_best_blocks, "sep_blocks_at_last_epoch": sep_last_blocks,
        "sep_deep_at_eval_epochs": traj,
        "acc_test": acc, "acc_test_pct": {k: round(100 * v, 2) for k, v in acc.items()},
        "acc_from_preds_npz": npz_check,
        "events_sha256": sha256(f"{last}/events.jsonl"),
        "layer_metrics_sha256": [sha256(f"{s}/layer_metrics.jsonl") for s in segs],
    })
    return info


# ----------------------------------------------------------------------------- statistics
def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def demean(v, groups):
    v = np.asarray(v, float).copy()
    out = v.copy()
    for g in set(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        out[idx] = v[idx] - v[idx].mean()
    return out


def perm_pooled(x, y, rng, n_perm=N_PERM):
    obs = spearman(x, y)
    y = np.asarray(y, float)
    cnt = 0
    for _ in range(n_perm):
        if abs(spearman(x, rng.permutation(y))) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def perm_within(x, y, groups, rng, n_perm=N_PERM):
    xd, yd = demean(x, groups), demean(y, groups)
    obs = spearman(xd, yd)
    idx_by_g = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by_g[g].append(i)
    cnt = 0
    for _ in range(n_perm):
        yp = yd.copy()
        for g, idx in idx_by_g.items():
            yp[idx] = yd[rng.permutation(idx)]
        if abs(spearman(xd, yp)) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def exact_perm_means(xm, ym):
    obs = spearman(xm, ym)
    perms = list(itertools.permutations(range(len(ym))))
    cnt = sum(1 for pmt in perms if abs(spearman(xm, np.asarray(ym)[list(pmt)])) >= abs(obs) - 1e-12)
    return obs, cnt / len(perms), len(perms)


def correlations(rows, sep_key, acc_key, group_key, seed_offset):
    keep = [r for r in rows if r[group_key] is not None]
    groups = [r[group_key] for r in keep]
    # drop singleton groups for the within-config test (demeaned value is identically 0)
    counts = defaultdict(int)
    for g in groups:
        counts[g] += 1
    keep_w = [r for r in keep if counts[r[group_key]] > 1]
    g_w = [r[group_key] for r in keep_w]
    x_all = [r[sep_key] for r in rows]
    y_all = [r["acc_test"][acc_key] * 100 for r in rows]
    rng = np.random.default_rng(PERM_SEED + seed_offset)
    rho_p, p_p = perm_pooled(x_all, y_all, rng)
    rng = np.random.default_rng(PERM_SEED + seed_offset + 1)
    rho_w, p_w = perm_within([r[sep_key] for r in keep_w], [r["acc_test"][acc_key] * 100 for r in keep_w], g_w, rng)
    gnames = sorted(set(groups))
    xm = [np.mean([r[sep_key] for r in keep if r[group_key] == g]) for g in gnames]
    ym = [np.mean([r["acc_test"][acc_key] * 100 for r in keep if r[group_key] == g]) for g in gnames]
    out = {
        "pooled_all_points": {"spearman_rho": rho_p, "perm_p_two_sided": p_p, "n": len(rows), "n_perm": N_PERM},
        "within_config_demeaned": {"spearman_rho": rho_w, "perm_p_two_sided": p_w, "n": len(keep_w),
                                    "n_groups": len(set(g_w)),
                                    "singletons_dropped": sorted(g for g in counts if counts[g] == 1),
                                    "n_perm": N_PERM,
                                    "null": "acc permuted within each config; Spearman on within-config-demeaned (sep, acc)"},
    }
    if len(gnames) <= 8:
        rho_m, p_m, n_m = exact_perm_means(xm, ym)
        out["cross_config_means"] = {"spearman_rho": rho_m, "perm_p_two_sided_exact": p_m,
                                     "n": len(gnames), "n_perm_exact": n_m}
    out["per_config_spearman_descriptive"] = {
        g: spearman([r[sep_key] for r in keep if r[group_key] == g],
                    [r["acc_test"][acc_key] for r in keep if r[group_key] == g])
        for g in gnames if counts[g] > 1}
    return out


def sample_sd(v):
    v = np.asarray(v, float)
    return float(v.std(ddof=1)) if len(v) > 1 else None


def main():
    wiki = json.load(open(WIKI_JSON))
    wiki_pts = {(q["config"], q["seed"]): q for q in wiki["points"]}
    rows = [load_point(p) for p in POINTS]
    assert len(rows) == 19 and set((r["config"], r["seed"]) for r in rows) == set(wiki_pts)
    gates = []
    for r in rows:
        w = wiki_pts[(r["config"], r["seed"])]
        ok = abs(r["sep_deep_at_last_epoch"] - w["sep"]) < 1e-9
        gates.append({"check": f"{r['config']}/{r['seed']} last-epoch sep == wiki", "got": r["sep_deep_at_last_epoch"], "want": w["sep"], "ok": ok})
        ok2 = abs(r["acc_test"]["S2-TTA"] * 100 - w["acc_pct"]) < 1e-6
        gates.append({"check": f"{r['config']}/{r['seed']} S2-TTA == wiki", "got": r["acc_test"]["S2-TTA"] * 100, "want": w["acc_pct"], "ok": ok2})

    # Grouping 1: the 6 published configs.  Grouping 2: split by actual run config (D7, D13).
    for r in rows:
        r["group_published6"] = r["config"]
        tag = r["config"]
        if r["config"] == "c10_gamma0":
            tag += "__bcl%s_dol%s" % (r["run_cfg"].get("block_curr_lambda"), r["run_cfg"].get("depth_order_lambda"))
        if r["config"] == "gated_l4d128_k0":
            tag += "__sam_%s" % ("on" if r["use_sam"] else "off")
        r["group_by_run_config"] = tag

    # Reproduction gate: wiki's numbers from last-epoch sep and S2-TTA (6 configs).
    rep = correlations(rows, "sep_deep_at_last_epoch", "S2-TTA", "group_published6", 0)
    gates.append({"check": "reproduce wiki within-config rho (last-epoch sep, S2-TTA)",
                  "got": rep["within_config_demeaned"]["spearman_rho"],
                  "want": wiki["correlations"]["within_config"]["spearman_rho"],
                  "ok": abs(rep["within_config_demeaned"]["spearman_rho"] - wiki["correlations"]["within_config"]["spearman_rho"]) < 1e-9})
    gates.append({"check": "reproduce wiki pooled rho (last-epoch sep, S2-TTA)",
                  "got": rep["pooled_all_points"]["spearman_rho"],
                  "want": wiki["correlations"]["pooled_all_points"]["spearman_rho"],
                  "ok": abs(rep["pooled_all_points"]["spearman_rho"] - wiki["correlations"]["pooled_all_points"]["spearman_rho"]) < 1e-9})
    gates.append({"check": "reproduce wiki cross-config-means rho (last-epoch sep, S2-TTA)",
                  "got": rep["cross_config_means"]["spearman_rho"],
                  "want": wiki["correlations"]["cross_config_means"]["spearman_rho"],
                  "ok": abs(rep["cross_config_means"]["spearman_rho"] - wiki["correlations"]["cross_config_means"]["spearman_rho"]) < 1e-9})
    all_ok = all(g["ok"] for g in gates)

    results = {}
    off = 10
    for sep_key, sep_label in [("sep_deep_at_best_epoch", "aligned_stage1_best_epoch"),
                               ("sep_deep_at_last_epoch", "last_stage1_epoch_as_published")]:
        results[sep_label] = {}
        for acc_key in ["S1", "S1-TTA", "S2", "S2-TTA"]:
            results[sep_label][acc_key] = {}
            for gk in ["group_published6", "group_by_run_config"]:
                results[sep_label][acc_key][gk] = correlations(rows, sep_key, acc_key, gk, off)
                off += 10

    per_config = {}
    for g in sorted(set(r["config"] for r in rows)):
        rr = [r for r in rows if r["config"] == g]
        per_config[g] = {
            "seeds": [r["seed"] for r in rr],
            "use_sam": [r["use_sam"] for r in rr],
            "stage1_best_epoch": [r["stage1_best_epoch"] for r in rr],
            "sep_aligned": [r["sep_deep_at_best_epoch"] for r in rr],
            "sep_aligned_mean": float(np.mean([r["sep_deep_at_best_epoch"] for r in rr])),
            "sep_aligned_sd": sample_sd([r["sep_deep_at_best_epoch"] for r in rr]),
            "sep_last": [r["sep_deep_at_last_epoch"] for r in rr],
            "sep_last_mean": float(np.mean([r["sep_deep_at_last_epoch"] for r in rr])),
            "sep_last_sd": sample_sd([r["sep_deep_at_last_epoch"] for r in rr]),
            **{f"acc_{k}_pct": [r["acc_test_pct"][k] for r in rr] for k in ["S1", "S1-TTA", "S2", "S2-TTA"]},
            **{f"acc_{k}_mean_pct": float(np.mean([r["acc_test"][k] * 100 for r in rr])) for k in ["S1", "S1-TTA", "S2", "S2-TTA"]},
            **{f"acc_{k}_sd_pct": sample_sd([r["acc_test"][k] * 100 for r in rr]) for k in ["S1", "S1-TTA", "S2", "S2-TTA"]},
        }

    out = {
        "title": "E1 tier 1: epoch-aligned deepest-block sep_curr_nl vs validation-selected accuracy (19 points)",
        "generated": "2026-09-25",
        "script": os.path.relpath(os.path.abspath(__file__), CR),
        "definitions": {
            "sep": "deepest block's sep_curr_nl from layer_metrics.jsonl (Stage-1 training-time running mean over the epoch, raw training weights, trainer's own HNM negatives)",
            "stage1_best_epoch": "first Stage-1 val-eval epoch whose EMA val top-1 (eval_top1, FF aggregate, val TTA per config) equals run_end.best_s1; trainers use strict '>' so this is the epoch whose checkpoint is stage1_best.pt",
            "acc": "final_test_eval in the final segment's events.jsonl: S1/S1-TTA = stage1_best EMA backbone with the goodness-sum rule; S2/S2-TTA = stage2_best attentive head on the frozen stage1_best EMA backbone",
            "resumed_logs": "segments concatenated by epoch; lineage asserted (first segment starts at 1, later segments start after the previous ends, identical config.json, resumed best_s1 equals max prior eval)",
            "perm_p": "two-sided on |rho|, p=(count+1)/(n_perm+1), numpy default_rng seeded from %d" % PERM_SEED,
            "groupings": {
                "group_published6": "the 6 configs of the published analysis",
                "group_by_run_config": "C10 gamma=0 split by (block_curr_lambda, depth_order_lambda) [seed 42 = 0/0, seeds 123/456 = 0.25/0.2, D7]; kappa=0 D128 split by SAM [seed 42 off, 123/456 on, D13]; singleton groups dropped from the within-config test",
            },
            "sd": "sample SD (ddof=1)",
        },
        "gates": {"all_passed": all_ok, "checks": gates},
        "points": [{k: r[k] for k in ["config", "family", "seed", "dataset", "use_sam", "group_published6", "group_by_run_config",
                                      "stage1_best_epoch", "stage1_last_epoch", "best_s1_val", "best_s2_val",
                                      "sep_deep_at_best_epoch", "sep_deep_at_last_epoch",
                                      "sep_blocks_at_best_epoch", "sep_blocks_at_last_epoch", "sep_deep_at_eval_epochs",
                                      "acc_test", "acc_test_pct", "acc_from_preds_npz", "run_cfg",
                                      "segments", "segment_stage1_epoch_ranges", "stage1_best_epoch_segment",
                                      "stage1_epochs_missing_from_logs", "superseded_restarted_from_scratch",
                                      "events_sha256", "layer_metrics_sha256"]} for r in rows],
        "per_config": per_config,
        "correlations": results,
        "reproduction_of_published_analysis": rep,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("gates all passed:", all_ok)
    for g in gates:
        if not g["ok"]:
            print("  FAIL", g)
    print(f"{'config':18s} {'seed':>4s} {'SAM':>4s} {'best_ep':>7s} {'sep_best':>9s} {'sep_last':>9s} {'S1-TTA':>7s} {'S2-TTA':>7s}")
    for r in rows:
        print(f"{r['config']:18s} {r['seed']:4d} {str(r['use_sam']):>4s} {r['stage1_best_epoch']:7d} {r['sep_deep_at_best_epoch']:9.4f} {r['sep_deep_at_last_epoch']:9.4f} {r['acc_test_pct']['S1-TTA']:7.2f} {r['acc_test_pct']['S2-TTA']:7.2f}")
    for sl in results:
        for ak in ["S1-TTA", "S2-TTA"]:
            for gk in results[sl][ak]:
                c = results[sl][ak][gk]
                print(f"{sl:32s} {ak:6s} {gk:20s} pooled rho={c['pooled_all_points']['spearman_rho']:+.3f} p={c['pooled_all_points']['perm_p_two_sided']:.3f} | within rho={c['within_config_demeaned']['spearman_rho']:+.3f} p={c['within_config_demeaned']['perm_p_two_sided']:.3f} (n={c['within_config_demeaned']['n']}) | means rho={c.get('cross_config_means', {}).get('spearman_rho', float('nan')):+.3f}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
