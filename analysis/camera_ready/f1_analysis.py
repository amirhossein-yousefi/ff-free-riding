#!/usr/bin/env python3
"""F1 (MGC block-0 auxiliary control) analysis -> <FF_OUT_DIR>/results/f1_block0_control.json.

Arms (CIFAR-10 L4/D128, softplus SYMBA, constant gamma = 0.7, SAM on, 180 + 10 epochs, seeds 42/123/456):
  a   = f1a_cum_sam     cumulative gamma = 0.7 with the depth-scaled, residual-weighted current-block
                        auxiliary [0.25, 0.5, 0.75, 1.0] (the SAM-matched cumulative anchor)
  b   = f1b_cum_noaux   cumulative gamma = 0.7, current-block auxiliary removed at every block, no compensator
  c   = f1c_mgc_keepl0  MGC (c_d = 1) with block 0's 0.25 auxiliary kept
  mgc = MGC-D128        existing MGC (c_d = 1) runs (analysis/config.py MGC_D128_RUNS)
Registered predictions: experiments/f1_mgc_block0/PREDICTIONS.md (written before any F1 run; shipped as
analysis/experiments/f1_mgc_block0/PREDICTIONS.md).

What this script does (CPU only, numpy only; reads raw run files read-only):
  * per run: the four protocols from events.jsonl::final_test_eval (cross-checked against preds_*.npz),
    last-Stage-1-epoch per-block sep_curr_nl from layer_metrics.jsonl, hardware (env.json), resumes
    (events.jsonl resume events, job-log segments), config checks against arms.json / config_diffs;
  * per arm: mean +- sample SD (ddof = 1) over seeds;
  * contrasts X - Y for MGC-a, b-a, c-a, c-MGC, b-MGC: per-seed differences, mean +- sample SD of the
    paired differences, and an example-level paired bootstrap pooled over the three seeds (n = 30,000)
    with the procedure of analysis/aggregate.py (paired_stats + boot_means: 5000 resamples, percentile
    95% CI, fresh numpy default_rng(0) per contrast x protocol, as for mgc_paired_bootstrap.json), plus
    per-seed exact McNemar and prediction churn;
  * decomposition of the deepest-block separation gain MGC - a = (b - a) + (MGC - b) and the block-0
    effect c - MGC;
  * scoring of P1-P4 as PASS/FAIL;
  * the optional F_d (gradient attenuation) snapshot of experiments/f1_mgc_block0/fd_snapshot_f1.py,
    if its outputs exist, with the E3 gamma = 0.7 SAM-off reference on the same images and wrong labels,
    paired per-example F_d differences, and the ratio of the mean cumulative-term gradient to its
    gamma = 0 counterfactual at the same margins (recomputed from the saved m / P_prev arrays);
  * a plain-language 'summary' list whose numbers are all formatted from the values above.

Usage (CPU only, about a minute):
  FF_HOME=... FF_OUT_DIR=... CUDA_VISIBLE_DEVICES='' python analysis/camera_ready/f1_analysis.py

Paths ([CR] supplement port; analysis/README.md): the F1 run directories, job logs, arms.json,
config_diffs/ and the fork trainer.py are read from the F1 experiment folder of the run archive
(config.F1_DIR: $FF_F1_DIR, default $FF_HOME/amir-porjects/neuroips_review/camera_ready/experiments/
f1_mgc_block0). The F_d snapshot outputs are read from $FF_F1_FD_DIR, else
<FF_OUT_DIR>/experiments/f1_mgc_block0/out/fd (written by analysis/experiments/f1_mgc_block0/
fd_snapshot_f1.py), else the archive's <F1_DIR>/out/fd. The E3 reference arrays are read from
$FF_E3_OUT_DIR, else analysis/experiments/e3_fd/out (written by the E3 scripts), else the archive's
experiments/e3_fd/out. mgc_d128.json and e3_fd_ema.json are read from <FF_OUT_DIR>/results, else from
metric_summaries/camera_ready/. Absolute path prefixes, Colab Drive mount points and host names are
neutralized in the written JSON, as in the distributed copy.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import aggregate as A  # noqa: E402  (helpers and the paired-bootstrap procedure)

PROT = list(C.PROTOCOLS)
SEEDS = list(C.SEEDS)
F1 = C.F1_DIR  # [CR] supplement port: F1 experiment folder of the run archive (config.py)
F1_RUNS_DIR = F1 / "runs"
F1_LOGS = F1 / "logs"
OUT_NAME = "f1_block0_control"
_SUPP_CR = Path(__file__).resolve().parents[2] / "metric_summaries" / "camera_ready"
_SUPP_E3_OUT = Path(__file__).resolve().parents[1] / "experiments" / "e3_fd" / "out"


def _first_dir(env, *cands):
    """[CR] supplement port: $env if set, else the first existing candidate (else the last one)."""
    if os.environ.get(env):
        return Path(os.environ[env])
    return next((p for p in cands if p.exists()), cands[-1])


FD_DIR = _first_dir("FF_F1_FD_DIR", C.CR / "experiments" / "f1_mgc_block0" / "out" / "fd", F1 / "out" / "fd")
E3_OUT = _first_dir("FF_E3_OUT_DIR", _SUPP_E3_OUT, C.CR_ARCHIVE / "experiments" / "e3_fd" / "out")


def result_json(name):
    """[CR] supplement port: <FF_OUT_DIR>/results/<name>.json, else this supplement's copy."""
    for d in (C.RESULTS, _SUPP_CR):
        if (d / f"{name}.json").exists():
            return json.load(open(d / f"{name}.json"))
    raise FileNotFoundError(name)


def neutral(o):
    """[CR] supplement port: strip the archive prefix and the Colab Drive mount point from every string and
    replace DGX host names, recursively (the distributed JSON carries no absolute path or host name)."""
    if isinstance(o, dict):
        return {neutral(k): neutral(v) for k, v in o.items()}
    if isinstance(o, list):
        return [neutral(v) for v in o]
    if isinstance(o, str):
        if str(C.HOME) not in ("", "."):
            o = o.replace(str(C.HOME).rstrip("/") + "/", "")
        o = re.sub(r"/\S*?drive/MyDrive/", "MyDrive/", o)  # Colab Drive mount point -> Drive-relative
        return re.sub(r"\bspark-[0-9a-z]+\b", "<DGX Spark host>", o)
    return o

# every run directory named explicitly (timestamp included), as in config.py
_F1_TS = {
    "a": ("f1a_cum_sam", {42: "20260926_224359", 123: "20260926_142507", 456: "20260926_142503"}),
    "b": ("f1b_cum_noaux", {42: "20260925_101141", 123: "20260926_134435", 456: "20260927_161811"}),
    "c": ("f1c_mgc_keepl0", {42: "20260926_001337", 123: "20260927_024824", 456: "20260926_224355"}),
}
TAGS = {arm: tag for arm, (tag, _) in _F1_TS.items()}
TAGS["mgc"] = "mgc_c1 (MGC-D128)"
RUNS = {arm: {s: F1_RUNS_DIR / f"ff_cifar10_{tag}_L4_D128_bs512_seed{s}_{ts}" for s, ts in d.items()}
        for arm, (tag, d) in _F1_TS.items()}
RUNS["mgc"] = dict(C.MGC_D128_RUNS)
ARMS = ["mgc", "a", "b", "c"]
CONTRASTS = [("mgc", "a"), ("b", "a"), ("c", "a"), ("c", "mgc"), ("b", "mgc")]

# expected flags per arm (arms.json; MGC-D128 config.json has use_mgc=true and no F1 keys)
EXPECT = {
    "a": {"use_mgc": False, "curr_aux_off": False, "mgc_keep_l0_aux": False},
    "b": {"use_mgc": False, "curr_aux_off": True, "mgc_keep_l0_aux": False},
    "c": {"use_mgc": True, "curr_aux_off": False, "mgc_keep_l0_aux": True},
    "mgc": {"use_mgc": True},
}
EXPECT_LAMBDA_EFF = {"b": [0.0, 0.0, 0.0, 0.0], "c": [0.25, 0.0, 0.0, 0.0]}
MGC_MGC_LOG_S42 = C.MR / "log_mgc_c1_seed42.txt"

# registered thresholds (PREDICTIONS.md, printed values)
REG = {"mgc_sep_mean_printed": 5.25, "band": [4.72, 5.78], "p2_below": 4.72,
       "lowest_mgc_seed_printed": 5.21, "s2tta_band": [86.27, 88.27], "p4_below": 4.72,
       "mgc_s2tta_mean_printed": 87.27}

r8, summ, rel = A.r8, A.summ, A.rel
GATES: list = []


def gate(check, ok, **kw):
    GATES.append({"check": check, "pass": bool(ok), **kw})
    return ok


# ----------------------------------------------------------------------------- per run
def job_log(arm, s):
    if arm == "mgc":
        return MGC_MGC_LOG_S42 if s == 42 else None
    tag = TAGS[arm]
    for p in (F1_LOGS / f"{tag}_seed{s}.log", F1_LOGS / "colab" / f"{tag}_seed{s}.log"):
        if p.exists():
            return p
    return None


def parse_log(p: Path):
    if p is None:
        return None
    txt = p.read_text(errors="replace")
    starts = re.findall(r"^=== .*START segment.*$", txt, flags=re.M)
    sha = sorted(set(re.findall(r"trainer sha256:? ([0-9a-f]{16})", txt)))
    env = dict(re.findall(r"^=== env (FF_[A-Z0-9_]+)=(\S*)$", txt, flags=re.M))
    hosts = sorted(set(re.findall(r"START segment:? .*? on (.+?)(?:;|$)", txt, flags=re.M)))
    return {"log": rel(p), "n_start_segments": len(starts), "segment_host_or_device": hosts,
            "n_from_scratch": len(re.findall(r"No checkpoint found\. Starting Stage", txt)),
            "n_checkpoint_resume_loads": len(re.findall(r"\[Checkpoint\] Loaded from ", txt)),
            "trainer_sha256_prefix": sha, "env": env,
            "segment_headers": [x.strip("= ").strip() for x in starts]}


def read_run(arm, s):
    d = RUNS[arm][s]
    assert d.is_dir(), d
    fe = A.final_eval(d)
    acc = A.accs(d)
    ev = [json.loads(l) for l in open(d / "events.jsonl") if l.strip()]
    resumes = [e for e in ev if str(e.get("event", "")).startswith("resume")]
    ep, last, rows = A.last_epoch_layers(d)
    nblk = len(last)
    # contiguity of the Stage-1 layer log (resumed runs would start later or repeat epochs)
    keys = [(r["epoch"], r["layer"]) for r in rows]
    eps = sorted(set(k[0] for k in keys))
    contiguous = eps == list(range(1, ep + 1)) and len(keys) == len(set(keys)) == ep * nblk
    env = json.load(open(d / "env.json"))
    cfg = json.load(open(d / "config.json"))
    # accuracy cross-check against the per-example predictions
    npz_acc = {}
    for p in PROT:
        y, yp = A.load_preds(d, p)
        npz_acc[p] = r8(100 * float((y == yp).mean()))
    rec = {
        "arm": arm, "tag": TAGS[arm], "seed": s, "run_dir": rel(d),
        **acc,
        "top5": {p: A.pp(fe[C.PROTOCOLS[p].replace("top1", "top5")]) for p in PROT},
        "final_stage1_epoch": ep,
        "first_stage1_epoch_logged": min(eps),
        "stage1_layer_log_contiguous_1_to_last": bool(contiguous),
        "sep_curr_nl": [r8(r["sep_curr_nl"]) for r in last],
        "sep_curr_ni": [r8(r["sep_curr_ni"]) for r in last],
        "gpos_cur": [r8(r["gpos_cur"]) for r in last],
        "hardware": {"device0": env.get("device0"), "torch": env.get("torch"), "cuda": env.get("cuda_version"),
                     "python": env.get("python"), "platform": env.get("platform"),
                     "short": "GB10" if "GB10" in str(env.get("device0")) else ("A100" if "A100" in str(env.get("device0")) else env.get("device0"))},
        "resume_events": [{k: v for k, v in e.items() if k != "ts"} for e in resumes],
        "config": {k: cfg.get(k) for k in ("seed", "use_sam", "gamma_scale", "gamma_gating_mode", "use_mgc",
                                           "curr_aux_off", "mgc_keep_l0_aux", "mgc_c0", "mgc_rho",
                                           "block_curr_lambda", "block_curr_depth_slope", "epochs_stage1",
                                           "epochs_stage2", "num_blocks", "d_model", "batch_size")},
        "acc_from_preds_npz": npz_acc,
    }
    lg = parse_log(job_log(arm, s))
    rec["job_log"] = lg
    n_seg = lg["n_start_segments"] if lg else None
    rec["resumed"] = (bool(resumes) or (n_seg is not None and n_seg > 1) or not contiguous
                      or bool(lg and lg["n_checkpoint_resume_loads"] > 0))
    rec["resume_evidence"] = (
        f"{len(resumes)} resume events in events.jsonl; "
        + (f"{n_seg} START segment(s), {lg['n_from_scratch']} from-scratch start(s) and "
           f"{lg['n_checkpoint_resume_loads']} checkpoint resume load(s) in the job log; " if lg and n_seg else
           (f"job log: {lg['n_from_scratch']} from-scratch start, {lg['n_checkpoint_resume_loads']} checkpoint resume "
            f"load(s); " if lg else "no job log on this host; "))
        + f"Stage-1 layer log epochs {min(eps)}-{ep}, contiguous={contiguous}"
        + ("" if arm != "mgc" else "; the MGC-D128 trainer did not log resume events, so a resume is detectable only "
           "through a new run directory / a layer log that starts after epoch 1"))
    if arm in EXPECT_LAMBDA_EFF:
        raw = [json.loads(l) for l in open(d / "layer_metrics_raw.jsonl") if l.strip()]
        lam = {}
        for r in raw:
            if r.get("stage", 1) == 1 and "block_curr_lambda_eff" in r:
                lam.setdefault(r["layer"], set()).add(round(float(r["block_curr_lambda_eff"]), 8))
        rec["block_curr_lambda_eff_logged_values"] = {str(k): sorted(v) for k, v in sorted(lam.items())}
    # gates
    gate(f"{arm} s{s}: one final_test_eval, final Stage-1 epoch 180", ep == 180)
    gate(f"{arm} s{s}: no resume", not rec["resumed"], evidence=rec["resume_evidence"])
    gate(f"{arm} s{s}: final_test_eval equals preds_*.npz accuracy (all 4 protocols, 0.005 pp)",
         all(abs(acc[p] - npz_acc[p]) <= 0.005 for p in PROT))
    gate(f"{arm} s{s}: config flags as registered", all(cfg.get(k) == v for k, v in EXPECT[arm].items())
         and cfg.get("use_sam") is True and cfg.get("gamma_scale") == 0.7 and cfg.get("gamma_gating_mode") == "constant"
         and cfg.get("seed") == s and cfg.get("epochs_stage1") == 180 and cfg.get("epochs_stage2") == 10)
    if arm in EXPECT_LAMBDA_EFF:
        want = EXPECT_LAMBDA_EFF[arm]
        gate(f"{arm} s{s}: logged block_curr_lambda_eff = {want} at every Stage-1 step log",
             rec["block_curr_lambda_eff_logged_values"] == {str(i): [w] for i, w in enumerate(want)})
    if lg and arm != "mgc":
        gate(f"{arm} s{s}: job log trainer sha256 prefix = local trainer.py", lg["trainer_sha256_prefix"] == [TRAINER_SHA[:16]])
        arm_env = {**ARMS_JSON["common_env"], **ARMS_JSON["arms"][arm]["env"], "FF_SEED": str(s), "FF_VERSION_TAG": TAGS[arm]}
        gate(f"{arm} s{s}: job-log env = arms.json env", all(lg["env"].get(k) == v for k, v in arm_env.items())
             and not any(k in lg["env"] for k in ("FF_CURR_AUX_OFF", "FF_MGC_KEEP_L0_AUX", "FF_USE_MGC") if k not in arm_env))
        # config diff vs the seed-matched MGC-D128 run equals the planned diff (config_diffs/)
        got = A.config_diff(RUNS["mgc"][s], d)
        got = {k: [("<MISSING>" if v[0] is None else v[0]), v[1]] for k, v in got.items()}
        want = CONFIG_DIFFS["arms"][arm]["seeds"][str(s)]["diff_vs_reference_same_seed"]
        gate(f"{arm} s{s}: config diff vs seed-matched MGC-D128 = planned diff", got == want, got=got, planned=want)
    return rec


# ----------------------------------------------------------------------------- statistics
def paired_seed_diffs(x, y, per_run):
    out = {}
    for p in PROT + ["sep_deepest"]:
        vals = []
        for s in SEEDS:
            a, b = per_run[x][s], per_run[y][s]
            if p == "sep_deepest":
                vals.append(a["sep_curr_nl"][-1] - b["sep_curr_nl"][-1])
            else:
                vals.append(a[p] - b[p])
        out[p] = {"per_seed": {str(s): r8(v) for s, v in zip(SEEDS, vals)}, **{k: v for k, v in summ(vals).items() if k != "values"}}
    nblk = len(per_run[x][SEEDS[0]]["sep_curr_nl"])
    out["sep_per_block"] = [
        {"per_seed": {str(s): r8(per_run[x][s]["sep_curr_nl"][bl] - per_run[y][s]["sep_curr_nl"][bl]) for s in SEEDS},
         **{k: v for k, v in summ([per_run[x][s]["sep_curr_nl"][bl] - per_run[y][s]["sep_curr_nl"][bl] for s in SEEDS]).items() if k != "values"}}
        for bl in range(nblk)]
    return out


def bootstrap_contrast(x, y):
    """analysis/aggregate.py procedure (as for mgc_paired_bootstrap.json): per protocol a fresh
    default_rng(0), pooled per-example correctness delta over seeds 42/123/456, 5000 resamples."""
    out = {}
    for p in PROT:
        rng = np.random.default_rng(C.BOOT_SEED)
        per_seed, delta = A.paired_stats(RUNS[x], RUNS[y], p)
        means = A.boot_means(delta, rng)
        blk = A.ci_block(delta, means)
        blk["ci_excludes_zero"] = bool(blk["ci95_pp"][0] > 0 or blk["ci95_pp"][1] < 0)
        churn = [per_seed[str(s)]["disagreement_pct"] for s in SEEDS]
        out[p] = {"per_seed": per_seed, "pooled": blk,
                  "churn_pct": {"per_seed": {str(s): c for s, c in zip(SEEDS, churn)},
                                **{k: v for k, v in summ(churn).items() if k != "values"}}}
    return out


def within_arm_seed_churn(arm):
    """Noise floor: prediction disagreement between two seeds of the same arm."""
    pairs = [(42, 123), (42, 456), (123, 456)]
    out = {}
    for p in PROT:
        vals = {}
        for s1, s2 in pairs:
            y1, p1 = A.load_preds(RUNS[arm][s1], p)
            y2, p2 = A.load_preds(RUNS[arm][s2], p)
            assert np.array_equal(y1, y2)
            vals[f"{s1}_vs_{s2}"] = r8(100 * float((p1 != p2).mean()))
        out[p] = {"pairs": vals, **{k: v for k, v in summ(list(vals.values())).items() if k != "values"}}
    return out


def verdict(ok):
    return "PASS" if ok else "FAIL"


# ----------------------------------------------------------------------------- F_d (optional)
def fd_block():
    d = FD_DIR
    files = sorted(d.glob("fd_*_*.json")) if d.exists() else []
    if not files:
        return {"status": "not run (experiments/f1_mgc_block0/fd_snapshot_f1.py outputs absent)"}
    recs = {f.stem: json.load(open(f)) for f in files}
    # "script": its path in the camera-ready repository (here: analysis/experiments/f1_mgc_block0/fd_snapshot_f1.py)
    out = {"status": "measured", "script": "experiments/f1_mgc_block0/fd_snapshot_f1.py",
           "definition": "F_d = mean over examples of 1 - min(1, R), R = (1 + e^{beta m}) / (1 + e^{beta (m + gamma P_prev)}), "
                         "beta = 4, gamma = 0.7; m = current-block goodness margin (true minus wrong label), P_prev = sum of earlier "
                         "blocks' margins; CI95 = 1.96 sd(ddof=0)/sqrt(N) (published convention; experiments/e3_fd/fd_stats.py)",
           "scope": "seed 42 only; first 1024 CIFAR-10 test images; EMA stage1_best weights; CPU. F_d measures the attenuation of the "
                    "gradient that the cumulative SYMBA term sends to block d's own margin; arm a additionally carries an unattenuated "
                    "residual-weighted current-block auxiliary, arm b does not.",
           "variants": {}}
    for k, r in recs.items():
        s = r["stats"]
        out["variants"][k] = {
            "arm": r["arm"], "seed": r["seed"], "neg_labels": r["neg_labels"],
            "F_d": [r8(x) for x in s["F_d"]], "F_d_ci95": [r8(x) for x in s["F_d_ci95"]],
            "median_R": s["median_R_d"], "coverage_pct": [r8(x) for x in s["assumption_coverage_pct"]],
            "m_mean": [r8(x) for x in s["m_mean"]], "P_prev_mean": [r8(x) for x in s["P_prev_mean"]],
            "frac_R_le_1e-2_pct": [r8(x) for x in s["frac_fully_starved_R_le_1e-2_pct"]],
            "grad_mean": [r8(x) for x in s["grad_mean"]], "grad_local_mean": [r8(x) for x in s["grad_local_mean"]],
            "ckpt": r["ckpt_meta"]["ckpt_origin"], "ckpt_md5": r["ckpt_meta"]["ckpt_md5"],
            "ckpt_epoch_stage1": r["ckpt_meta"]["ckpt_epoch_stage1"], "ckpt_cfg": r["ckpt_meta"]["ckpt_cfg"],
            "strict_load_missing_unexpected": [len(r["ckpt_meta"]["missing"]), len(r["ckpt_meta"]["unexpected"])],
            "acc_gate": r["acc_gate"],
        }
        ag = r["acc_gate"]
        if ag is not None:
            # E3's registered gate (experiments/e3_fd/PREDICTIONS.md): |acc_cpu - logged| <= 0.0002 (2 examples)
            # and per-example agreement with preds_stage1_test_no_tta.npz >= 99.9%. The snapshot script's own
            # 'passed' flag additionally required <= 2 disagreeing examples; that stricter, unregistered check is
            # reported, not gated (GPU-vs-CPU numerics and top-2 MoE routing flips move a few examples).
            agree = 1 - ag["n_disagree_with_logged_preds_npz"] / ag["n"]
            e3_ok = ag["abs_diff_vs_logged"] <= 0.0002 + 1e-12 and agree >= 0.999
            out["variants"][k]["acc_gate_e3_criteria_pass"] = bool(e3_ok)
            out["variants"][k]["acc_gate_strict_le2_disagreements_pass"] = bool(ag["n_disagree_with_logged_preds_npz"] <= 2)
            if r["neg_labels"] == "published":
                gate(f"F_d {k}: EMA S1 no-TTA CPU accuracy within 2 examples of the log, per-example agreement >= 99.9% "
                     f"(E3 criteria), strict net_ema load",
                     e3_ok and not r["ckpt_meta"]["missing"] and not r["ckpt_meta"]["unexpected"],
                     cpu=ag["acc_s1_no_tta_cpu"], logged=ag["logged_stage1_test_top1_no_tta"],
                     n_disagree=ag["n_disagree_with_logged_preds_npz"], agreement=r8(agree))
    # E3 reference (SAM-off published cumulative gamma = 0.7, same images and recovered wrong labels)
    e3 = result_json("e3_fd_ema")
    h = {x["gamma"]: x for x in e3["headline_ema_vs_published"]}
    out["e3_reference_same_images_and_labels"] = {
        "source": "results/e3_fd_ema.json headline_ema_vs_published (EMA, fixed std, published wrong labels)",
        "gamma0.7_cumulative_sam_off_seed42_F_d": [r8(x) for x in h[0.7]["F_d_ema"]],
        "gamma0.7_cumulative_sam_off_seed42_coverage_pct": [r8(x) for x in h[0.7]["coverage_pct_ema"]],
        "gamma0_F_d": [r8(x) for x in h[0.0]["F_d_ema"]],
    }
    # paired per-example comparisons on the same 1024 images and wrong labels
    pairs = {}

    def load(stem):
        z = np.load(FD_DIR / f"{stem}.npz")
        return z["m"], z["P_prev"]

    def Fvec(m, P, g=0.7):
        return 1 - np.minimum(1, np.exp(np.logaddexp(0, 4.0 * m) - np.logaddexp(0, 4.0 * (m + g * P))))

    def grad(M, alpha=4.0):
        """|d softplus(-alpha M) / dm| = alpha * sigmoid(-alpha M), overflow-safe."""
        return alpha * np.exp(-np.logaddexp(0, alpha * M))

    # recompute the stored gradient summaries from the arrays (consistency) and add the ratio of the mean
    # cumulative-term gradient to the mean gamma = 0 counterfactual gradient at the same margins
    for k in recs:
        m_, P_ = load(k)
        g_cum, g_loc = grad(m_ + 0.7 * P_).mean(axis=1), grad(m_).mean(axis=1)
        v = out["variants"][k]
        gate(f"F_d {k}: grad_mean / grad_local_mean recomputed from the npz arrays (1e-9)",
             np.allclose(g_cum, v["grad_mean"], atol=1e-9) and np.allclose(g_loc, v["grad_local_mean"], atol=1e-9)
             and np.allclose(Fvec(m_, P_).mean(axis=1), v["F_d"], atol=1e-8))
        v["grad_mean_over_grad_local_mean"] = [r8(a / b) if b > 0 else None for a, b in zip(g_cum, g_loc)]
    out["gradient_fields"] = ("grad_mean = mean over examples of beta * sigmoid(-beta (m + gamma P_prev)), the magnitude of the "
                              "cumulative SYMBA term's gradient w.r.t. block d's own margin; grad_local_mean = the same with gamma = 0 "
                              "at the same margins (a counterfactual, not a trained gamma = 0 model); their ratio is a mean-level "
                              "attenuation that weights hard examples (small or negative m) most. Snapshot at the stage1_best EMA "
                              "weights (Stage-1 epoch 180); it does not describe the training trajectory.")
    # the arm-a checkpoint is a local copy of the Colab/Drive file: its MD5 was compared with Drive by hand
    # (rclone md5sum gdrive:ffcr/f1_mgc_block0/checkpoints/, 2026-09-28)
    drive_md5_a42 = "4dcfc5ae328a25cbd6c9e32520f358da"
    for k in recs:
        if recs[k]["arm"] == "a" and recs[k]["seed"] == 42:
            out["variants"][k]["ckpt_md5_equals_drive_md5_2026-09-28"] = out["variants"][k]["ckpt_md5"] == drive_md5_a42
            gate(f"F_d {k}: local arm-a seed-42 checkpoint MD5 equals the Drive MD5", out["variants"][k]["ckpt_md5"] == drive_md5_a42)
    # the F1 snapshot read CIFAR-10 from F1's work/data_c10 link, E3 from revised_training/data_c10: same test batch?
    import hashlib
    tb = [Path(recs[next(iter(recs))]["setup"]["data_dir"]) / "cifar-10-batches-py" / "test_batch",
          C.CF / "revised_training" / "data_c10" / "cifar-10-batches-py" / "test_batch"]
    tb_md5 = [hashlib.md5(open(x, "rb").read()).hexdigest() for x in tb]
    out["test_batch_md5_f1_vs_e3_data_dir"] = {"files": [rel(x) for x in tb], "md5": tb_md5}
    gate("F_d: the F1 snapshot and E3 read the identical CIFAR-10 test_batch (MD5)", tb_md5[0] == tb_md5[1])
    e3_npz0 = E3_OUT / "ema_fixed_published" / "grad_attenuation_arrays_v1.npz"
    if e3_npz0.exists():
        z = np.load(e3_npz0)
        m_, P_ = z["m"].astype(np.float64), z["P_prev"].astype(np.float64)
        g_cum, g_loc = grad(m_ + 0.7 * P_).mean(axis=1), grad(m_).mean(axis=1)
        out["e3_reference_same_images_and_labels"].update({
            "gamma0.7_cumulative_sam_off_seed42_F_d_recomputed_from_arrays": [r8(x) for x in Fvec(m_, P_).mean(axis=1)],
            "gamma0.7_cumulative_sam_off_seed42_m_mean": [r8(x) for x in m_.mean(axis=1)],
            "gamma0.7_cumulative_sam_off_seed42_P_prev_mean": [r8(x) for x in P_.mean(axis=1)],
            "gamma0.7_cumulative_sam_off_seed42_grad_mean": [r8(x) for x in g_cum],
            "gamma0.7_cumulative_sam_off_seed42_grad_local_mean": [r8(x) for x in g_loc],
            "gamma0.7_cumulative_sam_off_seed42_grad_mean_over_grad_local_mean": [r8(a / b) for a, b in zip(g_cum, g_loc)],
            "arrays": rel(e3_npz0)})
        gate("F_d E3 reference: F_d recomputed from the E3 arrays equals results/e3_fd_ema.json (1e-8)",
             np.allclose(Fvec(m_, P_).mean(axis=1), out["e3_reference_same_images_and_labels"]["gamma0.7_cumulative_sam_off_seed42_F_d"], atol=1e-8))

    have = {k for k in recs}
    if {"fd_b42_published", "fd_a42_published"} <= have:
        fb, fa = Fvec(*load("fd_b42_published")), Fvec(*load("fd_a42_published"))
        dF = fb - fa
        pairs["b42_minus_a42_published_labels"] = {
            "delta_F_d": [r8(x) for x in dF.mean(axis=1)],
            "delta_F_d_ci95_paired": [r8(x) for x in 1.96 * dF.std(axis=1, ddof=1) / np.sqrt(dF.shape[1])]}
    e3_npz = E3_OUT / "ema_fixed_published" / "grad_attenuation_arrays_v1.npz"
    if "fd_b42_published" in have and e3_npz.exists():
        z = np.load(e3_npz)
        assert abs(float(z["gamma"]) - 0.7) < 1e-12
        fe3 = Fvec(z["m"].astype(np.float64), z["P_prev"].astype(np.float64))
        for k in ("b42", "a42"):
            if f"fd_{k}_published" in have:
                dF = Fvec(*load(f"fd_{k}_published")) - fe3
                pairs[f"{k}_minus_e3_gamma0.7_samoff"] = {
                    "delta_F_d": [r8(x) for x in dF.mean(axis=1)],
                    "delta_F_d_ci95_paired": [r8(x) for x in 1.96 * dF.std(axis=1, ddof=1) / np.sqrt(dF.shape[1])]}
    out["paired_per_example"] = pairs
    if {"fd_b42_published", "fd_a42_published"} <= have:
        vb, va = out["variants"]["fd_b42_published"], out["variants"]["fd_a42_published"]
        e3r = out["e3_reference_same_images_and_labels"]
        d3 = pairs["b42_minus_a42_published_labels"]
        out["answer"] = (
            f"Attenuation is still present in arm b. Seed 42, published wrong labels: F_d(blocks 1-3) = "
            f"{vb['F_d'][1]:.3f}/{vb['F_d'][2]:.3f}/{vb['F_d'][3]:.3f} in arm b (no auxiliary), "
            f"{va['F_d'][1]:.3f}/{va['F_d'][2]:.3f}/{va['F_d'][3]:.3f} in arm a (SAM-matched cumulative with the auxiliary), "
            f"{e3r['gamma0.7_cumulative_sam_off_seed42_F_d'][1]:.3f}/{e3r['gamma0.7_cumulative_sam_off_seed42_F_d'][2]:.3f}/"
            f"{e3r['gamma0.7_cumulative_sam_off_seed42_F_d'][3]:.3f} in the E3 SAM-off gamma = 0.7 reference; paired b - a at block 3: "
            f"{d3['delta_F_d'][3]:+.4f} +/- {d3['delta_F_d_ci95_paired'][3]:.4f}. Arm b's larger F_d follows its larger earlier-block "
            f"margins (P_prev mean at block 3 {vb['P_prev_mean'][3]:.2f} vs {va['P_prev_mean'][3]:.2f}). Its own block-3 margins are also "
            f"larger (m mean {vb['m_mean'][3]:.2f} vs {va['m_mean'][3]:.2f}), so the gamma = 0 counterfactual gradient is already small: "
            f"the mean cumulative-term gradient is {vb['grad_mean_over_grad_local_mean'][3]:.2f} of the counterfactual in arm b vs "
            f"{va['grad_mean_over_grad_local_mean'][3]:.2f} in arm a. The per-example index F_d therefore does not separate the "
            f"high-separation arm b from the low-separation arm a at this snapshot (n = 1 seed per arm, one checkpoint each).")
    return out


# ----------------------------------------------------------------------------- main
ARMS_JSON = json.load(open(F1 / "arms.json"))
CONFIG_DIFFS = json.load(open(F1 / "config_diffs" / "config_diffs.json"))
TRAINER_SHA = A.sha256(F1 / "trainer.py")


def main():
    per_run = {arm: {s: read_run(arm, s) for s in SEEDS} for arm in ARMS}

    # reference values printed in PREDICTIONS.md recomputed from raw and checked against results/mgc_d128.json
    ref = result_json("mgc_d128")["mgc"]
    for s in SEEDS:
        for p in PROT:
            gate(f"MGC-D128 s{s} {p} equals results/mgc_d128.json", abs(per_run["mgc"][s][p] - ref["per_seed"][str(s)][p]) < 1e-9)
        gate(f"MGC-D128 s{s} sep_curr_nl equals results/mgc_d128.json",
             np.allclose(per_run["mgc"][s]["sep_curr_nl"], ref["per_seed"][str(s)]["sep_curr_nl"], atol=1e-8))

    arms = {}
    for arm in ARMS:
        agg = {p: summ([per_run[arm][s][p] for s in SEEDS]) for p in PROT}
        nblk = len(per_run[arm][SEEDS[0]]["sep_curr_nl"])
        agg["sep_curr_nl_per_block"] = [summ([per_run[arm][s]["sep_curr_nl"][b] for s in SEEDS]) for b in range(nblk)]
        agg["sep_curr_nl_deepest"] = agg["sep_curr_nl_per_block"][-1]
        arms[arm] = {"tag": TAGS[arm], "n": len(SEEDS),
                     "hardware_by_seed": {str(s): per_run[arm][s]["hardware"]["short"] for s in SEEDS},
                     "resumed_by_seed": {str(s): per_run[arm][s]["resumed"] for s in SEEDS},
                     "agg": agg}

    contrasts = {}
    for x, y in CONTRASTS:
        name = f"{x}_minus_{y}"
        contrasts[name] = {
            "hardware_by_seed": {str(s): f"{per_run[x][s]['hardware']['short']} vs {per_run[y][s]['hardware']['short']}" for s in SEEDS},
            "same_hardware_seeds": [s for s in SEEDS if per_run[x][s]["hardware"]["device0"] == per_run[y][s]["hardware"]["device0"]],
            "same_software_stack_seeds": [s for s in SEEDS if per_run[x][s]["hardware"]["torch"] == per_run[y][s]["hardware"]["torch"]
                                          and per_run[x][s]["hardware"]["device0"] == per_run[y][s]["hardware"]["device0"]],
            "seed_paired_differences": paired_seed_diffs(x, y, per_run),
            "example_level_paired_bootstrap": bootstrap_contrast(x, y),
        }
        same_hw = contrasts[name]["same_hardware_seeds"]
        spd = contrasts[name]["seed_paired_differences"]
        contrasts[name]["same_hardware_seed_mean"] = {
            "seeds": same_hw,
            "note": "mean of the seed-paired differences over the seeds whose two runs used the same GPU model "
                    "(the MGC-D128 seed-42 run used torch 2.9.1 on the GB10, the F1 GB10 runs torch 2.14.0)",
            **{p: (r8(st.mean(spd[p]["per_seed"][str(s)] for s in same_hw)) if same_hw else None) for p in PROT + ["sep_deepest"]}}
        # the pooled preds-based delta must equal the mean of the final_test_eval deltas (same 10k test order)
        for p in PROT:
            pooled = contrasts[name]["example_level_paired_bootstrap"][p]["pooled"]["delta_pp"]
            gate(f"{name} {p}: pooled preds delta equals mean of seed deltas (0.005 pp)",
                 abs(pooled - contrasts[name]["seed_paired_differences"][p]["mean"]) <= 0.005)

    churn_floor = {arm: within_arm_seed_churn(arm) for arm in ARMS}

    # decomposition of the deepest-block separation gain
    def sep(arm, s, b=-1):
        return per_run[arm][s]["sep_curr_nl"][b]

    decomp_seed = {}
    for s in SEEDS:
        gain = sep("mgc", s) - sep("a", s)
        aux = sep("b", s) - sep("a", s)
        comp = sep("mgc", s) - sep("b", s)
        decomp_seed[str(s)] = {"gain_mgc_minus_a": r8(gain), "aux_removal_b_minus_a": r8(aux),
                               "compensation_mgc_minus_b": r8(comp), "aux_removal_share": r8(aux / gain),
                               "compensation_share": r8(comp / gain), "block0_effect_c_minus_mgc": r8(sep("c", s) - sep("mgc", s))}
    m = lambda arm: st.mean(sep(arm, s) for s in SEEDS)  # noqa: E731
    gain_m, aux_m, comp_m = m("mgc") - m("a"), m("b") - m("a"), m("mgc") - m("b")
    decomposition = {
        "metric": "deepest-block (block 3) sep_curr_nl at the last Stage-1 epoch (180)",
        "identity": "(MGC - a) = (b - a) [auxiliary removal] + (MGC - b) [compensation]; exact per seed",
        "per_seed": decomp_seed,
        "mean": {"gain_mgc_minus_a": r8(gain_m), "aux_removal_b_minus_a": r8(aux_m), "compensation_mgc_minus_b": r8(comp_m),
                 "aux_removal_share_of_mean_gain": r8(aux_m / gain_m), "compensation_share_of_mean_gain": r8(comp_m / gain_m),
                 "block0_effect_c_minus_mgc": r8(m("c") - m("mgc"))},
        "sd_of_seed_values": {k: r8(st.stdev([decomp_seed[str(s)][k] for s in SEEDS]))
                              for k in ("gain_mgc_minus_a", "aux_removal_b_minus_a", "compensation_mgc_minus_b",
                                        "aux_removal_share", "compensation_share", "block0_effect_c_minus_mgc")},
        "per_block_means": [
            {"block": b, "a": r8(st.mean(sep("a", s, b) for s in SEEDS)), "b": r8(st.mean(sep("b", s, b) for s in SEEDS)),
             "mgc": r8(st.mean(sep("mgc", s, b) for s in SEEDS)), "c": r8(st.mean(sep("c", s, b) for s in SEEDS)),
             "gain_mgc_minus_a": r8(st.mean(sep("mgc", s, b) - sep("a", s, b) for s in SEEDS)),
             "aux_removal_b_minus_a": r8(st.mean(sep("b", s, b) - sep("a", s, b) for s in SEEDS)),
             "compensation_mgc_minus_b": r8(st.mean(sep("mgc", s, b) - sep("b", s, b) for s in SEEDS)),
             "block0_effect_c_minus_mgc": r8(st.mean(sep("c", s, b) - sep("mgc", s, b) for s in SEEDS))}
            for b in range(4)],
    }

    # ---------------- registered predictions
    sd = {arm: arms[arm]["agg"]["sep_curr_nl_deepest"] for arm in ARMS}
    mgc_min_seed = min(sd["mgc"]["values"])
    band = REG["band"]
    exact_band = [0.9 * sd["mgc"]["mean"], 1.1 * sd["mgc"]["mean"]]
    p1_ok = band[0] <= sd["c"]["mean"] <= band[1]
    p2_mean_ok = sd["b"]["mean"] < REG["p2_below"]
    p2_seed_ok = all(v < mgc_min_seed for v in sd["b"]["values"])
    p3 = {arm: {"s2tta_mean": arms[arm]["agg"]["S2-TTA"]["mean"], "s2tta_sd": arms[arm]["agg"]["S2-TTA"]["sd"],
                "in_band": REG["s2tta_band"][0] <= arms[arm]["agg"]["S2-TTA"]["mean"] <= REG["s2tta_band"][1]} for arm in ("a", "b", "c")}
    p4_ok = sd["a"]["mean"] < REG["p4_below"]
    predictions = {
        "source": "experiments/f1_mgc_block0/PREDICTIONS.md (registered 2026-09-25, before any F1 run); scored at n = 3",
        "P1": {"statement": "(c) ~ MGC in deepest-block sep: (c) 3-seed mean in [4.72, 5.78] (5.25 +/- 10%)",
               "c_mean": sd["c"]["mean"], "c_sd": sd["c"]["sd"], "c_values": sd["c"]["values"], "band_registered": band,
               "band_from_exact_mgc_mean": [r8(x) for x in exact_band],
               "verdict": verdict(p1_ok), "verdict_exact_band": verdict(exact_band[0] <= sd["c"]["mean"] <= exact_band[1])},
        "P2": {"statement": "(b) < MGC in deepest-block sep: (b) 3-seed mean < 4.72 AND every (b) seed below the lowest MGC seed (5.21)",
               "b_mean": sd["b"]["mean"], "b_sd": sd["b"]["sd"], "b_values": sd["b"]["values"],
               "clause_mean_below_4.72": verdict(p2_mean_ok),
               "clause_every_seed_below_lowest_mgc_seed": verdict(p2_seed_ok), "lowest_mgc_seed_exact": r8(mgc_min_seed),
               "verdict": verdict(p2_mean_ok and p2_seed_ok),
               "unscored_expectation": "(b) lands near the cumulative regime (order of 1 or below)",
               "unscored_expectation_criterion": "(b) mean closer to the (a) mean than to the MGC mean",
               "unscored_expectation_met": bool(abs(sd["b"]["mean"] - sd["a"]["mean"]) < abs(sd["b"]["mean"] - sd["mgc"]["mean"])),
               "b_over_mgc_ratio": r8(sd["b"]["mean"] / sd["mgc"]["mean"]), "b_over_a_ratio": r8(sd["b"]["mean"] / sd["a"]["mean"])},
        "P3": {"statement": "for every arm run, the 3-seed mean S2-TTA lies in [86.27, 88.27] (87.27 +/- 1 pp)",
               "per_arm": {arm: {**v, "in_band": verdict(v["in_band"])} for arm, v in p3.items()},
               "verdict": verdict(all(v["in_band"] for v in p3.values()))},
        "P4": {"statement": "(secondary, arm a run) the (a) deepest-block sep mean is < 4.72",
               "a_mean": sd["a"]["mean"], "a_sd": sd["a"]["sd"], "a_values": sd["a"]["values"],
               "sam_off_cumulative_seed42_reference": 0.7421, "verdict": verdict(p4_ok)},
        "no_registered_direction_S1_S1TTA": {
            f"{x}_minus_{y}": {p: {"seed_mean": contrasts[f"{x}_minus_{y}"]["seed_paired_differences"][p]["mean"],
                                   "seed_sd": contrasts[f"{x}_minus_{y}"]["seed_paired_differences"][p]["sd"],
                                   "pooled_ci95_pp": contrasts[f"{x}_minus_{y}"]["example_level_paired_bootstrap"][p]["pooled"]["ci95_pp"]}
                               for p in ("S1", "S1-TTA")} for x, y in CONTRASTS},
        "registered_consequences_triggered": [],
    }
    if not p1_ok:
        predictions["registered_consequences_triggered"].append(
            "P1 fails: part of MGC's deep-block separation change is due to dropping block 0's auxiliary; report (c) as the clean MGC arm.")
    if not (p2_mean_ok and p2_seed_ok):
        predictions["registered_consequences_triggered"].append(
            "P2 fails ((b) >= 4.72): removing the current-block auxiliary alone reproduces most of MGC's separation. The MGC paragraph "
            "must then stop attributing the separation rise to missing-gradient compensation, and the 'approximately compensates' "
            "reading of App. A.1 is not supported at D128. This is the outcome that changes a claim, and it is reported as such. "
            "(PREDICTIONS.md, 'What each outcome means')")
    if not all(v["in_band"] for v in p3.values()):
        predictions["registered_consequences_triggered"].append("P3 fails for an arm: report that arm's per-seed values; the sub-1 pp wording must not cover it.")
    if not p4_ok:
        predictions["registered_consequences_triggered"].append("P4 fails: SAM drives part of the D128 separation contrast; use (a) for the MGC-vs-cumulative comparison.")

    # ---------------- plain-language summary (every number below is computed above)
    def ms(arm, p):
        a = arms[arm]["agg"][p] if p != "sep" else arms[arm]["agg"]["sep_curr_nl_deepest"]
        return f"{a['mean']:.2f} +/- {a['sd']:.2f}" if p != "sep" else f"{a['mean']:.3f} +/- {a['sd']:.3f}"

    def bs(name, p):
        b = contrasts[name]["example_level_paired_bootstrap"][p]["pooled"]
        return f"{b['delta_pp']:+.2f} pp [{b['ci95_pp'][0]:+.2f}, {b['ci95_pp'][1]:+.2f}]"

    dm, dsd = decomposition["mean"], decomposition["sd_of_seed_values"]
    spd = lambda n, p: contrasts[n]["seed_paired_differences"][p]  # noqa: E731
    churn_between = [c["example_level_paired_bootstrap"]["S2-TTA"]["churn_pct"]["mean"] for c in contrasts.values()]
    churn_within = [churn_floor[a]["S2-TTA"]["mean"] for a in ARMS]
    churn_rel = ("below" if max(churn_between) < min(churn_within) else
                 "above" if min(churn_between) > max(churn_within) else "overlapping")
    interpretation = [
        f"Deepest-block separation (block 3 sep_curr_nl, epoch 180; mean +/- sample SD, n = 3): MGC {ms('mgc', 'sep')}, "
        f"(c) {ms('c', 'sep')}, (b) {ms('b', 'sep')}, (a) {ms('a', 'sep')}.",
        f"P1 {predictions['P1']['verdict']}: keeping block 0's auxiliary under MGC moves the deepest-block separation by "
        f"{spd('c_minus_mgc', 'sep_deepest')['mean']:+.3f} +/- {spd('c_minus_mgc', 'sep_deepest')['sd']:.3f} (c - MGC; "
        f"{100 * spd('c_minus_mgc', 'sep_deepest')['mean'] / arms['mgc']['agg']['sep_curr_nl_deepest']['mean']:.1f}% of the MGC mean), "
        f"well inside the registered +/-10% band.",
        f"P2 {predictions['P2']['verdict']} (registered outcome that changes a claim): removing the current-block auxiliary without "
        f"compensation gives (b) {ms('b', 'sep')}, above the 4.72 threshold. Of the MGC - a gain of {dm['gain_mgc_minus_a']:.3f} "
        f"+/- {dsd['gain_mgc_minus_a']:.3f}, auxiliary removal (b - a) accounts for {dm['aux_removal_b_minus_a']:.3f} "
        f"+/- {dsd['aux_removal_b_minus_a']:.3f} ({100 * dm['aux_removal_share_of_mean_gain']:.1f}%) and the compensator (MGC - b) for "
        f"{dm['compensation_mgc_minus_b']:.3f} +/- {dsd['compensation_mgc_minus_b']:.3f} ({100 * dm['compensation_share_of_mean_gain']:.1f}%). "
        f"The compensator's increment is small but has the same sign in every seed (per seed "
        f"{', '.join(f'{v:+.3f}' for v in spd('b_minus_mgc', 'sep_deepest')['per_seed'].values())} for b - MGC); P2's second clause "
        f"(every (b) seed below the lowest MGC seed) holds, its first clause does not.",
        f"P4 {predictions['P4']['verdict']}: the SAM-matched cumulative anchor stays low ((a) {ms('a', 'sep')}; SAM-off seed-42 reference "
        f"0.7421), so SAM does not drive the D128 separation contrast.",
        f"P3 {predictions['P3']['verdict']}: S2-TTA (a) {ms('a', 'S2-TTA')}, (b) {ms('b', 'S2-TTA')}, (c) {ms('c', 'S2-TTA')}, "
        f"MGC {ms('mgc', 'S2-TTA')}; every arm lies within 1 pp of 87.27.",
        f"Accuracy contrasts (pooled example-level paired bootstrap, 95% CI, conditional on the checkpoints): S2-TTA c - MGC "
        f"{bs('c_minus_mgc', 'S2-TTA')}, b - MGC {bs('b_minus_mgc', 'S2-TTA')}, MGC - a {bs('mgc_minus_a', 'S2-TTA')}; S1 MGC - a "
        f"{bs('mgc_minus_a', 'S1')}, b - a {bs('b_minus_a', 'S1')}, c - a {bs('c_minus_a', 'S1')}. The CIs resample test examples "
        f"only; the seed-level SD of the paired differences (e.g. S2-TTA c - MGC {spd('c_minus_mgc', 'S2-TTA')['sd']:.2f}, "
        f"b - MGC {spd('b_minus_mgc', 'S2-TTA')['sd']:.2f} pp, n = 3) and the hardware mixing are not in them. Arm a ran entirely on "
        f"Colab A100, arm b entirely on the GB10, so b - a is fully confounded with hardware/software stack.",
        f"Prediction churn between two arms at the same seed is {min(churn_between):.2f}-{max(churn_between):.2f}% of test images at "
        f"S2-TTA (3-seed means per contrast), {churn_rel} the churn between two seeds of the same arm "
        f"({min(churn_within):.2f}-{max(churn_within):.2f}%).",
    ]

    hw_runs = {f"{arm}{s}": per_run[arm][s]["hardware"]["short"] for arm in ARMS for s in SEEDS}
    obj = {
        "_meta": {
            "description": "F1 MGC block-0 auxiliary control on CIFAR-10 L4/D128 (constant gamma = 0.7, SAM on, 180 + 10 epochs, "
                           "seeds 42/123/456): arms a (cumulative + depth-scaled residual-weighted current-block auxiliary), "
                           "b (auxiliary removed everywhere, no compensator), c (MGC with block 0's auxiliary kept), and the "
                           "existing MGC-D128 runs.",
            "generated_by": "analysis/f1_analysis.py", "generated": _dt.date.today().isoformat(),
            "sd_convention": "sample SD (ddof=1) over seeds",
            "protocols": C.PROTOCOL_LABELS,
            "accuracy_source": "events.jsonl::final_test_eval top-1 x 100 (cross-checked against preds_*.npz)",
            "sep_source": "layer_metrics.jsonl sep_curr_nl at the last Stage-1 epoch (180)",
            "bootstrap": {"n_boot": C.N_BOOT, "rng": "numpy default_rng(0), fresh per contrast x protocol",
                          "procedure": "analysis/aggregate.py paired_stats + boot_means + ci_block (as for results/mgc_paired_bootstrap.json)",
                          "pooling": "per-example correctness deltas pooled over seeds 42/123/456 (n = 30,000)",
                          "ci": "percentile 95%", "conditioning": "CIs are conditional on the trained checkpoints (test-set resampling only); seed variation is the sample SD of the paired seed differences"},
            "churn": "percent of the 10,000 CIFAR-10 test images whose predicted label differs between two runs of the same seed",
            "hardware_per_run": hw_runs,
            "hardware_note": "Runs are mixed across NVIDIA GB10 (DGX Spark; F1 runs torch 2.14.0+cu130, MGC-D128 seed 42 torch 2.9.1+cu130) "
                             "and Colab A100-SXM4-40GB (torch 2.11.0+cu128). GPU kernels are not bitwise deterministic across stacks, so "
                             "arm differences include hardware/software run-to-run noise; per-contrast hardware pairing is listed. "
                             "By arm: " + "; ".join(f"{arm}: " + ", ".join(f"s{s} {per_run[arm][s]['hardware']['short']}" for s in SEEDS)
                                                    for arm in ARMS)
                             + ". The F1 GB10 runs ran on the DGX Spark host named in their job logs ("
                             + ", ".join(sorted({h for arm in ARMS for s in SEEDS for h in ((per_run[arm][s]['job_log'] or {}).get('segment_host_or_device') or [])
                                                if 'A100' not in h})) + ").",
            "resume_note": ("No F1 run and no MGC-D128 run was resumed: one START segment and a from-scratch start in every job log, no "
                            "resume events, Stage-1 layer logs contiguous from epoch 1 to 180 (MGC-D128 seeds 123/456: no job log here; "
                            "their single run directories log epochs 1-180). The Colab poller lost its connection three times during "
                            "the arm-a seed 42/123 session (experiments/logs/colab_supervisor_ffcr2.log, EXEC-FAIL rc=124); training "
                            "continued in one segment.")
                           if not any(per_run[arm][s]["resumed"] for arm in ARMS for s in SEEDS) else
                           "RESUMED RUNS: " + ", ".join(f"{arm}{s}" for arm in ARMS for s in SEEDS if per_run[arm][s]["resumed"]),
            "trainer_sha256": TRAINER_SHA,
            "run_dirs": {arm: {str(s): rel(d) for s, d in RUNS[arm].items()} for arm in ARMS},
        },
        "summary": interpretation,
        "per_run": {arm: {str(s): per_run[arm][s] for s in SEEDS} for arm in ARMS},
        "arms": arms,
        "contrasts": contrasts,
        "within_arm_cross_seed_churn_pct": churn_floor,
        "deepest_sep_decomposition": decomposition,
        "predictions": predictions,
        "fd_gradient_attenuation": fd_block(),
        "gates": GATES,
        "all_gates_pass": all(g["pass"] for g in GATES),
    }
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    obj = neutral(obj)  # [CR] supplement port
    A.write(OUT_NAME, obj)

    # ---------------- console summary
    print("\nall gates pass:", obj["all_gates_pass"])
    for g in GATES:
        if not g["pass"]:
            print("  GATE FAIL:", g)
    for arm in ARMS:
        a = arms[arm]["agg"]
        print(f"{arm:4s} " + "  ".join(f"{p} {a[p]['mean']:.2f}+-{a[p]['sd']:.2f}" for p in PROT)
              + f"  sep3 {a['sep_curr_nl_deepest']['mean']:.3f}+-{a['sep_curr_nl_deepest']['sd']:.3f}  hw {arms[arm]['hardware_by_seed']}")
    for name, c in contrasts.items():
        sp = c["seed_paired_differences"]
        bs = c["example_level_paired_bootstrap"]
        print(f"{name:14s} sep3 {sp['sep_deepest']['mean']:+.3f}+-{sp['sep_deepest']['sd']:.3f} | "
              + " | ".join(f"{p} {bs[p]['pooled']['delta_pp']:+.2f} [{bs[p]['pooled']['ci95_pp'][0]:+.2f},{bs[p]['pooled']['ci95_pp'][1]:+.2f}] churn {bs[p]['churn_pct']['mean']:.2f}"
                           for p in PROT))
    print("decomposition:", decomposition["mean"])
    for k in ("P1", "P2", "P3", "P4"):
        print(k, predictions[k]["verdict"])
    fd = obj["fd_gradient_attenuation"]
    if fd.get("status") == "measured":
        for k, v in fd["variants"].items():
            print(k, "F_d", [round(x, 4) for x in v["F_d"]], "cov", [round(x, 1) for x in v["coverage_pct"]],
                  "grad/grad_local", v["grad_mean_over_grad_local_mean"])
        print("F_d:", fd.get("answer"))
    print()
    for line in interpretation:
        print("-", line)


if __name__ == "__main__":
    main()
