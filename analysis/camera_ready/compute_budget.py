#!/usr/bin/env python3
"""Compute budget of the reported runs, from the raw run logs (checklist item 8, H5).

Writes <FF_OUT_DIR>/results/compute_budget.json. Standard library only; reads the run directories
under FF_HOME (read-only). Paths come from environment variables (analysis/README.md).

Method
  * Device: env.json::device0 of the run directory. Run directories without env.json get a device
    only when it is documented elsewhere (see DEVICE_NOTES); otherwise "unknown".
  * Logged wall time: run_start -> run_end in events.jsonl of the result-bearing segment.
  * Full-run estimate: median Stage-1 epoch interval x configured Stage-1 epochs + median Stage-2
    epoch interval x configured Stage-2 epochs, with the medians taken over the consecutive
    epoch_metrics.jsonl timestamps of every listed segment of the run. The estimate covers resumed
    runs (whose earlier segments may be missing or partial) and excludes data download, setup and
    final evaluation, so it is slightly below the logged wall time of an uninterrupted run.
  * BP controls log per-epoch training time (history.jsonl::time_sec); their cost is its sum.
  * Runner-log durations (BP anchors, frozen-backbone readouts) pair the START and DONE/exit lines.
  * Every run directory is named explicitly (timestamps included); nothing is globbed.
Usage:  FF_HOME=... FF_OUT_DIR=... python analysis/camera_ready/compute_budget.py   (seconds, CPU only)
  The CPU-recomputation timings are read from the E3/equivalence result JSONs (FF_OUT_DIR/results,
  else this supplement's metric_summaries/camera_ready/) and from the E6 chunk outputs
  (FF_OUT_DIR/experiments/e6_l8/out); whatever is missing is recorded as null.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

HOME = C.HOME
CF = C.CF
# [CR] supplement port: optional inputs of the CPU-recomputation section
_SUPP_CR = Path(__file__).resolve().parents[2] / "metric_summaries" / "camera_ready"


def _opt_json(name):
    for d in (C.RESULTS, _SUPP_CR):
        p = d / name
        if p.exists():
            return json.load(open(p))
    return None
NOW = _dt.date.today().isoformat()


def rel(p: Path) -> str:
    return str(Path(p)).replace(str(HOME) + "/", "")


def jsonl(p: Path):
    out = []
    if not p.exists():
        return out
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def device_of(d: Path):
    e = d / "env.json"
    if e.exists():
        return json.load(open(e)).get("device0")
    return None


def med_interval(ts):
    ts = sorted(ts)
    dd = sorted(b - a for a, b in zip(ts, ts[1:]))
    return st.median(dd) if dd else None


def ff_run(segments, device_note=None, label=None):
    """segments: list of run dirs (first -> last). The last one carries the config and final eval."""
    segs = [Path(s) for s in segments]
    last = segs[-1]
    try:
        cfg = json.load(open(last / "config.json"))
    except (OSError, json.JSONDecodeError):
        cfg = {}              # e.g. an empty config.json; epochs then come from the epoch log
    e1 = cfg.get("epochs_stage1")
    e2 = cfg.get("epochs_stage2")
    epochs_source = "config.json" if isinstance(e1, int) else "max logged epoch"
    s1_ts, s2_ts, s1_int, s2_int = [], [], [], []
    first_s1 = None
    for s in segs:
        ep = jsonl(s / "epoch_metrics.jsonl")
        t1 = sorted(x["ts"] for x in ep if x.get("stage") == 1)
        t2 = sorted(x["ts"] for x in ep if x.get("stage") == 2)
        s1_int += [b - a for a, b in zip(t1, t1[1:])]
        s2_int += [b - a for a, b in zip(t2, t2[1:])]
        e1s = [x["epoch"] for x in ep if x.get("stage") == 1]
        if e1s and first_s1 is None:
            first_s1 = min(e1s)
        s1_ts += t1
        s2_ts += t2
    m1 = st.median(s1_int) if s1_int else None
    m2 = st.median(s2_int) if s2_int else None
    if not isinstance(e1, int):
        eps = [x for s in segs for x in jsonl(s / "epoch_metrics.jsonl")]
        e1 = max((x["epoch"] for x in eps if x.get("stage") == 1), default=None)
        e2 = max((x["epoch"] for x in eps if x.get("stage") == 2), default=None)
    est = None
    if m1 is not None and isinstance(e1, int):
        est = (m1 * e1 + (m2 or 0.0) * (e2 or 0)) / 3600.0
    ev = jsonl(last / "events.jsonl")
    rs = [x["ts"] for x in ev if x.get("event") == "run_start"]
    re_ = [x["ts"] for x in ev if x.get("event") == "run_end"]
    wall = (max(re_) - min(rs)) / 3600.0 if rs and re_ else None
    dev = None
    for s in reversed(segs):
        dev = device_of(s)
        if dev:
            break
    return {
        "label": label or last.name,
        "segments": [rel(s) for s in segs],
        "device": dev or device_note or "unknown",
        "device_source": "env.json" if dev else ("note" if device_note else None),
        "epochs_configured": [e1, e2],
        "epochs_source": epochs_source,
        "first_logged_stage1_epoch": first_s1,
        "median_stage1_epoch_s": round(m1, 1) if m1 else None,
        "median_stage2_epoch_s": round(m2, 1) if m2 else None,
        "est_full_run_h": round(est, 2) if est is not None else None,
        "logged_wall_h_last_segment": round(wall, 2) if wall is not None else None,
        "has_final_test_eval": any(x.get("event") == "final_test_eval" for x in ev),
    }


def bp_run(d, device_note):
    d = Path(d)
    h = jsonl(d / "history.jsonl")
    t = [x["time_sec"] for x in h if x.get("time_sec") is not None]
    fm = json.load(open(d / "final_metrics.json")) if (d / "final_metrics.json").exists() else {}
    return {
        "label": d.name,
        "segments": [rel(d)],
        "device": device_note,
        "device_source": "note",
        "epochs_logged": len(h),
        "sum_epoch_time_h": round(sum(t) / 3600.0, 2) if t else None,
        "final_metrics_device": fm.get("device"),
    }


def runner_pairs(log: Path, start_re: str, end_re: str):
    """Pair START and END lines of a runner log -> list of (name, start, end, hours)."""
    fmt = "%Y-%m-%d %H:%M:%S"
    starts, out = {}, []
    for line in open(log):
        m = re.search(start_re, line)
        if m:
            starts[m.group("key")] = _dt.datetime.strptime(m.group("ts"), fmt)
            continue
        m = re.search(end_re, line)
        if m and m.group("key") in starts:
            t0 = starts.pop(m.group("key"))
            t1 = _dt.datetime.strptime(m.group("ts"), fmt)
            out.append({"run": m.group("key"), "start": t0.strftime(fmt), "end": t1.strftime(fmt),
                        "hours": round((t1 - t0).total_seconds() / 3600.0, 2), "status": m.group("status")})
    # de-duplicate (the runner logs repeat their last lines after a relaunch)
    seen, uniq = set(), []
    for r in out:
        k = (r["run"], r["start"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


GB10_BY_RUNNER = "NVIDIA GB10 (DGX Spark; launched by a local runner script that waits on nvidia-smi; no env.json)"
A100_BY_APPENDIX = "NVIDIA A100-SXM4-40GB (Colab; stated in the training-cost appendix; final_metrics.json records only 'cuda')"

MR = C.MR
GA = C.GA
GA8 = CF / "basic_fair_revised_valfix_cp_fair" / "ff_gated_ablation_L8" / "runs"
CPF = CF / "basic_fair_revised_valfix_cp_fair"
COMP = CPF / "component_ablation_v2_from_drive" / "extracted" / "ff_component_ablation"
TINY = C.PR / "tiny_imagenet_fair"

# ============================================================================ camera-ready additions
# F1 (MGC block-0 control, CIFAR-10 L4/D128, SAM on, 180+10 epochs): nine single-segment runs, named
# explicitly (same directories as f1_analysis.py); arm a and c seed 456 ran on Colab A100, the
# rest on the DGX Spark GB10 (env.json of each run).
F1_RUNS_DIR = C.F1_DIR / "runs"  # [CR] supplement port: run archive (config.py)
F1_TS = {
    "f1a_cum_sam": {42: "20260926_224359", 123: "20260926_142507", 456: "20260926_142503"},
    "f1b_cum_noaux": {42: "20260925_101141", 123: "20260926_134435", 456: "20260927_161811"},
    "f1c_mgc_keepl0": {42: "20260926_001337", 123: "20260927_024824", 456: "20260926_224355"},
}
# F2 (CIFAR-10 gamma=0 seed-42 rerun in the seed-123/456 configuration, Colab A100): segment 1
# (..._20260925_080227) logged no epoch (VM lost during the CIFAR-10 download) and is not listed;
# segments 2-4 carry epochs 1-3 (+ a lost epoch 4), 4-274 and 275-362 + Stage 2 (results/f2_gamma0_s42.json).
F2_RUNS_DIR = C.F2_DIR / "runs"  # [CR] supplement port: run archive (config.py)
F2_SEGS = [F2_RUNS_DIR / f"ff_cifar10_cifar10_gamma0_seed42_matched_L4_D256_bs512_seed42_{ts}"
           for ts in ("20260925_130729", "20260926_105045", "20260927_212339")]
CR_GPU = {
    "mgc_c100": [ff_run([C.MGC_C100_RUNS[s]]) for s in C.SEEDS],
    "mgc_d128": [ff_run([C.MGC_D128_RUNS[s]]) for s in C.SEEDS],
    "hinge": [ff_run([C.HINGE_RUNS[a][s]]) for a in ("g07", "g0") for s in C.SEEDS],
    "f1_block0_control": [ff_run([F1_RUNS_DIR / f"ff_cifar10_{tag}_L4_D128_bs512_seed{s}_{ts[s]}"],
                                 label=f"{tag}_s{s}") for tag, ts in F1_TS.items() for s in C.SEEDS],
    "f2_gamma0_s42": [ff_run(F2_SEGS, label="f2_gamma0_s42_matched")],
}


def segment_recorded_wall_h(d: Path):
    """run_start -> run_end, or -> the last epoch row when the segment has no run_end (VM lost)."""
    ev = jsonl(d / "events.jsonl")
    rs = [x["ts"] for x in ev if x.get("event") == "run_start"]
    re_ = [x["ts"] for x in ev if x.get("event") == "run_end"]
    rows = [x["ts"] for x in jsonl(d / "epoch_metrics.jsonl")]
    end = max(re_) if re_ else (max(rows) if rows else None)
    return round((end - min(rs)) / 3600.0, 2) if rs and end is not None else None


F2_SEG_WALL = {d.name: segment_recorded_wall_h(d) for d in F2_SEGS}
assert all(v is not None for v in F2_SEG_WALL.values()), F2_SEG_WALL
CR_GPU["f2_gamma0_s42"][0]["segments_recorded_wall_h"] = F2_SEG_WALL
CR_GPU["f2_gamma0_s42"][0]["sum_segments_recorded_wall_h"] = round(sum(F2_SEG_WALL.values()), 2)
CR_GPU["f2_gamma0_s42"][0]["note"] = (
    "full-run estimate (median epoch time x configured epochs) excludes the repeated epoch 4 and the setup of "
    "each segment; segments_recorded_wall_h = run_start -> run_end (segment 4) or -> last epoch row (segments "
    "2-3, whose VMs were lost/stopped); a first segment (..._20260925_080227) logged no epoch and is excluded")
assert len(CR_GPU["f1_block0_control"]) == 9 and all(r["has_final_test_eval"] for r in CR_GPU["f1_block0_control"])
assert CR_GPU["f2_gamma0_s42"][0]["has_final_test_eval"]
bp_anchor_chain = runner_pairs(
    MR / "bp_runner.log",
    r"=== START (?P<key>\S+ seed \d+) \(attempt \d+\) (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)",
    r"=== (?P<status>DONE)\s+(?P<key>\S+ seed \d+) (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)",
)
# only DONE lines (seed verified in config.json) close a pair; a FAILED line carries no timestamp,
# so its START is superseded by the next START of the same run
bp_rerun_s42 = bp_run(C.BP_STRONG_S42_RERUN, GB10_BY_RUNNER)
s2ff = runner_pairs(
    MR / "s2ff_runner.log",
    r"=== S2-FF (?P<key>\w+) seed42 (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ===",
    r"=== (?P<key>\w+) exit=(?P<status>\d+) (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ===",
)
s2ffce = runner_pairs(
    MR / "s2ffce_runner.log",
    r"=== S2-FF-CE (?P<key>\w+) seed42 (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ===",
    r"=== (?P<key>\w+) exit=(?P<status>\d+) (?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ===",
)
aborted_gb10 = [ff_run([d]) for d in C.HINGE_ABORTED_DGX]
for r in aborted_gb10:
    ep = jsonl(HOME / r["segments"][0] / "epoch_metrics.jsonl")
    ts = [x["ts"] for x in ep]
    r["logged_epoch_span_h"] = round((max(ts) - min(ts)) / 3600.0, 2) if ts else None

# ============================================================================ submitted-paper runs
SUB = {
    # CIFAR-10 L4/D256, 32-expert MoE, SAM on
    "c10_d256_moe_a100": [
        ff_run([C.ABL / "no_all_collab/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260311_120026",
                C.C10_PAIR_RUNS["gamma0"][42]], label="gamma0_s42"),
        ff_run([C.ABL / "lcff_prefix/runs/ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260311_120438",
                C.ABL / "lcff_prefix/runs/ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260312_132019"],
               label="lcff_d256_s42"),
    ] + [
        ff_run([COMP / sub / n for n in names], label=f"component_{sub[5:]}") for sub, names in [
            ("runs_cpfair_baseline", ["ff_cifar10_ablation_cpfair_baseline_L4_D256_bs512_seed42_20260323_030540"]),
            ("runs_gamma0", ["ff_cifar10_ablation_gamma0_L4_D256_bs512_seed42_20260331_114340",
                             "ff_cifar10_ablation_gamma0_L4_D256_bs512_seed42_20260402_121810"]),
            ("runs_no_block_curr", ["ff_cifar10_ablation_no_block_curr_L4_D256_bs512_seed42_20260330_030138",
                                    "ff_cifar10_ablation_no_block_curr_L4_D256_bs512_seed42_20260330_223332"]),
            ("runs_no_depth_order", ["ff_cifar10_ablation_no_depth_order_L4_D256_bs512_seed42_20260327_055341",
                                     "ff_cifar10_ablation_no_depth_order_L4_D256_bs512_seed42_20260327_094917",
                                     "ff_cifar10_ablation_no_depth_order_L4_D256_bs512_seed42_20260329_220935"]),
            ("runs_no_hnm", ["ff_cifar10_ablation_no_hnm_L4_D256_bs512_seed42_20260325_164111",
                             "ff_cifar10_ablation_no_hnm_L4_D256_bs512_seed42_20260326_084811"]),
            ("runs_no_moe", ["ff_cifar10_ablation_no_moe_L4_D256_bs512_seed42_20260326_173825",
                             "ff_cifar10_ablation_no_moe_L4_D256_bs512_seed42_20260326_205045"]),
            ("runs_no_sam", ["ff_cifar10_ablation_no_sam_L4_D256_bs512_seed42_20260324_053035"]),
            ("runs_no_supcon", ["ff_cifar10_ablation_no_supcon_L4_D256_bs512_seed42_20260325_082544"]),
            ("runs_single_aspect", ["ff_cifar10_ablation_single_aspect_L4_D256_bs512_seed42_20260403_090931",
                                    "ff_cifar10_ablation_single_aspect_L4_D256_bs512_seed42_20260403_185707",
                                    "ff_cifar10_ablation_single_aspect_L4_D256_bs512_seed42_20260404_194358"]),
        ]
    ],
    "c10_d256_moe_gb10": [
        ff_run([CPF / "runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260218_144731",
                C.C10_PAIR_RUNS["cp_fair"][42]], label="cp_fair_s42"),
        ff_run([CPF / "runs_seed123/ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_bs512_seed123_20260318_001952",
                C.C10_PAIR_RUNS["cp_fair"][123]], label="cp_fair_s123"),
        ff_run([CPF / "runs_seed456/ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_bs512_seed456_20260324_135535",
                C.C10_PAIR_RUNS["cp_fair"][456]], label="cp_fair_s456"),
        ff_run([CPF / "runs_seed789/ff_cifar10_cifar10_cp_fair_clean_progressive_seed789_L4_D256_bs512_seed789_20260330_064018"],
               label="cp_fair_s789"),
        ff_run([C.C10_PAIR_RUNS["gamma0"][123]], label="gamma0_s123"),
        ff_run([C.C10_PAIR_RUNS["gamma0"][456]], label="gamma0_s456"),
        # 13-variant sweep rows (seed 42), identified by their final_test_eval values
        ff_run([CF / "basic_fair_revised_valfix_do_this_first/runs/ff_cifar10_cifar10_opt_do_this_first_ynorm_nomem_resid_L4_D256_bs512_seed42_20260128_084102"],
               label="sweep_nomem_l2"),
        ff_run([CF / "basic_fair_revised_valfix_cp_fair_v2/runs/ff_cifar10_cifar10_cp_fair_v2_low_contrastive_L4_D256_bs512_seed42_20260223_165508"],
               label="sweep_cp_fair_lc"),
        ff_run([CF / "atlas_ff/runs/ff_cifar10_cifar10_atlas_ff_triangle_multineg_div_probe_L4_D256_bs512_seed42_20260301_191325",
                CF / "atlas_ff/runs/ff_cifar10_cifar10_atlas_ff_triangle_multineg_div_probe_L4_D256_bs512_seed42_20260305_203646"],
               label="sweep_multi_tech"),
        ff_run([CF / "basic_fair_revised_valfix/runs/ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260106_173831",
                CF / "basic_fair_revised_valfix/runs/ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260108_214956",
                CF / "basic_fair_revised_valfix/runs/ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260110_171401"],
               label="sweep_baseline"),
        ff_run([CF / "basic_fair_revised_valfix_topk_depthsched/runs/ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_topkNL_depthsched_L4_D256_bs512_seed42_20260208_001416"],
               label="sweep_topk_depth"),
        ff_run([CF / "freez_layer_1_v2/runs/ff_cifar10_cifar10_opt_triangle_greedy_gaspectA_workersfix_v2_L4_D256_bs512_seed42_20260117_144151"],
               label="sweep_prefix_freeze"),
        ff_run([CF / "freez_layer_1/runs/ff_cifar10_cifar10_opt_triangle_greedy_gaspectA_workersfix_L4_D256_bs512_seed42_20260115_231345"],
               label="sweep_greedy_freeze"),
        ff_run([CF / "basic_fair_revised_valfix_dp_fair/runs/ff_cifar10_cifar10_dp_fair_progressive_depth_L4_D256_bs512_seed42_20260213_145420",
                CF / "basic_fair_revised_valfix_dp_fair/runs/ff_cifar10_cifar10_dp_fair_progressive_depth_L4_D256_bs512_seed42_20260218_131415"],
               label="sweep_dp_fair"),
    ],
    # CIFAR-10 L4/D128 gated ablation, seed 42, SAM off, nine arms
    "c10_l4_d128_gated_s42": [
        ff_run([GA / "runs" / n for n in names], label=names[-1].split("_L4_")[0][11:]) for names in [
            ["ff_cifar10_baseline_constant_L4_D128_bs512_seed42_20260313_232043"],
            ["ff_cifar10_no_all_collab_L4_D128_bs512_seed42_20260315_124117"],
            ["ff_cifar10_adaptive_t1_k0_L4_D128_bs512_seed42_20260314_032410"],
            ["ff_cifar10_lcff_prefix_L4_D128_bs512_seed42_20260315_084228"],
            ["ff_cifar10_adaptive_t1_k2_L4_D128_bs512_seed42_20260314_112422"],
            ["ff_cifar10_adaptive_t1_k4_L4_D128_bs512_seed42_20260314_152252"],
            ["ff_cifar10_adaptive_t3_k0_L4_D128_bs512_seed42_20260314_071928"],
            ["ff_cifar10_adaptive_prev_only_L4_D128_bs512_seed42_20260314_192927",
             "ff_cifar10_adaptive_prev_only_L4_D128_bs512_seed42_20260315_042603"],
            ["ff_cifar10_adaptive_prev_t1_k2_L4_D128_bs512_seed42_20260315_044213"],
        ]
    ],
    # CIFAR-10 L8/D128 gated ablation, seed 42, SAM off (arms without env.json: device unknown)
    "c10_l8_d128_gated_s42": [
        ff_run([GA8 / n], label=n.split("_L8_")[0][11:]) for n in [
            "ff_cifar10_baseline_constant_L8_L8_D128_bs512_seed42_20260318_204046",
            "ff_cifar10_no_all_collab_L8_L8_D128_bs512_seed42_20260318_062709",
            "ff_cifar10_lcff_prefix_L8_L8_D128_bs512_seed42_20260319_105650",
            "ff_cifar10_adaptive_prev_only_L8_L8_D128_bs512_seed42_20260317_031838",
            "ff_cifar10_adaptive_prev_t1_k2_L8_L8_D128_bs512_seed42_20260317_105615",
            "ff_cifar10_adaptive_t1_k0_L8_L8_D128_bs512_seed42_20260315_201449",
            "ff_cifar10_adaptive_t1_k2_L8_L8_D128_bs512_seed42_20260316_105222",
            "ff_cifar10_adaptive_t1_k4_L8_L8_D128_bs512_seed42_20260316_212730",
            "ff_cifar10_adaptive_t3_k0_L8_L8_D128_bs512_seed42_20260316_033950",
        ]
    ],
    "c10_gated_k0_multiseed": [
        ff_run([CF / "revised_training/extracted/ff_cifar10_gated_multiseed/runs_gated_k0_seed123/ff_cifar10_cifar10_gated_k0_seed123_L4_D128_bs512_seed123_20260426_071309"], label="k0_L4_s123"),
        ff_run([CF / "revised_training/extracted/ff_cifar10_gated_multiseed/runs_gated_k0_seed456/ff_cifar10_cifar10_gated_k0_seed456_L4_D128_bs512_seed456_20260426_150234"], label="k0_L4_s456"),
        ff_run([CF / "revised_training/extracted/ff_cifar10_l8_gated_multiseed/runs_gated_k0_L8_seed123/ff_cifar10_cifar10_gated_k0_L8_seed123_L8_D128_bs512_seed123_20260427_041411"], label="k0_L8_s123"),
        ff_run([CF / "revised_training/extracted/ff_cifar10_l8_gated_multiseed/runs_gated_k0_L8_seed456/ff_cifar10_cifar10_gated_k0_L8_seed456_L8_D128_bs512_seed456_20260427_114056"], label="k0_L8_s456"),
    ],
    "c10_l8_d128_32e_s42": [
        ff_run([CPF / "scaling_L8D128_32e/runs_scale_L8D128/ff_cifar10_scale_cpfair_G07_L8D128_L8_D128_bs512_seed42_20260315_071624",
                CPF / "scaling_L8D128_32e/runs_scale_L8D128/ff_cifar10_scale_cpfair_G07_L8D128_L8_D128_bs512_seed42_20260315_190153"],
               label="l8_32e_cp_fair"),
        ff_run([CPF / "scaling_L8D128_32e/runs_scale_L8D128/ff_cifar10_scale_lcff_G10_L8D128_L8_D128_bs512_seed42_20260316_140422"],
               label="l8_32e_lcff"),
        ff_run([CPF / "scaling_L8D128_32e/runs_scale_L8D128/ff_cifar10_scale_noall_G0_L8D128_L8_D128_bs512_seed42_20260314_015458",
                CPF / "scaling_L8D128_32e/runs_scale_L8D128/ff_cifar10_scale_noall_G0_L8D128_L8_D128_bs512_seed42_20260315_185311"],
               label="l8_32e_gamma0"),
    ],
    "c100_trio_dense": [ff_run([C.TRIO_RUNS[v][s]], label=f"{v}_s{s}") for v in C.TRIO_RUNS for s in C.SEEDS],
    "c100_32e_baseline": [
        ff_run([C.PR / "cifar_100_fair/runs/ff_cifar100_c100_baseline_constant_s123_L4_D256_bs256_seed123_20260405_094547"], label="c100_32e_s123"),
        ff_run([C.PR / "cifar_100_fair/runs/ff_cifar100_c100_baseline_constant_s456_L4_D256_bs256_seed456_20260411_095313"], label="c100_32e_s456"),
    ],
    "tiny_imagenet": [
        ff_run([TINY / "from_drive_1/extracted/ff_tiny_imagenet_training/runs_seed42/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed42_L4_D256_bs256_seed42_20260411_081714",
                TINY / "from_drive_1/extracted/ff_tiny_imagenet_training/runs_seed42/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed42_L4_D256_bs256_seed42_20260412_104050"],
               label="tiny_s42"),
        ff_run([TINY / "extracted_seeds/ff_tiny_imagenet_multiseed/runs_seed123/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed123_L4_D256_bs256_seed123_20260413_205048"], label="tiny_s123"),
        ff_run([TINY / "extracted_seeds/ff_tiny_imagenet_multiseed/runs_seed456/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed456_L4_D256_bs256_seed456_20260414_212958"], label="tiny_s456"),
    ],
    "bp_ff_controls_s42": [
        bp_run(C.BP_RUNS["bp_weak"][42], A100_BY_APPENDIX),
        bp_run(C.BP_RUNS["bp_strong"][42], A100_BY_APPENDIX),
        bp_run(CF / "ff_backbone_stripped/runs_ff_backbone_stripped/ff_backbone_stripped_cifar10_L4_D256_P2_seed42_20260306_153106", A100_BY_APPENDIX),
    ],
}


def h(r):
    for k in ("est_full_run_h", "sum_epoch_time_h", "hours"):
        if r.get(k) is not None:
            return r[k]
    return None


def dev_class(d: str | None) -> str:
    d = d or ""
    if "A100" in d:
        return "A100"
    if "GB10" in d:
        return "GB10"
    return "unknown"


def uninterrupted(r):
    """one segment that starts at Stage-1 epoch 1 and logs run_start and run_end"""
    return (len(r.get("segments", [])) == 1 and r.get("first_logged_stage1_epoch") == 1
            and r.get("logged_wall_h_last_segment") is not None)


def family_summary(rows):
    by = {}
    for r in rows:
        k = dev_class(r.get("device"))
        by.setdefault(k, []).append(h(r))
    out = {}
    for k, v in by.items():
        v = [x for x in v if x is not None]
        w = [r["logged_wall_h_last_segment"] for r in rows
             if dev_class(r.get("device")) == k and uninterrupted(r)]
        out[k] = {"n_runs": len(v), "per_run_h_min": round(min(v), 2) if v else None,
                  "per_run_h_max": round(max(v), 2) if v else None, "sum_h": round(sum(v), 1),
                  "uninterrupted_runs_logged_wall_h": [round(min(w), 2), round(max(w), 2), len(w)] if w else None}
    return out


def wall_or_est(r):
    """logged run_start->run_end for an uninterrupted run, else the full-run estimate."""
    if uninterrupted(r):
        return r["logged_wall_h_last_segment"]
    return h(r)


# L8 gated arms without env.json: attribute by median Stage-1 epoch time (A100 ~137-145 s; GB10 ~273 s at L8/D128)
for r in SUB["c10_l8_d128_gated_s42"]:
    if r["device"] == "unknown" and r.get("median_stage1_epoch_s"):
        m = r["median_stage1_epoch_s"]
        r["device"] = "A100 (inferred: median epoch time)" if m < 200 else "GB10 (inferred: median epoch time)"
        r["device_source"] = "inferred from median Stage-1 epoch time (A100 137-145 s, GB10 273 s for this config)"

sub_summary = {k: family_summary(v) for k, v in SUB.items()}
cr_summary = {k: family_summary(v) for k, v in CR_GPU.items()}

tot = {"A100": 0.0, "GB10": 0.0, "unknown": 0.0}
for fam in list(SUB.values()) + list(CR_GPU.values()):
    for r in fam:
        x = h(r)
        if x is not None:
            tot[dev_class(r.get("device"))] += x
bp_chain_h = sum(r["hours"] for r in bp_anchor_chain)
readouts_h = sum(r["hours"] for r in s2ff + s2ffce if r["status"] == "0")
aborted_h = sum(r["logged_epoch_span_h"] or 0 for r in aborted_gb10)
tot["GB10"] += bp_chain_h + (bp_rerun_s42["sum_epoch_time_h"] or 0) + readouts_h

cr_a100 = sum(h(r) for fam in CR_GPU.values() for r in fam if dev_class(r["device"]) == "A100")
cr_gb10 = sum(h(r) for fam in CR_GPU.values() for r in fam if dev_class(r["device"]) == "GB10") \
    + bp_chain_h + (bp_rerun_s42["sum_epoch_time_h"] or 0) + readouts_h

cr_a100_wall = sum(wall_or_est(r) for fam in CR_GPU.values() for r in fam if dev_class(r["device"]) == "A100")
cr_gb10_wall = sum(wall_or_est(r) for fam in CR_GPU.values() for r in fam if dev_class(r["device"]) == "GB10") \
    + bp_chain_h + (bp_rerun_s42["sum_epoch_time_h"] or 0) + readouts_h

# CPU recomputations (seconds recorded by the scripts themselves)
e3 = _opt_json("e3_fd_ema.json") or {"ema_accuracy_gate": []}
eq = _opt_json("supplement_equivalence.json") or {}
e6_chunks = sorted((C.CR / "experiments/e6_l8/out").glob("*_*_*.json"))
e6_s = [json.load(open(p)).get("seconds") for p in e6_chunks]
def _e1_e2_timings():
    """[CR] supplement port: the E1 tier-2 / E2 CPU timings. They need what analysis/experiments/run_cpu_batch.sh
    writes under FF_OUT_DIR/experiments (e2_exit/dumps, e1_sep/out, logs/batch.log and the per-step logs) and
    e2_exit.json (FF_OUT_DIR/results, else this supplement's metric_summaries/camera_ready/); without them every
    field is null, as for the other optional inputs."""
    keys = ("e2_exit_dump_s", "e2_exit_dump_c10_minutes_min_max", "e2_exit_dump_c100_hours_min_max",
            "e2_exit_select_s", "e2_exit_total_h", "e1_tier2_run_s", "e1_tier2_run_minutes_min_max",
            "e1_tier2_total_h", "e1_e2_cpu_batch")
    need = [C.CR / "experiments/e2_exit/dumps", C.CR / "experiments/e1_sep/out", C.CR / "experiments/logs/batch.log"]
    if not all(p.exists() for p in need) or _opt_json("e2_exit.json") is None:
        return dict.fromkeys(keys)
    # E2 (validation-selected early exit): per-checkpoint dump time recorded by dump_val_test_block_data.py
    # ("seconds_dump"; "seconds" of a --regate pass only re-scores an existing dump, so it is not used)
    e2_side = {p.stem: json.load(open(p)) for p in sorted((C.CR / "experiments/e2_exit/dumps").glob("c1*_seed*.json"))}
    e2_dump_s = {k: round(v["seconds_dump"], 1) for k, v in e2_side.items()}
    e2_c10 = [s for k, s in e2_dump_s.items() if k.startswith("c10_")]
    e2_c100 = [s for k, s in e2_dump_s.items() if k.startswith("c100_")]
    assert len(e2_c10) == 9 and len(e2_c100) == 9, e2_dump_s
    assert all(v.get("threads") == 6 for v in e2_side.values())
    e2_select_s = _opt_json("e2_exit.json")["seconds"]
    # E1 tier 2 (sep recomputed on the stage1_best EMA checkpoints): per-run wall time recorded by
    # measure_sep_tier2.py ("seconds" = test gate + 20-batch measurement); 18 points (ema_eval) plus the two
    # MGC-D128 seed-42 code-validation runs (ema_eval, raw_train)
    e1_side = {p.stem: json.load(open(p)) for p in sorted((C.CR / "experiments/e1_sep/out").glob("*__*.json"))}
    assert len(e1_side) == 20 and sum(k.startswith("c10_mgc_c1_seed42__") for k in e1_side) == 2, sorted(e1_side)
    assert all(v.get("status") == "OK" and v.get("threads") == 6 for v in e1_side.values())
    e1_run_s = {k: round(v["seconds"], 1) for k, v in e1_side.items()}
    # E1 tier 2 + E2 ran as ONE sequential CPU batch (experiments/run_cpu_batch.sh: CUDA hidden, 6 torch threads).
    # Whole-batch wall time from its step log (the "start" and "done" lines of experiments/logs/batch.log) and the
    # per-kind sums of the logged step durations. Steps whose output already existed (the smoke runs made before
    # the batch) are skipped by the runner; their per-step logs say "done already", and they are listed, not summed.
    BATCH_LOGS = C.CR / "experiments/logs"
    _bl = (BATCH_LOGS / "batch.log").read_text().splitlines()
    _b0 = [ln for ln in _bl if "===== run_cpu_batch.sh start" in ln]
    _b1 = [ln for ln in _bl if "===== run_cpu_batch.sh done" in ln]
    assert len(_b0) == 1 and len(_b1) == 1 and "done: 0 failed" in _b1[0], (_b0, _b1)
    _bsteps = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in
               (re.match(r"\[.{19}\] END   (\S+) rc=(-?\d+) \((\d+)s\)$", ln) for ln in _bl) if m]
    assert _bsteps and all(rc == 0 for _, rc, _ in _bsteps), _bsteps
    _bskip = [n for n, _, _ in _bsteps if "done already" in (BATCH_LOGS / f"{n}.log").read_text()]


    def _bkind(n: str) -> str:
        if n.startswith("e2_dump_"):
            return "e2_dumps"
        if n.startswith(("e2_select_", "e1_aggregate_")):
            return "aggregation"
        assert n.startswith("e1_"), n
        return "e1_points"


    _bts = [_dt.datetime.strptime(ln[1:20], "%Y-%m-%d %H:%M:%S") for ln in (_b0[0], _b1[0])]
    _bwall = (_bts[1] - _bts[0]).total_seconds()
    assert abs(sum(s for _, _, s in _bsteps) - _bwall) <= len(_bsteps), "steps did not run back to back"
    _bexec = {}
    for _k in ("e2_dumps", "e1_points", "aggregation"):
        _ss = [s for n, _, s in _bsteps if n not in _bskip and _bkind(n) == _k]
        _bexec[_k] = {"n_steps": len(_ss), "sum_s": sum(_ss), "sum_h": round(sum(_ss) / 3600, 2)}
    assert _bexec["e2_dumps"]["n_steps"] + sum(n.startswith("e2_dump_") for n in _bskip) == 18
    assert _bexec["e1_points"]["n_steps"] + sum(_bkind(n) == "e1_points" for n in _bskip) == 20
    e1e2_batch = {
        "log": "experiments/logs/batch.log (START/END line per step; per-step logs next to it)",
        "runner": "experiments/run_cpu_batch.sh",
        "device": "CPU only (CUDA_VISIBLE_DEVICES=''), 6 torch threads, steps run one after another",
        "start": _b0[0][1:20],
        "end": _b1[0][1:20],
        "wall_s": _bwall,
        "wall_h": round(_bwall / 3600, 1),
        "n_steps": len(_bsteps),
        "n_failed": 0,
        "executed": _bexec,
        "skipped_output_present": _bskip,
        "note": "The skipped steps were run before the batch as smoke tests; their own run times are in "
                "e2_exit_dump_s / e1_tier2_run_s, so e2_exit_total_h + e1_tier2_total_h exceeds the batch wall time "
                "by those runs. The E2 selector was re-run once afterwards (FLOP-model fix; e2_exit_select_s).",
    }
    return {
        "e2_exit_dump_s": e2_dump_s,
        "e2_exit_dump_c10_minutes_min_max": [round(min(e2_c10) / 60, 1), round(max(e2_c10) / 60, 1)],
        "e2_exit_dump_c100_hours_min_max": [round(min(e2_c100) / 3600, 2), round(max(e2_c100) / 3600, 2)],
        "e2_exit_select_s": e2_select_s,
        "e2_exit_total_h": round((sum(e2_dump_s.values()) + e2_select_s) / 3600, 1),
        "e1_tier2_run_s": e1_run_s,
        "e1_tier2_run_minutes_min_max": [round(min(e1_run_s.values()) / 60, 1), round(max(e1_run_s.values()) / 60, 1)],
        "e1_tier2_total_h": round(sum(e1_run_s.values()) / 3600, 1),
        "e1_e2_cpu_batch": e1e2_batch,
    }


def _f1_fd_timings():
    """[CR] supplement port: accuracy-gate seconds and thread count of the F1 F_d snapshot
    (analysis/experiments/f1_mgc_block0/fd_snapshot_f1.py), as recorded in each of its output JSONs under
    FF_OUT_DIR/experiments/f1_mgc_block0/out/fd (one gate per checkpoint; the *_seed variants re-use it).
    Without those outputs both fields are null, as for the other optional inputs."""
    d = C.CR / "experiments/f1_mgc_block0/out/fd"
    fd = {p.stem: json.load(open(p)) for p in sorted(d.glob("fd_*42_published.json"))} if d.exists() else {}
    if not fd:
        return {"f1_fd_ema_accuracy_gates_s": None, "f1_fd_threads": None}
    assert sorted(fd) == ["fd_a42_published", "fd_b42_published"], sorted(fd)
    return {"f1_fd_ema_accuracy_gates_s": {k: v["acc_gate"]["seconds"] for k, v in fd.items()},
            "f1_fd_threads": sorted({v["setup"]["threads"] for v in fd.values()})}


cpu = {
    "e3_ema_accuracy_gates_s": [round(g["seconds"], 1) for g in e3["ema_accuracy_gate"]],
    **_f1_fd_timings(),
    "e6_chunks": {p.name: round(s, 1) for p, s in zip(e6_chunks, e6_s) if s is not None},
    "e6_total_s": round(sum(s for s in e6_s if s is not None), 1),
    "supplement_equivalence_wall_s": eq["per_test_file"] if isinstance(eq.get("per_test_file"), dict) else None,
    **_e1_e2_timings(),
    "threads": 6,
    "note": "CPU jobs ran with CUDA hidden and at most 6 torch threads (8 for the F1 F_d snapshot, f1_fd_threads). "
            "E2 (experiments/e2_exit) is included: one "
            "score dump per checkpoint (val + test, all depths) plus the selector. E1 tier 2 (experiments/e1_sep) is "
            "included: one run per checkpoint (test gate + 20-batch sep measurement) for the 18 points, plus the two "
            "MGC-D128 seed-42 code-validation runs; the aggregation step (<1 min) is not counted. E1 tier 2 and E2 "
            "ran CPU-only as one sequential batch; e1_e2_cpu_batch gives its wall time from the batch log.",
}

out = {
    "_meta": {
        "description": "Compute budget of the runs reported in the paper, from raw run logs (checklist item 8).",
        "method": __doc__.strip().split("Method", 1)[1].split("Usage:")[0].strip(),
        "generated_by": "analysis/compute_budget.py",
        "generated": NOW,
        "submitted_checklist_values_superseded": {
            "D256 L4 run": "~4.5 h (submitted) vs ~30-32 A100-h (CIFAR-10 32-expert MoE) / ~15 A100-h (CIFAR-100 dense) measured",
            "Tiny ImageNet": "~8 h per seed (submitted; that is the last resumed segment) vs ~29 A100-h per seed measured",
            "hardware": "'A100 (Colab and an internal DGX)' (submitted) vs A100 on Colab plus a DGX Spark with a GB10 GPU (env.json)",
            "total": "~289 A100-h across 56 runs (submitted) vs the totals below",
        },
    },
    "camera_ready_gpu_runs": CR_GPU,
    "camera_ready_gpu_summary": cr_summary,
    "bp_anchor_chain_gb10": {"runs": bp_anchor_chain, "sum_h": round(bp_chain_h, 2),
                             "device": GB10_BY_RUNNER,
                             "accidental_seed42_rerun": bp_rerun_s42},
    "frozen_backbone_readouts_gb10": {"s2ff_trio": s2ff, "s2ffce_trio": s2ffce, "sum_h_exit0": round(readouts_h, 2),
                                      "device": GB10_BY_RUNNER,
                                      "not_measured": "S2-FF / S2-FF-CE on the three MGC backbones ran on Colab; their runner logs are not in the archive."},
    "aborted_or_smoke_gb10": {"aborted_hinge_seed42": aborted_gb10, "aborted_epoch_span_h": round(aborted_h, 2)},
    "camera_ready_totals_h": {"A100": round(cr_a100, 1), "GB10": round(cr_gb10, 1),
                              "method": "full-run estimates (median epoch time x configured epochs) + runner-log durations"},
    "camera_ready_totals_wall_h": {"A100": round(cr_a100_wall, 1), "GB10": round(cr_gb10_wall, 1),
                                   "method": "logged run_start->run_end for uninterrupted runs, full-run estimate for resumed runs, + runner-log durations"},
    "in_progress_not_included": [],
    "f1_f2_note": "F1 (experiments/f1_mgc_block0; 9 runs, GB10 and Colab A100) and F2 (experiments/f2_gamma0_s42; "
                  "Colab A100, 3 segments with 2 resumes) finished 2026-09-28 and are included in camera_ready_gpu_runs "
                  "and the totals.",
    "submitted_paper_runs": SUB,
    "submitted_paper_summary": sub_summary,
    "not_measured_reported_runs": [
        "13-variant sweep rows fixed-nl, block0-label and gamma-ramp (logs not identified in the local archive)",
        "CIFAR-100 32-expert gamma=0.7 baseline seed 42 (run not retained)",
        "plain-CNN BP/FF seed-42 rows of Table 2 (no per-epoch timing logged)",
        "HNM k-sweep and aspect/EMA deep-dive ablations",
        "text-domain runs (9; logs inside the text-run archive)",
    ],
    "totals_measured_h": {k: round(v, 1) for k, v in tot.items()},
    "cpu_recomputations": cpu,
}
C.RESULTS.mkdir(parents=True, exist_ok=True)
p = C.RESULTS / "compute_budget.json"
with open(p, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", rel(p))
print(json.dumps({"camera_ready_totals_h": out["camera_ready_totals_h"],
                  "camera_ready_totals_wall_h": out["camera_ready_totals_wall_h"],
                  "totals_measured_h": out["totals_measured_h"],
                  "bp_chain_h": round(bp_chain_h, 2), "readouts_h": round(readouts_h, 2),
                  "aborted_h": round(aborted_h, 2)}, indent=1))
for k, v in {**cr_summary, **sub_summary}.items():
    print(k, v)
