#!/usr/bin/env python3
"""F2 analysis: CIFAR-10 gamma=0 seed 42 re-run in the seed-123/456 configuration.

No model compute (numpy + scipy only; CPU). Reads the raw run files and writes
<FF_OUT_DIR>/results/f2_gamma0_s42.json with:

  * stitched per-epoch provenance of the F2 run across its four Colab segments
    (segment boundaries, resume events, RNG-restore status recorded by the [F2] fork,
    hardware/software per segment from env.json, epochs trained twice and which copy is used);
  * the final four test protocols (final_test_eval, cross-checked against the saved
    prediction arrays);
  * the deepest-block separation (layer 3 `sep_curr_nl`, layer_metrics.jsonl) at the last
    Stage-1 epoch (362, the registered P2 metric) and at the validation-selected
    Stage-1 epoch (the convention of the paper's tab:gamma0_multiseed note);
  * comparisons with the published seed 42 (old config: block_curr_lambda 0,
    depth_order_lambda 0) and with seeds 123/456 (matched config);
  * matched-config n = 3 aggregates (new seed 42, 123, 456) and the published n = 3,
    mean +/- sample SD (ddof = 1), four protocols + sep;
  * PASS/FAIL scoring of each registered prediction in
    experiments/f2_gamma0_s42/PREDICTIONS.md (P1, P1b, P2), exactly as registered (shipped as
    analysis/experiments/f2_gamma0_s42/PREDICTIONS.md).

Run (CPU only, seconds):
  FF_HOME=... FF_OUT_DIR=... CUDA_VISIBLE_DEVICES='' python analysis/camera_ready/f2_analysis.py

Paths ([CR] supplement port; analysis/README.md): FF_HOME is the root of the run archive. The F2 run
segments, the Colab job logs (logs/colab/), config_diffs/, test_equivalence_result.json and the fork
trainer.py are read from the F2 experiment folder of the archive ($FF_F2_DIR, default
$FF_HOME/amir-porjects/neuroips_review/camera_ready/experiments/f2_gamma0_s42). e1_tier1_sep_acc.json is
read from <FF_OUT_DIR>/results, else from metric_summaries/camera_ready/. The Colab Drive mount point and
the archive prefix are neutralized in the written JSON, as in the distributed copy. "CR/" and "CF/" in the
output name the camera-ready repository and cifar_10_fair/making_v1_v4_fair_comparison inside the archive.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

# [CR] supplement port: paths come from the environment (analysis/README.md); nothing is hard-coded.
HOME = Path(os.environ.get("FF_HOME", ""))               # root of the run archive (read-only)
PR = HOME / "amir-porjects"
CF = PR / "cifar_10_fair" / "making_v1_v4_fair_comparison"
CR = PR / "neuroips_review" / "camera_ready"             # the camera-ready repository inside the archive
EXP = Path(os.environ.get("FF_F2_DIR", str(CR / "experiments" / "f2_gamma0_s42")))
RUNS = EXP / "runs"
LOGS = EXP / "logs" / "colab"
OUT_ROOT = Path(os.environ.get("FF_OUT_DIR", "ff_out"))
OUT = OUT_ROOT / "results" / "f2_gamma0_s42.json"
_SUPP_CR = Path(__file__).resolve().parents[2] / "metric_summaries" / "camera_ready"

TAG = "ff_cifar10_cifar10_gamma0_seed42_matched_L4_D256_bs512_seed42"
# Every segment is named explicitly (no glob), oldest first.
F2_SEGMENTS = [
    dict(id="seg1", dir=RUNS / f"{TAG}_20260925_080227",
         launch_note=("Colab CLI VM (A100); VM reclaimed about 25 min after start, during the CIFAR-10 download, "
                      "before any Stage-1 epoch finished")),
    dict(id="seg2", dir=RUNS / f"{TAG}_20260925_130729",
         launch_note="Colab CLI VM (A100); fresh start; VM lost during epoch 5"),
    dict(id="seg3", dir=RUNS / f"{TAG}_20260926_105045",
         launch_note="browser Colab (A100); resumed from the Drive checkpoint"),
    dict(id="seg4", dir=RUNS / f"{TAG}_20260927_212339",
         launch_note="browser Colab (A100); resumed from the Drive checkpoint; finished Stage 1, Stage 2 and final_test_eval"),
]
LAUNCH_NOTE_SOURCE = ("launch mode (CLI VM vs browser) and the seg1 loss timing are not recorded in the run files; "
                      "they are taken from PROGRESS.md (2026-09-25 to 2026-09-28 entries) and the F2 task notes. "
                      "'VM lost during epoch 5' (seg2) is from the Drive console log, whose seg2 part ends at the "
                      "epoch-5 progress bar")

PUB42_SEGMENTS = [
    CF / "ablations" / "no_all_collab" / "runs" / "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260311_120026",
    CF / "ablations" / "no_all_collab" / "runs" / "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260312_132043",
]
S123 = [CF / "revised_training" / "out" / "gamma0_seed123" / "runs" / "ff_cifar10_cifar10_gamma0_seed123_L4_D256_bs512_seed123_20260417_235110"]
S456 = [CF / "revised_training" / "out" / "gamma0_seed456" / "runs" / "ff_cifar10_cifar10_gamma0_seed456_L4_D256_bs512_seed456_20260422_135437"]

PROTOCOLS = {"S1": "stage1_test_top1_no_tta", "S1-TTA": "stage1_test_top1_tta",
             "S2": "stage2_test_top1_no_tta", "S2-TTA": "stage2_test_top1_tta"}
TOP5 = {"S1": "stage1_test_top5_no_tta", "S1-TTA": "stage1_test_top5_tta",
        "S2": "stage2_test_top5_no_tta", "S2-TTA": "stage2_test_top5_tta"}
PRED_FILES = {"S1": "preds_stage1_test_no_tta.npz", "S1-TTA": "preds_stage1_test_tta.npz",
              "S2": "preds_stage2_test_no_tta.npz", "S2-TTA": "preds_stage2_test_tta.npz"}
N_TEST = 10000
DEEP = 3  # layer index of the deepest of the 4 blocks
SEP_KEY = "sep_curr_nl"

# Reference values registered in PREDICTIONS.md (2026-09-25). They are *checked* against the
# raw files below, never used in place of them.
REGISTERED_REF = {
    "pub42": {"S1": 90.52, "S1-TTA": 91.45, "S2": 90.37, "S2-TTA": 91.39, "sep_last": 3.2756},
    "s123": {"S1": 90.54, "S1-TTA": 91.40, "S2": 90.51, "S2-TTA": 91.51, "sep_last": 7.8521},
    "s456": {"S1": 90.44, "S1-TTA": 91.10, "S2": 90.44, "S2-TTA": 91.22, "sep_last": 7.9733},
    "published_n3_s1_tta": (91.32, 0.19),
}
# Selected-epoch seps printed in the paper's tab:gamma0_multiseed note (3.22/5.65/6.41 at epochs 320/260/260)
# come from results/e1_tier1_sep_acc.json; the raw-file values below are gated against that file.
E1_JSON = next((p for p in (OUT_ROOT / "results" / "e1_tier1_sep_acc.json", _SUPP_CR / "e1_tier1_sep_acc.json")
                if p.exists()), _SUPP_CR / "e1_tier1_sep_acc.json")  # [CR] supplement port
E1_KEYS = {"pub42": 42, "s123": 123, "s456": 456}
PAPER_SELECTED_2DP = {"pub42": (320, 3.22), "s123": (260, 5.65), "s456": (260, 6.41)}

# Registered predictions (PREDICTIONS.md, "Registered predictions"); bounds in % / sep units.
P1_TEXT = "S1-TTA lies within the seed-123/456 range, [91.10, 91.40]"
P1B_TEXT = "S1-TTA lies within +/-0.30 pp of the seed-123/456 mean (91.25), i.e. in [90.95, 91.55]"
P2_TEXT = ("the deepest-block sep (layer-3 sep_curr_nl at Stage-1 epoch 362) is near 7.9, like seeds 123/456, "
           "rather than 3.28. Band: [6.9, 8.9]. Decision rule: >= 5.6 counts as 'like seeds 123/456'; "
           "< 5.6 counts as 'like the published seed 42'")
P1B_HALF_WIDTH_PP = 0.30
P2_BAND = (6.9, 8.9)
P2_RULE = 5.6


# ----------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def utc(ts) -> str | None:
    if ts is None:
        return None
    return _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    p = Path(p)
    for base, name in ((CR, "CR"), (CF, "CF")):
        try:
            return f"{name}/{p.relative_to(base)}"
        except ValueError:
            pass
    return str(p)


def neutral(o):
    """[CR] supplement port: strip the archive prefix and the Colab Drive mount point from every string and
    replace DGX host names, recursively (the distributed JSON carries no absolute path or host name)."""
    if isinstance(o, dict):
        return {neutral(k): neutral(v) for k, v in o.items()}
    if isinstance(o, list):
        return [neutral(v) for v in o]
    if isinstance(o, str):
        if str(HOME) not in ("", "."):
            o = o.replace(str(HOME).rstrip("/") + "/", "")
        o = re.sub(r"/\S*?drive/MyDrive/", "MyDrive/", o)  # Colab Drive mount point -> Drive-relative
        return re.sub(r"\bspark-[0-9a-z]+\b", "<DGX Spark host>", o)
    return o


def sample_sd(v) -> float:
    return float(np.std(np.asarray(v, float), ddof=1))


def agg(values: list) -> dict:
    v = [float(x) for x in values]
    return {"values": v, "mean": float(np.mean(v)), "sd_sample": sample_sd(v), "n": len(v),
            "min": min(v), "max": max(v)}


GATES: list = []


def gate(check: str, ok: bool, **kw) -> bool:
    GATES.append({"check": check, "ok": bool(ok), **kw})
    return bool(ok)


# ----------------------------------------------------------------------------- run loading
def load_single_or_legacy(segs: list, label: str) -> dict:
    """Stitch contiguous segments of a run whose trainer has no resume events (legacy)
    or a single-segment run. Used for the published seed 42 and seeds 123/456."""
    em, lm, seg_of = {}, defaultdict(dict), {}
    ranges = []
    for si, s in enumerate(segs):
        e1 = [r for r in read_jsonl(s / "epoch_metrics.jsonl") if r.get("stage") == 1]
        l1 = [r for r in read_jsonl(s / "layer_metrics.jsonl") if r.get("stage") == 1]
        ranges.append((e1[0]["epoch"], e1[-1]["epoch"]))
        for r in e1:
            em[r["epoch"]] = r
            seg_of[r["epoch"]] = si
        for r in l1:
            lm[r["epoch"]][r["layer"]] = r
    assert ranges[0][0] == 1, (label, ranges)
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
        assert b0 == a1 + 1, (label, ranges)
    cfgs = [json.load(open(s / "config.json")) for s in segs]
    for c in cfgs[1:]:
        assert c == cfgs[0], label
    cfg = cfgs[-1]
    last = segs[-1]
    ev = read_jsonl(last / "events.jsonl")
    fte = [e for e in ev if e.get("event") == "final_test_eval"]
    rend = [e for e in ev if e.get("event") == "run_end"]
    assert len(fte) == 1 and len(rend) == 1, label
    return finish_run(label, segs, cfg, em, lm, seg_of, ranges, fte[0], rend[0], last)


def selected_epoch(em: dict, best_s1: float, label: str) -> int:
    evals = sorted((e, r["eval_top1"]) for e, r in em.items() if r.get("eval_top1") is not None)
    vmax = max(v for _, v in evals)
    assert abs(vmax - best_s1) < 1e-12, (label, vmax, best_s1)
    ea = min(e for e, v in evals if v == vmax)          # first epoch attaining the max (strict '>')
    eb = min(e for e, r in em.items() if r.get("best_s1") is not None and abs(r["best_s1"] - best_s1) < 1e-12)
    assert ea == eb, (label, ea, eb)
    return ea


def finish_run(label, segs, cfg, em, lm, seg_of, ranges, fte, rend, last) -> dict:
    L = int(cfg["num_blocks"])
    last_epoch = int(cfg["epochs_stage1"])
    missing = sorted(set(range(1, last_epoch + 1)) - set(em))
    missing_lm = sorted(e for e in range(1, last_epoch + 1) if sorted(lm.get(e, {})) != list(range(L)))
    assert not missing and not missing_lm, (label, missing, missing_lm)
    assert max(em) == last_epoch, label
    best_s1 = float(rend["best_s1"])
    e_sel = selected_epoch(em, best_s1, label)
    acc, top5, counts, npz = {}, {}, {}, {}
    y_pred = {}
    for k, key in PROTOCOLS.items():
        acc[k] = float(fte[key])
        top5[k] = float(fte[TOP5[k]])
        f = last / PRED_FILES[k]
        z = np.load(f)
        yt, yp = z["y_true"], z["y_pred"]
        assert len(yt) == N_TEST, (label, k, len(yt))
        c = int((yt == yp).sum())
        counts[k] = c
        npz[k] = {"file": rel(f), "sha256": sha256(f), "n": int(len(yt)), "correct": c}
        y_pred[k] = (yt.copy(), yp.copy())
        gate(f"{label} {k}: preds npz correct/10000 == final_test_eval", abs(c / N_TEST - acc[k]) < 5e-5,
             got=c / N_TEST, want=acc[k])
    traj = {int(e): lm[e][DEEP][SEP_KEY] for e in sorted(em) if em[e].get("eval_top1") is not None}
    return {
        "label": label,
        "segments": [rel(s) for s in segs],
        "segment_stage1_epoch_ranges": ranges,
        "config_json_sha256": sha256(segs[-1] / "config.json"),
        "env": json.load(open(segs[-1] / "env.json")),
        "env_per_segment": [json.load(open(s / "env.json")) for s in segs],
        "cfg": cfg,
        "best_s1_val": best_s1,
        "best_s2_val": float(rend["best_s2"]),
        "stage1_selected_epoch": e_sel,
        "stage1_selected_epoch_segment": rel(segs[seg_of[e_sel]]),
        "stage1_last_epoch": last_epoch,
        "sep_deep_last": float(lm[last_epoch][DEEP][SEP_KEY]),
        "sep_deep_selected": float(lm[e_sel][DEEP][SEP_KEY]),
        "sep_blocks_last": [float(lm[last_epoch][l][SEP_KEY]) for l in range(L)],
        "sep_blocks_selected": [float(lm[e_sel][l][SEP_KEY]) for l in range(L)],
        "sep_deep_at_eval_epochs": traj,
        "acc_frac": acc,
        "acc_pct": {k: round(100 * v, 2) for k, v in acc.items()},
        "test_correct_of_10000": counts,
        "top5_frac": top5,
        "preds_npz": npz,
        "final_test_eval_ts_utc": utc(fte.get("ts")),
        "events_sha256": sha256(last / "events.jsonl"),
        "layer_metrics_sha256": [sha256(s / "layer_metrics.jsonl") for s in segs],
        "_em": em, "_lm": lm, "_preds": y_pred,
    }


def split_trainer_log(path: Path) -> list:
    """Split the Drive trainer log into chunks at the runner's 'START segment' headers."""
    txt = path.read_text(errors="replace").replace("\r", "\n")
    hdr = re.compile(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) START segment .*?trainer sha256 ([0-9a-f]+)", re.M)
    ms = list(hdr.finditer(txt))
    chunks = []
    for i, m in enumerate(ms):
        body = txt[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(txt)]
        start = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc)
        non_env = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith("=== env ")]
        chunks.append({"header_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "header_ts": start.timestamp(),
                       "trainer_sha256_prefix": m.group(2), "body": body,
                       "n_trainer_output_lines": len(non_env)})
    return chunks


def load_f2() -> dict:
    segs = F2_SEGMENTS
    log_chunks = split_trainer_log(LOGS / "cifar10_gamma0_seed42_matched_seed42.log")
    runner_log = (LOGS / "runner.log").read_text()
    trainer_sha = sha256(EXP / "trainer.py")
    seg_info = []
    per_seg_rows = []
    for s in segs:
        d = s["dir"]
        ev = read_jsonl(d / "events.jsonl")
        e1 = [r for r in read_jsonl(d / "epoch_metrics.jsonl") if r.get("stage") == 1]
        e2 = [r for r in read_jsonl(d / "epoch_metrics.jsonl") if r.get("stage") == 2]
        l1 = [r for r in read_jsonl(d / "layer_metrics.jsonl") if r.get("stage") == 1]
        lraw = [r for r in read_jsonl(d / "layer_metrics_raw.jsonl") if r.get("stage") == 1]
        rs = [e for e in ev if e.get("event") == "run_start"]
        res = [e for e in ev if e.get("event") == "resume"]
        assert len(rs) == 1 and len(res) <= 1, s["id"]
        per_seg_rows.append((e1, e2, l1, lraw, ev))
        # console log chunk: header written by the runner a few seconds before the trainer's run_start
        ch = [c for c in log_chunks if 0 <= rs[0]["ts"] - c["header_ts"] <= 120]
        chunk = ch[0] if len(ch) == 1 else None
        console = {"present_in_drive_log": chunk is not None}
        if chunk is not None:
            body = chunk["body"]
            console.update({
                "header_utc": chunk["header_utc"],
                "trainer_sha256_prefix": chunk["trainer_sha256_prefix"],
                "trainer_sha256_prefix_matches_fork": trainer_sha.startswith(chunk["trainer_sha256_prefix"]),
                "n_trainer_output_lines": chunk["n_trainer_output_lines"],
                "rng_restore_line": next((ln.strip() for ln in body.splitlines() if "RNG state restored" in ln), None),
                "no_checkpoint_found_line": "No checkpoint found" in body,
                "checkpoint_saved_s1_epochs": sorted({int(x) for x in re.findall(r"Saved \(stage=1, s1_epoch=(\d+)", body)}),
                "end_line": next((ln.strip() for ln in body.splitlines() if ln.startswith("=== ") and "END segment" in ln), None),
            })
            if not chunk["n_trainer_output_lines"]:
                console["note"] = ("the Drive trainer log holds this segment's START header and env lines but no trainer "
                                   "output; the segment's evidence is its run directory (jsonl files)")
            sv = console["checkpoint_saved_s1_epochs"]
            if sv:
                console["checkpoint_saved_s1_epoch_range"] = [min(sv), max(sv)]
                console["checkpoint_saved_s1_epochs"] = f"{len(sv)} epochs ({min(sv)}-{max(sv)})" if len(sv) > 8 else sv
        else:
            console["note"] = "no START header of this segment in the Drive trainer log (the runner.log lists it)"
        runner_lines = []
        for ln in runner_log.splitlines():
            m = re.match(r"\[ffcr-runner (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", ln)
            if m:
                t = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc).timestamp()
                nxt = [x["ts"] for x in (read_jsonl(o["dir"] / "events.jsonl")[:1] for o in segs)
                       for x in x if x["ts"] > rs[0]["ts"] + 1]
                if rs[0]["ts"] - 120 <= t < (min(nxt) - 120 if nxt else float("inf")):
                    runner_lines.append(ln.strip())
        info = {
            "id": s["id"],
            "dir": rel(d),
            "launch_note": s["launch_note"],
            "run_start_utc": utc(rs[0]["ts"]),
            "env": json.load(open(d / "env.json")),
            "config_json_sha256": sha256(d / "config.json"),
            "split_indices_sha256": sha256(d / "split_indices.json") if (d / "split_indices.json").exists() else None,
            "stage1_epochs_logged": [e1[0]["epoch"], e1[-1]["epoch"]] if e1 else None,
            "n_stage1_epoch_rows": len(e1),
            "n_stage1_layer_rows": len(l1),
            "n_stage2_epoch_rows": len(e2),
            "resume_event": ({k: res[0][k] for k in res[0] if k != "ts"} | {"ts_utc": utc(res[0]["ts"])}) if res else None,
            "events": [e["event"] for e in ev],
            "first_row_utc": utc(e1[0]["ts"]) if e1 else None,
            "last_stage1_row_utc": utc(e1[-1]["ts"]) if e1 else None,
            "last_row_utc": utc((e2 or e1)[-1]["ts"]) if (e1 or e2) else None,
            "wall_h_run_start_to_last_row": (round(((e2 or e1)[-1]["ts"] - rs[0]["ts"]) / 3600, 2) if (e1 or e2) else None),
            "median_s_per_stage1_epoch": (float(np.median(np.diff([r["ts"] for r in e1]))) if len(e1) > 2 else None),
            "console_log": console,
            "runner_log_lines": runner_lines,
            "file_sha256": {fn: sha256(d / fn) for fn in ("events.jsonl", "epoch_metrics.jsonl", "layer_metrics.jsonl")},
        }
        fte = [e for e in ev if e.get("event") == "final_test_eval"]
        rend = [e for e in ev if e.get("event") == "run_end"]
        if rend:
            info["wall_h_run_start_to_run_end"] = round((rend[0]["ts"] - rs[0]["ts"]) / 3600, 2)
        info["has_final_test_eval"] = bool(fte)
        # RNG-restore status as recorded by the [F2] fork. events.jsonl 'rng_restored' covers python,
        # numpy and torch CPU; the torch-CUDA restore is only visible in the console line.
        if res:
            cl = console.get("rng_restore_line")
            info["rng_restore_status"] = {
                "status": "restored" if res[0].get("rng_restored") is True else f"not restored ({res[0].get('rng_restored')})",
                "from_checkpoint_stage1_epoch": res[0]["epoch_stage1"],
                "events_jsonl_rng_restored": res[0].get("rng_restored"),
                "console_line": cl,
                "torch_cuda_restore_confirmed": bool(cl and "torch CUDA" in cl),
                "dataloader_workers": "persistent workers re-created and re-seeded from the restored main-process torch RNG",
            }
        elif e1 and console.get("no_checkpoint_found_line"):
            info["rng_restore_status"] = {"status": "not applicable: fresh start ('No checkpoint found'); RNGs seeded from seed 42"}
        else:
            info["rng_restore_status"] = {"status": "not applicable: no checkpoint loaded and no epoch finished"}
        seg_info.append(info)

    # ---------------- lineage gates
    cfg_sha = {i["config_json_sha256"] for i in seg_info}
    gate("F2: config.json identical in all 4 segments", len(cfg_sha) == 1, got=sorted(cfg_sha))
    spl = {i["split_indices_sha256"] for i in seg_info if i["split_indices_sha256"]}
    gate("F2: split_indices.json identical in segments 2-4 (segment 1 never saved one)", len(spl) == 1, got=sorted(spl))
    env_keys = {json.dumps(i["env"], sort_keys=True) for i in seg_info}
    gate("F2: env.json identical in all 4 segments (A100-SXM4-40GB, same software)", len(env_keys) == 1)
    # segment 1: no epochs, no split; segment 2: fresh start (no resume)
    gate("F2 seg1: no Stage-1 rows (lost before epoch 1 finished)", seg_info[0]["n_stage1_epoch_rows"] == 0)
    gate("F2 seg2: fresh start (no resume event; 'No checkpoint found' in console log)",
         seg_info[1]["resume_event"] is None and seg_info[1]["console_log"].get("no_checkpoint_found_line") is True)

    # ---------------- stitch: a later segment's resume event (checkpoint epoch k) supersedes every
    # earlier row with epoch > k (those rows belong to an abandoned trajectory).
    canon_em, canon_lm, seg_of = {}, defaultdict(dict), {}
    raw_deep = {}
    superseded = []
    for si, (e1, e2, l1, lraw, ev) in enumerate(per_seg_rows):
        if not e1:
            continue
        res = seg_info[si]["resume_event"]
        k = res["epoch_stage1"] if res else 0
        gate(f"F2 {seg_info[si]['id']}: first logged Stage-1 epoch == checkpoint epoch + 1",
             e1[0]["epoch"] == k + 1, got=e1[0]["epoch"], want=k + 1)
        if res:
            gate(f"F2 {seg_info[si]['id']}: resume event rng_restored == true", res.get("rng_restored") is True,
                 got=res.get("rng_restored"))
            gate(f"F2 {seg_info[si]['id']}: resumed from a Stage-1 checkpoint", res.get("ckpt_stage") == 1)
            prior = [r["eval_top1"] for e, r in canon_em.items() if e <= k and r.get("eval_top1") is not None]
            want = max(prior) if prior else 0.0
            gate(f"F2 {seg_info[si]['id']}: resumed best_s1 == max val eval of epochs <= {k}",
                 abs(res["best_s1"] - want) < 1e-12, got=res["best_s1"], want=want)
            gate(f"F2 {seg_info[si]['id']}: first row best_s1 == resumed best_s1",
                 abs(e1[0]["best_s1"] - res["best_s1"]) < 1e-12)
        drop = sorted(e for e in canon_em if e > k)
        for e in drop:
            superseded.append({"epoch": e, "logged_in": seg_info[seg_of[e]]["id"],
                               "superseded_by": seg_info[si]["id"],
                               "reason": f"{seg_info[si]['id']} resumed from the epoch-{k} checkpoint"})
            del canon_em[e]
            del canon_lm[e]
            old_lm = [r for r in per_seg_rows[seg_of[e]][2] if r["epoch"] == e]
            superseded[-1]["_old_lm"] = {r["layer"]: r for r in old_lm}
            superseded[-1]["_old_em"] = next(r for r in per_seg_rows[seg_of[e]][0] if r["epoch"] == e)
            del seg_of[e]
        for r in e1:
            canon_em[r["epoch"]] = r
            seg_of[r["epoch"]] = si
        for r in l1:
            canon_lm[r["epoch"]][r["layer"]] = r
        for r in lraw:
            if r["layer"] == DEEP:
                raw_deep[r["epoch"]] = r.get(SEP_KEY)

    cfg = json.load(open(segs[-1]["dir"] / "config.json"))
    last = segs[-1]["dir"]
    ev = read_jsonl(last / "events.jsonl")
    fte = [e for e in ev if e.get("event") == "final_test_eval"]
    rend = [e for e in ev if e.get("event") == "run_end"]
    assert len(fte) == 1 and len(rend) == 1
    used = defaultdict(list)
    for e in sorted(canon_em):
        used[seg_of[e]].append(e)
    ranges = []
    for si in sorted(used):
        es = used[si]
        assert es == list(range(es[0], es[-1] + 1))
        ranges.append((es[0], es[-1]))
        seg_info[si]["stage1_epochs_used"] = [es[0], es[-1]]
    for i in seg_info:
        i.setdefault("stage1_epochs_used", None)
    run = finish_run("F2 new seed 42", [segs[si]["dir"] for si in sorted(used)], cfg, canon_em, canon_lm,
                     {e: sorted(used).index(si) for e, si in seg_of.items()}, ranges, fte[0], rend[0], last)
    run["segments_all"] = seg_info
    run["superseded_rows"] = superseded
    gate("F2: fixed-key and raw layer logs agree on the deepest-block sep at epoch 362",
         abs(raw_deep[362] - run["sep_deep_last"]) < 1e-12)
    # Stage 2 rows
    e2 = per_seg_rows[-1][1]
    run["stage2"] = {"epochs_logged": [e2[0]["epoch"], e2[-1]["epoch"]], "segment": seg_info[-1]["id"],
                     "val_evals": {int(r["epoch"]): r["eval_top1"] for r in e2 if r.get("eval_top1") is not None},
                     "backbone": "frozen stage1_best EMA backbone (console log: 'Loaded backbone weights from "
                                 "..._stage1_best.pt' before Stage 2 and before final_test_eval)"}
    body4 = next(c["body"] for c in log_chunks if abs(c["header_ts"] - read_jsonl(last / "events.jsonl")[0]["ts"]) <= 120)
    n_bb = body4.count("Loaded backbone weights from") if body4 else 0
    gate("F2 seg4 console: stage1_best backbone loaded for Stage 2 and for the final test eval", n_bb == 2, got=n_bb)
    m = re.findall(r"\[Final Test\]\[(Stage-1 FF|Stage-2 Head)\]\s+(no-TTA|TTA): Top1=([\d.]+)%", body4)
    con = {("S1" if a == "Stage-1 FF" else "S2") + ("" if b == "no-TTA" else "-TTA"): float(c) for a, b, c in m}
    gate("F2 seg4 console: printed final test top-1 == final_test_eval (x100)",
         con == {k: round(100 * v, 2) for k, v in run["acc_frac"].items()}, got=con)
    return run


# ----------------------------------------------------------------------------- main
def main():
    f2 = load_f2()
    pub = load_single_or_legacy(PUB42_SEGMENTS, "published seed 42")
    s123 = load_single_or_legacy(S123, "seed 123")
    s456 = load_single_or_legacy(S456, "seed 456")
    refs = {"pub42": pub, "s123": s123, "s456": s456}

    # ---------------- reference gates (raw files vs registered / printed values)
    e1_pts = json.load(open(E1_JSON))["points"]
    e1_point = {}
    for key, seed in E1_KEYS.items():
        m = [p for p in e1_pts if p["config"] == "c10_gamma0" and p["family"] == "c10_d256" and p["seed"] == seed]
        assert len(m) == 1, (key, len(m))
        e1_point[key] = m[0]
    for key, r in refs.items():
        for k in PROTOCOLS:
            gate(f"{r['label']} {k} == PREDICTIONS.md reference", r["acc_pct"][k] == REGISTERED_REF[key][k],
                 got=r["acc_pct"][k], want=REGISTERED_REF[key][k])
        gate(f"{r['label']} last-epoch deepest sep == PREDICTIONS.md reference (4 dp)",
             round(r["sep_deep_last"], 4) == REGISTERED_REF[key]["sep_last"],
             got=r["sep_deep_last"], want=REGISTERED_REF[key]["sep_last"])
        pt = e1_point[key]
        gate(f"{r['label']} selected epoch, selected/last-epoch sep and sep trajectory == results/e1_tier1_sep_acc.json (exact)",
             r["stage1_selected_epoch"] == pt["stage1_best_epoch"]
             and r["sep_deep_selected"] == pt["sep_deep_at_best_epoch"]
             and r["sep_deep_last"] == pt["sep_deep_at_last_epoch"]
             and {str(e): v for e, v in r["sep_deep_at_eval_epochs"].items()} == pt["sep_deep_at_eval_epochs"]
             and r["acc_pct"] == pt["acc_test_pct"],
             got=[r["stage1_selected_epoch"], r["sep_deep_selected"]],
             want=[pt["stage1_best_epoch"], pt["sep_deep_at_best_epoch"]])
        e, v = PAPER_SELECTED_2DP[key]
        gate(f"{r['label']} selected epoch / sep (2 dp) == tab:gamma0_multiseed note",
             r["stage1_selected_epoch"] == e and round(r["sep_deep_selected"], 2) == v,
             got=[r["stage1_selected_epoch"], round(r["sep_deep_selected"], 2)], want=[e, v])

    # ---------------- config relations
    def cdiff(a, b):
        return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    d123, d456, dpub = cdiff(f2["cfg"], s123["cfg"]), cdiff(f2["cfg"], s456["cfg"]), cdiff(f2["cfg"], pub["cfg"])
    gate("F2 config differs from seed 123 only in seed", set(d123) == {"seed"}, got=d123)
    gate("F2 config differs from seed 456 only in seed", set(d456) == {"seed"}, got=d456)
    gate("F2 config differs from the published seed 42 only in block_curr_lambda / depth_order_lambda",
         set(dpub) == {"block_curr_lambda", "depth_order_lambda"}, got=dpub)
    planned = json.load(open(EXP / "config_diffs" / "planned_seed42_matched.json"))
    gate("F2 config.json == planned config (config_diffs/planned_seed42_matched.json)", cdiff(f2["cfg"], planned) == {},
         got=cdiff(f2["cfg"], planned))
    split_same = sha256(F2_SEGMENTS[-1]["dir"] / "split_indices.json") == sha256(PUB42_SEGMENTS[-1] / "split_indices.json")
    gate("F2 train/val split == the published seed-42 split (split_indices.json sha256)", split_same)
    # lr / tau schedule per epoch identical to seed 123 (same config): scheduler state survived both resumes
    sched_bad = [e for e in range(1, 363)
                 if abs(f2["_em"][e]["lr"] - s123["_em"][e]["lr"]) > 1e-15 or abs(f2["_em"][e]["tau"] - s123["_em"][e]["tau"]) > 1e-15]
    gate("F2 per-epoch lr and router tau == seed 123's at every Stage-1 epoch (scheduler restored across resumes)",
         not sched_bad, got=sched_bad[:10])
    fork_sha = sha256(EXP / "trainer.py")
    tst = json.load(open(EXP / "test_equivalence_result.json"))
    gate("fork equivalence tests: 18/18 passed for this trainer sha256",
         tst["n_tests"] == 18 and tst["n_failed"] == 0 and tst["fork_sha256"] == fork_sha,
         got=[tst["n_tests"], tst["n_failed"]])

    # ---------------- epoch trained twice (seg2 epoch 4 vs seg3 epoch 4): resume reproducibility
    dup = []
    for s in f2["superseded_rows"]:
        e = s["epoch"]
        dup.append({
            "epoch": e, "logged_in": s["logged_in"], "superseded_by": s["superseded_by"], "reason": s["reason"],
            "sep_curr_nl_per_block_superseded": [s["_old_lm"][l][SEP_KEY] for l in range(4)],
            "sep_curr_nl_per_block_used": [f2["_lm"][e][l][SEP_KEY] for l in range(4)],
            "loss_total_per_block_superseded": [s["_old_lm"][l]["loss_total"] for l in range(4)],
            "loss_total_per_block_used": [f2["_lm"][e][l]["loss_total"] for l in range(4)],
            "max_abs_diff_sep_curr_nl": max(abs(s["_old_lm"][l][SEP_KEY] - f2["_lm"][e][l][SEP_KEY]) for l in range(4)),
            "max_abs_diff_loss_total": max(abs(s["_old_lm"][l]["loss_total"] - f2["_lm"][e][l]["loss_total"]) for l in range(4)),
            "bitwise_equal": all(s["_old_lm"][l] == {**f2["_lm"][e][l], "ts": s["_old_lm"][l]["ts"]} for l in range(4)),
            "note": ("both copies start from the same epoch-3 state, but the used copy started from the saved epoch-3 "
                     "checkpoint with the main-process RNG restored, on a different VM, with re-created (re-seeded) "
                     "persistent DataLoader workers; the fork documents that a resumed run is not bitwise equal to an "
                     "uninterrupted one"),
        })
    ckpt_note = None
    s2c = f2["segments_all"][1]["console_log"]
    if s2c.get("checkpoint_saved_s1_epoch_range") and f2["segments_all"][2]["resume_event"]:
        ckpt_note = (f"segment 2's console log reports Stage-1 checkpoint saves for epochs "
                     f"{s2c['checkpoint_saved_s1_epoch_range'][0]}-{s2c['checkpoint_saved_s1_epoch_range'][1]}, but segment 3 "
                     f"loaded the epoch-{f2['segments_all'][2]['resume_event']['epoch_stage1']} checkpoint, so the last "
                     f"logged save did not persist on Drive (most likely not yet synced when the CLI VM was lost; "
                     f"inference, not recorded in the files); epoch 4 was therefore trained twice")

    # ---------------- per-epoch provenance table
    seg_ids = [s["id"] for s in F2_SEGMENTS]
    used_seg = {}
    for i in f2["segments_all"]:
        if i["stage1_epochs_used"]:
            for e in range(i["stage1_epochs_used"][0], i["stage1_epochs_used"][1] + 1):
                used_seg[e] = i["id"]
    cols = ["epoch", "segment", "ts_utc", "lr", "tau", "val_eval_top1", "best_s1"] + [f"sep_curr_nl_b{l}" for l in range(4)]
    rows = []
    for e in range(1, 363):
        r = f2["_em"][e]
        rows.append([e, used_seg[e], utc(r["ts"]), r["lr"], r["tau"], r.get("eval_top1"), r["best_s1"]]
                    + [f2["_lm"][e][l][SEP_KEY] for l in range(4)])
    boundaries = []
    for a, b in zip(rows, rows[1:]):
        if a[1] != b[1]:
            boundaries.append({"after_epoch": a[0], "from": a[1], "to": b[1],
                               "gap_h_between_rows": round((_dt.datetime.fromisoformat(b[2].replace("Z", "+00:00"))
                                                            - _dt.datetime.fromisoformat(a[2].replace("Z", "+00:00"))).total_seconds() / 3600, 2)})

    # ---------------- comparisons
    new = f2
    comp_pub = {"protocols_pp": {}, "paired_test_predictions": {}}
    for k in PROTOCOLS:
        comp_pub["protocols_pp"][k] = {"new": new["acc_pct"][k], "published": pub["acc_pct"][k],
                                       "delta_new_minus_published": round(new["acc_pct"][k] - pub["acc_pct"][k], 2)}
        yt_n, yp_n = new["_preds"][k]
        yt_p, yp_p = pub["_preds"][k]
        assert np.array_equal(yt_n, yt_p)
        cn, cp = yp_n == yt_n, yp_p == yt_p
        b = int((cn & ~cp).sum())
        c = int((~cn & cp).sum())
        comp_pub["paired_test_predictions"][k] = {
            "new_right_published_wrong": b, "new_wrong_published_right": c,
            "prediction_agreement": float((yp_n == yp_p).mean()),
            "exact_mcnemar_p_two_sided": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
        }
    comp_pub["sep_last_epoch_362"] = {"new": new["sep_deep_last"], "published": pub["sep_deep_last"],
                                     "delta": new["sep_deep_last"] - pub["sep_deep_last"],
                                     "ratio": new["sep_deep_last"] / pub["sep_deep_last"]}
    comp_pub["sep_selected_epoch"] = {"new": new["sep_deep_selected"], "new_epoch": new["stage1_selected_epoch"],
                                     "published": pub["sep_deep_selected"], "published_epoch": pub["stage1_selected_epoch"]}
    comp_pub["best_val"] = {"new_best_s1": new["best_s1_val"], "published_best_s1": pub["best_s1_val"],
                            "new_best_s2": new["best_s2_val"], "published_best_s2": pub["best_s2_val"]}
    comp_pub["what_differs"] = {"config": dpub, "same_seed_and_split": split_same,
                                "hardware": "both Colab A100-SXM4-40GB (env.json of every segment)",
                                "software": {w: {k: r["env"][k] for k in ("torch", "cuda_version", "cudnn_version", "python", "platform")}
                                             for w, r in (("new", new), ("published", pub))},
                                "trainer": ("new: the [F2] fork of basic_fair_revised_valfix_cp_fair_fixed.py (the seed-123/456 "
                                            "trainer; RNG restored on resume). published: a pre-fork trainer that re-seeds instead "
                                            "of restoring the RNG on resume (PREDICTIONS.md provenance note); no trainer copy is "
                                            "stored with the published run, so code identity beyond identical config.json keys "
                                            "(values differ only in block_curr_lambda / depth_order_lambda) is not verified here"),
                                "resumes": "new: 2 resumes (after epochs 3 and 274); published: 1 resume (after epoch 273)"}

    comp_m = {"protocols_pp": {}}
    for k in PROTOCOLS:
        v123, v456 = s123["acc_pct"][k], s456["acc_pct"][k]
        lo, hi = min(v123, v456), max(v123, v456)
        comp_m["protocols_pp"][k] = {
            "new": new["acc_pct"][k], "s123": v123, "s456": v456, "s123_456_mean": round((v123 + v456) / 2, 3),
            "delta_new_minus_s123_456_mean": round(new["acc_pct"][k] - (v123 + v456) / 2, 3),
            "position": ("inside [min, max] of seeds 123/456" if lo <= new["acc_pct"][k] <= hi else
                         ("above" if new["acc_pct"][k] > hi else "below") + " seeds 123/456"),
        }
    comp_m["sep_last_epoch_362"] = {"new": new["sep_deep_last"], "s123": s123["sep_deep_last"], "s456": s456["sep_deep_last"]}
    comp_m["sep_selected_epoch"] = {"new": [new["stage1_selected_epoch"], new["sep_deep_selected"]],
                                    "s123": [s123["stage1_selected_epoch"], s123["sep_deep_selected"]],
                                    "s456": [s456["stage1_selected_epoch"], s456["sep_deep_selected"]]}
    comp_m["best_val"] = {"new": [new["best_s1_val"], new["best_s2_val"]], "s123": [s123["best_s1_val"], s123["best_s2_val"]],
                          "s456": [s456["best_s1_val"], s456["best_s2_val"]], "fields": ["best_s1", "best_s2"]}
    comp_m["hardware"] = {"new": f"{new['env']['device0']} (Colab), torch {new['env']['torch']}",
                          "s123": f"{s123['env']['device0']} (DGX Spark), torch {s123['env']['torch']}",
                          "s456": f"{s456['env']['device0']} (DGX Spark), torch {s456['env']['torch']}"}
    # sep trajectories at the common val-eval epochs (descriptive; explains selected-epoch differences)
    common = sorted(set(new["sep_deep_at_eval_epochs"]) & set(s123["sep_deep_at_eval_epochs"])
                    & set(s456["sep_deep_at_eval_epochs"]) & set(pub["sep_deep_at_eval_epochs"]))
    traj = {str(e): {"new42": new["sep_deep_at_eval_epochs"][e], "s123": s123["sep_deep_at_eval_epochs"][e],
                     "s456": s456["sep_deep_at_eval_epochs"][e], "pub42": pub["sep_deep_at_eval_epochs"][e]}
            for e in common}
    e0 = new["stage1_selected_epoch"]
    late = [e for e in common if e >= e0]
    m_late = {e: (s123["sep_deep_at_eval_epochs"][e] + s456["sep_deep_at_eval_epochs"][e]) / 2 for e in late}
    inside = [e for e in late if min(s123["sep_deep_at_eval_epochs"][e], s456["sep_deep_at_eval_epochs"][e])
              <= new["sep_deep_at_eval_epochs"][e] <= max(s123["sep_deep_at_eval_epochs"][e], s456["sep_deep_at_eval_epochs"][e])]
    below = [e for e in late if new["sep_deep_at_eval_epochs"][e]
             < min(s123["sep_deep_at_eval_epochs"][e], s456["sep_deep_at_eval_epochs"][e])]
    above = [e for e in late if new["sep_deep_at_eval_epochs"][e]
             > max(s123["sep_deep_at_eval_epochs"][e], s456["sep_deep_at_eval_epochs"][e])]
    new_dev = [new["sep_deep_at_eval_epochs"][e] - m_late[e] for e in late]
    pub_dev = [pub["sep_deep_at_eval_epochs"][e] - m_late[e] for e in late]
    pub_below_min = [e for e in late if pub["sep_deep_at_eval_epochs"][e]
                     < min(s123["sep_deep_at_eval_epochs"][e], s456["sep_deep_at_eval_epochs"][e])]
    traj_summary = {
        "epochs_considered": f"common val-eval epochs >= {e0} (the new run's selected epoch): {late}",
        "new42_inside_s123_s456_range_at": inside,
        "new42_below_s123_s456_min_at": below,
        "new42_above_s123_s456_max_at": above,
        "n_inside": len(inside), "n_below": len(below), "n_above": len(above), "n": len(late),
        "pub42_below_s123_s456_min_at": pub_below_min,
        "new42_minus_s123_456_mean_range": [min(new_dev), max(new_dev)],
        "pub42_minus_s123_456_mean_range": [min(pub_dev), max(pub_dev)],
        "at_epoch": {str(e): traj[str(e)] for e in (e0, 260, 320, 362) if str(e) in traj},
    }

    # ---------------- aggregates (mean +/- sample SD)
    def n3(runs):
        out = {k: agg([r["acc_pct"][k] for r in runs]) for k in PROTOCOLS}
        out["sep_last_epoch_362"] = agg([r["sep_deep_last"] for r in runs])
        out["sep_selected_epoch"] = agg([r["sep_deep_selected"] for r in runs])
        out["selected_epochs"] = [r["stage1_selected_epoch"] for r in runs]
        return out
    matched = n3([new, s123, s456])
    published = n3([pub, s123, s456])
    gate("published n=3 S1-TTA mean +/- SD == 91.32 +/- 0.19 (PREDICTIONS.md; headline)",
         round(published["S1-TTA"]["mean"], 2) == 91.32 and round(published["S1-TTA"]["sd_sample"], 2) == 0.19,
         got=[published["S1-TTA"]["mean"], published["S1-TTA"]["sd_sample"]])
    printed = {}
    for name, a in (("matched_n3", matched), ("published_n3", published)):
        printed[name] = {k: f"{a[k]['mean']:.2f} +/- {a[k]['sd_sample']:.2f}" for k in PROTOCOLS}
        printed[name]["sep_last_epoch_362"] = f"{a['sep_last_epoch_362']['mean']:.2f} +/- {a['sep_last_epoch_362']['sd_sample']:.2f}"
        printed[name]["sep_selected_epoch"] = f"{a['sep_selected_epoch']['mean']:.2f} +/- {a['sep_selected_epoch']['sd_sample']:.2f}"
    near_half = []
    for name, a in (("matched_n3", matched), ("published_n3", published)):
        for k in list(PROTOCOLS) + ["sep_last_epoch_362", "sep_selected_epoch"]:
            for f in ("mean", "sd_sample"):
                x = a[k][f] * 100
                if abs(abs(x - np.floor(x)) - 0.5) < 1e-6:
                    near_half.append([name, k, f, a[k][f]])
    gate("no printed 2-dp aggregate lies on a rounding boundary", not near_half, got=near_half)
    delta_means = {k: round(matched[k]["mean"] - published[k]["mean"], 4) for k in PROTOCOLS}
    delta_means["sep_last_epoch_362"] = matched["sep_last_epoch_362"]["mean"] - published["sep_last_epoch_362"]["mean"]
    delta_means["sep_selected_epoch"] = matched["sep_selected_epoch"]["mean"] - published["sep_selected_epoch"]["mean"]

    # ---------------- registered predictions (scored exactly as registered; integer test counts)
    c_new = new["test_correct_of_10000"]["S1-TTA"]
    c123, c456 = s123["test_correct_of_10000"]["S1-TTA"], s456["test_correct_of_10000"]["S1-TTA"]
    lo, hi = min(c123, c456), max(c123, c456)
    p1_pass = lo <= c_new <= hi
    gate("P1 bounds from raw files == registered [91.10, 91.40]", (lo, hi) == (9110, 9140), got=[lo / 100, hi / 100])
    mean_c = (c123 + c456) / 2  # 9125 -> 91.25 %
    band = (mean_c - P1B_HALF_WIDTH_PP * 100, mean_c + P1B_HALF_WIDTH_PP * 100)
    gate("P1b band from raw files == registered [90.95, 91.55]", band == (9095.0, 9155.0), got=[band[0] / 100, band[1] / 100])
    p1b_pass = band[0] <= c_new <= band[1]
    sep = new["sep_deep_last"]
    p2_pass = P2_BAND[0] <= sep <= P2_BAND[1]
    p2_rule = "like seeds 123/456" if sep >= P2_RULE else "like the published seed 42"
    predictions = {
        "source": rel(EXP / "PREDICTIONS.md") + " (written 2026-09-25, before launch)",
        "P1": {"registered": P1_TEXT, "observed_s1_tta_pct": c_new / 100, "interval_pct": [lo / 100, hi / 100],
               "result": "PASS" if p1_pass else "FAIL",
               "margin_to_nearest_bound_pp": round(min(c_new - lo, hi - c_new) / 100, 2),
               "margin_in_test_images": int(min(c_new - lo, hi - c_new)),
               "caveat_registered": ("if the three seeds are exchangeable, a third draw falls between the other two "
                                     "with probability 1/3; P1 is a strict check")},
        "P1b": {"registered": P1B_TEXT, "observed_s1_tta_pct": c_new / 100,
                "interval_pct": [band[0] / 100, band[1] / 100], "result": "PASS" if p1b_pass else "FAIL",
                "distance_from_center_pp": round((c_new - mean_c) / 100, 2)},
        "P2": {"registered": P2_TEXT, "metric": "layer-3 sep_curr_nl at Stage-1 epoch 362 (segments concatenated by (epoch, layer))",
               "observed": sep, "observed_2dp": round(sep, 2), "band": list(P2_BAND),
               "result": "PASS" if p2_pass else "FAIL", "decision_rule_threshold": P2_RULE,
               "decision_rule_outcome": p2_rule},
        "not_registered": {"S2_pct": new["acc_pct"]["S2"], "S2-TTA_pct": new["acc_pct"]["S2-TTA"],
                           "seeds_123_456_S2-TTA": [s123["acc_pct"]["S2-TTA"], s456["acc_pct"]["S2-TTA"]],
                           "selected_epoch_sep": new["sep_deep_selected"],
                           "note": "reported without a prediction (PREDICTIONS.md 'Not registered')"},
    }
    if p2_rule == "like seeds 123/456" and p1b_pass:
        branch = "P2 holds (sep >= 5.6) and P1b holds"
        template = ("Rerunning seed 42 in the seed-123/456 configuration gives S1-TTA X (seeds 123/456: 91.40, 91.10) "
                    "and deepest-block separation Y (7.85, 7.97). The 3.28 of the published seed-42 run therefore "
                    "reflects its configuration (block_curr_lambda 0, depth_order_lambda 0), not seed variance.")
        filled = template.replace("X", f"{c_new / 100:.2f}").replace("Y", f"{sep:.2f}")
    elif p2_rule != "like seeds 123/456":
        branch, filled = "P2 fails (sep < 5.6)", None
    else:
        branch, filled = "P1b fails", None
    predictions["registered_outcome_branch"] = {
        "branch": branch, "registered_sentence_filled": filled,
        "required_qualifiers": [
            f"the separation value is the last Stage-1 epoch (362); at the validation-selected epoch the new seed 42 gives "
            f"{new['sep_deep_selected']:.2f} (epoch {new['stage1_selected_epoch']}), close to the published seed 42's "
            f"{pub['sep_deep_selected']:.2f} (epoch {pub['stage1_selected_epoch']}) and below seeds 123/456 "
            f"({s123['sep_deep_selected']:.2f}/{s456['sep_deep_selected']:.2f}, epoch {s123['stage1_selected_epoch']}); "
            "see sep_convention_caveat",
            "one re-run of one seed; it does not separate block_curr_lambda from depth_order_lambda",
            (f"hardware: the new seed 42 ran on a Colab {new['env']['device0']}; seeds 123/456 ran on a DGX Spark "
             f"{s123['env']['device0']}; the published seed 42 also ran on a Colab {pub['env']['device0']}"),
            "the new run was resumed twice (epoch-3 and epoch-274 checkpoints; RNG restored, DataLoader workers re-seeded)",
        ],
        "p1_fails_but_p1b_holds_rule_applies": (not p1_pass) and p1b_pass,
        "scope_note": ("the registered sentence rests on one re-run seed: in the same seed and split, moving to the "
                       "seed-123/456 loss weights moved the last-epoch deepest sep from 3.28 to "
                       f"{sep:.2f} (both runs on Colab A100). So seed 42 with this split does not by itself produce the "
                       "low last-epoch separation under the matched configuration. The evidence is one re-run. It does "
                       "not separate block_curr_lambda from depth_order_lambda, and it does not exclude run-level "
                       f"nondeterminism (resumes; torch {pub['env']['torch']} vs {new['env']['torch']}; trainer code "
                       "identity is verified only through config.json) as a smaller contributor"),
    }
    t_sel = traj.get(str(e0), {})
    predictions["sep_convention_caveat"] = (
        f"P2 is registered on the last Stage-1 epoch (362) and passes ({sep:.2f}). The paper's tab:gamma0_multiseed note "
        f"prints the separation at each run's validation-selected epoch (3.22/5.65/6.41 at epochs "
        f"{pub['stage1_selected_epoch']}/{s123['stage1_selected_epoch']}/{s456['stage1_selected_epoch']}). On that "
        f"convention the new seed 42 gives {new['sep_deep_selected']:.2f} at epoch {e0}: below seeds 123/456 and close to "
        f"the published seed 42's {pub['sep_deep_selected']:.2f}, because its validation accuracy peaked {min(s123['stage1_selected_epoch'], s456['stage1_selected_epoch']) - e0} "
        f"epochs earlier, while separation was still rising. At matched epochs the new run stays much closer to seeds "
        f"123/456 than to the published seed 42 (epoch {e0}: new {t_sel.get('new42', float('nan')):.2f}, seeds 123/456 "
        f"{t_sel.get('s123', float('nan')):.2f}/{t_sel.get('s456', float('nan')):.2f}, published seed 42 "
        f"{t_sel.get('pub42', float('nan')):.2f}), but it mostly sits slightly below them: at the "
        f"{traj_summary['n']} common evaluation epochs from {e0} to 362 it lies "
        f"{-traj_summary['new42_minus_s123_456_mean_range'][0]:.2f} below to "
        f"{traj_summary['new42_minus_s123_456_mean_range'][1]:.2f} above the seed-123/456 mean; it is below both seeds at "
        f"{traj_summary['n_below']}, inside their range at {traj_summary['n_inside']} and above both at "
        f"{traj_summary['n_above']} of those epochs. The published seed 42 lies "
        f"{-traj_summary['pub42_minus_s123_456_mean_range'][1]:.2f} to "
        f"{-traj_summary['pub42_minus_s123_456_mean_range'][0]:.2f} below that mean (below both seeds at "
        f"{len(traj_summary['pub42_below_s123_s456_min_at'])} of {traj_summary['n']}). Any sentence that uses the "
        "registered result must say that it refers to the last epoch, and it must also report the selected-epoch value.")

    # ---------------- disclosures
    seg = f2["segments_all"]

    def arch(env):
        m = re.search(r"(x86_64|aarch64)", env["platform"])
        return m.group(1) if m else env["platform"]

    def sw(env):
        return (f"torch {env['torch']}, CUDA {env['cuda_version']}, cuDNN {env['cudnn_version']}, "
                f"Python {env['python']}, {arch(env)}")
    for key, r in refs.items():
        envs = {json.dumps(e, sort_keys=True) for e in r["env_per_segment"]}
        gate(f"{r['label']}: env.json identical across its segments", len(envs) == 1, got=len(envs))
    gate("seeds 123/456 on GB10, new and published seed 42 on A100 (env.json)",
         s123["env"]["device0"] == s456["env"]["device0"] == "NVIDIA GB10"
         and new["env"]["device0"] == pub["env"]["device0"] == "NVIDIA A100-SXM4-40GB")
    hardware_table = (
        [{"run": "F2 new seed 42", "segment": i["id"], "dir": i["dir"].split("/")[-1],
          "stage1_epochs_used": i["stage1_epochs_used"],
          "stage2_and_final_test_eval": i["has_final_test_eval"],
          **{k: i["env"][k] for k in ("device0", "torch", "cuda_version", "cudnn_version", "python", "platform")}}
         for i in seg]
        + [{"run": r["label"], "segment": f"segment {j + 1}", "dir": d.split("/")[-1],
            "stage1_epochs_used": list(r["segment_stage1_epoch_ranges"][j]),
            "stage2_and_final_test_eval": j == len(r["segments"]) - 1,
            **{k: r["env_per_segment"][j][k] for k in ("device0", "torch", "cuda_version", "cudnn_version", "python", "platform")}}
           for key, r in refs.items() for j, d in enumerate(r["segments"])])
    disclosures = [
        ("hardware mixing: the matched-config n = 3 combines the new seed 42, trained on a Colab "
         f"{new['env']['device0']} ({sw(new['env'])}), with seeds 123/456, trained on a DGX Spark "
         f"{s123['env']['device0']} ({sw(s123['env'])}). Its SD therefore mixes seed and hardware/software variation. "
         "The published n = 3 mixes the same two platforms (published seed 42 on a Colab "
         f"{pub['env']['device0']}, {sw(pub['env'])}), so the new-vs-published seed-42 comparison is on the same GPU "
         "model but not the same software stack."),
        ("resumes: the new seed-42 run was trained in 4 Colab segments; 2 resumes (from the epoch-3 checkpoint "
         "into segment 3 and from the epoch-274 checkpoint into segment 4), both with rng_restored = true in "
         "events.jsonl. The persistent DataLoader workers are re-seeded on resume, so the run is reproducible from its "
         "checkpoints but not bitwise equal to an uninterrupted run."),
        ckpt_note,
        (f"selected checkpoint: the validation-selected Stage-1 epoch is {new['stage1_selected_epoch']} "
         f"(val {new['best_s1_val']}) in {f2['stage1_selected_epoch_segment'].split('/')[-1]}; S1/S1-TTA therefore "
         "come from a backbone trained in segments 2-3 only; the Stage-2 head (S2/S2-TTA) and the epoch-362 sep "
         "(P2) come from segment 4, after the second resume"),
        ("console log: the Drive trainer log has no trainer output for segment 3 (START header only) and no header "
         "for segment 1; segment 3 is documented by its run-directory jsonl files; its CUDA-RNG restore is not "
         "separately recorded (events.jsonl records rng_restored = true; segment 4's console line confirms python, "
         "numpy, torch CPU and torch CUDA)"),
        ("checkpoint files are on Google Drive only and were not inspected; the selected epoch is identified from the "
         "logs (first val eval attaining best_s1, strict '>' update), as in results/e1_tier1_sep_acc.json"),
        (f"software differs from the plan's note: the F2 segments ran torch {new['env']['torch']} with Python "
         f"{new['env']['python']} and cuDNN {new['env']['cudnn_version']} (PREDICTIONS.md listed the published seed 42's "
         f"torch {pub['env']['torch']} and CR/.venv's 2.14.0+cu130 as known differences)"),
        (f"P1 passes by {predictions['P1']['margin_in_test_images']} test image(s): S1-TTA "
         f"{predictions['P1']['observed_s1_tta_pct']:.2f} against the upper bound "
         f"{predictions['P1']['interval_pct'][1]:.2f}; the registered caveat applies (P1 is a strict check)"
         if p1_pass and predictions['P1']['margin_in_test_images'] <= 5 else None),
    ]
    disclosures = [d for d in disclosures if d]

    seg_wall = {i["id"]: i.get("wall_h_run_start_to_run_end", i["wall_h_run_start_to_last_row"]) for i in seg}
    compute = {
        "segments_wall_h": seg_wall,
        "sum_recorded_wall_h": round(sum(v for v in seg_wall.values() if v), 2),
        "sum_note": ("A100 hours recorded in the run files; excludes seg1 (no rows) and the time between each lost "
                     "segment's last row and the VM loss"),
        "note": ("seg1 has no rows (end time not in the run files); seg2 is run_start to its last epoch row (the VM "
                 "was lost later, during epoch 5); seg3 is run_start to its last epoch row (no run_end); seg4 is "
                 "run_start to run_end"),
        "median_s_per_stage1_epoch": {i["id"]: i["median_s_per_stage1_epoch"] for i in seg if i["median_s_per_stage1_epoch"]},
    }

    def public(r):
        return {k: v for k, v in r.items() if not k.startswith("_") and k not in ("cfg", "segments_all", "superseded_rows")}

    all_ok = all(g["ok"] for g in GATES)
    out = {
        "title": "F2: CIFAR-10 gamma=0 seed 42 re-run in the seed-123/456 configuration (matched-config n = 3)",
        "generated": _dt.date.today().isoformat(),
        "script": "analysis/f2_analysis.py",
        "definitions": {
            "protocols": {"S1": "stage1_best EMA backbone, goodness-sum readout, single crop",
                          "S1-TTA": "same, flip TTA", "S2": "stage2_best attentive head on the frozen stage1_best EMA backbone",
                          "S2-TTA": "same, flip TTA", "source": "final_test_eval in the last segment's events.jsonl, "
                          "cross-checked against preds_*.npz (correct / 10000)"},
            "sep": ("deepest block (layer 3) sep_curr_nl from layer_metrics.jsonl: Stage-1 training-time running mean over "
                    "the epoch; last = epoch 362 (registered P2 metric); selected = the validation-selected Stage-1 epoch"),
            "selected_epoch": "first Stage-1 val-eval epoch whose eval_top1 equals run_end.best_s1 (strict '>' update)",
            "stitching": ("segments ordered by time; a segment's resume event (checkpoint epoch k) supersedes every earlier "
                          "row with epoch > k; the remaining rows must cover epochs 1-362 exactly once with 4 layers each"),
            "sd": "sample SD (ddof = 1)",
        },
        "gates": {"all_passed": all_ok, "n": len(GATES), "checks": GATES},
        "f2_run": {
            **public(f2),
            "config_vs_seed123": d123, "config_vs_seed456": d456, "config_vs_published_seed42": dpub,
            "trainer": {"path": rel(EXP / "trainer.py"), "sha256": fork_sha,
                        "equivalence_tests": f"{tst['n_tests'] - tst['n_failed']}/{tst['n_tests']} passed ({tst['finished']})"},
        },
        "provenance": {
            "segments": seg,
            "launch_note_source": LAUNCH_NOTE_SOURCE,
            "segment_boundaries_in_stitched_stage1_log": boundaries,
            "hardware_per_segment": hardware_table,
            "epochs_trained_twice": dup,
            "stage1_per_epoch": {"columns": cols, "rows": rows},
            "compute": compute,
        },
        "references": {k: public(r) | {"config_vs_f2": cdiff(r["cfg"], f2["cfg"])} for k, r in refs.items()},
        "comparison_vs_published_seed42": comp_pub,
        "comparison_vs_seeds_123_456": comp_m,
        "sep_trajectory_at_common_eval_epochs": traj,
        "sep_trajectory_summary": traj_summary,
        "aggregates": {
            "matched_config_n3": {"runs": ["F2 new seed 42 (Colab A100)", "seed 123 (GB10)", "seed 456 (GB10)"], **matched},
            "published_n3": {"runs": ["published seed 42 (bcl 0 / dol 0, Colab A100)", "seed 123 (GB10)", "seed 456 (GB10)"],
                             **published},
            "printed_2dp": printed,
            "delta_matched_minus_published_mean": delta_means,
            "note": ("the published n = 3 headline (S1-TTA 91.32 +/- 0.19) is not replaced; the matched n = 3 is reported "
                     "next to it (PREDICTIONS.md)"),
        },
        "predictions": predictions,
        "disclosures": disclosures,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(neutral(out), indent=1) + "\n")
    print(f"gates: {sum(g['ok'] for g in GATES)}/{len(GATES)} passed")
    for g in GATES:
        if not g["ok"]:
            print("  FAILED:", g["check"], {k: v for k, v in g.items() if k not in ("check", "ok")})
    print("F2 new s42:", new["acc_pct"], "sep last %.4f sel %.4f @%d" % (new["sep_deep_last"], new["sep_deep_selected"], new["stage1_selected_epoch"]))
    print("P1", predictions["P1"]["result"], "P1b", predictions["P1b"]["result"], "P2", predictions["P2"]["result"], "->", p2_rule)
    print("matched n3:", printed["matched_n3"])
    print("published n3:", printed["published_n3"])
    print("wrote", OUT)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
