"""Shared helpers for the CPU experiments E1 (tier 2) and E2 (camera-ready, NeurIPS 2026).

Everything under FF_HOME is read-only. This module guarantees that importing a trainer
never writes next to it:
  * sys.dont_write_bytecode / PYTHONDONTWRITEBYTECODE (no __pycache__ in the trainer dirs);
  * FF_CKPT_DIR / FF_RUNS_DIR point into $FF_WORK_DIR/_work (the trainers call
    os.makedirs(CKPT_DIR) at import time);
  * the working directory is $FF_WORK_DIR/_work, which holds read-only symlinks
    data_c10 -> CF/revised_training/data_c10 and data_c100 -> PR/cifar_100_fair/data_c100
    (the trainers' loaders use the relative paths ./data_c10 and ./data_c100 and call
    torchvision with download=True, which only verifies the md5 of the existing files).

Every checkpoint/run pairing is spelled out explicitly (timestamps included), so an aborted
duplicate run directory that sits next to the result-bearing one can never be picked up.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HOME = Path(FF_HOME)
PR = HOME / "amir-porjects"
CF = PR / "cifar_10_fair" / "making_v1_v4_fair_comparison"
CR = Path(FF_OUT)                    # output root (was the camera-ready repository)
FFN = HOME / "ff-next"
EXP = CR / "experiments"
WORK = Path(FF_WORK) / "_work"       # symlinked data dirs + dummy trainer ckpt/run dirs
RESULTS = CR / "results"

GA = CF / "basic_fair_revised_valfix_cp_fair_hardness_gated" / "ff_gated_ablation"
CPF = CF / "basic_fair_revised_valfix_cp_fair"
RT = CF / "revised_training"
TRIO_RUNS = RT / "extracted" / "ff_cifar100_multiseed"
C10_GATED_MS = RT / "extracted" / "ff_cifar10_gated_multiseed"
TRIO_CKPT = FFN / "checkpoints" / "trio"                      # flat layout, read-only
GATED_K0_CKPT = Path(FF_WORK) / "ckpts" / "gated_k0_d128"      # stage1_best extracted from the gated multiseed archive (see RELEASE_CHECKPOINTS.md)
MR = CF / "mgc_rebuttal"

DATA_LINKS = {
    "data_c10": RT / "data_c10",
    "data_c100": PR / "cifar_100_fair" / "data_c100",
}

# Normalization by trainer lineage (every trainer used here: hardness_gated, cp_fair,
# revised_training *_fixed, mgc_trainer -> CIFAR-10 std (0.2023,0.1994,0.2010); the v2
# lineage's (0.2470,0.2435,0.2616) is NOT used by any checkpoint here).
NORM = {
    10: dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    100: dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
}

# ---------------------------------------------------------------- trainers
TRAINERS = {
    # CP-FAIR (gamma=0.7 constant, D256, 32 experts top-4)
    "cp_fair": CPF / "basic_fair_revised_valfix_cp_fair.py",
    # FF-purity-fixed CP-FAIR trainer (C10 gamma=0 seeds 123/456 were trained by it)
    "cp_fair_fixed": RT / "basic_fair_revised_valfix_cp_fair_fixed.py",
    # hardness-gated trainer that trained the 9 GA seed-42 arms on Colab
    "gated": CF / "basic_fair_revised_valfix_cp_fair_hardness_gated" / "basic_fair_revised_valfix_cp_fair_hardness_gated.py",
    # FF-purity-fixed gated trainer (kappa=0 D128 seeds 123/456); byte-identical to the
    # copy shipped with the extracted multiseed runs
    "gated_fixed": RT / "basic_fair_revised_valfix_cp_fair_hardness_gated_fixed.py",
    # CIFAR-100 fixed trainer (trio); byte-identical to the copy in extracted/ff_cifar100_multiseed
    "c100_fixed": RT / "basic_fair_revised_valfix_cp_fair_cifar100_fixed.py",
    # MGC trainer (validation arm for E1)
    "mgc": MR / "mgc_trainer.py",
}


def _ga_run(arm, stamp):
    return GA / "runs" / f"ff_cifar10_{arm}_L4_D128_bs512_seed42_{stamp}"


def _ga_ckpt(arm):
    return GA / "checkpoints" / f"ff_cifar10_{arm}_L4_D128_P2_seed42_stage1_best.pt"


# --------------------------------------------------------------- registry
# name -> dict(dataset, trainer, run_dir, ckpt, config, seed, [cfg_from_ckpt])
REG = {}

# C10 gated L4/D128 seed-42 arms (GA). All trained by the (unfixed) hardness_gated trainer,
# SAM off. prev_only uses the completed re-run _042603 (the _192927 dir is an aborted start).
GA_ARMS = {
    "baseline_constant": "20260313_232043",
    "lcff_prefix": "20260315_084228",
    "no_all_collab": "20260315_124117",
    "adaptive_t1_k0": "20260314_032410",
    "adaptive_t3_k0": "20260314_071928",
    "adaptive_t1_k2": "20260314_112422",
    "adaptive_t1_k4": "20260314_152252",   # config.json is an HTML 404 -> ckpt['cfg']
    "adaptive_prev_only": "20260315_042603",
    "adaptive_prev_t1_k2": "20260315_044213",
}
for arm, stamp in GA_ARMS.items():
    REG[f"c10_gated_{arm}_seed42"] = dict(
        dataset="cifar10", trainer="gated", run_dir=_ga_run(arm, stamp), ckpt=_ga_ckpt(arm),
        config=f"c10_d128_{arm}", seed=42,
        cfg_from_ckpt=(arm == "adaptive_t1_k4"))

# C10 gated kappa=0 D128 seeds 123/456 (fixed gated trainer, SAM on)
for s, stamp in ((123, "20260426_071309"), (456, "20260426_150234")):
    REG[f"c10_gated_k0_seed{s}"] = dict(
        dataset="cifar10", trainer="gated_fixed",
        run_dir=C10_GATED_MS / f"runs_gated_k0_seed{s}" / f"ff_cifar10_cifar10_gated_k0_seed{s}_L4_D128_bs512_seed{s}_{stamp}",
        ckpt=GATED_K0_CKPT / f"ff_cifar10_cifar10_gated_k0_seed{s}_L4_D128_P2_seed{s}_stage1_best.pt",
        config="c10_d128_gated_k0", seed=s)

# C10 D256 CP-FAIR (gamma=0.7 constant) seeds 42/123/456/789 (cp_fair trainer, SAM on)
_CPF_RUNS = {
    42: ("runs", "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035",
         "checkpoints", "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_P2_seed42_stage1_best.pt"),
    123: ("runs_seed123", "ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_bs512_seed123_20260320_001527",
          "checkpoints_seed123", "ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_P2_seed123_stage1_best.pt"),
    456: ("runs_seed456", "ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_bs512_seed456_20260327_080127",
          "checkpoints_seed456", "ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_P2_seed456_stage1_best.pt"),
    789: ("runs_seed789", "ff_cifar10_cifar10_cp_fair_clean_progressive_seed789_L4_D256_bs512_seed789_20260330_064018",
          "checkpoints_seed789", "ff_cifar10_cifar10_cp_fair_clean_progressive_seed789_L4_D256_P2_seed789_stage1_best.pt"),
}
for s, (rd, rn, cd, cn) in _CPF_RUNS.items():
    REG[f"c10_cp_fair_seed{s}"] = dict(
        dataset="cifar10", trainer="cp_fair", run_dir=CPF / rd / rn, ckpt=CPF / cd / cn,
        config="c10_d256_cp_fair", seed=s)

# C10 D256 gamma=0 seeds 123/456 (fixed CP-FAIR trainer, FF_GAMMA_SCALE=0). Seed 42 (published
# run ablations/no_all_collab ..._20260312_132043) has no recoverable checkpoint -> not in E1 tier 2.
for s, stamp in ((123, "20260417_235110"), (456, "20260422_135437")):
    base = RT / "out" / f"gamma0_seed{s}"
    REG[f"c10_gamma0_seed{s}"] = dict(
        dataset="cifar10", trainer="cp_fair_fixed",
        run_dir=base / "runs" / f"ff_cifar10_cifar10_gamma0_seed{s}_L4_D256_bs512_seed{s}_{stamp}",
        ckpt=base / "checkpoints" / f"ff_cifar10_cifar10_gamma0_seed{s}_L4_D256_P2_seed{s}_stage1_best.pt",
        config="c10_d256_gamma0", seed=s)

# CIFAR-100 trio (c100_fixed trainer); result-bearing run dirs (the other dir per seed is an
# aborted start without final_test_eval)
_TRIO = {
    ("gamma0", 42): "20260420_101044", ("gamma0", 123): "20260421_124316", ("gamma0", 456): "20260422_142245",
    ("gated_k0", 42): "20260422_185437", ("gated_k0", 123): "20260423_150626", ("gated_k0", 456): "20260424_153513",
    ("cumulative", 42): "20260418_032149", ("cumulative", 123): "20260419_010632", ("cumulative", 456): "20260420_051234",
}
for (var, s), stamp in _TRIO.items():
    REG[f"c100_{var}_seed{s}"] = dict(
        dataset="cifar100", trainer="c100_fixed",
        run_dir=TRIO_RUNS / f"runs_{var}_seed{s}" / f"ff_cifar100_cifar100_{var}_seed{s}_L4_D256_bs256_seed{s}_{stamp}",
        ckpt=TRIO_CKPT / f"ff_cifar100_cifar100_{var}_seed{s}_L4_D256_P2_seed{s}_stage1_best.pt",
        config=f"c100_d256_{var}", seed=s)

# MGC D128 seed 42 (validation arm for the E1 tier-2 code; gradshare expected block-3 5.2346)
REG["c10_mgc_c1_seed42"] = dict(
    dataset="cifar10", trainer="mgc",
    run_dir=MR / "runs" / "ff_cifar10_mgc_c1_L4_D128_bs512_seed42_20260723_223516",
    ckpt=MR / "checkpoints" / "ff_cifar10_mgc_c1_L4_D128_P2_seed42_stage1_best.pt",
    config="c10_d128_mgc_c1", seed=42)

# Resumed runs: earlier run-dir segments whose Stage-1 logs precede the result-bearing (last)
# segment (the last segment's Stage-1 log does not start at epoch 1). The best-val epoch -- and so
# the stage1_best checkpoint -- can lie in an earlier segment, so logs are read across segments.
# (Segments that restarted from scratch -- c100 gamma0 s42 _20260417_215649, c100 gated_k0 s123
# _20260423_102837, GA prev_only _20260314_192927 -- are NOT prior segments.)
PRIOR_SEGMENTS = {
    "c10_cp_fair_seed42": [CPF / "runs" / "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260218_144731"],
    "c10_cp_fair_seed123": [CPF / "runs_seed123" / "ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_bs512_seed123_20260318_001952"],
    "c10_cp_fair_seed456": [CPF / "runs_seed456" / "ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_bs512_seed456_20260324_135535"],
    "c100_gamma0_seed123": [TRIO_RUNS / "runs_gamma0_seed123" / "ff_cifar100_cifar100_gamma0_seed123_L4_D256_bs256_seed123_20260421_015141"],
    "c100_gamma0_seed456": [TRIO_RUNS / "runs_gamma0_seed456" / "ff_cifar100_cifar100_gamma0_seed456_L4_D256_bs256_seed456_20260422_012047"],
    "c100_gated_k0_seed456": [TRIO_RUNS / "runs_gated_k0_seed456" / "ff_cifar100_cifar100_gated_k0_seed456_L4_D256_bs256_seed456_20260424_065206"],
    "c100_cumulative_seed123": [TRIO_RUNS / "runs_cumulative_seed123" / "ff_cifar100_cifar100_cumulative_seed123_L4_D256_bs256_seed123_20260418_190627"],
    "c100_cumulative_seed456": [TRIO_RUNS / "runs_cumulative_seed456" / "ff_cifar100_cifar100_cumulative_seed456_L4_D256_bs256_seed456_20260419_141800"],
}
for _n, _segs in PRIOR_SEGMENTS.items():
    REG[_n]["prior_segments"] = _segs

E1_POINTS = (
    ["c10_gamma0_seed123", "c10_gamma0_seed456"]
    + [f"c10_cp_fair_seed{s}" for s in (42, 123, 456, 789)]
    + ["c10_gated_adaptive_t1_k0_seed42", "c10_gated_k0_seed123", "c10_gated_k0_seed456"]
    + [f"c100_{v}_seed{s}" for v in ("gamma0", "gated_k0", "cumulative") for s in (42, 123, 456)]
)
E1_VALIDATION = "c10_mgc_c1_seed42"
E2_C10 = [f"c10_gated_{a}_seed42" for a in GA_ARMS]
E2_C100 = [f"c100_{v}_seed{s}" for v in ("gamma0", "gated_k0", "cumulative") for s in (42, 123, 456)]

# E1 configuration labels (for the within/across-config analysis). The kappa=0 D128 seed 42
# (GA, SAM off) and seeds 123/456 (fixed trainer, SAM on) are kept as SEPARATE configurations
# (D13); C10 gamma=0 has only seeds 123/456 (seed 42 unrecoverable, and it used a different
# config anyway -- D7).
E1_CONFIG = {
    "c10_gamma0_seed123": "c10_d256_gamma0", "c10_gamma0_seed456": "c10_d256_gamma0",
    **{f"c10_cp_fair_seed{s}": "c10_d256_cp_fair" for s in (42, 123, 456, 789)},
    "c10_gated_adaptive_t1_k0_seed42": "c10_d128_gated_k0_samoff",
    "c10_gated_k0_seed123": "c10_d128_gated_k0_samon", "c10_gated_k0_seed456": "c10_d128_gated_k0_samon",
    **{f"c100_{v}_seed{s}": f"c100_d256_{v}" for v in ("gamma0", "gated_k0", "cumulative") for s in (42, 123, 456)},
}


# ----------------------------------------------------------------- setup
def setup_env(threads=6):
    """Must be called before importing torch-heavy trainer modules."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", \
        "run with CUDA_VISIBLE_DEVICES='' (these recomputations run on CPU)"
    (WORK / "ckpt_dummy").mkdir(parents=True, exist_ok=True)
    (WORK / "runs_dummy").mkdir(parents=True, exist_ok=True)
    for name, target in DATA_LINKS.items():
        link = WORK / name
        if not link.exists():
            if link.is_symlink():
                link.unlink()
            link.symlink_to(target)
    os.environ["FF_CKPT_DIR"] = str(WORK / "ckpt_dummy")
    os.environ["FF_RUNS_DIR"] = str(WORK / "runs_dummy")
    os.chdir(WORK)
    import torch
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision("highest")
    assert not torch.cuda.is_available(), "CUDA must be hidden"
    return torch


_MODULES = {}


def _ensure_tqdm():
    """The trainers import `from tqdm.auto import tqdm` at module level (progress bars of the
    training loop only). If tqdm is not installed, rather than
    modify the shared venv, register a pass-through stub. Nothing measured here uses tqdm."""
    try:
        import tqdm.auto  # noqa: F401
        return "tqdm"
    except ImportError:
        import types

        class _Tqdm:
            def __init__(self, iterable=None, *a, **k):
                self.iterable = iterable

            def __iter__(self):
                return iter(self.iterable)

            def set_postfix(self, *a, **k):
                pass

            def write(self, s, *a, **k):
                print(s)

            def update(self, *a, **k):
                pass

            def close(self):
                pass

        import importlib.machinery
        import torchvision  # noqa: F401  (import torch._dynamo & co. before the stub exists)
        pkg = types.ModuleType("tqdm")
        pkg.__spec__ = importlib.machinery.ModuleSpec("tqdm", None, is_package=True)
        pkg.__path__ = []
        auto = types.ModuleType("tqdm.auto")
        auto.__spec__ = importlib.machinery.ModuleSpec("tqdm.auto", None)
        auto.tqdm = _Tqdm
        pkg.auto = auto
        pkg.tqdm = _Tqdm
        sys.modules["tqdm"] = pkg
        sys.modules["tqdm.auto"] = auto
        return "stub"


def load_trainer(key):
    """Import a trainer file under a unique module name without writing bytecode."""
    if key in _MODULES:
        return _MODULES[key]
    _ensure_tqdm()
    path = Path(TRAINERS[key])
    spec = importlib.util.spec_from_file_location(f"ffcr_trainer_{key}", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULES[key] = mod
    return mod


def persisted_config(entry, ckpt=None):
    """config.json of the run, or ckpt['cfg'] when config.json is unusable (t1_k4)."""
    path = Path(entry["run_dir"]) / "config.json"
    src = str(path)
    if entry.get("cfg_from_ckpt"):
        assert ckpt is not None and isinstance(ckpt.get("cfg"), dict)
        persisted, src = dict(ckpt["cfg"]), "ckpt['cfg'] (config.json is an HTML 404 page)"
    else:
        persisted = json.load(open(path))
        persisted = persisted.get("config", persisted)
    return persisted, src


def build_cfg(M, persisted):
    cfg = M.FFConfig()
    for k, v in persisted.items():
        if k == "device":
            continue
        setattr(cfg, k, v)
    cfg.device = "cpu"
    return cfg


def cfg_vs_ckpt_diff(persisted, ckpt):
    """Keys whose values differ between the run's config.json and ckpt['cfg'] (device ignored)."""
    c = ckpt.get("cfg")
    if not isinstance(c, dict):
        return None
    out = {}
    for k in sorted(set(persisted) | set(c)):
        if k == "device":
            continue
        if persisted.get(k, "<missing>") != c.get(k, "<missing>"):
            out[k] = [persisted.get(k, "<missing>"), c.get(k, "<missing>")]
    return out


def read_events(run_dir):
    fte = rend = None
    with open(Path(run_dir) / "events.jsonl") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("event") == "final_test_eval":
                fte = ev
            elif ev.get("event") == "run_end":
                rend = ev
    assert fte is not None and rend is not None, f"no final_test_eval/run_end in {run_dir}"
    return fte, rend


def segments(entry):
    """Run-dir segments of an entry, oldest first (prior resumed segments, then run_dir)."""
    if isinstance(entry, (str, Path)):
        return [Path(entry)]
    return [Path(p) for p in entry.get("prior_segments", [])] + [Path(entry["run_dir"])]


def best_epoch_row(entry):
    """epoch_metrics row of the best Stage-1 val epoch (the epoch that wrote stage1_best), read
    across resumed segments (a later segment's row for the same epoch overrides an earlier one)."""
    by_epoch = {}
    for seg in segments(entry):
        p = seg / "epoch_metrics.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("stage") == 1 and r.get("eval_top1") is not None:
                r["_segment"] = str(seg)
                by_epoch[int(r["epoch"])] = r
    best = None
    for ep in sorted(by_epoch):  # first epoch attaining the max (trainer saves on strict '>')
        r = by_epoch[ep]
        if best is None or r["eval_top1"] > best["eval_top1"]:
            best = r
    return best


def layer_metrics_at_epoch(entry, epoch):
    """Stage-1 per-layer logged metrics at `epoch`, across resumed segments (the latest segment
    that logged that epoch wins). Keys missing in layer_metrics.jsonl are filled from
    layer_metrics_raw.jsonl (which also carries e.g. margin_blk_nl, inc_margin_nl_std)."""
    found = {}
    for seg in segments(entry):
        rows = {}
        for fname in ("layer_metrics.jsonl", "layer_metrics_raw.jsonl"):
            p = seg / fname
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("stage") == 1 and r.get("epoch") == epoch:
                        row = rows.setdefault(int(r["layer"]), {"_segment": str(seg)})
                        for k, v in r.items():
                            row.setdefault(k, v)
        if rows:
            found = rows
    return found


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_id(path):
    st = os.stat(path)
    return dict(path=str(path), size=int(st.st_size), sha256=sha256(path))


def loaders(M, cfg, run_dir, workers=0):
    """The trainer's own Stage-1 loaders (train triple-view aug, val = own split, test).
    Asserts that the split regenerated from cfg.seed equals the run's split_indices.json."""
    train_loader, val_loader, test_loader, split_info = M.make_loaders_stage1(
        cfg.batch_size, cfg.test_batch_size, seed=cfg.seed,
        val_fraction=cfg.val_fraction, workers=workers)
    run_split = json.load(open(Path(run_dir) / "split_indices.json"))
    assert split_info["val_idx"] == run_split["val_idx"], "val split mismatch"
    assert split_info["train_idx"] == run_split["train_idx"], "train split mismatch"
    # normalization check (the C10 std trap)
    tf = test_loader.dataset.transform
    norm = [t for t in tf.transforms if t.__class__.__name__ == "Normalize"][0]
    exp = NORM[int(cfg.num_classes)]
    assert tuple(round(float(x), 4) for x in norm.std) == exp["std"], (norm.std, exp["std"])
    assert tuple(round(float(x), 4) for x in norm.mean) == exp["mean"], (norm.mean, exp["mean"])
    return train_loader, val_loader, test_loader, split_info, run_split


def load_nets(M, cfg, ckpt, which=("net_ema",)):
    nets = {}
    for w in which:
        key = "model" if w == "model" else "net_ema"
        net = M.FFHybridNet(cfg)
        net.load_state_dict(ckpt[key], strict=True)
        net.eval()
        nets[w] = net
    return nets


# ------------------------------------------------------------------ identity gate
# Strict gate (as specified): recomputed EMA S1 no-TTA test top-1 within 1e-4 (0.01 pp = one
# test image) of events.jsonl final_test_eval.stage1_test_top1_no_tta.
# Numerics fallback (pre-declared, reported separately and disclosed if ever used): the GA/C100
# trainers ran on A100s with TF32 matmuls/convs (set_float32_matmul_precision('high') in the
# trainers' __main__), while we recompute in exact FP32 on CPU. On the smoke checkpoint the
# stored GPU scores differ from ours by median 4e-4, 99.9th pct 0.27 (scores ~4, max 35), which
# flips a few near-tie examples. A checkpoint that misses the strict gate passes the fallback only
# if |delta| <= 3 test images AND >= 99.9% argmax agreement with the run's own stored
# preds_stage1_test_no_tta.npz (same checkpoint, same test images, same labels).
GATE_TOL = 1e-4
NUMERICS_MAX_EXAMPLES = 3
NUMERICS_MIN_AGREEMENT = 0.999


def test_gate(pred, y, anchor, run_dir, scores=None):
    import numpy as np
    pred = np.asarray(pred).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    n = len(y)
    acc = float((pred == y).mean())
    delta = abs(acc - float(anchor))
    strict = delta <= GATE_TOL + 1e-9
    cmp = None
    p = Path(run_dir) / "preds_stage1_test_no_tta.npz"
    if p.exists():
        z = np.load(p)
        cmp = dict(path=str(p), y_true_equal=bool(np.array_equal(z["y_true"].astype(np.int64), y)),
                   argmax_agreement=float((z["scores"].argmax(1) == pred).mean()),
                   n_disagree=int((z["scores"].argmax(1) != pred).sum()))
        if scores is not None:
            d = np.abs(z["scores"].astype(np.float64) - np.asarray(scores, dtype=np.float64))
            cmp.update(score_absdiff_median=float(np.median(d)), score_absdiff_p999=float(np.quantile(d, 0.999)),
                       score_absdiff_max=float(d.max()))
    numerics = bool((not strict) and delta <= NUMERICS_MAX_EXAMPLES / n + 1e-9 and cmp is not None
                    and cmp["y_true_equal"] and cmp["argmax_agreement"] >= NUMERICS_MIN_AGREEMENT)
    return dict(recomputed=acc, anchor=float(anchor), delta=delta, delta_examples=int(round(delta * n)),
                strict_pass=bool(strict), numerics_fallback_pass=numerics, pass_=bool(strict or numerics),
                stored_comparison=cmp,
                rule=f"strict |delta|<={GATE_TOL}; fallback |delta|<={NUMERICS_MAX_EXAMPLES}/{n} and "
                     f"argmax agreement with stored GPU preds >= {NUMERICS_MIN_AGREEMENT}")


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
