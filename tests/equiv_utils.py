#!/usr/bin/env python3
"""CPU-only helpers for the trainer equivalence tests (tests/test_equiv_*.py).

The camera-ready trainers in ../trainers add opt-in features (MGC, squared-hinge barrier,
F1 block-0 controls, RNG state in checkpoints) and wrap the previous-block depth-order
aggregation in torch.no_grad(). These helpers let a test run ONE real Stage-1 training step
(one fixed batch through every block: SAM two-pass update or plain AdamW step, EMA update,
queue update) with two trainer modules from the same seed and compare everything the step
touched, bitwise.

Reference trainers are located through environment variables (nothing is hard-coded):
  FF_V1_TRAINERS        directory with the published v1 trainers (the NeurIPS 2026 submission
                        supplement's trainers/ folder); identity is checked by SHA-256.
  FF_REBUTTAL_TRAINERS  directory with the rebuttal trainers mgc_trainer.py, mgc_c100_trainer.py
                        and hinge_trainer.py (checked by SHA-256).
  FF_F1_TRAINER         path to the F1 fork (MGC block-0 control), checked by SHA-256.
  FF_F2_TRAINER         path to the F2 fork (RNG save/restore on the CIFAR-10 trainer).
  FF_REFERENCE_RUNS     root under which the reference run directories (config.json) live.
Tests whose reference is not available are reported as "skipped", never as passed.

Run with CUDA hidden:  CUDA_VISIBLE_DEVICES='' python tests/run_all.py
"""
import copy
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("FF_TEST_THREADS", "6")))
if torch.cuda.is_available():
    raise SystemExit("refusing to run: CUDA is visible. Use CUDA_VISIBLE_DEVICES=''.")

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = os.path.dirname(HERE)
TRAINERS = os.path.join(SUPP, "trainers")

# SHA-256 of every reference file the tests compare against.
V1_SHA = {
    "cp_fair_cifar10.py": "340f6933ae09dc2c5b4b95ca47a8458f165e67d0c5b6f4145d2db7a4dcc1bc19",
    "cp_fair_cifar100.py": "0fa6c0427cb3ec66e5d89bf6bea2957d7a38da405f9686bcd2a7289180645a1d",
    "cp_fair_hardness_gated.py": "71d684e4e90500b777cc8b106ea0bb4c86c65492ca4224a70b1e21346348aa3b",
    "cp_fair_tiny_imagenet.py": "aef8aae711bb82af7419d297d640f5ca18ecbd12aab2eda616984d410bf578a7",
    "cp_fair_core.py": "23f640916d00abe2f36124944744705a9ecc652de819361143266ffb79cdf540",
}
REBUTTAL_SHA = {
    "mgc_trainer.py": "b0515dbb8e00288d003f4e2a8ffad792dc49fd7e8ca509eec8a02e130089cb24",
    "mgc_c100_trainer.py": "48ab31dd18e881a8827a7ae936ec0634ce2cb313f381812af995c660a6f3b98e",
    "hinge_trainer.py": "545d5c3aed4a1c090e67e53e3b6e21061e9afddc389502c408969fa5e58d3c0d",
}
F1_SHA = "83fa1519b4a5793e1e1fd0db5beebbc7180fbb5f2939ae71cc6f30e4daf1082d"
F2_SHA = "c78d850bd965a7bfaf0f6cfc7752820d0e4d3aa9518fc0348b1a31461ceb7e06"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ test bookkeeping
class Recorder:
    def __init__(self, name):
        self.out = {"test_file": name, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "torch": torch.__version__, "numpy": np.__version__,
                    "python": sys.version.split()[0], "threads": torch.get_num_threads(),
                    "references": {}, "tests": {}}
        self.failed = []
        self.skipped = []

    def ref(self, key, path, expected_sha):
        if not path or not os.path.exists(path):
            self.out["references"][key] = {"available": False}
            return None
        got = sha256(path)
        ok = (got == expected_sha) if expected_sha else True
        self.out["references"][key] = {"available": True, "sha256": got, "sha256_ok": ok,
                                       "file": os.path.basename(path)}
        if not ok:
            self.record(f"REF_{key}_sha256", False, {"expected": expected_sha, "got": got})
            return None
        return path

    def record(self, name, ok, detail=None):
        self.out["tests"][name] = {"status": "pass" if ok else "FAIL", "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail is not None else ""),
              flush=True)
        if not ok:
            self.failed.append(name)

    def skip(self, name, why):
        self.out["tests"][name] = {"status": "skipped", "detail": why}
        self.skipped.append(name)
        print(f"[SKIP] {name}  {why}", flush=True)

    def finish(self, path=None):
        self.out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.out["n_tests"] = len(self.out["tests"])
        self.out["n_failed"] = len(self.failed)
        self.out["n_skipped"] = len(self.skipped)
        self.out["failed"] = self.failed
        self.out["skipped"] = self.skipped
        if path:
            with open(path, "w") as f:
                json.dump(self.out, f, indent=1, sort_keys=True, default=str)
        n_run = self.out["n_tests"] - self.out["n_skipped"]
        print(f"\n{n_run - len(self.failed)}/{n_run} passed, {len(self.skipped)} skipped", flush=True)
        return 1 if self.failed else 0


def v1_path(name):
    d = os.environ.get("FF_V1_TRAINERS")
    return os.path.join(d, name) if d else None


def rebuttal_path(name):
    d = os.environ.get("FF_REBUTTAL_TRAINERS")
    return os.path.join(d, name) if d else None


# ------------------------------------------------------------------ module loading
_SCRATCH = tempfile.mkdtemp(prefix="ff_equiv_")
os.environ["FF_CKPT_DIR"] = os.path.join(_SCRATCH, "import_ckpt")  # trainers mkdir at import
os.environ["FF_RUNS_DIR"] = os.path.join(_SCRATCH, "import_runs")


def load_trainer(path, name):
    """Import `path` as a fresh module `name` without writing __pycache__ next to it."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cr_trainer(name):
    return load_trainer(os.path.join(TRAINERS, name), "cr_" + name[:-3])


# ------------------------------------------------------------------ one training step
def seed_all(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def build(mod, cfg, seed=1234):
    """Mirror of the trainers' main(): model, EMA, SAM optimizers, schedulers, queues, router tau."""
    seed_all(seed)
    net = mod.FFHybridNet(cfg).to("cpu")
    net_ema = copy.deepcopy(net).to("cpu") if cfg.use_ema else None
    if net_ema is not None:
        for p in net_ema.parameters():
            p.requires_grad_(False)
    opts, scheds = [], []
    for li, block in enumerate(net.blocks):
        params = list(block.parameters())
        if li == 0:
            params += list(net.embed.parameters())
        if getattr(cfg, "gamma_gating_mode", "constant") == "adaptive":
            if getattr(cfg, "gamma_kappa_learnable", False) and hasattr(net, "gamma_kappa_params"):
                params.append(net.gamma_kappa_params[li])
            if getattr(cfg, "gamma_tau_learnable", False) and hasattr(net, "gamma_tau_param"):
                params.append(net.gamma_tau_param)
        opt = mod.SAM(params, torch.optim.AdamW, rho=cfg.sam_rho, adaptive=cfg.sam_adaptive,
                      lr=cfg.lr, weight_decay=cfg.weight_decay)
        opts.append(opt)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt.base_optimizer,
            lr_lambda=lambda e, total=cfg.epochs_stage1, w=cfg.warmup_epochs:
            mod.cosine_warmup_lambda(e, total, w))
        scheds.append(sched)
    queues = [mod.ContrastiveQueue(cfg.queue_size, cfg.proj_dim, device="cpu")
              for _ in range(cfg.num_blocks)]
    tau = cfg.gate_tau_start  # router_tau_at(epoch - 1) for epoch 1
    net.set_router_tau(tau, depth_slope=cfg.gate_tau_depth_slope)
    return net, net_ema, opts, scheds, queues


def make_batch(cfg, B, seed=2026):
    g = torch.Generator().manual_seed(seed)
    x1, x2, x3 = (torch.randn(B, 3, cfg.img_size, cfg.img_size, generator=g) for _ in range(3))
    y = torch.arange(B) % cfg.num_classes
    return (x1, x2, x3, y)


def clone_state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()} if m is not None else None


def run_steps(mod, cfg, batches, build_seed=1234, step_seed=7):
    """len(batches) Stage-1 steps through the trainer's own train_epoch_stage1."""
    net, net_ema, opts, scheds, queues = build(mod, cfg, seed=build_seed)
    init = clone_state(net)
    seed_all(step_seed)
    stats = mod.train_epoch_stage1(net, net_ema, opts, list(batches), cfg, queues, epoch=1)
    return {
        "init": init,
        "stats": stats,
        "net": clone_state(net),
        "ema": clone_state(net_ema),
        "opt": [copy.deepcopy(o.base_optimizer.state_dict()) for o in opts],
        "queues": [(q.z.clone(), q.y.clone(), int(q.ptr), bool(q.full)) for q in queues],
        "rng_torch": torch.get_rng_state().clone(),
        "rng_numpy": np.random.get_state()[1].copy(),
        "rng_python": random.getstate(),
    }


def _maxdiff(a, b):
    """0.0 if bitwise equal; else max |a-b| (inf on structure/shape/dtype mismatch)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return math.inf
        if torch.equal(a, b):
            return 0.0
        if a.is_floating_point():
            return float((a.double() - b.double()).abs().max().item())
        return math.inf
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return 0.0 if np.array_equal(a, b) else math.inf
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf
        return max([_maxdiff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max([_maxdiff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return 0.0
        return abs(a - b)
    return 0.0 if a == b else math.inf


KEYS_ALL = ("init", "stats", "net", "ema", "opt", "queues", "rng_torch", "rng_numpy", "rng_python")


def compare(ra, rb, keys=KEYS_ALL):
    return {k: _maxdiff(ra[k], rb[k]) for k in keys}


def bitwise(d):
    return all(v == 0.0 for v in d.values())


def loss_per_block(r):
    return [float(l["loss_total"]) for l in r["stats"]]


MGC_TELEMETRY = ("loss_mgc", "mgc_lam_mean", "mgc_lam_p95")


def check_pair(rec, name, mod_a, cfg_a, mod_b, cfg_b, batches, keys=KEYS_ALL, info=None,
               a_only_zero_keys=()):
    """Run both modules on the same batches; record bitwise equality of every key.

    a_only_zero_keys: logged metrics that module A (a rebuttal trainer) always records and module B
    records only when the corresponding feature is on (e.g. the MGC telemetry, which the rebuttal
    trainers log as 0.0 even with MGC off). They must be exactly 0.0 in A; every other logged value,
    the parameters, EMA, optimizer state, queues and RNG state must be bitwise equal."""
    t0 = time.time()
    ra = run_steps(mod_a, cfg_a, batches)
    rb = run_steps(mod_b, cfg_b, batches)
    extra_ok = True
    extra = {}
    if a_only_zero_keys:
        sa = []
        for la, lb in zip(ra["stats"], rb["stats"]):
            la = dict(la)
            for k in a_only_zero_keys:
                if k in la and k not in lb:
                    extra.setdefault(k, []).append(la[k])
                    extra_ok &= (la.pop(k) == 0.0)
            sa.append(la)
        ra = dict(ra, stats=sa)
    d = compare(ra, rb, keys)
    ok = bitwise(d) and extra_ok
    detail = {"bitwise": bitwise(d), "max_abs_diff": d,
              "loss_total_per_block": [round(v, 6) for v in loss_per_block(rb)],
              "sec": round(time.time() - t0, 1)}
    if a_only_zero_keys:
        detail["reference_only_telemetry_all_zero"] = extra_ok
        detail["reference_only_telemetry"] = {k: v for k, v in extra.items()}
    if info:
        detail.update(info)
    rec.record(name, ok, detail)
    return ra, rb


# ------------------------------------------------------------------ planned config via main()
class _StopAfterConfig(Exception):
    pass


def planned_config(trainer_path, env):
    """Run the trainer's real main() under `env` until it has written config.json (the data
    loader is replaced by a function that raises). Returns the config dict."""
    saved = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix="ff_plan_", dir=_SCRATCH)
    try:
        for k in list(os.environ):
            if k.startswith("FF_") and k not in ("FF_TEST_THREADS",):
                del os.environ[k]
        os.environ.update({k: str(v) for k, v in env.items()})
        os.environ["FF_RUNS_DIR"] = os.path.join(tmp, "runs")
        os.environ["FF_CKPT_DIR"] = os.path.join(tmp, "ckpt")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        name = f"ff_plan_{abs(hash((trainer_path, tuple(sorted(env.items())))))}"
        mod = load_trainer(trainer_path, name)

        def _stop(*args, **kwargs):
            raise _StopAfterConfig()

        mod.make_loaders_stage1 = _stop
        try:
            mod.main()
        except _StopAfterConfig:
            pass
        else:
            raise RuntimeError("main() returned without reaching make_loaders_stage1")
        runs = sorted(glob.glob(os.path.join(tmp, "runs", "*")))
        if len(runs) != 1:
            raise RuntimeError(f"expected one run dir, found {runs}")
        with open(os.path.join(runs[0], "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(runs[0], "layer_metric_keys.json")) as f:
            keys = json.load(f)
        sys.modules.pop(name, None)
        return cfg, keys
    finally:
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def diff_configs(a, b):
    out = {}
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<MISSING>"), b.get(k, "<MISSING>")
        if va != vb:
            out[k] = [va, vb]
    return out


def check_config_vs_run(rec, name, trainer_path, env, run_config_path, allowed_extra_keys):
    """planned config.json under `env` versus a real run's config.json. Differences are allowed
    only for `device` and for camera-ready keys that the run's trainer did not have."""
    if not run_config_path or not os.path.exists(run_config_path):
        rec.skip(name, "reference run config.json not available (set FF_REFERENCE_RUNS)")
        return
    cfg, _ = planned_config(trainer_path, env)
    cfg["device"] = "cuda"
    with open(run_config_path) as f:
        ref = json.load(f)
    d = diff_configs(ref, cfg)
    bad = {k: v for k, v in d.items() if not (k in allowed_extra_keys and v[0] == "<MISSING>")}
    rec.record(name, not bad, {"diff_vs_run_config": d, "unexpected": bad,
                               "run": os.path.basename(os.path.dirname(run_config_path))})


def check_refuses(rec, name, trainer_path, env, exc=ValueError):
    try:
        planned_config(trainer_path, env)
        rec.record(name, False, "no error raised")
    except exc as e:
        rec.record(name, True, str(e)[:120])


# ------------------------------------------------------------------ RNG save / restore
def draw_all():
    return ([random.random() for _ in range(3)], np.random.rand(3).tolist(), np.random.randn(3).tolist(),
            torch.rand(3).tolist(), torch.randint(0, 10, (5,)).tolist())


def rng_snapshot():
    return (torch.get_rng_state().clone(), np.random.get_state()[1].copy(),
            np.random.get_state()[2:], random.getstate())


def rng_equal(s1, s2):
    return (torch.equal(s1[0], s2[0]) and np.array_equal(s1[1], s2[1])
            and s1[2] == s2[2] and s1[3] == s2[3])


def _head_for(mod, cfg):
    head = mod.AttentiveHybridHead(
        in_dim_g=cfg.num_blocks * cfg.n_aspects, in_dim_a=cfg.d_model * cfg.num_blocks,
        hidden=cfg.head_hidden, num_classes=cfg.num_classes, num_heads=cfg.head_num_heads,
        dropout=cfg.head_dropout)
    opt_head = torch.optim.AdamW(head.parameters(), lr=cfg.stage2_lr, weight_decay=cfg.stage2_wd)
    sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=cfg.epochs_stage2)
    return head, opt_head, sched_head


def _step_state(mod, net, ema, opts, queues, cfg, batch):
    stats = mod.train_epoch_stage1(net, ema, opts, [batch], cfg, queues, epoch=2)
    return {"stats": stats, "net": clone_state(net), "ema": clone_state(ema),
            "opt": [copy.deepcopy(o.base_optimizer.state_dict()) for o in opts],
            "queues": [(q.z.clone(), q.y.clone(), int(q.ptr), bool(q.full)) for q in queues],
            "rng_torch": torch.get_rng_state().clone(), "rng_numpy": np.random.get_state()[1].copy(),
            "rng_python": random.getstate()}


def run_rng_tests(rec, prefix, new, v1, cfg, batch_a, batch_b):
    """RNG save/restore of the camera-ready trainer `new`; `v1` is the published trainer."""
    tmp = tempfile.mkdtemp(prefix="ff_rng_", dir=_SCRATCH)
    net, ema, opts, scheds, queues = build(new, cfg, seed=11)
    seed_all(5)
    s0 = rng_snapshot()
    new._capture_rng_state()
    rec.record(f"{prefix}_capture_draws_nothing", rng_equal(s0, rng_snapshot()))

    # Stage-1 resume: after one real step, save; an uninterrupted run continues with batch B;
    # a resumed run (fresh objects from another seed, RNG re-seeded, then the loader) must
    # reproduce that continuation bitwise.
    seed_all(21)
    new.train_epoch_stage1(net, ema, opts, [batch_a], cfg, queues, epoch=1)
    p1 = os.path.join(tmp, "stage1_last.pt")
    new._save_checkpoint(stage=1, cfg=cfg, epoch_stage1=1, best_s1=0.5, net=net, net_ema=ema,
                         opts=opts, scheds=scheds, queues=queues, path=p1)
    ref_state = rng_snapshot()
    uninterrupted = _step_state(new, net, ema, opts, queues, cfg, batch_b)
    del net, ema, opts, scheds, queues
    ck = torch.load(p1, map_location="cpu", weights_only=True)
    rec.record(f"{prefix}_ckpt_has_rng_state_weights_only_load", "rng_state" in ck)
    del ck
    net2, ema2, opts2, scheds2, queues2 = build(new, cfg, seed=99)
    seed_all(cfg.seed)
    _, start, _ = new._load_checkpoint_for_stage1(cfg, net2, ema2, opts2, scheds2, queues2, path=p1)
    rec.record(f"{prefix}_stage1_resume_start_epoch_is_2", start == 2, start)
    rec.record(f"{prefix}_stage1_resume_rng_equals_saved", rng_equal(ref_state, rng_snapshot()))
    resumed = _step_state(new, net2, ema2, opts2, queues2, cfg, batch_b)
    d = compare(uninterrupted, resumed, keys=("stats", "net", "ema", "opt", "queues",
                                              "rng_torch", "rng_numpy", "rng_python"))
    rec.record(f"{prefix}_stage1_resumed_step_bitwise_equals_uninterrupted", bitwise(d), d)
    del net2, ema2, opts2, scheds2, queues2
    # the v1 loader ignores rng_state, so the same resume is NOT a continuation
    net3, ema3, opts3, scheds3, queues3 = build(v1, cfg, seed=99)
    seed_all(cfg.seed)
    v1._load_checkpoint_for_stage1(cfg, net3, ema3, opts3, scheds3, queues3, path=p1)
    v1_resumed = _step_state(v1, net3, ema3, opts3, queues3, cfg, batch_b)
    d1 = compare(uninterrupted, v1_resumed, keys=("net", "rng_torch"))
    rec.record(f"{prefix}_v1_loader_resume_differs_from_uninterrupted", not bitwise(d1), d1)
    del net3, ema3, opts3, scheds3, queues3, v1_resumed

    # Stage-2 checkpoint: the Stage-1 loader defers; the Stage-2 loader restores the RNG after
    # the head (whose init draws random numbers) has been built.
    net, ema, opts, scheds, queues = build(new, cfg, seed=12)
    head, opt_head, sched_head = _head_for(new, cfg)
    seed_all(33)
    draw_all()
    p2 = os.path.join(tmp, "stage2_last.pt")
    new._save_checkpoint(stage=2, cfg=cfg, epoch_stage1=cfg.epochs_stage1, best_s1=0.5, net=net,
                         net_ema=ema, opts=opts, scheds=scheds, queues=queues, epoch_stage2=2,
                         best_s2=0.6, head=head, opt_head=opt_head, sched_head=sched_head, path=p2)
    ref2 = draw_all()
    del net, ema, opts, scheds, queues, head, opt_head, sched_head
    net, ema, opts, scheds, queues = build(new, cfg, seed=77)
    seed_all(cfg.seed)
    before = rng_snapshot()
    ck2, _, _ = new._load_checkpoint_for_stage1(cfg, net, ema, opts, scheds, queues, path=p2)
    rec.record(f"{prefix}_stage1_loader_defers_for_stage2_ckpt", rng_equal(before, rng_snapshot()))
    head, opt_head, sched_head = _head_for(new, cfg)
    start2, _ = new._load_checkpoint_for_stage2(cfg, ck2, head, opt_head, sched_head)
    rec.record(f"{prefix}_stage2_resume_start_epoch_is_3", start2 == 3, start2)
    rec.record(f"{prefix}_stage2_resume_restores_rng", draw_all() == ref2)
    del net, ema, opts, scheds, queues, head, opt_head, sched_head, ck2

    # a v1 checkpoint (no rng_state) still loads and the RNG is left untouched
    net, ema, opts, scheds, queues = build(new, cfg, seed=13)
    p3 = os.path.join(tmp, "v1_stage1_last.pt")
    v1._save_checkpoint(stage=1, cfg=cfg, epoch_stage1=5, best_s1=0.4, net=net, net_ema=ema,
                        opts=opts, scheds=scheds, queues=queues, path=p3)
    seed_all(44)
    before = rng_snapshot()
    _, start3, _ = new._load_checkpoint_for_stage1(cfg, net, ema, opts, scheds, queues, path=p3)
    rec.record(f"{prefix}_v1_ckpt_loads_start_epoch_is_6", start3 == 6, start3)
    rec.record(f"{prefix}_v1_ckpt_rng_untouched", rng_equal(before, rng_snapshot()))
    rec.record(f"{prefix}_v1_ckpt_reported_not_restored",
               new.LAST_RNG_RESTORE.get("restored") is False)
    shutil.rmtree(tmp, ignore_errors=True)


def cleanup():
    shutil.rmtree(_SCRATCH, ignore_errors=True)


# ------------------------------------------------------------------ shared suite (RNG + P3 only trainers)
def make_cfg(mod, **over):
    cfg = mod.FFConfig()
    cfg.__post_init__()
    cfg.device = "cpu"
    cfg.workers = 0
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def run_simple_suite(rec, name, configs, p3_configs, rng_config, batch_size=16):
    """A: flags off == v1 (one step per config); B: three steps == v1 (depth-order locality);
    R: RNG save/restore. For trainers whose only camera-ready changes are RNG + P3."""
    new_path = os.path.join(TRAINERS, name)
    rec.out["trainer_sha256"] = sha256(new_path)
    new = cr_trainer(name)
    b1, b2, b3 = (make_batch(make_cfg(new), batch_size, seed=s) for s in (2026, 77, 5))
    v1p = rec.ref("v1_trainer", v1_path(name), V1_SHA[name])
    v1 = load_trainer(v1p, "v1_" + name[:-3]) if v1p else None
    for cname, over in configs.items():
        if v1 is None:
            rec.skip(f"A_flags_off_equals_v1_{cname}", "FF_V1_TRAINERS not set")
            continue
        check_pair(rec, f"A_flags_off_equals_v1_{cname}", v1, make_cfg(v1, **over), new,
                   make_cfg(new, **over), [b1])
    for cname in p3_configs:
        if v1 is None:
            rec.skip(f"B_p3_three_steps_equals_v1_{cname}", "FF_V1_TRAINERS not set")
            continue
        over = configs[cname]
        check_pair(rec, f"B_p3_three_steps_equals_v1_{cname}", v1, make_cfg(v1, **over), new,
                   make_cfg(new, **over), [b1, b2, b3])
    if v1 is not None:
        run_rng_tests(rec, "R", new, v1, make_cfg(new, **configs[rng_config]), b1, b2)
    else:
        rec.skip("R_rng_tests", "FF_V1_TRAINERS not set")
    return new, v1, (b1, b2, b3)
