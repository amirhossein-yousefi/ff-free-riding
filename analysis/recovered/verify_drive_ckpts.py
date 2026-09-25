#!/usr/bin/env python3
"""Content verification of the Drive-exported rebuttal checkpoints (MGC + hinge).

Same protocol as the trio verification: rebuild the net from the matching run's config,
run Stage-1 goodness-sum inference over ALL label hypotheses on the full test set, and
compare against the run's events.jsonl. Match within 0.02 pp proves identity.
Cheap CIFAR-10 jobs run first; the three CIFAR-100 jobs (100 hypotheses) run last.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import importlib.util, json, sys, glob, os
import torch

MG = FF_HOME + '/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/mgc_rebuttal'
CK = MG + '/checkpoints_from_drive'
RFC = MG + '/from_colab/runs_from_colab'

# (checkpoint file, trainer, run-dir glob)  — gates come from each run's own events.jsonl
JOBS = [
    ('ff_cifar10_mgc_c1_L4_D128_P2_seed123_stage1_best.pt',   'mgc_trainer.py',      f'{RFC}/ff_cifar10_mgc_c1_*seed123_*'),
    ('ff_cifar10_mgc_c1_L4_D128_P2_seed456_stage1_best.pt',   'mgc_trainer.py',      f'{RFC}/ff_cifar10_mgc_c1_*seed456_*'),
    ('ff_cifar10_hinge_g07_L4_D128_P2_seed123_stage1_best.pt','hinge_trainer.py',    f'{RFC}/ff_cifar10_hinge_g07_*seed123_*'),
    ('ff_cifar10_hinge_g07_L4_D128_P2_seed456_stage1_best.pt','hinge_trainer.py',    f'{RFC}/ff_cifar10_hinge_g07_*seed456_*'),
    ('ff_cifar10_hinge_g0_L4_D128_P2_seed123_stage1_best.pt', 'hinge_trainer.py',    f'{RFC}/ff_cifar10_hinge_g0_*seed123_*'),
    ('ff_cifar10_hinge_g0_L4_D128_P2_seed456_stage1_best.pt', 'hinge_trainer.py',    f'{RFC}/ff_cifar10_hinge_g0_*seed456_*'),
    ('ff_cifar100_mgc_c100_L4_D256_P2_seed42_stage1_best.pt', 'mgc_c100_trainer.py', f'{RFC}/ff_cifar100_mgc_c100_*seed42_*'),
    ('ff_cifar100_mgc_c100_L4_D256_P2_seed123_stage1_best.pt','mgc_c100_trainer.py', f'{RFC}/ff_cifar100_mgc_c100_*seed123_*'),
    ('ff_cifar100_mgc_c100_L4_D256_P2_seed456_stage1_best.pt','mgc_c100_trainer.py', f'{RFC}/ff_cifar100_mgc_c100_*seed456_*'),
]

dev = "cuda" if torch.cuda.is_available() else "cpu"
_loaded = {}

def load_trainer(name):
    if name not in _loaded:
        spec = importlib.util.spec_from_file_location("trainer_" + name.replace('.', '_'), f'{MG}/{name}')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _loaded[name] = mod
    return _loaded[name]

n_pass = 0
for ck_name, trainer, run_glob in JOBS:
    T = load_trainer(trainer)
    run = sorted(glob.glob(run_glob))[-1]
    cfgd = json.load(open(run + '/config.json'))
    ev = None
    for line in open(run + '/events.jsonl'):
        e = json.loads(line)
        if 'stage1_test_top1_no_tta' in e: ev = e
    gate = ev['stage1_test_top1_no_tta'] * 100

    cfg = T.FFConfig()
    for k, v in cfgd.items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, v)
            except Exception: pass
    net = T.FFHybridNet(cfg).to(dev)
    ck = torch.load(f'{CK}/{ck_name}', map_location=dev, weights_only=False)
    state = ck.get("net_ema") or ck.get("model")
    miss = net.load_state_dict(state, strict=False)
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)

    _, _, test, _ = T.make_loaders_stage1(batch_size=256, test_batch_size=256,
                                          seed=cfg.seed, val_fraction=0.1, workers=2)
    C = cfg.num_classes

    @torch.no_grad()
    def s1_acc():
        correct = tot = 0
        for batch in test:
            images = (batch[0] if not isinstance(batch[0], (list, tuple)) else batch[0][0]).to(dev)
            y = batch[-1].to(dev); B = images.size(0)
            scores = torch.zeros(B, C, device=dev)
            for k in range(C):
                yk = torch.full((B,), k, dtype=torch.long, device=dev)
                tokens = net.embed(images); g_hist = []; total = torch.zeros(B, device=dev)
                for blk in net.blocks:
                    g_prev = torch.stack(g_hist, dim=1).detach() if g_hist else None
                    a, g_vec, _, _, _ = blk(tokens, yk, g_prev)
                    total = total + blk.aggregate_goodness(g_vec)
                    g_hist.append(g_vec); tokens = T.l2_normalize_tokens(a)
                scores[:, k] = total
            correct += (scores.argmax(1) == y).sum().item(); tot += B
        return 100.0 * correct / tot

    acc = s1_acc()
    ok = abs(acc - gate) < 0.02 and len(miss.missing_keys) == 0
    n_pass += ok
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {ck_name}: recomputed {acc:.2f} vs gate {gate:.2f} "
          f"(Δ {abs(acc-gate):.3f} pp, missing_keys={len(miss.missing_keys)})", flush=True)
    del net; torch.cuda.empty_cache()

print(f"DONE {n_pass}/{len(JOBS)} passed", flush=True)
