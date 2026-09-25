#!/usr/bin/env python3
"""Definitive identity test for the archived CIFAR-100 trio checkpoints.

For each extracted checkpoint: rebuild the net from its own cfg, run the Stage-1
goodness-sum inference over all 100 label hypotheses on the full 10k test set, and
compare the top-1 against (a) the run dir's events.jsonl final_test_eval and
(b) the s1_sum_no_tta recorded by the S2-FF experiment.

A match to 0.01 pp proves the archived file IS the checkpoint those experiments used.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import importlib.util, json, sys, glob, os
import torch

MG = FF_HOME + '/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/mgc_rebuttal'
RUNS = FF_HOME + '/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/revised_training/extracted/ff_cifar100_multiseed'
SCR = os.path.dirname(os.path.abspath(__file__))
RUN_DIRS = {'gamma0': '20260420_101044', 'gated_k0': '20260422_185437', 'cumulative': '20260418_032149'}
# what the S2-FF / S2-FF-CE experiments recorded as the Stage-1 gate
S2FF_GATE = {'gamma0': 66.18, 'gated_k0': 66.07, 'cumulative': 66.70}

spec = importlib.util.spec_from_file_location("trainer_mod", MG + "/mgc_c100_trainer.py")
T = importlib.util.module_from_spec(spec); sys.modules["trainer_mod"] = T; spec.loader.exec_module(T)
dev = "cuda" if torch.cuda.is_available() else "cpu"

for variant in ('gamma0', 'gated_k0', 'cumulative'):
    ckpt_path = f'{SCR}/ckpt_verify/{variant}_seed42_stage1_best.pt'
    run = glob.glob(f'{RUNS}/runs_{variant}_seed42/*{RUN_DIRS[variant]}*')[0]
    cfgd = json.load(open(run + '/config.json'))

    cfg = T.FFConfig()
    for k, v in cfgd.items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, v)
            except Exception: pass
    net = T.FFHybridNet(cfg).to(dev)
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
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
    ev = None
    for line in open(run + '/events.jsonl'):
        e = json.loads(line)
        if 'stage1_test_top1_no_tta' in e: ev = e
    logged = ev['stage1_test_top1_no_tta'] * 100
    print(f"[{variant}] recomputed S1 no-TTA = {acc:.2f} | run log = {logged:.2f} "
          f"(Δ {abs(acc-logged):.3f} pp) | S2-FF gate recorded = {S2FF_GATE[variant]:.2f} "
          f"(Δ {abs(acc-S2FF_GATE[variant]):.3f} pp) | missing_keys={len(miss.missing_keys)}",
          flush=True)
print("DONE")
