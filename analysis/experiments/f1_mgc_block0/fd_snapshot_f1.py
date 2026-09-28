#!/usr/bin/env python3
"""F1 follow-up: free-riding index F_d (gradient attenuation) on F1 arm-b and arm-a seed-42 checkpoints.

Adapts experiments/e3_fd/snapshot_grad_attenuation_ema.py (the corrected E3 pipeline) to the F1 trainer
fork (experiments/f1_mgc_block0/trainer.py). Here the network is built, by default, from
trainers/cp_fair_hardness_gated.py, which reproduces the fork's training step bitwise under the F1 flags
(tests/README.md); the arm-b seed-42 margins, P_prev and R arrays computed with either file are bitwise
identical. Everything that defines the measurement is kept from E3:

  * weights: ckpt['net_ema'] of the run's stage1_best checkpoint, strict load (0 missing / 0 unexpected);
  * normalization: the trainer's CIFAR-10 mean/std (0.4914, 0.4822, 0.4465) / (0.2023, 0.1994, 0.2010);
  * examples: the first 4 x 256 = 1024 CIFAR-10 test images (shuffle=False);
  * wrong labels: the published snapshot's wrong labels recovered in E3
    (analysis/experiments/e3_fd/out/recovered_published_y_neg.npy), so the images and labels are the same as the
    E3 gamma=0.7 SAM-off reference; robustness: the CPU torch.manual_seed(42) draw;
  * margins: current-block aggregated goodness, true label minus wrong label, per block;
    P_prev = sum of the earlier blocks' margins; M = m + gamma * P_prev; beta = symba_alpha = 4;
  * statistics: analysis/experiments/e3_fd/fd_stats.py::stats (F_d = mean(1 - min(1, R)),
    R = (1 + e^{beta m}) / (1 + e^{beta (m + gamma P)}), published CI convention, coverage, median R);
  * EMA accuracy gate: full 10k test, FF goodness-sum rule, no TTA, vs the run's logged
    stage1_test_top1_no_tta and its preds_stage1_test_no_tta.npz. The 'passed' flag written here requires
    |acc - logged| <= 2 examples AND <= 2 disagreeing examples; this is stricter than E3's registered gate
    (per-example agreement >= 99.9%). analysis/camera_ready/f1_analysis.py gates on the E3 criteria and
    reports both.

F_d measures the attenuation of the gradient that the cumulative SYMBA term sends to block d's own margin.
It says nothing about the unattenuated current-block auxiliary that arm a (and the published cumulative
baseline) also carries; arm b has no such auxiliary.

Inputs: the stage1_best checkpoints of arm b seed 42 (a DGX Spark GB10 run) and arm a seed 42 (a Colab
A100 run; the local copy was MD5-checked against the Drive file). Neither is part of the checkpoint release
(RELEASE_CHECKPOINTS.md).

[CR] supplement port: paths come from the environment (analysis/README.md); nothing is hard-coded.
  FF_HOME            root of the run archive (read-only); the run directories are read from
                     $FF_F1_DIR/runs (default $FF_HOME/amir-porjects/neuroips_review/camera_ready/experiments/
                     f1_mgc_block0/runs)
  FF_F1B_S42_CKPT    arm-b seed-42 checkpoint (default <FF_WORK_DIR>/ckpts/f1_mgc_block0/
                     ff_cifar10_f1b_cum_noaux_L4_D128_P2_seed42_stage1_best.pt); FF_F1A_S42_CKPT likewise for arm a
  FF_F1_TRAINER      trainer file that defines the network (default trainers/cp_fair_hardness_gated.py)
  FF_E3_OUT_DIR      E3 outputs holding recovered_published_y_neg.npy (default analysis/experiments/e3_fd/out)
  FF_DATA_DIR        CIFAR-10 root (default $FF_HOME/.../revised_training/data_c10, as in E3)
  FF_WORK_DIR        scratch (the trainer creates its checkpoint directory at import time)
  FF_OUT_DIR         outputs: <FF_OUT_DIR>/experiments/f1_mgc_block0/out/fd/{fd_<tag>_<labels>.json, .npz},
                     read by analysis/camera_ready/f1_analysis.py and compute_budget.py
Run (CPU only, <= 8 threads):
  FF_HOME=... CUDA_VISIBLE_DEVICES='' python analysis/experiments/f1_mgc_block0/fd_snapshot_f1.py
Env: FF_ONLY=b42|a42 (default both), FF_NEG_LABELS=published|seed|both (default both), FF_ACC_GATE=1|0.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SUPP = HERE.parents[2]  # repository root
# [CR] supplement port: paths come from the environment (see the docstring and analysis/README.md).
FF_HOME = os.environ.get("FF_HOME", "")
FF_WORK = os.environ.get("FF_WORK_DIR", "ff_work")
FF_OUT = os.environ.get("FF_OUT_DIR", "ff_out")
CF = f"{FF_HOME}/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
F1_DIR = Path(os.environ.get("FF_F1_DIR", f"{FF_HOME}/amir-porjects/neuroips_review/camera_ready/experiments/f1_mgc_block0"))
SCRATCH_CKPT = os.environ.get("FF_SCRATCH_CKPT_DIR", FF_WORK + "/f1_trainer_ckpt_dir_unused")
os.environ["FF_CKPT_DIR"] = SCRATCH_CKPT  # the trainer makedirs(CKPT_DIR) at import time

import numpy as np  # noqa: E402
import torch  # noqa: E402

N_THREADS = min(8, int(os.environ.get("FF_NUM_THREADS", "8")))
torch.set_num_threads(N_THREADS)

TRAINER_PATH = Path(os.environ.get("FF_F1_TRAINER", str(SUPP / "trainers" / "cp_fair_hardness_gated.py"))).resolve()
_spec = importlib.util.spec_from_file_location("trainer", TRAINER_PATH)
trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer)  # the F1 fork in the camera-ready run; cp_fair_hardness_gated.py by default

sys.path.insert(0, str(HERE.parent / "e3_fd"))
from fd_stats import stats as fd_stats  # noqa: E402  (identical F_d statistics to E3)

FFConfig = trainer.FFConfig
FFHybridNet = trainer.FFHybridNet
l2_normalize_tokens = trainer.l2_normalize_tokens

RUNS = F1_DIR / "runs"
CKPTS = Path(FF_WORK) / "ckpts" / "f1_mgc_block0"
B42_CKPT = os.environ.get("FF_F1B_S42_CKPT", str(CKPTS / "ff_cifar10_f1b_cum_noaux_L4_D128_P2_seed42_stage1_best.pt"))
A42_CKPT = os.environ.get("FF_F1A_S42_CKPT", str(CKPTS / "ff_cifar10_f1a_cum_sam_L4_D128_P2_seed42_stage1_best.pt"))
VARIANTS = {
    "b42": {"arm": "b", "tag": "f1b_cum_noaux", "seed": 42,
            "ckpt": B42_CKPT,
            "ckpt_origin": "local DGX GB10 run (CR/experiments/ckpts/f1_mgc_block0)",
            "run_dir": RUNS / "ff_cifar10_f1b_cum_noaux_L4_D128_bs512_seed42_20260925_101141"},
    "a42": {"arm": "a", "tag": "f1a_cum_sam", "seed": 42,
            "ckpt": A42_CKPT,
            "ckpt_origin": "Colab A100 run; local copy of gdrive:ffcr/f1_mgc_block0/checkpoints/"
                           "ff_cifar10_f1a_cum_sam_L4_D128_P2_seed42_stage1_best.pt (MD5 matched Drive)",
            "run_dir": RUNS / "ff_cifar10_f1a_cum_sam_L4_D128_bs512_seed42_20260926_224359"},
}
REC_LABELS = Path(os.environ.get("FF_E3_OUT_DIR", str(HERE.parent / "e3_fd" / "out"))) / "recovered_published_y_neg.npy"
DATA_DIR = str(Path(os.environ.get("FF_DATA_DIR", f"{CF}/revised_training/data_c10")).resolve())
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
DEVICE = "cpu"
N_BATCHES, BATCH = 4, 256
ACC_TOL = 0.0002
OUT = Path(FF_OUT) / "experiments" / "f1_mgc_block0" / "out" / "fd"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


TRAINER_SHA = sha256(TRAINER_PATH)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_cfg_from_json(path: Path):
    """As in E3: default FFConfig, then every key of the run's config.json."""
    c = json.load(open(path))
    cfg = FFConfig()
    cfg.__post_init__()
    for k, v in c.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.aspect_loss_types = list(cfg.aspect_loss_types[: cfg.n_aspects])
    cfg.aspect_lambdas = list(cfg.aspect_lambdas[: cfg.n_aspects])
    cfg.aspect_theta_inits = list(cfg.aspect_theta_inits[: cfg.n_aspects])
    return cfg


def build_eval_loader(batch_size: int):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    ds = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)


@torch.no_grad()
def per_block_margins(net, images, y_pos, y_neg):
    """Unchanged from E3 / the published script."""
    net.eval()
    tokens0 = net.embed(images)
    out = []
    for y in (y_pos, y_neg):
        tokens, g_hist, g_curr = tokens0, [], []
        for blk in net.blocks:
            g_prev = torch.stack(g_hist, dim=1).detach() if g_hist else None
            a, g_vec, _, _, _ = blk(tokens, y, g_prev)
            g_curr.append(blk.aggregate_goodness(g_vec))
            g_hist.append(g_vec)
            tokens = l2_normalize_tokens(a)
        out.append(torch.stack(g_curr, dim=0))
    return out[0], out[1]


def random_wrong(y, num_classes):
    r = torch.randint(0, num_classes, y.shape, device=y.device)
    return (r + (r == y).long()) % num_classes


@torch.no_grad()
def accuracy_gate(net, run_dir: Path):
    loader = build_eval_loader(256)
    preds, labels = [], []
    t0 = time.time()
    for images, y in loader:
        scores = net.predict_scores(images.to(DEVICE))
        preds.append(scores.topk(1, dim=1).indices[:, 0].cpu())
        labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    acc = float((preds == labels).mean())
    ev = [json.loads(l) for l in open(run_dir / "events.jsonl") if l.strip()]
    logged = [e for e in ev if e.get("event") == "final_test_eval"][0]["stage1_test_top1_no_tta"]
    z = np.load(run_dir / "preds_stage1_test_no_tta.npz")
    assert np.array_equal(z["y_true"], labels)
    n_dis = int((z["y_pred"] != preds).sum())
    out = {"acc_s1_no_tta_cpu": acc, "n": int(len(labels)), "logged_stage1_test_top1_no_tta": logged,
           "abs_diff_vs_logged": abs(acc - logged), "n_disagree_with_logged_preds_npz": n_dis,
           "tolerance": ACC_TOL, "passed": bool(abs(acc - logged) <= ACC_TOL and n_dis <= 2),
           "seconds": round(time.time() - t0, 1)}
    print(f"  ACC GATE: cpu={acc:.4f} logged={logged:.4f} disagree_npz={n_dis} passed={out['passed']} "
          f"({out['seconds']:.0f}s)", flush=True)
    return out


def run(key: str, label_modes, acc_gate: bool):
    v = VARIANTS[key]
    print(f"\n=== {key}: {v['tag']} seed {v['seed']} ===", flush=True)
    cfg = build_cfg_from_json(v["run_dir"] / "config.json")
    cfg.device = DEVICE
    net = FFHybridNet(cfg).to(DEVICE)
    ckpt = torch.load(v["ckpt"], map_location=DEVICE, weights_only=False)
    sd = ckpt["net_ema"]
    res = net.load_state_dict(sd, strict=True)
    ccfg = ckpt.get("cfg") or {}
    meta = {"ckpt": v["ckpt"], "ckpt_origin": v["ckpt_origin"], "ckpt_md5": md5(v["ckpt"]),
            "state_dict_key": "net_ema", "n_keys": len(sd),
            "missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys),
            "ckpt_stage": ckpt.get("stage"), "ckpt_epoch_stage1": ckpt.get("epoch_stage1"),
            "ckpt_best_s1_val": ckpt.get("best_s1"),
            "ckpt_cfg": {k: ccfg.get(k) for k in ("use_sam", "gamma_scale", "gamma_gating_mode", "use_mgc",
                                                  "curr_aux_off", "mgc_keep_l0_aux", "block_curr_lambda",
                                                  "block_curr_depth_slope", "symba_alpha", "seed")},
            "run_dir": f"experiments/f1_mgc_block0/runs/{v['run_dir'].name}"}
    del ckpt
    gamma = float(cfg.gamma_scale)
    alpha = float(cfg.symba_alpha)
    assert cfg.gamma_gating_mode == "constant" and abs(gamma - 0.7) < 1e-12 and alpha == 4.0
    gate = accuracy_gate(net, v["run_dir"]) if acc_gate else None
    rec_labels = np.load(REC_LABELS)
    results = {}
    for lm in label_modes:
        t0 = time.time()
        loader = build_eval_loader(BATCH)
        all_m = []
        torch.manual_seed(42)
        for bi, (images, y) in enumerate(loader):
            if bi >= N_BATCHES:
                break
            y_neg = random_wrong(y, cfg.num_classes)
            if lm == "published":
                y_neg = torch.from_numpy(rec_labels[bi * BATCH:(bi + 1) * BATCH])
                assert y_neg.shape == y.shape and bool((y_neg != y).all())
            gp, gn = per_block_margins(net, images, y, y_neg)
            all_m.append((gp - gn).cpu())
        m = torch.cat(all_m, dim=1).numpy().astype(np.float64)
        P = np.zeros_like(m)
        for d in range(1, m.shape[0]):
            P[d] = m[:d].sum(axis=0)
        s, R = fd_stats(m, P, gamma)
        grad = alpha / (1 + np.exp(alpha * (m + gamma * P)))       # |d softplus(-alpha M)/dm|
        grad_local = alpha / (1 + np.exp(alpha * m))
        s["grad_mean"] = grad.mean(axis=1).tolist()
        s["grad_local_mean"] = grad_local.mean(axis=1).tolist()
        s["seconds"] = round(time.time() - t0, 1)
        rec = {"variant": key, "arm": v["arm"], "tag": v["tag"], "seed": v["seed"], "gamma": gamma,
               "beta": alpha, "neg_labels": lm, "stats": s, "ckpt_meta": meta, "acc_gate": gate,
               "setup": {"weights": "net_ema strict", "norm_mean": MEAN, "norm_std": STD,
                         "examples": f"first {N_BATCHES * BATCH} CIFAR-10 test images (shuffle=False)",
                         "data_dir": DATA_DIR, "device": DEVICE, "torch": torch.__version__,
                         "threads": N_THREADS, "trainer": f"{TRAINER_PATH.name} (sha256 {TRAINER_SHA})"}}
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / f"fd_{key}_{lm}.json", "w") as f:
            json.dump(rec, f, indent=1)
        np.savez_compressed(OUT / f"fd_{key}_{lm}.npz", m=m, P_prev=P, R=R, gamma=np.array(gamma))
        print(f"  [{lm}] F_d = {[round(x, 4) for x in s['F_d']]}  cov% = {[round(x, 1) for x in s['assumption_coverage_pct']]}"
              f"  m_mean = {[round(x, 3) for x in s['m_mean']]}  P_mean = {[round(x, 3) for x in s['P_prev_mean']]}"
              f"  median_R = {[f'{x:.2e}' for x in s['median_R_d']]}", flush=True)
        results[lm] = rec
    return results


def main():
    only = os.environ.get("FF_ONLY")
    keys = [only] if only else ["b42", "a42"]
    lab = os.environ.get("FF_NEG_LABELS", "both")
    modes = ["published", "seed"] if lab == "both" else [lab]
    gate = os.environ.get("FF_ACC_GATE", "1") == "1"
    for k in keys:
        run(k, modes, gate)


if __name__ == "__main__":
    main()
