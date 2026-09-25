#!/usr/bin/env python3
"""E3: gradient-attenuation snapshot on EMA weights with the trainer's normalization.

Copy of NeurIPS_FF_Net_Last_submitted/supplement_code/analysis/snapshot_grad_attenuation.py
(kept verbatim as orig_supplement_snapshot_grad_attenuation.py). The only intended changes are:

  1. Normalize std = (0.2023, 0.1994, 0.2010), the trainer's CIFAR-10 std (make_loaders*).
     The published script used (0.2470, 0.2435, 0.2616).                       [FF_NORM_STD=fixed]
  2. Load ckpt['net_ema'] (the weights that produced every reported accuracy) with strict=True.
     The published script fell through its key probes to ckpt['model'] (raw training weights)
     with strict=False.                                                        [FF_WEIGHTS=ema]
  3. EMA accuracy gate: full 10k CIFAR-10 test, FF goodness-sum rule, no TTA (the trainer's
     evaluate_ff top-1 protocol), compared with the run's logged stage1_test_top1_no_tta and with
     the per-example preds_stage1_test_no_tta.npz.                             [FF_ACC_GATE=1]

Everything else is unchanged: the three seed-42 L4/D128 stage1_best checkpoints, the first
N_BATCHES x BATCH = 1024 test images (shuffle=False), torch.manual_seed(42) random wrong labels,
current-block goodness margins, M = m + gamma * P_prev, |grad| = alpha * sigmoid(-alpha * M).

Housekeeping changes that do not touch the numbers: the device is forced to CPU; download=False;
the trainer import never writes bytecode into the read-only source tree; the trainer's
module-level CKPT_DIR makedirs is pointed at a scratch dir; outputs go to ./out/<weights>_<std>/.

The legacy combination (FF_WEIGHTS=raw FF_NORM_STD=legacy) is the reproduction gate against the
published grad_attenuation_arrays.npz.

Run (CPU only):
  CUDA_VISIBLE_DEVICES='' python snapshot_grad_attenuation_ema.py
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only trainer tree

HOME = FF_HOME
CF = f"{HOME}/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"
GA = f"{CF}/basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation"
SCRATCH_CKPT = os.environ.get(
    "FF_SCRATCH_CKPT_DIR",
    FF_WORK + "/trainer_ckpt_dir_unused")
os.environ["FF_CKPT_DIR"] = SCRATCH_CKPT  # trainer does os.makedirs(CKPT_DIR) at import time

import torch  # noqa: E402
import numpy as np  # noqa: E402

torch.set_num_threads(int(os.environ.get("FF_NUM_THREADS", "6")))

# ---- locate trainer module ----
TRAINER_DIR = Path(os.environ.get("FF_TRAINER_DIR", f"{CF}/NeurIPS_FF_Net_Last_submitted/supplement_code/trainers")).resolve()
TRAINER_MODULE = os.environ.get("FF_TRAINER_MODULE", "cp_fair_hardness_gated")
sys.path.insert(0, str(TRAINER_DIR))
try:  # the trainer imports tqdm for training progress bars only; the CR venv does not ship it
    import tqdm.auto  # noqa: F401
except ImportError:
    import importlib.machinery
    import types
    import torchvision  # noqa: F401  (import before the stub: torch._dynamo probes tqdm's __spec__)
    _tq = types.ModuleType("tqdm")
    _tqa = types.ModuleType("tqdm.auto")
    _tq.__spec__ = importlib.machinery.ModuleSpec("tqdm", None)
    _tqa.__spec__ = importlib.machinery.ModuleSpec("tqdm.auto", None)

    class _NoTqdm:  # minimal stand-in, never used on the analysis path
        def __init__(self, it=None, *a, **k):
            self.it = it

        def __iter__(self):
            return iter(self.it)

        def __getattr__(self, name):
            return lambda *a, **k: None

    _tqa.tqdm = _NoTqdm
    _tq.tqdm = _NoTqdm
    _tq.auto = _tqa
    sys.modules["tqdm"] = _tq
    sys.modules["tqdm.auto"] = _tqa
trainer = __import__(TRAINER_MODULE)

FFConfig = trainer.FFConfig
FFHybridNet = trainer.FFHybridNet
l2_normalize_tokens = trainer.l2_normalize_tokens

# ---- locate checkpoints + runs ----
CKPT_DIR = Path(os.environ.get("FF_CHECKPOINTS_DIR", f"{GA}/checkpoints")).resolve()
RUNS_DIR = Path(os.environ.get("FF_RUNS_DIR", f"{GA}/runs")).resolve()

WEIGHTS = os.environ.get("FF_WEIGHTS", "ema")          # ema | raw
NORM_STD = os.environ.get("FF_NORM_STD", "fixed")      # fixed | legacy
ACC_GATE = os.environ.get("FF_ACC_GATE", "1") == "1"
ONLY = os.environ.get("FF_ONLY_VARIANT")               # optional: run a single variant index
NEG_LABELS = os.environ.get("FF_NEG_LABELS", "published")  # published | seed
assert WEIGHTS in ("ema", "raw") and NORM_STD in ("fixed", "legacy") and NEG_LABELS in ("published", "seed")
# "seed": the published code path (torch.manual_seed(42) + torch.randint on this device). On CPU this
#         draws different labels than the published CUDA run did.
# "published": the exact wrong labels of the published snapshot, recovered by
#         recover_published_neg_labels.py (out/recovered_published_y_neg.npy). The RNG calls still run,
#         so nothing else changes.
REC_LABELS = Path(__file__).resolve().parent / "out" / "recovered_published_y_neg.npy"

STD = {"fixed": (0.2023, 0.1994, 0.2010), "legacy": (0.2470, 0.2435, 0.2616)}[NORM_STD]
MEAN = (0.4914, 0.4822, 0.4465)


def _find_first(dirpath: Path, pattern: str) -> Path:
    matches = sorted(dirpath.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find any file matching {pattern!r} under {dirpath}")
    return matches[0]


def _find_first_dir(dirpath: Path, pattern: str) -> Path:
    matches = sorted(p for p in dirpath.glob(pattern) if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find any directory matching {pattern!r} under {dirpath}")
    return matches[0]


VARIANTS = [
    {"tag": "gamma=0 (purely local)", "gamma": 0.0,
     "ckpt_pattern": "ff_cifar10_no_all_collab_*_stage1_best.pt", "run_pattern": "ff_cifar10_no_all_collab_*",
     "task_anchor_s1_no_tta": 0.8575},
    {"tag": "gamma=0.7 (CP-FAIR-style)", "gamma": 0.7,
     "ckpt_pattern": "ff_cifar10_baseline_constant_*_stage1_best.pt", "run_pattern": "ff_cifar10_baseline_constant_*",
     "task_anchor_s1_no_tta": 0.8617},
    {"tag": "gamma=1.0 (LCFF)", "gamma": 1.0,
     "ckpt_pattern": "ff_cifar10_lcff_prefix_*_stage1_best.pt", "run_pattern": "ff_cifar10_lcff_prefix_*",
     "task_anchor_s1_no_tta": 0.8579},
]
for v in VARIANTS:
    v["ckpt"] = _find_first(CKPT_DIR, v["ckpt_pattern"])
    v["run_dir"] = _find_first_dir(RUNS_DIR, v["run_pattern"])
    v["config"] = v["run_dir"] / "config.json"

DEVICE = "cpu"
N_BATCHES = 4   # 4 x 256 = 1024 examples per variant
BATCH = 256
DATA_DIR = str(Path(os.environ.get("FF_DATA_DIR", f"{CF}/revised_training/data_c10")).resolve())
ACC_TOL = 0.0002  # 2 of 10,000 test examples


def build_cfg_from_json(path: Path) -> "FFConfig":
    with open(path) as f:
        c = json.load(f)
    cfg = FFConfig()
    cfg.__post_init__()
    for k, v in c.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.aspect_loss_types = list(cfg.aspect_loss_types[: cfg.n_aspects])
    cfg.aspect_lambdas = list(cfg.aspect_lambdas[: cfg.n_aspects])
    cfg.aspect_theta_inits = list(cfg.aspect_theta_inits[: cfg.n_aspects])
    return cfg


def build_eval_loader(cfg, batch_size: int):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),   # CHANGE 1 (fixed) / published (legacy)
    ])
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=eval_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)


@torch.no_grad()
def per_block_margins(net, images, y_pos, y_neg):
    """Unchanged from the published script."""
    net.eval()
    L = net.num_blocks
    tokens0 = net.embed(images)
    tokens = tokens0
    g_hist_pos = []
    gpos_curr_list = []
    for blk in net.blocks:
        g_prev = torch.stack(g_hist_pos, dim=1) if g_hist_pos else None
        if g_prev is not None:
            g_prev = g_prev.detach()
        a, g_vec, _, _, _ = blk(tokens, y_pos, g_prev)
        gpos_curr_list.append(blk.aggregate_goodness(g_vec))
        g_hist_pos.append(g_vec)
        tokens = l2_normalize_tokens(a)
    gpos_curr = torch.stack(gpos_curr_list, dim=0)

    tokens = tokens0
    g_hist_nl = []
    gnl_curr_list = []
    for blk in net.blocks:
        g_prev = torch.stack(g_hist_nl, dim=1) if g_hist_nl else None
        if g_prev is not None:
            g_prev = g_prev.detach()
        a, g_vec, _, _, _ = blk(tokens, y_neg, g_prev)
        gnl_curr_list.append(blk.aggregate_goodness(g_vec))
        g_hist_nl.append(g_vec)
        tokens = l2_normalize_tokens(a)
    gnl_curr = torch.stack(gnl_curr_list, dim=0)
    return gpos_curr, gnl_curr


def random_wrong(y, num_classes):
    r = torch.randint(0, num_classes, y.shape, device=y.device)
    return (r + (r == y).long()) % num_classes


@torch.no_grad()
def accuracy_gate(net, cfg, variant):
    """Trainer protocol for stage1_test_top1_no_tta: FF goodness-sum over all 10 labels, top-1."""
    loader = build_eval_loader(cfg, 256)
    preds, labels = [], []
    t0 = time.time()
    for images, y in loader:
        scores = net.predict_scores(images.to(DEVICE))
        preds.append(scores.topk(1, dim=1, largest=True, sorted=True).indices[:, 0].cpu())
        labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = float((preds == labels).mean())
    ev = [json.loads(l) for l in open(variant["run_dir"] / "events.jsonl") if l.strip()]
    logged = [e for e in ev if e.get("event") == "final_test_eval"][0]["stage1_test_top1_no_tta"]
    z = np.load(variant["run_dir"] / "preds_stage1_test_no_tta.npz")
    assert np.array_equal(z["y_true"], labels)
    agree = float((z["y_pred"] == preds).mean())
    out = {"acc_s1_no_tta": acc, "n": int(len(labels)), "logged_stage1_test_top1_no_tta": logged,
           "task_anchor": variant["task_anchor_s1_no_tta"], "abs_diff_vs_logged": abs(acc - logged),
           "abs_diff_vs_task_anchor": abs(acc - variant["task_anchor_s1_no_tta"]),
           "per_example_agreement_with_logged_preds_npz": agree,
           "n_disagree_with_logged_preds_npz": int((z["y_pred"] != preds).sum()),
           "tolerance": ACC_TOL, "passed": abs(acc - logged) <= ACC_TOL, "seconds": time.time() - t0}
    print(f"  ACC GATE: acc={acc:.4f} logged={logged:.4f} anchor={variant['task_anchor_s1_no_tta']:.4f} "
          f"agree_npz={agree:.4f} passed={out['passed']} ({out['seconds']:.0f}s)")
    return out


def run_variant(variant):
    print(f"\n=== {variant['tag']} (gamma={variant['gamma']}) weights={WEIGHTS} std={NORM_STD} ===")
    cfg = build_cfg_from_json(variant["config"])
    cfg.device = DEVICE
    print(f"  L={cfg.num_blocks} D={cfg.d_model} alpha={cfg.symba_alpha} gamma_scale={cfg.gamma_scale}")

    net = FFHybridNet(cfg).to(DEVICE)
    ckpt = torch.load(variant["ckpt"], map_location=DEVICE, weights_only=False)
    key = "net_ema" if WEIGHTS == "ema" else "model"       # CHANGE 2
    sd = ckpt[key]
    res = net.load_state_dict(sd, strict=True)              # CHANGE 2 (strict)
    print(f"  loaded ckpt['{key}']: {len(sd)} keys, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    ck_meta = {"ckpt": str(variant["ckpt"]), "state_dict_key": key, "n_keys": len(sd),
               "missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys),
               "ckpt_epoch_stage1": ckpt.get("epoch_stage1"), "ckpt_best_s1": ckpt.get("best_s1"),
               "ckpt_cfg_use_sam": (ckpt.get("cfg") or {}).get("use_sam"),
               "ckpt_cfg_gamma_scale": (ckpt.get("cfg") or {}).get("gamma_scale")}
    del ckpt

    loader = build_eval_loader(cfg, BATCH)
    L = cfg.num_blocks
    alpha = float(cfg.symba_alpha)
    gamma = float(variant["gamma"])

    all_m = []
    torch.manual_seed(42)
    for bi, (images, y) in enumerate(loader):
        if bi >= N_BATCHES:
            break
        images = images.to(DEVICE, non_blocking=True)
        y_pos = y.to(DEVICE, non_blocking=True)
        y_neg = random_wrong(y_pos, cfg.num_classes)
        if NEG_LABELS == "published":
            y_neg = torch.from_numpy(np.load(REC_LABELS)[bi * BATCH:(bi + 1) * BATCH]).to(DEVICE)
            assert y_neg.shape == y_pos.shape and bool((y_neg != y_pos).all())
        gpos_curr, gnl_curr = per_block_margins(net, images, y_pos, y_neg)
        all_m.append((gpos_curr - gnl_curr).detach().cpu())

    m_LB = torch.cat(all_m, dim=1)
    L_, N = m_LB.shape
    assert L_ == L
    P_prev = torch.zeros_like(m_LB)
    for d in range(1, L):
        P_prev[d] = m_LB[:d].sum(dim=0)
    M = m_LB + gamma * P_prev
    grad = alpha * torch.sigmoid(-alpha * M)
    grad_local = alpha * torch.sigmoid(-alpha * m_LB)

    summary = []
    for d in range(L):
        summary.append({
            "block": d,
            "m_mean": float(m_LB[d].mean()), "m_median": float(m_LB[d].median()),
            "P_prev_mean": float(P_prev[d].mean()), "P_prev_median": float(P_prev[d].median()),
            "M_mean": float(M[d].mean()),
            "grad_mean": float(grad[d].mean()), "grad_median": float(grad[d].median()),
            "grad_p10": float(torch.quantile(grad[d], 0.10)), "grad_p90": float(torch.quantile(grad[d], 0.90)),
            "grad_local_mean": float(grad_local[d].mean()),
        })
    gate = accuracy_gate(net, cfg, variant) if ACC_GATE else None   # CHANGE 3
    return {"tag": variant["tag"], "gamma": gamma, "alpha": alpha, "L": L, "N": N, "summary": summary,
            "ckpt_meta": ck_meta, "acc_gate": gate,
            "m": m_LB.numpy(), "P_prev": P_prev.numpy(), "M": M.numpy(), "grad": grad.numpy()}


def main():
    out_dir = Path(__file__).resolve().parent / "out" / f"{WEIGHTS}_{NORM_STD}_{NEG_LABELS}"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = range(len(VARIANTS)) if ONLY is None else [int(ONLY)]
    meta = {"weights": WEIGHTS, "norm_std": list(STD), "norm_mean": list(MEAN), "neg_labels": NEG_LABELS,
            "device": DEVICE, "torch": torch.__version__, "trainer": str(TRAINER_DIR / (TRAINER_MODULE + ".py")),
            "n_examples": N_BATCHES * BATCH, "acc_gate_run": ACC_GATE}
    results = []
    for i in idx:  # one file pair per variant, so variants can run in separate <9-min calls
        r = run_variant(VARIANTS[i])
        results.append(r)
        light = {"tag": r["tag"], "gamma": r["gamma"], "alpha": r["alpha"], "summary": r["summary"],
                 "ckpt_meta": r["ckpt_meta"], "acc_gate": r["acc_gate"]}
        with open(out_dir / f"grad_attenuation_data_v{i}.json", "w") as f:
            json.dump({"meta": meta, "variant_index": i, "variant": light}, f, indent=2)
        np.savez_compressed(out_dir / f"grad_attenuation_arrays_v{i}.npz",
                            **{k: r[k] for k in ("m", "P_prev", "M", "grad")},
                            tag=np.array(r["tag"]), gamma=np.array(r["gamma"]))
    print("Saved ->", out_dir)
    for r in results:
        print(f"\n{r['tag']}:")
        for s in r["summary"]:
            print(f"  L{s['block']}  m={s['m_mean']:+.3f}  P_prev={s['P_prev_mean']:+.3f}  "
                  f"M={s['M_mean']:+.3f}  |grad|={s['grad_mean']:.4f}  |grad_local|={s['grad_local_mean']:.4f}")


if __name__ == "__main__":
    main()
