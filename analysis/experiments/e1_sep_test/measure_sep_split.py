#!/usr/bin/env python3
"""E1 on test images: the trainer's exact sep_curr_nl (sep^cur_nl(d), Eq. 7) measured on the TEST split and,
as a control, on CLEAN TRAINING images -- same checkpoints, weights, mode and hard-negative mining as E1 tier 2.

E1 tier 2 (experiments/e1_sep/measure_sep_tier2.py) measured it on the validation-selected EMA checkpoints
in eval mode, through the trainer's own forward path (compute_block_terms, imported unchanged from that
script), on 20 batches of the trainer's TRAINING loader (triple-view training augmentation). This script
changes only the data stream:
  test        : the official test split (eval transform: ToTensor + Normalize), shuffled with a fixed
                seed; the three views x1 = x2 = x3 = the clean image;
  train_clean : the run's own 45k training split (split_indices.json train_idx), same eval transform as the
                run's validation split (the trainer's CIFAR*SingleFromBase over the 50k train images), x1 =
                x2 = x3 = the clean image -- separates "train vs test" from "augmented vs clean";
  train_aug   : the trainer's training loader, exactly as E1 tier 2 (code validation: on CPU, with the same
                --n-batches and --data-seed, it reproduces tier 2 bit for bit; with --device cuda the NI,
                hard-negative and queue draws use the CUDA generator, so from batch 2 on the augmentation
                draws differ from tier 2's and the agreement is statistical only).
Everything else is tier 2's protocol: net_ema in eval mode (the weights behind every reported accuracy),
router tau = gate_tau_start, wrong labels hard-mined by the trainer's _choose_hard_negatives with the EMA
block (k candidates per block as in training), the NL token chain of the trainer (tokens passed on under
the previous block's mined label), torch/numpy/random seeded with --data-seed before the first batch,
SupCon queues restored from the checkpoint and updated as in training (they only affect loss diagnostics, and
the RNG stream through queue sampling). The identity gate (EMA S1 no-TTA test top-1 equals the logged value)
is reused from E2/E1 outputs for the same checkpoint SHA-256 when they exist, else recomputed on the CPU.
Per block, the reported value is the mean over all examples (= mean of the equal-size batch means; the
last incomplete batch is dropped), with the per-batch SD and SE = SD/sqrt(n_batches).

CPU: CUDA_VISIBLE_DEVICES='' python measure_sep_split.py ...
GPU: FFCR_ALLOW_GPU=1 python measure_sep_split.py --device cuda ... (FP32, TF32 off; the test and train_clean
images are deterministic, so they are identical on both devices). The "Test images" column of the paper's
App. Tab. e1_aligned was measured this way (--splits test train_clean --n-batches 19 for CIFAR-10, 39 for
CIFAR-100); the hard-negative candidates are then drawn on the GPU, so a CPU rerun reproduces the images but
not the candidate draws: its values should agree within the standard errors, not bit for bit.
"""
import argparse
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LAUNCH_CWD = Path.cwd()            # C.setup_env (run by the tier-2 import) chdirs into experiments/_work
sys.dont_write_bytecode = True     # no __pycache__ next to the imported scripts
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "e1_sep"))
import measure_sep_tier2 as T  # noqa: E402  (calls C.setup_env -> torch, threads from FFCR_THREADS)
import ffcr_common as C  # noqa: E402

torch = T.torch
import numpy as np  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

OUT_DIR = C.EXP / "e1_sep_test" / "out"
TIER2_OUT = T.OUT_DIR
DEFAULT_NAMES = (
    # C10 L4/D128: the four history rules of the paper's L4/D128 checks, the E1 points and MGC
    ["c10_gated_no_all_collab_seed42", "c10_gated_baseline_constant_seed42", "c10_gated_lcff_prefix_seed42",
     "c10_gated_adaptive_t1_k0_seed42", "c10_gated_k0_seed123", "c10_gated_k0_seed456", "c10_mgc_c1_seed42"]
    # CIFAR-100 trio (the central dissociation, Tab. 3)
    + [f"c100_{v}_seed{s}" for v in ("gamma0", "gated_k0", "cumulative") for s in (42, 123, 456)]
    # C10 L4/D256 32-expert (Tab. 30 / Figs. 4-5)
    + ["c10_gamma0_seed123", "c10_gamma0_seed456"] + [f"c10_cp_fair_seed{s}" for s in (42, 123, 456, 789)]
)


def make_stream(split, M, cfg, entry, args):
    """Builds the data stream BEFORE the global seeding (as tier 2 builds its loaders before seeding), and
    returns (iterable of (x1, x2, x3, labels), description)."""
    train_loader, val_loader, test_loader, split_info, run_split = C.loaders(M, cfg, entry["run_dir"], workers=0)
    if split == "train_aug":
        assert args.n_batches <= len(train_loader), f"{split}: only {len(train_loader)} full batches"
        return train_loader, ("trainer make_loaders_stage1 train loader (triple-view training aug, shuffle, "
                              "drop_last), workers=0")
    if split == "test":
        ds = test_loader.dataset
        desc = f"official test split ({len(ds)} images), eval transform"
    elif split == "train_clean":
        vd = val_loader.dataset
        assert isinstance(vd, Subset), type(vd)
        base = vd.dataset                                     # CIFAR*SingleFromBase over the 50k train images
        tr_idx = list(split_info["train_idx"])
        assert tr_idx == run_split["train_idx"] and len(tr_idx) + len(run_split["val_idx"]) == len(base)
        assert not (set(tr_idx) & set(run_split["val_idx"])), "train/val overlap"
        ds = base
        pool = tr_idx                                         # draw only from the run's training images
        desc = f"run's training split ({len(tr_idx)} images), eval transform of its validation split"
    else:
        raise ValueError(split)
    if split == "test":
        pool = list(range(len(ds)))
    tf = ds.transform
    names = [t.__class__.__name__ for t in tf.transforms]
    assert names == ["ToTensor", "Normalize"], names         # clean images only
    # a random subset in random order, drawn with its own generator (seeded apart from the global stream that
    # draws NI, HNM and queue samples); the indices into the CIFAR split are recorded
    B = int(cfg.batch_size)
    n = args.n_batches * B
    assert n <= len(pool), f"{split}: only {len(pool) // B} full batches"
    g = torch.Generator().manual_seed(args.data_seed + 1)
    idx = [pool[i] for i in torch.randperm(len(pool), generator=g)[:n].tolist()]
    loader = DataLoader(Subset(ds, idx), batch_size=B, shuffle=False, drop_last=True, num_workers=0)
    make_stream.indices = idx
    return ((images, images, images, labels) for images, labels in loader), desc + f"; transforms {names}"


def measure(name, split, args):
    t0 = time.time()
    entry = C.REG[name]
    M = C.load_trainer(entry["trainer"])
    ckpt_id = C.file_id(entry["ckpt"])
    try:
        ckpt = torch.load(entry["ckpt"], map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(entry["ckpt"], map_location="cpu", weights_only=False)
    persisted, cfg_src = C.persisted_config(entry, ckpt)
    cfg = C.build_cfg(M, persisted)
    cfg.device = args.device
    use_mgc = bool(getattr(cfg, "use_mgc", False))
    fte, rend = C.read_events(entry["run_dir"])
    E = int(ckpt["epoch_stage1"])
    assert abs(float(ckpt["best_s1"]) - float(rend["best_s1"])) < 1e-9, f"{name}: best_s1 mismatch"

    nets = C.load_nets(M, cfg, ckpt, which=("net_ema",))
    net_ema = nets["net_ema"]
    for p in net_ema.parameters():
        p.requires_grad_(False)
    net_ema.eval()

    gate = (T.e2_gate(name, ckpt_id["sha256"]) or T.own_gate(TIER2_OUT, name, ckpt_id["sha256"])
            or T.own_gate(args.out_dir, name, ckpt_id["sha256"]))
    if gate is None:
        _, _, test_loader, _, _ = C.loaders(M, cfg, entry["run_dir"], workers=0)
        tg = time.time()
        gate = T.recompute_test_gate(net_ema, test_loader, float(fte["stage1_test_top1_no_tta"]), entry["run_dir"])
        gate["source"] = f"recomputed here ({time.time() - tg:.0f}s)"
    assert gate["pass_"], f"{name}: identity gate failed {gate}"
    net_ema.to(args.device)                               # after the gate, which runs on CPU batches

    queues = [M.ContrastiveQueue(cfg.queue_size, cfg.proj_dim, device=args.device) for _ in range(cfg.num_blocks)]
    for q, qs in zip(queues, ckpt["queues"]):
        q.z.copy_(qs["z"])
        q.y.copy_(qs["y"])
        q.ptr = int(qs["ptr"])
        q.full = bool(qs["full"])
    del ckpt
    net = net_ema
    make_stream.indices = None
    data, data_desc = make_stream(split, M, cfg, entry, args)

    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    L = cfg.num_blocks
    rec = [[] for _ in range(L)]
    n_done = 0
    tb = time.time()
    with torch.no_grad():
        for x1, x2, x3, labels in data:
            if n_done >= args.n_batches:
                break
            if args.device != "cpu":
                same = (x2 is x1) and (x3 is x1)
                x1, labels = x1.to(args.device), labels.to(args.device)
                x2, x3 = (x1, x1) if same else (x2.to(args.device), x3.to(args.device))
            B = x1.size(0)
            x_ni = x1[T.ni_indices(labels, B, labels.device)]
            t_pos1 = t_pos2 = t_nl = t_ni = None
            g_hist_pos1, g_hist_pos2, g_hist_nl, g_hist_ni = [], [], [], []
            for li in range(L):
                total, caches, diags = T.compute_block_terms(
                    M, cfg, net, net_ema, li, x1, x2, x3, x_ni, labels,
                    t_pos1, t_pos2, t_nl, t_ni,
                    g_hist_pos1, g_hist_pos2, g_hist_nl, g_hist_ni,
                    queues, use_mgc)
                rec[li].append(diags)
                t_pos1 = M.l2_normalize_tokens(caches["a_pos1"].detach())
                t_pos2 = M.l2_normalize_tokens(caches["a_pos2"].detach())
                t_nl = M.l2_normalize_tokens(caches["a_nl"].detach())
                t_ni = M.l2_normalize_tokens(caches["a_ni"].detach())
                g_hist_pos1.append(caches["gpos1"].detach())
                g_hist_pos2.append(caches["gpos2"].detach())
                g_hist_nl.append(caches["gnl"].detach())
                g_hist_ni.append(caches["gni"].detach())
                queues[li].enqueue(caches["z1"].detach(), labels.detach())
                queues[li].enqueue(caches["z2"].detach(), labels.detach())
                del caches, total
            n_done += 1
            print(f"[{name}|{split}] batch {n_done}/{args.n_batches}: deepest sep_curr_nl="
                  f"{rec[L - 1][-1]['margin_curr_nl']:.4f} ({time.time() - tb:.0f}s)", flush=True)
    assert n_done == args.n_batches, f"only {n_done} full batches available"

    lm = C.layer_metrics_at_epoch(entry, E)
    # the paper's tables quote the LAST Stage-1 epoch; the checkpoint (best-val) epoch can be earlier
    E_last = max(int(r["epoch"]) for r in map(json.loads, open(Path(entry["run_dir"]) / "layer_metrics.jsonl"))
                 if r.get("stage", 1) == 1)
    lm_last = C.layer_metrics_at_epoch(entry, E_last)
    t2p = TIER2_OUT / f"{name}__ema_eval.json"
    t2 = json.load(open(t2p)) if t2p.exists() else None
    if t2 is not None and not (t2.get("status") == "OK" and t2["ckpt"]["sha256"] == ckpt_id["sha256"]
                               and t2["protocol"]["data_seed"] == args.data_seed):
        t2 = None                                         # only a tier-2 output of this checkpoint and seed
    clean = split in ("test", "train_clean")
    per_block = []
    for li in range(L):
        stats = {}
        for k in T.DIAG_KEYS:
            v = np.array([d[k] for d in rec[li]], dtype=np.float64)
            stats[k] = dict(mean=float(v.mean()),
                            sd_batches=float(v.std(ddof=1)) if len(v) > 1 else None,
                            se=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None)
        mine = [d["margin_curr_nl"] for d in rec[li]]
        row = dict(
            block=li, k_neg=rec[li][0]["k_neg"], measured=stats, sep_curr_nl_per_batch=mine,
            logged_train_sep_curr_nl_at_ckpt_epoch=(lm.get(li, {}) or {}).get("sep_curr_nl"),
            logged_train_sep_curr_nl_last_epoch=(lm_last.get(li, {}) or {}).get("sep_curr_nl"),
        )
        if t2 is not None:
            t2m = t2["per_block"][li]["measured"]["margin_curr_nl"]
            t2pb = t2["per_block"][li]["sep_curr_nl_per_batch"]
            row.update(tier2_train_aug_sep_curr_nl=t2m["mean"], tier2_train_aug_se=t2m["se"],
                       tier2_n_batches=len(t2pb), tier2_mean_first_n=float(np.mean(t2pb[:n_done])))
            if split == "train_aug":                      # code validation: must be exactly tier 2's batches
                row["tier2_per_batch_max_abs_diff"] = float(max(abs(a - b) for a, b in zip(mine, t2pb)))
        per_block.append(row)
    return dict(
        name=name, split=split, status="OK", config=C.E1_CONFIG.get(name, entry["config"]),
        device=(args.device if args.device == "cpu" else
                f"{args.device} ({torch.cuda.get_device_name(0)}; FP32, TF32 off)"),
        indices=make_stream.indices,
        dataset=entry["dataset"], seed=entry["seed"], ckpt=ckpt_id, run_dir=str(entry["run_dir"]),
        config_source=cfg_src, epoch_stage1=E, last_stage1_epoch=E_last, weights="net_ema, eval mode",
        protocol=dict(split=split, data=data_desc, data_seed=args.data_seed,
                      order_seed=(args.data_seed + 1 if clean else None), n_batches=n_done,
                      batch_size=int(cfg.batch_size), n_examples=int(n_done * cfg.batch_size),
                      hnm="trainer _choose_hard_negatives with the EMA block (use_ema_hnm=%s)" % bool(cfg.use_ema_hnm)),
        degenerate_keys=(["loss_total", "loss_block_raw"] if clean else []),   # x1 = x2: SupCon is degenerate
        gate=dict(source=gate.get("source"), pass_=bool(gate["pass_"]), delta=gate.get("delta")),
        gates=dict(test_gate=json.loads(json.dumps(dict(gate), default=lambda o: o.item() if hasattr(o, 'item')
                                                  else str(o)))),   # JSON-safe; own_gate() reads it on reruns
        accuracies_from_events=dict(s1_test_no_tta=fte.get("stage1_test_top1_no_tta"),
                                    s2_test_tta=fte.get("stage2_test_top1_tta")),
        per_block=per_block,
        deepest_block=dict(block=L - 1, sep_curr_nl=per_block[-1]["measured"]["margin_curr_nl"]["mean"],
                           se=per_block[-1]["measured"]["margin_curr_nl"]["se"]),
        seconds=round(time.time() - t0, 1), seconds_measure=round(time.time() - tb, 1),
        torch=torch.__version__, threads=torch.get_num_threads(),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    ap.add_argument("--splits", nargs="*", default=["test", "train_clean"],
                    choices=["test", "train_clean", "train_aug"])
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--data-seed", type=int, default=20260725)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    if args.device == "cuda":                             # needs FFCR_ALLOW_GPU=1 (see C.setup_env)
        assert os.environ.get("FFCR_ALLOW_GPU") == "1" and torch.cuda.is_available(), "GPU not enabled"
        torch.backends.cuda.matmul.allow_tf32 = False     # exact FP32, comparable with the CPU runs
        torch.backends.cudnn.allow_tf32 = False
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else (LAUNCH_CWD / out_dir).resolve()
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for name in args.names:
        for split in args.splits:
            path = out_dir / f"{name}__{split}.json"
            if path.exists() and not args.force:
                prev = json.load(open(path))
                if prev.get("status") == "OK" and prev["protocol"]["n_batches"] == args.n_batches \
                        and prev["protocol"]["data_seed"] == args.data_seed:
                    print(f"[{name}|{split}] done already; skipping", flush=True)
                    continue
            print(f"\n===== {name} | {split} =====", flush=True)
            try:
                out = measure(name, split, args)
            except AssertionError as err:                 # record and go on with the other runs
                n_fail += 1
                C.write_json_atomic(path, dict(name=name, split=split, status="FAIL", error=str(err)))
                print(f"[{name}|{split}] FAIL: {err}", flush=True)
                continue
            C.write_json_atomic(path, out)
            dp = out["deepest_block"]
            t2s = [round(b["tier2_train_aug_sep_curr_nl"], 3) if "tier2_train_aug_sep_curr_nl" in b else None
                   for b in out["per_block"]]
            se = "n/a" if dp["se"] is None else f"{dp['se']:.4f}"     # no SE from a single batch
            print(f"[{name}|{split}] deepest sep_curr_nl = {dp['sep_curr_nl']:.4f} +- {se} | per block "
                  f"{[round(b['measured']['margin_curr_nl']['mean'], 3) for b in out['per_block']]} | tier2 aug-train "
                  f"{t2s} | {out['seconds']}s", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
