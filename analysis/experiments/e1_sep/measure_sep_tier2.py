#!/usr/bin/env python3
"""E1 tier 2 (CR-18): deepest-block sep_curr_nl RECOMPUTED on each validation-selected
Stage-1 checkpoint, through the trainer's own diagnostic path.

Port of analysis/recovered/gradshare/measure_grad_share.py (the rebuttal gradient-share run):
  * compute_block_terms() is kept verbatim (it mirrors train_epoch_stage1 -> forward_block and
    records margin_curr_nl = mean(gpos_curr - gnl_curr), which is exactly the trainer's logged
    `sep_curr_nl`), with two changes: the constant-gating assert is relaxed (the hardness-gated
    kappa=0 arms use gamma_gating_mode='adaptive'; the adaptive gate is applied with the trainer's
    own _compute_adaptive_gamma / _prev_only_goodness, so the cumulative-margin diagnostics are
    right too -- sep_curr_nl itself never depends on gamma), and no gradients are taken;
  * paths come from experiments/ffcr_common.py (no hard-coded temporary or home paths).

Modes
  ema_eval  (E1 tier 2, the reported number): the network is net_ema (the weights that produce
            every reported accuracy) in eval mode; hard negatives come from the EMA teacher via
            the trainer's _choose_hard_negatives (use_ema_hnm=True in every run), i.e. from the
            same EMA network; router tau = gate_tau_start (the trainer never sets tau on the EMA
            net, so this is the tau at which the EMA was evaluated and used for HNM).
  raw_train (code validation only): gradshare's own path -- raw ckpt['model'] in train mode,
            router tau at the checkpoint epoch, EMA-teacher HNM. On MGC-D128 seed 42 the gradshare
            run (GPU, 20 batches, data_seed 20260725) measured block-3 margin_curr_nl = 5.2346.

Averaging protocol (both modes, fixed so the number is reproducible)
  * stream: the trainer's own make_loaders_stage1() training loader (45k train split of the run,
    asserted equal to the run's split_indices.json), triple-view training augmentation, shuffle,
    drop_last, batch = cfg.batch_size (512 C10, 256 C100), workers=0 (all randomness in-process);
  * seeding: torch/numpy/random seeded with --data-seed (default 20260725, the gradshare seed)
    immediately before the first batch; NI indices, HNM candidate labels and queue samples are
    drawn from the same CPU generator in the trainer's order;
  * N = --n-batches consecutive batches (default 20); per block, the reported value is the mean
    over all N*B examples (= mean of the N equal-size batch means), with the per-batch SD (ddof=1)
    and SE = SD/sqrt(N);
  * SupCon queues are restored from the checkpoint and updated as in training (they only affect
    loss diagnostics, and the RNG stream through queue sampling);
  * no optimizer step, no EMA update.

Identity gate (before measuring): EMA S1 no-TTA top-1 on the official test split must equal the
run's events.jsonl final_test_eval stage1_test_top1_no_tta within 1e-4 (0.01 pp = one test image);
pre-declared numerics fallback in ffcr_common.test_gate (|delta| <= 3 images and >= 99.9% argmax
agreement with the run's stored GPU predictions; the trainers evaluated with TF32). The gate may be
taken from the E2 dump sidecar of the same checkpoint (same SHA-256) when that dump passed, or from
an earlier output of this script for the same checkpoint, to avoid repeated full test passes
(a CIFAR-100 test pass is ~40+ CPU-min). Also checked:
ckpt['best_s1'] == run_end best_s1 and ckpt['epoch_stage1'] == the best-val epoch in
epoch_metrics.jsonl.

CPU only: run with CUDA_VISIBLE_DEVICES=''; torch threads = 6.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ffcr_common as C  # noqa: E402

torch = C.setup_env(threads=int(os.environ.get("FFCR_THREADS", "6")))
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT_DIR = C.EXP / "e1_sep" / "out"
E2_DUMPS = C.EXP / "e2_exit" / "dumps"
GRADSHARE_MGC_BLOCK3 = 5.234605669975281   # AN/gradient_share_results.json, c10_d128_mgc_c1_seed42, block 3
GRADSHARE_JSON = C.PR / "ff-wiki" / "_build" / "analyses" / "gradient_share_results.json"


# --------------------------------------------------------------------------------------
# compute_block_terms: verbatim port of measure_grad_share.py:132-354, gating assert relaxed.
# --------------------------------------------------------------------------------------
def compute_block_terms(M, cfg, net, net_ema, li, x1, x2, x3, x_ni, labels,
                        t_pos1, t_pos2, t_nl, t_ni,
                        g_hist_pos1, g_hist_pos2, g_hist_nl, g_hist_ni,
                        queues, use_mgc):
    """Faithful mirror of the trainer's forward_block(). Returns terms, total, caches, diags."""
    block = net.blocks[li]
    B = x1.size(0)
    gating = getattr(cfg, "gamma_gating_mode", "constant") or "constant"
    assert gating in ("constant", "adaptive"), gating   # relaxed (was: == "constant")

    k_neg = int(round(
        cfg.hard_negative_k_first +
        (cfg.hard_negative_k_last - cfg.hard_negative_k_first) * (li / max(1, cfg.num_blocks - 1))
    ))
    hnm_block = block
    if net_ema is not None and getattr(cfg, "use_ema_hnm", False):
        hnm_block = net_ema.blocks[li]

    if li == 0:
        t_pos1_local = net.embed(x1)
        t_pos2_local = net.embed(x2)
        t_ni_local = net.embed(x_ni)
        t_nl_local = net.embed(x3)
    else:
        t_pos1_local, t_pos2_local, t_nl_local, t_ni_local = t_pos1, t_pos2, t_nl, t_ni

    def get_g_prev(hist):
        g_prev = torch.stack(hist, dim=1) if hist else None
        return g_prev.detach() if g_prev is not None else None

    g_prev_pos1 = get_g_prev(g_hist_pos1)
    g_prev_pos2 = get_g_prev(g_hist_pos2)
    g_prev_nl = get_g_prev(g_hist_nl)
    g_prev_ni = get_g_prev(g_hist_ni)

    # positives
    a_pos1, gpos1, lb_pos1, zloss_pos1, recon1 = block(t_pos1_local, labels, g_prev_pos1)
    a_pos2, gpos2, lb_pos2, zloss_pos2, recon2 = block(t_pos2_local, labels, g_prev_pos2)
    # wrong-label negatives (NL) with hard-negative mining
    y_nl = M._choose_hard_negatives(hnm_block, t_nl_local, labels, g_prev_nl, cfg.num_classes, k_neg)
    a_nl, gnl, lb_nl, zloss_nl, _ = block(t_nl_local, y_nl, g_prev_nl)
    # wrong-image negatives (NI)
    a_ni, gni, lb_ni, zloss_ni, _ = block(t_ni_local, labels, g_prev_ni)

    gamma_pos1 = M._sum_prev_goodness(net, g_hist_pos1)
    gamma_nl = M._sum_prev_goodness(net, g_hist_nl)
    gamma_ni = M._sum_prev_goodness(net, g_hist_ni)
    zero = a_pos1.new_zeros(B)
    if gamma_pos1 is None:
        gamma_pos1 = zero
    if gamma_nl is None:
        gamma_nl = zero
    if gamma_ni is None:
        gamma_ni = zero

    # ---- hardness-gated adaptive gamma (trainer's own helpers; added for the gated arms) ----
    gate_pos1 = gate_nl = gate_ni = None
    if gating == "adaptive" and li > 0:
        if getattr(cfg, "gamma_M_mode", "cumulative") == "cumulative":
            M_pos1, M_nl, M_ni = gamma_pos1, gamma_nl, gamma_ni
        else:  # "prev_only"
            M_pos1 = M._prev_only_goodness(net, g_hist_pos1)
            M_nl = M._prev_only_goodness(net, g_hist_nl)
            M_ni = M._prev_only_goodness(net, g_hist_ni)
        gate_pos1 = M._compute_adaptive_gamma(net, li, M_pos1, cfg)
        gate_nl = M._compute_adaptive_gamma(net, li, M_nl, cfg)
        gate_ni = M._compute_adaptive_gamma(net, li, M_ni, cfg)

    # ---- aspect losses ----
    thetas = block.aspect_thetas()
    loss_aspects = 0.0
    for k in range(cfg.n_aspects):
        lp = gpos1[:, k]
        ln_nl = gnl[:, k]
        ln_ni = gni[:, k]
        if cfg.aspect_loss_types[k] == "margin":
            loss_k_nl = M.ff_margin_loss(lp, ln_nl, thetas[k])
            loss_k_ni = M.ff_margin_loss(lp, ln_ni, thetas[k])
        else:
            loss_k_nl = M.pairwise_rank_loss(lp, ln_nl)
            loss_k_ni = M.pairwise_rank_loss(lp, ln_ni)
        loss_k = (1.0 - cfg.sff_ni_weight) * loss_k_nl + cfg.sff_ni_weight * loss_k_ni
        loss_aspects += cfg.aspect_lambdas[k] * loss_k

    # ---- block goodness (+ gamma) ----
    gpos_curr = block.aggregate_goodness(gpos1)
    gnl_curr = block.aggregate_goodness(gnl)
    gni_curr = block.aggregate_goodness(gni)

    # ---- current-block discrimination ----
    if li > 0:
        with torch.no_grad():
            prev_margin_nl = (gamma_pos1 - gamma_nl).detach()
            prev_margin_ni = (gamma_pos1 - gamma_ni).detach()
            w_nl = torch.sigmoid(-cfg.symba_alpha * prev_margin_nl)
            w_ni = torch.sigmoid(-cfg.symba_alpha * prev_margin_ni)
            w_nl = w_nl / (w_nl.mean() + 1e-6)
            w_ni = w_ni / (w_ni.mean() + 1e-6)
        loss_block_curr_nl_vec = F.softplus(-cfg.symba_alpha * (gpos_curr - gnl_curr))
        loss_block_curr_ni_vec = F.softplus(-cfg.symba_alpha * (gpos_curr - gni_curr))
        loss_block_curr_nl = (loss_block_curr_nl_vec * w_nl).mean()
        loss_block_curr_ni = (loss_block_curr_ni_vec * w_ni).mean()
    else:
        loss_block_curr_nl = M.symba_loss(gpos_curr, gnl_curr, alpha=cfg.symba_alpha)
        loss_block_curr_ni = M.symba_loss(gpos_curr, gni_curr, alpha=cfg.symba_alpha)
    loss_block_curr = ((1.0 - cfg.sff_ni_weight) * loss_block_curr_nl
                       + cfg.sff_ni_weight * loss_block_curr_ni)

    if gate_pos1 is not None:   # adaptive per-sample gate
        gpos_blk = gpos_curr + gate_pos1 * gamma_pos1
        gnl_blk = gnl_curr + gate_nl * gamma_nl
        gni_blk = gni_curr + gate_ni * gamma_ni
    else:                       # constant gamma (or block 0)
        gpos_blk = gpos_curr + cfg.gamma_scale * gamma_pos1
        gnl_blk = gnl_curr + cfg.gamma_scale * gamma_nl
        gni_blk = gni_curr + cfg.gamma_scale * gamma_ni

    # ---- block discrimination (SYMBA, cumulative) ----
    loss_block_nl = M.symba_loss(gpos_blk, gnl_blk, alpha=cfg.symba_alpha)
    loss_block_ni = M.symba_loss(gpos_blk, gni_blk, alpha=cfg.symba_alpha)
    loss_block = (1.0 - cfg.sff_ni_weight) * loss_block_nl + cfg.sff_ni_weight * loss_block_ni

    # ---- MGC compensated local term (mgc_trainer only) ----
    loss_mgc_local = None
    mgc_lam_mean = None
    if use_mgc:
        if li > 0:
            with torch.no_grad():
                m_nl_d = (gpos_curr - gnl_curr).detach()
                m_ni_d = (gpos_curr - gni_curr).detach()
                M_nl_d = (gpos_blk - gnl_blk).detach()
                M_ni_d = (gpos_blk - gni_blk).detach()
                s_m_nl = torch.sigmoid(-cfg.symba_alpha * m_nl_d)
                s_m_ni = torch.sigmoid(-cfg.symba_alpha * m_ni_d)
                s_M_nl = torch.sigmoid(-cfg.symba_alpha * M_nl_d)
                s_M_ni = torch.sigmoid(-cfg.symba_alpha * M_ni_d)
                c_d = float(cfg.mgc_c0) * (
                    1.0 + float(cfg.mgc_rho) * li / max(1, cfg.num_blocks - 1))
                lam_nl = torch.clamp(c_d - s_M_nl / (s_m_nl + cfg.mgc_eps), min=0.0)
                lam_ni = torch.clamp(c_d - s_M_ni / (s_m_ni + cfg.mgc_eps), min=0.0)
                mgc_lam_mean = float((0.5 * (lam_nl.mean() + lam_ni.mean())).item())
            mgc_nl = (lam_nl * F.softplus(-cfg.symba_alpha * (gpos_curr - gnl_curr))).mean()
            mgc_ni = (lam_ni * F.softplus(-cfg.symba_alpha * (gpos_curr - gni_curr))).mean()
            loss_mgc_local = (1.0 - cfg.sff_ni_weight) * mgc_nl + cfg.sff_ni_weight * mgc_ni
        else:
            c0_excess = max(0.0, float(cfg.mgc_c0) * (1.0 if cfg.mgc_rho == 0.0 else 1.0) - 1.0)
            if c0_excess > 0.0:
                mgc_nl = c0_excess * F.softplus(-cfg.symba_alpha * (gpos_curr - gnl_curr)).mean()
                mgc_ni = c0_excess * F.softplus(-cfg.symba_alpha * (gpos_curr - gni_curr)).mean()
                loss_mgc_local = (1.0 - cfg.sff_ni_weight) * mgc_nl + cfg.sff_ni_weight * mgc_ni

    # ---- depth ordering ----
    loss_depth = 0.0
    if li > 0 and getattr(cfg, "depth_order_lambda", 0.0) > 0.0:
        prev_pos_blk = net.blocks[li - 1].aggregate_goodness(g_hist_pos1[-1])
        loss_depth += F.softplus(cfg.depth_margin_pos - (gpos_blk - prev_pos_blk)).mean()
        prev_nl_blk = net.blocks[li - 1].aggregate_goodness(g_hist_nl[-1])
        loss_depth += F.softplus(cfg.depth_margin_neg - (prev_nl_blk - gnl_blk)).mean()
        prev_ni_blk = net.blocks[li - 1].aggregate_goodness(g_hist_ni[-1])
        loss_depth += F.softplus(cfg.depth_margin_neg - (prev_ni_blk - gni_blk)).mean()
        loss_depth *= cfg.depth_order_lambda

    # ---- MoE aux ----
    loss_bal = (lb_pos1 + lb_pos2 + lb_nl + lb_ni) / 4.0
    loss_z = (zloss_pos1 + zloss_pos2 + zloss_nl + zloss_ni) / 4.0

    # ---- reconstruction (None in eval mode) ----
    loss_recon = 0.0
    if getattr(cfg, "recon_lambda", 0.0) > 0.0 and recon1 is not None and recon2 is not None:
        loss_recon += F.mse_loss(recon1, t_pos1_local.detach())
        loss_recon += F.mse_loss(recon2, t_pos2_local.detach())
        loss_recon *= cfg.recon_lambda

    # ---- contrastive ----
    z1 = block.contrastive_embed(a_pos1)
    z2 = block.contrastive_embed(a_pos2)
    z_all = torch.cat([z1, z2], dim=0)
    y_all = torch.cat([labels, labels], dim=0)
    if getattr(cfg, "queue_neg_samples", 0) > 0:
        zq, yq = queues[li].sample(cfg.queue_neg_samples)
        if zq is not None:
            z_all = torch.cat([z_all, zq.detach()], dim=0)
            y_all = torch.cat([y_all, yq.detach()], dim=0)
    loss_con = M.supcon_generic(z_all, y_all, temperature=cfg.contrastive_temp)

    moe_bal_w = float(getattr(cfg, "moe_balance_coef", 0.0))
    moe_z_w = float(getattr(cfg, "moe_zloss_coef", 0.0))
    block_curr_lambda_eff = float(cfg.block_curr_lambda)
    slope = float(getattr(cfg, "block_curr_depth_slope", 0.0))
    if slope != 0.0:
        block_curr_lambda_eff = float(cfg.block_curr_lambda) * (
            1.0 + slope * (li / max(1, cfg.num_blocks - 1)))
    if use_mgc:
        block_curr_lambda_eff = 0.0

    total = (loss_aspects
             + cfg.block_lambda * loss_block
             + block_curr_lambda_eff * loss_block_curr
             + (cfg.block_lambda * loss_mgc_local if (use_mgc and loss_mgc_local is not None) else 0.0)
             + loss_depth
             + moe_bal_w * loss_bal
             + moe_z_w * loss_z
             + cfg.contrastive_lambda * loss_con
             + loss_recon)

    with torch.no_grad():
        m_curr_nl = (gpos_curr - gnl_curr)
        m_curr_ni = (gpos_curr - gni_curr)
        diags = dict(
            loss_block_raw=float(loss_block.item()),
            loss_total=float(total.item()),
            margin_blk_nl=float((gpos_blk - gnl_blk).mean().item()),
            margin_curr_nl=float(m_curr_nl.mean().item()),          # == trainer's sep_curr_nl
            margin_curr_ni=float(m_curr_ni.mean().item()),          # == trainer's sep_curr_ni
            margin_curr_nl_sd=float(m_curr_nl.std(unbiased=False).item()),  # == inc_margin_nl_std
            sep_nl_frac=float((gpos_blk > gnl_blk).float().mean().item()),  # == trainer's sep_nl
            gpos_cur=float(gpos_curr.mean().item()),
            gnl_cur=float(gnl_curr.mean().item()),
            gamma_pos=float(gamma_pos1.mean().item()),
            mgc_lam_mean=mgc_lam_mean,
            k_neg=k_neg,
        )

    caches = dict(a_pos1=a_pos1, a_pos2=a_pos2, a_nl=a_nl, a_ni=a_ni,
                  gpos1=gpos1, gpos2=gpos2, gnl=gnl, gni=gni, z1=z1, z2=z2)
    return total, caches, diags


DIAG_KEYS = ("margin_curr_nl", "margin_curr_ni", "margin_curr_nl_sd", "margin_blk_nl", "sep_nl_frac",
             "gpos_cur", "gnl_cur", "gamma_pos", "loss_block_raw", "loss_total")
LOGGED_KEYS = ("sep_curr_nl", "sep_curr_ni", "inc_margin_nl_std", "margin_blk_nl", "sep_nl",
               "gpos_cur", "gnl_cur", "gamma_pos", "loss_block", "loss_total")


def ni_indices(labels, B, device):
    """NI sampling -- verbatim trainer logic."""
    if B > 1:
        base = torch.arange(B, device=device)
        indices_ni = torch.randint(0, B, (B,), device=device)
        bad = (indices_ni == base) | (labels[indices_ni] == labels)
        for _ in range(10):
            if not bad.any():
                break
            indices_ni[bad] = torch.randint(0, B, (int(bad.sum().item()),), device=device)
            bad = (indices_ni == base) | (labels[indices_ni] == labels)
        if bad.any():
            shift = torch.randint(1, B, (1,), device=device).item()
            indices_ni = (base + shift) % B
    else:
        indices_ni = torch.arange(B, device=device)
    return indices_ni


def e2_gate(name, ckpt_sha):
    """Reuse the E2 dump's test gate for the same checkpoint (identical SHA-256), if it passed."""
    p = E2_DUMPS / f"{name}.json"
    if not p.exists():
        return None
    e = json.load(open(p))
    if e.get("ckpt_sha256") != ckpt_sha or not e.get("gate_test_pass") \
            or e.get("status") not in ("PASS", "PASS_NUMERICS"):
        return None
    return dict(e["gate_test"], source=f"E2 dump {p}")


def own_gate(out_dir, name, ckpt_sha):
    """Reuse a test gate this script already computed for the same checkpoint (other mode)."""
    for p in sorted(Path(out_dir).glob(f"{name}__*.json")):
        try:
            e = json.load(open(p))
        except Exception:
            continue
        g = e.get("gates", {}).get("test_gate", {})
        if e.get("status") == "OK" and e.get("ckpt", {}).get("sha256") == ckpt_sha and g.get("pass_") \
                and str(g.get("source", "")).startswith("recomputed here"):
            return dict(g, source=f"E1 earlier output, same sha256 ({p})")
    return None


@torch.no_grad()
def recompute_test_gate(net_ema, test_loader, anchor, run_dir):
    """The trainer's evaluate_ff(tta=False) computation (argmax of predict_scores), keeping the
    predictions so that the numerics fallback can compare them with the stored GPU predictions."""
    net_ema.eval()
    preds, ys, scores = [], [], []
    for batch in test_loader:
        images, labels = batch[0], batch[-1]
        sc = net_ema.predict_scores(images)
        scores.append(sc.numpy())
        preds.append(sc.argmax(1).numpy())
        ys.append(labels.numpy())
    return C.test_gate(np.concatenate(preds), np.concatenate(ys), anchor, run_dir, scores=np.concatenate(scores))


def measure(name, args):
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
    cfg_diff = None if entry.get("cfg_from_ckpt") else C.cfg_vs_ckpt_diff(persisted, ckpt)
    use_mgc = bool(getattr(cfg, "use_mgc", False))
    fte, rend = C.read_events(entry["run_dir"])
    anchor_test = float(fte["stage1_test_top1_no_tta"])

    # ---- checkpoint identity ----
    E = int(ckpt["epoch_stage1"])
    ckpt_best_s1 = float(ckpt["best_s1"])
    best_row = C.best_epoch_row(entry)
    id_best_s1 = abs(float(ckpt["best_s1"]) - float(rend["best_s1"])) < 1e-9
    id_epoch = (best_row is not None and int(best_row["epoch"]) == E)
    assert id_best_s1, f"{name}: ckpt best_s1 {ckpt['best_s1']} != run_end {rend['best_s1']}"

    train_loader, _, test_loader, split_info, _ = C.loaders(M, cfg, entry["run_dir"], workers=0)

    nets = C.load_nets(M, cfg, ckpt, which=("net_ema",) + (("model",) if args.mode == "raw_train" else ()))
    net_ema = nets["net_ema"]
    for p in net_ema.parameters():
        p.requires_grad_(False)

    # ---- identity gate: EMA S1 no-TTA test top-1 vs events.jsonl (strict 1e-4; numerics fallback) ----
    if args.timing_probe:
        gate = dict(pass_=True, strict_pass=False, recomputed=anchor_test, anchor=anchor_test, delta=0.0,
                    source="SKIPPED (--timing-probe; output is not a result)")
    else:
        gate = None if args.no_e2_gate else (e2_gate(name, ckpt_id["sha256"])
                                             or own_gate(args.out_dir, name, ckpt_id["sha256"]))
        if gate is None:
            tg = time.time()
            gate = recompute_test_gate(net_ema, test_loader, anchor_test, entry["run_dir"])
            gate["source"] = f"recomputed here ({time.time() - tg:.0f}s)"
    acc, delta, gate_pass = float(gate["recomputed"]), float(gate["delta"]), bool(gate["pass_"])
    print(f"[{name}] gate: EMA S1 no-TTA test top1 {acc:.4f} vs events {anchor_test:.4f} "
          f"(|d|={delta:.1e}, strict={gate.get('strict_pass')}, {gate['source']}) -> "
          f"{'PASS' if gate_pass else 'FAIL'}; best_s1 match={id_best_s1}; "
          f"ckpt epoch {E} == best-val epoch: {id_epoch}", flush=True)
    if not gate_pass:
        return dict(name=name, status="GATE_FAIL", ckpt=ckpt_id, gates=dict(test_gate=gate))

    # ---- network under measurement ----
    queues = [M.ContrastiveQueue(cfg.queue_size, cfg.proj_dim, device="cpu") for _ in range(cfg.num_blocks)]
    for q, qs in zip(queues, ckpt["queues"]):
        q.z.copy_(qs["z"])
        q.y.copy_(qs["y"])
        q.ptr = int(qs["ptr"])
        q.full = bool(qs["full"])
    tau_base = None
    if args.mode == "raw_train":
        net = nets["model"]
        t_frac = (E - 1) / max(1, cfg.epochs_stage1 - 1)
        tau_base = cfg.gate_tau_start + t_frac * (cfg.gate_tau_end - cfg.gate_tau_start)
        net.set_router_tau(tau_base, depth_slope=cfg.gate_tau_depth_slope)
        net.train()
    else:
        net = net_ema          # EMA weights, eval mode, tau = gate_tau_start (as trained/evaluated)
    net_ema.eval()
    del ckpt

    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    L = cfg.num_blocks
    rec = [[] for _ in range(L)]
    n_done = 0
    tb = time.time()
    with torch.no_grad():
        for x1, x2, x3, labels in train_loader:
            if n_done >= args.n_batches:
                break
            B = x1.size(0)
            x_ni = x1[ni_indices(labels, B, "cpu")]
            t_pos1 = t_pos2 = t_nl = t_ni = None
            g_hist_pos1, g_hist_pos2, g_hist_nl, g_hist_ni = [], [], [], []
            for li in range(L):
                total, caches, diags = compute_block_terms(
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
            print(f"[{name}] batch {n_done}/{args.n_batches}: deepest sep_curr_nl="
                  f"{rec[L - 1][-1]['margin_curr_nl']:.4f} ({time.time() - tb:.0f}s)", flush=True)

    lm = C.layer_metrics_at_epoch(entry, E)
    per_block = []
    for li in range(L):
        stats = {}
        for k in DIAG_KEYS:
            v = np.array([d[k] for d in rec[li]], dtype=np.float64)
            stats[k] = dict(mean=float(v.mean()),
                            sd_batches=float(v.std(ddof=1)) if len(v) > 1 else None,
                            se=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None)
        row = lm.get(li, {})
        per_block.append(dict(
            block=li, k_neg=rec[li][0]["k_neg"],
            measured=stats,
            sep_curr_nl_per_batch=[d["margin_curr_nl"] for d in rec[li]],
            logged_at_ckpt_epoch={k: row.get(k) for k in LOGGED_KEYS},
        ))

    deep = per_block[-1]
    out = dict(
        name=name, status=("TIMING_ONLY" if args.timing_probe else "OK"), mode=args.mode,
        config=C.E1_CONFIG.get(name, entry["config"]), dataset=entry["dataset"], seed=entry["seed"],
        ckpt=ckpt_id, run_dir=str(entry["run_dir"]),
        trainer=dict(path=str(C.TRAINERS[entry["trainer"]]), sha256=C.sha256(C.TRAINERS[entry["trainer"]])),
        config_source=cfg_src, config_json_vs_ckpt_cfg_diff=cfg_diff,
        epoch_stage1=E, best_val_epoch_from_epoch_metrics=(int(best_row["epoch"]) if best_row else None),
        router_tau=("gate_tau_start=%.3f (EMA net; never set by the trainer)" % cfg.gate_tau_start
                    if args.mode == "ema_eval" else "set_router_tau(%.6f, slope %.3f) on raw net" % (tau_base, cfg.gate_tau_depth_slope)),
        weights=("net_ema, eval mode" if args.mode == "ema_eval" else "ckpt['model'], train mode"),
        protocol=dict(data_seed=args.data_seed, n_batches=n_done, batch_size=int(cfg.batch_size),
                      n_examples=int(n_done * cfg.batch_size), loader="trainer make_loaders_stage1 train "
                      "loader (triple-view training aug, shuffle, drop_last), workers=0",
                      hnm="trainer _choose_hard_negatives with the EMA block (use_ema_hnm=%s)" % bool(cfg.use_ema_hnm)),
        gates=dict(anchor_final_test_s1_top1_no_tta=anchor_test, recomputed_ema_test_top1_no_tta=acc,
                   delta=delta, gate_pass=bool(gate_pass), gate_strict=bool(gate.get("strict_pass")),
                   gate_source=gate["source"], test_gate=gate,
                   ckpt_best_s1=ckpt_best_s1,
                   run_end_best_s1=float(rend["best_s1"]), best_s1_match=bool(id_best_s1),
                   ckpt_epoch_is_best_val_epoch=bool(id_epoch), split_indices_match=True),
        accuracies_from_events=dict(
            s1_test_no_tta=fte.get("stage1_test_top1_no_tta"), s1_test_tta=fte.get("stage1_test_top1_tta"),
            s2_test_no_tta=fte.get("stage2_test_top1_no_tta"), s2_test_tta=fte.get("stage2_test_top1_tta"),
            best_s1_val_tta=rend.get("best_s1")),
        config_summary=dict(gamma_scale=float(cfg.gamma_scale),
                            gamma_gating_mode=getattr(cfg, "gamma_gating_mode", None),
                            gamma_kappa=getattr(cfg, "gamma_kappa", None), use_mgc=use_mgc,
                            use_sam=bool(cfg.use_sam), d_model=int(cfg.d_model), num_blocks=L,
                            n_experts=int(cfg.n_experts), moe_top_k=int(cfg.moe_top_k),
                            hard_negative_k=[int(cfg.hard_negative_k_first), int(cfg.hard_negative_k_last)],
                            block_curr_lambda=float(cfg.block_curr_lambda),
                            depth_order_lambda=float(cfg.depth_order_lambda), epochs_stage1=int(cfg.epochs_stage1)),
        deepest_block=dict(block=L - 1,
                           sep_curr_nl=deep["measured"]["margin_curr_nl"]["mean"],
                           sep_curr_nl_se=deep["measured"]["margin_curr_nl"]["se"],
                           logged_sep_curr_nl_at_ckpt_epoch=deep["logged_at_ckpt_epoch"]["sep_curr_nl"]),
        per_block=per_block,
        seconds=round(time.time() - t0, 1), seconds_measure=round(time.time() - tb, 1),
        torch=torch.__version__, threads=torch.get_num_threads(),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
    )
    if name == C.E1_VALIDATION and args.mode == "raw_train":
        se = out["deepest_block"]["sep_curr_nl_se"]
        d = out["deepest_block"]["sep_curr_nl"] - GRADSHARE_MGC_BLOCK3
        out["validation_vs_gradshare"] = dict(
            gradshare_json=str(GRADSHARE_JSON), gradshare_block3_margin_curr_nl=GRADSHARE_MGC_BLOCK3,
            delta=d, se_this_run=se,
            z_approx=(d / (math.sqrt(2.0) * se)) if se else None,
            criterion="|delta| <= 3*sqrt(2)*SE (the gradshare run drew an independent GPU RNG stream, so "
                      "agreement is statistical, not bitwise; its own SE is taken equal to ours)",
            pass_=bool(se and abs(d) <= 3 * math.sqrt(2.0) * se))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=list(C.E1_POINTS))
    ap.add_argument("--mode", choices=["ema_eval", "raw_train"], default="ema_eval")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--data-seed", type=int, default=20260725)
    ap.add_argument("--no-e2-gate", action="store_true", help="always recompute the test gate here")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--timing-probe", action="store_true",
                    help="skip the identity gate to time/memory-probe a few batches; refuses the real out dir")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.timing_probe:
        assert out_dir.resolve() != OUT_DIR.resolve(), "--timing-probe output must not go to the real out dir"
    n_fail = 0
    for name in args.names:
        path = out_dir / f"{name}__{args.mode}.json"
        if path.exists() and not args.force:
            prev = json.load(open(path))
            if prev.get("status") == "OK" and prev["protocol"]["n_batches"] == args.n_batches \
                    and prev["protocol"]["data_seed"] == args.data_seed:
                print(f"[{name}] done already ({path}); skipping", flush=True)
                continue
        print(f"\n===== {name} ({args.mode}) =====", flush=True)
        out = measure(name, args)
        C.write_json_atomic(path, out)
        if out["status"] == "TIMING_ONLY":
            print(f"[{name}] TIMING_ONLY: {out['seconds_measure']}s for {out['protocol']['n_batches']} batches "
                  f"({out['seconds_measure'] / max(1, out['protocol']['n_batches']):.1f}s/batch)", flush=True)
            continue
        if out["status"] != "OK":
            n_fail += 1
            print(f"[{name}] {out['status']} -> wrote {path}", flush=True)
            continue
        dp = out["deepest_block"]
        msg = (f"[{name}] deepest-block sep_curr_nl = {dp['sep_curr_nl']:.4f} +- {dp['sep_curr_nl_se']:.4f} (SE) "
               f"| logged at epoch {out['epoch_stage1']}: {dp['logged_sep_curr_nl_at_ckpt_epoch']} "
               f"| {out['seconds']}s")
        if "validation_vs_gradshare" in out:
            v = out["validation_vs_gradshare"]
            msg += f" | vs gradshare 5.2346: delta={v['delta']:+.4f}, z~{v['z_approx']:.2f}, pass={v['pass_']}"
        print(msg, flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
