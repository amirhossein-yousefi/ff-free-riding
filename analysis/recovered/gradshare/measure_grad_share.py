#!/usr/bin/env python3
"""GRADIENT-SHARE MEASUREMENT at convergence (rebuttal: gradient-share analysis).

For each converged Stage-1 checkpoint, mirror the trainer's own per-block Stage-1
loss (importing the training module for every reusable function: block forward,
_choose_hard_negatives, _sum_prev_goodness, symba_loss, ff_margin_loss,
pairwise_rank_loss, supcon_generic, l2_normalize_tokens, ContrastiveQueue,
make_loaders_stage1, evaluate_ff) and, on ~20 training-transform batches, compute
per loss term t and per block d:
    g_t = autograd.grad(term_t, params_d)   (same forward graph for all terms)
    share_t = ||g_t|| / ||g_total||,  cos_t = <g_t, g_total>/(||g_t||*||g_total||)
params_d = exactly the trainer's per-block optimizer partition
(block.parameters() + net.embed.parameters() for block 0).

Weights differentiated = ckpt['model'] (the raw training weights the per-block
AdamW/SAM optimizers act on). EMA weights are used only for (a) the accuracy
gate against events.jsonl final_test_eval and (b) EMA-HNM, exactly as in training.
Queues are restored from the checkpoint (the trainer's own resume path).
Router tau: net gets set_router_tau(tau at ckpt epoch) as the training loop did;
net_ema keeps gate_tau_start (the trainer never calls set_router_tau on the EMA).

Vector caveat reported in the JSON: gradients of terms sum exactly to the total
gradient (linearity; checked per block-batch), but their NORMS do not.

Single GPU process, blocks/batches strictly sequential. No optimizer stepping.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRATCH = FF_WORK
WORKDIR = os.path.join(SCRATCH, "gradshare")
OUT_JSON = os.path.join(SCRATCH, "gradient_share_results.json")
ROOT = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison"

os.environ.setdefault("FF_CKPT_DIR", os.path.join(WORKDIR, "checkpoints_dummy"))

ARMS = {
    "c10_d128_gamma07_baseline_constant_seed42": dict(
        module_dir=os.path.join(ROOT, "revised_training"),
        module="basic_fair_revised_valfix_cp_fair_hardness_gated_fixed",
        run_dir=os.path.join(ROOT, "basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/runs/ff_cifar10_baseline_constant_L4_D128_bs512_seed42_20260313_232043"),
        ckpt=os.path.join(ROOT, "basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/checkpoints/ff_cifar10_baseline_constant_L4_D128_P2_seed42_stage1_best.pt"),
    ),
    "c10_d128_gamma0_no_all_collab_seed42": dict(
        module_dir=os.path.join(ROOT, "revised_training"),
        module="basic_fair_revised_valfix_cp_fair_hardness_gated_fixed",
        run_dir=os.path.join(ROOT, "basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/runs/ff_cifar10_no_all_collab_L4_D128_bs512_seed42_20260315_124117"),
        ckpt=os.path.join(ROOT, "basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/checkpoints/ff_cifar10_no_all_collab_L4_D128_P2_seed42_stage1_best.pt"),
    ),
    "c10_d128_mgc_c1_seed42": dict(
        module_dir=os.path.join(ROOT, "mgc_rebuttal"),
        module="mgc_trainer",
        run_dir=os.path.join(ROOT, "mgc_rebuttal/runs/ff_cifar10_mgc_c1_L4_D128_bs512_seed42_20260723_223516"),
        ckpt=os.path.join(ROOT, "mgc_rebuttal/checkpoints/ff_cifar10_mgc_c1_L4_D128_P2_seed42_stage1_best.pt"),
    ),
    # D256 pair (secondary): CP-FAIR headline arm (gamma=0.7) + fixed-trainer gamma=0 re-run.
    # NOTE: D256 gamma0 seed42 (ablations/no_all_collab, H6) kept NO checkpoint -> seed123 used.
    "c10_d256_gamma07_cp_fair_seed42": dict(
        module_dir=os.path.join(ROOT, "revised_training"),
        module="basic_fair_revised_valfix_cp_fair_fixed",
        run_dir=os.path.join(ROOT, "basic_fair_revised_valfix_cp_fair/runs/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035"),
        ckpt=os.path.join(ROOT, "basic_fair_revised_valfix_cp_fair/checkpoints/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_P2_seed42_stage1_best.pt"),
    ),
    "c10_d256_gamma0_seed123": dict(
        module_dir=os.path.join(ROOT, "revised_training"),
        module="basic_fair_revised_valfix_cp_fair_fixed",
        run_dir=os.path.join(ROOT, "revised_training/out/gamma0_seed123/runs/ff_cifar10_cifar10_gamma0_seed123_L4_D256_bs512_seed123_20260417_235110"),
        ckpt=os.path.join(ROOT, "revised_training/out/gamma0_seed123/checkpoints/ff_cifar10_cifar10_gamma0_seed123_L4_D256_P2_seed123_stage1_best.pt"),
    ),
}

TERM_ORDER = ["aspects", "block_cum", "block_curr", "mgc_local", "depth",
              "moe_bal", "moe_z", "supcon", "recon"]


def load_module(module_dir, module_name):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    return __import__(module_name)


def build_cfg(M, config_json_path):
    cfg = M.FFConfig()
    persisted = json.load(open(config_json_path))
    persisted = persisted.get("config", persisted)
    for k, v in persisted.items():
        if k == "device":
            continue
        setattr(cfg, k, v)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg, persisted


def read_anchors(run_dir):
    fte = rend = None
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("event") == "final_test_eval":
                fte = ev
            elif ev.get("event") == "run_end":
                rend = ev
    return fte, rend


def read_layer_metrics_at_epoch(run_dir, epoch):
    rows = {}
    with open(os.path.join(run_dir, "layer_metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("stage") == 1 and r.get("epoch") == epoch:
                rows[int(r["layer"])] = r
    return rows


def flat_grad(scalar, params):
    gs = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(gs, params)
    ])


def compute_block_terms(M, cfg, net, net_ema, li, x1, x2, x3, x_ni, labels,
                        t_pos1, t_pos2, t_nl, t_ni,
                        g_hist_pos1, g_hist_pos2, g_hist_nl, g_hist_ni,
                        queues, use_mgc):
    """Faithful mirror of the trainer's forward_block() (constant-gamma path;
    MGC term included when cfg.use_mgc). Returns terms, total, caches, diags."""
    block = net.blocks[li]
    B = x1.size(0)
    assert getattr(cfg, "gamma_gating_mode", "constant") == "constant"

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

    # constant gamma (all measured arms use gamma_gating_mode == "constant")
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

    # ---- reconstruction ----
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

    # ---- weights (epoch-dependent pieces are inert here: moe_aux_ramp_epochs=0) ----
    moe_bal_w = float(getattr(cfg, "moe_balance_coef", 0.0))
    moe_z_w = float(getattr(cfg, "moe_zloss_coef", 0.0))
    block_curr_lambda_eff = float(cfg.block_curr_lambda)
    slope = float(getattr(cfg, "block_curr_depth_slope", 0.0))
    if slope != 0.0:
        block_curr_lambda_eff = float(cfg.block_curr_lambda) * (
            1.0 + slope * (li / max(1, cfg.num_blocks - 1)))
    if use_mgc:
        block_curr_lambda_eff = 0.0

    # total, constructed exactly as the trainer does
    total = (loss_aspects
             + cfg.block_lambda * loss_block
             + block_curr_lambda_eff * loss_block_curr
             + (cfg.block_lambda * loss_mgc_local if (use_mgc and loss_mgc_local is not None) else 0.0)
             + loss_depth
             + moe_bal_w * loss_bal
             + moe_z_w * loss_z
             + cfg.contrastive_lambda * loss_con
             + loss_recon)

    terms = {
        "aspects": loss_aspects,
        "block_cum": cfg.block_lambda * loss_block,
        "block_curr": (block_curr_lambda_eff * loss_block_curr) if block_curr_lambda_eff != 0.0 else None,
        "mgc_local": (cfg.block_lambda * loss_mgc_local) if (use_mgc and loss_mgc_local is not None) else None,
        "depth": loss_depth if torch.is_tensor(loss_depth) else None,
        "moe_bal": moe_bal_w * loss_bal if moe_bal_w != 0.0 else None,
        "moe_z": moe_z_w * loss_z if moe_z_w != 0.0 else None,
        "supcon": cfg.contrastive_lambda * loss_con,
        "recon": loss_recon if torch.is_tensor(loss_recon) else None,
    }

    with torch.no_grad():
        diags = dict(
            loss_block_raw=float(loss_block.item()),
            loss_aspects_raw=float(loss_aspects.item()),
            loss_block_curr_raw=float(loss_block_curr.item()) if torch.is_tensor(loss_block_curr) else 0.0,
            loss_con_raw=float(loss_con.item()),
            loss_recon_raw=float(loss_recon.item()) if torch.is_tensor(loss_recon) else 0.0,
            loss_depth_w=float(loss_depth.item()) if torch.is_tensor(loss_depth) else 0.0,
            loss_total=float(total.item()),
            margin_blk_nl=float((gpos_blk - gnl_blk).mean().item()),
            margin_curr_nl=float((gpos_curr - gnl_curr).mean().item()),
            gamma_pos=float(gamma_pos1.mean().item()),
            sat_factor_nl=float(torch.sigmoid(-cfg.symba_alpha * (gpos_blk - gnl_blk)).mean().item()),
            sat_factor_curr_nl=float(torch.sigmoid(-cfg.symba_alpha * (gpos_curr - gnl_curr)).mean().item()),
            mgc_lam_mean=mgc_lam_mean,
            k_neg=k_neg,
        )

    caches = dict(a_pos1=a_pos1, a_pos2=a_pos2, a_nl=a_nl, a_ni=a_ni,
                  gpos1=gpos1, gpos2=gpos2, gnl=gnl, gni=gni, z1=z1, z2=z2)
    return terms, total, caches, diags


def measure_arm(name, arm, args, device):
    t0 = time.time()
    M = load_module(arm["module_dir"], arm["module"])
    cfg, persisted = build_cfg(M, os.path.join(arm["run_dir"], "config.json"))
    use_mgc = bool(getattr(cfg, "use_mgc", False))
    fte, rend = read_anchors(arm["run_dir"])
    anchor_test = float(fte["stage1_test_top1_no_tta"])

    # loaders from the training module itself (training-transform triple views)
    train_loader, _, test_loader, split_info = M.make_loaders_stage1(
        cfg.batch_size, cfg.test_batch_size, seed=cfg.seed,
        val_fraction=cfg.val_fraction, workers=args.workers)
    run_split = json.load(open(os.path.join(arm["run_dir"], "split_indices.json")))
    assert split_info["val_idx"] == run_split["val_idx"], "val split mismatch"
    assert split_info["train_idx"] == run_split["train_idx"], "train split mismatch"

    ckpt = torch.load(arm["ckpt"], map_location="cpu", weights_only=False)
    assert abs(float(ckpt["best_s1"]) - float(rend["best_s1"])) < 1e-9, \
        f"ckpt best_s1 {ckpt['best_s1']} != run_end {rend['best_s1']}"
    E = int(ckpt["epoch_stage1"])

    net = M.FFHybridNet(cfg).to(device)
    net.load_state_dict(ckpt["model"], strict=True)
    net_ema = M.FFHybridNet(cfg).to(device)
    net_ema.load_state_dict(ckpt["net_ema"], strict=True)
    for p in net_ema.parameters():
        p.requires_grad_(False)

    # ---- GATE: recompute EMA S1 no-TTA test top-1, must match events.jsonl ----
    acc = M.evaluate_ff(net_ema, test_loader, device, tta=False, topk=(1,))[1]
    delta = abs(acc - anchor_test)
    gate_pass = delta <= args.tol
    print(f"[{name}] gate: recomputed EMA test top1 {acc:.4f} vs anchor {anchor_test:.4f} "
          f"(delta {delta:.2e}) -> {'PASS' if gate_pass else 'FAIL'}", flush=True)
    assert gate_pass, f"accuracy gate failed for {name}"

    # ---- queues restored from checkpoint (trainer's own resume behavior) ----
    queues = [M.ContrastiveQueue(cfg.queue_size, cfg.proj_dim, device=device)
              for _ in range(cfg.num_blocks)]
    for q, qs in zip(queues, ckpt["queues"]):
        q.z.copy_(qs["z"].to(device))
        q.y.copy_(qs["y"].to(device))
        q.ptr = int(qs["ptr"])
        q.full = bool(qs["full"])

    # ---- router tau exactly as the training loop set it at epoch E ----
    t_frac = (E - 1) / max(1, cfg.epochs_stage1 - 1)
    tau_base = cfg.gate_tau_start + t_frac * (cfg.gate_tau_end - cfg.gate_tau_start)
    net.set_router_tau(tau_base, depth_slope=cfg.gate_tau_depth_slope)
    # net_ema: trainer never calls set_router_tau on the EMA net -> stays at gate_tau_start

    net.train()
    net_ema.eval()

    # per-block optimizer param partition (exactly main()'s construction)
    params_per_block = []
    for li, block in enumerate(net.blocks):
        params = [p for p in block.parameters() if p.requires_grad]
        if li == 0:
            params = params + [p for p in net.embed.parameters() if p.requires_grad]
        params_per_block.append(params)

    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    L = cfg.num_blocks
    acc_rec = [{t: dict(share=[], cos=[], norm=[]) for t in TERM_ORDER} for _ in range(L)]
    tot_rec = [dict(norm=[], lin_rel=[], disc_all_share=[]) for _ in range(L)]
    diag_rec = [[] for _ in range(L)]

    n_done = 0
    for bi, (x1, x2, x3, labels) in enumerate(train_loader):
        if n_done >= args.n_batches:
            break
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        labels = labels.to(device)
        B = x1.size(0)

        # NI sampling — verbatim trainer logic
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
        x_ni = x1[indices_ni]

        t_pos1 = t_pos2 = t_nl = t_ni = None
        g_hist_pos1, g_hist_pos2, g_hist_nl, g_hist_ni = [], [], [], []

        for li in range(L):
            terms, total, caches, diags = compute_block_terms(
                M, cfg, net, net_ema, li, x1, x2, x3, x_ni, labels,
                t_pos1, t_pos2, t_nl, t_ni,
                g_hist_pos1, g_hist_pos2, g_hist_nl, g_hist_ni,
                queues, use_mgc)
            params = params_per_block[li]

            flats = {}
            for tname in TERM_ORDER:
                tv = terms.get(tname)
                if tv is not None and torch.is_tensor(tv) and tv.requires_grad:
                    flats[tname] = flat_grad(tv, params)
                else:
                    flats[tname] = None
            g_tot = flat_grad(total, params)
            n_tot = float(g_tot.norm().item())

            g_sum = None
            for v in flats.values():
                if v is not None:
                    g_sum = v.clone() if g_sum is None else g_sum + v
            lin_rel = float((g_sum - g_tot).norm().item() / (n_tot + 1e-30))

            for tname in TERM_ORDER:
                v = flats[tname]
                if v is None:
                    acc_rec[li][tname]["share"].append(0.0)
                    acc_rec[li][tname]["cos"].append(float("nan"))
                    acc_rec[li][tname]["norm"].append(0.0)
                else:
                    nv = float(v.norm().item())
                    cv = float((v @ g_tot).item() / (nv * n_tot + 1e-30))
                    acc_rec[li][tname]["share"].append(nv / (n_tot + 1e-30))
                    acc_rec[li][tname]["cos"].append(cv)
                    acc_rec[li][tname]["norm"].append(nv)

            # combined discrimination pressure at this block (cum + curr + mgc)
            g_disc = None
            for tname in ("block_cum", "block_curr", "mgc_local"):
                v = flats[tname]
                if v is not None:
                    g_disc = v.clone() if g_disc is None else g_disc + v
            disc_share = float(g_disc.norm().item() / (n_tot + 1e-30)) if g_disc is not None else 0.0

            tot_rec[li]["norm"].append(n_tot)
            tot_rec[li]["lin_rel"].append(lin_rel)
            tot_rec[li]["disc_all_share"].append(disc_share)
            diag_rec[li].append(diags)

            # cache for next block + queue update, exactly as the trainer
            t_pos1 = M.l2_normalize_tokens(caches["a_pos1"].detach())
            t_pos2 = M.l2_normalize_tokens(caches["a_pos2"].detach())
            t_nl = M.l2_normalize_tokens(caches["a_nl"].detach())
            t_ni = M.l2_normalize_tokens(caches["a_ni"].detach())
            g_hist_pos1.append(caches["gpos1"].detach())
            g_hist_pos2.append(caches["gpos2"].detach())
            g_hist_nl.append(caches["gnl"].detach())
            g_hist_ni.append(caches["gni"].detach())
            with torch.no_grad():
                queues[li].enqueue(caches["z1"].detach(), labels.detach())
                queues[li].enqueue(caches["z2"].detach(), labels.detach())

            del flats, g_tot, g_sum, g_disc, terms, total, caches
        n_done += 1
        if (n_done % 5) == 0:
            print(f"[{name}] batch {n_done}/{args.n_batches} done "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate ----
    lm_run = read_layer_metrics_at_epoch(arm["run_dir"], E)
    per_block = []
    for li in range(L):
        tstats = {}
        for tname in TERM_ORDER:
            sh = np.array(acc_rec[li][tname]["share"], dtype=np.float64)
            co = np.array(acc_rec[li][tname]["cos"], dtype=np.float64)
            no = np.array(acc_rec[li][tname]["norm"], dtype=np.float64)
            present = bool(np.any(no > 0))
            tstats[tname] = dict(
                present=present,
                share_mean=float(sh.mean()), share_std=float(sh.std(ddof=1)) if len(sh) > 1 else 0.0,
                cos_mean=(float(np.nanmean(co)) if present else None),
                grad_norm_mean=float(no.mean()),
            )
        dmean = {k: float(np.mean([d[k] for d in diag_rec[li]]))
                 for k in ("loss_block_raw", "loss_aspects_raw", "loss_block_curr_raw",
                           "loss_con_raw", "loss_recon_raw", "loss_depth_w", "loss_total",
                           "margin_blk_nl", "margin_curr_nl", "gamma_pos",
                           "sat_factor_nl", "sat_factor_curr_nl")}
        if use_mgc and li > 0:
            dmean["mgc_lam_mean"] = float(np.mean([d["mgc_lam_mean"] for d in diag_rec[li]
                                                   if d["mgc_lam_mean"] is not None]))
        run_row = lm_run.get(li, {})
        crosscheck = {
            "run_epochE_loss_block": run_row.get("loss_block"),
            "meas_loss_block": dmean["loss_block_raw"],
            "run_epochE_loss_total": run_row.get("loss_total"),
            "meas_loss_total": dmean["loss_total"],
            "run_epochE_gamma_pos": run_row.get("gamma_pos"),
            "meas_gamma_pos": dmean["gamma_pos"],
            "run_epochE_margin_blk_nl": run_row.get("margin_blk_nl"),
            "meas_margin_blk_nl": dmean["margin_blk_nl"],
        }
        per_block.append(dict(
            block=li,
            n_params=int(sum(p.numel() for p in params_per_block[li])),
            total_grad_norm_mean=float(np.mean(tot_rec[li]["norm"])),
            total_grad_norm_std=float(np.std(tot_rec[li]["norm"], ddof=1)),
            linearity_rel_err_max=float(np.max(tot_rec[li]["lin_rel"])),
            disc_all_share_mean=float(np.mean(tot_rec[li]["disc_all_share"])),
            terms=tstats,
            loss_value_means=dmean,
            layer_metrics_crosscheck=crosscheck,
        ))

    out = dict(
        arm=name,
        ckpt=arm["ckpt"],
        run_dir=arm["run_dir"],
        trainer_module=os.path.join(arm["module_dir"], arm["module"] + ".py"),
        weights_differentiated="ckpt['model'] (raw Stage-1 training weights at best epoch)",
        epoch_stage1=E,
        router_tau_base=float(tau_base),
        config_summary=dict(
            gamma_scale=float(cfg.gamma_scale),
            gamma_gating_mode=cfg.gamma_gating_mode,
            use_mgc=use_mgc,
            mgc_c0=(float(getattr(cfg, "mgc_c0", 0.0)) if use_mgc else None),
            block_lambda=float(cfg.block_lambda),
            block_curr_lambda=float(cfg.block_curr_lambda),
            block_curr_depth_slope=float(cfg.block_curr_depth_slope),
            depth_order_lambda=float(cfg.depth_order_lambda),
            contrastive_lambda=float(cfg.contrastive_lambda),
            recon_lambda=float(cfg.recon_lambda),
            moe_balance_coef=float(cfg.moe_balance_coef),
            moe_zloss_coef=float(cfg.moe_zloss_coef),
            use_sam=bool(cfg.use_sam),
            use_ema_hnm=bool(cfg.use_ema_hnm),
            batch_size=int(cfg.batch_size),
            d_model=int(cfg.d_model),
            seed=int(cfg.seed),
        ),
        gates=dict(
            anchor_final_test_s1_top1_no_tta=anchor_test,
            recomputed_ema_test_top1_no_tta=float(acc),
            delta=float(delta),
            tol=args.tol,
            gate_accuracy_pass=bool(gate_pass),
            ckpt_best_s1=float(ckpt["best_s1"]),
            run_end_best_s1=float(rend["best_s1"]),
            split_indices_match=True,
        ),
        n_batches=int(n_done),
        batch_size=int(cfg.batch_size),
        per_block=per_block,
        seconds=round(time.time() - t0, 1),
    )

    del net, net_ema, queues, ckpt
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS.keys()))
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)  # 0 = in-process loading; workers>0 deadlocked at arm transition (fork-after-CUDA, a hazard the trainer itself documents)
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--data-seed", type=int, default=20260725)
    args = ap.parse_args()

    os.chdir(WORKDIR)  # data_c10 symlink lives here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if os.path.exists(OUT_JSON):
        results = json.load(open(OUT_JSON))
    else:
        results = {"_header": {}, "arms": {}}

    results["_header"] = {
        "description": ("Per-term gradient share of the Stage-1 per-block loss at convergence "
                        "(rebuttal gradient-share analysis). For each block d and loss term t: share = "
                        "||grad_t|| / ||grad_total|| over the block's own optimizer parameters, "
                        "plus cos(grad_t, grad_total). All terms differentiated from the SAME "
                        "training-mode forward (same dropout/droppath/routing/HNM draws)."),
        "created": time.strftime("%Y-%m-%d"),
        "script": os.path.abspath(__file__),
        "method": {
            "batches": f"{args.n_batches} training-transform batches (triple-view aug, HNM NL negatives + NI negatives), data_seed={args.data_seed}",
            "weights": "raw net = ckpt['model'] (what the per-block optimizers train); net_ema used only for the accuracy gate and EMA-HNM, as in training",
            "queues": "SupCon queues restored from the checkpoint (trainer resume path), updated during measurement as in training",
            "router_tau": "net.set_router_tau(tau at ckpt epoch, depth_slope) as the training loop did; EMA net keeps gate_tau_start (trainer never sets tau on the EMA)",
            "no_step": "no optimizer stepping, no EMA update; within a batch, later blocks see caches from pre-update weights (training would have stepped earlier blocks first)",
            "terms": {
                "aspects": "sum_k aspect_lambdas[k] * (margin|rank) losses on per-aspect goodness",
                "block_cum": "block_lambda * SYMBA on CUMULATIVE goodness g_curr + gamma*sum_prev (the discrimination term of Theorem 1)",
                "block_curr": "block_curr_lambda_eff * current-block-only SYMBA (residual-weighted for d>0)",
                "mgc_local": "block_lambda * MGC compensated local term (MGC arm only)",
                "depth": "depth-ordering softplus margins (weighted)",
                "moe_bal": "moe_balance_coef * Switch load-balance",
                "moe_z": "moe_zloss_coef * ST-MoE router z-loss",
                "supcon": "contrastive_lambda * SupCon (with queue negatives)",
                "recon": "recon_lambda * token reconstruction MSE",
            },
            "vector_caveat": ("Gradients of terms sum exactly to the total gradient (linearity; "
                              "verified per block-batch as linearity_rel_err_max), but their NORMS "
                              "do not sum to ||grad_total||; shares can exceed 1 or sum above 1 when "
                              "terms partially cancel. cos(grad_t, grad_total) is reported for direction."),
            "caveats": [
                "MGC arm trained with SAM (use_sam=true): measured gradient is the first (unperturbed) SAM pass, i.e. grad of the loss at the checkpoint weights; the D128 gated arms trained with plain AdamW (use_sam=false).",
                "Gradient norms are pre-clipping (trainer clips at grad_clip=1.0 before stepping in the non-SAM branch); shares are scale-invariant to clipping.",
                "The D128 baseline_constant / no_all_collab checkpoints were trained on an A100 (colab) by the gated ablation runner; loss code measured here is the FF-purity-FIXED trainer (revised_training/..._fixed.py), the canonical loss definition (wiki H1).",
            ],
        },
        "arms_measured": sorted(set(list(results.get("arms", {}).keys()) + args.arms)),
    }

    for name in args.arms:
        print(f"\n===== {name} =====", flush=True)
        out = measure_arm(name, ARMS[name], args, device)
        results["arms"][name] = out
        json.dump(results, open(OUT_JSON, "w"), indent=1)
        print(f"[{name}] done in {out['seconds']}s; block_cum shares by block: "
              + ", ".join(f"d{pb['block']}={pb['terms']['block_cum']['share_mean']:.4f}"
                          for pb in out["per_block"]), flush=True)

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
