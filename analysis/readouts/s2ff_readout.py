#!/usr/bin/env python3
"""S2-FF: a strict-Forward-Forward readout replacing the backprop Stage-2 probe.

Rebuttal question (FF-only readout): can the Stage-2 attentive head be replaced by a readout
trained with the FF objective alone?  This trains exactly that: ONE extra FF Hybrid Block
stacked on the frozen Stage-1 backbone, trained only with the local goodness objective
(softplus margin between the correct-label and a hard-mined wrong-label hypothesis). No
cross-entropy, no backprop into or across the frozen blocks — the frozen prefix is run under
torch.no_grad() and all its parameters have requires_grad=False, so locality is guaranteed by
construction.

Reported per checkpoint:
  s1_sum        : frozen goodness-sum over the original blocks          (the paper's Stage-1)
  s2ff_alone    : argmax over the new FF block's own goodness           (pure FF readout)
  s2ff_sum      : argmax over frozen-sum + new block goodness           (FF readout, cumulative)
and the S2 attentive-head anchor is taken from the published aggregates for comparison.

Usage:
  python3 s2ff_readout.py --ckpt <stage1_best.pt> --config <run>/config.json \\
      --events <run>/events.jsonl --out s2ff_<name>.json [--epochs 30]
"""
import argparse, importlib.util, json, os, sys
import torch
import torch.nn.functional as F


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("trainer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainer_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    # [CR] default: the camera-ready CIFAR-100 trainer (the rebuttal runs used mgc_c100_trainer.py,
    # which builds the identical network; see tests/test_equiv_cp_fair_cifar100.py)
    ap.add_argument("--trainer", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "..", "..", "trainers", "cp_fair_cifar100.py"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--events", default=None, help="run events.jsonl, for the S1 anchor gate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hnm_k", type=int, default=4, help="hard-negative candidates per example")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    T = load_trainer(args.trainer)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfgd = json.load(open(args.config))

    # ---- rebuild the trained net exactly as the run did ----
    cfg = T.FFConfig()
    for k, v in cfgd.items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, v)
            except Exception: pass
    net = T.FFHybridNet(cfg).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    state = ck.get("net_ema") or ck.get("model")          # eval numbers come from the EMA weights
    missing = net.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)                            # frozen backbone, hard guarantee

    # ---- the S2-FF readout: one more FF Hybrid Block, the only trainable module ----
    dpr = 0.0
    readout = T.FFHybridBlock(
        d_model=cfg.d_model, num_classes=cfg.num_classes, num_heads=cfg.num_heads,
        n_experts=cfg.n_experts, mlp_mult=cfg.mlp_mult, dropout=cfg.dropout,
        gate_tau=cfg.gate_tau_end, max_depth=cfg.num_blocks + 1, n_aspects=cfg.n_aspects,
        aspect_theta_inits=cfg.aspect_theta_inits, block_theta_init=cfg.block_theta_init,
        use_softmax_weights=cfg.use_softmax_weights, proj_dim=cfg.proj_dim, drop_path=dpr,
        moe_top_k=cfg.moe_top_k, use_rope=cfg.use_rope, rotary_emb=net.rotary_emb,
        recon_mult=cfg.recon_mult, proto_temp=cfg.proto_temp,
        label_embed_dim=cfg.label_embed_dim, use_mem=cfg.use_mem,
    ).to(dev)
    n_train = sum(p.numel() for p in readout.parameters() if p.requires_grad)
    print(f"[s2ff] trainable readout params: {n_train:,}")

    tr, val, test, _ = T.make_loaders_stage1(
        batch_size=args.batch_size, test_batch_size=args.batch_size,
        seed=cfg.seed, val_fraction=0.1, workers=args.workers)

    @torch.no_grad()
    def frozen_prefix(images, y):
        """Run the frozen backbone under label hypothesis y -> (tokens, summed goodness)."""
        tokens = net.embed(images)
        g_hist, total = [], torch.zeros(images.size(0), device=dev)
        for blk in net.blocks:
            g_prev = torch.stack(g_hist, dim=1).detach() if g_hist else None
            a, g_vec, _, _, _ = blk(tokens, y, g_prev)
            total = total + blk.aggregate_goodness(g_vec)
            g_hist.append(g_vec)
            tokens = T.l2_normalize_tokens(a)
        return tokens, total

    def readout_goodness(tokens, y):
        _, g_vec, _, _, _ = readout(tokens, y, None)       # gamma=0: purely local readout
        return readout.aggregate_goodness(g_vec)

    opt = torch.optim.AdamW(readout.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    C = cfg.num_classes

    for ep in range(1, args.epochs + 1):
        readout.train(); run_loss = n = 0.0
        for batch in tr:
            images = (batch[0] if not isinstance(batch[0], (list, tuple)) else batch[0][0]).to(dev, non_blocking=True)
            y = batch[-1].to(dev, non_blocking=True)
            B = images.size(0)
            with torch.no_grad():
                tok_pos, _ = frozen_prefix(images, y)
                # hard-negative mining among k random wrong labels, scored by the current readout
                cand = torch.randint(0, C, (B, args.hnm_k), device=dev)
                clash = cand == y.unsqueeze(1)
                cand[clash] = (cand[clash] + 1) % C
                best, best_g = cand[:, 0], None
                for j in range(args.hnm_k):
                    yj = cand[:, j]
                    tok_j, _ = frozen_prefix(images, yj)
                    gj = readout_goodness(tok_j, yj)
                    if best_g is None:
                        best_g, best = gj, yj
                    else:
                        take = gj > best_g
                        best_g = torch.where(take, gj, best_g)
                        best = torch.where(take, yj, best)
                tok_neg, _ = frozen_prefix(images, best)
            g_pos = readout_goodness(tok_pos, y)
            g_neg = readout_goodness(tok_neg, best)
            loss = F.softplus(-cfg.symba_alpha * (g_pos - g_neg)).mean()   # the FF criterion
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run_loss += loss.item() * B; n += B
        sched.step()
        if ep % 5 == 0 or ep == args.epochs:
            print(f"[s2ff] epoch {ep:3d}/{args.epochs}  loss {run_loss/max(n,1):.4f}", flush=True)

    # ---- evaluation: all C hypotheses through frozen prefix + readout ----
    @torch.no_grad()
    def evaluate(loader):
        readout.eval()
        c_s1 = c_alone = c_sum = tot = 0
        for batch in loader:
            images = (batch[0] if not isinstance(batch[0], (list, tuple)) else batch[0][0]).to(dev)
            y = batch[-1].to(dev)
            B = images.size(0)
            s1 = torch.zeros(B, C, device=dev); ro = torch.zeros(B, C, device=dev)
            for k in range(C):
                yk = torch.full((B,), k, dtype=torch.long, device=dev)
                tok, tot_g = frozen_prefix(images, yk)
                s1[:, k] = tot_g
                ro[:, k] = readout_goodness(tok, yk)
            c_s1 += (s1.argmax(1) == y).sum().item()
            c_alone += (ro.argmax(1) == y).sum().item()
            c_sum += ((s1 + ro).argmax(1) == y).sum().item()
            tot += B
        return c_s1 / tot, c_alone / tot, c_sum / tot

    s1_acc, alone_acc, sum_acc = evaluate(test)
    res = {
        "checkpoint": args.ckpt, "config": args.config, "seed": cfg.seed,
        "epochs": args.epochs, "hnm_k": args.hnm_k, "trainable_params": n_train,
        "s1_sum_no_tta": round(s1_acc * 100, 2),
        "s2ff_alone_no_tta": round(alone_acc * 100, 2),
        "s2ff_sum_no_tta": round(sum_acc * 100, 2),
        "note": ("S2-FF readout trained ONLY with the local FF goodness objective on a frozen "
                 "backbone; no cross-entropy and no backprop into or across the frozen blocks."),
    }
    if args.events and os.path.exists(args.events):
        anchor = None
        for line in open(args.events):
            e = json.loads(line)
            if e.get("event") == "final_test_eval":
                anchor = e.get("stage1_test_top1_no_tta")
        if anchor is not None:
            res["s1_anchor_from_events"] = round(anchor * 100, 2)
            res["s1_gate_delta_pp"] = round(abs(anchor * 100 - s1_acc * 100), 3)
            res["s1_gate_ok"] = bool(abs(anchor - s1_acc) < 0.01)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
