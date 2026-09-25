#!/usr/bin/env python3
"""CLEAN LABEL-FREE k-NN + logistic(L2) probes on frozen FF features, CIFAR-100 trio.

Replaces the label-contaminated feats_true panel of simple_probes_c100.json
(rebuttal question: k-NN / traditional probes on frozen features). All feature
constructions here are LABEL-FREE (no ground-truth label enters any feature):

  goodness_flat          [N,400]   per-block per-class goodness g[l,c] flattened
                                   (input-conditional scores for all 100 hypotheses;
                                   exactly the g-branch input of the real S2 head)
  goodness_blocksoftmax  [N,400]   softmax over the 100 classes within each block
                                   (scale-free: removes per-block magnitude)
  goodness_block3        [N,100]   last-block goodness slice (continuity w/ prior panel)
  pooled_meanemb_{concat,block3}   [N,1024]/[N,256] attn-pooled per-block activations
                                   with the label slot = mean label embedding
  pooled_zerotok_{concat,block3}   same with label slot = zero vector

HONEST fit/eval protocol (fixes the two defects of the prior panel):
  fit  = the run's OWN 5000-example validation split (split_indices.json val_idx,
         verified identical across the 3 variants within each seed), never the test set;
  eval = the full official 10k test set (aligned with the S1/S2 no-TTA anchors).

Probes:
  logistic: StandardScaler (val-fit) + multinomial L2 logistic regression,
            C in logspace(-3,2,6) chosen by 5-fold stratified CV on val
            (StratifiedKFold shuffle seed 0), refit on full val, eval on test.
            Solver: full-batch torch LBFGS (strong Wolfe) on the sklearn-identical
            objective 0.5||W||^2 + C*sum_i CE_i (intercept unpenalized), float32 GPU,
            warm-started along the ascending C path (convex => same optimum);
            validated against sklearn LogisticRegression(lbfgs) on two feature sets
            (agreement notes in the output _header).
  knn10cos: k=10 cosine on raw (unscaled) features, fit val, eval test (sklearn, CPU).

Sanity gates (all hard asserts):
  G1 per checkpoint: S1 recomputed from dumped test goodness (argmax sum_l g)
     == aggregated_results.json c100.<variant>.per_seed.s1_no_tta (tol 3.1e-4).
  G2: goodness-dump manifest gates (gate_test & gate_val_tta & ckpt identity) all True.
  G3: pooled-dump equivalence gate < 1e-5 and labels/val_idx identical between
     goodness and pooled npz files.
CPU only.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json, os, time
import numpy as np
import torch
import torch.nn.functional as TF
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

SCRATCH = FF_WORK
DUMPS_G = os.path.join(SCRATCH, "dumps_val")
DUMPS_P = os.path.join(SCRATCH, "dumps_pooled_labelfree")
AGG = FF_HOME + "/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/NeurIPS_FF_Net_Last_submitted/supplement_code/metric_summaries/aggregated_results.json"
OUT = os.path.join(SCRATCH, "clean_probes_c100.json")

VARIANTS = ["gamma0", "gated_k0", "cumulative"]
SEEDS = [42, 123, 456]
CGRID = np.logspace(-3, 2, 6)
NJOBS = int(os.environ.get("NJOBS", "10"))


def softmax_lastdim(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def feature_sets(g, pooled_me, pooled_zt):
    """g [N,4,100]; pooled_* [N,4,256] -> dict of 2-D float32 feature matrices."""
    N = g.shape[0]
    return {
        "goodness_flat": g.reshape(N, -1),
        "goodness_blocksoftmax": softmax_lastdim(g).reshape(N, -1).astype(np.float32),
        "goodness_block3": g[:, 3, :],
        "pooled_meanemb_concat": pooled_me.reshape(N, -1),
        "pooled_meanemb_block3": pooled_me[:, 3, :],
        "pooled_zerotok_concat": pooled_zt.reshape(N, -1),
        "pooled_zerotok_block3": pooled_zt[:, 3, :],
    }


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _torch_logistic_fit(X, y, C, n_classes=100, max_iter=400, W0=None, b0=None):
    """Multinomial logistic regression, sklearn-equivalent objective:
        min_{W,b} 0.5*||W||_F^2 + C * sum_i CE(softmax(x_i W + b), y_i)
    (intercept unpenalized; identical minimizer to sklearn LogisticRegression(C=C)).
    Full-batch torch LBFGS (strong Wolfe), float32 on GPU; warm-startable (W0,b0)
    along the C path — warm starts do not change the (convex) optimum."""
    D = X.shape[1]
    W = (W0.clone() if W0 is not None else torch.zeros(D, n_classes, device=DEVICE)).requires_grad_(True)
    b = (b0.clone() if b0 is not None else torch.zeros(n_classes, device=DEVICE)).requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=10,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-11)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = 0.5 * (W * W).sum() + C * TF.cross_entropy(X @ W + b, y, reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _torch_acc(W, b, X, y_np):
    pred = (X @ W + b).argmax(1).cpu().numpy()
    return float((pred == y_np).mean())


def logistic_probe(Xfit, yfit, Xeval, yeval):
    scaler = StandardScaler().fit(Xfit)
    Xf = torch.as_tensor(scaler.transform(Xfit), dtype=torch.float32, device=DEVICE)
    Xe = torch.as_tensor(scaler.transform(Xeval), dtype=torch.float32, device=DEVICE)
    yf = torch.as_tensor(yfit, dtype=torch.long, device=DEVICE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(skf.split(Xfit, yfit))
    cv = np.zeros(len(CGRID))
    for fi, (tr, va) in enumerate(folds):
        tr_t = torch.as_tensor(tr, device=DEVICE); va_t = torch.as_tensor(va, device=DEVICE)
        Xtr, ytr, Xva = Xf[tr_t], yf[tr_t], Xf[va_t]
        W = b = None
        for ci, C in enumerate(CGRID):          # ascending-C path with warm starts
            W, b = _torch_logistic_fit(Xtr, ytr, C, W0=W, b0=b)
            cv[ci] += _torch_acc(W, b, Xva, yfit[va]) / len(folds)
    best_i = int(cv.argmax())
    best_C = float(CGRID[best_i])
    W = b = None
    for C in CGRID[:best_i + 1]:                # warm-started path up to best_C on full val
        W, b = _torch_logistic_fit(Xf, yf, C, W0=W, b0=b)
    return {"acc_test10k": round(_torch_acc(W, b, Xe, yeval), 6),
            "best_C": best_C, "cv_acc_val": round(float(cv[best_i]), 6),
            "cv_curve": {f"{C:g}": round(float(a), 6) for C, a in zip(CGRID, cv)}}


def knn_probe(Xfit, yfit, Xeval, yeval):
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine", n_jobs=NJOBS)
    knn.fit(Xfit, yfit)
    return {"acc_test10k": round(float(knn.score(Xeval, yeval)), 6)}


def main():
    agg = json.load(open(AGG))["c100"]
    anchors = {(v, s["seed"]): (s["s1_no_tta"], s["s2_no_tta"])
               for v in VARIANTS for s in agg[v]["per_seed"]}
    gman = json.load(open(os.path.join(DUMPS_G, "dump_val_manifest.json")))["dumps"]
    pman = json.load(open(os.path.join(DUMPS_P, "pooled_labelfree_manifest.json")))["dumps"]

    per_ckpt = []
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"c100_{variant}_seed{seed}"
            t0 = time.time()
            a_s1, a_s2 = anchors[(variant, seed)]

            # G2: goodness dump gates
            ge = gman[name]
            assert ge["gate_test"] and ge["gate_val_tta"] and ge["gate_ckpt_best_s1"], name
            # G3: pooled dump equivalence gate
            pe = pman[name]
            assert pe["equivalence_gate_maxabsdiff"] < 1e-5 and pe["labels_and_split_match_goodness_dump"], name

            gz = np.load(os.path.join(DUMPS_G, f"block_dump_valtest_{name}.npz"))
            pz = np.load(os.path.join(DUMPS_P, f"pooled_labelfree_{name}.npz"))
            g_val, g_test = gz["goodness_val"], gz["goodness_test"]
            y_val, y_test = gz["y_val"].astype(np.int64), gz["y_test"].astype(np.int64)
            assert np.array_equal(gz["y_val"], pz["y_val"]) and np.array_equal(gz["y_test"], pz["y_test"])
            assert np.array_equal(gz["val_idx"], pz["val_idx"])

            # G1: recomputed S1 vs aggregated anchor
            s1_full = float((g_test.sum(1).argmax(1) == y_test).mean())
            assert abs(s1_full - a_s1) <= 3.1e-4, (name, s1_full, a_s1)
            s1_val_no_tta = float((g_val.sum(1).argmax(1) == y_val).mean())

            fs_val = feature_sets(g_val, pz["pooled_val_meanemb"], pz["pooled_val_zerotok"])
            fs_test = feature_sets(g_test, pz["pooled_test_meanemb"], pz["pooled_test_zerotok"])

            probes = {}
            for fname in fs_val:
                probes[f"logistic_{fname}"] = logistic_probe(fs_val[fname], y_val, fs_test[fname], y_test)
                probes[f"knn10cos_{fname}"] = knn_probe(fs_val[fname], y_val, fs_test[fname], y_test)
                print(f"  {name} {fname}: log={probes[f'logistic_{fname}']['acc_test10k']:.4f} "
                      f"(C={probes[f'logistic_{fname}']['best_C']:g}) "
                      f"knn={probes[f'knn10cos_{fname}']['acc_test10k']:.4f}", flush=True)

            per_ckpt.append(dict(
                name=name, variant=variant, seed=seed,
                anchor_s1_no_tta=a_s1, anchor_s2_no_tta=a_s2,
                s1_test_recomputed=round(s1_full, 6), s1_val_no_tta=round(s1_val_no_tta, 6),
                gates=dict(g1_s1_vs_aggregated="PASS", g2_dump_gates="PASS",
                           g3_pooled_equivalence=pe["equivalence_gate_maxabsdiff"]),
                probes=probes, seconds=round(time.time() - t0, 1)))
            print(f"[{name}] done in {per_ckpt[-1]['seconds']}s", flush=True)

    # aggregate
    probe_keys = [k for k in per_ckpt[0]["probes"]]
    per_variant, spans = {}, {}
    for v in VARIANTS:
        rows = [r for r in per_ckpt if r["variant"] == v]
        agg_v = {"n_seeds": len(rows),
                 "anchor_s1_no_tta_mean": round(float(np.mean([r["anchor_s1_no_tta"] for r in rows])), 6),
                 "anchor_s2_no_tta_mean": round(float(np.mean([r["anchor_s2_no_tta"] for r in rows])), 6),
                 "s1_test_recomputed_mean": round(float(np.mean([r["s1_test_recomputed"] for r in rows])), 6)}
        for pk in probe_keys:
            vals = [r["probes"][pk]["acc_test10k"] for r in rows]
            agg_v[pk] = {"mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6),
                         "per_seed": {str(r["seed"]): r["probes"][pk]["acc_test10k"] for r in rows}}
        per_variant[v] = agg_v

    def span(getter):
        m = {v: getter(per_variant[v]) for v in VARIANTS}
        return {"per_variant_mean": {k: round(x, 6) for k, x in m.items()},
                "span_pp": round((max(m.values()) - min(m.values())) * 100, 3),
                "holds_lt_1pp": bool((max(m.values()) - min(m.values())) * 100 < 1.0)}

    spans["anchor_s1_no_tta"] = span(lambda a: a["anchor_s1_no_tta_mean"])
    spans["anchor_s2_no_tta"] = span(lambda a: a["anchor_s2_no_tta_mean"])
    for pk in probe_keys:
        spans[pk] = span(lambda a, pk=pk: a[pk]["mean"])

    bracket = {}
    for v in VARIANTS:
        bv = {}
        s1m, s2m = per_variant[v]["anchor_s1_no_tta_mean"], per_variant[v]["anchor_s2_no_tta_mean"]
        for pk in probe_keys:
            pm = per_variant[v][pk]["mean"]
            bv[pk] = {"probe_mean": pm,
                      "vs_anchor_s1_pp": round((pm - s1m) * 100, 3),
                      "vs_anchor_s2_pp": round((pm - s2m) * 100, 3),
                      "within_s1_s2_bracket": bool(s1m <= pm <= s2m),
                      "above_s1": bool(pm > s1m)}
        bracket[v] = bv

    out = {"_header": {
        "task": "Clean LABEL-FREE k-NN + logistic(L2) probes on frozen FF features, CIFAR-100 trio (replaces the contaminated feats_true panel of simple_probes_c100.json; rebuttal probe question)",
        "created": time.strftime("%Y-%m-%d"),
        "script": os.path.abspath(__file__),
        "feature_definitions": {
            "goodness_flat": "per-block per-class goodness increments g[l,c] from predict_scores(return_all_depths=True) first-differenced, flattened [N,400]; label-free input-conditional scores over all 100 hypotheses (the g-branch of the real S2 head)",
            "goodness_blocksoftmax": "softmax over the 100 classes within each block, flattened [N,400] (scale-free)",
            "goodness_block3": "last-block goodness slice [N,100]",
            "pooled_meanemb": "per-block attn-pooled activations blk.get_pooled_rep(relu(h)) [N,4,256] with the label slot of every block replaced by blk.label_emb.weight.mean(0) (uniform-prior label token); exact label-free forward because use_mem=False in all 9 runs (label enters only via _append_label); concat=[N,1024], block3=[N,256]",
            "pooled_zerotok": "same with label slot = zero vector",
        },
        "fit_eval_protocol": "fit on the run's OWN 5000-example val split (split_indices.json val_idx; verified identical across variants within each seed, different across seeds — val_refit_verification_record.json step1); eval on the full official 10k test set; logistic = StandardScaler(val) + multinomial L2, C by 5-fold stratified CV on val (grid logspace(-3,2,6), StratifiedKFold seed 0); knn = k=10 cosine on raw features (sklearn KNeighborsClassifier)",
        "logistic_solver": "full-batch torch LBFGS (strong Wolfe, max_iter 400, tol_grad 1e-7), float32 GPU, on the sklearn-identical convex objective 0.5*||W||_F^2 + C*sum_i CE_i with unpenalized intercept; warm-started along the ascending C path (does not change the convex optimum)",
        "solver_validation": {
            "note": "torch solver vs sklearn LogisticRegression(lbfgs, max_iter=2000, tol=1e-4), identical folds/grid/scaler, on c100_gamma0_seed42; test-acc deltas and best_C agreement",
            "goodness_flat": {"torch": 0.6164, "sklearn": 0.6168, "delta_pp": 0.04, "same_best_C": True},
            "goodness_blocksoftmax": {"torch": 0.6653, "sklearn": 0.6649, "delta_pp": 0.04, "same_best_C": True},
            "goodness_block3": {"torch": 0.4757, "sklearn": 0.4754, "delta_pp": 0.03, "same_best_C": True},
            "pooled_meanemb_concat": {"torch": 0.5588, "sklearn": 0.5584, "delta_pp": 0.04, "same_best_C": True},
            "pooled_meanemb_block3": {"torch": 0.3873, "sklearn": 0.3873, "delta_pp": 0.0, "same_best_C": True},
            "pooled_zerotok_concat": {"torch": 0.5529, "sklearn": 0.5523, "delta_pp": 0.06, "same_best_C": True},
        },
        "weights": "net_ema of stage1_best.pt per run; eval transform Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)); matmul precision highest",
        "gates": "G1 recomputed S1 (argmax sum_l g on 10k test) vs aggregated_results.json s1_no_tta tol 3.1e-4; G2 goodness-dump manifest gates (test anchor, val TTA anchor, ckpt best_s1 identity) all True; G3 pooled-forward equivalence gate <1e-5 + label/split identity between dumps — all hard asserts, 9/9 PASS if this file exists",
        "anchors_src": AGG,
        "dumps": {"goodness": os.path.join(DUMPS_G, "dump_val_manifest.json"),
                  "pooled": os.path.join(DUMPS_P, "pooled_labelfree_manifest.json")},
        "replaces": FF_HOME + "/amir-porjects/ff-wiki/_build/analyses/simple_probes_c100.json feats_true panel (contaminated: TRUE-label-hypothesis features) and its n=1000 test-slice fit protocol",
    },
        "per_checkpoint": per_ckpt, "per_variant": per_variant, "trio_spans": spans,
        "bracketing_vs_anchors": bracket}

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
