#!/usr/bin/env python3
"""Readout-spectrum refit on the TRUE VALIDATION SPLIT (CIFAR-100 trio, 9 checkpoints).

Fit R1/R2/R3 on each run's own 5000-example val split (no TTA), evaluate on the FULL
official 10k test set. R0 (plain sum) must reproduce the events.jsonl anchors.

Ladder (protocol of readout_spectrum_results.json _header, reimplemented):
  R0: plain sum over blocks
  R1: per-depth scalar weights w_d (unconstrained), NLL fit, init w=1
  R2: per-depth inverse temperature invT_d = exp(rho_d) > 0, logits = sum_d g_d * invT_d,
      init rho=0; reported as T_d = 1/invT_d
  R3: margin-gated w_d(x) = sigmoid(a_d * m_d(x) + b_d); m_d = (top1-top2 of g_d(x,.)),
      standardized by FIT-split per-depth std (ddof=0); logits = s3 * sum_d w_d(x) g_d(x,.)
      with FIXED s3 = 2*max(invT_R2); warm start a=0, b=logit(invT_R2/s3)
Fit: full-batch Adam(0.05, 400 steps) + LBFGS polish (strong Wolfe), float64, torch CPU,
torch.manual_seed(0). Selection: best_by_fit = argmax fit accuracy among {R1,R2,R3}
(ties -> earliest rung); best_by_eval analogous on eval accuracy.
Bootstrap: 5000 folds, np.random.default_rng(0); shared index matrices (10k per-checkpoint,
30k pooled-per-variant); interaction folds reuse the per-checkpoint fold means.
NOTE: the bootstrap RNG stream necessarily differs from the earlier (lost) script, so CI
endpoints can shift by ~0.01-0.03pp; all point estimates are deterministic.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRATCH = FF_WORK
DUMP_MANIFEST = os.path.join(SCRATCH, "dumps_val", "dump_val_manifest.json")
OUT = os.path.join(SCRATCH, "val_refit_readout_results.json")
PREV = FF_HOME + "/amir-porjects/ff-wiki/_build/analyses/readout_spectrum_results.json"
N_BOOT, TOL = 5000, 5e-4
VARIANTS = ["gamma0", "gated_k0", "cumulative"]
SEEDS = [42, 123, 456]
RUNGS = ["R1", "R2", "R3"]

torch.set_num_threads(8)


def nll_and_acc(logits, y):
    nll = float(F.cross_entropy(logits, y).item())
    acc = float((logits.argmax(1) == y).float().mean().item())
    return nll, acc


def optimize(params, loss_fn, adam_steps=400, adam_lr=0.05):
    opt = torch.optim.Adam(params, lr=adam_lr)
    for _ in range(adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    lb = torch.optim.LBFGS(params, lr=1.0, max_iter=500, history_size=100,
                           tolerance_grad=1e-12, tolerance_change=1e-14,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    lb.step(closure)
    with torch.no_grad():
        return float(loss_fn().item())


def fit_R1(gf, yf):
    w = torch.ones(gf.shape[1], dtype=torch.float64, requires_grad=True)
    fn = lambda: F.cross_entropy(torch.einsum("nlc,l->nc", gf, w), yf)
    optimize([w], fn)
    return w.detach()


def fit_R2(gf, yf):
    rho = torch.zeros(gf.shape[1], dtype=torch.float64, requires_grad=True)
    fn = lambda: F.cross_entropy(torch.einsum("nlc,l->nc", gf, torch.exp(rho)), yf)
    optimize([rho], fn)
    return torch.exp(rho).detach()


def margins(g):
    top2 = torch.topk(g, 2, dim=2).values  # [n,l,2]
    return top2[:, :, 0] - top2[:, :, 1]


def fit_R3(gf, yf, m_fit_std, invT_R2):
    s3 = 2.0 * float(invT_R2.max().item())
    mf = margins(gf) / m_fit_std  # [n,l]
    p = torch.clamp(invT_R2 / s3, 1e-9, 1 - 1e-9)
    b = torch.log(p / (1 - p)).clone().requires_grad_(True)
    a = torch.zeros_like(invT_R2, requires_grad=True)

    def fn():
        w = torch.sigmoid(a.unsqueeze(0) * mf + b.unsqueeze(0))  # [n,l]
        return F.cross_entropy(s3 * torch.einsum("nlc,nl->nc", gf, w), yf)

    optimize([a, b], fn)
    return a.detach(), b.detach(), s3


def r3_logits(g, m_std, a, b, s3):
    m = margins(g) / m_std
    w = torch.sigmoid(a.unsqueeze(0) * m + b.unsqueeze(0))
    return s3 * torch.einsum("nlc,nl->nc", g, w)


def boot_means(vec, idx, chunk=1000):
    """vec [n] float; idx [B,n] int -> [B] fold means."""
    out = np.empty(idx.shape[0], dtype=np.float64)
    for s in range(0, idx.shape[0], chunk):
        out[s:s + chunk] = vec[idx[s:s + chunk]].mean(axis=1)
    return out


def bstats(delta_vec, idx):
    bm = boot_means(delta_vec, idx) * 100
    return {
        "delta_pp": round(float(delta_vec.mean()) * 100, 2),
        "ci95_pp": [round(float(np.percentile(bm, 2.5)), 2), round(float(np.percentile(bm, 97.5)), 2)],
        "frac_boot_le_0": round(float((bm <= 0).mean()), 4),
    }, bm


def main():
    t0 = time.time()
    torch.manual_seed(0)
    man = json.load(open(DUMP_MANIFEST))["dumps"]
    assert len(man) == 9 and all(e["gate_test"] and e["gate_val_tta"] for e in man.values())

    rng = np.random.default_rng(0)
    idx10 = rng.integers(0, 10000, size=(N_BOOT, 10000), dtype=np.int32)
    idx30 = rng.integers(0, 30000, size=(N_BOOT, 30000), dtype=np.int32)

    per_ckpt, store = {}, {}
    y_test_ref = None
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"c100_{variant}_seed{seed}"
            e = man[name]
            z = np.load(e["npz"])
            gv = torch.tensor(z["goodness_val"], dtype=torch.float64)
            gt = torch.tensor(z["goodness_test"], dtype=torch.float64)
            gvf = torch.tensor(z["goodness_val_flip"], dtype=torch.float64)
            yv = torch.tensor(z["y_val"].astype(np.int64))
            yt = torch.tensor(z["y_test"].astype(np.int64))
            if y_test_ref is None:
                y_test_ref = z["y_test"].copy()
            assert np.array_equal(y_test_ref, z["y_test"]), f"test labels differ for {name}"

            # ---- R0 anchors (gate again from the arrays themselves)
            r0_test_logits = gt.sum(1)
            r0_full = float((r0_test_logits.argmax(1) == yt).float().mean())
            r0_val_logits = gv.sum(1)
            r0_val_no = float((r0_val_logits.argmax(1) == yv).float().mean())
            val_tta = float(((r0_val_logits + gvf.sum(1)).argmax(1) == yv).float().mean())
            g_test_ok = abs(r0_full - e["anchor_test_s1_top1_no_tta"]) <= TOL
            g_val_ok = abs(val_tta - e["anchor_val_best_s1_tta"]) <= TOL
            assert g_test_ok and g_val_ok, name

            # ---- fits on VAL
            w1 = fit_R1(gv, yv)
            invT2 = fit_R2(gv, yv)
            m_std = margins(gv).std(dim=0, unbiased=False)  # ddof=0, fit split
            a3, b3, s3 = fit_R3(gv, yv, m_std, invT2)

            logits = {
                "R0": (r0_val_logits, r0_test_logits),
                "R1": (torch.einsum("nlc,l->nc", gv, w1), torch.einsum("nlc,l->nc", gt, w1)),
                "R2": (torch.einsum("nlc,l->nc", gv, invT2), torch.einsum("nlc,l->nc", gt, invT2)),
                "R3": (r3_logits(gv, m_std, a3, b3, s3), r3_logits(gt, m_std, a3, b3, s3)),
            }
            entry = {
                "group": "c100", "variant": variant, "seed": seed,
                "anchor_s1_top1_no_tta": e["anchor_test_s1_top1_no_tta"],
                "R0_full_acc": round(r0_full, 4),
                "val_anchor_best_s1_tta": e["anchor_val_best_s1_tta"],
                "r0_val_tta_recomputed": round(val_tta, 4),
                "r0_val_no_tta": round(r0_val_no, 4),
                "gate_pass": True, "n_fit": 5000, "n_eval": 10000,
                "readouts": {},
            }
            correct = {}
            for r, (lf, le) in logits.items():
                fit_nll, fit_acc = nll_and_acc(lf, yv)
                eval_acc = float((le.argmax(1) == yt).float().mean())
                correct[r] = (le.argmax(1) == yt).numpy().astype(np.float32)
                rd = {"eval_acc": round(eval_acc, 4), "fit_acc": round(fit_acc, 4)}
                if r == "R0":
                    rd["full_acc"] = round(r0_full, 4)
                    rd["params"] = {"w": [1.0] * gv.shape[1]}
                else:
                    rd["fit_nll"] = round(fit_nll, 4)
                    rd["delta_vs_R0_eval_pp"] = round((eval_acc - r0_full) * 100, 2)
                if r == "R1":
                    rd["params"] = {"w": [round(x, 4) for x in w1.tolist()]}
                elif r == "R2":
                    rd["params"] = {"T": [round(1.0 / x, 4) for x in invT2.tolist()]}
                elif r == "R3":
                    rd["params"] = {"a": [round(x, 4) for x in a3.tolist()],
                                    "b": [round(x, 4) for x in b3.tolist()],
                                    "fixed_prescale_s3": round(s3, 4),
                                    "margin_std_fit": [round(x, 4) for x in m_std.tolist()]}
                entry["readouts"][r] = rd

            entry["solo_block_eval_acc"] = [
                round(float((gt[:, l, :].argmax(1) == yt).float().mean()), 4) for l in range(gt.shape[1])]
            fit_accs = [entry["readouts"][r]["fit_acc"] for r in RUNGS]
            eval_accs = [entry["readouts"][r]["eval_acc"] for r in RUNGS]
            best_fit = RUNGS[int(np.argmax(fit_accs))]
            best_eval = RUNGS[int(np.argmax(eval_accs))]
            entry["best_readout_by_fit"] = best_fit
            entry["best_readout_by_eval"] = best_eval

            bs, bm_store = {}, {}
            for r in RUNGS:
                bs[r], bm_store[r] = bstats(correct[r] - correct["R0"], idx10)
            for tag, rr in (("best_by_fit", best_fit), ("best_by_eval", best_eval)):
                bs[tag] = dict(bs[rr])
                bs[tag]["readout"] = rr
            entry["bootstrap_vs_R0"] = bs
            per_ckpt[name] = entry
            store[name] = {"correct": correct, "boot_means": bm_store,
                           "best_fit": best_fit, "r0_full": r0_full,
                           "eval": {r: entry["readouts"][r]["eval_acc"] for r in ["R0"] + RUNGS},
                           "delta_pp": {r: entry["readouts"][r]["delta_vs_R0_eval_pp"] for r in RUNGS}}
            print(f"{name}: R0={r0_full:.4f} " +
                  " ".join(f"{r}={entry['readouts'][r]['eval_acc']:.4f}" for r in RUNGS) +
                  f" best_fit={best_fit} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- aggregates ----------------
    aggregates = {}
    for variant in VARIANTS:
        names = [f"c100_{variant}_seed{s}" for s in SEEDS]
        agg = {"group": "c100", "n_seeds": 3, "checkpoints": names, "readouts": {}}
        r0s = np.array([store[n]["eval"]["R0"] for n in names])
        agg["readouts"]["R0"] = {"eval_acc_mean": round(float(r0s.mean()), 5),
                                 "eval_acc_std": round(float(r0s.std(ddof=0)), 5)}
        for r in RUNGS:
            accs = np.array([store[n]["eval"][r] for n in names])
            dl = np.array([store[n]["delta_pp"][r] for n in names])
            pooled = np.concatenate([store[n]["correct"][r] - store[n]["correct"]["R0"] for n in names])
            bm = boot_means(pooled, idx30) * 100
            agg["readouts"][r] = {
                "eval_acc_mean": round(float(accs.mean()), 5),
                "eval_acc_std": round(float(accs.std(ddof=0)), 5),
                "delta_vs_R0_pp_mean": round(float(dl.mean()), 3),
                "delta_vs_R0_pp_std": round(float(dl.std(ddof=0)), 3),
                "pooled_bootstrap": {
                    "delta_pp": round(float(pooled.mean()) * 100, 3),
                    "ci95_pp": [round(float(np.percentile(bm, 2.5)), 3),
                                round(float(np.percentile(bm, 97.5)), 3)],
                    "frac_boot_le_0": round(float((bm <= 0).mean()), 4),
                },
            }
        pooled_bf = np.concatenate([store[n]["correct"][store[n]["best_fit"]] - store[n]["correct"]["R0"]
                                    for n in names])
        bm = boot_means(pooled_bf, idx30) * 100
        agg["best_by_fit_pooled_bootstrap"] = {
            "delta_pp": round(float(pooled_bf.mean()) * 100, 3),
            "ci95_pp": [round(float(np.percentile(bm, 2.5)), 3), round(float(np.percentile(bm, 97.5)), 3)],
            "frac_boot_le_0": round(float((bm <= 0).mean()), 4),
        }
        agg["best_by_fit_readouts"] = [store[n]["best_fit"] for n in names]
        aggregates[f"c100/{variant}"] = agg

    # ---------------- interaction (repaired vs cumulative) ----------------
    def rung_of(n, tag):
        return store[n]["best_fit"] if tag == "best_by_fit" else tag

    interaction = {}
    for tag in RUNGS + ["best_by_fit"]:
        sec = {}
        gain_folds, gain_mean = {}, {}
        for variant in VARIANTS:
            names = [f"c100_{variant}_seed{s}" for s in SEEDS]
            folds = np.mean([store[n]["boot_means"][rung_of(n, tag)] for n in names], axis=0)
            mg = float(np.mean([(store[n]["correct"][rung_of(n, tag)] - store[n]["correct"]["R0"]).mean()
                                for n in names])) * 100
            gain_folds[variant], gain_mean[variant] = folds, mg
            sec[variant] = {"mean_gain_pp": round(mg, 3)}
        for repaired in ["gamma0", "gated_k0"]:
            d = gain_folds[repaired] - gain_folds["cumulative"]
            sec[f"{repaired}_minus_cumulative"] = {
                "diff_pp": round(gain_mean[repaired] - gain_mean["cumulative"], 3),
                "ci95_pp": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                "frac_boot_le_0": round(float((d <= 0).mean()), 4),
            }
        interaction[tag] = sec

    # ---------------- trio span ----------------
    span = {}
    for r in ["R0"] + RUNGS:
        vm = {v: round(float(np.mean([store[f'c100_{v}_seed{s}']["eval"][r] for s in SEEDS])), 5)
              for v in VARIANTS}
        span[r] = {"variant_means": vm,
                   "span_pp": round((max(vm.values()) - min(vm.values())) * 100, 3),
                   "best_variant": max(vm, key=vm.get)}

    header = {
        "description": "Readout-spectrum refit on the TRUE VALIDATION SPLIT (CIFAR-100 trio only). Upgrade of readout_spectrum_results.json removing its CAVEAT_TEST_SPLIT_FITTED.",
        "fit_protocol_honest": "R1-R3 fit on each run's own 5000-example validation split (CIFAR-100 TRAIN images selected by the run's persisted split_indices.json val_idx, eval transform, EMA weights) and evaluated on the FULL official 10k test set. No test example influences any fitted parameter. Val dumps anchored: recomputed TTA val top-1 == events.jsonl run_end best_s1 (gate tol 5e-4); R0 full-test top-1 == events.jsonl final_test_eval stage1_test_top1_no_tta (gate tol 5e-4); stage1_best.pt best_s1 field == run_end best_s1 (checkpoint identity).",
        "split_protocol": "split_indices.json is IDENTICAL ACROSS VARIANTS WITHIN A SEED (md5-verified shared-split trio protocol: seed 42/123/456 each have one split shared by gamma0, gated_k0, cumulative; val_idx md5 e954bec857/8db77f3d46/82ace8b632) but DIFFERS ACROSS SEEDS (carve = random.Random(seed).shuffle(range(50000)), val = sorted first 5000; cross-seed val overlap 516/5000 = chance). Test set identical for all 9 checkpoints (y_test asserted equal), so the interaction contrast pairs across variants on the full 10k test.",
        "readout_ladder": {
            "R0": "plain sum over blocks (control; reproduces asserted S1 no-TTA top-1 on full 10k)",
            "R1": "per-depth scalar weights w_d, NLL(=logistic-regression) fit, L params",
            "R2": "per-depth temperature T_d, log-opinion pool of per-block softmax(g_d/T_d) (= per-block PRODUCT of experts). NOTE: pooled softmax == softmax(sum_d g/T_d), so R2 is exactly R1 constrained to positive weights; argmax family identical.",
            "R3": "margin-gated input-dependent w_d(x)=sigmoid(a_d*m_d(x)+b_d), m_d = block top1-top2 goodness margin (standardized by fit-split=VAL per-depth std, ddof=0), 2 params/depth, NLL fit, fixed prescale s3=2*max(invT_R2), warm start at R2's exact solution.",
        },
        "fit": "full-batch Adam(0.05, 400 steps) + LBFGS polish (strong Wolfe, max_iter 500), float64, torch CPU, torch.manual_seed(0)",
        "bootstrap": f"{N_BOOT}-fold paired bootstrap over the FULL 10k test (np.random.default_rng(0), shared index matrices; pattern of paired_bootstrap_cifar10.py); best readout selected on the VAL fit split (best_by_fit); best_by_eval also reported. CI endpoints depend on the RNG stream and may differ ~0.01-0.03pp from other reruns; all point estimates are deterministic.",
        "eval_protocol": "S1 no-TTA, EMA weights, official full 10k test split",
        "n_checkpoints": 9,
        "predecessor": PREV,
        "val_dumps": DUMP_MANIFEST,
        "dump_script": os.path.join(SCRATCH, "dump_val_test_block_data.py"),
        "fit_script": os.path.abspath(__file__),
        "created": time.strftime("%Y-%m-%d"),
    }

    out = {"per_checkpoint": per_ckpt, "aggregates": aggregates, "interaction": interaction,
           "c100_trio_span_per_readout": span, "_header": header}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
