"""Phase-2 PAT fixes (P3, P4, P14/A6, P17, P18, P22, P23): every number written into
the paper by these fixes, read from raw run logs / configs.

No torch; pure JSON reading. Writes results/pat_values.json.
Sample SD (ddof=1). Accuracies in percent.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import glob
import json
import os
import statistics as st
import time
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

PR = FF_HOME + "/amir-porjects"
CF = f"{PR}/cifar_10_fair/making_v1_v4_fair_comparison"
TINY = f"{PR}/tiny_imagenet_fair"
TEX = f"{CF}/NeurIPS_FF_Net_Last_submitted"
CR = Path(FF_OUT)  # output root
OUT = CR / "results" / "pat_values.json"

SCFF_TINY_TOP1 = 35.67  # scff2025 (Nature Communications 16:5978), as printed in the paper


def r2(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def events(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def final_eval(run):
    fte = [e for e in events(f"{run}/events.jsonl") if e.get("event") == "final_test_eval"]
    return fte[-1] if fte else None


def cfg(run):
    c = json.load(open(f"{run}/config.json"))
    return c.get("cfg", c)


def layer_rows(run):
    with open(f"{run}/layer_metrics.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def last_epoch_layers(run, epoch=None):
    rows = layer_rows(run)
    ep = epoch if epoch is not None else max(r["epoch"] for r in rows)
    return ep, sorted([r for r in rows if r["epoch"] == ep], key=lambda r: r["layer"])


def main():
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "sd": "sample (ddof=1)"}

    # ---------- P14 / A6: Tiny ImageNet single-crop S1 ----------
    tiny_runs = {
        42: f"{TINY}/from_drive_1/extracted/ff_tiny_imagenet_training/runs_seed42/"
            "ff_tiny_imagenet_tiny_imagenet_cp_fair_seed42_L4_D256_bs256_seed42_20260412_104050",
        123: f"{TINY}/extracted_seeds/ff_tiny_imagenet_multiseed/runs_seed123/"
             "ff_tiny_imagenet_tiny_imagenet_cp_fair_seed123_L4_D256_bs256_seed123_20260413_205048",
        456: f"{TINY}/extracted_seeds/ff_tiny_imagenet_multiseed/runs_seed456/"
             "ff_tiny_imagenet_tiny_imagenet_cp_fair_seed456_L4_D256_bs256_seed456_20260414_212958",
    }
    per = {}
    for s, run in tiny_runs.items():
        f = final_eval(run)
        c = cfg(run)
        per[s] = {
            "run": run,
            "s1_top1_no_tta": 100 * f["stage1_test_top1_no_tta"],
            "s1_top1_tta": 100 * f["stage1_test_top1_tta"],
            "s2_top1_tta": 100 * f["stage2_test_top1_tta"],
            "gamma_scale": c.get("gamma_scale"), "use_sam": c.get("use_sam"),
            "hnm_k": [c.get("hard_negative_k_first"), c.get("hard_negative_k_last")],
            "num_classes": c.get("num_classes"),
        }
    agg = {}
    for k in ("s1_top1_no_tta", "s1_top1_tta", "s2_top1_tta"):
        v = [per[s][k] for s in (42, 123, 456)]
        agg[k] = {"mean": r2(st.mean(v)), "sd": r2(st.stdev(v)), "values": [r2(x) for x in v],
                  "mean_raw": st.mean(v), "sd_raw": st.stdev(v)}
    gaps = {k: round(agg[k]["mean_raw"] - SCFF_TINY_TOP1, 2) for k in agg}
    out["tiny"] = {"per_seed": per, "agg": agg,
                   "scff_top1": SCFF_TINY_TOP1, "gap_vs_scff_pp": gaps}

    # ---------- P17: HNM unique-label coverage ----------
    def cov(C, k):
        return 1 - (1 - 1 / (C - 1)) ** k
    out["hnm_coverage"] = {
        "formula": "1-(1-1/(C-1))^k (draws uniform over the C-1 wrong labels, with replacement)",
        "cifar10_k8": round(cov(10, 8), 4), "cifar10_k16": round(cov(10, 16), 4),
        "tiny_k20": round(cov(200, 20), 4), "tiny_k40": round(cov(200, 40), 4),
        "k_ratio_tiny_over_c10": 20 / 8,
        "class_ratio": 200 / 10,
        "sources": ["trainers/cp_fair_cifar10.py:_choose_hard_negatives (randint over C, resample y)",
                    "CIFAR-10 CP-FAIR config.json hard_negative_k_first/last = 8/16",
                    "Tiny config.json hard_negative_k_first/last = 20/40 (all 3 seeds)"],
    }

    # ---------- P3: hyperparameters of the published CP-FAIR run ----------
    cp = f"{CF}/basic_fair_revised_valfix_cp_fair/runs/" \
         "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035"
    c = cfg(cp)
    keys = ["symba_alpha", "sff_ni_weight", "block_lambda", "block_curr_lambda",
            "block_curr_depth_slope", "depth_order_lambda", "depth_margin_pos",
            "depth_margin_neg", "recon_lambda", "aspect_theta_inits", "aspect_lambdas",
            "aspect_loss_types", "n_aspects", "contrastive_lambda", "contrastive_temp",
            "moe_balance_coef", "moe_zloss_coef", "gamma_scale", "use_mem", "use_sam",
            "hard_negative_k_first", "hard_negative_k_last", "proto_temp",
            "use_softmax_weights"]
    out["cp_fair_hparams"] = {"run": cp, **{k: c.get(k) for k in keys}}

    # ---------- P23: nomem-l2 vs baseline goodness magnitude ----------
    nomem = f"{CF}/basic_fair_revised_valfix_do_this_first/runs/" \
            "ff_cifar10_cifar10_opt_do_this_first_ynorm_nomem_resid_L4_D256_bs512_seed42_20260128_084102"
    base_final = f"{CF}/basic_fair_revised_valfix/runs/" \
                 "ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260110_171401"
    base_prev = f"{CF}/basic_fair_revised_valfix/runs/" \
                "ff_cifar10_cifar10_opt_load_last_fair_reviewfix_moe_valsplit_workersfix_L4_D256_bs512_seed42_20260108_214956"
    ep_n, Ln = last_epoch_layers(nomem)
    ep_b, Lb = last_epoch_layers(base_final)
    ep_b2, Lb2 = last_epoch_layers(base_prev)
    fn, fb = final_eval(nomem), final_eval(base_final)
    out["p23_goodness_magnitude"] = {
        "nomem_l2": {"run": nomem, "epoch": ep_n, "gpos_L3": round(Ln[-1]["gpos"], 3),
                     "s1_tta": r2(100 * fn["stage1_test_top1_tta"]), "use_mem": cfg(nomem).get("use_mem")},
        "baseline": {"run": base_final, "epoch": ep_b, "gpos_L3": round(Lb[-1]["gpos"], 3),
                     "gpos_L3_epoch286_previous_segment": round(Lb2[-1]["gpos"], 3),
                     "s1_tta": r2(100 * fb["stage1_test_top1_tta"])},
        "ratio_final_epochs": round(Lb[-1]["gpos"] / Ln[-1]["gpos"], 2),
        "ratio_using_epoch286": round(Lb2[-1]["gpos"] / Ln[-1]["gpos"], 2),
        "note": "baseline trajectory is split across resumed dirs (1-144, 145-286, 287); "
                "final Stage-1 epoch is 287",
    }

    # ---------- P18: L8 L7/L0 ratio is current-block g+ ----------
    l8 = f"{CF}/basic_fair_revised_valfix_cp_fair/ff_gated_ablation_L8/runs"
    ratios = {}
    for run in sorted(glob.glob(f"{l8}/ff_cifar10_*")):
        ep, L = last_epoch_layers(run)
        if len(L) != 8:
            continue
        ratios[os.path.basename(run)] = {
            "epoch": ep,
            "gpos_cur_L7_over_L0": round(L[-1]["gpos_cur"] / L[0]["gpos_cur"], 4),
            "gpos_cumulative_L7_over_L0": round(L[-1]["gpos"] / L[0]["gpos"], 4),
        }
    out["p18_l8_ratio"] = ratios

    # ---------- P4 / P22: aspect-ablation configs ----------
    zips = sorted(glob.glob(f"{CF}/basic_fair_revised_valfix_cp_fair/new_ablation_2/ff_aspect_ema_ablation-*.zip"))
    asp = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if not n.endswith("config.json"):
                    continue
                run = n.rsplit("/", 1)[0]
                cc = json.loads(zf.read(n))
                cc = cc.get("cfg", cc)
                fe = None
                ev_name = f"{run}/events.jsonl"
                if ev_name in zf.namelist():
                    evs = [json.loads(l) for l in zf.read(ev_name).decode().splitlines() if l.strip()]
                    fl = [e for e in evs if e.get("event") == "final_test_eval"]
                    if fl:
                        fe = {k: v for k, v in fl[-1].items() if "top1" in k}
                asp[run] = {"zip": os.path.basename(z), "n_aspects": cc.get("n_aspects"),
                            "aspect_lambdas": cc.get("aspect_lambdas"), "use_mem": cc.get("use_mem"),
                            "final_test_eval": fe}
    out["p4_aspect_ablation_configs"] = asp
    sa = f"{CF}/basic_fair_revised_valfix_cp_fair/component_ablation_v2_from_drive/extracted/" \
         "ff_component_ablation/runs_single_aspect/" \
         "ff_cifar10_ablation_single_aspect_L4_D256_bs512_seed42_20260404_194358"
    cs = cfg(sa)
    out["p22_single_aspect"] = {
        "run": sa, "n_aspects": cs.get("n_aspects"), "use_mem": cs.get("use_mem"),
        "final_test_eval": {k: v for k, v in final_eval(sa).items() if "top1" in k},
        "slot_kept": "g_list[:n_aspects] with g_list=[g_alignment, g_energy, g_attn, g_learned] "
                     "(basic_fair_revised_valfix_cp_fair.py:1018-1019) -> prototype alignment",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"tiny": out["tiny"]["agg"], "gaps": out["tiny"]["gap_vs_scff_pp"],
                      "hnm": out["hnm_coverage"], "p23": out["p23_goodness_magnitude"]["ratio_final_epochs"]},
                     indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
