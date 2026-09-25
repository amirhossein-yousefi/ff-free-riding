"""Phase-2 errata (D1, D2, D4, D5, D7, D13): read every value from raw run logs.

No torch; pure JSON reading. Writes results/d_errata_values.json.
Sample SD (ddof=1) throughout. Accuracies in percent, rounded to 2 dp
(ROUND_HALF_UP on the decimal mean).
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
from decimal import ROUND_HALF_UP, Decimal

CF = (FF_HOME + "/amir-porjects/"
      "cifar_10_fair/making_v1_v4_fair_comparison")
GA = f"{CF}/basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/runs"
GA8 = f"{CF}/basic_fair_revised_valfix_cp_fair/ff_gated_ablation_L8/runs"
EX = f"{CF}/revised_training/extracted"

RUNS = {
    # L4/D128 seed-42 gated ablation (SAM off)
    "l4_k0_s42": f"{GA}/ff_cifar10_adaptive_t1_k0_L4_D128_bs512_seed42_20260314_032410",
    "l4_const_s42": f"{GA}/ff_cifar10_baseline_constant_L4_D128_bs512_seed42_20260313_232043",
    "l4_lcff_s42": f"{GA}/ff_cifar10_lcff_prefix_L4_D128_bs512_seed42_20260315_084228",
    "l4_g0_s42": f"{GA}/ff_cifar10_no_all_collab_L4_D128_bs512_seed42_20260315_124117",
    # L4/D128 kappa=0 seeds 123/456 (Colab, from_drive zip -001, extracted copy verified identical)
    "l4_k0_s123": f"{EX}/ff_cifar10_gated_multiseed/runs_gated_k0_seed123/"
                  "ff_cifar10_cifar10_gated_k0_seed123_L4_D128_bs512_seed123_20260426_071309",
    "l4_k0_s456": f"{EX}/ff_cifar10_gated_multiseed/runs_gated_k0_seed456/"
                  "ff_cifar10_cifar10_gated_k0_seed456_L4_D128_bs512_seed456_20260426_150234",
    # L8/D128
    "l8_k0_s42": f"{GA8}/ff_cifar10_adaptive_t1_k0_L8_L8_D128_bs512_seed42_20260315_201449",
    "l8_k0_s123": f"{EX}/ff_cifar10_l8_gated_multiseed/runs_gated_k0_L8_seed123/"
                  "ff_cifar10_cifar10_gated_k0_L8_seed123_L8_D128_bs512_seed123_20260427_041411",
    "l8_k0_s456": f"{EX}/ff_cifar10_l8_gated_multiseed/runs_gated_k0_L8_seed456/"
                  "ff_cifar10_cifar10_gated_k0_L8_seed456_L8_D128_bs512_seed456_20260427_114056",
    "l8_g0_s42": f"{GA8}/ff_cifar10_no_all_collab_L8_L8_D128_bs512_seed42_20260318_062709",
    "l8_const_s42": f"{GA8}/ff_cifar10_baseline_constant_L8_L8_D128_bs512_seed42_20260318_204046",
    "l8_lcff_s42": f"{GA8}/ff_cifar10_lcff_prefix_L8_L8_D128_bs512_seed42_20260319_105650",
    # CIFAR-10 gamma=0 D256
    "c10_g0_s42": f"{CF}/ablations/no_all_collab/runs/"
                  "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260312_132043",
    "c10_g0_s123": glob.glob(f"{CF}/revised_training/out/gamma0_seed123/runs/*")[0],
    "c10_g0_s456": glob.glob(f"{CF}/revised_training/out/gamma0_seed456/runs/*")[0],
    # CIFAR-10 D256 L4 gamma sweep (tab:freeriding)
    "c10_cpfair_s42": f"{CF}/basic_fair_revised_valfix_cp_fair/runs/"
                      "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035",
    "c10_lcff_s42_completed": f"{CF}/ablations/lcff_prefix/runs/"
                              "ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260312_132019",
    "c10_lcff_s42_aborted_ep277": f"{CF}/ablations/lcff_prefix/runs/"
                                  "ff_cifar10_cifar10_cp_fair_ablation_lcff_prefix_L4_D256_bs512_seed42_20260311_120438",
}
CFG_KEYS = ["seed", "use_sam", "num_blocks", "d_model", "n_experts", "moe_top_k", "gamma_scale",
            "gamma_gating_mode", "gamma_kappa", "gamma_tau", "block_curr_lambda",
            "depth_order_lambda", "epochs_stage1", "epochs_stage2"]


def r2(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def run_record(d):
    rec = {"dir": d}
    try:
        c = json.load(open(f"{d}/config.json"))
        rec["config"] = {k: c.get(k) for k in CFG_KEYS}
    except Exception as e:  # noqa: BLE001
        rec["config"] = f"unreadable: {e}"
    fe = [e for e in jl(f"{d}/events.jsonl") if e.get("event") == "final_test_eval"]
    if fe:
        e = fe[-1]
        rec["test"] = {k: round(100 * e[k], 2) for k in
                       ["stage1_test_top1_no_tta", "stage1_test_top1_tta",
                        "stage2_test_top1_no_tta", "stage2_test_top1_tta"]}
        rec["test_source"] = f"{d}/events.jsonl final_test_eval"
    else:
        rec["test"] = None
    em = jl(f"{d}/epoch_metrics.jsonl")
    s1 = [r for r in em if r.get("stage") == 1]
    s2 = [r for r in em if r.get("stage") == 2]
    rec["best_val_s1"] = round(100 * max((r.get("best_s1") or 0) for r in s1), 2) if s1 else None
    rec["best_val_s2_tta"] = round(100 * max((r.get("best_s2") or 0) for r in s2), 2) if s2 else None
    lm = [r for r in jl(f"{d}/layer_metrics.jsonl") if r.get("stage", 1) == 1]
    if lm:
        last = max(r["epoch"] for r in lm)
        L = sorted([r for r in lm if r["epoch"] == last], key=lambda r: r["layer"])
        rec["stage1_last_epoch"] = last
        rec["sep_curr_nl_last"] = [round(r["sep_curr_nl"], 4) for r in L]
        rec["sep_nl_last"] = [round(r["sep_nl"], 4) for r in L]
    return rec


def agg(vals):
    return {"values": vals, "n": len(vals), "mean": r2(st.mean(vals)),
            "sd_sample": round(st.stdev(vals), 4) if len(vals) > 1 else None,
            "sd_population": round(st.pstdev(vals), 4)}


def main(out):
    R = {k: run_record(d) for k, d in RUNS.items()}
    P = ["stage1_test_top1_no_tta", "stage1_test_top1_tta", "stage2_test_top1_no_tta", "stage2_test_top1_tta"]
    A = {}
    A["D1_l4_k0_n3"] = {p: agg([R[k]["test"][p] for k in ["l4_k0_s42", "l4_k0_s123", "l4_k0_s456"]]) for p in P}
    A["D1_l4_k0_sep_deep_n3"] = agg([R[k]["sep_curr_nl_last"][-1] for k in ["l4_k0_s42", "l4_k0_s123", "l4_k0_s456"]])
    A["D2_l8_k0_test_n2"] = {p: agg([R[k]["test"][p] for k in ["l8_k0_s123", "l8_k0_s456"]]) for p in P}
    A["D2_l8_k0_s42_val_only_best_s2_tta"] = R["l8_k0_s42"]["best_val_s2_tta"]
    A["D2_l8_k0_sep_deep_n3"] = agg([R[k]["sep_curr_nl_last"][-1] for k in ["l8_k0_s42", "l8_k0_s123", "l8_k0_s456"]])
    A["D4_c10_g0_n3"] = {p: agg([R[k]["test"][p] for k in ["c10_g0_s42", "c10_g0_s123", "c10_g0_s456"]]) for p in P}
    A["D5_sep_ratio_g0_over_lcff"] = round(R["c10_g0_s42"]["sep_curr_nl_last"][-1] / R["c10_lcff_s42_completed"]["sep_curr_nl_last"][-1], 3)
    A["D5_sep_ratio_g0_over_cpfair"] = round(R["c10_g0_s42"]["sep_curr_nl_last"][-1] / R["c10_cpfair_s42"]["sep_curr_nl_last"][-1], 3)
    A["D1_l4_d128_range_s2tta"] = r2(R["l4_const_s42"]["test"]["stage2_test_top1_tta"] - R["l4_lcff_s42"]["test"]["stage2_test_top1_tta"])
    A["D2_l8_best_mean_minus_lcff"] = r2(A["D2_l8_k0_test_n2"]["stage2_test_top1_tta"]["mean"] - R["l8_lcff_s42"]["test"]["stage2_test_top1_tta"])
    res = {"_header": {"items": "D1 D2 D4 D5 D7 D13", "script": os.path.abspath(__file__),
                       "created": time.strftime("%Y-%m-%d"), "sd": "sample (ddof=1) unless labelled population"},
           "runs": R, "aggregates": A}
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(A, indent=1))


if __name__ == "__main__":
    os.makedirs(os.path.join(FF_OUT, "results"), exist_ok=True)
    main(os.path.join(FF_OUT, "results", "d_errata_values.json"))
