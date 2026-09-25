#!/usr/bin/env python3
"""Print the headline gradient-share table from gradient_share_results.json."""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json
import sys

P = FF_WORK + "/gradient_share_results.json"
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else P))

for name, a in r["arms"].items():
    g = a["gates"]
    print(f"\n=== {name} (gamma={a['config_summary']['gamma_scale']}, use_mgc={a['config_summary']['use_mgc']}, "
          f"block_curr_lambda={a['config_summary']['block_curr_lambda']}, epoch={a['epoch_stage1']}, "
          f"n_batches={a['n_batches']}x{a['batch_size']})")
    print(f"    gate: recomputed {g['recomputed_ema_test_top1_no_tta']:.4f} vs anchor "
          f"{g['anchor_final_test_s1_top1_no_tta']:.4f} (delta {g['delta']:.1e}) pass={g['gate_accuracy_pass']}")
    hdr = f"{'blk':>3} {'||g_tot||':>9} {'lin_err':>8} | " + " ".join(f"{t:>15}" for t in
          ["block_cum", "mgc_local", "block_curr", "aspects", "depth", "supcon", "recon", "moe_bal", "moe_z"]) + " | disc_all"
    print(hdr)
    for pb in a["per_block"]:
        cells = []
        for t in ["block_cum", "mgc_local", "block_curr", "aspects", "depth", "supcon", "recon", "moe_bal", "moe_z"]:
            s = pb["terms"][t]
            if s["present"]:
                cells.append(f"{s['share_mean']:.3f}±{s['share_std']:.3f}")
            else:
                cells.append("-")
        print(f"{pb['block']:>3} {pb['total_grad_norm_mean']:>9.4f} {pb['linearity_rel_err_max']:>8.1e} | "
              + " ".join(f"{c:>15}" for c in cells) + f" | {pb['disc_all_share_mean']:.3f}")
    print("    cos(term,total) block_cum: " + ", ".join(
        f"d{pb['block']}={pb['terms']['block_cum']['cos_mean']:.2f}" for pb in a["per_block"]))
    print("    margins blk_nl: " + ", ".join(
        f"d{pb['block']}={pb['loss_value_means']['margin_blk_nl']:.2f}" for pb in a["per_block"])
        + " | curr_nl: " + ", ".join(
        f"d{pb['block']}={pb['loss_value_means']['margin_curr_nl']:.2f}" for pb in a["per_block"]))
    print("    saturation sigmoid(-a*margin_blk): " + ", ".join(
        f"d{pb['block']}={pb['loss_value_means']['sat_factor_nl']:.4f}" for pb in a["per_block"]))
