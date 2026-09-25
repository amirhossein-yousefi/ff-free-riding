#!/usr/bin/env python3
"""Equivalence tests for trainers/cp_fair_cifar100.py (CIFAR-100, L4/D256). CPU only.

A. Flags off == v1: one Stage-1 step (batch of 16 through all 4 blocks) on the real configs of
   the dense dissociation trio (gamma=0, kappa=0 adaptive, cumulative gamma=0.7; no MoE, no SAM)
   and of the v1 default (32-expert MoE, SAM on): bitwise equal.
B. Depth-order locality (P3): three consecutive steps (trio cumulative, SAM off; MoE default,
   SAM on): bitwise equal to v1.
C. MGC on == rebuttal mgc_c100_trainer.py on the MGC-C100 config (two steps, bitwise).
   Note: the camera-ready MGC code also carries the block-0 branch for c_0 > 1 (from the
   CIFAR-10 MGC trainer); for the configuration that was run (c_0 = 1) it is inactive.
F. Semantics: implied current-block weight 0 at every block under MGC; compensator zero at
   block 0 and positive at blocks 1-3.
G. main(): planned config.json under the documented env equals the real MGC-C100 seed-42 run
   and the trio cumulative seed-42 run (only camera-ready keys may be added).
R. RNG save/restore in checkpoints.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equiv_utils as U  # noqa: E402

NAME = "cp_fair_cifar100.py"
B = 16
CR_NEW_CFG_KEYS = {"use_mgc", "mgc_c0", "mgc_rho", "mgc_eps"}
TRIO = dict(n_experts=1, moe_top_k=1, use_sam=False, num_heads=8)


def make_cfg(mod, **over):
    cfg = mod.FFConfig()
    cfg.__post_init__()
    cfg.device = "cpu"
    cfg.workers = 0
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


CONFIGS = {
    "trio_gamma0": dict(TRIO, gamma_scale=0.0),
    "trio_kappa0_adaptive": dict(TRIO, gamma_gating_mode="adaptive", gamma_0=0.7, gamma_kappa=0.0,
                                 gamma_tau=1.0),
    "trio_cumulative": dict(TRIO, gamma_scale=0.7),
    "v1_default_moe32_samon": dict(),
}


def implied_lambda(cfg, layer):
    rest = (layer["loss_aspects"] + cfg.block_lambda * layer["loss_block"]
            + cfg.block_lambda * layer.get("loss_mgc", 0.0) + layer["loss_depth"]
            + layer["moe_bal_w"] * layer["loss_bal"] + layer["moe_z_w"] * layer["loss_z"]
            + cfg.contrastive_lambda * layer["loss_con"] + layer["loss_recon"])
    return (layer["loss_total"] - rest) / layer["loss_block_curr"]


def main():
    rec = U.Recorder(os.path.basename(__file__))
    out_json = sys.argv[1] if len(sys.argv) > 1 else None
    new_path = os.path.join(U.TRAINERS, NAME)
    rec.out["trainer_sha256"] = U.sha256(new_path)
    new = U.cr_trainer(NAME)
    b1, b2, b3 = (U.make_batch(make_cfg(new), B, seed=s) for s in (2026, 77, 5))
    v1p = rec.ref("v1_trainer", U.v1_path(NAME), U.V1_SHA[NAME])
    v1 = U.load_trainer(v1p, "v1_c100") if v1p else None

    for cname, over in CONFIGS.items():
        if v1 is None:
            rec.skip(f"A_flags_off_equals_v1_{cname}", "FF_V1_TRAINERS not set")
            continue
        U.check_pair(rec, f"A_flags_off_equals_v1_{cname}", v1, make_cfg(v1, **over),
                     new, make_cfg(new, **over), [b1])
    for cname in ("trio_cumulative", "v1_default_moe32_samon"):
        if v1 is None:
            rec.skip(f"B_p3_three_steps_equals_v1_{cname}", "FF_V1_TRAINERS not set")
            continue
        over = CONFIGS[cname]
        U.check_pair(rec, f"B_p3_three_steps_equals_v1_{cname}", v1, make_cfg(v1, **over),
                     new, make_cfg(new, **over), [b1, b2, b3])

    mgc_over = dict(CONFIGS["trio_cumulative"], use_mgc=True, gamma_gating_mode="constant")
    mp = rec.ref("rebuttal_mgc_c100_trainer", U.rebuttal_path("mgc_c100_trainer.py"),
                 U.REBUTTAL_SHA["mgc_c100_trainer.py"])
    if mp:
        mgc = U.load_trainer(mp, "rebuttal_mgc_c100")
        ra, rb = U.check_pair(rec, "C_mgc_on_equals_rebuttal_mgc_c100_trainer", mgc, make_cfg(mgc, **mgc_over),
                              new, make_cfg(new, **mgc_over), [b1, b2])
        rec.record("C_mgc_loss_per_block_matches_rebuttal", U.loss_per_block(ra) == U.loss_per_block(rb),
                   {"rebuttal": U.loss_per_block(ra), "camera_ready": U.loss_per_block(rb)})
        del mgc
    else:
        rec.skip("C_mgc_on_equals_rebuttal_mgc_c100_trainer", "FF_REBUTTAL_TRAINERS not set")

    for arm, over, exp in (("cumulative", CONFIGS["trio_cumulative"], [0.25, 0.5, 0.75, 1.0]),
                           ("mgc", mgc_over, [0.0] * 4)):
        cfg = make_cfg(new, **over)
        r = U.run_steps(new, cfg, [b1])
        lam = [implied_lambda(cfg, l) for l in r["stats"]]
        err = max(abs(a - b) for a, b in zip(lam, exp))
        terms = [l.get("loss_mgc") for l in r["stats"]]
        comp_ok = (terms[0] == 0.0 and all(v > 0.0 for v in terms[1:])) if cfg.use_mgc \
            else all(v is None for v in terms)
        rec.record(f"F_implied_current_block_weight_{arm}", err < 1e-3 and comp_ok,
                   {"implied": [round(v, 6) for v in lam], "expected": exp, "loss_mgc_per_block": terms})

    root = os.environ.get("FF_REFERENCE_RUNS")
    mgc_env = dict(FF_USE_MGC="1", FF_MGC_C0="1.0", FF_N_EXPERTS="1", FF_MOE_TOP_K="1", FF_USE_SAM="0",
                   FF_NUM_HEADS="8", FF_GAMMA_GATING_MODE="constant", FF_GAMMA_SCALE="0.7",
                   FF_VERSION_TAG="mgc_c100", FF_SEED="42")
    p = (os.path.join(root, "mgc_rebuttal", "from_colab", "runs_from_colab",
                      "ff_cifar100_mgc_c100_L4_D256_bs256_seed42_20260724_220559", "config.json")
         if root else None)
    U.check_config_vs_run(rec, "G_config_equals_run_mgc_c100_seed42", new_path, mgc_env, p, CR_NEW_CFG_KEYS)
    trio_env = dict(FF_NUM_BLOCKS="4", FF_D_MODEL="256", FF_NUM_HEADS="8", FF_EPOCHS_STAGE1="362",
                    FF_USE_SAM="0", FF_N_EXPERTS="1", FF_MOE_TOP_K="1", FF_GAMMA_SCALE="0.7",
                    FF_VERSION_TAG="cifar100_cumulative_seed42", FF_SEED="42")
    p = (os.path.join(root, "revised_training", "extracted", "ff_cifar100_multiseed", "runs_cumulative_seed42",
                      "ff_cifar100_cifar100_cumulative_seed42_L4_D256_bs256_seed42_20260418_032149",
                      "config.json") if root else None)
    U.check_config_vs_run(rec, "G_config_equals_run_trio_cumulative_seed42_flags_off", new_path, trio_env, p,
                          CR_NEW_CFG_KEYS)
    _, keys_mgc = U.planned_config(new_path, mgc_env)
    _, keys_off = U.planned_config(new_path, dict(trio_env))
    i = keys_mgc.index("loss_block_curr")
    rec.record("G_layer_metric_keys", "loss_mgc" not in keys_off
               and keys_mgc[i + 1:i + 4] == ["loss_mgc", "mgc_lam_mean", "mgc_lam_p95"], keys_mgc[i:i + 5])

    if v1 is not None:
        U.run_rng_tests(rec, "R", new, v1, make_cfg(new, **mgc_over), b1, b2)
    else:
        rec.skip("R_rng_tests", "FF_V1_TRAINERS not set")
    rc = rec.finish(out_json)
    U.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
