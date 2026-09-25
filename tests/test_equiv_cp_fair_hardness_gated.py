#!/usr/bin/env python3
"""Equivalence tests for trainers/cp_fair_hardness_gated.py (CIFAR-10, L4/D128 family). CPU only.

A. Flags off == v1. One Stage-1 step (batch of 16 through all 4 blocks) on the real L4/D128
   config, same seeds, v1 trainer vs camera-ready trainer: loss/metrics, parameters, EMA,
   optimizer state, queues and RNG state after the step are bitwise equal. Four configs:
   kappa=0 adaptive with SAM off (seed-42 anchor) and on (seeds 123/456), constant gamma=0.7
   with SAM on (the MGC / hinge base) and gamma=0 block-local with SAM off.
B. Depth-order locality (P3): three consecutive steps with depth ordering on, SAM on and off:
   bitwise equal to v1 (the v1 leak into blocks[d-1].alpha_raw.grad is discarded before use).
C. MGC on == rebuttal mgc_trainer.py (two steps, bitwise, including the logged MGC terms).
D. Squared hinge on == rebuttal hinge_trainer.py, gamma=0.7 and gamma=0 arms (two steps, bitwise).
E. F1 flags on == the F1 fork (FF_CURR_AUX_OFF; FF_USE_MGC + FF_MGC_KEEP_L0_AUX), two steps.
   (D, E: the rebuttal/F1 trainers always log the three MGC telemetry scalars, as 0.0 when MGC
   is off; the camera-ready trainer logs them only under MGC. Those zeros are checked, every
   other logged value and all state is compared bitwise.)
F. Semantics: implied current-block weight per block recovered from the logged loss terms
   (cumulative 0.25/0.5/0.75/1, MGC 0/0/0/0, F1-b 0/0/0/0, F1-c 0.25/0/0/0) and the MGC
   compensator is zero at block 0 and positive at blocks 1-3.
G. main(): the planned config.json under the documented env equals the config.json of the real
   MGC-D128 and hinge runs (only camera-ready keys may be added), and invalid flag combinations
   are refused.
R. RNG save/restore in checkpoints.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equiv_utils as U  # noqa: E402

NAME = "cp_fair_hardness_gated.py"
B = 16
CR_NEW_CFG_KEYS = {"use_mgc", "mgc_c0", "mgc_rho", "mgc_eps", "mgc_keep_l0_aux", "curr_aux_off",
                   "barrier_type", "barrier_margin"}


def make_cfg(mod, **over):
    cfg = mod.FFConfig()
    cfg.__post_init__()
    cfg.device = "cpu"
    cfg.workers = 0
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


CONFIGS = {
    "k0_adaptive_samoff": dict(use_sam=False),
    "k0_adaptive_samon": dict(use_sam=True),
    "const_g07_samon": dict(gamma_gating_mode="constant", gamma_scale=0.7, use_sam=True),
    "g0_blocklocal_samoff": dict(gamma_gating_mode="constant", gamma_scale=0.0, block_curr_lambda=0.0,
                                 depth_order_lambda=0.0, use_sam=False),
}
BASE = CONFIGS["const_g07_samon"]


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
    v1 = U.load_trainer(v1p, "v1_gated") if v1p else None

    # A. flags off == v1 (one step)
    for cname, over in CONFIGS.items():
        if v1 is None:
            rec.skip(f"A_flags_off_equals_v1_{cname}", "FF_V1_TRAINERS not set")
            continue
        U.check_pair(rec, f"A_flags_off_equals_v1_{cname}", v1, make_cfg(v1, **over),
                     new, make_cfg(new, **over), [b1])

    # B. P3 depth-order locality, three steps
    for cname in ("k0_adaptive_samoff", "const_g07_samon"):
        if v1 is None:
            rec.skip(f"B_p3_three_steps_equals_v1_{cname}", "FF_V1_TRAINERS not set")
            continue
        over = CONFIGS[cname]
        U.check_pair(rec, f"B_p3_three_steps_equals_v1_{cname}", v1, make_cfg(v1, **over),
                     new, make_cfg(new, **over), [b1, b2, b3])

    # C. MGC on == rebuttal mgc_trainer.py
    mgcp = rec.ref("rebuttal_mgc_trainer", U.rebuttal_path("mgc_trainer.py"), U.REBUTTAL_SHA["mgc_trainer.py"])
    runs = {}
    if mgcp:
        mgc = U.load_trainer(mgcp, "rebuttal_mgc_d128")
        ra, rb = U.check_pair(rec, "C_mgc_on_equals_rebuttal_mgc_trainer", mgc,
                              make_cfg(mgc, use_mgc=True, **BASE), new, make_cfg(new, use_mgc=True, **BASE),
                              [b1, b2])
        rec.record("C_mgc_loss_per_block_matches_rebuttal", U.loss_per_block(ra) == U.loss_per_block(rb),
                   {"rebuttal": U.loss_per_block(ra), "camera_ready": U.loss_per_block(rb)})
        # c0 > 1 exercises the block-0 compensator branch (identical code in both trainers)
        U.check_pair(rec, "C_mgc_c0_1p5_rho_0p5_equals_rebuttal", mgc,
                     make_cfg(mgc, use_mgc=True, mgc_c0=1.5, mgc_rho=0.5, **BASE), new,
                     make_cfg(new, use_mgc=True, mgc_c0=1.5, mgc_rho=0.5, **BASE), [b1])
        del mgc
    else:
        rec.skip("C_mgc_on_equals_rebuttal_mgc_trainer", "FF_REBUTTAL_TRAINERS not set")

    # D. squared hinge on == rebuttal hinge_trainer.py (module-level barrier switch)
    hp = rec.ref("rebuttal_hinge_trainer", U.rebuttal_path("hinge_trainer.py"), U.REBUTTAL_SHA["hinge_trainer.py"])
    if hp:
        hinge = U.load_trainer(hp, "rebuttal_hinge")
        for arm, g in (("g07", 0.7), ("g0", 0.0)):
            over = dict(BASE, gamma_scale=g)
            hinge.BARRIER_TYPE, hinge.BARRIER_MARGIN = "sq_hinge", 2.0
            new.BARRIER_TYPE, new.BARRIER_MARGIN = "sq_hinge", 2.0
            try:
                ra, rb = U.check_pair(rec, f"D_sq_hinge_{arm}_equals_rebuttal_hinge_trainer", hinge,
                                      make_cfg(hinge, barrier_type="sq_hinge", barrier_margin=2.0, **over),
                                      new, make_cfg(new, barrier_type="sq_hinge", barrier_margin=2.0, **over),
                                      [b1, b2], a_only_zero_keys=U.MGC_TELEMETRY)
            finally:
                new.BARRIER_TYPE, new.BARRIER_MARGIN = "softplus", 2.0
        # the hinge barrier actually changes the objective (guards against a no-op switch)
        r_soft = U.run_steps(new, make_cfg(new, **BASE), [b1])
        new.BARRIER_TYPE = "sq_hinge"
        try:
            r_hinge = U.run_steps(new, make_cfg(new, barrier_type="sq_hinge", **BASE), [b1])
        finally:
            new.BARRIER_TYPE = "softplus"
        rec.record("D_sq_hinge_changes_loss_block", [l["loss_block"] for l in r_soft["stats"]]
                   != [l["loss_block"] for l in r_hinge["stats"]])
        del hinge
    else:
        rec.skip("D_sq_hinge_equals_rebuttal_hinge_trainer", "FF_REBUTTAL_TRAINERS not set")

    # E. F1 flags == F1 fork
    f1p = rec.ref("f1_fork_trainer", os.environ.get("FF_F1_TRAINER"), U.F1_SHA)
    if f1p:
        f1 = U.load_trainer(f1p, "f1_fork")
        for arm, over in (("b_curr_aux_off", dict(curr_aux_off=True)),
                          ("c_mgc_keep_l0_aux", dict(use_mgc=True, mgc_keep_l0_aux=True))):
            U.check_pair(rec, f"E_f1_{arm}_equals_f1_fork", f1, make_cfg(f1, **over, **BASE),
                         new, make_cfg(new, **over, **BASE), [b1, b2], a_only_zero_keys=U.MGC_TELEMETRY)
        del f1
    else:
        rec.skip("E_f1_flags_equal_f1_fork", "FF_F1_TRAINER not set")

    # F. semantics of the flags on the camera-ready trainer
    expected = {"cumulative": ({}, [0.25, 0.5, 0.75, 1.0]),
                "mgc": (dict(use_mgc=True), [0.0] * 4),
                "f1_b": (dict(curr_aux_off=True), [0.0] * 4),
                "f1_c": (dict(use_mgc=True, mgc_keep_l0_aux=True), [0.25, 0.0, 0.0, 0.0])}
    for arm, (over, exp) in expected.items():
        cfg = make_cfg(new, **over, **BASE)
        r = U.run_steps(new, cfg, [b1])
        lam = [implied_lambda(cfg, l) for l in r["stats"]]
        err = max(abs(a - b) for a, b in zip(lam, exp))
        mgc_terms = [l.get("loss_mgc") for l in r["stats"]]
        if cfg.use_mgc:
            comp_ok = mgc_terms[0] == 0.0 and all(v > 0.0 for v in mgc_terms[1:])
        else:
            comp_ok = all(v is None for v in mgc_terms)  # MGC metrics absent when MGC is off
        rec.record(f"F_implied_current_block_weight_{arm}", err < 1e-3 and comp_ok,
                   {"implied": [round(v, 6) for v in lam], "expected": exp, "max_err": err,
                    "loss_mgc_per_block": mgc_terms})

    # G. main(): planned config vs the real runs' config.json; refusals
    root = os.environ.get("FF_REFERENCE_RUNS")
    mr = os.path.join(root, "mgc_rebuttal", "from_colab", "runs_from_colab") if root else None
    runs_ref = {
        "mgc_d128_seed123": (dict(FF_USE_MGC="1", FF_MGC_C0="1.0", FF_MGC_RHO="0.0",
                                  FF_GAMMA_GATING_MODE="constant", FF_GAMMA_SCALE="0.7",
                                  FF_VERSION_TAG="mgc_c1", FF_SEED="123"),
                             "ff_cifar10_mgc_c1_L4_D128_bs512_seed123_20260723_205123"),
        "hinge_g07_seed42": (dict(FF_BARRIER_TYPE="sq_hinge", FF_BARRIER_MARGIN="2.0",
                                  FF_GAMMA_GATING_MODE="constant", FF_GAMMA_SCALE="0.7",
                                  FF_VERSION_TAG="hinge_g07", FF_SEED="42"),
                             "ff_cifar10_hinge_g07_L4_D128_bs512_seed42_20260723_214017"),
        "hinge_g0_seed456": (dict(FF_BARRIER_TYPE="sq_hinge", FF_BARRIER_MARGIN="2.0",
                                  FF_GAMMA_GATING_MODE="constant", FF_GAMMA_SCALE="0.0",
                                  FF_VERSION_TAG="hinge_g0", FF_SEED="456"),
                             "ff_cifar10_hinge_g0_L4_D128_bs512_seed456_20260725_210209"),
    }
    for tname, (env, run) in runs_ref.items():
        path = os.path.join(mr, run, "config.json") if mr else None
        U.check_config_vs_run(rec, f"G_config_equals_run_{tname}", new_path, env, path, CR_NEW_CFG_KEYS)
    kpath = (os.path.join(root, "revised_training", "extracted", "ff_cifar10_gated_multiseed",
                          "runs_gated_k0_seed123",
                          "ff_cifar10_cifar10_gated_k0_seed123_L4_D128_bs512_seed123_20260426_071309",
                          "config.json") if root else None)
    U.check_config_vs_run(rec, "G_config_equals_run_gated_k0_seed123_flags_off", new_path,
                          dict(FF_VERSION_TAG="cifar10_gated_k0_seed123", FF_SEED="123"), kpath,
                          CR_NEW_CFG_KEYS)
    cfg_mgc, keys_mgc = U.planned_config(new_path, dict(FF_USE_MGC="1", FF_SEED="42"))
    cfg_off, keys_off = U.planned_config(new_path, dict(FF_SEED="42"))
    v1_keys = U.load_trainer(new_path, "keys_probe").FAIR_LAYER_KEYS
    rec.record("G_layer_metric_keys_v1_when_mgc_off", keys_off == v1_keys and "loss_mgc" not in keys_off)
    i = keys_mgc.index("loss_block_curr")
    rec.record("G_layer_metric_keys_include_mgc_when_on",
               keys_mgc[i + 1:i + 4] == ["loss_mgc", "mgc_lam_mean", "mgc_lam_p95"], keys_mgc[i:i + 5])
    base_env = dict(FF_GAMMA_GATING_MODE="constant", FF_GAMMA_SCALE="0.7", FF_SEED="42")
    for tname, env in (("keep_l0_aux_without_mgc", dict(base_env, FF_MGC_KEEP_L0_AUX="1")),
                       ("curr_aux_off_with_mgc", dict(base_env, FF_USE_MGC="1", FF_CURR_AUX_OFF="1")),
                       ("sq_hinge_with_mgc", dict(base_env, FF_USE_MGC="1", FF_BARRIER_TYPE="sq_hinge")),
                       ("unknown_barrier", dict(base_env, FF_BARRIER_TYPE="hinge"))):
        U.check_refuses(rec, f"G_refuses_{tname}", new_path, env)

    # R. RNG save/restore
    if v1 is not None:
        U.run_rng_tests(rec, "R", new, v1, make_cfg(new, use_mgc=True, **BASE), b1, b2)
    else:
        rec.skip("R_rng_tests", "FF_V1_TRAINERS not set")
    rc = rec.finish(out_json)
    U.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
