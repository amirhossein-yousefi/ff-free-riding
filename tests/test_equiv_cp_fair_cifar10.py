#!/usr/bin/env python3
"""Equivalence tests for trainers/cp_fair_cifar10.py (CIFAR-10, L4/D256, 32-expert MoE). CPU only.

The camera-ready changes to this trainer are RNG state in checkpoints and the no_grad wrap of
the previous-block depth-order aggregation (P3); no objective was added.
A. Flags off == v1: one Stage-1 step on the real CP-FAIR config (gamma=0.7, depth-scaled
   current-block term, SAM on) and on the gamma=0 config of seeds 123/456 (bitwise).
B. Depth-order locality (P3): three consecutive steps, SAM on and off (bitwise equal to v1).
R. RNG save/restore in checkpoints.
F2. The planned F2 run (CIFAR-10 gamma=0 seed 42 in the seed-123/456 config) was prepared with a
   fork that adds only RNG save/restore; the camera-ready trainer reproduces that fork's step
   bitwise, and its planned config.json equals the seed-123 run's config.json except for the seed.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equiv_utils as U  # noqa: E402

NAME = "cp_fair_cifar10.py"
CONFIGS = {
    "cp_fair_default_samon": dict(),
    "gamma0_seed123_456_config": dict(gamma_scale=0.0),
    "cp_fair_samoff": dict(use_sam=False),
}


def main():
    rec = U.Recorder(os.path.basename(__file__))
    out_json = sys.argv[1] if len(sys.argv) > 1 else None
    new, v1, (b1, b2, b3) = U.run_simple_suite(rec, NAME, CONFIGS,
                                               ["cp_fair_default_samon", "cp_fair_samoff"],
                                               "gamma0_seed123_456_config")
    f2p = rec.ref("f2_fork_trainer", os.environ.get("FF_F2_TRAINER"), U.F2_SHA)
    if f2p:
        f2 = U.load_trainer(f2p, "f2_fork")
        U.check_pair(rec, "F2_step_equals_f2_fork", f2, U.make_cfg(f2, gamma_scale=0.0), new,
                     U.make_cfg(new, gamma_scale=0.0), [b1, b2])
    else:
        rec.skip("F2_step_equals_f2_fork", "FF_F2_TRAINER not set")
    root = os.environ.get("FF_REFERENCE_RUNS")
    p = (os.path.join(root, "revised_training", "out", "gamma0_seed123", "runs",
                      "ff_cifar10_cifar10_gamma0_seed123_L4_D256_bs512_seed123_20260417_235110",
                      "config.json") if root else None)
    U.check_config_vs_run(rec, "G_config_equals_run_c10_gamma0_seed123", os.path.join(U.TRAINERS, NAME),
                          dict(FF_GAMMA_SCALE="0", FF_VERSION_TAG="cifar10_gamma0_seed123", FF_SEED="123"),
                          p, set())
    rc = rec.finish(out_json)
    U.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
