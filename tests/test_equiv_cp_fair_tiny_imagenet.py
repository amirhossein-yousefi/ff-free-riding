#!/usr/bin/env python3
"""Equivalence tests for trainers/cp_fair_tiny_imagenet.py (Tiny ImageNet, 64x64, L4/D256 dense).

The camera-ready changes to this trainer are RNG state in checkpoints and the no_grad wrap of
the previous-block depth-order aggregation (P3). CPU only.
A. Flags off == v1: one Stage-1 step on the reported configuration (the v1 defaults: dense MLP,
   no SAM, no depth ordering) and with depth ordering + SAM on (exercises the P3 path).
B. Depth-order locality (P3): three consecutive steps with depth ordering on (bitwise).
R. RNG save/restore in checkpoints.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equiv_utils as U  # noqa: E402

NAME = "cp_fair_tiny_imagenet.py"
CONFIGS = {
    "reported_defaults": dict(),
    "depth_order_on_samon": dict(depth_order_lambda=0.2, use_sam=True),
    "depth_order_on_samoff": dict(depth_order_lambda=0.2, use_sam=False),
}


def main():
    rec = U.Recorder(os.path.basename(__file__))
    out_json = sys.argv[1] if len(sys.argv) > 1 else None
    U.run_simple_suite(rec, NAME, CONFIGS, ["depth_order_on_samon", "depth_order_on_samoff"],
                       "depth_order_on_samoff", batch_size=8)
    rc = rec.finish(out_json)
    U.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
