#!/usr/bin/env python3
"""Equivalence tests for trainers/cp_fair_core.py (shared FF block library; its own main() is an
exploratory D384 Tiny ImageNet configuration that backs no reported number). CPU only.

The camera-ready changes are RNG state in checkpoints and the P3 no_grad wrap. Because the
module's default configuration (D384, 32 experts, 64x64) is slow on CPU, the tests use a
reduced configuration; the code paths are identical.
A. Flags off == v1: one Stage-1 step, SAM on and off (bitwise).
B. Depth-order locality (P3): three consecutive steps (bitwise).
R. RNG save/restore in checkpoints.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equiv_utils as U  # noqa: E402

NAME = "cp_fair_core.py"
SMALL = dict(d_model=128, num_heads=4, n_experts=4, moe_top_k=2, proj_dim=128, label_embed_dim=32,
             stem_channels=128, head_hidden=256, queue_size=1024, queue_neg_samples=256,
             hard_negative_k_first=8, hard_negative_k_last=16)
CONFIGS = {
    "reduced_samon": dict(SMALL, use_sam=True),
    "reduced_samoff": dict(SMALL, use_sam=False),
}


def main():
    rec = U.Recorder(os.path.basename(__file__))
    out_json = sys.argv[1] if len(sys.argv) > 1 else None
    rec.out["config_note"] = "reduced configuration: " + str(SMALL)
    U.run_simple_suite(rec, NAME, CONFIGS, ["reduced_samon", "reduced_samoff"], "reduced_samoff",
                       batch_size=8)
    rc = rec.finish(out_json)
    U.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
