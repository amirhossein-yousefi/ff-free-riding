#!/usr/bin/env bash
# Reproduce the BP/FF calibration rows of Tab. 2 at n=3 (seeds 42/123/456) with the ORIGINAL run
# scripts in trainers/bp_original/ (seed 42 = the submitted runs; seeds 123/456 = rebuttal runs).
# Pass --seed explicitly: the scripts' argparse default (42) overrides FF_SEED.
# The scripts read CIFAR-10 from ./data_c10 (downloaded by torchvision if absent).
set -euo pipefail
cd "$(dirname "$0")/.."

for SEED in 42 123 456; do
  python trainers/bp_original/plain_cnn_bp_cifar10.py --seed "$SEED"                  # Plain CNN + BP
  python trainers/bp_original/plain_cnn_ff_cifar10.py --seed "$SEED"                  # Plain CNN + FF
  python trainers/bp_original/ff_backbone_stripped_bp_cifar10.py --seed "$SEED"       # FF backbone + BP, weak aug
  python trainers/bp_original/ff_backbone_stripped_bp_strongaug_cifar10.py --seed "$SEED"  # FF backbone + BP, strong aug
done
echo "Expected test top-1 (no TTA, mean +/- sample SD over 3 seeds): see repro_manifest.csv (tab:bp_ff_ablation rows)"
echo "and metric_summaries/camera_ready/bp_calibration.json. trainers/bp_control.py is a compact re-implementation."
