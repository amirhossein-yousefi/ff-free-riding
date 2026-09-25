#!/usr/bin/env bash
# Reproduce the headline gamma=0 rows of Tab. 1 (CIFAR-10, L4/D256, 3 seeds).
# Each run writes runs/<run_dir>/ (events.jsonl, layer_metrics.jsonl, predictions) and checkpoints/.
# Wall-clock depends on the GPU; each run's events.jsonl records its timestamps.
set -euo pipefail
cd "$(dirname "$0")/.."

# Seed 42 as published: block_curr_lambda=0 and depth_order_lambda=0 (erratum D7);
# seeds 123/456 use the trainer defaults (0.25 / 0.2). All three use SAM.
FF_GAMMA_SCALE=0.0 FF_BLOCK_CURR_LAMBDA=0 FF_DEPTH_ORDER_LAMBDA=0 FF_SEED=42 \
  python trainers/cp_fair_cifar10.py
for SEED in 123 456; do
  FF_GAMMA_SCALE=0.0 FF_SEED=$SEED python trainers/cp_fair_cifar10.py
done
python analysis/aggregate_multiseed.py --runs_dir runs --tag gamma0
echo "Aggregated metric summary: metric_summaries/aggregated_results.json [c10_gamma0]"
echo "Expected (mean +/- sample SD): S1_no_tta = 90.50 +/- 0.05; S1_tta = 91.32 +/- 0.19"
