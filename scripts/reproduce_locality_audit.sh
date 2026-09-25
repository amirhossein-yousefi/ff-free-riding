#!/usr/bin/env bash
# Asserts strict block-local gradient flow for the image-domain trainers (CPU, a few minutes).
#  1. cumulative-goodness path (v1 check), all four image trainers + the shared block library;
#  2. depth-order path on the real Stage-1 loop (camera-ready check). If FF_V1_TRAINERS points at
#     the v1 supplement's trainers/, it also asserts bit-identical parameters versus v1.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
OUT="${1:-locality_depth_order.json}"

for m in cifar10 cifar100 tiny gated text; do
  python analysis/verify_locality.py --module "$m"
done
for m in cifar10 cifar100 tiny gated text; do
  python analysis/verify_locality.py --module "$m" --depth-order --json "$OUT"
done
echo "Locality contract verified (cumulative and depth-order paths) across cifar10, cifar100, tiny, gated, text."
