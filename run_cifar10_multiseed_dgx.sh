#!/usr/bin/env bash
# Multi-seed reproduction dispatcher for the headline CIFAR-10 + L8/D128
# results in the paper. Covers exactly the runs cited as multi-seed in
# the supplement manifest:
#
#   Tier A   gamma=0  L4/D256, seeds 42, 123, 456     (3-seed; tab:gamma0_multiseed)
#   Tier B   CP-FAIR  L4/D256, seeds 42, 123, 456, 789 (4-seed; tab:multiseed)
#   Tier C   gated kappa=0 L4/D128, seeds 42, 123, 456 (3-seed; tab:gated_k0_multiseed L4 row)
#   Tier D   gated kappa=0 L8/D128, seeds 42, 123, 456 (3-seed; tab:gated_k0_multiseed L8 row)
#
# Usage:
#   bash run_cifar10_multiseed_dgx.sh                  # all tiers, sequential GPU 0
#   bash run_cifar10_multiseed_dgx.sh --tier A         # tier A only (or B/C/D)
#   bash run_cifar10_multiseed_dgx.sh --tier A,B       # multiple tiers
#   bash run_cifar10_multiseed_dgx.sh --parallel       # one job per GPU starting at 0
#   bash run_cifar10_multiseed_dgx.sh --gpu 3          # pin to a specific GPU
#   bash run_cifar10_multiseed_dgx.sh --dry-run        # print commands, don't execute
#
# Each run writes metrics to ./out/<tag>/runs/, stdout to ./out/<tag>/stdout.log,
# and checkpoints to ./out/<tag>/checkpoints/. Already-complete runs (detected
# by stage2_best.pt) are skipped so you can re-invoke to resume.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINERS_DIR="${TRAINERS_DIR:-${SCRIPT_DIR}/trainers}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/out}"
PY="${PY:-python3}"

MODE="sequential"   # sequential | parallel
GPU_OVERRIDE=""
DRY_RUN=0
TIER_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel) MODE="parallel"; shift ;;
    --gpu)      GPU_OVERRIDE="$2"; shift 2 ;;
    --tier)     TIER_FILTER="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -n 25
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Each job = tag | script | extra env vars (space-separated KEY=VAL)
# Tag prefix encodes the tier (A/B/C/D) so --tier filtering works.
JOBS=(
  # Tier A: gamma=0 L4/D256, 3 seeds
  # (seed 42 as published: block_curr_lambda=0 and depth_order_lambda=0; seeds 123/456 use the defaults, erratum D7)
  "A_gamma0_seed42  | cp_fair_cifar10.py        | FF_GAMMA_SCALE=0.0 FF_BLOCK_CURR_LAMBDA=0 FF_DEPTH_ORDER_LAMBDA=0 FF_SEED=42"
  "A_gamma0_seed123 | cp_fair_cifar10.py        | FF_GAMMA_SCALE=0.0 FF_SEED=123"
  "A_gamma0_seed456 | cp_fair_cifar10.py        | FF_GAMMA_SCALE=0.0 FF_SEED=456"
  # Tier B: CP-FAIR L4/D256, 4 seeds (defaults are CP-FAIR)
  "B_cp_fair_seed42  | cp_fair_cifar10.py       | FF_SEED=42"
  "B_cp_fair_seed123 | cp_fair_cifar10.py       | FF_SEED=123"
  "B_cp_fair_seed456 | cp_fair_cifar10.py       | FF_SEED=456"
  "B_cp_fair_seed789 | cp_fair_cifar10.py       | FF_SEED=789"
  # Tier C: gated kappa=0 L4/D128, 3 seeds
  # (seed 42 as published ran with SAM off; seeds 123/456 with SAM on, erratum D13)
  "C_gated_k0_L4D128_seed42  | cp_fair_hardness_gated.py | FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0 FF_NUM_BLOCKS=4 FF_D_MODEL=128 FF_USE_SAM=0 FF_SEED=42"
  "C_gated_k0_L4D128_seed123 | cp_fair_hardness_gated.py | FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0 FF_NUM_BLOCKS=4 FF_D_MODEL=128 FF_SEED=123"
  "C_gated_k0_L4D128_seed456 | cp_fair_hardness_gated.py | FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0 FF_NUM_BLOCKS=4 FF_D_MODEL=128 FF_SEED=456"
  # Tier D: gated kappa=0 L8/D128, 3 seeds (all three ran with SAM off)
  "D_gated_k0_L8D128_seed42  | cp_fair_hardness_gated.py | FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0 FF_NUM_BLOCKS=8 FF_D_MODEL=128 FF_USE_SAM=0 FF_SEED=42"
  "D_gated_k0_L8D128_seed123 | cp_fair_hardness_gated.py | FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0 FF_NUM_BLOCKS=8 FF_D_MODEL=128 FF_USE_SAM=0 FF_SEED=123"
  "D_gated_k0_L8D128_seed456 | cp_fair_hardness_gated.py | FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0 FF_NUM_BLOCKS=8 FF_D_MODEL=128 FF_USE_SAM=0 FF_SEED=456"
)

# Apply --tier filter if given (comma-separated list of A/B/C/D)
if [[ -n "${TIER_FILTER}" ]]; then
  IFS=',' read -ra REQUESTED <<< "${TIER_FILTER}"
  FILTERED=()
  for job in "${JOBS[@]}"; do
    tag="${job%%|*}"; tag="${tag## }"; tag="${tag%% *}"
    tier="${tag%%_*}"
    for r in "${REQUESTED[@]}"; do
      if [[ "${r}" == "${tier}" ]]; then
        FILTERED+=("${job}")
        break
      fi
    done
  done
  JOBS=("${FILTERED[@]}")
fi

mkdir -p "${OUT_ROOT}"

run_one() {
  local tag="$1" script="$2" extra_env="$3" gpu_id="$4"
  local out_dir="${OUT_ROOT}/${tag}"
  local log_file="${out_dir}/stdout.log"
  local done_marker="${out_dir}/checkpoints/stage2_best.pt"

  mkdir -p "${out_dir}/runs" "${out_dir}/checkpoints"

  if [[ -f "${done_marker}" ]]; then
    echo "[skip] ${tag}: stage2_best.pt already exists at ${done_marker}"
    return 0
  fi

  local gpu_prefix=""
  if [[ -n "${gpu_id}" ]]; then
    gpu_prefix="CUDA_VISIBLE_DEVICES=${gpu_id}"
  fi

  local cmd="${gpu_prefix} ${extra_env} FF_RUNS_DIR=${out_dir}/runs FF_CKPT_DIR=${out_dir}/checkpoints FF_VERSION_TAG=cifar10_${tag} ${PY} -u ${TRAINERS_DIR}/${script}"

  echo "[run ] ${tag}"
  echo "       cmd: ${cmd}"
  echo "       log: ${log_file}"

  if [[ ${DRY_RUN} -eq 1 ]]; then
    return 0
  fi

  # start-time + end-time markers for simple accounting
  local t0
  t0=$(date +%s)
  bash -c "${cmd}" > "${log_file}" 2>&1
  local rc=$?
  local t1
  t1=$(date +%s)
  echo "[done] ${tag}: rc=${rc}, elapsed=$(( (t1 - t0) / 60 )) min"
  return ${rc}
}

if [[ "${MODE}" == "sequential" ]]; then
  local_gpu="${GPU_OVERRIDE}"
  for job in "${JOBS[@]}"; do
    IFS='|' read -r tag script extra <<< "${job}"
    tag=$(echo "${tag}" | xargs)
    script=$(echo "${script}" | xargs)
    extra=$(echo "${extra}" | xargs)
    run_one "${tag}" "${script}" "${extra}" "${local_gpu}" || echo "[warn] ${tag} exited non-zero; continuing"
  done
else
  # parallel: one job per GPU starting at gpu 0
  pids=()
  i=0
  for job in "${JOBS[@]}"; do
    IFS='|' read -r tag script extra <<< "${job}"
    tag=$(echo "${tag}" | xargs)
    script=$(echo "${script}" | xargs)
    extra=$(echo "${extra}" | xargs)
    gpu="${GPU_OVERRIDE:-${i}}"
    run_one "${tag}" "${script}" "${extra}" "${gpu}" &
    pids+=($!)
    i=$((i + 1))
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  if [[ ${fail} -eq 1 ]]; then
    echo "[warn] one or more parallel jobs exited non-zero; see per-run stdout.log files"
  fi
fi

echo
echo "=== CIFAR-10 multi-seed dispatch complete ==="
echo "Results root: ${OUT_ROOT}"
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag _ _ <<< "${job}"
  tag=$(echo "${tag}" | xargs)
  log="${OUT_ROOT}/${tag}/stdout.log"
  if [[ -f "${log}" ]]; then
    echo
    echo "--- ${tag} (last 5 lines) ---"
    tail -n 5 "${log}"
  else
    echo "[miss] ${tag}: no log at ${log}"
  fi
done
