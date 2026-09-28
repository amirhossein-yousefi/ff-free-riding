#!/usr/bin/env bash
# CPU batch for the camera-ready experiments E1 tier 2 (CR-18) and E2 (CR-19).
#
#   * CPU only: CUDA_VISIBLE_DEVICES='' for every step; nice -n 19; torch intra-op threads = 6.
#   * Sequential: one checkpoint per Python process (memory is released between steps).
#   * Resumable: every step skips outputs that already exist and passed (E1: out/<name>__<mode>.json
#     with status OK and the same N/data seed; E2: dumps/<name>.json with status PASS + its npz).
#     Re-running the script after an interruption continues where it stopped.
#   * A failing step is logged and the batch continues; the final aggregation lists what is missing.
#   * Single instance: guarded by flock on $FF_OUT_DIR/experiments/logs/run_cpu_batch.lock.
#   * Logs: $FF_OUT_DIR/experiments/logs/batch.log (step summary) and .../<step>.log (per step).
#
# Configuration (environment variables; see analysis/README.md):
#   FF_HOME      root of the run archive (required; read-only)
#   FF_WORK_DIR  working directory for symlinked data and dummy trainer dirs (default ./ff_work)
#   FF_OUT_DIR   output root (default ./ff_out): experiments/e1_sep/out, experiments/e2_exit/dumps,
#                experiments/logs and results/{e1_sep_tier2,e2_exit}.json
#   PYTHON       interpreter (default python3)
#
# Timing estimates (2026-09-25, DGX Spark CPU shared with other CPU jobs; load average ~13):
#   E2 dump, C10 L4/D128 arm (30k image passes x 10 labels) ....... 10 min   x 8 remaining  = ~1.3 h
#   E1 point, C10 D128 (test gate ~4.5-6 min + 20 batches ~3.5 min)  ~10 min   x 2 remaining  = ~0.3 h
#   E1 point, C10 D256 (test gate ~13-25 min + 20 x 53 s)           ~31-43 min x 6           = ~3.1-4.3 h
#   E2 dump, C100 D256 (20k image passes x 100 labels, ~0.25 s/img)  ~83-120 min x 9         = ~12.5-18 h
#   E1 point, C100 (gate reused from E2; 20 x 30 s; peak RSS ~30 GB) ~11 min   x 9           = ~1.7 h
#   aggregation (select_exit.py, aggregate_e1_tier2.py) ............ < 1 min
#   TOTAL ~19 h if the machine is otherwise idle, ~25-27 h under the contention above.
#   (Smoke outputs already present and skipped: E2 dump + E1 point for c10_gated_adaptive_t1_k0_seed42,
#    E1 MGC validation in both modes.)
# Measured: the batch ran from 2026-09-25 07:19 to 2026-09-26 08:20 (25.0 h wall-clock, 42 steps, 0 failed;
# metric_summaries/camera_ready/compute_budget.json, cpu_recomputations.e1_e2_cpu_batch).
#
# Launch (detached; survives the terminal), from the supplement root:
#   mkdir -p ff_out && FF_HOME=/path/to/run-archive FF_OUT_DIR=$PWD/ff_out \
#     nohup setsid bash analysis/experiments/run_cpu_batch.sh > ff_out/run_cpu_batch.out 2>&1 < /dev/null &
# Progress:  tail -f "$FF_OUT_DIR/experiments/logs/batch.log"
# Stop:      pkill -f run_cpu_batch.sh; pkill -f 'measure_sep_tier2.py|dump_val_test_block_data.py'
set -u

: "${FF_HOME:?set FF_HOME to the root of the run archive (see analysis/README.md)}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # this folder: ffcr_common.py, e1_sep/, e2_exit/
# absolute paths: ffcr_common.setup_env() changes the working directory
FF_HOME="$(cd "$FF_HOME" && pwd)" || exit 1
mkdir -p "${FF_WORK_DIR:-ff_work}" "${FF_OUT_DIR:-ff_out}"
FF_WORK_DIR="$(cd "${FF_WORK_DIR:-ff_work}" && pwd)"
FF_OUT_DIR="$(cd "${FF_OUT_DIR:-ff_out}" && pwd)"
export FF_HOME FF_WORK_DIR FF_OUT_DIR
PY="${PYTHON:-python3}"
LOG="$FF_OUT_DIR/experiments/logs"
mkdir -p "$LOG"

exec 9>"$LOG/run_cpu_batch.lock"
if ! flock -n 9; then
    echo "another run_cpu_batch.sh is running (lock $LOG/run_cpu_batch.lock); exiting" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=''
export FFCR_THREADS=6
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
cd "$EXP" || exit 1

N_FAIL=0
step() {  # step <log-name> <script> [args...]
    local name="$1"; shift
    local t0; t0=$(date +%s)
    echo "[$(date '+%F %T')] START $name" | tee -a "$LOG/batch.log"
    nice -n 19 "$PY" "$@" >> "$LOG/$name.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "[$(date '+%F %T')] END   $name rc=$rc (${dt}s)" | tee -a "$LOG/batch.log"
    if [ $rc -ne 0 ]; then N_FAIL=$((N_FAIL + 1)); fi
    return 0
}

echo "[$(date '+%F %T')] ===== run_cpu_batch.sh start (pid $$) =====" | tee -a "$LOG/batch.log"

# ---------------------------------------------------------------- phase A: CIFAR-10 L4/D128
# E1 code validation on MGC-D128 seed 42 (gradshare path, expected block-3 sep ~5.2346), then the
# same checkpoint through the tier-2 EMA path (reported next to the 18 points, not one of them).
step e1_mgc_raw_train  e1_sep/measure_sep_tier2.py --names c10_mgc_c1_seed42 --mode raw_train
step e1_mgc_ema_eval   e1_sep/measure_sep_tier2.py --names c10_mgc_c1_seed42 --mode ema_eval

# E2 dumps: the 9 gated L4/D128 seed-42 arms (val + flipped val + test + flipped test)
for arm in baseline_constant lcff_prefix no_all_collab adaptive_t1_k0 adaptive_t3_k0 \
           adaptive_t1_k2 adaptive_t1_k4 adaptive_prev_only adaptive_prev_t1_k2; do
    step "e2_dump_c10_gated_${arm}_seed42" e2_exit/dump_val_test_block_data.py --names "c10_gated_${arm}_seed42"
done

# E1 tier 2, D128 points (t1_k0 reuses the E2 test gate of the same checkpoint)
for n in c10_gated_adaptive_t1_k0_seed42 c10_gated_k0_seed123 c10_gated_k0_seed456; do
    step "e1_${n}" e1_sep/measure_sep_tier2.py --names "$n" --mode ema_eval
done

# interim aggregation (cheap)
step e2_select_interim e2_exit/select_exit.py

# ---------------------------------------------------------------- phase B: CIFAR-10 L4/D256
for n in c10_cp_fair_seed42 c10_cp_fair_seed123 c10_cp_fair_seed456 c10_cp_fair_seed789 \
         c10_gamma0_seed123 c10_gamma0_seed456; do
    step "e1_${n}" e1_sep/measure_sep_tier2.py --names "$n" --mode ema_eval
done
step e1_aggregate_interim e1_sep/aggregate_e1_tier2.py

# ---------------------------------------------------------------- phase C: CIFAR-100 trio
# per run: E2 dump first (its full test pass is also E1's identity gate), then the E1 point
for v in gamma0 gated_k0 cumulative; do
    for s in 42 123 456; do
        n="c100_${v}_seed${s}"
        step "e2_dump_${n}" e2_exit/dump_val_test_block_data.py --names "$n"
        step "e1_${n}" e1_sep/measure_sep_tier2.py --names "$n" --mode ema_eval
    done
done

# ---------------------------------------------------------------- final aggregation
step e2_select_final     e2_exit/select_exit.py
step e1_aggregate_final  e1_sep/aggregate_e1_tier2.py

echo "[$(date '+%F %T')] ===== run_cpu_batch.sh done: ${N_FAIL} failed step(s) =====" | tee -a "$LOG/batch.log"
exit 0
