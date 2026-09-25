#!/usr/bin/env bash
# Reproduce Figure 1 (gradient-attenuation snapshot + plot).
#
# REQUIREMENT: This script needs trained CIFAR-10 checkpoints for the
# three γ variants (no_all_collab, baseline_constant, lcff_prefix) under
# the L4/D128 hardness-gated trainer. Without checkpoints, only the
# tabular summary (Table 4 in the paper) is reproducible from the
# bundled `metric_summaries/grad_attenuation_data.json`; the per-example
# scatter requires re-running snapshot_grad_attenuation.py against the
# checkpoints to produce analysis/grad_attenuation_arrays.npz.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f analysis/grad_attenuation_arrays.npz ] && [ -f analysis/grad_attenuation_data.json ]; then
  echo "[Fig 1] Bundled per-example arrays found; replotting from cached data."
  python analysis/plot_grad_attenuation.py
else
  if [ -z "${FF_CHECKPOINTS_DIR:-}" ] || [ -z "${FF_RUNS_DIR:-}" ]; then
    cat <<'EOF' >&2
[Fig 1] Per-example arrays are not bundled (size budget); re-running the
        snapshot from checkpoints requires:

          FF_CHECKPOINTS_DIR=<path-to-checkpoints>
          FF_RUNS_DIR=<path-to-run-dirs>
          FF_DATA_DIR=<path-to-CIFAR10>
          bash scripts/reproduce_fig1_grad_attenuation.sh

        The bundled metric_summaries/grad_attenuation_data.json lets you
        verify Table 4 (batch-mean diagnostics) without retraining; for
        the per-example scatter (Fig. 1b) the snapshot script is needed.
EOF
    exit 2
  fi
  echo "[Fig 1] Snapshotting per-example gradients from checkpoints..."
  python analysis/snapshot_grad_attenuation.py
  python analysis/plot_grad_attenuation.py
fi
echo "Figure written to fig_grad_attenuation.{pdf,png}"
