#!/usr/bin/env bash
# Re-render the camera-ready figures listed in README.md from the raw run logs (CPU only).
#   FF_HOME=<run archive> bash run_figure_regen.sh OUT_DIR
# Each script writes into its own sub-folder of OUT_DIR; OUT_DIR/paper/ then holds copies under the
# file names that paper/figures/ uses, for a direct comparison.
set -euo pipefail
OUT=${1:?usage: FF_HOME=... bash run_figure_regen.sh OUT_DIR}
: "${FF_HOME:?set FF_HOME to the run archive root (it contains amir-porjects/)}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
mkdir -p "$OUT"/{generate_figures,plot_dynamics,gated_L4,gated_L8,tiny_imagenet,magnitude_vs_separation,paper}

"$PY" "$HERE/generate_figures_cr.py" "$OUT/generate_figures"
FF_FIG_DIR="$OUT/plot_dynamics" "$PY" "$HERE/plot_dynamics_cr.py"
"$PY" "$HERE/analyze_free_riding_L4_cr.py" --output-dir "$OUT/gated_L4"
"$PY" "$HERE/analyze_free_riding_L8_cr.py" --output-dir "$OUT/gated_L8"
"$PY" "$HERE/tinyimagenet_figs_cr.py" "$OUT/tiny_imagenet"
"$PY" "$HERE/magnitude_vs_separation_cr.py" "$OUT/magnitude_vs_separation"

# script output -> paper/figures file name
while read -r src dst; do
  cp "$OUT/$src" "$OUT/paper/$dst"
done <<'EOF'
generate_figures/fig2_freeriding_separation.pdf fig2_freeriding_separation.pdf
generate_figures/fig_scaling_freeriding_L8D128.pdf fig_scaling_freeriding_L8D128.pdf
generate_figures/fig6_training_dynamics.pdf fig6_training_dynamics.pdf
generate_figures/fig7_moe_routing.pdf fig7_moe_routing.pdf
plot_dynamics/fig8_freeriding_onset.pdf fig8_freeriding_onset.pdf
plot_dynamics/fig9_convergence_speed.pdf fig9_convergence_speed.pdf
plot_dynamics/fig10_depth_gap.pdf fig10_depth_gap.pdf
plot_dynamics/fig11_training_stability.pdf fig11_training_stability.pdf
gated_L4/fig5_goodness_decomposition.pdf fig_goodness_decomposition.pdf
gated_L4/fig4_depth_truncation.pdf fig_depth_truncation.pdf
gated_L8/fig5_goodness_decomposition.pdf fig_goodness_decomposition_L8.pdf
gated_L8/fig3_training_dynamics.pdf fig_training_dynamics_L8.pdf
tiny_imagenet/fig_tinyimagenet_crossdata.pdf fig_tinyimagenet_crossdata.pdf
tiny_imagenet/fig_tinyimagenet_perseed.pdf fig_tinyimagenet_perseed.pdf
tiny_imagenet/fig_tinyimagenet_dynamics.pdf fig_tinyimagenet_dynamics.pdf
magnitude_vs_separation/fig5_magnitude_vs_separation.pdf fig5_magnitude_vs_separation.pdf
EOF
echo "figures under $OUT/paper:"
ls "$OUT/paper"
