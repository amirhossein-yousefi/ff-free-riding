# Camera-ready figure regeneration

These scripts render the data figures of the camera-ready paper that were re-plotted for the
camera-ready version. They are copies of the project's plotting scripts (`../generate_figures.py`,
`../scripts/plot_dynamics.py`, and the gated-ablation and Tiny ImageNet plotting scripts), and each
change is marked `[CR]`/`[CR2]` in the code. The changes are presentation only (legend placement,
axis labels and ticks, a log axis for Fig. 9, larger type, TrueType fonts, paper names in legends,
no "(ours)" label). The logged values and the plotted quantities are unchanged, with two
exceptions, both Tiny ImageNet figures. Fig. 19 now plots every logged segment of each resumed run;
the submitted version showed only the last segments. Fig. 17 recomputes its Block-0 points from the
logs; the submitted points did not match any logged row (App. O, item B16). The two gated-ablation
scripts (`analyze_free_riding_L4_cr.py`, `analyze_free_riding_L8_cr.py`) keep only the code and the
data loads behind their paper figures.

All scripts run on CPU and read only the raw run logs (`layer_metrics.jsonl`,
`epoch_metrics.jsonl`, `events.jsonl`). Paths come from environment variables, as in
`../camera_ready/config.py`:

| Variable | Meaning |
|---|---|
| `FF_HOME` | Root of the run archive (it contains `amir-porjects/`). Required by `run_figure_regen.sh`; a single script can instead be pointed at its data with `FF_BASE_DIR` (every script except `tinyimagenet_figs_cr.py`), `FF_TINY_DIR` (`tinyimagenet_figs_cr.py`) or, for the two `analyze_free_riding_*_cr.py` scripts, `--runs-dir` (the folder that holds their run folders). |
| `FF_BASE_DIR` | CIFAR-10 project folder (default `$FF_HOME/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison`). All CIFAR-10 paths below are relative to it. |
| `FF_TINY_DIR` | Tiny ImageNet project folder (default `$FF_HOME/amir-porjects/tiny_imagenet_fair`). |
| `FF_FIG_DIR` | Output folder when no output argument is given (default `./figures`). |

Run everything (tested with matplotlib 3.11.2 and numpy 2.5.2 from `requirements.txt`):

```bash
FF_HOME=/path/to/run_archive bash analysis/figure_regen/run_figure_regen.sh ff_figs_out
```

Each script writes into its own subfolder of the output directory. `run_figure_regen.sh` then copies
the PDFs into `<out>/paper/` under the file names the paper's LaTeX uses. With matplotlib 3.11.2,
every PDF renders pixel-identical (150 dpi) to the figure in the camera-ready PDF. Only the
creation-date metadata differs.

| Script (command) | Output file → paper file | Paper figure | Raw logs read (under `FF_BASE_DIR` / `FF_TINY_DIR`) |
|---|---|---|---|
| `generate_figures_cr.py OUT` | `fig2_freeriding_separation.pdf` | Fig. 4 (`fig:freeriding`) | `<variant>/runs/<run>/layer_metrics.jsonl` of the baseline, CP-FAIR, Fixed-NL and NoMem-L2 seed-42 runs in `RUN_REGISTRY` (last logged Stage-1 epoch) |
| | `fig_scaling_freeriding_L8D128.pdf` | Fig. 7 (`fig:scaling_L8`) | `basic_fair_revised_valfix_cp_fair/scaling_L8D128_32e/runs_scale_L8D128/<run>/layer_metrics.jsonl` (γ = 0 / 0.7 / 1, epoch 180) |
| | `fig6_training_dynamics.pdf` | Fig. 13 (`fig:training_dynamics`) | `layer_metrics.jsonl` of the four Fig. 4 runs (sep_nl per epoch, epochs > 50) |
| | `fig7_moe_routing.pdf` | Fig. 14 (`fig:moe_routing`) | `epoch_metrics.jsonl` of the CP-FAIR, baseline and Multi-Tech runs in `RUN_REGISTRY` |
| `FF_FIG_DIR=OUT plot_dynamics_cr.py` | `fig8_freeriding_onset.pdf` | Fig. 5 (`fig:freeriding_onset`) | `layer_metrics.jsonl` of the seed-42 CP-FAIR, γ = 0 and LCFF-prefix segments listed in `VARIANTS` |
| | `fig9_convergence_speed.pdf` | Fig. 12 (`fig:convergence_speed`) | `epoch_metrics.jsonl` of the same segments (`eval_top1`) |
| | `fig10_depth_gap.pdf` | Fig. 15 (`fig:depth_gap`) | same (`eval_top1_d2`, `eval_top1_d4`) |
| | `fig11_training_stability.pdf` | Fig. 16 (`fig:training_stability`) | same (`eval_top1`; rolling population SD, ddof = 0, over five evaluations) |
| `analyze_free_riding_L4_cr.py --output-dir OUT` | `fig5_goodness_decomposition.pdf` → `fig_goodness_decomposition.pdf` | Fig. 6 (`fig:goodness_decomposition`) | `basic_fair_revised_valfix_cp_fair/from_colab/basic_fair_revised_valfix_cp_fair_hardness_gated/ff_gated_ablation/runs/<run>/layer_metrics.jsonl` of the four runs in `FIG6_RUNS` (constant γ = 0.7, adaptive τ = 1, κ = 0, γ = 0 and LCFF; L4/D128, seed 42; `gpos` and `gpos_cur` at the last Stage-1 epoch, 180) |
| | `fig4_depth_truncation.pdf` → `fig_depth_truncation.pdf` | Fig. 11 (`fig:depth_truncation`) | `epoch_metrics.jsonl` of all nine runs in `EXPERIMENT_ORDER`, same folder (`eval_top1_d1`–`eval_top1_d4` at the Stage-1 epoch with the highest `eval_top1`) |
| `analyze_free_riding_L8_cr.py --output-dir OUT` | `fig5_goodness_decomposition.pdf` → `fig_goodness_decomposition_L8.pdf` | Fig. 8 (`fig:goodness_decomposition_L8`) | `.../ff_gated_ablation_L8/runs/<run>/layer_metrics.jsonl` of the four adaptive runs in `EXPERIMENT_LABELS` (τ = 1, κ = 0; τ = 3, κ = 0; prev-only; prev τ = 1, κ = 2; L8/D128, seed 42; `gpos` and `gpos_cur` at the last Stage-1 epoch, 180) |
| | `fig3_training_dynamics.pdf` → `fig_training_dynamics_L8.pdf` | Fig. 9 (`fig:training_dynamics_L8`) | the same files of the two cumulative κ = 0 runs in `FIG9_RUNS` (`gpos_cur` of Blocks 0, 1, 3, 5 and 7 at every Stage-1 epoch) |
| `tinyimagenet_figs_cr.py OUT` | `fig_tinyimagenet_crossdata.pdf` | Fig. 17 (`fig:tinyimagenet_crossdata`) | epoch-362 `layer_metrics.jsonl` rows of the final segment of seeds 42/123/456 (`SEGMENTS`) |
| | `fig_tinyimagenet_perseed.pdf` | Fig. 18 (`fig:tinyimagenet_perseed`) | `final_test_eval` events of the same segments |
| | `fig_tinyimagenet_dynamics.pdf` | Fig. 19 (`fig:tinyimagenet_dynamics`) | `epoch_metrics.jsonl` (`best_s1`) of every segment, the earlier ones read from the archive's `from_drive_2` zip |
| `magnitude_vs_separation_cr.py OUT` | `fig5_magnitude_vs_separation.pdf` | Fig. 24 (`fig:magnitude_vs_sep`) | Block-3 `gpos` and `sep_nl` at the last logged Stage-1 epoch of the 12 runs in `generate_figures_cr.py`, and its `TEST_ACC` table |

The analysis scripts write their outputs as `fig4_*`, `fig5_*` and `fig3_*`. `run_figure_regen.sh`
copies these files to the paper file names given after the arrow. Both analysis scripts take the
latest run folder (by timestamp) of each configuration. For the L4 prev-only run, that folder is the
resumed final segment (Stage-1 epochs 175–180 and Stage 2); its best Stage-1 epoch, 180, is also the
best of the whole run.

The submitted generator of Fig. 24 did not survive. `magnitude_vs_separation_cr.py` redraws the
submitted layout: the badge positions are taken from the submitted PDF, and the data come from the
logs. The redrawn figure differs from the submitted one only in the legend label "CP-FAIR" (it was
"CP-FAIR (ours)"). The figures not listed here come from other scripts. Figs. 1, 2, 3 and 10 come
from `../camera_ready/make_figures.py`. Figs. 20–23 are TikZ drawings in the LaTeX source.
