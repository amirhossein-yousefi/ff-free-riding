# Analysis scripts

The top-level scripts in this folder are the v1 supplement's (aggregation, locality audit,
gradient-attenuation snapshot and plot, paired-bootstrap statistics, figure generation). The
subfolders hold the scripts behind the camera-ready numbers. No script contains a hard-coded
location; paths come from environment variables:

| Variable | Meaning |
|---|---|
| `FF_HOME` | Root of the run archive. It contains `amir-porjects/` (the original project layout: run directories with `config.json`, `events.jsonl`, `layer_metrics.jsonl`, `split_indices.json`, prediction files) and `ff-next/checkpoints/trio`. Read-only. Every run path in `repro_manifest.csv` and `metric_summaries/camera_ready/*.json` is relative to it. |
| `FF_WORK_DIR` | Working directory for dumps, extracted checkpoints and intermediate files (default `./ff_work`). |
| `FF_OUT_DIR` | Output root for regenerated results (default `./ff_out`; results go to `<FF_OUT_DIR>/results/*.json`, tables to `<FF_OUT_DIR>/tables`). |
| `FF_PAPER_DIR` | `camera_ready/check_numbers.py` only: a git checkout of the paper sources (with `FF_BASELINE_COMMIT`). |
| `FF_F1_DIR`, `FF_F2_DIR` | The F1 / F2 experiment folders of the run archive: `runs/` (run directories), `logs/` (job logs), the launch specification (`arms.json` / `jobs.json`, `config_diffs/`) and the fork trainer. Default `$FF_HOME/amir-porjects/neuroips_review/camera_ready/experiments/{f1_mgc_block0,f2_gamma0_s42}`. Read-only. |

All scripts run on CPU. Hide GPUs with `CUDA_VISIBLE_DEVICES=''` where a script imports torch.

**Figure and theorem numbers in the v1 scripts.** The top-level v1 scripts
(`generate_figures.py`, `plot_grad_attenuation.py`, `scripts/plot_dynamics.py`) and
`recovered/gradshare/measure_grad_share.py` keep the submitted version's numbering in their section
headers, plot titles and descriptions ("Figure 2", "Figure 5", "Figure 7 (Appendix)", "Prop.~1",
"Theorem 1", and so on). Those numbers do not match the camera-ready paper; use the LaTeX labels.
Examples: separation across depth is `fig:freeriding` (App. H.1), goodness magnitude vs. separation
is `fig:magnitude_vs_sep` (App. N), MoE routing is `fig:moe_routing` (App. H.5), and the
attenuation theorem is `thm:attenuation` (Theorem 3.1). Figure 1 of the camera-ready paper,
`fig:grad_attenuation`, is rendered by `camera_ready/make_figures.py`, not by `plot_grad_attenuation.py`.
`figure_regen/` holds the camera-ready copies of the plotting scripts behind Figs. 4–9, 11–19 and 24. The changes are presentation-only except in the Tiny ImageNet Figs. 17 and 19: Fig. 19 now plots every logged segment, and Fig. 17 recomputes its Block-0 points from the logs (App. O, item B16). Its `README.md` gives each script's figure, raw logs and command.

Several scripts, and the `source` fields of a few `metric_summaries/camera_ready/*.json`, name
`amir-porjects/ff-wiki/_build/analyses/`. That directory is the author's internal archive of the
rebuttal-era analysis JSONs (the outputs of the `recovered/` scripts); "wiki" in check labels, such
as the gates of `e1_tier1_sep_acc.json`, refers to it. It is not distributed: the values the paper
takes from it are carried in `metric_summaries/camera_ready/`.

## `camera_ready/`

| Script | Produces |
|---|---|
| `config.py` | every run directory used below, named explicitly (timestamps included) so that aborted duplicate runs next to the result-bearing ones are never picked up |
| `aggregate.py` | `c100_trio`, `c100_trio_paired`, `c10_paired`, `mgc_c100`, `mgc_d128`, `mgc_paired_bootstrap`, `churn`, `hinge`, `bp_calibration`, `s2ff_mgc`, `probes`, `readout_ladder`, `gradient_share`, `redistribution` (all in `metric_summaries/camera_ready/`) from raw run logs and the rebuttal analysis JSONs; sample SD throughout, 5000-resample paired bootstraps |
| `d_errata_values.py` | `d_errata_values.json` (errata D1, D2, D4, D5, D7, D13 from raw run logs) |
| `count_params_d3.py` | `d3_param_counts.json` (total / active / non-expert parameter counts from checkpoints) |
| `pat_values.py`, `phase3_framing_values.py` | values quoted in the camera-ready text |
| `make_tables.py`, `make_figures.py` | the auto-generated LaTeX tables and the camera-ready figures, including the aligned-checkpoint separation table (E1), the early-exit table and figure (E2), the MGC control table (F1, Table 6) and the γ=0 multi-seed table with the seed-42 rerun (F2, Table 22). With `FF_OUT_DIR/results` holding the files of `metric_summaries/camera_ready/`, `make_tables.py` regenerates all 13 generated tables of the paper byte for byte. |
| `f1_analysis.py` | `f1_block0_control.json`: the MGC control (F1). Per-run accuracies (cross-checked against the prediction files), last-epoch per-block separation, hardware and resume checks, config checks against the launch specification; per-arm mean ± sample SD; seed-paired differences and pooled example-level paired bootstraps (the `aggregate.py` procedure) for five contrasts; the separation decomposition; the scoring of the registered predictions; the F_d snapshot (`experiments/f1_mgc_block0/`). Reads `mgc_d128.json` and `e3_fd_ema.json` from `FF_OUT_DIR/results`, else from `metric_summaries/camera_ready/`. |
| `f2_analysis.py` | `f2_gamma0_s42.json`: the γ=0 seed-42 rerun (F2). Stitches its three Colab segments (resume events, epochs trained twice), reads the four test protocols and the deepest-block separation, compares with the published seed 42 and seeds 123/456, computes the matched and published n=3, and scores the registered predictions. |
| `early_exit.py` | the early-exit summaries that `make_tables.py` and `make_figures.py` share, all derived from `e2_exit.json` and asserted equal to the selector's own family aggregates (the figure's values: `fig_early_exit_pareto_values.json`) |
| `check_numbers.py` | traces every numeral added to the paper since the submitted version to a results JSON |
| `compute_budget.py` | `compute_budget.json` (device and GPU-hours of every reported run, including the F1 and F2 runs, from `env.json`, `epoch_metrics.jsonl` timestamps and runner logs, and the CPU time of the camera-ready recomputations, including the E1 tier-2 / E2 batch from its log and the F1 F_d snapshot; the checklist's compute statement) |

## `readouts/`

`s2ff_readout.py` (one extra FF block on the frozen Stage-1 backbone, trained only with the local
goodness objective) and `s2ff_ce_readout.py` (the same block trained with sampled-softmax
cross-entropy). Both default to `trainers/cp_fair_cifar100.py` to rebuild the backbone; the
rebuttal runs used `mgc_c100_trainer.py`, which builds the identical network. Results:
`metric_summaries/camera_ready/s2ff_mgc.json`.

## `recovered/`

Rebuttal-era scripts behind results the paper cites. Most of them lived only in a session
scratchpad and were **reconstructed** by replaying the recorded file writes and edits; they are
reconstructions, not the original files, and they were not re-executed in the camera-ready pass.
Their outputs (the rebuttal analysis JSONs) are the inputs of `camera_ready/aggregate.py`.

| Script | Origin | Behind |
|---|---|---|
| `dump_block_data.py`, `dump_val_test_block_data.py`, `extract_labelfree_pooled.py`, `extract_probe_feats_c100.py` | reconstructed | per-block goodness / feature dumps of the CIFAR-100 trio (inputs of the probes and readout ladder) |
| `clean_probes_c100.py`, `clean_probes_addendum.py`, `merge_clean_probes.py`, `summarize_clean_probes.py` | reconstructed | `probes.json` (tab:probes) |
| `val_refit_readout.py` | reconstructed | `readout_ladder.json` (tab:readout_ladder) |
| `gradshare/measure_grad_share.py`, `gradshare/summarize.py` | reconstructed | `gradient_share.json` (tab:gradshare) |
| `fd_figure.py`, `paired_bootstrap_cifar10.py`, `redistribution_demo.py` | original files | F_d statistics (Fig. 1), `c10_paired.json`, `redistribution.json` |
| `verify_ckpts.py`, `verify_drive_ckpts.py` | original files | checkpoint identity checks cited in `RELEASE_CHECKPOINTS.md` |

## `experiments/`

Camera-ready CPU experiments; the registered predictions and READMEs of E1–E6 stay in the camera-ready
repository. For the two camera-ready GPU experiments, F1 and F2, the registered predictions are
included (`f1_mgc_block0/PREDICTIONS.md`, `f2_gamma0_s42/PREDICTIONS.md`); both are byte-identical to
the files written before the runs started, and `camera_ready/f1_analysis.py` and
`camera_ready/f2_analysis.py` score them exactly as written. Their plan-item labels (C3, C4, X1/D7)
and `CR/` paths refer to the camera-ready repository. `ffcr_common.py` holds the run/checkpoint registry shared by E1 tier 2 and E2.
`run_cpu_batch.sh` runs both as one sequential batch on the CPU (CUDA hidden, six torch threads). It
needs `FF_HOME`; `FF_WORK_DIR` and `FF_OUT_DIR` are optional. Per-run outputs go to
`<FF_OUT_DIR>/experiments/` (`e1_sep/out/`, `e2_exit/dumps/`, `logs/`) and the result JSONs to
`<FF_OUT_DIR>/results/`. The batch behind the paper's numbers took 25.0 wall-clock hours, CPU only
(`compute_budget.json`, `cpu_recomputations.e1_e2_cpu_batch`). The E2 score dumps (about 300 MB) are not
distributed. The κ=0 L4/D128 seed-123/456 points of E1 read their `stage1_best` checkpoints from
`<FF_WORK_DIR>/ckpts/gated_k0_d128/` (extracted from the gated multiseed archive; `RELEASE_CHECKPOINTS.md`).
CR-18 (E1 tier 2) and CR-19 (E2) in docstrings and JSON headers are camera-ready plan item numbers.

| Folder | Experiment | Results |
|---|---|---|
| `e1_sep/` | E1: separation versus accuracy at each run's selected checkpoint (tier 1: logged values at the best-validation epoch; tier 2: recomputed on the EMA checkpoints through the trainer's own diagnostic path) | `e1_tier1_sep_acc.json` (19 points), `e1_sep_tier2.json` (18 points) |
| `e2_exit/` | E2: validation-selected early exit. `dump_val_test_block_data.py` dumps each run's per-depth cumulative class scores on its own validation split and on the test split (identity-gated against the logged test accuracy); `select_exit.py` chooses a truncation depth (rules T, T′) or a dynamic-exit threshold (rule D) on validation only and evaluates each choice once on test | `e2_exit.json` |
| `e3_fd/` | E3: F_d and gradient attenuation on EMA weights with the correct normalization | `e3_fd_ema.json` |
| `e6_l8/` | E6: test evaluation of the L8/D128 κ=0 seed-42 checkpoints | `e6_l8_k0_s42.json` |
| `f1_mgc_block0/` | F1 (MGC control, a GPU experiment): `PREDICTIONS.md` (registered before the runs) and `fd_snapshot_f1.py`, the CPU F_d snapshot on the arm-a and arm-b seed-42 `stage1_best` EMA checkpoints (the E3 pipeline; the checkpoints are not part of the release). Outputs go to `<FF_OUT_DIR>/experiments/f1_mgc_block0/out/fd/`; the variables are listed in its docstring | `f1_block0_control.json` (`fd_gradient_attenuation`, via `camera_ready/f1_analysis.py`) |
| `f2_gamma0_s42/` | F2 (γ=0 seed-42 rerun, a GPU experiment): `PREDICTIONS.md` (registered before the run) | `f2_gamma0_s42.json` (via `camera_ready/f2_analysis.py`) |

Re-running `e2_exit/select_exit.py` on the recorded dumps reproduces `e2_exit.json` exactly, apart
from its path and timing fields. Re-running `e1_sep/aggregate_e1_tier2.py` on the recorded per-point
outputs reproduces `e1_sep_tier2.json` exactly, apart from its path and date fields. The stratified
(within-configuration) test draws its per-configuration permutations in sorted configuration order,
so the permutation p-values do not depend on `PYTHONHASHSEED`. (An earlier version iterated in
Python set order, which moved these p-values by at most 0.004 between re-runs, within the Monte Carlo
error of 100k permutations; none of the values quoted in the paper changed when this was fixed.)

Run on the run archive, `camera_ready/f1_analysis.py` and `camera_ready/f2_analysis.py` reproduce
`f1_block0_control.json` and `f2_gamma0_s42.json` exactly, apart from their date fields, and
`camera_ready/compute_budget.py` (with the optional CPU-timing inputs present) reproduces
`compute_budget.json`. `experiments/f1_mgc_block0/fd_snapshot_f1.py` with the default
`trainers/cp_fair_hardness_gated.py` gives the same arm-b margins, bit for bit, as the F1 fork did.
