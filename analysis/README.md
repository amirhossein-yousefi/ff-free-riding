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

All scripts run on CPU. Hide GPUs with `CUDA_VISIBLE_DEVICES=''` where a script imports torch.

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
| `make_tables.py`, `make_figures.py` | the auto-generated LaTeX tables and the camera-ready figures |
| `check_numbers.py` | traces every numeral added to the paper since the submitted version to a results JSON |
| `compute_budget.py` | `compute_budget.json` (device and GPU-hours of every reported run, from `env.json`, `epoch_metrics.jsonl` timestamps and runner logs; the checklist's compute statement) |

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

Camera-ready CPU experiments, with their registered predictions and README in the camera-ready
repository. `ffcr_common.py` holds the run/checkpoint registry shared by E1 tier 2.

| Folder | Experiment | Results |
|---|---|---|
| `e1_sep/` | E1: separation versus accuracy at each run's selected checkpoint (tier 1: logged values at the best-validation epoch; tier 2: recomputed on the EMA checkpoints) | `e1_tier1_sep_acc.json` (tier 2 still running at the time of writing) |
| `e3_fd/` | E3: F_d and gradient attenuation on EMA weights with the correct normalization | `e3_fd_ema.json` |
| `e6_l8/` | E6: test evaluation of the L8/D128 κ=0 seed-42 checkpoints | `e6_l8_k0_s42.json` |
