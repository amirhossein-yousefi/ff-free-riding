# Camera-ready result summaries

Each JSON has a `_meta` or `_header` block naming the script that wrote it (see `analysis/README.md`)
and its raw sources. Run paths are relative to `FF_HOME` (the root of the run archive). Spreads are
sample SDs (ddof=1) unless a key says otherwise. The files are copies of the camera-ready
repository's `results/*.json`. Only strings were edited: absolute path prefixes were removed, and in the
F1/F2 files Colab paths are given relative to the Google Drive root (`MyDrive/...`) and the name of the
DGX Spark host is replaced by `<DGX Spark host>`. Every numeric value is unchanged (checked leaf by leaf
when copying). `f2_gamma0_s42.json` abbreviates run paths as `CR/...`
(`amir-porjects/neuroips_review/camera_ready/...`) and `CF/...`
(`amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison/...`). Sources under
`amir-porjects/ff-wiki/_build/analyses/` are the author's internal archive of rebuttal-era analysis
JSONs, which is not distributed (see `analysis/README.md`).

| File | Contents |
|---|---|
| `c100_trio.json`, `c100_trio_paired.json` | CIFAR-100 dissociation trio, four protocols; paired bootstrap CIs |
| `c10_paired.json` | CIFAR-10 CP-FAIR vs γ=0 paired statistics |
| `mgc_c100.json`, `mgc_d128.json`, `mgc_paired_bootstrap.json`, `churn.json` | missing-gradient compensation |
| `hinge.json` | squared-hinge replication |
| `bp_calibration.json` | BP/FF calibration rows of Table 2 (n=3) |
| `s2ff_mgc.json`, `readout_ladder.json`, `probes.json` | frozen-backbone readouts, readout ladder, probes |
| `gradient_share.json`, `redistribution.json` | per-term gradient shares; redistribution demonstration |
| `d_errata_values.json`, `d3_param_counts.json`, `pat_values.json`, `phase3_framing_values.json` | errata values, parameter counts and other quoted values, all read from raw logs |
| `e1_tier1_sep_acc.json`, `e1_sep_tier2.json`, `e3_fd_ema.json`, `e6_l8_k0_s42.json` | camera-ready experiments E1 (tier 1: separation logged at the selected epoch; tier 2: recomputed on the selected checkpoints), E3, E6 |
| `e2_exit.json` | camera-ready experiment E2: validation-selected early exit (per-run operating points, gates, family summaries; test-selected values for reference only) |
| `f1_block0_control.json` | camera-ready experiment F1, the MGC control at CIFAR-10 L4/D128 (Table 6): per-run values, arm means ± sample SD, seed-paired differences and pooled paired bootstraps, the separation decomposition, the scored registered predictions and the F_d snapshot; written by `analysis/camera_ready/f1_analysis.py` |
| `f2_gamma0_s42.json` | camera-ready experiment F2, the CIFAR-10 γ=0 seed-42 rerun in the seed-123/456 configuration (Table 22): segment provenance, test protocols, separation, matched and published n=3, scored registered predictions; written by `analysis/camera_ready/f2_analysis.py` |
| `fd_figure_check.json`, `fig3_4_bpff_values.json`, `fig_accuracy_vs_ratio_values.json`, `fig_early_exit_pareto_values.json` | values plotted in the regenerated figures |
| `fig_dissociation_summary_values.json` | values of a summary figure that is not in the camera-ready PDF (the same values are in Tables 3 and 15) |
| `p3_depth_order_leak_check.json` | the depth-order leak check on the v1 trainers (superseded by `tests/locality_depth_order.json`) |
| `compute_budget.json` | compute budget of the reported runs (device, per-run GPU-hours, totals, including the F1 and F2 runs) from the raw run logs, and the CPU time of the recomputations (E1 tier 2 and E2: per run, and the batch's wall time, CPU only; the F1 F_d snapshot); written by `analysis/camera_ready/compute_budget.py` |
