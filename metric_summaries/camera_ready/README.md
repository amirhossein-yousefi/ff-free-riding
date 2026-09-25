# Camera-ready result summaries

Each JSON has a `_meta` or `_header` block naming the script that wrote it (see `analysis/README.md`)
and its raw sources. Run paths are relative to `FF_HOME` (the root of the run archive). Spreads are
sample SDs (ddof=1) unless a key says otherwise. The files are copies of the camera-ready
repository's `results/*.json`; only absolute path prefixes were removed, and every numeric value is
unchanged (checked leaf by leaf when copying). Sources under `amir-porjects/ff-wiki/_build/analyses/`
are the author's internal archive of rebuttal-era analysis JSONs, which is not distributed (see
`analysis/README.md`).

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
| `e1_tier1_sep_acc.json`, `e3_fd_ema.json`, `e6_l8_k0_s42.json` | camera-ready experiments E1 (tier 1), E3, E6 |
| `fd_figure_check.json`, `fig3_4_bpff_values.json`, `fig_accuracy_vs_ratio_values.json`, `fig_dissociation_summary_values.json` | values plotted in the regenerated figures |
| `p3_depth_order_leak_check.json` | the depth-order leak check on the v1 trainers (superseded by `tests/locality_depth_order.json`) |
| `compute_budget.json` | compute budget of the reported runs (device, per-run GPU-hours, totals) from the raw run logs; written by `analysis/camera_ready/compute_budget.py` |

Not included: E1 tier 2 and E2 (validation-selected early exit), which were still running when this
archive was built.
