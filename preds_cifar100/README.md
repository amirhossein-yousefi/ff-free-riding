# CIFAR-100 trio predictions (paired-bootstrap input)

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (this directory only; the
code is Apache-2.0). Copyright 2026 Amirhossein Yousefiramandi.

This directory holds the nine Stage-2 TTA test-prediction files of the CIFAR-100 dissociation trio
(γ=0 / hardness-gated κ=0 / cumulative γ=0.7, seeds 42/123/456). With them, the paired-bootstrap
statistics of the trio under the S2-TTA protocol can be recomputed without retraining: the S2-TTA
column of the trio Δ rows in Table 3 (`tab:dissociation_merged`), the S2-TTA column of the per-seed
table in App. B.7 (`tab:trio_perseed`), and the S2-TTA disagreement rates.

Paths below refer to the code repository (<https://github.com/amirhossein-yousefi/ff-free-riding>),
which is where these files are published. They are not duplicated in the minimal checkpoint
release (<https://huggingface.co/Amirhossein75/ff-free-riding>).
The files are byte-identical (SHA-256) to `preds_stage2_test_tta.npz` in the result-bearing run
directories listed in `analysis/camera_ready/config.py` (`TRIO_RUNS`). They are unchanged from the
v1 supplement.

The other three protocols (S1, S1-TTA, S2) of the trio, the MGC rows, and the CIFAR-10 pair need
the full prediction sets of those runs (about 127 MB for the trio alone). Those sets are not
bundled because the archive would exceed 100 MB. You can regenerate them from the checkpoints
listed in `RELEASE_CHECKPOINTS.md`:

- The seed-42 Stage-1 checkpoints of the trio and of MGC are public; they give S1 and S1-TTA.
- The others, including every Stage-2 head, are available on request.
- The exception is the CIFAR-10 γ=0 seed-42 run, whose checkpoint is not recoverable.

The camera-ready values are in `metric_summaries/camera_ready/c100_trio_paired.json`,
`mgc_paired_bootstrap.json` and `c10_paired.json`.

## Layout

```
preds_cifar100/
├── runs_gamma0_seed{42,123,456}/preds_stage2_test_tta.npz
├── runs_gated_k0_seed{42,123,456}/preds_stage2_test_tta.npz
└── runs_cumulative_seed{42,123,456}/preds_stage2_test_tta.npz
```

Each `.npz` file contains:

- `y_true` (int64, 10 000): CIFAR-100 test-set labels in deterministic order. The order is the
  same in all nine files, and the loader verifies this.
- `y_pred` (int64, 10 000): top-1 predictions under Stage-2 TTA (horizontal flip).
- `scores` (float32, 10 000 × 100): per-class scores before the argmax.

## Reproduce the paired bootstrap

```bash
python analysis/paired_dissociation_stats.py          # reads ./preds_cifar100 by default
```

`FF_C100_PREDS=<dir>` overrides the input directory, and `FF_PAIRED_OUT=<dir>` overrides the
output directory. By default the output goes next to the script as
`60_paired_dissociation.{json,md}`. The script's pooled differences and 95% CIs (5000 resamples,
`default_rng(0)`) equal the `S2-TTA` entries of
`metric_summaries/camera_ready/c100_trio_paired.json`: γ=0 − cumulative +0.28 pp [−0.02, +0.56],
κ=0 − cumulative +0.11 pp [−0.20, +0.42], γ=0 − κ=0 +0.17 pp [−0.13, +0.46]. The mean pairwise
disagreement is 13.0–13.7%.

Size: about 30 MB (9 files of about 3.4 MB each).
