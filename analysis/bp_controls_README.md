# BP control runs (Table 2 of the main paper)

The BP rows in `Table 2: BP vs FF Controlled Ablation` use the same
backbone as the FF rows but trained with a stock BP loop (PyTorch
`nn.CrossEntropyLoss` + AdamW). In the camera-ready version the four
calibration rows are reported at n=3 (seeds 42/123/456; test top-1 without
TTA, mean ± sample SD; values in
`metric_summaries/camera_ready/bp_calibration.json`):

| Row | Backbone | Test top-1 (%), n=3 |
| --- | --- | ---: |
| Plain CNN + BP | Plain CNN (1.15 M, ch. 64/128/256) | 90.08 ± 0.28 |
| Plain CNN + FF | same plain CNN, FF pipeline | 29.85 ± 0.50 |
| FF backbone + BP, weak aug | FF backbone stripped of goodness heads (5,539,530 parameters = 5.54 M) | 87.43 ± 0.62 |
| FF backbone + BP, strong aug | same as above | 93.85 ± 0.18 |

These numbers were produced by the ORIGINAL run scripts, shipped in
[`trainers/bp_original/`](../trainers/bp_original/) (seed 42 = the submitted
runs; seeds 123/456 = the rebuttal runs; the only difference between the two
copies was the seed plumbing). Pass `--seed` explicitly: the scripts'
argparse default (42) overrides `FF_SEED`. `scripts/reproduce_table2_bp_controls.sh`
runs all twelve.

Two parameter-matched BP controls from the v1 supplement are not reported in the paper; they are
single-seed:

| Row | Backbone | Augmentation | Reported acc |
| --- | --- | --- | ---: |
| Plain CNN big (7.16 M) BP strong | Plain CNN big (ch. 160/320/640) | matched FF augmentation | 92.10% |
| MoE Transformer small (~2 M) BP strong | MoE-T (L=2 D=128 4 experts top-2) | matched FF augmentation | 91.33% |

## Compact re-implementation (`trainers/bp_control.py`)

A compact BP control trainer is also bundled at
[`trainers/bp_control.py`](../trainers/bp_control.py).
It is a thin reference re-implementation (not the script that produced the reported numbers): the FF-side
co-design (goodness aspects, MoE routing, depth-scaled loss) belongs
in `cp_fair_cifar10.py`; this file just trains the same backbone
families with stock CrossEntropy + AdamW. Its docstring still calls the `plain_big` and
`moe_small` runs "The two parameter-matched controls of the same table"; they are not reported in
the paper. The file is left unchanged because `tests/equivalence_results.json` records its SHA-256.

```bash
# Reproduce each Table 2 BP row (one command per row):

# Plain CNN BP strong  →  ~89.86% test top-1
FF_BACKBONE=plain_cnn  FF_AUG=strong FF_SEED=42 python trainers/bp_control.py

# FF backbone BP weak  →  ~86.90%
FF_BACKBONE=ff_stripped FF_AUG=weak   FF_SEED=42 python trainers/bp_control.py

# FF backbone BP strong (matched)  →  ~93.90%
FF_BACKBONE=ff_stripped FF_AUG=strong FF_SEED=42 python trainers/bp_control.py

# Parameter-matched controls (not reported in the paper):
FF_BACKBONE=plain_big   FF_AUG=strong FF_SEED=42 python trainers/bp_control.py   # ~92.10%
FF_BACKBONE=moe_small   FF_AUG=strong FF_SEED=42 python trainers/bp_control.py   # ~91.33%
```

Recipe: AdamW (lr 3e-4, weight_decay 0.05) + cosine schedule, batch 512,
label smoothing 0.1. Each run writes `preds_test.npz` (same `(y_true,
y_pred, scores)` convention as the FF preds files) and a `log.json`
to `out/bp_<backbone>_<aug>_seed<seed>/`. Override the output dir with
`FF_OUT_DIR`.

## Verification

`metric_summaries/aggregated_remaining_experiments.json` contains the
final test accuracies for the four parameter-matched control runs (not
reported in the paper). Each value was read from the run's
`final_metrics.json` or saved prediction file (the `src` field of each
entry); running the commands above reproduces them within seed std.
