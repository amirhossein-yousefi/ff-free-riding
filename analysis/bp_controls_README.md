# BP control runs (Table 2 of the main paper)

The BP rows in `Table 2: BP vs FF Controlled Ablation` use the same
backbone as the FF rows but trained with a stock BP loop (PyTorch
`nn.CrossEntropyLoss` + AdamW). In the camera-ready version the four
calibration rows are reported at n=3 (seeds 42/123/456; test top-1 without
TTA, mean ± sample SD; values in
`metric_summaries/camera_ready/bp_calibration.json`):

| Row | Backbone | Augmentation | Test top-1 (%), n=3 |
| --- | --- | --- | ---: |
| Plain CNN + BP | Plain CNN (1.15 M, ch. 64/128/256) | weak: random crop (padding 4) + flip | 90.08 ± 0.28 |
| Plain CNN + FF | same plain CNN, trained with FF | weak, as the BP row | 29.85 ± 0.50 |
| FF backbone + BP, weak aug | FF backbone stripped of goodness heads (5,539,530 parameters = 5.54 M) | weak | 87.43 ± 0.62 |
| FF backbone + BP, strong aug | same as above | FF's per-view recipe (one view; the FF runs draw three) | 93.85 ± 0.18 |

"FF's per-view recipe" is the FF trainers' `create_aug(10)`: RandomResizedCrop(32, scale=(0.3, 1.0)),
flip, ColorJitter(0.4, 0.4, 0.2, 0.1) with p=0.8, RandAugment(num_ops=2, magnitude=10) and
GaussianBlur(3, sigma in [0.1, 2.0]) with p=0.5 (paper Tab. 19). The FF runs draw three such views
per image, at RandAugment magnitudes 10/8/10.

These numbers were produced by the ORIGINAL run scripts, shipped in
[`trainers/bp_original/`](../trainers/bp_original/) (seed 42 = the submitted
runs; seeds 123/456 = the rebuttal runs; the only difference between the two
copies was the seed plumbing). Pass `--seed` explicitly: the scripts'
argparse default (42) overrides `FF_SEED`. `scripts/reproduce_table2_bp_controls.sh`
runs all twelve.

The nine BP networks of these rows (plain CNN + BP and both stripped-backbone BP rows, seeds
42/123/456) are public in the checkpoint release (group `bp_calibration`,
[`RELEASE_CHECKPOINTS.md`](../RELEASE_CHECKPOINTS.md)); each file is its run's validation-selected
`checkpoints/best.pt`. The plain-CNN FF row is not part of the release.

Two parameter-matched BP controls from the v1 supplement are not reported in the paper; they are
single-seed:

| Row | Backbone | Augmentation | Reported acc |
| --- | --- | --- | ---: |
| Plain CNN big (7.16 M) BP | Plain CNN big (ch. 160/320/640) | weak: random crop + flip (`plain_cnn_bp_cifar10.py`) | 92.10% |
| MoE Transformer small (~2 M) BP strong | MoE-T (L=2 D=128 4 experts top-2) | FF's per-view recipe (one view) | 91.33% |

## Compact re-implementation (`trainers/bp_control.py`)

A compact BP control trainer is also bundled at
[`trainers/bp_control.py`](../trainers/bp_control.py).
It is a thin reference re-implementation (not the script that produced the reported numbers): the FF-side
co-design (goodness aspects, MoE routing, depth-scaled loss) belongs
in `cp_fair_cifar10.py`; this file just trains the same backbone
families with stock CrossEntropy + AdamW. Its docstring still calls the `plain_big` and
`moe_small` runs "The two parameter-matched controls of the same table"; they are not reported in
the paper. Its `aug_strong` docstring ("Matched to the FF training pipeline ... Same recipe as the
strong-aug FF runs") is also wrong: `aug_strong` uses RandomResizedCrop scale (0.8, 1.0), RandAugment
magnitude 9 and always-on ColorJitter(0.2, 0.2, 0.2) and GaussianBlur sigma in [0.1, 1.5], which is
not the FF recipe; the plain-CNN rows of Table 2 used weak augmentation, not `aug_strong`. The file is
left unchanged because `tests/equivalence_results.json` records its SHA-256; the commands below are
therefore approximations of the reported rows, not reproductions.

```bash
# Approximate each Table 2 BP row (one command per row; see above):

# Plain CNN BP (weak aug)  →  reported seed 42: 89.86% test top-1
FF_BACKBONE=plain_cnn  FF_AUG=weak   FF_SEED=42 python trainers/bp_control.py

# FF backbone BP weak  →  reported seed 42: 86.90%
FF_BACKBONE=ff_stripped FF_AUG=weak   FF_SEED=42 python trainers/bp_control.py

# FF backbone BP strong (aug_strong, not FF's recipe; see above)  →  reported seed 42: 93.90%
FF_BACKBONE=ff_stripped FF_AUG=strong FF_SEED=42 python trainers/bp_control.py

# Parameter-matched controls (not reported in the paper):
FF_BACKBONE=plain_big   FF_AUG=weak   FF_SEED=42 python trainers/bp_control.py   # ~92.10%
FF_BACKBONE=moe_small   FF_AUG=strong FF_SEED=42 python trainers/bp_control.py   # ~91.33%
```

Recipe of `bp_control.py`: AdamW (lr 3e-4, weight_decay 0.05) + cosine schedule without
warmup, 200 epochs (`FF_EPOCHS`), batch 512, label smoothing 0.1, no gradient clipping. The
reported BP rows used a different recipe, which `trainers/bp_original/` implements and paper
App. D.3 states: AdamW, lr 1e-3, 362 epochs with 15 warmup epochs, no label smoothing; weight
decay 0.1, batch 256 and gradient clip 1.0 on the stripped backbone; weight decay 1e-4, batch 512
and no clipping for the plain CNN (the plain-CNN FF row, which `bp_control.py` cannot train, used
per-layer AdamW with lr 3e-4, weight decay 1e-4, batch 256 and no clipping on the same schedule).
Each `bp_control.py` run writes `preds_test.npz` (same `(y_true,
y_pred, scores)` convention as the FF preds files) and a `log.json`
to `out/bp_<backbone>_<aug>_seed<seed>/`. Override the output dir with
`FF_OUT_DIR`.

## Verification

`metric_summaries/aggregated_remaining_experiments.json` contains the
final test accuracies for the four parameter-matched control runs (not
reported in the paper). Each value was read from the run's
`final_metrics.json` or saved prediction file (the `src` field of each
entry). The commands above only approximate the two BP runs (B, D; see above), and
`bp_control.py`, which trains with BP only, cannot produce the two FF runs (A, C).
