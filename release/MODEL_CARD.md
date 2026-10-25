---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - forward-forward
  - local-learning
  - cifar-10
  - cifar-100
  - tiny-imagenet
  - layer-free-riding
  - neurips-2026
datasets:
  - uoft-cs/cifar10
  - uoft-cs/cifar100
metrics:
  - accuracy
---

# Forward-Forward free-riding checkpoints (NeurIPS 2026)

This repository holds 29 trained checkpoints from *Cumulative-Goodness Free-Riding in
Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant* (Amirhossein
Yousefiramandi, NeurIPS 2026). Each file is the validation-selected Stage-1 network
(`stage1_best.pt`) of one training run:

| Group | Dataset and setup | Runs | Files | Paper (main tables) |
|---|---|---|---:|---|
| `c100_trio` | CIFAR-100, L4/D256 dense, no SAM, 362 Stage-1 epochs: block-local γ=0, hardness-gated κ=0, cumulative γ=0.7 | 3 arms × seeds 42/123/456 | 9 | Tables 3, 5 (cumulative reference row), 12, 15, 24, 29 and 39; Tables 9–11 (readouts and probes); Tables 13 and 36 (re-scored from these files) |
| `mgc_c100` | CIFAR-100, same setup: missing-gradient compensation (MGC, c_d=1) | seeds 42/123/456 | 3 | Tables 3 and 5; Table 9 (readouts) |
| `mgc_d128` | CIFAR-10, L4/D128, 16-expert MoE, SAM on, 180 Stage-1 epochs: MGC | seeds 42/123/456 | 3 | Tables 5 and 6; Table 8 (gradient share, seed 42) |
| `hinge` | CIFAR-10, same setup as `mgc_d128`: squared-hinge barrier, γ=0.7 and γ=0 arms | 2 arms × seeds 42/123/456 | 6 | Table 7 |
| `c10_cp_fair` | CIFAR-10, L4/D256, 32-expert MoE, SAM on, 362 Stage-1 epochs: CP-FAIR (γ=0.7 with a depth-scaled current-block term) | seeds 42/123/456/789 | 4 | Tables 1, 3, 21 and 27; Tables 2, 14, 30 and 41 (seed 42); Table 36 (re-scored from these files) |
| `c10_nomem_l2` | CIFAR-10, same network and 362-epoch Stage-1 schedule as `c10_cp_fair`: NoMem-L2, which keeps a uniform current-block weight (0.5) where CP-FAIR uses its depth-scaled term, with a depth-order weight of 0.1 and margin 0.05 where CP-FAIR uses 0.2 and 0.1 | seed 42 | 1 | Tables 27 and 41; App. N (Stage-2 summary) |
| `tin_gamma0` | Tiny ImageNet, L4/D256 dense, no SAM, 362 Stage-1 epochs: block-local γ=0 | seeds 42/123/456 | 3 | Tables 15, 25 and 29; Table 40 (seed 42) |

Table and appendix numbers refer to the camera-ready paper. The first release (2026-10-01) held
the four seed-42 CIFAR-100 files of Table 3 (γ=0, κ=0, cumulative and MGC). The other 25 were
added on 2026-10-24.

`release_manifest.json` lists 77 further checkpoint files with their SHA-256 hashes, which are
available from the author on request (see [Other checkpoints](#other-checkpoints-on-request)):
the Stage-2 heads of all 53 runs in the manifest, and the Stage-1 networks of its
24 other runs (CIFAR-10 γ=0, gated ablation and hardness-gated κ=0).

The checkpoints are research artifacts. They let you re-evaluate and probe the trained networks
without retraining Stage 1. They are not general-purpose image classifiers.

- **Paper:** [arXiv:2605.06240](https://arxiv.org/abs/2605.06240) ·
  [OpenReview](https://openreview.net/forum?id=TN0FvatX6A)
- **Project page:** <https://amirhossein-yousefi.github.io/ff-free-riding/>
- **Code** (trainers, analysis, reproduction manifest; Apache-2.0):
  <https://github.com/amirhossein-yousefi/ff-free-riding>
- **Checkpoints** (this repository; CC BY 4.0): <https://huggingface.co/Amirhossein75/ff-free-riding>

## Files

| File | Run | Selected epoch (val. top-1, flip TTA) | Bytes | SHA-256 |
|---|---|---|---:|---|
| `checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed42_L4_D256_P2_seed42_stage1_best.pt` | CIFAR-100 γ=0, seed 42 | 362 (65.72 %) | 211467565 | `d2e2a0548597355d633c7c3acad1ef01c274323f279ca1055e77a1c807e2848c` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed123_L4_D256_P2_seed123_stage1_best.pt` | CIFAR-100 γ=0, seed 123 | 360 (67.44 %) | 211486873 | `decc3110d2195112b5f851dafc73043588a58255ba22c75ee228897d1496a450` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed456_L4_D256_P2_seed456_stage1_best.pt` | CIFAR-100 γ=0, seed 456 | 340 (67.02 %) | 211486873 | `09b878e2669545e045ef76de77025da11caac3369160ad8dc80faaf8cff74f54` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gated_k0_seed42_L4_D256_P2_seed42_stage1_best.pt` | CIFAR-100 κ=0, seed 42 | 340 (66.22 %) | 211484697 | `58e0c770f80a4f70d9d6c814884ecbb1caf1162a270caf3294282b67d69fd4ca` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gated_k0_seed123_L4_D256_P2_seed123_stage1_best.pt` | CIFAR-100 κ=0, seed 123 | 320 (67.44 %) | 211487237 | `4ad924afe1d33fd1b31edf54587e82ca2df1217ddef224156ae5c51db26a1dcf` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gated_k0_seed456_L4_D256_P2_seed456_stage1_best.pt` | CIFAR-100 κ=0, seed 456 | 360 (67.64 %) | 211489477 | `dfc6fac39b92f648b83720b40200dc4202163aea52c466f1ba0e59f61deb036d` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_cumulative_seed42_L4_D256_P2_seed42_stage1_best.pt` | CIFAR-100 cumulative γ=0.7, seed 42 | 340 (66.22 %) | 211487237 | `537cf9e2e27e73009596a9de9f52097ee98d33ad1055783a8f434e20ab44cc3b` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_cumulative_seed123_L4_D256_P2_seed123_stage1_best.pt` | CIFAR-100 cumulative γ=0.7, seed 123 | 340 (67.42 %) | 211539185 | `21cbdd0eed82bf2846ace41fb03cca1c41f446c15d10cfd3d3ad0b48e0ad6d3f` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_cumulative_seed456_L4_D256_P2_seed456_stage1_best.pt` | CIFAR-100 cumulative γ=0.7, seed 456 | 320 (68.30 %) | 211539121 | `354783dfa1827d5e8d3b130d88e765c141db5cdc473b445e42aeb5cbd28593d6` |
| `checkpoints/mgc_c100/ff_cifar100_mgc_c100_L4_D256_P2_seed42_stage1_best.pt` | CIFAR-100 MGC, seed 42 | 362 (66.96 %) | 211449657 | `6776a87e9afa3dcf6abc993b0e17646724b3b9f185ada825eebd54977c2c1a66` |
| `checkpoints/mgc_c100/ff_cifar100_mgc_c100_L4_D256_P2_seed123_stage1_best.pt` | CIFAR-100 MGC, seed 123 | 340 (66.02 %) | 211450767 | `d5504353839c8ff15041a67963cbe66ab097893b4e90bcc70474fb55d1473dd8` |
| `checkpoints/mgc_c100/ff_cifar100_mgc_c100_L4_D256_P2_seed456_stage1_best.pt` | CIFAR-100 MGC, seed 456 | 340 (66.20 %) | 211453007 | `09a475deb24e8771d38561f3e8233cbe13bac33e37cc92eb015dd69143f756ea` |
| `checkpoints/mgc_d128/ff_cifar10_mgc_c1_L4_D128_P2_seed42_stage1_best.pt` | CIFAR-10 MGC-D128, seed 42 | 180 (86.22 %) | 241632967 | `41ee1f5735885b3e942bb3c65301c064e8975caf0ceb89320deaad83fea67a10` |
| `checkpoints/mgc_d128/ff_cifar10_mgc_c1_L4_D128_P2_seed123_stage1_best.pt` | CIFAR-10 MGC-D128, seed 123 | 180 (86.74 %) | 241635341 | `a4a69e7b46bc21878f9c55866fca2f6b6c1180ae7494475624595c80689f2820` |
| `checkpoints/mgc_d128/ff_cifar10_mgc_c1_L4_D128_P2_seed456_stage1_best.pt` | CIFAR-10 MGC-D128, seed 456 | 180 (85.48 %) | 241635341 | `411bebeacc7da9cccec8c647053270ce4642dee8d380d996cc3a0b3a9d29cb2f` |
| `checkpoints/hinge/ff_cifar10_hinge_g07_L4_D128_P2_seed42_stage1_best.pt` | CIFAR-10 hinge γ=0.7, seed 42 | 180 (86.22 %) | 241640089 | `ed7f1482a7294e19489fc788c8b531f61ac2d893e7825639d713ff5001e19033` |
| `checkpoints/hinge/ff_cifar10_hinge_g07_L4_D128_P2_seed123_stage1_best.pt` | CIFAR-10 hinge γ=0.7, seed 123 | 160 (86.30 %) | 241642399 | `623842bba962ce27e7dc2533ecd26a097d69375e63bd3b39d0a8f012250e2809` |
| `checkpoints/hinge/ff_cifar10_hinge_g07_L4_D128_P2_seed456_stage1_best.pt` | CIFAR-10 hinge γ=0.7, seed 456 | 180 (86.78 %) | 241647199 | `b8c4a39a77bd8f9e8a5f16334f8296706faf7d896f0b736772346dca9a996b8b` |
| `checkpoints/hinge/ff_cifar10_hinge_g0_L4_D128_P2_seed42_stage1_best.pt` | CIFAR-10 hinge γ=0, seed 42 | 180 (86.38 %) | 241637779 | `8d128a88350c16793d7a7fb798180393af50f0afdaa207e169aa01f66a8cff16` |
| `checkpoints/hinge/ff_cifar10_hinge_g0_L4_D128_P2_seed123_stage1_best.pt` | CIFAR-10 hinge γ=0, seed 123 | 180 (86.62 %) | 241640089 | `ddd58ae95630f77c96dff821f748369733b9dfb71de8740ad144a41d3e76f048` |
| `checkpoints/hinge/ff_cifar10_hinge_g0_L4_D128_P2_seed456_stage1_best.pt` | CIFAR-10 hinge γ=0, seed 456 | 160 (86.12 %) | 241640089 | `7fd228475664c635e763e11d20682c12ab3a521022f9bd7bbe9cbec268c23ccb` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_P2_seed42_stage1_best.pt` | CIFAR-10 CP-FAIR, seed 42 | 340 (91.90 %) | 1776204713 | `d8c7f2c358850170ff75d1828aa1a340240bca1b8579c02833ed042dd29dbdf9` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_P2_seed123_stage1_best.pt` | CIFAR-10 CP-FAIR, seed 123 | 240 (91.76 %) | 1776237023 | `a88fe83598fd22031e5e9fe434fc87e8bbc718c02c4912782ba2ab94a75f7a70` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_P2_seed456_stage1_best.pt` | CIFAR-10 CP-FAIR, seed 456 | 220 (92.00 %) | 1776237023 | `025018a5b4dceb7f8b8379f9a91edba0bf2a52381fa4a6af8ccf10d48f682429` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_seed789_L4_D256_P2_seed789_stage1_best.pt` | CIFAR-10 CP-FAIR, seed 789 | 260 (91.44 %) | 1776229407 | `14aad1dbfdbbb623fb3a0fa0475f4a5a640293a12a11c65444632611fcb9136a` |
| `checkpoints/c10_nomem_l2/ff_cifar10_cifar10_opt_do_this_first_ynorm_nomem_resid_L4_D256_P2_seed42_stage1_best.pt` | CIFAR-10 NoMem-L2, seed 42 | 320 (91.84 %) | 1776232933 | `957cf6e18c384ef3f41282f116771fa2f5f0e8b5fb9a9cc262e051f707203db8` |
| `checkpoints/tin_gamma0/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed42_L4_D256_P4_seed42_stage1_best.pt` | Tiny ImageNet γ=0, seed 42 | 362 (49.29 %) | 232472607 | `2ac2d23d6facec8a6464b8f9fe5c3c0b8e97e6af5d06939fff6cd85a46146c1b` |
| `checkpoints/tin_gamma0/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed123_L4_D256_P4_seed123_stage1_best.pt` | Tiny ImageNet γ=0, seed 123 | 340 (48.80 %) | 232474827 | `4381293b00bb06e4d2ec6e738ca17f9d6fd86c930bb237bb80dda81b67b10a26` |
| `checkpoints/tin_gamma0/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed456_L4_D256_P4_seed456_stage1_best.pt` | Tiny ImageNet γ=0, seed 456 | 360 (49.00 %) | 232474827 | `bd1b08ba4ef6e43d1be21d75711ebd56b95350dc839875ee6d2efd59c40a1b49` |

The 29 files total 14,291,136,349 bytes (14.3 GB). The selected epoch is the file's
`epoch_stage1`. The validation top-1 is its `best_s1`: the S1 readout with flip TTA on the run's
held-out validation split, which selected the checkpoint. The file names keep the run tags of the
training scripts: the Tiny ImageNet files are tagged `tiny_imagenet_cp_fair_seed<seed>`, but their
stored `cfg` has `gamma_scale=0` (block-local γ=0). The repository also contains:

```
README.md                this model card
SHA256SUMS               sha256sum-compatible list of the 29 checkpoint files
release_manifest.json    the 106 checkpoint files of the release (53 runs): run, paper tables, trainer,
                         source, lineage, size, SHA-256 and the checks below; `availability` is
                         "public_hf" for the 29 files here and "on_request" for the others
```

Each file stores the following keys:

- `net_ema`: the EMA weights, which give the S1 numbers;
- `model`: the raw weights;
- `cfg`: the run's full configuration, identical to the run's `config.json` except `device`;
- `epoch_stage1`: the selected epoch;
- `best_s1`: the validation top-1 at that epoch (S1 readout with flip TTA);
- `stage`: 1;
- `opt_states`, `sched_states` and `queues`: the optimizer states, the scheduler states and the
  contrastive feature queues.

## Numbers behind each file

| Run | S1 | S1-TTA | S2 (logged) | S2-TTA (logged) |
|---|---:|---:|---:|---:|
| CIFAR-100 γ=0, seed 42 | 66.17 | 66.92 | 68.27 | 68.94 |
| CIFAR-100 γ=0, seed 123 | 66.36 | 66.80 | 68.60 | 69.16 |
| CIFAR-100 γ=0, seed 456 | 66.57 | 67.07 | 68.76 | 69.60 |
| CIFAR-100 κ=0, seed 42 | 66.07 | 66.85 | 68.29 | 68.64 |
| CIFAR-100 κ=0, seed 123 | 66.17 | 67.09 | 68.02 | 68.79 |
| CIFAR-100 κ=0, seed 456 | 66.82 | 68.10 | 69.00 | 69.77 |
| CIFAR-100 cumulative γ=0.7, seed 42 | 66.70 | 67.18 | 68.09 | 68.57 |
| CIFAR-100 cumulative γ=0.7, seed 123 | 66.55 | 67.24 | 68.36 | 68.88 |
| CIFAR-100 cumulative γ=0.7, seed 456 | 67.10 | 67.82 | 69.07 | 69.42 |
| CIFAR-100 MGC, seed 42 | 66.05 | 66.79 | 68.71 | 69.19 |
| CIFAR-100 MGC, seed 123 | 65.36 | 66.30 | 67.70 | 68.78 |
| CIFAR-100 MGC, seed 456 | 65.79 | 66.66 | 68.30 | 69.29 |
| CIFAR-10 MGC-D128, seed 42 | 85.23 | 86.64 | 86.00 | 87.05 |
| CIFAR-10 MGC-D128, seed 123 | 85.88 | 87.09 | 86.48 | 87.60 |
| CIFAR-10 MGC-D128, seed 456 | 85.25 | 86.89 | 85.73 | 87.17 |
| CIFAR-10 hinge γ=0.7, seed 42 | 85.22 | 86.59 | 85.65 | 87.04 |
| CIFAR-10 hinge γ=0.7, seed 123 | 85.77 | 87.10 | 86.00 | 87.20 |
| CIFAR-10 hinge γ=0.7, seed 456 | 85.63 | 86.86 | 86.18 | 87.01 |
| CIFAR-10 hinge γ=0, seed 42 | 85.59 | 86.80 | 85.56 | 86.95 |
| CIFAR-10 hinge γ=0, seed 123 | 85.85 | 86.96 | 86.33 | 87.25 |
| CIFAR-10 hinge γ=0, seed 456 | 85.27 | 86.30 | 85.61 | 86.59 |
| CIFAR-10 CP-FAIR, seed 42 | 90.45 | 91.36 | 90.42 | 91.27 |
| CIFAR-10 CP-FAIR, seed 123 | 90.45 | 91.34 | 90.39 | 91.21 |
| CIFAR-10 CP-FAIR, seed 456 | 90.43 | 91.39 | 90.59 | 91.44 |
| CIFAR-10 CP-FAIR, seed 789 | 90.09 | 91.22 | 90.19 | 91.19 |
| CIFAR-10 NoMem-L2, seed 42 | 90.56 | 91.31 | 90.32 | 91.19 |
| Tiny ImageNet γ=0, seed 42 | 48.75 | 49.44 | 51.71 | 52.60 |
| Tiny ImageNet γ=0, seed 123 | 48.45 | 49.16 | 51.61 | 52.42 |
| Tiny ImageNet γ=0, seed 456 | 48.41 | 48.67 | 51.23 | 51.95 |

The values are the test top-1 (%) that each run logged in its final evaluation. For Tiny ImageNet,
the test set is the labeled validation split, because the test labels are not public. The paper
prints per-seed values for most of these runs:

- Table 5 (MGC) and Table 21 (CP-FAIR): all four readouts;
- Table 25 (Tiny ImageNet): S1 in its caption, S1-TTA and S2-TTA in the table; S2 in the caption
  of Fig. 18;
- Table 27 (NoMem-L2): S1 and S1-TTA;
- Table 36 (CIFAR-100 trio): S1 and S2-TTA; the seed-42 S1 and S2 also in Table 9.

The other values of the trio and all values of the squared hinge appear only as 3-seed means
(Tables 3 and 7). The per-seed values of every CIFAR-100 and CIFAR-10 L4/D128 run are in
`metric_summaries/camera_ready/` of the code repository.

- **S1 and S1-TTA** are the Stage-1 readouts of these files. S1 is the goodness-sum readout of the
  EMA weights on a single crop; S1-TTA adds the horizontally flipped image.
- **S2 and S2-TTA** come from the Stage-2 attentive head. This head is a backprop-trained probe on
  the frozen backbone, not part of the FF method, and its checkpoints (`stage2_best.pt`) are not in
  this repository. To get S2 numbers, retrain the probe as described in
  [Retraining the Stage-2 probe](#retraining-the-stage-2-probe), or ask for the original heads,
  which are available on request. For NoMem-L2 the paper prints no Stage-2 value of its own: its
  S2-TTA enters the Stage-2 summary of App. N, which is taken over the 12 seed-42 variants of
  Table 27 that were run through Stage 2.
- **Deepest-block separation** (the "sep" columns of Tables 3, 5 and 7) is a training-log quantity:
  it was read at the last Stage-1 epoch, not recomputed from the validation-selected checkpoint.
  The code repository ships the per-seed values in `metric_summaries/camera_ready/`. Table 36 gives
  the deepest-block separation recomputed on the validation-selected checkpoints themselves (EMA
  weights, eval mode), including the CIFAR-100 trio and CP-FAIR files here.

### Checks before upload

Every file was checked on CPU before upload (2026-10-24). `release_manifest.json` records the
results for each file (`lineage_check`, `loading_spot_check`):

- Its SHA-256, size and CRC-32 equal the manifest. The CIFAR-100 trio files, extracted from Google
  Drive export archives, also have the same CRC-32 and size as their archive members, and the Tiny
  ImageNet files are the archive members themselves, whose CRC-32 zipfile verifies on read.
- The result-bearing run's `events.jsonl` names the file in its final test evaluation, and the
  logged S1 and S1-TTA equal the values behind the paper's tables.
- The file's `best_s1` equals the value in the run's `run_end` event, and its stored `cfg` equals
  the run's `config.json`.
- The file loads with `torch.load(..., weights_only=True)`. Its `net_ema` loads into the group's
  trainer with every key matched, and every key of its stored `cfg` is a field of that trainer's
  `FFConfig`.
- The loading code below, run on CPU in float32, gave the same single-crop prediction as the run's own saved test predictions (`preds_stage1_test_no_tta.npz`) for 1,854 of 1,856 file–image pairs: each of the 29 files scored the first 64 images of its test set. The dense files agree on 960 of 960, with every per-class score within 0.03 of the saved one. The MoE files (`mgc_d128`, `hinge`, `c10_cp_fair`, `c10_nomem_l2`) agree on 894 of 896, with every per-class score within 1.15 of the saved one. The original evaluations ran on GPUs with TF32; see Precision below.

S1 was also re-predicted on the full test set for 22 of the files, each within 0.02 pp of its run
log. Twelve of them have identity checks of their own (the seed-42 CIFAR-100 trio, all MGC CIFAR-100
files, and seeds 123/456 of MGC-D128 and of both squared-hinge arms; each within 0.01 pp), and the
re-scoring behind Tables 13 and 36 covers all nine CIFAR-100 trio files and the four CP-FAIR files.
`RELEASE_CHECKPOINTS.md` in the code repository lists them.

## Loading

Each group has its own trainer in the code repository. The file's stored `cfg` configures it.

| Group | Trainer | Test set | Normalization (mean / std) |
|---|---|---|---|
| `c100_trio`, `mgc_c100` | `trainers/cp_fair_cifar100.py` | CIFAR-100 test set | (0.5071, 0.4867, 0.4408) / (0.2675, 0.2565, 0.2761) |
| `mgc_d128`, `hinge` | `trainers/cp_fair_hardness_gated.py` | CIFAR-10 test set | (0.4914, 0.4822, 0.4465) / (0.2023, 0.1994, 0.2010) |
| `c10_cp_fair`, `c10_nomem_l2` | `trainers/cp_fair_cifar10.py` | CIFAR-10 test set | (0.4914, 0.4822, 0.4465) / (0.2023, 0.1994, 0.2010) |
| `tin_gamma0` | `trainers/cp_fair_tiny_imagenet.py` | Tiny ImageNet validation split (64×64) | (0.4802, 0.4481, 0.3975) / (0.2770, 0.2691, 0.2821) |

MGC and the squared-hinge barrier change only the training loss, so their files load like the
others. NoMem-L2 has the same network as CP-FAIR and differs only in its training settings.

```bash
git clone https://github.com/amirhossein-yousefi/ff-free-riding && cd ff-free-riding
pip install -r requirements.txt huggingface_hub
```

```python
import importlib.util, os, sys, torch
from huggingface_hub import hf_hub_download
from torchvision import datasets, transforms

FILE = "checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed42_L4_D256_P2_seed42_stage1_best.pt"
TRAINER = "trainers/cp_fair_cifar100.py"         # the group's trainer (table above)

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision("high")  # the runs' GPU setting (TF32); see Precision below

spec = importlib.util.spec_from_file_location("ff_trainer", TRAINER)
T = importlib.util.module_from_spec(spec); sys.modules["ff_trainer"] = T; spec.loader.exec_module(T)

path = hf_hub_download("Amirhossein75/ff-free-riding", FILE)
ckpt = torch.load(path, map_location="cpu", weights_only=True)
cfg = T.FFConfig()
for k, v in ckpt["cfg"].items():                 # the run's configuration is stored in the checkpoint
    if hasattr(cfg, k):
        setattr(cfg, k, v)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
net = T.FFHybridNet(cfg).to(cfg.device)
net.load_state_dict(ckpt["net_ema"])             # EMA weights: the S1 / S1-TTA numbers
net.eval()

norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # the group's statistics
tf = transforms.Compose([transforms.ToTensor(), norm])
test = datasets.CIFAR100("./data_c100", train=False, download=True, transform=tf)
# CIFAR-10:      test = datasets.CIFAR10("./data_c10", train=False, download=True, transform=tf)
# Tiny ImageNet: test = datasets.ImageFolder(
#                    os.path.join(T.download_and_prepare_tiny_imagenet(), "val_restructured"), transform=tf)
loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)
acc = T.evaluate_ff(net, loader, cfg.device, tta=False, topk=(1,))   # tta=True gives S1-TTA
print(f"S1 top-1: {100 * acc[1]:.2f} %")         # logged: 66.17 for this file
```

- **`weights_only`.** The files are PyTorch pickles. With PyTorch 2.14.0, all 29 load with
  `weights_only=True`, because they contain only tensors and plain Python containers. If your
  PyTorch version refuses a file, `torch.load(..., weights_only=False)` will load it, but that
  setting executes arbitrary pickle code. Use it only after the file's SHA-256 matches: compare
  `sha256sum <file>` with the Files table, or run the check in [Integrity](#integrity). The
  trainers' own `torch.load` calls, used by the Stage-2 retraining below, do not pass the argument,
  so they fall back to PyTorch's default.
- **Precision.** The runs were evaluated on GPUs with TF32: the trainers call
  `torch.set_float32_matmul_precision('high')` on GPUs of compute capability 8 or higher, and cuDNN
  convolutions use TF32 by default. Top-k expert routing is discontinuous, so in the MoE files
  (`mgc_d128`, `hinge`, `c10_cp_fair`, `c10_nomem_l2`) a small numerical difference can change which
  experts a token uses. In the spot check above, the float32 CPU evaluation changed 2 of the 896 MoE predictions (`hinge`, `mgc_d128`) and none of the dense ones. To match the logged values closely, evaluate on such a GPU
  after calling `torch.set_float32_matmul_precision('high')`.
- **Side effects.** Importing a trainer creates an empty `./checkpoints/` directory.
  `download=True` fetches CIFAR into `./data_c100/` or `./data_c10/`. The Tiny ImageNet helper
  `T.download_and_prepare_tiny_imagenet()` downloads the dataset from the CS231n site into
  `./data_tiny_imagenet/` and copies the validation images into one folder per class
  (`val_restructured/`).
- **Speed.** The goodness-sum readout scores every label hypothesis for every image: 10 for
  CIFAR-10, 100 for CIFAR-100 and 200 for Tiny ImageNet. On CPU, a CIFAR-100 file took about 1 s
  per test image with 4 threads, so use a GPU for full test sets.
- **Size.** The CIFAR-10 CP-FAIR and NoMem-L2 files hold a 32-expert MoE network with 107 M
  parameters (19 M active per token), so each takes 1.8 GB.

## Retraining the Stage-2 probe

When a Stage-1 checkpoint is complete, the trainer skips Stage 1 and does the following:

1. It trains only the Stage-2 head on the frozen EMA backbone, with the run's train/validation
   split, which the trainer regenerates from the seed.
2. It runs the final test evaluation: S1, S1-TTA, S2 and S2-TTA, written to `events.jsonl` in a
   new run directory.

This needs a GPU. The commands below are for the four seed-42 CIFAR-100 files (20 Stage-2 epochs,
AdamW). They use `hf`, the command-line tool that `pip install huggingface_hub` installs; current
`huggingface_hub` releases no longer run `huggingface-cli`.

```bash
# one arm per invocation; EP must equal the checkpoint's epoch_stage1
ARM=gamma0;     EP=362; ENV="FF_GAMMA_SCALE=0"
# ARM=gated_k0; EP=340; ENV="FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0"
# ARM=cumulative; EP=340; ENV="FF_GAMMA_SCALE=0.7"
F=ff_cifar100_cifar100_${ARM}_seed42_L4_D256_P2_seed42_stage1_best.pt
hf download Amirhossein75/ff-free-riding "checkpoints/c100_trio/$F" --local-dir ./hf && \
mkdir -p ckpt_s2 && cp "hf/checkpoints/c100_trio/$F" ckpt_s2/ && cp "ckpt_s2/$F" "ckpt_s2/${F/_best/_last}" && \
env $ENV FF_CKPT_DIR=./ckpt_s2 FF_VERSION_TAG=cifar100_${ARM}_seed42 FF_EPOCHS_STAGE1=$EP \
  FF_NUM_BLOCKS=4 FF_D_MODEL=256 FF_NUM_HEADS=8 FF_USE_SAM=0 FF_N_EXPERTS=1 FF_MOE_TOP_K=1 \
  FF_SEED=42 python trainers/cp_fair_cifar100.py

# MGC (EP=362)
F=ff_cifar100_mgc_c100_L4_D256_P2_seed42_stage1_best.pt
hf download Amirhossein75/ff-free-riding "checkpoints/mgc_c100/$F" --local-dir ./hf && \
mkdir -p ckpt_s2 && cp "hf/checkpoints/mgc_c100/$F" ckpt_s2/ && cp "ckpt_s2/$F" "ckpt_s2/${F/_best/_last}" && \
FF_CKPT_DIR=./ckpt_s2 FF_VERSION_TAG=mgc_c100 FF_USE_MGC=1 FF_MGC_C0=1.0 FF_N_EXPERTS=1 FF_MOE_TOP_K=1 \
  FF_USE_SAM=0 FF_NUM_HEADS=8 FF_GAMMA_GATING_MODE=constant FF_GAMMA_SCALE=0.7 FF_EPOCHS_STAGE1=362 FF_SEED=42 \
  python trainers/cp_fair_cifar100.py
```

The `&&` chain stops before the trainer starts if the download or a copy fails. Without a
checkpoint in `FF_CKPT_DIR`, the trainer would start Stage 1 from scratch.

The trainer loads two files from `FF_CKPT_DIR`:

- It resumes Stage 1 from `<run>_stage1_last.pt`.
- It starts Stage 2 and the final Stage-1 evaluation from `<run>_stage1_best.pt`.

Both copies are therefore needed.

If `FF_EPOCHS_STAGE1` is larger than the checkpoint's `epoch_stage1` (340 for κ=0 and cumulative),
the trainer resumes Stage-1 training instead of skipping it. The per-arm environment matches the
arm's row in the code repository's `repro_manifest.csv`, with `FF_EPOCHS_STAGE1` set to the selected
epoch.

For any other file, use its group's trainer and the environment of its row in `repro_manifest.csv`
(every group here except NoMem-L2 has one). Set `FF_SEED` to the run's seed, `FF_EPOCHS_STAGE1` to
the file's `epoch_stage1`, and `FF_VERSION_TAG` to the part of the file name between
`ff_<dataset>_` and `_L<blocks>`, so that the trainer finds the file under its run name
(`ff_<dataset>_<tag>_L<blocks>_D<width>_P<patch>_seed<seed>`). These other commands were not
checked one by one. The trainer builds its configuration from its own defaults and these
environment variables, not from the stored `cfg`, so two groups need a change in the trainer's
`FFConfig`:

- **Tiny ImageNet:** the runs' Stage-2 head had `head_hidden=1536`, but
  `trainers/cp_fair_tiny_imagenet.py` defaults to 2048 and has no environment variable for it. Set
  `head_hidden` to 1536 before retraining.
- **NoMem-L2:** it has no row. Its settings are in its stored `cfg`, for example 300 Stage-2
  epochs. The `cfg` predates the trainer's `block_curr_depth_slope` setting, which must be 0 for
  this run (a uniform current-block weight of 0.5). This setting affects training only, not the
  loading code above.

A retrained head is not bitwise identical to the paper's. The trainers that produced these runs did
not store RNG state, so the Stage-2 data order differs. Expect S2 values close to the logged ones,
not equal to them.

## Run history and disclosures

- **Resumed runs.** 13 of the 29 runs were resumed after an interrupted session:
  γ=0 seeds 123/456, κ=0 seed 456, cumulative seeds 123/456 and MGC seed 456 on CIFAR-100; squared-hinge γ=0.7 seed 456 and CP-FAIR seeds 42/123/456 on CIFAR-10; and all three Tiny ImageNet runs. Their final evaluations are complete. The trainers that produced them did not
  save RNG state, so a resumed run is not a bitwise continuation of an uninterrupted one. The
  `resumed_run` field of `release_manifest.json` gives the run segments. The other runs, including
  the four seed-42 CIFAR-100 files of the first release, ran without interruption.
- **Lineage.**
  - Most runs were trained on Colab A100 GPUs. The CIFAR-10 CP-FAIR and NoMem-L2 runs and MGC-D128
    seed 42 were trained on an NVIDIA DGX Spark (GB10).
  - The CIFAR-100 trio files are the extracted copies of the runs' Google Drive archive members,
    with equal CRC-32 and size. The Tiny ImageNet files are the archive members themselves.
  - The squared-hinge seed-42 files come from the Colab runs. Two aborted DGX runs of that seed
    have no final evaluation and are not released.
- **Trainers.**
  - The files were produced by earlier versions of the trainers: the v1 CIFAR-100 trainer (trio),
    the rebuttal-era MGC and squared-hinge trainers, and earlier versions of the CIFAR-10 and Tiny
    ImageNet trainers.
  - The camera-ready trainers in the code repository build the same networks. Every file's
    `net_ema` loads into its group's trainer with every key matched, and every key of its stored
    `cfg` is a field of that trainer's `FFConfig`. In the spot check above, the predictions match
    each run's saved ones, apart from 2 of the 896 MoE predictions (see Precision).
    The code repository's CPU equivalence tests compare the training step with the v1 and rebuttal
    trainers.
- **MGC.** The MGC arm replaces the baseline's depth-scaled current-block term: block 0 loses its
  auxiliary and has no compensator. The implementation is faithful but not an exact recovery
  (ε=10⁻⁶, positive-part clamp).
- **SAM.** The CIFAR-10 runs used SAM. The CIFAR-100 and Tiny ImageNet runs did not.

## Other checkpoints (on request)

`release_manifest.json` lists 106 checkpoint files: the `stage1_best.pt` and `stage2_best.pt` of
53 runs on CIFAR-10, CIFAR-100 and Tiny ImageNet. For each file it gives the run, the main paper
tables the file supports (not an exhaustive list), its trainer and configuration, its source and
lineage, its size and its SHA-256.

The 77 files marked `"availability": "on_request"` are available from the author on request
(amir.usefi75@gmail.com). A recipient can check them against the SHA-256 values in the manifest.
The following are not available at all:

- the published CIFAR-10 γ=0 L4/D256 seed-42 checkpoint, which is not recoverable;
- the text-domain and BP-control checkpoints (kept, but not released);
- the checkpoints of the CIFAR-100 32-expert γ=0.7 baseline (kept, but not released).

The manifest does not cover every run of the paper. For example, it does not list the checkpoints
of the single-seed CIFAR-10 variants of Tables 1 and 27 other than NoMem-L2, the
component ablation (Table 26), arms (a)–(c) of the MGC control (Table 6), the CIFAR-10 γ=0 seed-42
rerun (Table 22) or the three seeds of the CIFAR-100 32-expert γ=0.7 baseline (Table 38).

## Integrity

`SHA256SUMS` pins the 29 checkpoint files by their SHA-256 over the full file bytes, and
`release_manifest.json` records the same hashes. To verify a download, run the following in the
folder that holds `SHA256SUMS` and the downloaded `checkpoints/` directory (for example the
`--local-dir` of `hf download`); `--ignore-missing` skips the checkpoints you did not download:

```bash
sha256sum -c --ignore-missing SHA256SUMS
```

The release was staged with `release/stage_release.py --subset public` from the code repository.

## License

- **Checkpoints:** Creative Commons Attribution 4.0 International (CC BY 4.0). If you use them,
  please cite the paper (below).
- **Code:** Apache-2.0 (the `LICENSE` file of the GitHub repository).
- **Datasets:** the checkpoints were trained on CIFAR-10, CIFAR-100 and Tiny ImageNet, which are
  not redistributed here. A checkpoint stores network weights, optimizer and scheduler states and
  the contrastive feature queues, which hold projected features of training images and their
  labels. It contains no images. Use of the datasets themselves follows their authors' terms.

## Datasets

- **CIFAR-10 and CIFAR-100** (Krizhevsky, 2009), loaded with torchvision. The 50,000 training
  images are split into 45,000 for training and 5,000 for validation. The split is seeded with the
  run seed, so runs with the same seed share it. The 10,000-image test set is used only for the
  final evaluation.
- **Tiny ImageNet** (Le and Yang, 2015; 200 classes, 64×64 pixels), downloaded by the trainer. The
  100,000 training images are split into 90,000 for training and 10,000 for validation, seeded with
  the run seed. The labeled 10,000-image validation split serves as the test set and is used only
  for the final evaluation.

## Intended use and limitations

These checkpoints are for studying FF training. Possible uses:

- per-block separation and goodness;
- readouts and probes of the frozen backbones;
- prediction churn between arms and between seeds.

The paper reports the accuracies, protocols and seed-level variation: Tables 3, 5, 7, 21, 25 and 27,
and Appendix B. For the CIFAR-100 trio, the paired statistics can be recomputed as follows:

- S1 and S1-TTA: from these files, which now cover all three seeds;
- S2-TTA: from the prediction files in the code repository (`preds_cifar100/`);
- S2: with the Stage-2 heads, which are available on request (retrained probes give only
  approximate values).

The models are CIFAR- and Tiny ImageNet-scale classifiers (9.5 M to 107 M parameters) trained with
a local learning rule. They were not evaluated for robustness, fairness or out-of-distribution use.

## Citation

```bibtex
@inproceedings{yousefiramandi2026cumulative,
  title     = {Cumulative-Goodness Free-Riding in Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant},
  author    = {Yousefiramandi, Amirhossein},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.06240}
}
```
