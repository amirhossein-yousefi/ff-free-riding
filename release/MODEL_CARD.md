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
  - text-classification
  - layer-free-riding
  - neurips-2026
datasets:
  - uoft-cs/cifar10
  - uoft-cs/cifar100
  - stanfordnlp/imdb
  - fancyzhx/ag_news
metrics:
  - accuracy
---

# Forward-Forward free-riding checkpoints (NeurIPS 2026)

This repository holds 85 trained checkpoint files from *Cumulative-Goodness Free-Riding in
Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant* (Amirhossein
Yousefiramandi, NeurIPS 2026). They come from 47 training runs:

- **FF runs on images (29 runs, 58 files).** Each run has its validation-selected Stage-1
  network (`stage1_best.pt`) and its Stage-2 file (`stage2_best.pt`), which holds the same network
  plus the Stage-2 head.
- **BP calibration runs (9 runs, 9 files).** These are the CIFAR-10 backpropagation networks
  behind the BP rows of Table 2. Each file is the run's validation-selected network.
- **Text-domain runs (9 runs, 18 files).** These are the FF networks of the frozen-BERT check
  of App. K (Table 44). Each run has its Stage-1 network and its Stage-2 head.

| Group | Dataset and setup | Runs | Files | Paper (main tables) |
|---|---|---|---:|---|
| `c100_trio` | CIFAR-100, L4/D256 dense, no SAM, 362 Stage-1 epochs: block-local γ=0, hardness-gated κ=0, cumulative γ=0.7 | 3 arms × seeds 42/123/456 | 18 | Tables 3, 5 (cumulative reference row), 12, 15, 24, 29 and 39; Tables 9–11 (readouts and probes); Tables 13 and 36 (re-scored from the Stage-1 files) |
| `mgc_c100` | CIFAR-100, same setup: missing-gradient compensation (MGC, c_d=1) | seeds 42/123/456 | 6 | Tables 3 and 5; Table 9 (readouts) |
| `mgc_d128` | CIFAR-10, L4/D128, 16-expert MoE, SAM on, 180 Stage-1 epochs: MGC | seeds 42/123/456 | 6 | Tables 5 and 6; Table 8 (gradient share, seed 42) |
| `hinge` | CIFAR-10, same setup as `mgc_d128`: squared-hinge barrier, γ=0.7 and γ=0 arms | 2 arms × seeds 42/123/456 | 12 | Table 7 |
| `c10_cp_fair` | CIFAR-10, L4/D256, 32-expert MoE, SAM on, 362 Stage-1 epochs: CP-FAIR (γ=0.7 with a depth-scaled current-block term) | seeds 42/123/456/789 | 8 | Tables 1, 3, 21 and 27; Tables 2, 14, 30 and 41 (seed 42); Table 36 (re-scored from the Stage-1 files) |
| `c10_nomem_l2` | CIFAR-10, same network and 362-epoch Stage-1 schedule as `c10_cp_fair`: NoMem-L2, which keeps a uniform current-block weight (0.5) where CP-FAIR uses its depth-scaled term, with a depth-order weight of 0.1 and margin 0.05 where CP-FAIR uses 0.2 and 0.1 | seed 42 | 2 | Tables 27 and 41; App. N (Stage-2 summary) |
| `tin_gamma0` | Tiny ImageNet, L4/D256 dense, no SAM, 362 Stage-1 epochs: block-local γ=0 | seeds 42/123/456 | 6 | Tables 15, 25 and 29; Table 40 (seed 42) |
| `bp_calibration` | CIFAR-10, backpropagation, 362 epochs, AdamW: a plain CNN (1.15 M parameters) with weak augmentation, and the FF backbone stripped of its goodness heads (5.54 M) with weak and with strong (FF per-view) augmentation | 3 rows × seeds 42/123/456 | 9 | Table 2 (BP rows) |
| `text_domain` | IMDb, 20 Newsgroups and AG News: an FF L4/D256 network with a 16-expert MoE (top-4), γ=0, no SAM, on frozen `bert-base-uncased` sentence embeddings (8 virtual tokens); 25–40 Stage-1 and 8–10 Stage-2 epochs | 3 datasets × seeds 42/43/44 | 18 | Table 44 (App. K); Sec. 4.3 |

Table and appendix numbers refer to the camera-ready paper. The release grew in three steps:

- **2026-09-25:** the four seed-42 CIFAR-100 Stage-1 files of Table 3 (γ=0, κ=0, cumulative
  and MGC), uploaded with the code release.
- **2026-10-24:** 25 further Stage-1 files.
- **2026-10-26:** the Stage-2 files of these 29 runs, the BP calibration networks and the
  text-domain files.

`release_manifest.json` lists 48 further checkpoint files with their SHA-256 hashes. They are
available from the author on request (see [Other checkpoints](#other-checkpoints-on-request)): the
Stage-1 and Stage-2 files of the 24 CIFAR-10 γ=0, gated-ablation and hardness-gated κ=0 runs.

The checkpoints are research artifacts. They let you re-evaluate and probe the trained networks
without retraining them. They are not general-purpose classifiers.

- **Paper:** [arXiv:2605.06240](https://arxiv.org/abs/2605.06240) ·
  [OpenReview](https://openreview.net/forum?id=TN0FvatX6A)
- **Project page:** <https://amirhossein-yousefi.github.io/ff-free-riding/>
- **Code** (trainers, analysis, reproduction manifest; Apache-2.0):
  <https://github.com/amirhossein-yousefi/ff-free-riding>
- **Checkpoints** (this repository; CC BY 4.0): <https://huggingface.co/Amirhossein75/ff-free-riding>

## Files

The 85 files total 40,281,074,872 bytes (40.3 GB). Every table below gives each file's
size and SHA-256. The FF files keep the names the training scripts gave them, which carry the run
tags. The text-domain files are each run's `stage1_best.pt` and `stage2_best.pt`, prefixed with the
run name on upload, and the BP files are renamed (see their table). The Tiny ImageNet files are
tagged `tiny_imagenet_cp_fair_seed<seed>`, but their stored `cfg` has `gamma_scale=0` (block-local
γ=0).

### FF Stage-1 networks (`stage1_best.pt`)

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

The selected epoch is the file's `epoch_stage1`. The validation top-1 is its `best_s1`: the S1
readout with flip TTA on the run's held-out validation split, which selected the checkpoint.

### FF Stage-2 files (`stage2_best.pt`)

| File | Run | Stage-2 epoch (val. top-1, flip TTA) | Bytes | SHA-256 |
|---|---|---|---:|---|
| `checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed42_L4_D256_P2_seed42_stage2_best.pt` | CIFAR-100 γ=0, seed 42 | 20 (67.46 %) | 514031757 | `a3cd9ac3d04e81c679320b5562f67bbc371f3b36db449fc49a53276075179ad6` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed123_L4_D256_P2_seed123_stage2_best.pt` | CIFAR-100 γ=0, seed 123 | 20 (68.96 %) | 514052817 | `1575c53e078fb3d3cce9c02ca96c465d9c88d1dd3abf1dd783b88d22bfe26826` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed456_L4_D256_P2_seed456_stage2_best.pt` | CIFAR-100 γ=0, seed 456 | 20 (68.80 %) | 514052817 | `8b70f66de851c2d0ab31456d1ba848e2545b4c8bdba0b31a2552ca91f7e3da25` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gated_k0_seed42_L4_D256_P2_seed42_stage2_best.pt` | CIFAR-100 κ=0, seed 42 | 20 (67.26 %) | 514050385 | `ef050a42e114a183752bed960268d96f5934aedfa309e538568da56e43b629f2` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gated_k0_seed123_L4_D256_P2_seed123_stage2_best.pt` | CIFAR-100 κ=0, seed 123 | 20 (68.90 %) | 514053141 | `ff6a8df11f1b3d48a0e925da35521aae4a9a49caa020684c751a9a8677e66b78` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gated_k0_seed456_L4_D256_P2_seed456_stage2_best.pt` | CIFAR-100 κ=0, seed 456 | 20 (69.20 %) | 514055573 | `7943f0a9f34b564453e6c770cd3be874b34805a85c404a74c1e1fd13b2f4ac32` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_cumulative_seed42_L4_D256_P2_seed42_stage2_best.pt` | CIFAR-100 cumulative γ=0.7, seed 42 | 20 (67.20 %) | 514053141 | `00d8b2adc0ebf2fa708fbcfdc449a4611a0edc5802106ae4500b0e043a3cf0d4` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_cumulative_seed123_L4_D256_P2_seed123_stage2_best.pt` | CIFAR-100 cumulative γ=0.7, seed 123 | 20 (68.10 %) | 514108953 | `515bd427dc2ddff6dc50ff4deec1c92067b736a53724665f6bc4a79ce704cb2d` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_cumulative_seed456_L4_D256_P2_seed456_stage2_best.pt` | CIFAR-100 cumulative γ=0.7, seed 456 | 20 (69.46 %) | 514108953 | `46f7c323ec02b89a321e0040677cbd6304bddac769d89747df6e53a69431b35e` |
| `checkpoints/mgc_c100/ff_cifar100_mgc_c100_L4_D256_P2_seed42_stage2_best.pt` | CIFAR-100 MGC, seed 42 | 20 (67.92 %) | 514012849 | `45a94a638f1a534f785e3db05076122d652c412b42d70696413c2dfaadd81a44` |
| `checkpoints/mgc_c100/ff_cifar100_mgc_c100_L4_D256_P2_seed123_stage2_best.pt` | CIFAR-100 MGC, seed 123 | 20 (67.80 %) | 514014035 | `54854ea5ea8312baed19cfaddb5283a780ef8ba6497c905316cce02a0974a929` |
| `checkpoints/mgc_c100/ff_cifar100_mgc_c100_L4_D256_P2_seed456_stage2_best.pt` | CIFAR-100 MGC, seed 456 | 20 (68.28 %) | 514016403 | `31131d835ba96bf9b4f7d2621ac8829b2cb21241e469698a44df10021556c62f` |
| `checkpoints/mgc_d128/ff_cifar10_mgc_c1_L4_D128_P2_seed42_stage2_best.pt` | CIFAR-10 MGC-D128, seed 42 | 10 (86.54 %) | 317430107 | `acc0843e910f591a172cfec3f606cc7495217a13c1299dba78d747edc09a357a` |
| `checkpoints/mgc_d128/ff_cifar10_mgc_c1_L4_D128_P2_seed123_stage2_best.pt` | CIFAR-10 MGC-D128, seed 123 | 10 (87.40 %) | 317432557 | `8a72f497fcb5767ac1a0abb6c64729813b0261622738cf1984a872f0ceb0e356` |
| `checkpoints/mgc_d128/ff_cifar10_mgc_c1_L4_D128_P2_seed456_stage2_best.pt` | CIFAR-10 MGC-D128, seed 456 | 10 (86.24 %) | 317432557 | `b1ba4b7053bcb0731cac8c0c119798107621c15022e6a3523983024a3626e343` |
| `checkpoints/hinge/ff_cifar10_hinge_g07_L4_D128_P2_seed42_stage2_best.pt` | CIFAR-10 hinge γ=0.7, seed 42 | 10 (86.44 %) | 317437393 | `e2bd42c772587805c2cf12e7f65022a0e9f7d5d973d52ee427173026fd82663b` |
| `checkpoints/hinge/ff_cifar10_hinge_g07_L4_D128_P2_seed123_stage2_best.pt` | CIFAR-10 hinge γ=0.7, seed 123 | 10 (86.84 %) | 317439779 | `2d979b898a7c78722ffba31d0e086f1f3c3ed09470356318433105062ad8f753` |
| `checkpoints/hinge/ff_cifar10_hinge_g07_L4_D128_P2_seed456_stage2_best.pt` | CIFAR-10 hinge γ=0.7, seed 456 | 10 (86.80 %) | 317444771 | `3938cd3fe88bbe81f3bc4f80c5caaca22cbc52a371d8d2986ed9cfd07e0fc59f` |
| `checkpoints/hinge/ff_cifar10_hinge_g0_L4_D128_P2_seed42_stage2_best.pt` | CIFAR-10 hinge γ=0, seed 42 | 10 (86.74 %) | 317435007 | `fd1546f7f855e2bc5d1c441c1131720c7cf44cdd6b22f5e3aa74891bd9dc63cc` |
| `checkpoints/hinge/ff_cifar10_hinge_g0_L4_D128_P2_seed123_stage2_best.pt` | CIFAR-10 hinge γ=0, seed 123 | 10 (86.86 %) | 317437393 | `9bd92640b1e5b292ca8568407c84542f2ef19fd98cc30f4d04da5fbd43e98176` |
| `checkpoints/hinge/ff_cifar10_hinge_g0_L4_D128_P2_seed456_stage2_best.pt` | CIFAR-10 hinge γ=0, seed 456 | 10 (86.30 %) | 317437393 | `49ae58e124dc524e06a1094cc10dfd9d75e6f9414c1192b5b861f07270945014` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_P2_seed42_stage2_best.pt` | CIFAR-10 CP-FAIR, seed 42 | 20 (91.80 %) | 2078774721 | `1aad86b0ee575a84f05c24923e1a26f80b0c96309032ddd85a1d0d420628daab` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_P2_seed123_stage2_best.pt` | CIFAR-10 CP-FAIR, seed 123 | 20 (91.50 %) | 2078807779 | `c9dffedd3230a2e35ed5600d112c192873c3905d1ea25790979fd58382872ca6` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_P2_seed456_stage2_best.pt` | CIFAR-10 CP-FAIR, seed 456 | 20 (91.76 %) | 2078807779 | `20901dd307b0bd549bfc8a929b1bd50756baa2db727c8b0d1af6bfcc289b3d8f` |
| `checkpoints/c10_cp_fair/ff_cifar10_cifar10_cp_fair_clean_progressive_seed789_L4_D256_P2_seed789_stage2_best.pt` | CIFAR-10 CP-FAIR, seed 789 | 20 (91.32 %) | 2078799907 | `c87a91d18ea44ba39375dcee72feb9ca89342bb4190aa073f81f5380537a9d3c` |
| `checkpoints/c10_nomem_l2/ff_cifar10_cifar10_opt_do_this_first_ynorm_nomem_resid_L4_D256_P2_seed42_stage2_best.pt` | CIFAR-10 NoMem-L2, seed 42 | 40 (91.92 %) | 2078803509 | `4c9a6c49c2b546837d8a3131b19beaffec2806ec8619b3258a01c86b7fa37431` |
| `checkpoints/tin_gamma0/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed42_L4_D256_P4_seed42_stage2_best.pt` | Tiny ImageNet γ=0, seed 42 | 20 (51.67 %) | 535042691 | `a42d38224ca8ffcdb547bfad7f871c7f286939a5a17df07ac7254baaac1179f7` |
| `checkpoints/tin_gamma0/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed123_L4_D256_P4_seed123_stage2_best.pt` | Tiny ImageNet γ=0, seed 123 | 20 (51.75 %) | 535045063 | `e2005ac186b6dabdd0324eab581bac6f91922e253ce4446e99160585eac799f6` |
| `checkpoints/tin_gamma0/ff_tiny_imagenet_tiny_imagenet_cp_fair_seed456_L4_D256_P4_seed456_stage2_best.pt` | Tiny ImageNet γ=0, seed 456 | 20 (52.11 %) | 535045063 | `bbedea198b61c43925816ff0344f5e9d204dbae8ac3f957ea18b22e9009b8c42` |

Stage 2 trained the head for 20 epochs (MGC-D128 and the squared hinge: 10; NoMem-L2: 300). The
trainers scored the head on the validation split every 20 epochs and at the last epoch, and kept the
head with the highest score. Every file except NoMem-L2's therefore holds the last-epoch head;
NoMem-L2's holds the epoch-40 head. The Stage-2 epoch is the file's `epoch_stage2`, and the
validation top-1 is its `best_s2`, the S2 readout with flip TTA on the same validation split.

### BP calibration networks (`bp_calibration`)

| File | Run | Selected epoch (val. top-1) | Bytes | SHA-256 |
|---|---|---|---:|---|
| `checkpoints/bp_calibration/plain_cnn_bp_seed42_best.pt` | Plain CNN + BP, seed 42 | 281 (90.78 %) | 13820613 | `ec91c970758bf841097880cd3ad0de489ab0ffff3a7cfdb32589fd8d704fb460` |
| `checkpoints/bp_calibration/plain_cnn_bp_seed123_best.pt` | Plain CNN + BP, seed 123 | 266 (90.82 %) | 13820613 | `864cedf3c32acd7e9063e82cd8b9065fc1dc89c8c448995577584ff1ac2eadd9` |
| `checkpoints/bp_calibration/plain_cnn_bp_seed456_best.pt` | Plain CNN + BP, seed 456 | 341 (91.14 %) | 13820613 | `2fa46a0301ca6fb599864685013c530d079283b61a346adf9ca90606b4173d60` |
| `checkpoints/bp_calibration/bp_weak_seed42_best.pt` | Stripped backbone + BP, weak aug., seed 42 | 360 (87.34 %) | 66560075 | `f9a48b227fc0f5d992d6e771ad58cadff29898eb2405d861e501990da996bdb7` |
| `checkpoints/bp_calibration/bp_weak_seed123_best.pt` | Stripped backbone + BP, weak aug., seed 123 | 362 (87.38 %) | 66560075 | `bf04311efc21bd8f1d328edcbf133a5e4e8aa087687db6a4d4bf8a6009e8b1a2` |
| `checkpoints/bp_calibration/bp_weak_seed456_best.pt` | Stripped backbone + BP, weak aug., seed 456 | 360 (88.76 %) | 66560075 | `7a0b317a5611b22e3ff0a338cd97f8b5280ac3de46954aaa4831908d125e8648` |
| `checkpoints/bp_calibration/bp_strong_seed42_best.pt` | Stripped backbone + BP, strong aug., seed 42 | 362 (93.86 %) | 66560075 | `ba8127bed716cf0ee9efc1f75e98750215429397097c3409c36aebbb6ccf7529` |
| `checkpoints/bp_calibration/bp_strong_seed123_best.pt` | Stripped backbone + BP, strong aug., seed 123 | 362 (93.92 %) | 66560075 | `390abfdee6c4b79ca80b73d96192db4b1ab71e248f00ee194fe3b56ae3f8930e` |
| `checkpoints/bp_calibration/bp_strong_seed456_best.pt` | Stripped backbone + BP, strong aug., seed 456 | 360 (94.20 %) | 66560075 | `12c5a72a6828fccef7dfc7019c06fef1de2332c6ed88558600a62940b6c4ddf6` |

Each file is the run's `checkpoints/best.pt`, renamed on upload. Every run scored its validation
split during training (the plain CNN every epoch, the stripped backbone every 20 epochs and at the
last) and saved the network at the epoch with the highest validation top-1 (single crop).

### Text-domain networks and heads (`text_domain`)

| File | Run | Selected epoch (val. top-1) | Bytes | SHA-256 |
|---|---|---|---:|---|
| `checkpoints/text_domain/imdb_bert_base_uncased_L4_D256_V8_seed42_stage1_best.pt` | IMDb, seed 42 (Stage 1) | 26 (87.60 %) | 462854447 | `0f8537cfee811e658065f45aea5ec58cdf4a18c71675130482599abd215214af` |
| `checkpoints/text_domain/imdb_bert_base_uncased_L4_D256_V8_seed42_stage2_best.pt` | IMDb, seed 42 (Stage-2 head) | 3 (87.84 %) | 46270286 | `43aa9fc8c011305a359bfdc0a58039d20b629c51c82af7d221af4b2422ca2c18` |
| `checkpoints/text_domain/imdb_bert_base_uncased_L4_D256_V8_seed43_stage1_best.pt` | IMDb, seed 43 (Stage 1) | 32 (87.48 %) | 462854447 | `ee9690c9fdd072bdb4e9406b81a8074350ba6e38d9a7c46c08fbdb64d8e240cf` |
| `checkpoints/text_domain/imdb_bert_base_uncased_L4_D256_V8_seed43_stage2_best.pt` | IMDb, seed 43 (Stage-2 head) | 2 (87.60 %) | 46270286 | `10a018e9717a38cf0b51abde4bf7e2034c19edba9d97dd52e107023b352beb55` |
| `checkpoints/text_domain/imdb_bert_base_uncased_L4_D256_V8_seed44_stage1_best.pt` | IMDb, seed 44 (Stage 1) | 28 (88.64 %) | 462854447 | `d8ea65e07f1a3e4573a358ec4767fcdbf07558c0672e0f55e2af129d44336ce7` |
| `checkpoints/text_domain/imdb_bert_base_uncased_L4_D256_V8_seed44_stage2_best.pt` | IMDb, seed 44 (Stage-2 head) | 4 (89.04 %) | 46270286 | `e1ef077edabbeffb887fcea9230e1996c9c962ffa991713adb636c4831017091` |
| `checkpoints/text_domain/20newsgroups_bert_base_uncased_L4_D256_V8_seed42_stage1_best.pt` | 20 Newsgroups, seed 42 (Stage 1) | 34 (67.67 %) | 463038767 | `a780df67b079ad5fd05f09f287d5cd2f87fdab59024e49e5aa40905a540dd3d8` |
| `checkpoints/text_domain/20newsgroups_bert_base_uncased_L4_D256_V8_seed42_stage2_best.pt` | 20 Newsgroups, seed 42 (Stage-2 head) | 6 (68.82 %) | 46270286 | `829fb9a4eee367c271c6cbe1328657eda175089d839a7e5c71690ed3062def07` |
| `checkpoints/text_domain/20newsgroups_bert_base_uncased_L4_D256_V8_seed43_stage1_best.pt` | 20 Newsgroups, seed 43 (Stage 1) | 35 (63.52 %) | 463038767 | `c59827625a8e5c957a60ca61c65348353be8a1fbb95c75e4e50a98f292165386` |
| `checkpoints/text_domain/20newsgroups_bert_base_uncased_L4_D256_V8_seed43_stage2_best.pt` | 20 Newsgroups, seed 43 (Stage-2 head) | 8 (65.99 %) | 46270286 | `598631d8b425434514e4d06b6e168a7704e211292ee1ebd4d452f69f9a6fe76f` |
| `checkpoints/text_domain/20newsgroups_bert_base_uncased_L4_D256_V8_seed44_stage1_best.pt` | 20 Newsgroups, seed 44 (Stage 1) | 35 (65.99 %) | 463038767 | `067e55d3f48efd9e78e6797bb1d7c5bc259843691befd31f776d5dba78b44d18` |
| `checkpoints/text_domain/20newsgroups_bert_base_uncased_L4_D256_V8_seed44_stage2_best.pt` | 20 Newsgroups, seed 44 (Stage-2 head) | 8 (67.49 %) | 46270286 | `790dff470ff423caaeadf48ed643b1786849d8dd197e34b6dee6aabbb8b6cc1b` |
| `checkpoints/text_domain/ag_news_bert_base_uncased_L4_D256_V8_seed42_stage1_best.pt` | AG News, seed 42 (Stage 1) | 19 (93.36 %) | 462874927 | `f628c300db6dfc82feed3ac4e1c9db4b5a472ac7ca2e738fc665f18470dd3e0c` |
| `checkpoints/text_domain/ag_news_bert_base_uncased_L4_D256_V8_seed42_stage2_best.pt` | AG News, seed 42 (Stage-2 head) | 8 (93.26 %) | 26841934 | `388485f897f5f125962ad1ca48fb76f1a8799f06778ce7d9569970d059640f23` |
| `checkpoints/text_domain/ag_news_bert_base_uncased_L4_D256_V8_seed43_stage1_best.pt` | AG News, seed 43 (Stage 1) | 15 (93.14 %) | 462874927 | `a64698e4b7d049b62546b31c64f082b81d0ff92805d91cf0c7c168e6755ce845` |
| `checkpoints/text_domain/ag_news_bert_base_uncased_L4_D256_V8_seed43_stage2_best.pt` | AG News, seed 43 (Stage-2 head) | 6 (93.17 %) | 26841934 | `6262bf566c4061c97368ebf67b29a0bd77ff6382548db81c6ba7e06a425b20f1` |
| `checkpoints/text_domain/ag_news_bert_base_uncased_L4_D256_V8_seed44_stage1_best.pt` | AG News, seed 44 (Stage 1) | 20 (93.27 %) | 462874927 | `27e76519473485683067cdb2472d341c6420bb03172a1b3af2d7344be418c156` |
| `checkpoints/text_domain/ag_news_bert_base_uncased_L4_D256_V8_seed44_stage2_best.pt` | AG News, seed 44 (Stage-2 head) | 3 (93.36 %) | 26841934 | `27ce497b9285b5aa7f42887ba238a9eb69fe3a92fd347101c2998bec131d3eb5` |

For a Stage-1 file, the selected epoch is the Stage-1 epoch with the highest validation top-1 of
the EMA network (single pass, no TTA). For a Stage-2 file, it is the Stage-2 epoch with the highest
validation top-1 of the head. Each run holds out a stratified 10% of its training set for this
selection.

### Other files and file contents

The repository also contains:

```
README.md                this model card
SHA256SUMS               sha256sum-compatible list of the 85 checkpoint files
release_manifest.json    the 133 checkpoint files of the release (71 runs): run, paper tables, trainer,
                         source, lineage, size, SHA-256 and the checks below; `availability` is
                         "public_hf" for the 85 files here and "on_request" for the others
```

The files store the following keys:

- **FF `stage1_best.pt`:**
  - `net_ema`: the EMA weights, which give the S1 numbers;
  - `model`: the raw weights;
  - `cfg`: the run's full configuration, identical to the run's `config.json` except `device`;
  - `epoch_stage1`: the selected epoch;
  - `best_s1`: the validation top-1 at that epoch (S1 readout with flip TTA);
  - `stage`: 1;
  - `opt_states`, `sched_states` and `queues`: the optimizer states, the scheduler states and the
    contrastive feature queues.
- **FF `stage2_best.pt`:** the same keys, plus the following:
  - `head_state`: the Stage-2 head;
  - `opt_head_state` and `sched_head_state`: the head's optimizer and scheduler states;
  - `epoch_stage2` and `best_s2`: the epoch of the saved head and its validation top-1;
  - `stage`: 2.

  Its `model` and `net_ema` equal those of the run's `stage1_best.pt`, tensor for tensor, and so
  does its `best_s1`. Its `epoch_stage1` is the run's last Stage-1 epoch (362; MGC-D128 and the
  squared hinge: 180), not the selected one. Stage 2 starts from the selected network.
- **BP `best.pt`:**
  - `model`: the network weights;
  - `optimizer` and `scheduler`: their states;
  - `epoch` and `best_val_top1`: the selected epoch and its validation top-1;
  - `version_tag`: names the trainer;
  - the run's configuration: `config` for the stripped backbone, `cfg` for the plain CNN. Both
    equal the run's `config.json` except `device`;
  - `best_epoch`: the stripped-backbone files only.
- **Text `stage1_best.pt`:**
  - `ema`: the EMA weights, which give the Stage-1 numbers;
  - `model`: the raw weights;
  - `best_val_top1`: the validation top-1 at the selected epoch;
  - `cfg`: the run's configuration, equal to its `config.json`.

  It holds the FF network only. The BERT encoder is not included, so the network takes
  `bert-base-uncased` sentence embeddings as input.
- **Text `stage2_best.pt`:**
  - `head`: the Stage-2 head;
  - `best_val_top1`: the head's validation top-1;
  - `cfg`: the same configuration.

  The head runs on the EMA network of the run's `stage1_best.pt`.

## Numbers behind each file

### FF runs

| Run | S1 | S1-TTA | S2 | S2-TTA |
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

- **S1 and S1-TTA** are the Stage-1 readouts. S1 is the goodness-sum readout of the EMA weights on
  a single crop; S1-TTA adds the horizontally flipped image.
- **S2 and S2-TTA** come from the Stage-2 attentive head in `stage2_best.pt`. This head is a
  backprop-trained probe on the frozen backbone, not part of the FF method. For NoMem-L2 the paper
  prints no Stage-2 value of its own: its S2-TTA enters the Stage-2 summary of App. N, which is
  taken over the 12 seed-42 variants of Table 27 that were run through Stage 2.
- **Deepest-block separation** (the "sep" columns of Tables 3, 5 and 7) is a training-log quantity:
  it was read at the last Stage-1 epoch, not recomputed from the validation-selected checkpoint.
  The code repository ships the per-seed values in `metric_summaries/camera_ready/`. Table 36 gives
  the deepest-block separation recomputed on the validation-selected checkpoints themselves (EMA
  weights, eval mode), including the CIFAR-100 trio and CP-FAIR files here.

### BP calibration runs

| Run | Parameters | Test top-1 |
|---|---:|---:|
| Plain CNN + BP, seed 42 | 1,148,874 | 89.86 |
| Plain CNN + BP, seed 123 | 1,148,874 | 89.98 |
| Plain CNN + BP, seed 456 | 1,148,874 | 90.39 |
| Stripped backbone + BP, weak aug., seed 42 | 5,539,530 | 86.90 |
| Stripped backbone + BP, weak aug., seed 123 | 5,539,530 | 87.29 |
| Stripped backbone + BP, weak aug., seed 456 | 5,539,530 | 88.11 |
| Stripped backbone + BP, strong aug., seed 42 | 5,539,530 | 93.90 |
| Stripped backbone + BP, strong aug., seed 123 | 5,539,530 | 94.00 |
| Stripped backbone + BP, strong aug., seed 456 | 5,539,530 | 93.65 |

The test top-1 is the single-crop value that each run logged in `final_metrics.json` after
reloading its selected network. Table 2 prints the mean and SD over the three seeds of each row
(per-seed values: `metric_summaries/camera_ready/bp_calibration.json` of the code repository).

### Text-domain runs

| Run | Stage 1 (test top-1) | Stage 2 (test top-1) |
|---|---:|---:|
| IMDb, seed 42 | 86.89 | 87.24 |
| IMDb, seed 43 | 86.91 | 87.10 |
| IMDb, seed 44 | 87.11 | 87.41 |
| 20 Newsgroups, seed 42 | 61.64 | 62.44 |
| 20 Newsgroups, seed 43 | 61.71 | 62.61 |
| 20 Newsgroups, seed 44 | 62.16 | 62.92 |
| AG News, seed 42 | 93.45 | 93.45 |
| AG News, seed 43 | 93.24 | 93.34 |
| AG News, seed 44 | 93.46 | 93.30 |

The values are the test top-1 (%) that each run logged in `final_metrics.json`. Table 44 prints
the mean and SD over the three seeds of each dataset. The test sets are the standard test splits:
IMDb 25,000 reviews, 20 Newsgroups 7,532 posts and AG News 7,600 articles.

### Checks before upload

Every file was checked on CPU before upload. `release_manifest.json` records the results for each
file (`lineage_check`, `loading_spot_check`).

**FF Stage-1 files (2026-10-24):**

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
- The loading code below, run on CPU in float32, gave the same single-crop prediction as the run's own saved test predictions (`preds_stage1_test_no_tta.npz`) for 1,854 of 1,856 file–image pairs: each of the 29 Stage-1 files scored the first 64 images of its test set. The dense files agree on 960 of 960, with every per-class score within 0.03 of the saved one. The MoE files (`mgc_d128`, `hinge`, `c10_cp_fair`, `c10_nomem_l2`) agree on 894 of 896, with every per-class score within 1.15 of the saved one. The original evaluations ran on GPUs with TF32; see Precision below.

S1 was also re-predicted on the full test set for 22 of the files, each within 0.02 pp of its run
log. Twelve of them have identity checks of their own (the seed-42 CIFAR-100 trio, all MGC CIFAR-100
files, and seeds 123/456 of MGC-D128 and of both squared-hinge arms; each within 0.01 pp), and the
re-scoring behind Tables 13 and 36 covers all nine CIFAR-100 trio files and the four CP-FAIR files.
`RELEASE_CHECKPOINTS.md` in the code repository lists them.

**FF Stage-2 files (2026-10-26):**

- Its SHA-256, size and CRC-32 equal the manifest.
- The run's final test evaluation names the file, and its `best_s2` equals the run's `run_end`
  value.
- The logged S2 and S2-TTA equal the per-seed values behind the paper's tables. For Tiny ImageNet
  these are S2-TTA top-1 and top-5; NoMem-L2 has no per-file value.
- Its stored `cfg` equals the run's `config.json`.
- Its `model` and `net_ema` equal the run's public `stage1_best.pt`, tensor for tensor.
- It loads with `weights_only=True`, and its `net_ema` and `head_state` load into the group's
  trainer with every key matched.
- The Stage-2 loading code below, run on CPU in float32, gave the same single-crop prediction as the run's own saved Stage-2 test predictions (`preds_stage2_test_no_tta.npz`) for 1,856 of 1,856 file–image pairs (each file scored the first 64 images of its test set). The dense files agree on 960 of 960, with every class logit within 0.09 of the saved one. The MoE files agree on 896 of 896, with every class logit within 2.21 of the saved one.

**BP calibration files (2026-10-26):**

- The run's `final_metrics.json` names the file. Its stored epoch and validation top-1 equal
  `final_metrics.json` and the run's `history.jsonl`.
- Its stored configuration equals the run's `config.json`, and its `version_tag` names the
  trainer.
- The run's logged test top-1 is the per-seed value behind Table 2.
- It loads with `weights_only=True` into the trainer's model class with every key matched, and the
  parameter count equals the run's.
- On CPU in float32, each network scored the full CIFAR-10 test set. Its top-1 is within 2 test images (0.02 pp) of the run's logged value, with 3 of the 9 equal. The 6 stripped-backbone runs saved their test predictions (`preds_test.npz`), and 9,998 to 10,000 of the 10,000 CPU predictions of each agree with them, with every logit within 0.04 of the saved one.

**Text-domain files (2026-10-26):**

- Each run's files are members of a per-run `.tar.gz` inside a Google Drive export zip. Both layers
  were streamed, and zipfile verified the tarball's CRC-32.
- The tarball's `final_metrics.json` equals the archived copies (the export's `metrics/` folder and
  run summary), and the means over the three seeds equal Table 44.
- Each file's `best_val_top1` equals `final_metrics.json` and the maximum of the run's history, and
  its `cfg` equals the run's `config.json`.
- Both files load with `weights_only=True`. The `ema` weights load into the released text
  trainer's network, and the `head` into the core's attentive head, each with every key matched.
- On CPU, the loading code scored the full test set of each run from `bert-base-uncased` embeddings re-encoded on another machine (not the runs' Colab cache), at the routing temperature of the final evaluation. Against each run's logged values, the number of correct test texts differs by −5 to +1 (Stage 1) and −2 to +2 (Stage 2). The validation split, scored at the routing temperature of the selected epoch, differs by −1 to +1 correct texts from the value that selected the Stage-1 file.

## Loading

### FF files

Each FF group has its own trainer in the code repository. The file's stored `cfg` configures it.

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

For a `stage2_best.pt` file, load the network from the same file and add the head:

```python
FILE = "checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed42_L4_D256_P2_seed42_stage2_best.pt"
ckpt = torch.load(hf_hub_download("Amirhossein75/ff-free-riding", FILE), map_location="cpu", weights_only=True)
# ... build cfg and net from ckpt["cfg"] and ckpt["net_ema"] exactly as above, then:
head = T.AttentiveHybridHead(in_dim_g=cfg.num_blocks * cfg.n_aspects, in_dim_a=cfg.d_model * cfg.num_blocks,
                             hidden=cfg.head_hidden, num_classes=cfg.num_classes,
                             num_heads=cfg.head_num_heads, dropout=cfg.head_dropout).to(cfg.device)
head.load_state_dict(ckpt["head_state"])
head.eval()
acc = T.evaluate_stage2(net, head, loader, cfg.device, tta=False, topk=(1,))   # tta=True gives S2-TTA
print(f"S2 top-1: {100 * acc[1]:.2f} %")         # logged: 68.27 for this file
```

### BP calibration files

The trainers are the original run scripts in `trainers/bp_original/`. The `version_tag` stored in
each file names its script: `plain_cnn_bp_cifar10.py` (plain CNN),
`ff_backbone_stripped_bp_cifar10.py` (stripped backbone, weak augmentation) and
`ff_backbone_stripped_bp_strongaug_cifar10.py` (stripped backbone, strong augmentation; same model
class). The test set is the CIFAR-10 test set with the CIFAR-10 normalization above, and there is
no TTA.

```python
import importlib.util, sys, torch
from huggingface_hub import hf_hub_download
from torchvision import datasets, transforms

FILE = "checkpoints/bp_calibration/bp_strong_seed42_best.pt"
ckpt = torch.load(hf_hub_download("Amirhossein75/ff-free-riding", FILE), map_location="cpu", weights_only=True)
TRAINER = "trainers/bp_original/" + ckpt["version_tag"].removesuffix("_v1") + ".py"   # the script that trained the file
spec = importlib.util.spec_from_file_location("bp_trainer", TRAINER)
B = importlib.util.module_from_spec(spec); sys.modules["bp_trainer"] = B; spec.loader.exec_module(B)
device = "cuda" if torch.cuda.is_available() else "cpu"
if "config" in ckpt:                             # stripped backbone (bp_weak, bp_strong)
    cfg = B.StrippedBPConfig()
    for k, v in ckpt["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = device
    model = B.StrippedBackboneBP(cfg)
else:                                            # plain CNN (plain_cnn_bp)
    model = B.PlainCNN(num_classes=10, channels=ckpt["cfg"]["channels"], dropout=ckpt["cfg"]["dropout"])
model.load_state_dict(ckpt["model"])
model = model.to(device).eval()

norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
test = datasets.CIFAR10("./data_c10", train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor(), norm]))
correct = 0
with torch.no_grad():
    for x, y in torch.utils.data.DataLoader(test, batch_size=500):
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
print(f"test top-1: {100 * correct / len(test):.2f} %")   # logged: 93.90 for this file
```

### Text-domain files

The trainer is `trainers/text_original/cp_fair_text_benchmark.py`, the runner that produced the
nine runs. It imports `trainers/cp_fair_core.py` and needs `pip install transformers datasets
scikit-learn`. The network's input is a frozen `bert-base-uncased` sentence embedding: the masked
mean of the last hidden states, L2-normalized, with the run's `max_length` (IMDb and 20 Newsgroups
256, AG News 128). The trainer's `encode_texts` computes it.

```python
import importlib.util, sys, torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

spec = importlib.util.spec_from_file_location("text_trainer", "trainers/text_original/cp_fair_text_benchmark.py")
TB = importlib.util.module_from_spec(spec); sys.modules["text_trainer"] = TB; spec.loader.exec_module(TB)

RUN = "checkpoints/text_domain/imdb_bert_base_uncased_L4_D256_V8_seed42"
s1 = torch.load(hf_hub_download("Amirhossein75/ff-free-riding", RUN + "_stage1_best.pt"), map_location="cpu", weights_only=True)
s2 = torch.load(hf_hub_download("Amirhossein75/ff-free-riding", RUN + "_stage2_best.pt"), map_location="cpu", weights_only=True)
cfg = TB.TextFFConfig()
for k, v in s1["cfg"].items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
net = TB.TextFFHybridNet(cfg).to(cfg.device)
net.load_state_dict(s1["ema"])                   # EMA weights: the Stage-1 numbers
net.eval()
net.set_router_tau(cfg.gate_tau_end, cfg.gate_tau_depth_slope)   # routing temperature of the final evaluation
head = TB.core.AttentiveHybridHead(in_dim_g=cfg.num_blocks * cfg.n_aspects, in_dim_a=cfg.d_model * cfg.num_blocks,
                                   hidden=cfg.head_hidden, num_classes=cfg.num_classes,
                                   num_heads=cfg.head_num_heads, dropout=cfg.head_dropout).to(cfg.device)
head.load_state_dict(s2["head"])
head.eval()

_, _, test_texts, test_labels, _ = TB.load_text_classification_data(cfg.dataset)
x = TB.encode_texts(test_texts, cfg.encoder_model, cfg.max_length, 64, cfg.device, cfg.encoder_pool)
loader = DataLoader(TB.EmbeddingSingleDataset(x, torch.as_tensor(test_labels)), batch_size=cfg.test_batch_size)
s1_acc = TB.core.evaluate_ff(net, loader, cfg.device, tta=False, topk=(1,))[1]
s2_acc = TB.core.evaluate_stage2(net, head, loader, cfg.device, tta=False, topk=(1,))[1]
print(f"Stage 1: {100 * s1_acc:.2f} %, Stage 2: {100 * s2_acc:.2f} %")   # logged: 86.89 and 87.24
```

- **Routing temperature.** The MoE router's temperature is not part of the state dict. The runs
  annealed it from `gate_tau_start` (2.0) to `gate_tau_end` (0.75) over Stage 1 and ran their final
  evaluation at the last epoch's value, `gate_tau_end`. A freshly built network starts at
  `gate_tau_start`. Left at `gate_tau_start`, the nine networks' Stage-1 test top-1 moved by up to 0.45 pp in the CPU check.
- **Embeddings.** The trainer caches the embeddings in float16 (`--cache_dir`). A run that found a
  cache used that copy, and a run that encoded the texts itself used float32. The two differ only by
  float16 rounding.

### Notes

- **`weights_only`.** The files are PyTorch pickles. With PyTorch 2.14.0, all 85 load with
  `weights_only=True`, because they contain only tensors and plain Python containers. If your
  PyTorch version refuses a file, `torch.load(..., weights_only=False)` will load it, but that
  setting executes arbitrary pickle code. Use it only after the file's SHA-256 matches: compare
  `sha256sum <file>` with the tables above, or run the check in [Integrity](#integrity). The
  trainers' own `torch.load` calls, used by the Stage-2 retraining below, do not pass the argument,
  so they fall back to PyTorch's default.
- **`map_location`.** Keep `map_location="cpu"` (or your device) as in the code above. Every file
  holds tensors saved on the GPU (the text-domain files hold only GPU tensors), so on a machine
  without CUDA, `torch.load` fails without it.
- **Precision.** The runs were evaluated on GPUs with TF32: the trainers call
  `torch.set_float32_matmul_precision('high')` on the GPU (the FF image trainers on compute
  capability 8 or higher), and cuDNN convolutions use TF32 by default. Top-k expert routing is discontinuous, so in the MoE files
  (`mgc_d128`, `hinge`, `c10_cp_fair`, `c10_nomem_l2`, `text_domain`) a small numerical difference
  can change which experts a token uses. In the FF spot checks above (the first 64 test images of each file), the float32 CPU evaluation changed 2 of the 896 MoE Stage-1 predictions (`hinge`, `mgc_d128`), none of the 896 MoE Stage-2 predictions and none of the 1,920 dense ones. On the full CIFAR-10 test set, it changed 0 to 2 of the 10,000 predictions of each stripped-backbone BP network, which is dense. To match the logged values closely,
  evaluate on such a GPU after calling `torch.set_float32_matmul_precision('high')`.
- **Side effects.** Importing an FF trainer creates an empty `./checkpoints/` directory.
  `download=True` fetches CIFAR into `./data_c100/` or `./data_c10/`. The Tiny ImageNet helper
  `T.download_and_prepare_tiny_imagenet()` downloads the dataset from the CS231n site into
  `./data_tiny_imagenet/` and copies the validation images into one folder per class
  (`val_restructured/`). The text runner downloads IMDb and AG News through Hugging Face `datasets`,
  20 Newsgroups through scikit-learn, and `bert-base-uncased` through `transformers`.
- **Speed.** The goodness-sum readout and the Stage-2 head score every label hypothesis for every
  image: 10 for CIFAR-10, 100 for CIFAR-100 and 200 for Tiny ImageNet. On CPU, a CIFAR-100 file
  took about 1 s per test image with 4 threads, so use a GPU for full test sets. The BP networks
  score the full CIFAR-10 test set on CPU in a few minutes. For the text files, encoding the test
  texts with BERT dominates the cost.
- **Size.** The CIFAR-10 CP-FAIR and NoMem-L2 files hold a 32-expert MoE network with 107 M
  parameters (19 M active per token). Each Stage-1 file takes 1.8 GB, and each Stage-2 file 2.1 GB.

## Retraining the Stage-2 probe

The original Stage-2 heads are in this repository (`stage2_best.pt`). To retrain the probe from a
Stage-1 file instead, use the trainer. When a Stage-1 checkpoint is complete, the trainer skips
Stage 1 and does the following:

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
(every FF group here except NoMem-L2 has one). Set `FF_SEED` to the run's seed, `FF_EPOCHS_STAGE1`
to the file's `epoch_stage1`, and `FF_VERSION_TAG` to the part of the file name between
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

- **Resumed runs.** 13 of the 29 FF runs were resumed after an interrupted session:
  γ=0 seeds 123/456, κ=0 seed 456, cumulative seeds 123/456 and MGC seed 456 on CIFAR-100; squared-hinge γ=0.7 seed 456 and CP-FAIR seeds 42/123/456 on CIFAR-10; and all three Tiny ImageNet runs. Their final evaluations are complete. The trainers that produced them did not
  save RNG state, so a resumed run is not a bitwise continuation of an uninterrupted one. The
  `resumed_run` field of `release_manifest.json` gives the run segments. The other FF runs,
  including the four seed-42 CIFAR-100 runs of the first release, ran without interruption, and
  so did the BP and text-domain runs.
- **Lineage.**
  - Most FF runs were trained on Colab A100 GPUs. The CIFAR-10 CP-FAIR and NoMem-L2 runs and
    MGC-D128 seed 42 were trained on an NVIDIA DGX Spark (GB10).
  - The CIFAR-100 trio Stage-1 files are the extracted copies of the runs' Google Drive archive
    members, with equal CRC-32 and size. The trio's Stage-2 files and all Tiny ImageNet files are
    the archive members themselves.
  - The squared-hinge seed-42 files come from the Colab runs. Two aborted DGX runs of that seed
    have no final evaluation and are not released.
  - The BP runs of seeds 123 and 456 (rebuttal) and the two stripped-backbone runs of seed 42
    (submitted paper) were trained on the DGX Spark. The device of the plain-CNN seed-42 run is not
    recorded.
  - The text-domain runs were trained on Colab GPUs. Each run directory was archived as a
    `.tar.gz` in a Google Drive export.
- **Trainers.**
  - The FF files were produced by earlier versions of the trainers: the v1 CIFAR-100 trainer
    (trio), the rebuttal-era MGC and squared-hinge trainers, and earlier versions of the CIFAR-10
    and Tiny ImageNet trainers.
  - The camera-ready trainers in the code repository build the same networks. Every FF file's
    `net_ema` (and every Stage-2 file's `head_state`) loads into its group's trainer with every key
    matched, and every key of its stored `cfg` is a field of that trainer's `FFConfig`. In the
    Stage-1 spot check above, the predictions match each run's saved ones, apart from 2 of the
    896 MoE predictions (see Precision). The code repository's CPU equivalence tests compare the
    training step with the v1 and rebuttal trainers.
  - The BP files come from the run scripts shipped in `trainers/bp_original/`. The text-domain files
    come from the runner in `trainers/text_original/`, which imports the camera-ready core instead of
    the pre-camera-ready copy the runs used (see that folder's README).
- **MGC.** The MGC arm replaces the baseline's depth-scaled current-block term: block 0 loses its
  auxiliary and has no compensator. The implementation is faithful but not an exact recovery
  (ε=10⁻⁶, positive-part clamp).
- **SAM.** The CIFAR-10 FF runs used SAM. The CIFAR-100, Tiny ImageNet and text-domain runs did not.

## Other checkpoints (on request)

`release_manifest.json` lists 133 checkpoint files of 71 runs. For each file it gives the run, the
main paper tables the file supports (not an exhaustive list), its trainer and configuration, its
source and lineage, its size and its SHA-256.

The 48 files marked `"availability": "on_request"` are the `stage1_best.pt` and `stage2_best.pt`
of the 24 CIFAR-10 γ=0, gated-ablation and hardness-gated κ=0 runs. They are available from the
author on request (amir.usefi75@gmail.com), and a recipient can check them against the SHA-256
values in the manifest. The following are not available at all:

- the published CIFAR-10 γ=0 L4/D256 seed-42 checkpoint, which is not recoverable;
- the checkpoints of the CIFAR-100 32-expert γ=0.7 baseline (kept, but not released).

The manifest does not cover every run of the paper. For example, it does not list the checkpoints
of the single-seed CIFAR-10 variants of Tables 1 and 27 other than NoMem-L2, the
component ablation (Table 26), arms (a)–(c) of the MGC control (Table 6), the CIFAR-10 γ=0 seed-42
rerun (Table 22), the three seeds of the CIFAR-100 32-expert γ=0.7 baseline (Table 38), or the FF
rows of Table 2 on the plain CNN and on the stripped backbone.

## Integrity

`SHA256SUMS` pins the 85 checkpoint files by their SHA-256 over the full file bytes, and
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
- **Datasets:** the checkpoints were trained on CIFAR-10, CIFAR-100, Tiny ImageNet, IMDb,
  20 Newsgroups and AG News, which are not redistributed here.
  - An FF image checkpoint stores network weights, optimizer and scheduler states and the
    contrastive feature queues, which hold projected features of training images and their labels.
    It contains no images.
  - A BP checkpoint stores network weights, optimizer and scheduler states and the configuration.
  - A text-domain checkpoint stores network weights and the configuration, and no text.

  Use of the datasets themselves follows their authors' terms.

## Datasets

- **CIFAR-10 and CIFAR-100** (Krizhevsky, 2009), loaded with torchvision. The 50,000 training
  images are split into 45,000 for training and 5,000 for validation. The split is seeded with the
  run seed, so runs with the same seed share it. The 10,000-image test set is used only for the
  final evaluation. The BP runs split the training images in the same proportions (45,000 and
  5,000), also seeded with the run seed.
- **Tiny ImageNet** (Le and Yang, 2015; 200 classes, 64×64 pixels), downloaded by the trainer. The
  100,000 training images are split into 90,000 for training and 10,000 for validation, seeded with
  the run seed. The labeled 10,000-image validation split serves as the test set and is used only
  for the final evaluation.
- **IMDb** (Maas et al., 2011), **20 Newsgroups** (Lang, 1995; headers, footers and quotes removed,
  as in scikit-learn) and **AG News** (Zhang et al., 2015). Each text run holds out a stratified 10%
  of the training split for validation, seeded with the run seed. The standard test split is used
  only for the final evaluation.

## Intended use and limitations

These checkpoints are for studying FF training. Possible uses:

- per-block separation and goodness;
- readouts and probes of the frozen backbones, including the original Stage-2 heads;
- prediction churn between arms and between seeds;
- the BP calibration of Table 2 and the text-domain check of App. K.

The paper reports the accuracies, protocols and seed-level variation: Tables 2, 3, 5, 7, 21, 25,
27 and 44, and Appendix B. With the Stage-2 files, the S2 and S2-TTA values of every FF run in this
repository can be recomputed from the original heads, for example the paired CIFAR-100 trio
statistics.

The image models are CIFAR- and Tiny ImageNet-scale classifiers (1.1 M to 107 M parameters), and
the text networks are 58 M-parameter FF networks that rely on a frozen, backprop-trained BERT
encoder. They were trained for this study and were not evaluated for robustness, fairness or
out-of-distribution use.

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
