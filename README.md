# Cumulative-Goodness Free-Riding in Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant

**Amirhossein Yousefiramandi** (Independent Researcher) · NeurIPS 2026 (poster)

- **Paper:** [arXiv:2605.06240](https://arxiv.org/abs/2605.06240)
- **Code** (this repository): <https://github.com/amirhossein-yousefi/ff-free-riding> (Apache-2.0,
  full text in [`LICENSE`](LICENSE))
- **Checkpoints:** <https://huggingface.co/Amirhossein75/ff-free-riding> (CC BY 4.0). This is a
  minimal release: the seed-42 Stage-1 checkpoints of the four CIFAR-100 arms of Table 3. The
  other result-bearing checkpoints are listed with SHA-256 in
  [`RELEASE_CHECKPOINTS.md`](RELEASE_CHECKPOINTS.md) and are available on request.

This repository contains the training code, analysis scripts, metric summaries and reproduction
manifest for the paper. It is the camera-ready version of the anonymized supplement submitted
for review; [`CHANGELOG.md`](CHANGELOG.md) lists every difference from that version.

## Quickstart

```bash
git clone https://github.com/amirhossein-yousefi/ff-free-riding && cd ff-free-riding
pip install -r requirements.txt     # Python 3.12; install the torch/torchvision wheels for your CUDA

# CPU, about a second: paired-bootstrap CIs of the CIFAR-100 trio (S2-TTA column of Table 3)
# from the nine bundled prediction files in preds_cifar100/
FF_PAIRED_OUT=./out python analysis/paired_dissociation_stats.py

# GPU, about 16 h per run on an A100: train one arm of the CIFAR-100 trio from scratch
# (here gamma=0, seed 42; the other arms and seeds are in scripts/reproduce_cifar100_dissociation.sh)
FF_VERSION_TAG=cifar100_gamma0_seed42 FF_NUM_BLOCKS=4 FF_D_MODEL=256 FF_NUM_HEADS=8 FF_USE_SAM=0 \
  FF_N_EXPERTS=1 FF_MOE_TOP_K=1 FF_GAMMA_SCALE=0 FF_SEED=42 python trainers/cp_fair_cifar100.py
```

To evaluate a released checkpoint, see the loading snippet in the model card
([`release/MODEL_CARD.md`](release/MODEL_CARD.md), also the README of the Hugging Face
repository). It rebuilds the network with `trainers/cp_fair_cifar100.py` and scores the CIFAR-100
test set with the paper's Stage-1 goodness-sum readout. To get S2 numbers, retrain the Stage-2
probe from a released Stage-1 checkpoint as described in
[`RELEASE_CHECKPOINTS.md`](RELEASE_CHECKPOINTS.md#retraining-the-stage-2-probe).

## Citation

```bibtex
@inproceedings{yousefiramandi2026cumulative,
  title     = {Cumulative-Goodness Free-Riding in Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant},
  author    = {Yousefiramandi, Amirhossein},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026}
}
```

The same metadata is in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository").

## Coverage at a glance

| Result | Status |
| --- | --- |
| CIFAR-10 main FF runs (γ=0, CP-FAIR, ablations) | Trainer + env commands; metric summaries bundled |
| CIFAR-100 dissociation trio (γ=0 / κ=0 / cumulative, 3 seeds) | Trainer + env commands; the nine S2-TTA prediction files bundled in `preds_cifar100/` |
| Tiny ImageNet γ=0 (3 seeds) | Trainer + seed pins |
| L4/D128 and L8/D128 hardness-gated runs | Trainer + env commands (SAM setting per seed in the manifest) |
| Missing-gradient compensation (MGC), CIFAR-100 and CIFAR-10 D128 | Opt-in flags in the shipped trainers (`FF_USE_MGC=1`) |
| MGC control at CIFAR-10 L4/D128 (F1: SAM-matched cumulative arm, auxiliary removed, block-0 auxiliary kept; 3 seeds each; Table 6, App. B.1.1) | Trainer flags + env commands in the manifest; `analysis/camera_ready/f1_analysis.py` (paired statistics, scoring of the registered predictions) and `analysis/experiments/f1_mgc_block0/` (F_d snapshot, registered predictions); results `metric_summaries/camera_ready/f1_block0_control.json` |
| CIFAR-10 γ=0 seed 42 rerun in the seed-123/456 configuration (F2; Table 22) | Trainer + env command; `analysis/camera_ready/f2_analysis.py`, registered predictions in `analysis/experiments/f2_gamma0_s42/`; results `metric_summaries/camera_ready/f2_gamma0_s42.json` |
| Squared-hinge barrier replication | Opt-in flag (`FF_BARRIER_TYPE=sq_hinge`) |
| Frozen-backbone readouts (strict-FF block, CE block) | `analysis/readouts/` |
| BP/FF calibration rows of Table 2 (n=3) | Original run scripts in `trainers/bp_original/` |
| Gradient-attenuation snapshot (Fig. 1) and its EMA-weight recomputation | `analysis/snapshot_grad_attenuation.py`, `analysis/experiments/e3_fd/` |
| Separation vs. accuracy at the selected checkpoints (E1: logged, and recomputed on the checkpoints) | `analysis/experiments/e1_sep/` (CPU; E1 tier 2 and E2 run together with `analysis/experiments/run_cpu_batch.sh`) |
| Validation-selected early exit (E2; the cumulative runs exit early too, so the saving is not specific to the repairs) | `analysis/experiments/e2_exit/` (CPU); table and figure from `analysis/camera_ready/make_tables.py` / `make_figures.py --fig early_exit` |
| Paired-bootstrap dissociation statistics | `analysis/paired_dissociation_stats.py` on `preds_cifar100/` (S2-TTA; the other protocols need the full prediction sets, regenerated from the checkpoints, most of them available on request) |
| Block-locality audit, including the depth-order path | `scripts/reproduce_locality_audit.sh` |
| Text-domain runs (App. K) | Wrapper trainer + recipe (`analysis/text_domain_README.md`) |
| Trained checkpoints | Public on Hugging Face: the 4 seed-42 Stage-1 checkpoints of the CIFAR-100 γ=0 / κ=0 / cumulative / MGC arms (CC BY 4.0). All 98 result-bearing checkpoints are listed with SHA-256, and the other 94 are available on request: [`RELEASE_CHECKPOINTS.md`](RELEASE_CHECKPOINTS.md), model card [`release/MODEL_CARD.md`](release/MODEL_CARD.md) |

## Layout

```
trainers/
  cp_fair_cifar10.py           CIFAR-10 trainer (γ sweep, depth-scaled loss, MoE)
  cp_fair_cifar100.py          CIFAR-100 trainer (dense dissociation trio; MGC via FF_USE_MGC=1)
  cp_fair_tiny_imagenet.py     Tiny ImageNet trainer (γ=0)
  cp_fair_hardness_gated.py    CIFAR-10 hardness-gated trainer (L4/L8 D128; MGC, squared hinge, F1 flags)
  cp_fair_text_domain.py       text-domain wrapper (frozen BERT stem + FF blocks; App. K)
  cp_fair_core.py              shared FF block library (its own main() is an exploratory D384 config)
  bp_control.py                compact BP control re-implementation
  bp_original/                 the original BP/FF calibration scripts behind Table 2
tests/                         CPU equivalence tests of the camera-ready trainers (tests/README.md)
analysis/
  aggregate_multiseed.py, verify_locality.py, snapshot_grad_attenuation.py, plot_grad_attenuation.py,
  paired_dissociation_stats.py, generate_figures.py, scripts/plot_dynamics.py    (v1 scripts)
  camera_ready/                aggregation, tables, figures, compute budget and number tracing for the
                               camera-ready paper; F1/F2 analysis (f1_analysis.py, f2_analysis.py)
  readouts/                    S2-FF (strict-FF readout block) and S2-FF-CE readouts
  recovered/                   rebuttal-era scripts behind cited results (see analysis/README.md)
  experiments/                 camera-ready CPU experiments E1 (separation vs accuracy), E2 (validation-
                               selected early exit), E3 (F_d on EMA weights), E6 (L8 κ=0 seed-42 test
                               evaluation); run_cpu_batch.sh runs E1 tier 2 and E2 (CPU only);
                               f1_mgc_block0/ (F_d snapshot of the MGC control) and the registered
                               predictions of the F1/F2 GPU runs
  *_README.md                  recipes for the BP, component-ablation and text-domain rows
metric_summaries/              v1 summaries (aggregated_gated_l4d128.json corrected) + camera_ready/ results
                               (incl. compute_budget.json)
preds_cifar100/                the nine CIFAR-100 trio S2-TTA test-prediction files (30 MB; paired bootstrap; CC BY 4.0)
notebooks/                     Colab launchers (Drive sync + resume)
scripts/                       push-button reproduction wrappers
release/stage_release.py       stages and verifies the checkpoint release (--subset minimal: the 4 public files; the script itself uploads nothing)
release/MODEL_CARD.md          Hugging Face model card of the checkpoint release (license: cc-by-4.0)
repro_manifest.csv             paper row -> trainer -> env -> reference run -> expected value
CITATION.cff                   citation metadata (GitHub "Cite this repository")
release_manifest.json          checkpoint sources, lineage, sizes, SHA-256, availability (public / on request)
```

## Reproducibility manifest

[`repro_manifest.csv`](repro_manifest.csv) maps paper table rows to the trainer, seeds, SAM setting,
environment variables, the reference run directories and the expected values (mean ± sample SD).
`tests/check_manifest_commands.py` replays each command's `main()` up to the point where it writes
`config.json` and checks that the configuration equals the reference run's own `config.json`
(all 53 command/seed pairs pass, including the ten of the F1/F2 rows; only keys added by the
camera-ready trainers may differ; `tests/manifest_check.json`).

## Reproducing the headline results

All trainers read their configuration from environment variables, so one trainer file produces
every variant.

### CIFAR-10 γ=0 (Table 1; S1-TTA 91.32 ± 0.19 %)

```bash
pip install -r requirements.txt
# seed 42 as published: block_curr_lambda=0 and depth_order_lambda=0 (seeds 123/456 use the defaults)
FF_GAMMA_SCALE=0 FF_BLOCK_CURR_LAMBDA=0 FF_DEPTH_ORDER_LAMBDA=0 FF_SEED=42 python trainers/cp_fair_cifar10.py
for SEED in 123 456; do FF_GAMMA_SCALE=0 FF_SEED=$SEED python trainers/cp_fair_cifar10.py; done
python analysis/aggregate_multiseed.py --runs_dir runs --tag gamma0
# F2: seed 42 rerun in the seed-123/456 configuration (Table 22, next to the published n=3)
FF_VERSION_TAG=cifar10_gamma0_seed42_matched FF_GAMMA_SCALE=0 FF_USE_SAM=1 FF_SEED=42 python trainers/cp_fair_cifar10.py
```

The published n=3 pools two configurations (seed 42 used `block_curr_lambda=0`,
`depth_order_lambda=0`; erratum D7) and remains the headline. The F2 rerun of seed 42 in the
seed-123/456 configuration reaches S1-TTA 91.39% and a last-epoch deepest-block separation of 7.72
(published seed 42: 3.28); the matched n=3 is 91.30 ± 0.17% S1-TTA, next to the published
91.32 ± 0.19%. Its registered predictions (`analysis/experiments/f2_gamma0_s42/PREDICTIONS.md`) all
pass; `analysis/camera_ready/f2_analysis.py` writes `f2_gamma0_s42.json` from the raw run files.

### CIFAR-100 dissociation trio (3 variants × 3 seeds)

`scripts/reproduce_cifar100_dissociation.sh`, or `notebooks/cifar100_multiseed_colab.ipynb` on Colab.

### Missing-gradient compensation (MGC)

```bash
# CIFAR-100 L4/D256, dense, no SAM (Colab runs of the paper)
FF_USE_MGC=1 FF_MGC_C0=1.0 FF_N_EXPERTS=1 FF_MOE_TOP_K=1 FF_USE_SAM=0 FF_NUM_HEADS=8 \
  FF_GAMMA_GATING_MODE=constant FF_GAMMA_SCALE=0.7 FF_VERSION_TAG=mgc_c100 FF_SEED=42 \
  python trainers/cp_fair_cifar100.py
# CIFAR-10 L4/D128, SAM on
FF_USE_MGC=1 FF_MGC_C0=1.0 FF_MGC_RHO=0.0 FF_GAMMA_GATING_MODE=constant FF_GAMMA_SCALE=0.7 \
  FF_VERSION_TAG=mgc_c1 FF_SEED=42 python trainers/cp_fair_hardness_gated.py
```

MGC replaces the depth-scaled current-block auxiliary with the compensated local term
λ_i·softplus(−β m_i), λ_i = [c_d − s(M_i)/(s(m_i)+ε)]₊ (stop-gradient); it is not added on top of
the auxiliary. With the default c₀ = 1, block 0 loses its 0.25 auxiliary and receives no
compensation.

### MGC control at CIFAR-10 L4/D128 (F1; Table 6, App. B.1.1)

Three arms next to the MGC-D128 runs, all with constant γ=0.7, SAM on, 180+10 epochs, seeds
42/123/456:

```bash
COMMON="FF_GAMMA_GATING_MODE=constant FF_GAMMA_SCALE=0.7 FF_MGC_C0=1.0 FF_MGC_RHO=0.0 FF_USE_SAM=1"
for s in 42 123 456; do
  env $COMMON FF_USE_MGC=0 FF_VERSION_TAG=f1a_cum_sam FF_SEED=$s \
    python trainers/cp_fair_hardness_gated.py                          # (a) SAM-matched cumulative arm
  env $COMMON FF_USE_MGC=0 FF_CURR_AUX_OFF=1 FF_VERSION_TAG=f1b_cum_noaux FF_SEED=$s \
    python trainers/cp_fair_hardness_gated.py                          # (b) auxiliary removed, no compensator
  env $COMMON FF_USE_MGC=1 FF_MGC_KEEP_L0_AUX=1 FF_VERSION_TAG=f1c_mgc_keepl0 FF_SEED=$s \
    python trainers/cp_fair_hardness_gated.py                          # (c) MGC with block 0's auxiliary kept
done
```

Against arm (a), MGC raises the deepest-block separation from 0.75 ± 0.01 to 5.25 ± 0.05. Arm (b),
without the residual-weighted auxiliary that MGC replaces and without a compensator, reaches
4.88 ± 0.05, 92% of that gain, so the registered prediction that it would stay below 4.72 failed;
arm (c) leaves the separation unchanged (5.28 ± 0.03). Every paired 95% CI of the accuracy
differences lies inside ±1 pp. The registered predictions are in
`analysis/experiments/f1_mgc_block0/PREDICTIONS.md` (written before the runs; P2 failed, P1/P3/P4
passed). `analysis/camera_ready/f1_analysis.py` computes `f1_block0_control.json` (per-run values,
paired bootstraps, the separation decomposition, the prediction scores, and the F_d snapshot of
`analysis/experiments/f1_mgc_block0/fd_snapshot_f1.py`), and `analysis/camera_ready/make_tables.py`
generates Table 6 from it.

### Squared-hinge barrier

```bash
for G in 0.7 0.0; do
  FF_BARRIER_TYPE=sq_hinge FF_BARRIER_MARGIN=2.0 FF_GAMMA_GATING_MODE=constant FF_GAMMA_SCALE=$G \
    FF_SEED=42 python trainers/cp_fair_hardness_gated.py
done
```

### Hardness-gated κ=0 (L4/D128 and L8/D128)

`run_cifar10_multiseed_dgx.sh --tier C` (L4) and `--tier D` (L8). The seed-42 L4 run and all L8
runs used SAM off (`FF_USE_SAM=0`); the L4 seeds 123/456 used SAM on. The corrected L4 aggregate is
`metric_summaries/aggregated_gated_l4d128.json`.

### Gradient-attenuation figure (Fig. 1)

Point the snapshot script at the three L4/D128 seed-42 checkpoints (`baseline_constant`,
`no_all_collab`, `lcff_prefix`; available on request, see `RELEASE_CHECKPOINTS.md`) and their run
directories:

```bash
FF_CHECKPOINTS_DIR=<checkpoints> FF_RUNS_DIR=<runs> FF_DATA_DIR=./data_c10 \
  python analysis/snapshot_grad_attenuation.py
python analysis/plot_grad_attenuation.py
```

The camera-ready recomputation on EMA weights with the correct normalization is in
`analysis/experiments/e3_fd/` (results: `metric_summaries/camera_ready/e3_fd_ema.json`).

### Locality audit (CPU, a few minutes)

```bash
FF_V1_TRAINERS=/path/to/v1/trainers bash scripts/reproduce_locality_audit.sh
```

This checks both the cumulative-goodness path and the depth-order path on the real Stage-1 loop.
In the v1 trainers the depth-order loss of block l evaluated block l−1's aggregation weights with
gradient enabled; that gradient was discarded before block l−1's next update, so no published
parameter was affected. The camera-ready trainers evaluate it under `torch.no_grad()`, and the
audit confirms bit-identical parameters with and without the change.

## Environment-variable interface

All variables are optional; unset means the trainer's default configuration (for the new flags:
the v1 behavior, bit for bit; see `tests/`).

| Variable | Effect | Trainers |
|---|---|---|
| `FF_SEED` | random seed | all |
| `FF_GAMMA_SCALE` | constant γ of the cumulative goodness (LCFF 1.0, CP-FAIR 0.7, block-local 0) | all image trainers |
| `FF_GAMMA_GATING_MODE` | `constant` or `adaptive` (hardness-gated); the gated trainer defaults to `adaptive` | cifar100, gated |
| `FF_GAMMA_0`, `FF_GAMMA_TAU`, `FF_GAMMA_KAPPA` | adaptive-γ base, sharpness and threshold | cifar100, gated |
| `FF_BLOCK_CURR_LAMBDA`, `FF_DEPTH_ORDER_LAMBDA` | current-block auxiliary and depth-order weights | image trainers |
| `FF_USE_SAM` | SAM on/off (defaults differ per trainer; see the manifest) | image trainers |
| `FF_N_EXPERTS`, `FF_MOE_TOP_K` | MoE experts and routing top-k (1/1 = dense MLP; `FF_MOE_TOP_K` is read by cifar100 only) | cifar10, cifar100 |
| `FF_NUM_BLOCKS`, `FF_D_MODEL`, `FF_NUM_HEADS` | depth, width, heads (`FF_D_MODEL`/`FF_NUM_HEADS` not read by the gated trainer) | image trainers |
| `FF_EPOCHS_STAGE1`, `FF_EPOCHS_STAGE2` | epoch overrides (`FF_EPOCHS_STAGE2`: cifar100 and gated only) | image trainers |
| `FF_K_FIRST`, `FF_K_LAST`, `FF_N_ASPECTS`, `FF_CONTRASTIVE_LAMBDA` | HNM k schedule, aspect count, SupCon weight | cifar10 |
| `FF_BATCH_SIZE` | batch size | tiny |
| `FF_USE_MGC`, `FF_MGC_C0`, `FF_MGC_RHO` | missing-gradient compensation (off by default) | cifar100, gated |
| `FF_BARRIER_TYPE`, `FF_BARRIER_MARGIN` | block-level barrier `softplus` (default) or `sq_hinge`, and its margin | gated |
| `FF_MGC_KEEP_L0_AUX`, `FF_CURR_AUX_OFF` | F1 MGC-control arms: keep block 0's auxiliary under MGC (arm c); remove the current-block auxiliary, without MGC (arm b). Off by default | gated |
| `FF_VERSION_TAG`, `FF_RUNS_DIR`, `FF_CKPT_DIR` | run name tag, run-log and checkpoint directories | all image trainers |

The full list for each trainer is at the top of its `main()`. Checkpoints now also store the RNG
state (Python, NumPy, torch CPU/CUDA), which is restored on resume; v1 checkpoints still load.

## Hardware

The reported runs were trained on NVIDIA A100-SXM4-40GB GPUs (Google Colab) and on NVIDIA DGX
Spark (GB10) machines; every run directory's `env.json` records the device and software versions. Reduce
`batch_size` or `n_experts` if you run out of memory. The tests, the locality audit and the
analysis scripts run on CPU.

## Colab notebooks

The notebooks in `notebooks/` sync checkpoints and logs to a Google Drive folder (`MyDrive/ff_*`);
change `DRIVE_BASE` in the second cell to use another location. Re-running a notebook resumes from
whatever is on Drive.

## License

- **Code** (this repository): Apache-2.0, see [`LICENSE`](LICENSE) (full license text) and
  [`NOTICE`](NOTICE) (copyright notice).
- **Checkpoints** (public: <https://huggingface.co/Amirhossein75/ff-free-riding>; the others on
  request) and the prediction files in `preds_cifar100/`: Creative Commons Attribution 4.0
  International (CC BY 4.0); see [`RELEASE_CHECKPOINTS.md`](RELEASE_CHECKPOINTS.md) and the model
  card [`release/MODEL_CARD.md`](release/MODEL_CARD.md).
- The datasets (CIFAR-10, CIFAR-100, Tiny ImageNet, and the text datasets of App. K) are not
  redistributed; they are downloaded from their original sources under their own terms.
