# Changelog

## Camera-ready corrections (2026-10-25)

- **Text domain (App. K):** the original runner that produced the nine text-domain runs is now
  bundled in `trainers/text_original/` (`cp_fair_text_benchmark.py` and its launcher
  `run_cp_fair_text_multirun.py`). Its README gives the runs' exact command and the changes
  from the run copies: the import of the released `trainers/cp_fair_core.py` and three reworded
  docstring lines. The earlier wrapper
  `trainers/cp_fair_text_domain.py` does not run (it calls functions that
  `trainers/cp_fair_cifar10.py` does not define); it is kept unchanged because
  `tests/equivalence_results.json` records its SHA-256. `analysis/text_domain_README.md`, the
  README and the `tab:text_domain_full` row of `repro_manifest.csv` now point to the original
  runner, and the text README no longer calls every published target an FF method (the AG News
  target is Karimi et al.'s layer-wise triplet-loss method).
- **Augmentation of the Table 2 calibration rows:** both plain-CNN rows (BP and FF) used random
  crop and flip only, not strong augmentation, and the BP-strong row uses the FF trainers'
  per-view recipe (one view; the FF runs draw three). `analysis/bp_controls_README.md`,
  `analysis/camera_ready/make_tables.py` and `make_figures.py`, and
  `metric_summaries/camera_ready/fig3_4_bpff_values.json` now say so; no number changes. The
  unreported parameter-matched "Plain CNN big" BP run also used crop and flip only.
  `trainers/bp_control.py`'s `aug_strong` docstring ("matched to the FF training pipeline") is
  wrong; the file is unchanged (its SHA-256 is recorded), and the README says how it differs.
- **Optimizer wording:** the generated `t_gamma0_multiseed` caption now says adaptive SAM: every
  SAM-on run used the trainers' `sam_adaptive=True` mode, whose ascent step is scaled element-wise
  by |w| (paper App. D.3).
- **Compute budget:** `analysis/camera_ready/compute_budget.py` now labels the three seed-42 BP/FF
  control runs timed in the paper's training-cost appendix as DGX Spark (GB10) runs, not Colab A100
  runs: the BP-strong run is bitwise identical to its GB10 rerun (all 362 epochs' training loss
  and `preds_test.npz`), and the run names follow local time. The measured totals in
  `metric_summaries/camera_ready/compute_budget.json` become 844.6 A100-hours and 2,202.2
  GB10-hours (were 822.0 and 1,911.4): the relabeling moves about 20 hours from the A100 to the
  GB10 total, and the seed-42 run of the CIFAR-100 32-expert baseline (42.4 A100-hours) and the
  FIXED-NL and BLOCK0-LABEL runs (270.8 GB10-hours), listed as untimed before, are now timed. The
  HNM k-sweep and aspect/EMA deep-dive runs and the incomplete GAMMA-RAMP run stay untimed;
  `not_measured_reported_runs` now gives their estimated cost. The JSON's method note now says
  how runs without an `env.json` get their device (a note, or the median epoch time for four
  L8/D128 arms).
- **Component ablation, γ=0 row:** that leave-one-out run also set `block_curr_lambda=0` and
  `depth_order_lambda=0` (the configuration of the published γ=0 seed-42 run), so it removes
  γ accumulation and two loss terms, not γ alone. `analysis/component_ablation_README.md` and the
  `tab:component_ablation` row of `repro_manifest.csv` now say so and give its full setting; the
  README gave `FF_GAMMA_SCALE=0` alone ("γ=0 only") and the manifest said each row drops one
  flag. `FF_GAMMA_SCALE=0` alone keeps the default block-curr and depth-order weights (the
  seed-123/456 configuration of `tab:gamma0_multiseed`). The paper's Sec. 4.5 and
  Tab. 26 are corrected the same way.
- **Docs:** `analysis/component_ablation_README.md` now gives the paper's leave-one-out effects
  (it printed placeholder ± values), the components without an environment-variable hook, and the
  k values of the HNM sweep (4, 8, 16; k=1 is the no-HNM row). The `tab:main_results` Multi-Tech row
  of `repro_manifest.csv` names its original trainer (`atlas_ff.py`, not bundled) instead of a
  configuration edit that `cp_fair_cifar10.py` cannot make. The CIFAR-100 32-expert baseline's
  checkpoints (all three seeds) are kept but not released; the manifest, `RELEASE_CHECKPOINTS.md`,
  the model card and the `tab:cifar100_multiseed` row said its seed-42 run was not retained.
  `analysis/bp_controls_README.md` no longer says the compact trainer reproduces the four
  parameter-matched runs (two of them are FF runs), and it now gives the recipe of the reported
  rows next to that of `bp_control.py`. The text-domain and BP-control checkpoints are kept but not
  released (the docs said the paper reports only their run logs and metrics). The Multi-Tech row
  of `repro_manifest.csv` now cites the original run's `final_test_eval` and its SAM setting.
  `analysis/component_ablation_README.md` names the trainer's `FFConfig` dataclass (it said
  `Config`). The README and `requirements.txt` no longer say that every run directory has an
  `env.json`, and `requirements.txt` now also lists PyTorch 2.14.0+cu130 (the five GB10 runs of
  the MGC control) and Python 3.13 (its four Colab runs and the γ=0 seed-42 rerun). The README
  now lists the rows that need a configuration edit and the nine Table 27 variants (and the
  stripped-backbone FF row of Table 2) that have no bundled trainer;
  `analysis/component_ablation_README.md` gives the settings of the NL-only and NI-only rows
  (`sff_ni_weight`) and of both EMA rows (`use_ema`, `use_ema_hnm`); the Greedy-Freeze row of
  `repro_manifest.csv` gives its SAM setting.

## Expanded checkpoint release (2026-10-24)

The Hugging Face repository now holds the Stage-1 checkpoints (`stage1_best.pt`) of 29 runs
instead of 4:

- **Added:** seeds 123/456 of the CIFAR-100 γ=0, κ=0, cumulative and MGC arms; all seeds of the
  CIFAR-10 squared-hinge (both arms), MGC-D128 and CP-FAIR runs; the Tiny ImageNet γ=0 runs
  (seeds 42/123/456); and the CIFAR-10 NoMem-L2 run (seed 42). The four seed-42 CIFAR-100 files of
  the first release are unchanged.
- **`release_manifest.json`:**
  - `availability` is "public_hf" for the 29 files.
  - Eight new entries: the Stage-1 and Stage-2 files of the three Tiny ImageNet runs and of the
    NoMem-L2 run. The four Stage-1 files are public; the four Stage-2 heads are available on
    request. The manifest now lists 106 checkpoint files of 53 runs.
  - A `public` subset. The `minimal` subset stays as the record of the first release.
  - Resumed-run disclosures for the three Tiny ImageNet runs and for two gated-ablation runs that
    were missing them (`gated_l4_adaptive_prev_only_seed42`, `gated_l8_adaptive_t1_k4_seed42`).
  - For every public file, the checks made before the upload: `lineage_check` and
    `loading_spot_check`. RELEASE_CHECKPOINTS.md and the model card describe them.
  - More paper tables per group, for example Tables 13 and 36, which re-score checkpoint files
    (Table 13: the nine CIFAR-100 trio files and the nine CIFAR-10 L4/D128 gated-ablation files;
    Table 36: 18 files of the trio, CP-FAIR, CIFAR-10 γ=0 and CIFAR-10 L4/D128 κ=0 runs), and
    `_meta.not_covered`, which names runs of the paper whose checkpoints the manifest does not
    list. The earlier documents called the listed files every result-bearing checkpoint
    of the paper, which overstated their coverage.
- **`release/stage_release.py`:** `--subset public` stages the Hugging Face release, and `--subset
  minimal` now stages the 4 files of the first release. A staged file that fails its SHA-256 check
  is now deleted from `--dest`.
- **Model card, `RELEASE_CHECKPOINTS.md`, README and `preds_cifar100/README.md`:** updated for the
  29 public files. The model card gives the trainer and test-set normalization of each checkpoint
  group, the logged numbers of every public file, and the run history of the resumed runs. The
  documented commands use `hf`, because current `huggingface_hub` releases no longer run
  `huggingface-cli`.
- **`repro_manifest.csv`:** the notes of the two `tab:readouts` rows and of the trio
  paired-bootstrap row of `tab:dissociation_merged` (their Stage-1 checkpoints are all public now),
  and of the Tiny ImageNet row (the runs' Stage-2 head used `head_hidden=1536`; the trainer's
  default is 2048).

- **`analysis/experiments/ffcr_common.py`:** two comments only. The gated-ablation
  `adaptive_prev_only` seed-42 run was resumed (Stage-1 epochs 1-174, then 175-180), not
  restarted; its selected epoch (180) is in the final segment, so no result changes.

The paper and the trainers are unchanged.

## Update after the first public release: F1, F2, E1 tier 2 and E2

The paper now reports two camera-ready GPU experiments that were still running when this repository
was first published, and two CPU analyses. This update adds their results and scripts:

- **F1, the MGC control at CIFAR-10 L4/D128** (Table 6, App. B.1.1): arms (a) SAM-matched cumulative,
  (b) cumulative without the current-block auxiliary and without a compensator, and (c) MGC with
  block 0's auxiliary kept, 3 seeds each.
  - Results: `metric_summaries/camera_ready/f1_block0_control.json` (per-run values, paired
    bootstraps, the separation decomposition, the scored predictions and the F_d snapshot).
  - Scripts: `analysis/camera_ready/f1_analysis.py` and
    `analysis/experiments/f1_mgc_block0/fd_snapshot_f1.py` (CPU).
  - Registered predictions: `analysis/experiments/f1_mgc_block0/PREDICTIONS.md`, byte-identical to
    the file written before the first F1 run. P2 (arm (b) stays below 4.72) failed; P1, P3 and P4
    passed.
- **F2, the CIFAR-10 γ=0 seed-42 rerun in the seed-123/456 configuration** (Table 22):
  `metric_summaries/camera_ready/f2_gamma0_s42.json`, `analysis/camera_ready/f2_analysis.py`, and the
  registered predictions `analysis/experiments/f2_gamma0_s42/PREDICTIONS.md` (all pass). The
  published n=3 stays the headline, and the matched n=3 is reported next to it.
- `repro_manifest.csv`: the three planned F1/F2 rows are replaced by four completed rows (F1 arms
  (a), (b), (c) and the F2 run), with reference runs, expected values and the GPU of each seed. All
  53 command/seed pairs reproduce their reference runs' `config.json` (`tests/manifest_check.json`;
  43 before this update).
- `analysis/camera_ready/make_tables.py` also generates Table 6 and Table 22. All 13 generated
  tables of the paper regenerate byte-identically from `metric_summaries/camera_ready/`.
- `analysis/camera_ready/compute_budget.py` and `metric_summaries/camera_ready/compute_budget.json`
  include the F1 and F2 runs (the camera-ready GPU additions now total about 177 A100-hours and
  108 GB10-hours) and the CPU time of the F1 F_d snapshot. `analysis/camera_ready/config.py` names
  the F1/F2 experiment folders (`FF_F1_DIR`, `FF_F2_DIR`).
- E1 tier 2 and E2 (see Additions below) were also added after the first public release.
- README: F1/F2 commands and results; the hardware section now says "NVIDIA DGX Spark (GB10)
  machines", as the paper does.

The trainers are unchanged. The F1 and F2 runs were trained with forks of the rebuttal MGC trainer
and of the v1 CIFAR-10 trainer (available on request). The camera-ready trainers reproduce their
training step bitwise under the F1 flags and for the F2 configuration (`tests/`), and the manifest
commands reproduce every run's `config.json`.

## Changes from the submitted (v1) supplement

New or changed code is marked `[CR]` in the trainers. With every new flag off, one Stage-1
training step of each camera-ready trainer is bitwise identical to the v1 trainer (loss terms,
parameters, EMA, optimizer state, contrastive queues, RNG state); see `tests/`.

### Trainers

- **Missing-gradient compensation (MGC)**, previously the separate rebuttal files
  `mgc_trainer.py` (CIFAR-10 D128) and `mgc_c100_trainer.py` (CIFAR-100), is folded into
  `cp_fair_hardness_gated.py` and `cp_fair_cifar100.py` behind `FF_USE_MGC=1` (`FF_MGC_C0`,
  `FF_MGC_RHO`). With the flag on, two training steps are bitwise identical to the rebuttal files.
  The rebuttal comment that called MGC an "additive missing-gradient compensation" is corrected:
  MGC replaces the depth-scaled current-block auxiliary at every block (block 0 included).
  The CIFAR-100 file now also carries the c₀ > 1 block-0 branch of the CIFAR-10 file; it is
  inactive for the configuration that was run (c₀ = 1). MGC telemetry (`loss_mgc`,
  `mgc_lam_mean`, `mgc_lam_p95`) is logged only when MGC is on.
- **Squared-hinge barrier** (rebuttal file `hinge_trainer.py`) folded into
  `cp_fair_hardness_gated.py`: `FF_BARRIER_TYPE=sq_hinge`, `FF_BARRIER_MARGIN` (default
  `softplus`, the v1 criterion). MGC together with the hinge barrier is refused.
- **F1 block-0 controls** in `cp_fair_hardness_gated.py`: `FF_MGC_KEEP_L0_AUX=1` (under MGC,
  block 0 keeps its 0.25 auxiliary) and `FF_CURR_AUX_OFF=1` (cumulative objective without the
  current-block auxiliary). Mislabeling combinations are refused.
- **RNG state in checkpoints** (all five image trainers): Python, NumPy, torch CPU and CUDA RNG
  states are saved in every checkpoint and restored on resume (Stage-1 checkpoints at load time,
  Stage-2 checkpoints after the head is rebuilt); resumes are logged in `events.jsonl`. v1
  checkpoints still load (RNG left as seeded, as in v1).
- **Depth-order locality (P3)**: the previous-block `aggregate_goodness` calls of the depth-order
  loss now run under `torch.no_grad()` in all five image trainers. In v1, block l's backward wrote
  a gradient into `blocks[l-1].alpha_raw.grad` that was discarded before use; parameters after
  training are bit-identical with and without the change (`analysis/verify_locality.py --depth-order`).
- `FF_EPOCHS_STAGE1` / `FF_EPOCHS_STAGE2` env hooks added to the gated trainer and
  `FF_EPOCHS_STAGE2` to the CIFAR-100 trainer (unset = v1 defaults).
- `config.json` of a new run contains the new configuration fields with their default values.
- `trainers/bp_original/`: the original BP/FF calibration scripts behind the n=3 Table 2 rows.

### Corrections

- `metric_summaries/aggregated_gated_l4d128.json` (errata D1, D13): the v1 file stored 85.87 (a
  Stage-1 value) as seed 42's S2-TTA; the genuine value is 87.27, and the n=3 S2-TTA is
  87.15 ± 0.17 (sample SD; v1: 86.69 ± 0.72). SAM differs across seeds (seed 42 off, 123/456 on);
  the file now records it per seed, with all four protocols and the unrounded separations.
- `repro_manifest.csv`: rebuilt with `use_sam`, `env`, `reference_runs` and `notes` columns; every
  row's command is checked against its reference run's `config.json`
  (`tests/check_manifest_commands.py`). Row-level fixes: the Tiny ImageNet row was malformed CSV;
  the text-domain seeds are 42/43/44 and those runs used SAM off; the "energy-only" row is
  replaced by the four aspect-isolation rows (per-aspect weight overrides, not `FF_N_ASPECTS=1`,
  which keeps prototype alignment only); the gated L4/D128 seed-42 rows carry their SAM-off and
  constant-gating settings and genuine S2 values; the CIFAR-10 γ=0 seed-42 row carries
  `block_curr_lambda=0`, `depth_order_lambda=0` (erratum D7); the L8 κ=0 row uses seed 42's
  post hoc test evaluation (87.12 ± 0.35); CP-FAIR S1 no-TTA SD is 0.18; new rows for MGC
  (CIFAR-100, CIFAR-10 D128), squared hinge, BP n=3, the S2-FF / S2-FF-CE readouts, and the
  F1/F2 runs (see the update section above).
- `run_cifar10_multiseed_dgx.sh`, `scripts/reproduce_*.sh`: commands match the runs as trained
  (SAM off for the seed-42 L4 and all L8 κ=0 runs; the seed-42 γ=0 configuration); unverified
  wall-clock estimates removed; the BP script uses the original run scripts at n=3.
- `analysis/component_ablation_README.md`, `analysis/bp_controls_README.md`,
  `analysis/text_domain_README.md`: aspect-isolation recipe, 5.54 M parameters for the stripped
  BP backbone, n=3 BP values, text seeds 42/43/44 and SAM off; camera-ready appendix letters
  (E, K); the two parameter-matched BP controls are marked as not reported in the paper.
- `requirements.txt`: the camera-ready tested environment (adds scipy, scikit-learn; see the file
  for why the v1 pins changed).

### Additions

- `tests/`: CPU equivalence tests for every modified trainer, and the manifest command check.
- `analysis/camera_ready/`, `analysis/readouts/`, `analysis/recovered/`, `analysis/experiments/`:
  the scripts behind the camera-ready numbers, with every path taken from environment variables
  (`analysis/README.md`).
- `metric_summaries/camera_ready/`: the camera-ready result JSONs (paths made relative to
  `FF_HOME`; every numeric value unchanged).
- `RELEASE_CHECKPOINTS.md`, `release_manifest.json`, `release/stage_release.py`: the checkpoint
  release with SHA-256 per file.
  - The public Hugging Face release (<https://huggingface.co/Amirhossein75/ff-free-riding>) is
    minimal: 4 files, the seed-42 `stage1_best.pt` of the CIFAR-100 γ=0, κ=0, cumulative and MGC
    arms of Table 3.
  - The other 94 result-bearing files are listed with SHA-256 and are available on request
    (`availability` in the manifest).
  - `stage_release.py --subset minimal` stages the public files, and `--subset full` stages
    everything (with `SHA256SUMS` also covering the prediction files).
  - `release/MODEL_CARD.md` is the model card of the Hugging Face repository.
  - `RELEASE_CHECKPOINTS.md` explains how to retrain the Stage-2 probe from a public Stage-1
    checkpoint.
- `analysis/camera_ready/compute_budget.py` and `metric_summaries/camera_ready/compute_budget.json`:
  the compute budget of the reported runs from the raw run logs (checklist, compute item).
- E1 tier 2 (separation recomputed on the selected checkpoints) and E2 (validation-selected early exit):
  `analysis/experiments/e2_exit/`, `analysis/experiments/run_cpu_batch.sh` (the CPU batch that ran both;
  the E1 scripts were already in `analysis/experiments/e1_sep/`), `analysis/camera_ready/early_exit.py`
  and the E1/E2 table and figure code in `make_tables.py` / `make_figures.py`, the results
  `metric_summaries/camera_ready/{e1_sep_tier2,e2_exit,fig_early_exit_pareto_values}.json`, two
  `repro_manifest.csv` rows, and their CPU time in `compute_budget.json` (the batch took 25.0
  wall-clock hours, CPU only).
- `CITATION.cff`: citation metadata for GitHub's "Cite this repository" (the same title, author and
  venue as the BibTeX entry in `README.md`).
- Licenses: the code stays Apache-2.0 (`LICENSE`, now the full license text; copyright notice in
  `NOTICE`); the released checkpoints and prediction files are CC BY 4.0 (`RELEASE_CHECKPOINTS.md`,
  `release/MODEL_CARD.md`, `preds_cifar100/README.md`).

### Unchanged data

- `preds_cifar100/`: the nine CIFAR-100 trio Stage-2 TTA prediction files (30 MB) are bundled
  unchanged (byte-identical to the v1 files and to the result-bearing run directories); they are
  published with the code and are not duplicated in the minimal checkpoint release.
  Only their README changed: it points to the camera-ready tables (Table 3, App. B.7) and states
  their license.

### Removed

- The anonymization notes; author, license holder and citation added.
