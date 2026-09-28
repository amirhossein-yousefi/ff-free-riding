# F1: registered predictions (written 2026-09-25, before any F1 run was launched)

## What F1 controls for

Under MGC the trainer zeroes the depth-scaled current-block auxiliary at every block
(`mgc_trainer.py:2468-2469`), while the compensator exists only for blocks d > 0
(`:2328-2329`; at d = 0 it is [c_0 - 1]_+ = 0 for c_0 = 1). So block 0 loses its 0.25
auxiliary with nothing in its place. Its objective drops from 1.25 x SYMBA to 1.0 x SYMBA.
MGC therefore changes two things at once: the compensator at blocks 1-3 and the block-0
objective. The CPU test (`test_equivalence.py`, F2) confirms that after one step, block 0's
parameters under MGC differ from those under cumulative.

Arms (L4/D128 CIFAR-10, softplus, constant gamma = 0.7, **SAM on**, 180+10 epochs,
seeds 42/123/456; env in `arms.json`):

| arm | env on top of the MGC-D128 recipe | block-0 objective | blocks 1-3 |
|---|---|---|---|
| MGC-D128 (existing, n=3) | `FF_USE_MGC=1` | SYMBA only (aux removed) | cumulative + compensator, no aux |
| (b) `f1b_cum_noaux` | `FF_USE_MGC=0 FF_CURR_AUX_OFF=1` | SYMBA only (= MGC, bitwise after one step) | cumulative, no aux, no compensator |
| (c) `f1c_mgc_keepl0` | `FF_USE_MGC=1 FF_MGC_KEEP_L0_AUX=1` | SYMBA + 0.25 aux (= cumulative, bitwise after one step) | = MGC |
| (a) `f1a_cum_sam`, optional | `FF_USE_MGC=0` | SYMBA + 0.25 aux | cumulative + aux [0.5, 0.75, 1.0] |

- (c) vs MGC isolates the block-0 auxiliary.
- (b) vs MGC isolates the compensator, with block 0 identical.
- (a) is the SAM-matched cumulative anchor. The published D128 cumulative run
  (`ff_cifar10_baseline_constant_L4_D128_..._seed42_20260313_232043`) has `use_sam: false`.
  MGC-D128 has `use_sam: true`, although the header comment of `run_mgc_seeds.sh` says
  "SAM off". So the existing MGC-vs-cumulative D128 comparison also differs in SAM
  (`config_diffs/config_diffs.json`).

Planned configs differ from the seed-matched MGC-D128 `config.json` only in `use_mgc` (b, a) and
the two new keys `mgc_keep_l0_aux` / `curr_aux_off` (all arms). See `config_diffs/`.

## Reference values (verified from raw logs on 2026-09-25; sample SD, ddof = 1)

MGC-D128. Seed 42 is `mgc_rebuttal/runs/..._seed42_20260723_223516` (GB10). Seeds 123 and 456
are `mgc_rebuttal/from_colab/runs_from_colab/..._seed{123,456}_*` (Colab A100). Values are
`final_test_eval` and `layer_metrics.jsonl`, S1 epoch 180:

| metric | s42 | s123 | s456 | mean +/- SD |
|---|---|---|---|---|
| S1 (no TTA) | 85.23 | 85.88 | 85.25 | 85.45 +/- 0.37 |
| S1-TTA | 86.64 | 87.09 | 86.89 | 86.87 +/- 0.23 |
| S2 (no TTA) | 86.00 | 86.48 | 85.73 | 86.07 +/- 0.38 |
| S2-TTA | 87.05 | 87.60 | 87.17 | 87.27 +/- 0.29 |
| deepest-block sep (block 3 `sep_curr_nl`, last S1 epoch) | 5.2393 | 5.3062 | 5.2072 | 5.25 +/- 0.05 |

SAM-off cumulative gamma = 0.7 D128, seed 42 (context only; a different SAM setting): deepest-block sep 0.7421,
S2-TTA 87.42.

## Metric definitions (fixed now)

- **Deepest-block sep**: `sep_curr_nl` of layer 3 in `layer_metrics.jsonl` at the last Stage-1 epoch (180).
  If a run was resumed, concatenate segments by (epoch, layer). This is the metric behind the 5.25 above.
- **Accuracy**: `final_test_eval` top-1 x 100 (S1 / S1-TTA from stage1_best, S2 / S2-TTA from stage2_best).
- Arm summaries are 3-seed means with sample SD. Arms are compared per seed, arm minus MGC-D128 of the same seed.
- Predictions are scored only at n = 3. At n = 2 (cut-line, Oct 18) values are reported descriptively
  and labelled "not scored (n = 2)".

## Registered predictions

- **P1: (c) ~ MGC in deepest-block sep.** The (c) 3-seed mean lies within +/-10% of 5.25, i.e. in [4.72, 5.78].
- **P2: (b) < MGC in deepest-block sep.** The (b) 3-seed mean is < 4.72 (below the P1 band), and every (b) seed
  is below the lowest MGC seed (5.21). Unscored expectation: (b) lands near the cumulative regime (order of
  1 or below), because nothing local pushes the deep blocks.
- **P3: accuracy.** For every arm that is run, the 3-seed mean S2-TTA lies within +/-1 pp of 87.27, i.e. in [86.27, 88.27].
- **P4 (secondary, only if (a) is run):** the (a) deepest-block sep mean is < 4.72.
- **No registered direction** for S1 / S1-TTA. The block-0 change could carry part of MGC's Stage-1 cost. We
  report the per-seed (c) - MGC and (b) - MGC deltas for all four protocols, plus a paired bootstrap
  (5000 resamples, as for C4) if the 10k-prediction npz files exist for every run.

## What each outcome means for the paper sentence

The C3 text already discloses the confound: "the MGC arm replaces the baseline's depth-scaled current-block
term; block 0 therefore loses its 0.25 auxiliary; the compensator is zero at block 0". F1 adds one sentence
to the MGC paragraph and one appendix table (arms x 4 protocols + deepest-block sep, n per arm).

- **P1, P2 and P3 hold.** Sentence: "Keeping block 0's auxiliary under MGC leaves the deepest-block separation
  unchanged ((c) X +/- s vs 5.25 +/- 0.05, n = 3). Removing the current-block auxiliary from the cumulative
  objective without the compensator does not raise it ((b) Y +/- s). The separation gain therefore comes from
  the compensator, not from the block-0 change. S2-TTA stays within 1 pp of MGC in every arm." The confound
  is disclosed and resolved.
- **P1 fails** ((c) outside [4.72, 5.78]). Part of MGC's deep-block separation change is due to dropping block
  0's auxiliary. The appendix reports (c) as the clean MGC arm. Any D128 separation-ratio statement is
  recomputed with (c) or qualified. The C100 MGC result keeps the disclosed confound (F1 is D128 only).
- **P2 fails** ((b) >= 4.72). Removing the current-block auxiliary alone reproduces most of MGC's separation.
  The MGC paragraph must then stop attributing the separation rise to missing-gradient compensation, and the
  "approximately compensates" reading of App. A.1 is not supported at D128. This is the outcome that changes a
  claim, and it is reported as such.
- **P3 fails for an arm.** Report that arm's per-seed values and the delta. The sub-1 pp wording must not cover
  that arm. The MGC Stage-1-cost framing stays and is not extended to other rule changes.
- **P4 fails** ((a) >= 4.72 with SAM on). SAM, not MGC, would be driving part of the D128 separation contrast.
  The D128 MGC-vs-cumulative comparison must then use (a) instead of the SAM-off published run.

## Known non-flag differences (disclosed with the results)

- Hardware and software. The MGC-D128 references are seed 42 on GB10 (torch 2.9.1+cu130) and seeds 123/456 on Colab
  A100 (torch 2.11.0+cu128). `CR/.venv` has torch 2.14.0+cu130. GPU kernels are not bitwise deterministic
  across stacks, so arm-vs-reference differences include run-to-run noise. The best mirror of the reference
  lineage is seed 42 on the GB10 and seeds 123/456 on a Colab A100. Record `env.json` either way.
- Resumes. The fork restores the python, numpy, torch-CPU and CUDA RNG on resume. Persistent DataLoader
  workers are re-seeded from the restored RNG, so a resumed run is reproducible from its checkpoint but not
  bitwise equal to an uninterrupted one. Disclose every `resume` event.
