# F2: registered predictions (written 2026-09-25, before the run was launched)

## What F2 fixes (X1 / D7)

The published CIFAR-10 gamma=0 n=3 mixes two configurations (`config_diffs/config_diffs.json`):

| run | block_curr_lambda | depth_order_lambda | SAM | everything else |
|---|---|---|---|---|
| seed 42 (published), `ablations/no_all_collab/runs/..._seed42_20260312_132043` | 0.0 | 0.0 | on | identical |
| seeds 123 / 456, `revised_training/out/gamma0_seed{123,456}/runs/*` | 0.25 | 0.2 | on | identical |

F2 reruns seed 42 in the seed-123/456 config with the same trainer
(`basic_fair_revised_valfix_cp_fair_fixed.py`). The fork only adds RNG save/restore; one training step
is bitwise identical to the original (`test_equivalence.py`). Launch env: `FF_GAMMA_SCALE=0.0 FF_SEED=42
FF_VERSION_TAG=cifar10_gamma0_seed42_matched FF_USE_SAM=1` (SAM is the default and is set explicitly).
The planned `config.json` differs from seeds 123/456 only in `seed`. It differs from the published seed 42
only in `block_curr_lambda` 0 -> 0.25 and `depth_order_lambda` 0 -> 0.2.

Provenance note: the published seed-42 run is two segments. Epochs 1-273 are in
`..._seed42_20260311_120026` (no run_end). Epochs 274-362 plus Stage 2 are in `..._seed42_20260312_132043`.
The original trainer re-seeds instead of restoring the RNG on resume. Its best_s1 (0.9198) was reached at
epoch 320, in the second segment.

## Reference values (verified from events.jsonl / layer_metrics.jsonl on 2026-09-25)

| metric (test top-1 x 100) | s42 published (bcl 0 / dol 0) | s123 | s456 |
|---|---|---|---|
| S1 (no TTA) | 90.52 | 90.54 | 90.44 |
| S1-TTA | 91.45 | 91.40 | 91.10 |
| S2 (no TTA) | 90.37 | 90.51 | 90.44 |
| S2-TTA | 91.39 | 91.51 | 91.22 |
| deepest-block sep (block 3 `sep_curr_nl`, S1 epoch 362) | 3.2756 | 7.8521 | 7.9733 |

Published n = 3 S1-TTA: 91.32 +/- 0.19 (sample SD). This headline is never silently replaced.

## Metric definitions (fixed now)

- **S1-TTA**: `final_test_eval.stage1_test_top1_tta` x 100 of the matched seed-42 run.
- **Deepest-block sep**: layer-3 `sep_curr_nl` in `layer_metrics.jsonl` at Stage-1 epoch 362. If the run
  was resumed, concatenate segments by (epoch, layer).

## Registered predictions

- **P1 (as specified): S1-TTA lies within the seed-123/456 range, [91.10, 91.40].** Caveat, written down now:
  if the three seeds are exchangeable, a third draw falls between the other two with probability 1/3. P1 is
  therefore a strict check and is reported as is.
- **P1b (tolerance band, the criterion the paper sentence relies on): S1-TTA lies within +/-0.30 pp of the
  seed-123/456 mean (91.25), i.e. in [90.95, 91.55].** The width is about 1.5 x the published n = 3 SD.
- **P2: the deepest-block sep is near 7.9, like seeds 123/456, rather than 3.28.** Band: [6.9, 8.9].
  Decision rule for the sentence: a value >= 5.6 (closer to 7.9 than to 3.28) counts as "like seeds
  123/456"; a value < 5.6 counts as "like the published seed 42".
- **Not registered.** S2 and S2-TTA are reported next to seeds 123/456 (S2-TTA 91.51 / 91.22), with no
  prediction.

## What each outcome means for the paper sentence (D7)

The D7 disclosure (per-seed config in `tab:gamma0_multiseed`, "identical hyperparameters" removed, no
"seed variance" for 3.28 -> 7.9) is written regardless of F2. F2 adds a matched-config n = 3 row
(new s42, s123, s456) **next to** the published n = 3.

- **P2 holds (sep >= 5.6) and P1b holds.** "Rerunning seed 42 in the seed-123/456 configuration gives S1-TTA
  X (seeds 123/456: 91.40, 91.10) and deepest-block separation Y (7.85, 7.97). The 3.28 of the published
  seed-42 run therefore reflects its configuration (block_curr_lambda 0, depth_order_lambda 0), not seed
  variance." Report the matched n = 3 next to the published 91.32 +/- 0.19.
- **P2 fails (sep < 5.6).** "Under the matched configuration seed 42 still separates less (Y), so the spread
  is not explained by the configuration difference alone." The config disclosure stays. The paper may then
  describe the remaining spread as seed-level variation, and must say it was checked.
- **P1b fails.** Report the matched-config S1-TTA and its n = 3 mean +/- SD next to the published one. The
  gamma=0 accuracy statements are given for both n = 3 sets. The headline is unchanged, and any sentence
  that relies on the published n = 3 SD is re-checked against the matched n = 3.
- **P1 fails but P1b holds.** Nothing changes in the text (see the P1 caveat). Both are reported.

## Known non-flag differences

- Hardware and software. Seeds 123/456 ran on a GB10 (torch 2.9.1+cu130). The published seed 42 ran on a
  Colab A100 (torch 2.10.0+cu128). `CR/.venv` has torch 2.14.0+cu130. Record `env.json`.
- Duration. About 110 h on a GB10 alone (seeds 123/456: 110.1 h and 110.2 h wall), or about 32 h on an
  A100 (the published seed-42 segments: 23.9 h + 8.0 h). A Colab run needs at least one resume. The fork
  restores the RNG; persistent DataLoader workers are re-seeded, so a resumed run is reproducible from its
  checkpoint but not bitwise equal to an uninterrupted one. Disclose every `resume` event.
