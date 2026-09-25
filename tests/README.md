# Equivalence tests for the camera-ready trainers (CPU only)

Each modified trainer has a test file. A test builds the model, EMA copy, per-block SAM(AdamW)
optimizers, schedulers and contrastive queues exactly as the trainer's `main()` does, seeds every
RNG, and runs real Stage-1 training steps through the trainer's own `train_epoch_stage1` (one fixed
batch through every block: SAM two-pass update or plain AdamW step, EMA update, queue update). It
then compares, bitwise, everything the steps touched: logged loss terms and metrics, parameters,
EMA parameters, optimizer state, queues and the RNG state after the steps.

| Test file | Checks |
|---|---|
| `test_equiv_cp_fair_hardness_gated.py` | flags off == v1 (4 configurations); three-step depth-order locality == v1; `FF_USE_MGC=1` == rebuttal `mgc_trainer.py` (also with c₀=1.5, ρ=0.5); `FF_BARRIER_TYPE=sq_hinge` == rebuttal `hinge_trainer.py` (γ=0.7 and γ=0); F1 flags == the F1 fork; implied current-block weight per block for each arm; planned `config.json` == the real MGC-D128, hinge and κ=0 runs; refused flag combinations; RNG save/restore |
| `test_equiv_cp_fair_cifar100.py` | flags off == v1 (trio γ=0, κ=0, cumulative; 32-expert MoE default); three-step locality == v1; `FF_USE_MGC=1` == rebuttal `mgc_c100_trainer.py`; implied weights; planned `config.json` == the real MGC-C100 and trio runs; RNG save/restore |
| `test_equiv_cp_fair_cifar10.py` | flags off == v1 (CP-FAIR, γ=0, SAM off); three-step locality == v1; step == the F2 fork; planned `config.json` == the γ=0 seed-123 run; RNG save/restore |
| `test_equiv_cp_fair_tiny_imagenet.py` | flags off == v1 (reported defaults; depth ordering on with SAM on/off); three-step locality == v1; RNG save/restore |
| `test_equiv_cp_fair_core.py` | same on a reduced configuration (the module backs no reported number) |
| `check_manifest_commands.py` | every `repro_manifest.csv` command reproduces its reference run's `config.json` |

`equiv_utils.py` holds the shared helpers. The RNG tests check that capturing the state draws no
random numbers, that a checkpoint loads with `torch.load(weights_only=True)`, that a Stage-1 resume
continues bitwise like an uninterrupted run (while the v1 loader does not), that a Stage-2 resume
restores the RNG after the head is rebuilt, and that v1 checkpoints still load.

## Running

```bash
CUDA_VISIBLE_DEVICES='' \
FF_V1_TRAINERS=/path/to/v1/supplement_code/trainers \
FF_REBUTTAL_TRAINERS=/path/to/rebuttal/trainers \
FF_F1_TRAINER=/path/to/f1/trainer.py FF_F2_TRAINER=/path/to/f2/trainer.py \
FF_REFERENCE_RUNS=/path/to/FF_HOME/amir-porjects/cifar_10_fair/making_v1_v4_fair_comparison \
python tests/run_all.py --out tests/equivalence_results.json

CUDA_VISIBLE_DEVICES='' FF_REFERENCE_HOME=/path/to/FF_HOME python tests/check_manifest_commands.py
```

Every reference file is identified by SHA-256 (listed in `equiv_utils.py`); a test whose reference
is not supplied is reported as skipped, never as passed. The v1 trainers are the `trainers/` folder
of the submitted supplement; the rebuttal and F1/F2 trainers are available on request. The
recorded results of the author's run are in `tests/equivalence_results.json` (its
`_meta.post_run_edits` lists the comment-only trainer edits made after the run, each checked to
compile to identical bytecode) and `tests/manifest_check.json`; the depth-order locality audit
is in `tests/locality_depth_order.json`.
