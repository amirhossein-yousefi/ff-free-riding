# Component leave-one-out ablation (Appendix E of the main paper)

The component LOO ablation (`tab:component_ablation`) drops one
training/architecture flag at a time and re-runs Stage-1 + Stage-2
under otherwise identical config (single seed 42, L=4, D=256, 362+20
epochs). The exception is the γ=0 row: like the published γ=0 seed-42
run, it also sets `block_curr_lambda=0` and `depth_order_lambda=0`, so it
removes γ accumulation together with the block-curr and depth-order
losses. `FF_GAMMA_SCALE=0` alone keeps the trainer defaults
(`block_curr_lambda=0.25`, `depth_order_lambda=0.2`), which is the
seed-123/456 configuration of `tab:gamma0_multiseed`, not this row.

## Components covered

| Flag                | Effect on S2 TTA |
| ------------------- | ---------------: |
| no HNM              |        −0.45 pp  |
| 1-aspect goodness (prototype alignment only) | −0.41 pp |
| no SAM              |        +0.25 pp  |
| no SupCon           |        +0.25 pp  |
| no γ, block-curr, depth-order (γ=λ_curr=λ_depth=0) | +0.25 pp |
| no MoE              |        +0.17 pp  |
| no block-curr       |        +0.16 pp  |
| no depth-order      |         0.00 pp  |

## Trainer env-var interface

`trainers/cp_fair_cifar10.py` reads the following environment variables
at start-up and overrides the corresponding fields of its `FFConfig`
dataclass (see `main()` for the exact mapping):

| env var | overrides | typical use |
| ------- | --------- | ----------- |
| `FF_SEED`                 | `cfg.seed`                 | seed for the run                                |
| `FF_NUM_BLOCKS`           | `cfg.num_blocks`           | model depth (default 4)                         |
| `FF_D_MODEL`              | `cfg.d_model`              | hidden dim (default 256)                        |
| `FF_NUM_HEADS`            | `cfg.num_heads`            | attention heads                                 |
| `FF_USE_SAM`              | `cfg.use_sam`              | toggle SAM (set to `0` to disable)              |
| `FF_EPOCHS_STAGE1`        | `cfg.epochs_stage1`        | shorten S1 epochs                               |
| `FF_GAMMA_SCALE`          | `cfg.gamma_scale`          | cumulative-loss coefficient (γ in the paper)    |
| `FF_BLOCK_CURR_LAMBDA`    | `cfg.block_curr_lambda`    | depth-scaled current-block weight (λ_curr base) |
| `FF_DEPTH_ORDER_LAMBDA`   | `cfg.depth_order_lambda`   | depth-order auxiliary loss weight               |
| `FF_CONTRASTIVE_LAMBDA`   | `cfg.contrastive_lambda`   | SupCon weight                                   |
| `FF_K_FIRST` / `FF_K_LAST`| `cfg.hard_negative_k_*`    | HNM k schedule                                  |
| `FF_N_EXPERTS`            | `cfg.n_experts`            | MoE expert count (set to 1 to disable MoE)      |
| `FF_N_ASPECTS`            | `cfg.n_aspects`            | goodness aspect count (1 keeps only aspect 0, prototype alignment) |

For ablations whose flag does not have a one-line env-var override
(e.g. the per-aspect weights and the EMA switches), edit
the `FFConfig` dataclass of `cp_fair_cifar10.py` (just above `main()`; the per-aspect weights are
set in its `__post_init__`) directly.

## Reproducing each row

Each row is reproduced by combining the env-var overrides above with
the seed pin. The `(no HNM)` row, for example, is reproduced by
setting both HNM widths to 1 (effectively disabling hard-negative
selection) and re-running:

```bash
FF_K_FIRST=1 FF_K_LAST=1 FF_SEED=42 python trainers/cp_fair_cifar10.py
```

The other LOO rows correspond to:

| LOO row             | reproduce by                                                                 |
| ------------------- | ---------------------------------------------------------------------------- |
| no HNM              | `FF_K_FIRST=1 FF_K_LAST=1 FF_SEED=42 python trainers/cp_fair_cifar10.py`     |
| 1-aspect goodness   | `FF_N_ASPECTS=1 FF_SEED=42 python trainers/cp_fair_cifar10.py`               |
| no SAM              | `FF_USE_SAM=0 FF_SEED=42 python trainers/cp_fair_cifar10.py`                 |
| no SupCon           | `FF_CONTRASTIVE_LAMBDA=0 FF_SEED=42 python trainers/cp_fair_cifar10.py`      |
| no γ, block-curr, depth-order | `FF_GAMMA_SCALE=0 FF_BLOCK_CURR_LAMBDA=0 FF_DEPTH_ORDER_LAMBDA=0 FF_SEED=42 python trainers/cp_fair_cifar10.py` |
| no MoE              | `FF_N_EXPERTS=1 FF_SEED=42 python trainers/cp_fair_cifar10.py`               |
| no block-curr       | `FF_BLOCK_CURR_LAMBDA=0 FF_SEED=42 python trainers/cp_fair_cifar10.py`       |
| no depth-order      | `FF_DEPTH_ORDER_LAMBDA=0 FF_SEED=42 python trainers/cp_fair_cifar10.py`      |

Each run writes `preds_stage{1,2}_test_{tta,no_tta}.npz` to its
`runs/<run_id>/` dir. S2 TTA test accuracy is then
`(z['y_pred'] == z['y_true']).mean()` on `preds_stage2_test_tta.npz`.

## Aspect / EMA sweep (`tab:aspect_ema_ablation`)

The aspect-isolation rows keep all four aspect slots (`n_aspects=4`; three are
active, because the attention-sharpness slot is zero with memory disabled) and
zero the other aspects' loss weights. There is no environment hook for the
per-aspect weights: set `aspect_lambdas` in `FFConfig.__post_init__` of
`trainers/cp_fair_cifar10.py` (default `[1.5, 0.2, 0.3, 0.3]`):

| Row | `aspect_lambdas` | S2 TTA (seed 42) |
| --- | --- | ---: |
| Proto alignment only | `[1.5, 0.0, 0.0, 0.0]` | 90.97% |
| Energy (rank) only | `[0.0, 0.2, 0.0, 0.0]` | 91.51% |
| No aspect loss (attention-sharpness slot only) | `[0.0, 0.0, 0.3, 0.0]` | 91.26% |
| Learned head only | `[0.0, 0.0, 0.0, 0.3]` | 91.33% |

`FF_N_ASPECTS=1` is a different setting: it truncates the aspect list to aspect 0,
i.e. prototype alignment only (it is the "1-aspect goodness" row of the
leave-one-out table), not energy only. The two EMA rows edit the `FFConfig` dataclass
directly (no env-var hook): `use_ema = False` for "No EMA teacher (eval-EMA also
disabled)" (90.90% S2 TTA), and `use_ema_hnm = False`, with `use_ema` left `True`, for
"EMA eval retained; HNM uses online model" (91.48%).

## HNM k-sweep (`tab:hnm_ksweep`)

Sweep both ends of the k schedule together: `FF_K_FIRST=K
FF_K_LAST=K` for `K ∈ {4, 8, 16}` (K=1, random negatives, is the no-HNM row). `K=4` is the optimum at 91.43%
S2 TTA.

The NL-only and NI-only rows keep the k schedule (8→16) and change the NL/NI blend, which has
no env-var hook: set `sff_ni_weight` in the `FFConfig` dataclass (default `0.5`, the 50/50
blend; the aspect, block and current-block losses weight their wrong-label and wrong-image terms
by `1 - sff_ni_weight` and `sff_ni_weight`) to `0.0` for NL only (91.24% S2 TTA) or `1.0` for
NI only (90.37%), then run `FF_SEED=42 python trainers/cp_fair_cifar10.py`.
