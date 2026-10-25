# Cross-modality / text-domain check (Appendix K of the main paper)

The text-domain results in `tab:text_domain_full` use a frozen
`bert-base-uncased` encoder (BP-trained, public checkpoint) followed
by an FF hybrid head (L=4 D=256). We treat this as a generalization
sanity check, not as evidence about FF-from-scratch on text; the main
mechanistic claims rely on the vision experiments.

## Trainer

The reported runs were produced by the original text runner, now bundled in
[`trainers/text_original/`](../trainers/text_original/) (`cp_fair_text_benchmark.py` and its
launcher `run_cp_fair_text_multirun.py`; its README gives the exact command of the nine runs and
the changes from the run copies: the import of the released `trainers/cp_fair_core.py` and three
reworded docstring lines). The
runner embeds each text with the frozen `bert-base-uncased` encoder (masked mean of the last
hidden states, 768-d, L2-normalized), maps it through LayerNorm and a two-layer GELU MLP to 8
virtual tokens of width 256, and trains the L=4, D=256 FF blocks (16 experts,
γ=0) and the Stage-2 attentive readout from the core module. The BERT weights are downloaded by
`transformers` on first run, so no model assets ship in this repository.

`trainers/cp_fair_text_domain.py`, the wrapper listed here before, does not run (it calls
functions that `trainers/cp_fair_cifar10.py` does not define); it is kept unchanged because
`tests/equivalence_results.json` records its SHA-256.

## Reproduction recipe

See [`trainers/text_original/README.md`](../trainers/text_original/README.md). The reported runs
used seeds 42/43/44 (the v1 supplement listed 42/123/456 by mistake). All nine archived run
configs record `use_sam: false`, so Stage 1 of the text runs did not use SAM; the runs used
learning rate 1e-3 and gradient clipping at norm 1.0 in both stages, and weight decay 0.05 in Stage 1
(0 for the Stage-2 head).

## Expected numbers

Across 9 runs (3 datasets × 3 seeds: 42, 43, 44) the Stage-2 means exceed
the published target cited for each dataset in `tab:text_domain_full`
(Gandhi et al. 2023 for IMDb, Yang et al. 2024 for 20 Newsgroups, Karimi et
al. 2024 for AG News) by +1.02 to +2.39 pp; the across-seed sample SD is at
most 0.24 pp. Caveat: the BERT encoder is BP-trained, and the AG News target
(Karimi et al. 2024) is a layer-wise triplet-loss method that does not use
the FF goodness function, so this is not a like-for-like strict-FF comparison.

## Where the numbers live

`tab:text_domain_full` in `appendix/H_text_domain.tex` lists per-row
mean ± SD and the target each row exceeds. The machine-readable per-run
metrics and `dataset_aggregates.json` are inside the text-run archive
(metrics available on request; the archive's checkpoints are not released); the aggregate values are IMDb 87.25 ± 0.15,
20 Newsgroups 62.66 ± 0.24 and AG News 93.36 ± 0.07 (Stage-2 test top-1, %,
mean ± sample SD).
