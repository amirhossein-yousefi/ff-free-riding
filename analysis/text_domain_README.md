# Cross-modality / text-domain check (Appendix K of the main paper)

The text-domain results in `tab:text_domain_full` use a frozen
`bert-base-uncased` encoder (BP-trained, public checkpoint) followed
by an FF hybrid head (L=4 D=256). We treat this as a generalization
sanity check, not as evidence about FF-from-scratch on text; the main
mechanistic claims rely on the vision experiments.

## Bundled wrapper trainer

`trainers/cp_fair_text_domain.py` is a thin wrapper around
`trainers/cp_fair_cifar10.py`: it swaps the convolutional stem for a
frozen `bert-base-uncased` encoder with a 768→256 linear projection,
and reuses the FF hybrid block, hard-negative mining, locality
contract, and Stage-2 attentive readout from the canonical core /
CIFAR-10 modules. The BERT weights are downloaded by `transformers`
on first run, so no model assets ship in this archive.

## Reproduction recipe

```bash
pip install transformers datasets scikit-learn
for DS in imdb agnews 20news; do
  for SEED in 42 43 44; do
    FF_TEXT_DATASET=$DS FF_SEED=$SEED \
      python trainers/cp_fair_text_domain.py
  done
done
```

The reported runs used seeds 42/43/44 (the v1 supplement listed 42/123/456 by
mistake) and were produced by the original multi-run text runner
(`run_cp_fair_text_multirun.py` + `cp_fair_text_benchmark.py`; not bundled,
available on request). All nine archived run configs record `use_sam: false`,
so Stage 1 of the text runs did not use SAM.

## Expected numbers

Across 9 runs (3 datasets × 3 seeds: 42, 43, 44) the Stage-2 means exceed
the strongest published FF target on each dataset by +1.02 to +2.39 pp;
the across-seed sample SD is at most 0.24 pp. Caveat: the BERT encoder is
BP-trained, and one of the published targets is itself a non-strict-FF
method, so this is not a like-for-like strict-FF comparison.

## Where the numbers live

`tab:text_domain_full` in `appendix/H_text_domain.tex` lists per-row
mean ± SD and the target each row exceeds. The machine-readable per-run
metrics and `dataset_aggregates.json` are inside the text-run archive
(available on request); the aggregate values are IMDb 87.25 ± 0.15,
20 Newsgroups 62.66 ± 0.24 and AG News 93.36 ± 0.07 (Stage-2 test top-1, %,
mean ± sample SD).
