# Original text-domain runner (Appendix K, `tab:text_domain_full`)

These two scripts produced the nine text-domain runs of the paper (IMDb, 20 Newsgroups and AG News;
seeds 42, 43 and 44; run on Colab). They are the run copies, with two kinds of change:

- `cp_fair_text_benchmark.py` imported `cp_fair_core_from_user.py`, the pre-camera-ready copy of
  [`../cp_fair_core.py`](../cp_fair_core.py). This copy imports `../cp_fair_core.py` instead
  (the commented import block near the top). The released core adds only no-grad guards on the
  history and depth-order paths, which leave the applied updates unchanged
  (`analysis/verify_locality.py`), plus RNG save/restore and resume logging.
- Three docstring and comment lines of `cp_fair_text_benchmark.py` are reworded. No other line
  differs, and `run_cp_fair_text_multirun.py` is unchanged.

SHA-256 of the run copies: `cp_fair_text_benchmark.py` 0fbdb92576396894e934d78359795176d8cda407e29a8c0e3c737b6f75ba810a,
`run_cp_fair_text_multirun.py` 49f6fa9b1ce7b0b68c10713843b71e2aa24500840b3ee9b3a1faa06549c2efdf,
`cp_fair_core_from_user.py` f47c69d2fc7f5f1b17aa0c67f82d9404e88fa3b20d8d75789b80b9b34cad3fca.

## Commands

The archived summary of the nine runs records one command, identical apart from the dataset and the
seed (Colab paths shortened):

```bash
pip install transformers datasets scikit-learn
for DS in imdb 20newsgroups ag_news; do
  for SEED in 42 43 44; do
    python trainers/text_original/cp_fair_text_benchmark.py --dataset $DS \
      --encoder_model bert-base-uncased --encoder_pool mean --encode_batch_size 64 \
      --d_model 256 --num_blocks 4 --num_heads 8 --n_experts 16 --proj_dim 256 \
      --num_virtual_tokens 8 --seed $SEED --workers 2 --device cuda --val_fraction 0.1 \
      --lr 0.001 --stage2_lr 0.001 --weight_decay 0.05 --gamma_scale 0.0 \
      --depth_order_lambda 0.0 --run_root ./runs_text --cache_dir ./cache_text
  done
done
```

`run_cp_fair_text_multirun.py` is the launcher that issued these commands and collected the
summaries (`--datasets imdb,20newsgroups,ag_news --seeds 42,43,44 --drive_mode none` runs the
same grid without Google Drive). The BERT weights and the datasets are downloaded on first use:
IMDb and AG News through Hugging Face `datasets`, 20 Newsgroups through scikit-learn. The runs
used no SAM (`--use_sam` is off by default). Gradients are clipped at norm 1.0 in Stage 1 (per
block) and in Stage 2 (`TextFFConfig.grad_clip`).

The networks (`stage1_best.pt`) and Stage-2 heads (`stage2_best.pt`) of the nine runs are public in
the checkpoint release (group `text_domain`); the model card (`release/MODEL_CARD.md`) gives the
loading code, which builds the network with this runner's classes.

`trainers/cp_fair_text_domain.py`, the earlier wrapper, does not run: it calls functions that
`trainers/cp_fair_cifar10.py` does not define. It is kept unchanged because
`tests/equivalence_results.json` records its SHA-256; use the scripts in this folder instead.
