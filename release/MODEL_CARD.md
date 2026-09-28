---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - forward-forward
  - local-learning
  - cifar-100
  - layer-free-riding
  - neurips-2026
datasets:
  - uoft-cs/cifar100
metrics:
  - accuracy
---

# Forward-Forward free-riding checkpoints (NeurIPS 2026)

This repository holds four trained checkpoints from *Cumulative-Goodness Free-Riding in
Forward-Forward Networks: Real, Repairable, but Not Accuracy-Dominant* (Amirhossein
Yousefiramandi, NeurIPS 2026). Each file is the validation-selected Stage-1 network of the
seed-42 run of one of the four central CIFAR-100 arms in the paper's Table 3:

- block-local training (γ=0);
- hardness-gated collaboration (κ=0);
- cumulative-goodness training (γ=0.7);
- missing-gradient compensation (MGC, c_d=1).

This is a minimal release. The paper's other 94 result-bearing checkpoints are listed with their
SHA-256 hashes in `release_manifest.json` and are available from the author on request (see
[Other checkpoints](#other-checkpoints-on-request)). They cover all seeds, the Stage-2 heads and
the CIFAR-10 runs.

The checkpoints are research artifacts. They let you re-evaluate and probe the four Table 3 arms
without retraining Stage 1. They are not general-purpose image classifiers.

- **Paper:** [arXiv:2605.06240](https://arxiv.org/abs/2605.06240)
- **Code** (trainers, analysis, reproduction manifest; Apache-2.0):
  <https://github.com/amirhossein-yousefi/ff-free-riding>
- **Checkpoints** (this repository; CC BY 4.0): <https://huggingface.co/Amirhossein75/ff-free-riding>

## Files

| File | Arm | Selected Stage-1 epoch (val. top-1) | Bytes | SHA-256 |
|---|---|---|---:|---|
| `checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed42_L4_D256_P2_seed42_stage1_best.pt` | γ=0 (block-local) | 362 (65.72 %) | 211467565 | `d2e2a0548597355d633c7c3acad1ef01c274323f279ca1055e77a1c807e2848c` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_gated_k0_seed42_L4_D256_P2_seed42_stage1_best.pt` | hardness-gated κ=0 | 340 (66.22 %) | 211484697 | `58e0c770f80a4f70d9d6c814884ecbb1caf1162a270caf3294282b67d69fd4ca` |
| `checkpoints/c100_trio/ff_cifar100_cifar100_cumulative_seed42_L4_D256_P2_seed42_stage1_best.pt` | cumulative γ=0.7 | 340 (66.22 %) | 211487237 | `537cf9e2e27e73009596a9de9f52097ee98d33ad1055783a8f434e20ab44cc3b` |
| `checkpoints/mgc_c100/ff_cifar100_mgc_c100_L4_D256_P2_seed42_stage1_best.pt` | MGC (c_d=1) | 362 (66.96 %) | 211449657 | `6776a87e9afa3dcf6abc993b0e17646724b3b9f185ada825eebd54977c2c1a66` |

The four files total 845,889,156 bytes (846 MB). All four arms share one setup: CIFAR-100,
L4/D256 dense (no MoE), no SAM, 362 Stage-1 epochs, seed 42, and the same seed-42 train/validation
split. The repository also contains:

```
README.md                this model card
SHA256SUMS               sha256sum-compatible list of the four checkpoint files
release_manifest.json    all 98 result-bearing checkpoints of the paper: run, paper tables, trainer,
                         source, lineage, size, SHA-256; `availability` is "public_hf" for the four
                         files here and "on_request" for the others
```

Each file stores the following keys:

- `net_ema`: the EMA weights, which give the S1 numbers;
- `model`: the raw weights;
- `cfg`: the run's full configuration, identical to the run's `config.json` except `device`;
- `epoch_stage1`: the selected epoch;
- `best_s1`: the validation top-1 at that epoch;
- `stage`: 1;
- `opt_states`, `sched_states` and `queues`: the optimizer states, the scheduler states and the
  contrastive feature queues.

## Numbers behind each file

| Arm | S1 | S1-TTA | S2 (logged) | S2-TTA (logged) |
|---|---:|---:|---:|---:|
| γ=0 | 66.17 | 66.92 | 68.27 | 68.94 |
| κ=0 (gated) | 66.07 | 66.85 | 68.29 | 68.64 |
| Cumulative γ=0.7 | 66.70 | 67.18 | 68.09 | 68.57 |
| MGC | 66.05 | 66.79 | 68.71 | 69.19 |

The values are CIFAR-100 test top-1 (%) of the seed-42 runs. They enter the 3-seed means of
Table 3 (`tab:dissociation_merged`). The seed-42 columns of the frozen-backbone readout table
(`tab:readouts`, appendix) are computed on these four backbones.

- **S1 and S1-TTA** are the Stage-1 readouts of these files. S1 is the goodness-sum readout of the
  EMA weights on a single crop; S1-TTA adds the horizontally flipped image. S1 was re-predicted
  from each file on the full test set and matched the run logs to within 0.01 pp: 66.18, 66.07,
  66.70 and 66.06 against logged S1 values of 66.17, 66.07, 66.70 and 66.05. S1-TTA was not
  re-predicted; it is tied to these files by lineage (each run's `events.jsonl` names the file in
  its final test evaluation; see [Run history and disclosures](#run-history-and-disclosures)).
- **S2 and S2-TTA** come from the Stage-2 attentive head. This head is a backprop-trained probe on
  the frozen backbone, not part of the FF method, and its checkpoints (`stage2_best.pt`) are not in
  this release. To get S2 numbers, retrain the probe as described in
  [Retraining the Stage-2 probe](#retraining-the-stage-2-probe). The original heads are available
  on request.
- **Deepest-block separation** (the "sep" column of Table 3) is a training-log quantity: it was
  read at the last Stage-1 epoch (362), not recomputed from the validation-selected checkpoint.
  The code repository ships the per-seed values in `metric_summaries/camera_ready/`.

## Loading

The network is defined by the CIFAR-100 trainer in the code repository, which works for all four
files. MGC changes only the training loss, not the network.

```bash
git clone https://github.com/amirhossein-yousefi/ff-free-riding && cd ff-free-riding
pip install -r requirements.txt huggingface_hub
```

```python
import importlib.util, sys, torch
from huggingface_hub import hf_hub_download
from torchvision import datasets, transforms

path = hf_hub_download("Amirhossein75/ff-free-riding",
    "checkpoints/c100_trio/ff_cifar100_cifar100_gamma0_seed42_L4_D256_P2_seed42_stage1_best.pt")

spec = importlib.util.spec_from_file_location("ff_c100", "trainers/cp_fair_cifar100.py")
T = importlib.util.module_from_spec(spec); sys.modules["ff_c100"] = T; spec.loader.exec_module(T)

ckpt = torch.load(path, map_location="cpu", weights_only=True)
cfg = T.FFConfig()
for k, v in ckpt["cfg"].items():            # the run's configuration is stored in the checkpoint
    if hasattr(cfg, k):
        setattr(cfg, k, v)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
net = T.FFHybridNet(cfg).to(cfg.device)
net.load_state_dict(ckpt["net_ema"])        # EMA weights: the S1 / S1-TTA numbers
net.eval()

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
test = datasets.CIFAR100("./data_c100", train=False, download=True, transform=tf)
loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)
acc = T.evaluate_ff(net, loader, cfg.device, tta=False, topk=(1,))   # tta=True gives S1-TTA
print(f"S1 top-1: {100 * acc[1]:.2f} %")    # logged: 66.17 for this file
```

- **`weights_only`.** The files are PyTorch pickles. With PyTorch 2.14.0, all four load with
  `weights_only=True`, because they contain only tensors and plain Python containers. If your
  PyTorch version refuses a file, `torch.load(..., weights_only=False)` will load it, but that
  setting executes arbitrary pickle code. Use it only after `sha256sum -c SHA256SUMS` passes. The
  trainer's own `torch.load` calls, used by the Stage-2 retraining below, do not pass the argument,
  so they fall back to PyTorch's default.
- **Side effects.** Importing the trainer creates an empty `./checkpoints/` directory, and
  `download=True` fetches CIFAR-100 into `./data_c100/`.
- **Speed.** The goodness-sum readout scores all 100 label hypotheses for every image. On CPU it
  took about 1 s per test image with 4 threads, so use a GPU for the full test set.

## Retraining the Stage-2 probe

When a Stage-1 checkpoint is complete, the trainer skips Stage 1 and does the following:

1. It trains only the Stage-2 head: 20 epochs, AdamW, on the frozen EMA backbone, with the same
   seed-42 train/validation split, which the trainer regenerates from the seed.
2. It runs the final test evaluation: S1, S1-TTA, S2 and S2-TTA, written to `events.jsonl` in a
   new run directory.

This needs a GPU.

```bash
# one arm per invocation; EP must equal the checkpoint's epoch_stage1
ARM=gamma0;     EP=362; ENV="FF_GAMMA_SCALE=0"
# ARM=gated_k0; EP=340; ENV="FF_GAMMA_GATING_MODE=adaptive FF_GAMMA_0=0.7 FF_GAMMA_KAPPA=0.0 FF_GAMMA_TAU=1.0"
# ARM=cumulative; EP=340; ENV="FF_GAMMA_SCALE=0.7"
F=ff_cifar100_cifar100_${ARM}_seed42_L4_D256_P2_seed42_stage1_best.pt
huggingface-cli download Amirhossein75/ff-free-riding "checkpoints/c100_trio/$F" --local-dir ./hf
mkdir -p ckpt_s2 && cp "hf/checkpoints/c100_trio/$F" ckpt_s2/ && cp "ckpt_s2/$F" "ckpt_s2/${F/_best/_last}"
env $ENV FF_CKPT_DIR=./ckpt_s2 FF_VERSION_TAG=cifar100_${ARM}_seed42 FF_EPOCHS_STAGE1=$EP \
  FF_NUM_BLOCKS=4 FF_D_MODEL=256 FF_NUM_HEADS=8 FF_USE_SAM=0 FF_N_EXPERTS=1 FF_MOE_TOP_K=1 \
  FF_SEED=42 python trainers/cp_fair_cifar100.py

# MGC (EP=362)
F=ff_cifar100_mgc_c100_L4_D256_P2_seed42_stage1_best.pt
huggingface-cli download Amirhossein75/ff-free-riding "checkpoints/mgc_c100/$F" --local-dir ./hf
mkdir -p ckpt_s2 && cp "hf/checkpoints/mgc_c100/$F" ckpt_s2/ && cp "ckpt_s2/$F" "ckpt_s2/${F/_best/_last}"
FF_CKPT_DIR=./ckpt_s2 FF_VERSION_TAG=mgc_c100 FF_USE_MGC=1 FF_MGC_C0=1.0 FF_N_EXPERTS=1 FF_MOE_TOP_K=1 \
  FF_USE_SAM=0 FF_NUM_HEADS=8 FF_GAMMA_GATING_MODE=constant FF_GAMMA_SCALE=0.7 FF_SEED=42 \
  python trainers/cp_fair_cifar100.py
```

The trainer loads two files from `FF_CKPT_DIR`:

- It resumes Stage 1 from `<run>_stage1_last.pt`.
- It starts Stage 2 and the final Stage-1 evaluation from `<run>_stage1_best.pt`.

Both copies are therefore needed.

If `FF_EPOCHS_STAGE1` is larger than the checkpoint's `epoch_stage1` (340 for κ=0 and cumulative),
the trainer resumes Stage-1 training instead of skipping it. The per-arm environment matches the
arm's row in the code repository's `repro_manifest.csv`, with `FF_EPOCHS_STAGE1` set to the selected
epoch.

A retrained head is not bitwise identical to the paper's. The trainer that produced these runs did
not store RNG state, so the Stage-2 data order differs. Expect S2 values close to the logged ones,
not equal to them.

## Run history and disclosures

- **Not resumed.** None of the four released runs was resumed. Each ran from Stage-1 epoch 1 to
  epoch 362 in one run directory, and its layer log starts at epoch 1. The resumed runs reported in
  the paper are all among the on-request files: γ=0 seeds 123/456, κ=0 seed 456, cumulative seeds
  123/456 and MGC seed 456 on CIFAR-100, plus some CIFAR-10 runs. The trainers that produced them
  did not save RNG state, so a resumed run is not a bitwise continuation of an uninterrupted one.
  `RELEASE_CHECKPOINTS.md` in the code repository lists them.
- **Lineage.**
  - All four runs were trained on Colab A100 GPUs.
  - The three trio files are the extracted copies of the runs' Google Drive archive members, with
    equal CRC-32 and size.
  - Each file is the `stage1_best` that the result-bearing run's `events.jsonl` names in its final
    test evaluation.
  - Each file's `best_s1` equals the run's logged best validation accuracy, and its stored `cfg`
    equals the run's `config.json`.
- **Trainers.**
  - The trio was trained with the v1 CIFAR-100 trainer, and the MGC run with the rebuttal-era MGC
    trainer.
  - The camera-ready `trainers/cp_fair_cifar100.py`, with its switches off, or with `FF_USE_MGC=1`
    for MGC, builds the same network. The code repository's CPU equivalence tests check this, and
    all four `net_ema` state dicts load into it with every key matched.
- **MGC.** The MGC arm replaces the baseline's depth-scaled current-block term: block 0 loses its
  auxiliary and has no compensator. The implementation is faithful but not an exact recovery
  (ε=10⁻⁶, positive-part clamp).

## Other checkpoints (on request)

`release_manifest.json` lists all 98 result-bearing checkpoint files of the paper:

- the `stage1_best.pt` and `stage2_best.pt` of 49 runs on CIFAR-10 and CIFAR-100;
- for each file: its run, the paper tables it supports, its trainer and configuration, its source
  and lineage, its size and its SHA-256.

The 94 files marked `"availability": "on_request"` are available from the author on request
(amir.usefi75@gmail.com). A recipient can check them against the SHA-256 values in the manifest.
The following are not available at all:

- the published CIFAR-10 γ=0 L4/D256 seed-42 checkpoint, which is not recoverable;
- the Tiny ImageNet, text-domain and BP-control checkpoints (the paper reports their run logs and
  metrics only);
- the seed-42 run of the CIFAR-100 32-expert γ=0.7 baseline, which was not retained.

## Integrity

`SHA256SUMS` pins the four checkpoint files by their SHA-256 over the full file bytes, and
`release_manifest.json` records the same hashes. To verify a download:

```bash
sha256sum -c SHA256SUMS
```

The release was staged with `release/stage_release.py --subset minimal` from the code repository.

## License

- **Checkpoints:** Creative Commons Attribution 4.0 International (CC BY 4.0). If you use them,
  please cite the paper (below).
- **Code:** Apache-2.0 (the `LICENSE` file of the GitHub repository).
- **Datasets:** the checkpoints were trained on CIFAR-100, which is not redistributed here. A
  checkpoint stores network weights, optimizer and scheduler states and the contrastive feature
  queues, which hold projected features of training images and their labels. It contains no
  images. Use of the dataset itself follows its authors' terms.

## Dataset

The four files were trained on CIFAR-100 (Krizhevsky, 2009), loaded with torchvision:

- The 50,000 training images are split into 45,000 for training and 5,000 for validation.
- The split is seeded with the run seed (42 for all four files), so the four share it.
- The 10,000-image test set is used only for the final evaluation.

## Intended use and limitations

These checkpoints are for studying FF training. Possible uses:

- per-block separation and goodness;
- readouts and probes of the frozen backbones;
- prediction churn between the four arms.

The paper reports the accuracies, protocols and seed-level variation: Table 3, Appendix B and the
appendix readout tables. A single seed per arm cannot reproduce the paper's paired multi-seed
statistics, because those need the on-request files or retraining. The models are small
CIFAR-scale classifiers trained with a local learning rule. They were not evaluated for
robustness, fairness or out-of-distribution use.

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
