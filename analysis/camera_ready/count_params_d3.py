"""D3: recount FF model parameters from checkpoints (CPU only).

Run as:
  CUDA_VISIBLE_DEVICES='' python analysis/camera_ready/count_params_d3.py

Counts are taken from each checkpoint's `net_ema` state dict (the weights the
final test evaluation uses). BatchNorm running statistics / counters are
buffers and are excluded. "Active per token" = everything outside the MoE
experts plus top-k experts in each block. "Training-only" heads are the
SupCon projection (`contrastive_proj`) and the token reconstructor
(`reconstructor`), which the trainers call only inside the loss
(`contrastive_embed`, and `self.reconstructor(h) if self.training`).
`prototypes` and `lin_head` are NOT training-only: they produce the
prototype-alignment and learned goodness aspects that `predict_scores`
sums at inference (see `_compute_aspects` / `predict_scores` in
basic_fair_revised_valfix_cp_fair.py).
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json
import os
import sys
import time

import torch

torch.set_num_threads(6)

CF = (FF_HOME + "/amir-porjects/"
      "cifar_10_fair/making_v1_v4_fair_comparison")
FFN = FF_HOME + "/ff-next"
CKPTS = {
    "c10_cp_fair_seed42": f"{CF}/basic_fair_revised_valfix_cp_fair/checkpoints/"
                          "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_P2_seed42_stage1_best.pt",
    "c10_gamma0_seed123": f"{CF}/revised_training/out/gamma0_seed123/checkpoints/"
                          "ff_cifar10_cifar10_gamma0_seed123_L4_D256_P2_seed123_stage1_best.pt",
    "c100_gamma0_dense_seed42": f"{FFN}/checkpoints/trio/"
                                "ff_cifar100_cifar100_gamma0_seed42_L4_D256_P2_seed42_stage1_best.pt",
}
BUF = ("running_mean", "running_var", "num_batches_tracked")
TRAIN_ONLY = ("contrastive_proj", "reconstructor")


def count(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["net_ema"] if ck.get("net_ema") is not None else ck["model"]
    cfg = ck.get("cfg")
    c = cfg if isinstance(cfg, dict) else (vars(cfg) if cfg is not None else {})
    E, K, L = c.get("n_experts"), c.get("moe_top_k"), c.get("num_blocks")
    tot = exp = train_only = buffers = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if any(b in k for b in BUF):
            buffers += v.numel()
            continue
        tot += v.numel()
        if ".moe.experts." in k:
            exp += v.numel()
        if any(t in k for t in TRAIN_ONLY):
            train_only += v.numel()
    per_expert = exp // (E * L)
    active = (tot - exp) + L * K * per_expert
    st = os.stat(path)
    return {
        "checkpoint": path,
        "state_dict": "net_ema",
        "file_bytes": st.st_size,
        "file_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
        "cfg": {k: c.get(k) for k in ["n_experts", "moe_top_k", "num_blocks", "d_model",
                                      "num_classes", "n_aspects", "gamma_scale", "use_sam"]},
        "total_params": tot,
        "expert_params": exp,
        "non_expert_params": tot - exp,
        "params_per_expert": per_expert,
        "active_per_token": active,
        "training_only_heads(contrastive_proj+reconstructor)": train_only,
        "active_per_token_excluding_training_only_heads": active - train_only,
        "excluded_buffers(bn_running_stats)": buffers,
    }


def main(out):
    res = {"_header": {"item": "D3", "script": os.path.abspath(__file__),
                       "created": time.strftime("%Y-%m-%d")}}
    for name, p in CKPTS.items():
        res[name] = count(p)
        print(name, json.dumps({k: v for k, v in res[name].items() if "param" in k or "active" in k}))
    # counts logged by the non-FF runs (read from their final_metrics.json)
    fm = {
        "stripped_backbone_FF_seed42": f"{CF}/ff_backbone_stripped/runs_ff_backbone_stripped/"
                                       "ff_backbone_stripped_cifar10_L4_D256_P2_seed42_20260306_153106/final_metrics.json",
        "stripped_backbone_BP_weak_seed42": f"{CF}/ablations/ff_backbone_stripped_bp/runs_ff_backbone_bp/"
                                            "bp_stripped_backbone_cifar10_L4_D256_P2_seed42_20260307_035223/final_metrics.json",
        "stripped_backbone_BP_strong_seed42": f"{CF}/ablations/ff_backbone_stripped_bp_strongaug/runs_ff_backbone_bp_strongaug/"
                                              "bp_stripped_backbone_cifar10_L4_D256_P2_seed42_20260310_143157/final_metrics.json",
        "full_backbone_BP_seed42": f"{CF}/ablations/full_backbone_bp/runs_full_backbone_bp/"
                                   "bp_full_backbone_full_backbone_bp_cifar10_v1_L4_D256_P2_E32_K4_seed42_20260307_143455/final_metrics.json",
    }
    for name, p in fm.items():
        d = json.load(open(p))
        res[name] = {"final_metrics": p, "num_parameters": d.get("num_parameters"),
                     "test_top1": d.get("test_top1"), "val_top1": d.get("val_top1")}
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(os.path.join(FF_OUT, "results"), exist_ok=True)
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(FF_OUT, "results", "d3_param_counts.json"))
