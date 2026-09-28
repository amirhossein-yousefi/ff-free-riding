"""Every path used by the camera-ready analysis scripts, in one place.

Paths come from environment variables (no hard-coded locations):
  FF_HOME     root of the run-log archive; it contains amir-porjects/ (the original project
              layout: run directories, events.jsonl, layer_metrics.jsonl, config.json) and
              ff-next/checkpoints/trio (CIFAR-100 trio checkpoints). Read-only.
  FF_OUT_DIR  output root; results go to <FF_OUT_DIR>/results, tables to <FF_OUT_DIR>/tables.
  FF_PAPER_DIR  (check_numbers.py only) a git checkout of the paper sources.

Rules:
  * every run directory is named explicitly (timestamp included) so that aborted duplicates
    that live next to the result-bearing runs can never be picked up by a glob;
  * everything under FF_HOME is read-only; scripts write only to FF_OUT_DIR.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
from pathlib import Path

HOME = Path(FF_HOME)
PR = HOME / "amir-porjects"
CF = PR / "cifar_10_fair" / "making_v1_v4_fair_comparison"
TEX = CF / "NeurIPS_FF_Net_Last_submitted"              # submitted LaTeX + v1 supplement (read-only)
SUPP_METRICS = Path(__file__).resolve().parents[2] / "metric_summaries"  # this supplement
MR = CF / "mgc_rebuttal"
COLAB = MR / "from_colab" / "runs_from_colab"            # result-bearing Colab runs
AN = PR / "ff-wiki" / "_build" / "analyses"              # rebuttal-era analysis JSONs
TRIO = CF / "revised_training" / "extracted" / "ff_cifar100_multiseed"
GA = CF / "basic_fair_revised_valfix_cp_fair_hardness_gated" / "ff_gated_ablation"
ABL = CF / "ablations"
BPA = MR / "bp_anchors"
FFN = HOME / "ff-next"

CR = Path(FF_OUT)                                        # output root (was the camera-ready repository)
RESULTS = CR / "results"
TABLES = CR / "tables"
PAPER = Path(_ffp_os.environ.get("FF_PAPER_DIR", str(CR / "paper")))
BASELINE_COMMIT = "389b1fa"                              # commit 0 = submitted version
# [CR] Experiment folders of the camera-ready GPU runs F1 (MGC block-0 control) and F2 (gamma=0 seed-42
# rerun) inside the run archive: runs/ (run directories), logs/ (job logs), the launch specification and
# the fork trainer. Read-only; used by f1_analysis.py, f2_analysis.py and compute_budget.py.
CR_ARCHIVE = PR / "neuroips_review" / "camera_ready"     # the camera-ready repository inside the archive
F1_DIR = Path(_ffp_os.environ.get("FF_F1_DIR", str(CR_ARCHIVE / "experiments" / "f1_mgc_block0")))
F2_DIR = Path(_ffp_os.environ.get("FF_F2_DIR", str(CR_ARCHIVE / "experiments" / "f2_gamma0_s42")))

SEEDS = (42, 123, 456)
# final_test_eval keys (fractions in events.jsonl) and short protocol labels
PROTOCOLS = {
    "S1": "stage1_test_top1_no_tta",
    "S1-TTA": "stage1_test_top1_tta",
    "S2": "stage2_test_top1_no_tta",
    "S2-TTA": "stage2_test_top1_tta",
}
PRED_FILES = {
    "S1": "preds_stage1_test_no_tta.npz",
    "S1-TTA": "preds_stage1_test_tta.npz",
    "S2": "preds_stage2_test_no_tta.npz",
    "S2-TTA": "preds_stage2_test_tta.npz",
}
PROTOCOL_LABELS = {
    "S1": "Stage-1 goodness-sum readout, single crop (val-selected EMA stage1_best)",
    "S1-TTA": "Stage-1 goodness-sum readout, flip TTA (val-selected EMA stage1_best)",
    "S2": "Stage-2 attentive probe, single crop (val-selected stage2_best)",
    "S2-TTA": "Stage-2 attentive probe, flip TTA (val-selected stage2_best)",
}

# ---------------------------------------------------------------- CIFAR-100 trio (L4/D256)
# Only these dirs carry final_test_eval; the sibling dirs with other timestamps are
# aborted duplicates (CAMERA_READY_PLAN A1: cumulative ONLY from the three runs below).
_TRIO_TS = {
    "cumulative": {42: "20260418_032149", 123: "20260419_010632", 456: "20260420_051234"},
    "gamma0": {42: "20260420_101044", 123: "20260421_124316", 456: "20260422_142245"},
    "gated_k0": {42: "20260422_185437", 123: "20260423_150626", 456: "20260424_153513"},
}
TRIO_RUNS = {
    v: {s: TRIO / f"runs_{v}_seed{s}" / f"ff_cifar100_cifar100_{v}_seed{s}_L4_D256_bs256_seed{s}_{ts}"
        for s, ts in d.items()}
    for v, d in _TRIO_TS.items()
}
TRIO_EXCLUDED = {  # aborted duplicates (no final_test_eval) — listed so the JSONs can say so
    "cumulative": [TRIO / "runs_cumulative_seed123" / "ff_cifar100_cifar100_cumulative_seed123_L4_D256_bs256_seed123_20260418_190627",
                   TRIO / "runs_cumulative_seed456" / "ff_cifar100_cifar100_cumulative_seed456_L4_D256_bs256_seed456_20260419_141800"],
}

# ---------------------------------------------------------------- MGC
MGC_C100_RUNS = {
    42: COLAB / "ff_cifar100_mgc_c100_L4_D256_bs256_seed42_20260724_220559",
    123: COLAB / "ff_cifar100_mgc_c100_L4_D256_bs256_seed123_20260724_191009",
    456: COLAB / "ff_cifar100_mgc_c100_L4_D256_bs256_seed456_20260725_194446",  # resumed; layer log from ep 192
}
MGC_D128_RUNS = {
    42: MR / "runs" / "ff_cifar10_mgc_c1_L4_D128_bs512_seed42_20260723_223516",   # local DGX run (H8 lineage)
    123: COLAB / "ff_cifar10_mgc_c1_L4_D128_bs512_seed123_20260723_205123",
    456: COLAB / "ff_cifar10_mgc_c1_L4_D128_bs512_seed456_20260724_053002",
}
MGC_TRAINER = MR / "mgc_c100_trainer.py"

# ---------------------------------------------------------------- squared hinge (C10 L4/D128)
HINGE_RUNS = {
    "g07": {42: COLAB / "ff_cifar10_hinge_g07_L4_D128_bs512_seed42_20260723_214017",
            123: COLAB / "ff_cifar10_hinge_g07_L4_D128_bs512_seed123_20260724_191321",
            456: COLAB / "ff_cifar10_hinge_g07_L4_D128_bs512_seed456_20260725_194352"},  # resumed; layer log from ep 171
    "g0": {42: COLAB / "ff_cifar10_hinge_g0_L4_D128_bs512_seed42_20260724_063123",
           123: COLAB / "ff_cifar10_hinge_g0_L4_D128_bs512_seed123_20260725_035433",
           456: COLAB / "ff_cifar10_hinge_g0_L4_D128_bs512_seed456_20260725_210209"},
}
HINGE_ABORTED_DGX = [  # never aggregate (stop at ep 46 / 23, no final_test_eval)
    MR / "runs" / "ff_cifar10_hinge_g07_L4_D128_bs512_seed42_20260724_120246",
    MR / "runs" / "ff_cifar10_hinge_g0_L4_D128_bs512_seed42_20260724_152743",
]

# ---------------------------------------------------------------- C10 L4/D128 seed-42 gated-ablation anchors (SAM off)
GA_RUNS = {
    "gamma07_constant": GA / "runs" / "ff_cifar10_baseline_constant_L4_D128_bs512_seed42_20260313_232043",
    "gamma0_no_all_collab": GA / "runs" / "ff_cifar10_no_all_collab_L4_D128_bs512_seed42_20260315_124117",
    "gated_k0_t1": GA / "runs" / "ff_cifar10_adaptive_t1_k0_L4_D128_bs512_seed42_20260314_032410",
    "lcff_prefix": GA / "runs" / "ff_cifar10_lcff_prefix_L4_D128_bs512_seed42_20260315_084228",
}

# ---------------------------------------------------------------- BP / FF calibration (C10, Table 2)
BP_RUNS = {
    "bp_strong": {
        42: ABL / "ff_backbone_stripped_bp_strongaug" / "runs_ff_backbone_bp_strongaug" / "bp_stripped_backbone_cifar10_L4_D256_P2_seed42_20260310_143157",
        123: BPA / "ff_backbone_stripped_bp_strongaug" / "runs_ff_backbone_bp_strongaug" / "bp_stripped_backbone_cifar10_L4_D256_P2_seed123_20260725_173842",
        456: BPA / "ff_backbone_stripped_bp_strongaug" / "runs_ff_backbone_bp_strongaug" / "bp_stripped_backbone_cifar10_L4_D256_P2_seed456_20260726_013559",
    },
    "bp_weak": {
        42: ABL / "ff_backbone_stripped_bp" / "runs_ff_backbone_bp" / "bp_stripped_backbone_cifar10_L4_D256_P2_seed42_20260307_035223",
        123: BPA / "ff_backbone_stripped_bp" / "runs_ff_backbone_bp" / "bp_stripped_backbone_cifar10_L4_D256_P2_seed123_20260725_221323",
        456: BPA / "ff_backbone_stripped_bp" / "runs_ff_backbone_bp" / "bp_stripped_backbone_cifar10_L4_D256_P2_seed456_20260726_061013",
    },
    "plain_cnn_bp": {
        42: ABL / "plain_cnn_ff_baseline" / "runs_plain_cnn_bp" / "bp_plaincnn_plain_cnn_bp_cifar10_v1_seed42_20260306_104748",
        123: BPA / "plain_cnn_ff_baseline" / "runs_plain_cnn_bp" / "bp_plaincnn_plain_cnn_bp_cifar10_v1_seed123_20260726_005331",
        456: BPA / "plain_cnn_ff_baseline" / "runs_plain_cnn_bp" / "bp_plaincnn_plain_cnn_bp_cifar10_v1_seed456_20260726_085009",
    },
    "plain_cnn_ff": {
        42: ABL / "plain_cnn_ff_baseli_ff" / "runs_plain_cnn_ff" / "ff_plaincnn_plain_cnn_ff_cifar10_v1_seed42_20260306_121504",
        123: BPA / "plain_cnn_ff_baseli_ff" / "runs_plain_cnn_ff" / "ff_plaincnn_plain_cnn_ff_cifar10_v1_seed123_20260725_202326",
        456: BPA / "plain_cnn_ff_baseli_ff" / "runs_plain_cnn_ff" / "ff_plaincnn_plain_cnn_ff_cifar10_v1_seed456_20260726_042004",
    },
}
# a July-2026 rerun of strong-aug BP seed 42 (bit-identical preds) kept as a determinism check only
BP_STRONG_S42_RERUN = BPA / "ff_backbone_stripped_bp_strongaug" / "runs_ff_backbone_bp_strongaug" / "bp_stripped_backbone_cifar10_L4_D256_P2_seed42_20260725_144750"

# ---------------------------------------------------------------- CIFAR-10 flagship pair (L4/D256): CP-FAIR vs gamma=0
C10_PAIR_RUNS = {
    "gamma0": {
        42: ABL / "no_all_collab" / "runs" / "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260312_132043",
        123: CF / "revised_training" / "out" / "gamma0_seed123" / "runs" / "ff_cifar10_cifar10_gamma0_seed123_L4_D256_bs512_seed123_20260417_235110",
        456: CF / "revised_training" / "out" / "gamma0_seed456" / "runs" / "ff_cifar10_cifar10_gamma0_seed456_L4_D256_bs512_seed456_20260422_135437",
    },
    "cp_fair": {
        42: CF / "basic_fair_revised_valfix_cp_fair" / "runs" / "ff_cifar10_cifar10_cp_fair_clean_progressive_L4_D256_bs512_seed42_20260220_202035",
        123: CF / "basic_fair_revised_valfix_cp_fair" / "runs_seed123" / "ff_cifar10_cifar10_cp_fair_clean_progressive_seed123_L4_D256_bs512_seed123_20260320_001527",
        456: CF / "basic_fair_revised_valfix_cp_fair" / "runs_seed456" / "ff_cifar10_cifar10_cp_fair_clean_progressive_seed456_L4_D256_bs512_seed456_20260327_080127",
    },
}
SUPP_AGG = SUPP_METRICS / "aggregated_results.json"

# ---------------------------------------------------------------- S2-FF / S2-FF-CE readouts (per-run JSONs)
S2FF_TRIO = {v: MR / "s2ff_results" / f"s2ff_c100_{v}_seed42.json" for v in ("gamma0", "gated_k0", "cumulative")}
S2FFCE_TRIO = {v: MR / "s2ff_ce_results" / f"s2ffce_c100_{v}_seed42.json" for v in ("gamma0", "gated_k0", "cumulative")}
S2FF_MGC = {s: MR / "s2ff_from_colab" / "s2ff_results" / f"s2ff_mgc_c100_seed{s}.json" for s in SEEDS}
S2FFCE_MGC = {s: MR / "from_colab" / "s2ff_ce" / "s2ff_ce_results" / f"s2ffce_mgc_c100_seed{s}.json" for s in SEEDS}

# ---------------------------------------------------------------- analysis JSONs used as sources (AN)
AN_VAL_REFIT = AN / "val_refit_readout_results.json"
AN_CLEAN_PROBES = AN / "clean_probes_c100.json"        # the ONLY probe source allowed (H24 label leak elsewhere)
AN_GRADSHARE = AN / "gradient_share_results.json"
AN_REDIST = AN / "redistribution_demo.json"
AN_C10_PAIRED = AN / "paired_bootstrap_cifar10.json"
AN_MGC_D128 = AN / "mgc_3seed_aggregate.json"
AN_BP = AN / "bp_anchors_multiseed.json"
AN_S2FF_FULL = AN / "s2ff_full_results.json"
AN_S2FFCE = AN / "s2ffce_results.json"
AN_S2FF_TRIO = AN / "s2ff_trio_results.json"
C100_PAIRED_AUDIT = CF / "audit" / "60_paired_dissociation.json"  # published S2-TTA trio CIs

N_BOOT = 5000
BOOT_SEED = 0
