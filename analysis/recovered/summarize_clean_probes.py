#!/usr/bin/env python3
"""Add the summary block to clean_probes_c100.json and write it to ff-wiki/_build/analyses/
(the author's archive of rebuttal-era analysis JSONs; not distributed, see analysis/README.md)."""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json, os, shutil

SCRATCH = FF_WORK
SRC = os.path.join(SCRATCH, "clean_probes_c100.json")
PRIOR = FF_HOME + "/amir-porjects/ff-wiki/_build/analyses/simple_probes_c100.json"
DEST = FF_HOME + "/amir-porjects/ff-wiki/_build/analyses/clean_probes_c100.json"

d = json.load(open(SRC))
prior = json.load(open(PRIOR))
pv, spans, br = d["per_variant"], d["trio_spans"], d["bracketing_vs_anchors"]
VARIANTS = ["gamma0", "gated_k0", "cumulative"]

probe_keys = [k for k in pv["gamma0"] if k.startswith(("logistic_", "knn10cos_"))]

# contamination quantification: prior feats_true probes vs clean pooled probes
prior_pv = prior["per_variant"]
contam = {}
for v in VARIANTS:
    contam[v] = {
        "prior_feats_true_logistic_concat_eval9000": prior_pv[v]["logistic_concat"]["mean"],
        "clean_pooled_meanemb_logistic_concat_test10k": pv[v]["logistic_pooled_meanemb_concat"]["mean"],
        "drop_pp": round((prior_pv[v]["logistic_concat"]["mean"]
                          - pv[v]["logistic_pooled_meanemb_concat"]["mean"]) * 100, 2),
        "prior_feats_true_knn_concat_eval9000": prior_pv[v]["knn10cos_concat"]["mean"],
        "clean_pooled_meanemb_knn_concat_test10k": pv[v]["knn10cos_pooled_meanemb_concat"]["mean"],
    }

# underfit fix quantification: prior goodness panel (n=1000 test-slice fit) vs clean val-5k fit
prior_good = prior["goodness_panel_label_free"]["per_variant"]
underfit_fix = {v: {
    "prior_logistic_goodness_all_fit1000": prior_good[v]["logistic_goodness_all"]["mean"],
    "clean_logistic_goodness_flat_fit5000val": pv[v]["logistic_goodness_flat"]["mean"],
    "gain_pp": round((pv[v]["logistic_goodness_flat"]["mean"]
                      - prior_good[v]["logistic_goodness_all"]["mean"]) * 100, 2),
    "prior_knn_goodness_all_fit1000": prior_good[v]["knn10cos_goodness_all"]["mean"],
    "clean_knn_goodness_flat_fit5000val": pv[v]["knn10cos_goodness_flat"]["mean"],
} for v in VARIANTS}

span_table = {k: {"span_pp": spans[k]["span_pp"], "holds_lt_1pp": spans[k]["holds_lt_1pp"]}
              for k in ["anchor_s1_no_tta", "anchor_s2_no_tta"] + probe_keys}

# which probes bracket S1..S2 on every variant
bracket_all = {pk: all(br[v][pk]["within_s1_s2_bracket"] for v in VARIANTS) for pk in probe_keys}
above_s1_all = {pk: all(br[v][pk]["above_s1"] for v in VARIANTS) for pk in probe_keys}

d["summary"] = {
    "headline": None,  # filled below
    "dissociation_span_pp_by_probe": span_table,
    "probes_bracketing_s1_s2_on_all_variants": bracket_all,
    "probes_above_s1_on_all_variants": above_s1_all,
    "contamination_quantified_vs_prior_feats_true": contam,
    "prior_goodness_panel_underfit_fixed": underfit_fix,
    "cross_reference_val_refit_R1": {
        "note": "structured per-depth-weight readout (val-fit, honest) from val_refit_readout_results.json for comparison with the full-rank logistic on the same goodness features",
        "R1_eval_acc_mean": {"gamma0": 0.67763, "gated_k0": 0.67737, "cumulative": 0.67387},
        "R1_trio_span_pp": 0.376,
    },
}

# headline
s1span = spans["anchor_s1_no_tta"]["span_pp"]; s2span = spans["anchor_s2_no_tta"]["span_pp"]
best_probe = max(probe_keys, key=lambda k: min(pv[v][k]["mean"] for v in VARIANTS))
best_str = ", ".join("{} {:.4f}".format(v, pv[v][best_probe]["mean"]) for v in VARIANTS)
d["summary"]["headline"] = (
    "Honest label-free probes (fit on each run's own 5k val split, eval on the full 10k test): "
    f"best probe = {best_probe} ({best_str}); "
    f"anchor spans S1 {s1span}pp / S2 {s2span}pp. "
    "See dissociation_span_pp_by_probe for the trio span under every probe; "
    "prior feats_true probes (0.986-0.998) were label-contaminated — clean pooled label-free probes land "
    f"logistic concat {pv['gamma0']['logistic_pooled_meanemb_concat']['mean']:.4f}/"
    f"{pv['gated_k0']['logistic_pooled_meanemb_concat']['mean']:.4f}/"
    f"{pv['cumulative']['logistic_pooled_meanemb_concat']['mean']:.4f} (gamma0/gated_k0/cumulative)."
)

json.dump(d, open(SRC, "w"), indent=1)
shutil.copyfile(SRC, DEST)
print("summary added; published to", DEST)
print(json.dumps(d["summary"]["dissociation_span_pp_by_probe"], indent=1))
print("bracket_all:", json.dumps(bracket_all, indent=1))
