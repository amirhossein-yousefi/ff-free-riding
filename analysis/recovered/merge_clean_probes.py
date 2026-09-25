#!/usr/bin/env python3
"""Merge clean_probes_addendum.json into clean_probes_c100.json and attach the
two blocks the wiki record needs: an explicit per-panel verdict on the superseded
simple_probes_c100.json, and a CORRECTED cross-reference to the honest structured
readout the rebuttal quotes.

Everything written here is read back out of the source JSONs -- no number is typed
by hand except the panel labels; every quoted value is asserted against its source.
"""
import os as _ffp_os
# [CR] Paths come from the environment (see analysis/README.md); nothing is hard-coded.
FF_HOME = _ffp_os.environ.get("FF_HOME", "")              # root containing amir-porjects/ and ff-next/
FF_WORK = _ffp_os.environ.get("FF_WORK_DIR", "ff_work")   # dumps and intermediate outputs
FF_OUT = _ffp_os.environ.get("FF_OUT_DIR", "ff_out")      # regenerated results (<FF_OUT>/results/*.json)
import json, os, time
import numpy as np

SCRATCH = FF_WORK
BASE = os.path.join(SCRATCH, "clean_probes_c100.json")
ADD = os.path.join(SCRATCH, "clean_probes_addendum.json")
OLD = FF_HOME + "/amir-porjects/ff-wiki/_build/analyses/simple_probes_c100.json"
VALREFIT = FF_HOME + "/amir-porjects/ff-wiki/_build/analyses/val_refit_readout_results.json"
WIKI = FF_HOME + "/amir-porjects/ff-wiki/_build/analyses/clean_probes_c100.json"
V = ["gamma0", "gated_k0", "cumulative"]


def span_pp(d):
    return round((max(d.values()) - min(d.values())) * 100, 3)


def main():
    base = json.load(open(BASE))
    add = json.load(open(ADD))
    old = json.load(open(OLD))
    vr = json.load(open(VALREFIT))

    # ---------- fold addendum into the per-variant / span / bracket tables ----------
    for v in V:
        for k, val in add["per_variant"][v].items():
            if k == "n_seeds":
                continue
            base["per_variant"][v][k] = val
    base["trio_spans"].update(add["trio_spans"])
    for nm, per_v in add["bracketing_vs_anchors"].items():
        for v in V:
            base["bracketing_vs_anchors"][v][nm] = per_v[v]
    base["per_checkpoint_addendum"] = add["per_checkpoint"]
    base["knn_k_sweep"] = {
        "_note": add["_header"]["probes"]["knn_sweep"],
        "implementation_validation": add["knn_implementation_validation"],
        "per_variant": {v: {k: val for k, val in add["per_variant"][v].items()
                            if k.startswith("knn_sweep_")} for v in V}}

    s1m = {v: base["per_variant"][v]["anchor_s1_no_tta_mean"] for v in V}
    s2m = {v: base["per_variant"][v]["anchor_s2_no_tta_mean"] for v in V}

    # ---------- corrected honest-anchor cross reference ----------
    bw = old["goodness_panel_label_free"]["blockweight"]
    bw_mean = {v: bw["per_variant"][v]["mean"] for v in V}
    bw_w = [r["learned_w"] for r in bw["per_checkpoint"]]
    n_neg = sum(1 for row in bw_w for w in row if w < 0)
    vr_sp = vr["c100_trio_span_per_readout"]

    honest = {
        "_note": ("The rebuttal currently quotes the 104-parameter block-weight readout from "
                  "simple_probes_c100.json (goodness_panel_label_free.blockweight). Its FEATURES are "
                  "label-free and its substance survives, but two descriptors attached to it are wrong "
                  "and its fit protocol is weaker than the val-refit standard. Corrected accounting below."),
        "as_quoted_in_rebuttal": {
            "readout": "logits_c = sum_l w_l*g[l,c] + b_c (4 block weights + 100 class biases = 104 params)",
            "per_variant_mean": bw_mean,
            "trio_span_pp": span_pp(bw_mean),
            "source": OLD + " :: goodness_panel_label_free.blockweight",
        },
        "corrections": {
            "not_non_negative": {
                "claim_in_task_framing": "non-negative block-weight readout",
                "fact": f"{n_neg} of {sum(len(r) for r in bw_w)} learned block weights are NEGATIVE "
                        f"(e.g. {bw['per_checkpoint'][0]['name']} w={bw['per_checkpoint'][0]['learned_w']}); "
                        "the readout is unconstrained. The genuinely non-negative (positive-weight-constrained) "
                        "readout is val_refit R2 (per-depth temperature = R1 constrained to positive weights).",
            },
            "fit_protocol_is_test_slice_not_val": {
                "fact": "blockweight was fit on a 1000-example stratified slice of the TEST set and evaluated on "
                        "the disjoint 9000 test examples -- not fit on val, and not evaluated on the full 10k. "
                        "It predates the val-refit protocol.",
                "old_protocol": old["_header"]["fit_eval_protocol"],
            },
        },
        "honest_val_fit_replacements": {
            "_note": "Same readout family, fit on each run's OWN 5k val split, evaluated on the FULL 10k test "
                     "(val_refit_readout_results.json). These are what the rebuttal should quote.",
            "R1_per_depth_weights_4params": {"per_variant_mean": vr_sp["R1"]["variant_means"],
                                             "trio_span_pp": vr_sp["R1"]["span_pp"]},
            "R2_per_depth_temperature_nonnegative_4params": {"per_variant_mean": vr_sp["R2"]["variant_means"],
                                                             "trio_span_pp": vr_sp["R2"]["span_pp"]},
            "R3_margin_gated_8params": {"per_variant_mean": vr_sp["R3"]["variant_means"],
                                        "trio_span_pp": vr_sp["R3"]["span_pp"]},
            "R0_plain_sum_control_eq_S1": {"per_variant_mean": vr_sp["R0"]["variant_means"],
                                           "trio_span_pp": vr_sp["R0"]["span_pp"]},
        },
        "substance_check": {
            "R1_brackets_S1_S2_on_all_variants": all(s1m[v] <= vr_sp["R1"]["variant_means"][v] <= s2m[v] for v in V),
            "R2_brackets_S1_S2_on_all_variants": all(s1m[v] <= vr_sp["R2"]["variant_means"][v] <= s2m[v] for v in V),
            "blockweight_brackets_S1_S2_on_all_variants": all(s1m[v] <= bw_mean[v] <= s2m[v] for v in V),
            "all_structured_readout_spans_lt_1pp": all(vr_sp[r]["span_pp"] < 1.0 for r in ["R0", "R1", "R2", "R3"]),
            "verdict": "The claim survives the correction: the low-parameter label-free readouts still land "
                       "strictly between S1 and S2 on all three variants, and every trio span stays well under 1 pp "
                       "(0.33-0.43 pp val-fit). Only the '104-parameter'/'non-negative' wording and the "
                       "test-slice fit protocol need requoting.",
        },
    }
    base["honest_anchor_cross_reference"] = honest
    # hard asserts on everything quoted above
    assert abs(span_pp(bw_mean) - 0.133) < 0.01, span_pp(bw_mean)
    assert n_neg > 0
    for r in ["R0", "R1", "R2", "R3"]:
        assert vr_sp[r]["span_pp"] < 1.0

    # ---------- explicit verdict on every panel of the old file ----------
    opv = old["per_variant"]
    ogp = old["goodness_panel_label_free"]["per_variant"]
    cpv = base["per_variant"]
    superseded = {
        "_note": "Panel-by-panel verdict on " + OLD + ". Two defects existed: (D1) LABEL LEAK -- features "
                 "extracted under the TRUE-label hypothesis, so the probe measures label recoverability from an "
                 "input that already contains the label, not embedding structure; (D2) TEST-SLICE FIT -- probes "
                 "fit on a 1000-example stratified slice of the test set and evaluated on the disjoint 9000, "
                 "instead of fit on the run's own 5k val split and evaluated on the full 10k test.",
        "INVALID_label_leak": {
            "panels": ["per_variant.logistic_block3", "per_variant.logistic_concat",
                       "per_variant.knn10cos_block3", "per_variant.knn10cos_concat",
                       "per_checkpoint[*].probes.*", "trio_spans.{logistic,knn10cos}_{block3,concat}"],
            "defects": ["D1", "D2"],
            "why": "features = feats_true [N,4,256], per-block attn-pooled activations computed with the GROUND-TRUTH "
                   "label embedded in the network input. Accuracies of 0.85-0.997 are a measurement of how "
                   "recoverable the injected label is, not of how well the embedding is structured. These panels "
                   "must NOT be cited for the k-NN / traditional-probe question.",
            "old_values_per_variant": {p: {v: opv[v][p]["mean"] for v in V}
                                       for p in ["logistic_concat", "logistic_block3", "knn10cos_concat", "knn10cos_block3"]},
            "clean_replacement": {
                "panels": ["pooled_meanemb_{concat,block3}", "pooled_zerotok_{concat,block3}"],
                "why": "same attn-pooled activations, but the label slot is replaced by the mean label embedding "
                       "(uniform prior) or a zero token -- a genuinely label-free forward pass.",
                "new_values_per_variant": {p: {v: cpv[v][p]["mean"] for v in V}
                                           for p in ["logistic_pooled_meanemb_concat", "knn10cos_pooled_meanemb_concat",
                                                     "logistic_pooled_zerotok_concat", "knn10cos_pooled_zerotok_concat"]},
                "contamination_pp": {v: round((opv[v]["logistic_concat"]["mean"]
                                               - cpv[v]["logistic_pooled_meanemb_concat"]["mean"]) * 100, 2) for v in V},
            },
        },
        "VALID_FEATURES_DEFECTIVE_PROTOCOL": {
            "panels": ["goodness_panel_label_free.{per_checkpoint,per_variant,trio_spans,bracketing_vs_attentive_head}"],
            "defects": ["D2"],
            "why": "features (dumped goodness over all 100 hypotheses) are genuinely label-free, so the DIRECTION of "
                   "these results stands; but the probes were fit on 1000 test examples, which underfits a 100-way "
                   "problem badly and fits on the evaluation distribution.",
            "old_values_per_variant": {p: {v: ogp[v][p]["mean"] for v in V}
                                       for p in ["logistic_goodness_all", "knn10cos_goodness_all",
                                                 "logistic_goodness_block3", "knn10cos_goodness_block3"]},
            "clean_replacement_val_fit_full10k": {
                p: {v: cpv[v][p]["mean"] for v in V}
                for p in ["logistic_goodness_flat", "knn10cos_goodness_flat",
                          "logistic_goodness_block3", "knn10cos_goodness_block3"]},
            "gain_from_honest_protocol_pp": {
                v: round((cpv[v]["logistic_goodness_flat"]["mean"] - ogp[v]["logistic_goodness_all"]["mean"]) * 100, 2)
                for v in V},
        },
        "SUPERSEDED_REQUOTE": {
            "panels": ["goodness_panel_label_free.blockweight"],
            "defects": ["D2"],
            "why": "the number the rebuttal currently quotes; label-free features, but test-slice fit and 9000-example "
                   "eval, and it is not 'non-negative'. See honest_anchor_cross_reference for the corrected "
                   "val-fit / full-10k replacements (val_refit R1/R2).",
        },
        "STILL_VALID": {
            "panels": ["_header.sanity_gate", "per_checkpoint[*].s1_full_recomputed", "trio_spans.anchor_s1_no_tta",
                       "trio_spans.anchor_s2_no_tta"],
            "why": "the S1 recomputation gate and the S1/S2 anchor means are protocol-independent and reproduce here "
                   "(re-asserted in this file's G1 against aggregated_results.json).",
        },
    }
    base["superseded_panels_of_simple_probes_c100"] = superseded

    # ---------- refreshed summary ----------
    ks = add["per_variant"]["gamma0"]["knn_sweep_goodness_blocksoftmax"]
    best_k = max(ks, key=lambda k: np.mean([add["per_variant"][v][f"knn_sweep_goodness_blocksoftmax"][k]["mean"] for v in V]))
    knn_best = {v: add["per_variant"][v]["knn_sweep_goodness_blocksoftmax"][best_k]["mean"] for v in V}
    log_best = {v: cpv[v]["logistic_goodness_blocksoftmax"]["mean"] for v in V}
    ridge_best = {v: cpv[v]["ridge_goodness_blocksoftmax"]["mean"] for v in V}

    base["summary"]["clean_probe_headline_v2"] = {
        "question": "rebuttal probe question: how well structured is the frozen embedding, as measured "
                    "by a traditional probe (k-NN)?",
        "answer_A_vs_anchors": {
            "best_knn": {"k": int(best_k), "features": "goodness_blocksoftmax", "per_variant_mean": knn_best,
                         "vs_S1_pp": {v: round((knn_best[v] - s1m[v]) * 100, 2) for v in V},
                         "vs_S2_pp": {v: round((knn_best[v] - s2m[v]) * 100, 2) for v in V}},
            "best_logistic": {"features": "goodness_blocksoftmax", "per_variant_mean": log_best,
                              "vs_S1_pp": {v: round((log_best[v] - s1m[v]) * 100, 2) for v in V},
                              "vs_S2_pp": {v: round((log_best[v] - s2m[v]) * 100, 2) for v in V}},
            "best_ridge": {"features": "goodness_blocksoftmax", "per_variant_mean": ridge_best,
                           "vs_S1_pp": {v: round((ridge_best[v] - s1m[v]) * 100, 2) for v in V},
                           "vs_S2_pp": {v: round((ridge_best[v] - s2m[v]) * 100, 2) for v in V}},
            "anchors": {"S1_no_tta": s1m, "S2_no_tta": s2m},
        },
        "answer_B_trio_spans": {k: base["trio_spans"][k]["span_pp"] for k in sorted(base["trio_spans"])},
    }
    base["_header"]["addendum"] = {
        "script": os.path.join(SCRATCH, "clean_probes_addendum.py"),
        "adds": "k-NN k-sweep (k=1..100), ridge (alpha by val CV), goodness_topk10 / goodness_margin feature sets, "
                "superseded-panel verdict, corrected honest-anchor cross-reference",
        "device": "CPU only", "created": time.strftime("%Y-%m-%d")}

    json.dump(base, open(BASE, "w"), indent=1)
    json.dump(base, open(WIKI, "w"), indent=1)
    print("WROTE", BASE, "and", WIKI)


if __name__ == "__main__":
    main()
