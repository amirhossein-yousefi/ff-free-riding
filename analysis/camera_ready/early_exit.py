"""E2 (validation-selected early exit): summaries shared by make_tables.py and make_figures.py.

Source: results/e2_exit.json (experiments/e2_exit/select_exit.py). Every value is derived from that file;
nothing is typed by hand. Protocol (see the file's _header): S1 goodness-sum readout, single crop, EMA
weights of the validation-selected stage1_best checkpoint; each operating point is chosen on the run's own
5k validation split and evaluated once on the 10k test split; delta = test accuracy at the operating point
minus full-depth test accuracy of the same checkpoint (paired, same images); SD = sample SD (ddof=1).

Objective families (for the repaired-vs-cumulative comparison):
  cumulative  constant-gamma cumulative goodness: CIFAR-10 constant gamma=0.7 and LCFF gamma=1.0 (seed 42),
              CIFAR-100 trio "cumulative" (gamma=0.7, 3 seeds);
  gamma0      block-local repair (gamma=0): CIFAR-10 no_all_collab, CIFAR-100 trio gamma0;
  gated       hardness-gated repair: the six CIFAR-10 adaptive arms, CIFAR-100 trio gated_k0.
"repaired" = gamma0 + gated. CIFAR-100 comparisons are paired by seed; the CIFAR-10 arms are one seed each,
so they are compared descriptively only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import config as C

RULES = ("T_iso_val", "T_argmax_val", "D_dynamic_exit")
SEEDS = (42, 123, 456)

C10_ARMS = {  # run name -> (objective family, short LaTeX label)
    "c10_gated_baseline_constant_seed42": ("cumulative", r"constant $\gamma{=}0.7$"),
    "c10_gated_lcff_prefix_seed42": ("cumulative", r"LCFF $\gamma{=}1.0$"),
    "c10_gated_no_all_collab_seed42": ("gamma0", r"$\gamma{=}0$"),
    "c10_gated_adaptive_t1_k0_seed42": ("gated", r"$\tau{=}1,\kappa{=}0$"),
    "c10_gated_adaptive_t3_k0_seed42": ("gated", r"$\tau{=}3,\kappa{=}0$"),
    "c10_gated_adaptive_t1_k2_seed42": ("gated", r"$\tau{=}1,\kappa{=}2$"),
    "c10_gated_adaptive_t1_k4_seed42": ("gated", r"$\tau{=}1,\kappa{=}4$"),
    "c10_gated_adaptive_prev_only_seed42": ("gated", r"prev-only $\kappa{=}0$"),
    "c10_gated_adaptive_prev_t1_k2_seed42": ("gated", r"prev-only $\kappa{=}2$"),
}
C100_VARIANT_FAMILY = {"gamma0": "gamma0", "gated_k0": "gated", "cumulative": "cumulative"}


def load(results_dir: Path = C.RESULTS):
    return json.load(open(Path(results_dir) / "e2_exit.json"))


def family_of(name: str) -> str:
    if name.startswith("c100_"):
        return C100_VARIANT_FAMILY[name[len("c100_"):].rsplit("_seed", 1)[0]]
    return C10_ARMS[name][0]


def op(run: dict, rule: str) -> dict:
    """The selected operating point of one run under one rule, in percent / pp."""
    r = run["dynamic_exit"] if rule == "D_dynamic_exit" else run["truncation"][rule]
    out = dict(saved_pct=100.0 * r["flops_saved"], delta_pp=float(r["delta_test_pp"]),
               exit_acc_pct=100.0 * r["test_acc"], full_acc_pct=100.0 * r["test_acc_full"])
    if rule == "D_dynamic_exit":
        out.update(mean_blocks=float(r["test_mean_blocks"]), tau=float(r["tau"]))
    else:
        out.update(depth=int(r["depth"]))
    return out


def stats(v):
    a = np.asarray(v, dtype=np.float64)
    return dict(n=int(a.size), mean=float(a.mean()), sd=float(a.std(ddof=1)) if a.size > 1 else None,
                min=float(a.min()), max=float(a.max()), values=[float(x) for x in a])


def group_summary(runs: dict, names: list) -> dict:
    s = {"members": list(names)}
    for rule in RULES:
        pts = [op(runs[n], rule) for n in names]
        g = dict(delta_pp=stats([p["delta_pp"] for p in pts]), saved_pct=stats([p["saved_pct"] for p in pts]),
                 exit_acc_pct=stats([p["exit_acc_pct"] for p in pts]),
                 n_delta_ge_0=sum(p["delta_pp"] >= 0 for p in pts), n_delta_gt_0=sum(p["delta_pp"] > 0 for p in pts))
        if rule == "D_dynamic_exit":
            g["mean_blocks"] = stats([p["mean_blocks"] for p in pts])
        else:
            g["depths"] = [p["depth"] for p in pts]
            g["runs_at_depth"] = [sum(p["depth"] == d for p in pts) for d in (1, 2, 3, 4)]
        s[rule] = g
    return s


def summarize(e2: dict) -> dict:
    R = e2["runs"]
    assert e2["_header"]["n_runs"] == 18 and not e2["_header"]["missing_or_failed"], e2["_header"]["missing_or_failed"]
    assert all(r["gates"]["status"] in ("PASS", "PASS_NUMERICS") for r in R.values())
    # FLOP model sanity (the pre-fix selector costed CIFAR-100 runs with the CIFAR-10 model)
    for n, r in R.items():
        fl = e2["_header"]["flops"]["c100" if n.startswith("c100_") else "c10"]
        full = (fl["stem_flops"] + fl["arch"]["num_classes"] * sum(fl["block_flops"])) / 1e9
        assert abs(r["gflops_by_depth"][-1] - full) < 1e-9, (n, r["gflops_by_depth"], full)
    c10 = list(C10_ARMS)
    c100 = [f"c100_{v}_seed{s}" for v in C100_VARIANT_FAMILY for s in SEEDS]
    assert sorted(c10 + c100) == sorted(R), sorted(set(R) ^ set(c10 + c100))
    groups = {
        "c10_all": c10,
        "c10_cumulative": [n for n in c10 if family_of(n) == "cumulative"],
        "c10_repaired": [n for n in c10 if family_of(n) != "cumulative"],
        "c100_all": c100,
        **{f"c100_{v}": [f"c100_{v}_seed{s}" for s in SEEDS] for v in C100_VARIANT_FAMILY},
    }
    out = {g: group_summary(R, names) for g, names in groups.items()}

    # cross-check against the selector's own family aggregates
    fam = e2["families"]
    for g, f in (("c10_all", "c10_l4d128_gated_seed42"), ("c100_all", "c100_l4d256_trio"),
                 ("c100_gamma0", "c100_gamma0"), ("c100_gated_k0", "c100_gated_k0"),
                 ("c100_cumulative", "c100_cumulative")):
        assert out[g]["members"] == fam[f]["members"], (g, f)
        for rule in RULES:
            a, b = out[g][rule], fam[f][rule]
            assert math.isclose(a["delta_pp"]["mean"], b["delta_test_pp_mean"], abs_tol=1e-9)
            assert math.isclose(a["delta_pp"]["sd"], b["delta_test_pp_sd"], abs_tol=1e-9)
            assert math.isclose(a["saved_pct"]["mean"], 100 * b["flops_saved_mean"], abs_tol=1e-7)
            assert math.isclose(a["saved_pct"]["sd"], 100 * b["flops_saved_sd"], abs_tol=1e-7)
            assert a["n_delta_ge_0"] == b["n_delta_ge_0"] and a["n_delta_gt_0"] == b["n_delta_gt_0"]
        assert out[g]["T_iso_val"]["depths"] == fam[f]["T_iso_val_depths"]
        assert out[g]["T_argmax_val"]["depths"] == fam[f]["T_argmax_val_depths"]

    # CIFAR-100: repaired minus cumulative, paired by seed (same test images, same seed/split)
    paired = {}
    for v in ("gamma0", "gated_k0"):
        paired[f"{v}_minus_cumulative"] = {}
        for rule in RULES:
            rows = []
            for s in SEEDS:
                a, b = op(R[f"c100_{v}_seed{s}"], rule), op(R[f"c100_cumulative_seed{s}"], rule)
                rows.append(dict(seed=s, saved_pp=a["saved_pct"] - b["saved_pct"], delta_pp=a["delta_pp"] - b["delta_pp"],
                                 exit_acc_pp=a["exit_acc_pct"] - b["exit_acc_pct"],
                                 full_acc_pp=a["full_acc_pct"] - b["full_acc_pct"]))
            paired[f"{v}_minus_cumulative"][rule] = dict(
                per_seed=rows, **{k: stats([r[k] for r in rows]) for k in ("saved_pp", "delta_pp", "exit_acc_pp", "full_acc_pp")})
    out["c100_paired"] = paired

    cl = e2["claim_87_40"]["t3_k0"]
    out["claim_87_40_pct"] = dict(
        arm="c10_gated_adaptive_t3_k0_seed42",
        note="submitted 87.40 = VALIDATION flip-TTA at depth 1 (Block 0); 87.06 = logged full-depth val flip-TTA",
        val_tta_d1_recomputed=100 * cl["val_tta_by_depth_recomputed"][0],
        val_tta_full_logged=100 * cl["val_tta_by_depth_logged"][-1],
        test_no_tta_d1=100 * cl["test_no_tta_by_depth"][0], test_no_tta_full=100 * cl["test_no_tta_by_depth"][-1],
        test_tta_d1=100 * cl["test_tta_by_depth"][0], test_tta_full=100 * cl["test_tta_by_depth"][-1],
        selected_depth={rule: op(R["c10_gated_adaptive_t3_k0_seed42"], rule).get("depth") for rule in RULES[:2]},
        dynamic_exit_mean_blocks=op(R["c10_gated_adaptive_t3_k0_seed42"], "D_dynamic_exit")["mean_blocks"],
    )
    gates = [r["gates"]["status"] for r in R.values()]
    out["gates"] = dict(n_pass=gates.count("PASS"), n_pass_numerics=gates.count("PASS_NUMERICS"),
                        pass_numerics=[n for n, r in R.items() if r["gates"]["status"] == "PASS_NUMERICS"],
                        max_test_delta_examples=max(r["gates"]["test_delta_examples"] for r in R.values()))
    out["flops_saved_pct_by_depth"] = {
        ds: [100 * (1 - (fl["stem_flops"] + fl["arch"]["num_classes"] * sum(fl["block_flops"][:d]))
                    / (fl["stem_flops"] + fl["arch"]["num_classes"] * sum(fl["block_flops"]))) for d in (1, 2, 3, 4)]
        for ds, fl in e2["_header"]["flops"].items()}
    return out
