#!/usr/bin/env python3
"""Check that repro_manifest.csv's commands reproduce the configurations of the real runs. CPU only.

For every manifest row that names a camera-ready trainer, an `env` and `reference_runs`, this
script runs the trainer's real main() with that environment (plus FF_SEED) until it has written
config.json (no data is loaded, no training step is taken) and compares the result with the
reference run's own config.json. The only differences allowed are `device` and configuration
keys that the camera-ready trainer added (absent from the older run's config.json).

One functionally inert difference is also accepted and reported: block_curr_depth_slope when
block_curr_lambda is 0 in both configurations (the depth-scaled weight is then 0 at every block;
the LCFF D256 run recorded slope 0, and the trainer has no environment hook for it).

The `env` column format:  "KEY=VAL KEY=VAL | @42 KEY=VAL | @123,456 KEY=VAL"
(the part before the first "|" applies to every seed; "@<seeds>" parts only to those seeds).

    CUDA_VISIBLE_DEVICES='' FF_REFERENCE_HOME=/path/to/FF_HOME python tests/check_manifest_commands.py \
        [--out manifest_check.json]
FF_REFERENCE_HOME is the directory that contains amir-porjects/ (the run-log archive root).
"""
import argparse
import csv
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equiv_utils as U  # noqa: E402

CHECKED_TRAINERS = {"trainers/cp_fair_cifar10.py", "trainers/cp_fair_cifar100.py",
                    "trainers/cp_fair_hardness_gated.py"}
CR_KEYS = {"use_mgc", "mgc_c0", "mgc_rho", "mgc_eps", "mgc_keep_l0_aux", "curr_aux_off",
           "barrier_type", "barrier_margin"}


def parse_env(spec, seed):
    env = {}
    parts = [p.strip() for p in spec.split("|")]
    for i, p in enumerate(parts):
        if not p:
            continue
        toks = p.split()
        if toks[0].startswith("@"):
            seeds = {int(s) for s in toks[0][1:].split(",")}
            if seed not in seeds:
                continue
            toks = toks[1:]
        for t in toks:
            k, v = t.split("=", 1)
            env[k] = v
    env["FF_SEED"] = str(seed)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(U.SUPP, "repro_manifest.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    home = os.environ.get("FF_REFERENCE_HOME")
    rec = U.Recorder(os.path.basename(__file__))
    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    rec.out["manifest_rows"] = len(rows)
    for i, r in enumerate(rows, 1):
        if r["trainer"] not in CHECKED_TRAINERS or not r["reference_runs"]:
            continue
        for item in r["reference_runs"].split(";"):
            seed, run = item.strip().split(":", 1)
            seed = int(seed)
            name = f"row{i:02d}_{r['paper_table']}|{r['row_name']}|seed{seed}"
            if not home:
                rec.skip(name, "FF_REFERENCE_HOME not set")
                continue
            path = os.path.join(home, run, "config.json")
            if not os.path.exists(path):
                rec.skip(name, "reference run config.json not found")
                continue
            cfg, _ = U.planned_config(os.path.join(U.SUPP, r["trainer"]), parse_env(r["env"], seed))
            cfg["device"] = "cuda"
            ref = json.load(open(path))
            d = U.diff_configs(ref, cfg)
            bad = {k: v for k, v in d.items() if not (k in CR_KEYS and v[0] == "<MISSING>")}
            inert = {}
            if set(bad) == {"block_curr_depth_slope"} and ref.get("block_curr_lambda") == 0.0 \
                    and cfg.get("block_curr_lambda") == 0.0:
                inert, bad = bad, {}
            rec.record(name, not bad, {"diff_vs_run_config": d, "unexpected": bad,
                                       "inert_diff_accepted": inert, "run": os.path.basename(run)})
    rc = rec.finish(args.out)
    U.cleanup()
    return rc


if __name__ == "__main__":
    sys.exit(main())
