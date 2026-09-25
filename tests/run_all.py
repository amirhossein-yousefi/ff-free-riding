#!/usr/bin/env python3
"""Run every trainer equivalence test (each in a fresh process) and write one JSON summary.

    CUDA_VISIBLE_DEVICES='' \
    FF_V1_TRAINERS=/path/to/v1/supplement_code/trainers \
    FF_REBUTTAL_TRAINERS=/path/to/mgc_rebuttal \
    FF_F1_TRAINER=/path/to/f1_mgc_block0/trainer.py \
    FF_F2_TRAINER=/path/to/f2_gamma0_s42/trainer.py \
    FF_REFERENCE_RUNS=/path/to/run/root \
    python tests/run_all.py --out tests/equivalence_results.json

Only FF_V1_TRAINERS is needed for the flags-off tests; tests whose reference is not given are
reported as skipped. CPU only (about 10-20 min with 6 threads).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_equiv_cp_fair_hardness_gated.py", "test_equiv_cp_fair_cifar100.py",
         "test_equiv_cp_fair_cifar10.py", "test_equiv_cp_fair_tiny_imagenet.py",
         "test_equiv_cp_fair_core.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "equivalence_results.json"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--logdir", default=None, help="write each test's full stdout/stderr here")
    args = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}, "log_tail": {}}
    rc_all = 0
    for t in TESTS:
        if args.only and not any(o in t for o in args.only):
            continue
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            jpath = f.name
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(HERE, t), jpath], env=env,
                           capture_output=True, text=True)
        if args.logdir:
            os.makedirs(args.logdir, exist_ok=True)
            with open(os.path.join(args.logdir, t.replace(".py", ".log")), "w") as f:
                f.write(p.stdout + "\n---- stderr ----\n" + p.stderr)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith(("[PASS]", "[FAIL]", "[SKIP]"))]
        print(f"== {t} (rc={p.returncode})\n" + "\n".join(ln[:200] for ln in lines), flush=True)
        if p.returncode:
            print(p.stderr[-3000:], flush=True)
        try:
            with open(jpath) as f:
                res = json.load(f)
        except Exception as e:  # crashed before writing
            res = {"crashed": True, "error": repr(e), "returncode": p.returncode,
                   "stderr_tail": p.stderr[-4000:]}
        os.unlink(jpath)
        res["wall_sec"] = round(time.time() - t0, 1)
        res["returncode"] = p.returncode
        summary["results"][t] = res
        rc_all |= p.returncode
    tot = sum(r.get("n_tests", 0) for r in summary["results"].values())
    fail = sum(r.get("n_failed", 0) for r in summary["results"].values())
    skip = sum(r.get("n_skipped", 0) for r in summary["results"].values())
    crashed = [t for t, r in summary["results"].items() if r.get("crashed")]
    summary.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), n_tests=tot, n_failed=fail,
                   n_skipped=skip, n_passed=tot - fail - skip, crashed=crashed,
                   all_passed=(fail == 0 and not crashed and rc_all == 0))
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    print(f"\n== {tot - fail - skip}/{tot - skip} passed, {skip} skipped, {len(crashed)} crashed -> {args.out}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
