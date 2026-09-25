#!/usr/bin/env python3
"""Diff-scoped number tracer for the camera-ready paper.

Every numeral on a line ADDED since commit 0 (389b1fa) in paper/**/*.tex must be traceable to
  (a) a numeric value anywhere in results/*.json, matched at 0-4 decimal places on the absolute
      value (e.g. 0.02498 traces "0.02", "0.025", "0.0250"), or
  (b) a numeral inside a "value" field of results/text_edits_ledger.json (if that file exists), or
  (c) the small, printed WHITELIST below (structural constants, not results).

Added lines = `git diff 389b1fa -- paper/` (tracked files, committed or not) plus every line of
untracked *.tex files under paper/. Comments, \\ref/\\cite/\\label-style arguments, lengths,
table-structure arguments and cross-reference numbers ("Thm. 3.1", "App. A.2", "Fig. 7") are
stripped before extraction. Numerals glued to a letter (L4, D256, S1, R3) are identifiers, not values.

Paths: FF_PAPER_DIR is a git checkout of the paper whose paper/ subtree is traced (the camera-ready
repository); FF_BASELINE_COMMIT (default: the submitted-version commit below) is the diff base;
FF_OUT_DIR/results holds the results JSONs (or point FF_OUT_DIR at a directory whose results/ is
this supplement's metric_summaries/).

Usage:  python analysis/camera_ready/check_numbers.py [--quiet]   exit 1 if anything is untraceable
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

CR = C.PAPER  # git checkout that contains paper/ (FF_PAPER_DIR)
C.BASELINE_COMMIT = __import__('os').environ.get('FF_BASELINE_COMMIT', C.BASELINE_COMMIT)

# ----------------------------------------------------------------------------- whitelist
WHITELIST = {
    # small integers: counts, block/seed indices, thresholds (+-1 pp), n=3 seeds, L4/L8 depths
    **{str(i): "small integer (count / index / threshold)" for i in range(0, 13)},
    "42": "seed", "123": "seed", "456": "seed", "789": "seed (4th CP-FAIR seed)",
    "95": "95% CI level", "97.5": "percentile of a 95% CI", "2.5": "percentile of a 95% CI",
    "16": "block count / width constant", "20": "Stage-2 epochs (C100) / #batches", "32": "MoE experts",
    "64": "width / channels", "128": "width D128", "256": "width D256 / channels", "384": "stem channels",
    "512": "batch size", "180": "Stage-1 epochs (D128)", "190": "total epochs D128", "362": "Stage-1 epochs (D256)",
    "382": "total epochs D256", "100": "classes / percent", "200": "classes (Tiny)", "1000": "k", "5000": "bootstrap resamples / val size",
    "10000": "test-set size", "400": "probe feature dimension (4 blocks x 100 classes)", "30000": "pooled paired test predictions (3 x 10k)", "50000": "train-set size",
    "2024": "year", "2025": "year", "2026": "year",
}

_ARGS = r"\s*(?:\[[^\]]*\])*\s*(?:\([^)]*\))?\s*"
# commands whose every argument is structural (labels, keys, paths, specs, lengths)
_STRIP_ALL = ("ref|eqref|autoref|cref|Cref|pageref|label|cite[pt]?\\*?|citealp|citeauthor|citeyear|input|include|"
              "includegraphics|url|bibitem|usepackage|documentclass|begin|end|setlength|addtolength|hspace\\*?|vspace\\*?|"
              "rule|cmidrule|cline|setcounter|addtocounter|definecolor|fontsize|linespread|arraystretch|tabcolsep")
# commands whose FIRST n arguments are structural; later arguments carry text and are kept
_STRIP_N = {"multicolumn": 2, "multirow": 2, "resizebox": 2, "scalebox": 1, "textcolor": 1, "color": 1, "href": 1,
            "newcommand": 1, "renewcommand": 1, "providecommand": 1, "hyperref": 0}
STRIP_PATTERNS = [r"\\(?:" + _STRIP_ALL + r")" + _ARGS + r"(?:\{[^{}]*\})*"]
STRIP_PATTERNS += [r"\\" + c + _ARGS + r"(?:\{[^{}]*\}){" + str(n) + "}" for c, n in _STRIP_N.items()]
STRIP_PATTERNS += [
    # float placement
    r"\[(?:!?[htbpH]+)\]",
    # dimensions
    r"[-+]?\d*\.?\d+\s*(?:pt|em|ex|cm|mm|in|bp|sp|mu)\b",
    r"[-+]?\d*\.?\d+\s*\\(?:textwidth|linewidth|columnwidth|textheight|baselineskip)",
    # cross references by name: Thm. 3.1, Prop.~3.2, Sec. 4, App. A.1, Fig. 7(b), Table 2, Eq. (3), rows 21-23
    r"(?:Thm|Theorem|Prop|Proposition|Lemma|Cor|Corollary|Def|Definition|Remark|Sec|Section|Secs|App|Appendix|Fig|Figs|"
    r"Figure|Figures|Tab|Table|Tables|Eq|Eqs|Equation|Alg|Algorithm|Step|line|lines|row|rows)\.?(?:~|\s)*\(?[A-Z]?\.?\d+(?:\.\d+)*"
    r"(?:\([a-z]\))?\)?(?:\s*(?:--|-|–|and|,)\s*[A-Z]?\d+(?:\.\d+)*)*",
    r"§\s*\d+(?:\.\d+)*",
]
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)")


def strip_comment(line: str) -> str:
    out, i = [], 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2]); i += 2; continue
        if ch == "%":
            break
        out.append(ch); i += 1
    return "".join(out)


def normalize(text: str) -> str:
    text = text.replace("{,}", "").replace("\\,", " ").replace("~", " ")
    text = re.sub(r"(?<=\d),(?=000\b)", "", text)             # 30,000 -> 30000 (but "42,123,456" is a list)
    for pat in STRIP_PATTERNS:
        text = re.sub(pat, " ", text)
    return text


def harvest_json(o, allowed: set):
    if isinstance(o, dict):
        for v in o.values():
            harvest_json(v, allowed)
    elif isinstance(o, list):
        for v in o:
            harvest_json(v, allowed)
    elif isinstance(o, bool) or o is None:
        return
    elif isinstance(o, (int, float)):
        a = abs(float(o))
        if isinstance(o, int):
            allowed.add(str(abs(o)))
        for nd in range(0, 5):
            allowed.add(f"{a:.{nd}f}")


def ledger_values(o, allowed: set):
    """numerals inside any 'value' field of text_edits_ledger.json (string or number)."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "value":
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    harvest_json(v, allowed)
                elif isinstance(v, str):
                    for m in NUM_RE.finditer(normalize(v)):
                        allowed.add(m.group(1))
                elif isinstance(v, list):
                    for x in v:
                        ledger_values({"value": x}, allowed)
            else:
                ledger_values(v, allowed)
    elif isinstance(o, list):
        for v in o:
            ledger_values(v, allowed)


def numerals(raw: str):
    return [m.group(1) for m in NUM_RE.finditer(normalize(strip_comment(raw)))]


def added_lines():
    """{file: [(new_line_no, text, carried)]} for lines added since the baseline commit.

    `carried` is a multiset (dict numeral->count) of numerals present on the REMOVED lines of the
    same diff hunk: a numeral that merely survives an edit of its line is not new.
    """
    out = defaultdict(list)
    diff = subprocess.run(["git", "-C", str(CR), "diff", "--no-color", "--unified=0", C.BASELINE_COMMIT, "--", "paper/"],
                          capture_output=True, text=True, check=True).stdout
    cur, ln = None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            f = line[4:].strip()
            cur = f[2:] if f.startswith("b/") else None
            if cur and not cur.endswith(".tex"):
                cur = None
            continue
        if line.startswith("--- "):
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            ln = int(m.group(1)); carried = defaultdict(int); continue
        if cur is None:
            continue
        if line.startswith("-"):
            for n in numerals(line[1:]):
                carried[n] += 1
        elif line.startswith("+"):
            out[cur].append((ln, line[1:], carried)); ln += 1
        elif line.startswith(" "):
            ln += 1
    untracked = subprocess.run(["git", "-C", str(CR), "ls-files", "--others", "--exclude-standard", "paper/"],
                               capture_output=True, text=True, check=True).stdout.split()
    for f in untracked:
        if f.endswith(".tex"):
            for i, t in enumerate(open(CR / f, encoding="utf-8", errors="replace").read().splitlines(), 1):
                out[f].append((i, t, defaultdict(int)))
    return out


def main():
    quiet = "--quiet" in sys.argv
    strict = "--strict" in sys.argv          # also check numerals carried over on edited lines
    allowed: set = set()
    n_json = 0
    for p in sorted(C.RESULTS.glob("*.json")):
        try:
            o = json.load(open(p))
        except Exception as e:  # a JSON being written concurrently by another process
            print(f"WARN: could not parse {p.name}: {e}")
            continue
        n_json += 1
        harvest_json(o, allowed)
        if p.name == "text_edits_ledger.json":
            ledger_values(o, allowed)

    if not quiet:
        print(f"check_numbers: baseline {C.BASELINE_COMMIT}; traced against {n_json} results/*.json files "
              f"({len(allowed)} admissible numeral strings)")
        print("WHITELIST (structural constants, never results):")
        for k in sorted(WHITELIST, key=lambda x: float(x)):
            print(f"  {k:>6}  {WHITELIST[k]}")

    bad = []
    n_checked = n_carried = 0
    for f, lines in sorted(added_lines().items()):
        for ln, raw, carried in lines:
            for s in numerals(raw):
                if not strict and carried.get(s, 0) > 0:
                    carried[s] -= 1          # same numeral was on the line(s) this edit replaced
                    n_carried += 1
                    continue
                n_checked += 1
                if s in WHITELIST or s in allowed:
                    continue
                bad.append((f, ln, s, raw.strip()[:140]))

    print(f"numerals checked on added lines: {n_checked} (+{n_carried} carried over unchanged from the replaced "
          f"lines{'' if not strict else ', checked too (--strict)'}); untraceable: {len(bad)}")
    if bad:
        print("UNTRACEABLE NUMERALS (file:line  numeral  | context):")
        for f, ln, s, ctx in bad:
            print(f"  {f}:{ln}  {s}  | {ctx}")
        sys.exit(1)
    print("check_numbers: all added numerals trace to results/*.json or the whitelist")


if __name__ == "__main__":
    main()
