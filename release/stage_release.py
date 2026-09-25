#!/usr/bin/env python3
"""Stage the checkpoint release described in release_manifest.json into one directory, verify every
SHA-256, and write SHA256SUMS. Nothing is uploaded; see RELEASE_CHECKPOINTS.md for the upload step to
https://huggingface.co/Amirhossein75/ff-free-riding.

Two subsets (release_manifest.json, _meta.subsets):

  --subset minimal   the public Hugging Face release: the 4 files with availability "public_hf"
                     (seed-42 stage1_best of the CIFAR-100 gamma=0, gated kappa=0, cumulative and MGC
                     arms of main-text Table 3; 846 MB). Writes README.md (the model card,
                     release/MODEL_CARD.md), release_manifest.json (the full manifest; each entry's
                     `availability` says whether it is on the Hub) and SHA256SUMS (the 4 checkpoints).
  --subset full      every result-bearing checkpoint (98 files, about 55 GB), for sharing the
                     "on_request" files; also copies the bundled CIFAR-100 prediction files
                     (preds_cifar100/) to predictions/preds_cifar100/, and SHA256SUMS lists the
                     checkpoints and every file under predictions/. This is the default.

    FF_HOME=/path/to/archive/root python release/stage_release.py --subset minimal --dest ./hf_release [--copy]

FF_HOME is the directory that contains amir-porjects/ and ff-next/ (every `source` in the manifest is
relative to it). Regular files are hard-linked into --dest (use --copy on another filesystem, or to
keep the staged files independent of the sources); archive members ("archive.zip::member") are
streamed out of the zip. The script is resumable: files already present with the right SHA-256 are
skipped. CPU and disk only.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CH = 8 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CH), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(HERE), "release_manifest.json"))
    ap.add_argument("--dest", required=True)
    ap.add_argument("--subset", choices=("full", "minimal"), default="full",
                    help="minimal: the 4 public Hugging Face files; full: all 98 files (default)")
    ap.add_argument("--copy", action="store_true", help="copy instead of hard-linking regular files")
    ap.add_argument("--only-group", default=None)
    args = ap.parse_args()
    home = os.environ.get("FF_HOME")
    if not home:
        sys.exit("set FF_HOME to the directory that contains amir-porjects/ and ff-next/")
    man = json.load(open(args.manifest))
    entries = man["checkpoints"]
    if args.subset == "minimal":
        entries = [e for e in entries if e.get("availability") == "public_hf"]
        want = man["_meta"]["subsets"]["minimal"]
        if sorted(e["hf_path"] for e in entries) != sorted(want["files"]):
            sys.exit("manifest inconsistent: availability flags differ from _meta.subsets.minimal.files")
    os.makedirs(args.dest, exist_ok=True)
    sums, bad = [], []
    for e in entries:
        if args.only_group and e["group"] != args.only_group:
            continue
        dst = os.path.join(args.dest, e["hf_path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not (os.path.exists(dst) and os.path.getsize(dst) == e["size_bytes"] and sha256_file(dst) == e["sha256"]):
            if os.path.exists(dst):
                os.remove(dst)
            if e["source_kind"] == "file":
                src = os.path.join(home, e["source"])
                if args.copy:
                    shutil.copyfile(src, dst)
                else:
                    os.link(src, dst)
            else:
                zpath, member = e["source"].split("::", 1)
                with zipfile.ZipFile(os.path.join(home, zpath)) as zf, zf.open(member) as fi, open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo, CH)
            got = sha256_file(dst)
            if got != e["sha256"]:
                bad.append((e["hf_path"], got))
                print(f"SHA-256 MISMATCH {e['hf_path']}", flush=True)
                continue
            print(f"staged {e['hf_path']}", flush=True)
        sums.append(f"{e['sha256']}  {e['hf_path']}")
    n_ckpt = len(sums)
    shutil.copyfile(args.manifest, os.path.join(args.dest, "release_manifest.json"))
    shutil.copyfile(os.path.join(HERE, "MODEL_CARD.md"), os.path.join(args.dest, "README.md"))
    if args.subset == "full":
        # The full bundle also carries the CIFAR-100 trio S2-TTA prediction files; the public
        # (minimal) release leaves them in the code repository (preds_cifar100/).
        preds = os.path.join(os.path.dirname(HERE), "preds_cifar100")
        if os.path.isdir(preds):
            shutil.copytree(preds, os.path.join(args.dest, "predictions", "preds_cifar100"), dirs_exist_ok=True)
        # SHA256SUMS also pins every file under predictions/ (hashed from the staged copies).
        pred_root = os.path.join(args.dest, "predictions")
        for root, dirs, files in os.walk(pred_root):
            dirs.sort()
            for name in sorted(files):
                p = os.path.join(root, name)
                rel = os.path.relpath(p, args.dest).replace(os.sep, "/")
                sums.append(f"{sha256_file(p)}  {rel}")
    with open(os.path.join(args.dest, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    print(f"subset={args.subset}: {n_ckpt} checkpoints verified, {len(sums) - n_ckpt} prediction files hashed; "
          f"{len(bad)} mismatches -> {args.dest}/SHA256SUMS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
