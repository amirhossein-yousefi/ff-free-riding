#!/usr/bin/env python3
"""Stage the checkpoint release described in release_manifest.json into one directory, verify every
SHA-256, and write SHA256SUMS. Nothing is uploaded; see RELEASE_CHECKPOINTS.md for the upload step to
https://huggingface.co/Amirhossein75/ff-free-riding.

Three subsets (release_manifest.json, _meta.subsets):

  --subset public    the public Hugging Face release (since 2026-10-26): the 85 files with availability
                     "public_hf" (40.3 GB): stage1_best and stage2_best of the CIFAR-100 trio and MGC
                     arms, the CIFAR-10 squared-hinge, MGC-D128, CP-FAIR and NoMem-L2 runs and the Tiny
                     ImageNet gamma=0 runs; the best.pt of the nine CIFAR-10 BP calibration runs; and
                     stage1_best and stage2_best of the nine text-domain runs. Writes README.md (the model
                     card, release/MODEL_CARD.md), release_manifest.json (the full manifest; each entry's
                     `availability` says whether it is on the Hub) and SHA256SUMS (the public checkpoints).
  --subset minimal   the first release (uploaded 2026-09-25): the 4 seed-42 stage1_best files of the
                     CIFAR-100 gamma=0, gated kappa=0, cumulative and MGC arms (846 MB); same outputs.
  --subset full      every checkpoint listed in the manifest (133 files, about 66 GB), for sharing the
                     "on_request" files; also copies the bundled CIFAR-100 prediction files
                     (preds_cifar100/) to predictions/preds_cifar100/, and SHA256SUMS lists the
                     checkpoints and every file under predictions/. This is the default.

    FF_HOME=/path/to/archive/root python release/stage_release.py --subset public --dest ./hf_release [--copy]

FF_HOME is the directory that contains amir-porjects/ and ff-next/ (every `source` in the manifest is
relative to it). Regular files are hard-linked into --dest (use --copy on another filesystem, or to
keep the staged files independent of the sources); archive members ("archive.zip::member") are
streamed out of the zip, and members of a tarball inside a zip ("archive.zip::run.tar.gz::member")
out of both. Each file is written as <name>.part and renamed only after its SHA-256 check passes. The
script is resumable: files already present with the right SHA-256 are skipped. CPU and disk only.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
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
    ap.add_argument("--subset", choices=("full", "public", "minimal"), default="full",
                    help="public: the Hugging Face release (availability public_hf); minimal: the first "
                         "4-file release; full: every listed file (default)")
    ap.add_argument("--copy", action="store_true", help="copy instead of hard-linking regular files")
    ap.add_argument("--only-group", default=None)
    args = ap.parse_args()
    home = os.environ.get("FF_HOME")
    if not home:
        sys.exit("set FF_HOME to the directory that contains amir-porjects/ and ff-next/")
    man = json.load(open(args.manifest))
    entries = man["checkpoints"]
    if args.subset == "public":
        entries = [e for e in entries if e.get("availability") == "public_hf"]
        want = man["_meta"]["subsets"]["public"]
        if sorted(e["hf_path"] for e in entries) != sorted(want["files"]):
            sys.exit("manifest inconsistent: availability flags differ from _meta.subsets.public.files")
    elif args.subset == "minimal":
        want = set(man["_meta"]["subsets"]["minimal"]["files"])
        entries = [e for e in entries if e["hf_path"] in want]
        if len(entries) != len(want) or any(e.get("availability") != "public_hf" for e in entries):
            sys.exit("manifest inconsistent: _meta.subsets.minimal.files are not all listed as public_hf")
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
            # write under a temporary name; the file takes its final name only after its SHA-256 check,
            # so an interrupted or failed copy never leaves a partial file under a release path
            part = dst + ".part"
            if os.path.exists(part):
                os.remove(part)
            if e["source_kind"] == "file":
                src = os.path.join(home, e["source"])
                if args.copy:
                    shutil.copyfile(src, part)
                else:
                    os.link(src, part)
            elif e["source_kind"] == "archive_member":
                zpath, member = e["source"].split("::", 1)
                with zipfile.ZipFile(os.path.join(home, zpath)) as zf, zf.open(member) as fi, open(part, "wb") as fo:
                    shutil.copyfileobj(fi, fo, CH)
            elif e["source_kind"] == "tar_member":
                # a member of a .tar.gz that is itself a member of a zip: stream both layers
                zpath, tpath, member = e["source"].split("::", 2)
                found = False
                with zipfile.ZipFile(os.path.join(home, zpath)) as zf, zf.open(tpath) as zt, \
                        tarfile.open(fileobj=zt, mode="r|gz") as tf, open(part, "wb") as fo:
                    for ti in tf:
                        if ti.name == member:
                            shutil.copyfileobj(tf.extractfile(ti), fo, CH)
                            found = True
                            break
                if not found:
                    os.remove(part)
                    sys.exit(f"{member} not found in {zpath}::{tpath} ({e['hf_path']})")
            else:
                sys.exit(f"unknown source_kind {e['source_kind']!r} ({e['hf_path']})")
            got = sha256_file(part)
            if got != e["sha256"]:
                bad.append((e["hf_path"], got))
                os.remove(part)  # never leave a file that failed its check in --dest
                print(f"SHA-256 MISMATCH {e['hf_path']} (removed from --dest)", flush=True)
                continue
            os.replace(part, dst)
            print(f"staged {e['hf_path']}", flush=True)
        sums.append(f"{e['sha256']}  {e['hf_path']}")
    n_ckpt = len(sums)
    shutil.copyfile(args.manifest, os.path.join(args.dest, "release_manifest.json"))
    shutil.copyfile(os.path.join(HERE, "MODEL_CARD.md"), os.path.join(args.dest, "README.md"))
    if args.subset == "full":
        # The full bundle also carries the CIFAR-100 trio S2-TTA prediction files; the public
        # release leaves them in the code repository (preds_cifar100/).
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
