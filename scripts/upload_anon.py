#!/usr/bin/env python3
"""
Anonymous one-shot upload to a HuggingFace dataset repo.

Uploads:
  - README.md                     (dataset card)
  - tasks.parquet, tasks.csv      (42-task manifest, viewer-renderable)
  - small.tar.gz / medium.tar.gz / large.tar.gz   (re-packed from current tasks/)
  - manifest.json                 (SHA256 of every tarball + task metadata)

Usage:
  # In your shell (token already set via huggingface-cli login OR HF_TOKEN env):
  python3 scripts/upload_anon.py --repo-id Accel-Eval/AccelEval-data
  # Add --skip-pack to reuse existing build/<size>.tar.gz (skip re-tarring)
  # Add --sizes small to only upload one scale
"""
import argparse, os, json, hashlib, tarfile, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILES = ("input.bin", "expected_output.txt", "cpu_time_ms.txt", "requests.txt")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(size, out_dir):
    out = out_dir / f"{size}.tar.gz"
    print(f"[pack] {size} → {out}")
    n = 0
    with tarfile.open(out, "w:gz", compresslevel=6) as t:
        for task_dir in sorted((ROOT / "tasks").iterdir()):
            if not task_dir.is_dir(): continue
            d = task_dir / "data" / size
            if not d.is_dir(): continue
            for f in DATA_FILES:
                p = d / f
                if p.exists():
                    t.add(p, arcname=f"{task_dir.name}/data/{size}/{f}")
                    n += 1
    print(f"  added {n} files, {out.stat().st_size/1e6:.1f} MB")
    return out


def build_manifest(out_dir, tarballs):
    entries = {}
    for tb in tarballs:
        entries[tb.name] = {
            "size_bytes": tb.stat().st_size,
            "sha256": sha256(tb),
        }
    tasks = {}
    for tj in sorted((ROOT / "tasks").glob("*/task.json")):
        with open(tj) as f: t = json.load(f)
        tasks[tj.parent.name] = {
            "category": t.get("category"),
            "difficulty": t.get("difficulty"),
            "interface_mode": t.get("interface_mode", "compute_only"),
            "input_sizes": t.get("input_sizes"),
            "tags": t.get("tags") or [],
        }
    m = {
        "version": "1.0",
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "num_tasks": len(tasks),
        "tarballs": entries,
        "tasks": tasks,
    }
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(m, indent=2))
    print(f"[manifest] {p}  ({p.stat().st_size} bytes, {len(tasks)} tasks)")
    return p


def upload(repo_id, files, private):
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"[repo] {repo_id} ready")
    except Exception as e:
        print(f"[repo] note: {e}")
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"[upload] {f.name}  ({size_mb:.1f} MB) → {repo_id}")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="e.g. Accel-Eval/AccelEval-data")
    ap.add_argument("--sizes", default="small,medium,large",
                    help="comma list, default: small,medium,large")
    ap.add_argument("--private", action="store_true",
                    help="create as private (you can flip to public later)")
    ap.add_argument("--skip-pack", action="store_true",
                    help="don't re-pack; reuse existing build/<size>.tar.gz")
    ap.add_argument("--metadata-only", action="store_true",
                    help="upload README + tasks.parquet/csv only, skip tarballs")
    args = ap.parse_args()

    out_dir = ROOT / "build"
    out_dir.mkdir(exist_ok=True)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]

    # 1) tarballs
    tarballs = []
    if not args.metadata_only:
        for sz in sizes:
            target = out_dir / f"{sz}.tar.gz"
            if args.skip_pack and target.exists():
                print(f"[pack] reusing {target}")
            else:
                target = pack(sz, out_dir)
            tarballs.append(target)

    # 2) manifest (always recompute so SHA256 reflects what we're uploading)
    manifest = build_manifest(out_dir, tarballs) if tarballs else None

    # 3) what to upload
    to_upload = []
    for name in ("tasks.parquet", "tasks.csv", "README.md"):
        p = out_dir / name
        if p.exists(): to_upload.append(p)
    to_upload += tarballs
    if manifest: to_upload.append(manifest)

    print()
    print(f"=== Uploading {len(to_upload)} files to {args.repo_id} ===")
    for f in to_upload:
        print(f"  {f.name:<24} {f.stat().st_size/1e6:>8.1f} MB")
    print()

    upload(args.repo_id, to_upload, args.private)
    print(f"\n✅ Done. https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
