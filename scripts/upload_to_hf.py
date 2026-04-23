#!/usr/bin/env python3
"""
Upload ORBench data to HuggingFace Hub as a dataset repo.

Usage:
  # First-time setup:
  pip install huggingface_hub
  huggingface-cli login        # needs a write-access token from hf.co/settings/tokens

  # Upload:
  python3 scripts/upload_to_hf.py --repo-id <org>/orbench-data

  # Only upload one size:
  python3 scripts/upload_to_hf.py --repo-id <org>/orbench-data --size small
"""
import argparse, os, glob, tarfile, sys, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files we distribute (exclude transient output.txt, timing.json, etc.)
DATA_FILES = ("input.bin", "expected_output.txt", "cpu_time_ms.txt", "requests.txt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_size(size: str, out_dir: Path) -> Path:
    """Pack one size (small/medium/large) into a tar.gz."""
    out_path = out_dir / f"{size}.tar.gz"
    print(f"[pack] {size} → {out_path}")
    added = 0
    with tarfile.open(out_path, "w:gz", compresslevel=6) as tar:
        for task_dir in sorted((ROOT / "tasks").iterdir()):
            if not task_dir.is_dir():
                continue
            data_dir = task_dir / "data" / size
            if not data_dir.is_dir():
                continue
            for fname in DATA_FILES:
                f = data_dir / fname
                if f.exists():
                    tar.add(f, arcname=f"{task_dir.name}/data/{size}/{fname}")
                    added += 1
    print(f"  added {added} files, size: {out_path.stat().st_size / 1e6:.1f} MB")
    return out_path


def build_manifest(out_dir: Path, tarballs: list) -> Path:
    """Write a manifest.json with SHA256 + task list + sizes."""
    entries = {}
    for tar in tarballs:
        entries[tar.name] = {
            "size_bytes": tar.stat().st_size,
            "sha256": sha256(tar),
        }
    # Also include per-task metadata
    tasks = {}
    for tj_path in sorted((ROOT / "tasks").glob("*/task.json")):
        tid = tj_path.parent.name
        with open(tj_path) as f: tj = json.load(f)
        tasks[tid] = {
            "category": tj.get("category"),
            "difficulty": tj.get("difficulty"),
            "interface_mode": tj.get("interface_mode", "compute_only"),
            "input_sizes": tj.get("input_sizes"),
        }
    manifest = {
        "version": "1.0",
        "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "tarballs": entries,
        "tasks": tasks,
        "num_tasks": len(tasks),
    }
    p = out_dir / "manifest.json"
    with open(p, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {p}")
    return p


def upload(repo_id: str, files: list, private: bool = False):
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"[warn] create_repo: {e}")

    for f in files:
        print(f"[upload] {f.name} ({f.stat().st_size/1e6:.1f} MB) → {repo_id}")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    # Upload README if present
    readme = ROOT / "scripts" / "data_readme_template.md"
    if readme.exists():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="HuggingFace dataset repo id, e.g. yourname/orbench-data")
    ap.add_argument("--size", choices=["small", "medium", "large", "all"],
                    default="all")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--pack-only", action="store_true",
                    help="Only build tarballs, don't upload")
    ap.add_argument("--out-dir", default=str(ROOT / "build"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = ["small", "medium", "large"] if args.size == "all" else [args.size]
    tarballs = [pack_size(s, out_dir) for s in sizes]
    manifest = build_manifest(out_dir, tarballs)

    if args.pack_only:
        print(f"\nDone (pack-only). Files in: {out_dir}")
        for t in tarballs: print(f"  {t} ({t.stat().st_size/1e6:.1f} MB)")
        return

    upload(args.repo_id, tarballs + [manifest], private=args.private)
    print(f"\n✅ Done. Repo: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
