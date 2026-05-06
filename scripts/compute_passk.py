#!/usr/bin/env python3
"""Compute pass@k statistics from a single generation run.

For each (model, size) cell, ``pass@k`` is the geometric mean of the
**best-of-k** speedup over each task that has at least one correct sample.
A run dir is expected to contain ``eval_results_*.json`` files with
``sample_id`` 0..k-1 already evaluated.

This script does NOT merge results from different generation runs (e.g.
April pass@1 with May pass@2). Cross-batch merging is methodologically
risky --- the model's behavior, prompt scaffolding, and timing harness
may all differ between runs --- so we leave it out of the public
release. Run a single ``generate --samples k`` followed by ``eval``
and feed that one run dir to this script.

Usage::

  # 1) generate k samples per (model, task) into one run
  python3 run.py generate --models <model> --levels 3 --samples 3
  python3 run.py eval --run <run_dir> --sizes small,medium,large

  # 2) compute pass@k aggregates
  python3 scripts/compute_passk.py --runs <run_dir> [<run_dir2> ...] [--k 3]
"""
import argparse, json, glob, math, os, sys
from collections import defaultdict


SIZES = ("small", "medium", "large")


def load_run(run_dir):
    """Return ((task, sample_id, size) -> speedup) for all correct samples in run_dir."""
    out = {}
    for fp in sorted(glob.glob(f"{run_dir}/eval_results_*.json")):
        try:
            data = json.load(open(fp))
        except Exception:
            continue
        for k, v in data.items():
            if not isinstance(v, dict) or not v.get("correct"):
                continue
            t = v.get("task_id", "")
            sid = v.get("sample_id", 0)
            b = v.get("benchmark") or {}
            sz = b.get("size_name")
            sp = b.get("speedup_e2e")
            if not sz or sp is None or sp <= 0:
                continue
            out[(t, sid, sz)] = max(sp, out.get((t, sid, sz), 0.0))
    return out


def best_per_task(samples, k):
    """Group (task, sid, sz) -> sp by (task, sz) and keep best over the first k samples."""
    by_task_sz = defaultdict(list)
    for (t, sid, sz), sp in samples.items():
        if sid < k:
            by_task_sz[(t, sz)].append(sp)
    return {(t, sz): max(sps) for (t, sz), sps in by_task_sz.items()}


def gm(values):
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more run dirs (each will be reported separately).")
    ap.add_argument("--k", type=int, default=None,
                    help="Restrict to first k samples (default: use all samples found).")
    args = ap.parse_args()

    print(f"{'run':<60} {'size':<7} {'n':>4} {'GM':>9}")
    print("-" * 86)
    summaries = {}
    for run in args.runs:
        if not os.path.isdir(run):
            print(f"  [skip] {run} not a directory", file=sys.stderr)
            continue
        samples = load_run(run)
        if not samples:
            print(f"  [skip] {run}: no correct samples", file=sys.stderr)
            continue
        max_sid = max(sid for (_, sid, _) in samples) + 1
        k = args.k if args.k else max_sid
        best = best_per_task(samples, k)
        per_size = defaultdict(list)
        for (t, sz), sp in best.items():
            per_size[sz].append(sp)
        run_summary = {}
        for sz in SIZES:
            if not per_size.get(sz):
                continue
            g = gm(per_size[sz])
            print(f"  {os.path.basename(run):<58} {sz:<7} {len(per_size[sz]):>4} {g:>8.2f}x")
            run_summary[sz] = {"n": len(per_size[sz]), "gm": g}
        summaries[run] = {"k": k, "by_size": run_summary}

    # Save aggregate
    os.makedirs("runs/reports", exist_ok=True)
    out_fp = "runs/reports/passk_summary.json"
    with open(out_fp, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved: {out_fp}", file=sys.stderr)


if __name__ == "__main__":
    main()
