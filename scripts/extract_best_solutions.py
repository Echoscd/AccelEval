#!/usr/bin/env python3
"""
Stage 0 of the strategy-transfer pipeline.

For each task in the medium-size leaderboard, pick the winner model
(argmax speedup_e2e among compiled+correct samples) and copy its
solution.cu into runs/strategy_transfer/best/<task>/.

Usage:
    python3 scripts/extract_best_solutions.py
    python3 scripts/extract_best_solutions.py --size medium --out runs/strategy_transfer/best
"""
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Canonical Stage-1 runs (medium + large are in these dirs)
RUNS = {
    "gemini-3.1-pro-preview-openrouter": "gemini-3.1-pro-preview-openrouter_l3_20260421_1712",
    "claude-opus-4.6-openrouter":       "claude-opus-4.6-openrouter_l3_20260421_1905",
    "openai/gpt-5.4":                    "openai/gpt-5.4_l3_20260421_1905",
    "qwen/qwen3.6-plus":                 "qwen/qwen3.6-plus_l3_20260421_1905",
    "kimi-k2.5-openrouter":              "kimi-k2.5-openrouter_l3_20260421_1905",
    "glm-5.1-openrouter":                "glm-5.1-openrouter_l3_20260421_1905",
    "deepseek-v3.2-openrouter":          "deepseek-v3.2-openrouter_l3_20260421_1905",
}


def latest_per_task(run_dir: Path, size: str) -> dict:
    """
    Scan ALL eval_results_*.json in run_dir and return {task_id: record}
    keeping the most recent record per (task_id, size) by file mtime.

    This correctly handles the case where some tasks were re-evaluated later
    (e.g., after medium data was regenerated) — fresher entries win.
    """
    per_task: dict[str, tuple[float, dict]] = {}  # task -> (mtime, record)
    for ev in run_dir.glob("eval_results_*.json"):
        try:
            mt = ev.stat().st_mtime
            d = json.loads(ev.read_text())
            res = d.get("results", d)
            for key, v in res.items():
                if not isinstance(v, dict):
                    continue
                bm = v.get("benchmark") or {}
                if bm.get("size_name") != size:
                    continue
                task_id = v.get("task_id") or key.rsplit("_sample_", 1)[0]
                prev = per_task.get(task_id)
                if prev is None or mt > prev[0]:
                    per_task[task_id] = (mt, v)
        except Exception:
            continue
    return {t: rec for t, (_, rec) in per_task.items()}


def active_tasks() -> set:
    """Tasks with a current task.json in tasks/ (excludes _deprecated)."""
    active = set()
    tdir = ROOT / "tasks"
    for t in tdir.iterdir():
        if t.is_dir() and (t / "task.json").exists():
            active.add(t.name)
    return active


def collect(size: str) -> dict:
    """Return {task_id: [(model, speedup, cu_path), ...]}, sorted desc by speedup."""
    active = active_tasks()
    by_task: dict[str, list] = {}
    for model, run_name in RUNS.items():
        run_dir = ROOT / "runs" / run_name
        if not run_dir.exists():
            print(f"  skip {model}: run dir missing")
            continue
        records = latest_per_task(run_dir, size)
        if not records:
            print(f"  skip {model}: no {size} eval")
            continue
        for task_id, v in records.items():
            if task_id not in active:
                continue  # deprecated / removed task
            if not (v.get("compiled") and v.get("correct")):
                continue
            bm = v.get("benchmark") or {}
            sp = bm.get("speedup_e2e")
            if not sp or sp <= 0:
                continue
            sample_id = v.get("sample_id", 0)
            cu = run_dir / task_id / f"sample_{sample_id}.cu"
            if not cu.exists():
                continue
            by_task.setdefault(task_id, []).append((model, float(sp), str(cu)))
    for t in by_task:
        by_task[t].sort(key=lambda x: -x[1])
    return by_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--out", default="runs/strategy_transfer/best")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    by_task = collect(args.size)
    manifest = {}

    for task_id, rank in sorted(by_task.items()):
        winner_model, winner_sp, winner_cu = rank[0]
        task_out = out / task_id
        task_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(winner_cu, task_out / "solution.cu")
        manifest[task_id] = {
            "winner_model": winner_model,
            "winner_speedup": winner_sp,
            "source": winner_cu,
            "rank_table": [
                {"model": m, "speedup": round(s, 2)} for m, s, _ in rank
            ],
            "n_correct": len(rank),
        }
        if not args.quiet:
            print(f"  {task_id:40s} {winner_model:35s} {winner_sp:>8.1f}x  (n={len(rank)})")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps({
        "size": args.size,
        "tasks": manifest,
    }, indent=2))
    print(f"\n  {len(manifest)} tasks → {out}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
