#!/usr/bin/env python3
"""
Consolidate all eval_results across multiple run dirs into a single canonical
JSON: keep the most recent (task_id, size, model) triple.

Output:
  runs/reports/acceleval_consolidated_<timestamp>.json
     {
       "generated_at": "...",
       "active_tasks": [...],
       "models": { model_name: canonical_run_dir },
       "cpu_baselines": { size: { task: ms } },
       "records": { size: { task: { model: record_dict_including_benchmark } } },
     }

Usage:
    python3 scripts/consolidate_eval_data.py
"""
import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUNS_L3 = {
    "Gemini 3.1 Pro":  "gemini-3.1-pro-preview-openrouter_l3_20260421_1712",
    "Claude Opus 4.6": "claude-opus-4.6-openrouter_l3_20260421_1905",
    "GPT-5.4":         "openai/gpt-5.4_l3_20260421_1905",
    "Qwen 3.6 Plus":   "qwen/qwen3.6-plus_l3_20260421_1905",
    "Kimi K2.5":       "kimi-k2.5-openrouter_l3_20260421_1905",
    "GLM 5.1":         "glm-5.1-openrouter_l3_20260421_1905",
    "DeepSeek V3.2":   "deepseek-v3.2-openrouter_l3_20260421_1905",
    "DeepSeek V4 Pro": "deepseek-v4-pro-openrouter_l3_20260424_1706",
}


def active_tasks():
    tdir = ROOT / "tasks"
    return sorted(
        t.name for t in tdir.iterdir()
        if t.is_dir() and (t / "task.json").exists()
    )


def _filename_ts(p: Path) -> int:
    """Parse YYYYMMDD_HHMMSS from eval_results_<ts>.json filename. Falls back to
    mtime when the filename does not match. Using the filename timestamp avoids
    the foot-gun where editing a file (e.g. relabeling) bumps the mtime even
    though the underlying eval is older."""
    import re as _re
    m = _re.match(r"eval_results_(\d{8})_(\d{6})", p.stem)
    if m:
        return int(m.group(1) + m.group(2))
    return int(p.stat().st_mtime)


def latest_records(run_dir: Path, active: set) -> dict:
    """Return {(task, size): record} with most recent record per key, where
    'most recent' is judged by the timestamp embedded in the filename."""
    lat = {}
    for ev in run_dir.glob("eval_results_*.json"):
        ts = _filename_ts(ev)
        try:
            d = json.loads(ev.read_text())
            res = d.get("results", d)
            for k, v in res.items():
                if not isinstance(v, dict):
                    continue
                tid = v.get("task_id") or k.rsplit("_sample_", 1)[0]
                if tid not in active:
                    continue
                bm = v.get("benchmark") or {}
                size = bm.get("size_name")
                if not size:
                    # Compile-fail record -- no size tag; emit entries for all 3 sizes
                    for s in ("small", "medium", "large"):
                        key = (tid, s)
                        if key not in lat or ts > lat[key][0]:
                            lat[key] = (ts, v)
                    continue
                key = (tid, size)
                if key not in lat or ts > lat[key][0]:
                    lat[key] = (ts, v)
        except Exception:
            continue
    return {k: v for k, (_, v) in lat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    active = active_tasks()
    print(f"Active tasks: {len(active)}")

    # Per-model latest records
    records = {s: {t: {} for t in active} for s in ("small", "medium", "large")}
    cpu_baselines = {s: {} for s in ("small", "medium", "large")}

    for mname, run_name in RUNS_L3.items():
        rd = ROOT / "runs" / run_name
        if not rd.exists():
            print(f"  [warn] missing run dir: {run_name}")
            continue
        lat = latest_records(rd, set(active))
        for (t, s), rec in lat.items():
            records[s][t][mname] = rec
            bm = rec.get("benchmark") or {}
            cpu = bm.get("cpu_baseline_ms")
            if cpu and cpu > 0 and t not in cpu_baselines[s]:
                cpu_baselines[s][t] = cpu
        print(f"  {mname:20s}  records merged from {run_name}")

    # Fill CPU baselines from disk for tasks where no record has them
    for t in active:
        for s in ("small", "medium", "large"):
            if t not in cpu_baselines[s]:
                p = ROOT / "tasks" / t / "data" / s / "cpu_time_ms.txt"
                if p.exists():
                    try:
                        cpu_baselines[s][t] = float(p.read_text().strip())
                    except Exception:
                        pass

    # Compact records: drop raw_response-style bloat, keep essentials
    for s in records:
        for t in records[s]:
            for m in records[s][t]:
                r = records[s][t][m]
                keep = {
                    "compiled": r.get("compiled"),
                    "correct": r.get("correct"),
                    "compile_error": (r.get("compile_error") or "")[:500] or None,
                    "correctness_detail": r.get("correctness_detail"),
                }
                if r.get("benchmark"):
                    bm = r["benchmark"]
                    keep["benchmark"] = {
                        "speedup_e2e": bm.get("speedup_e2e"),
                        "speedup_kernel": bm.get("speedup_kernel"),
                        "e2e_time_ms_mean": (bm.get("e2e_time_ms") or {}).get("mean") if isinstance(bm.get("e2e_time_ms"), dict) else bm.get("e2e_time_ms"),
                        "kernel_time_ms": bm.get("kernel_time_ms"),
                        "memcpy_overhead_ms": bm.get("memcpy_overhead_ms"),
                        "num_kernel_launches": bm.get("num_kernel_launches"),
                        "cpu_baseline_ms": bm.get("cpu_baseline_ms"),
                        "size_name": bm.get("size_name"),
                        "hardware": bm.get("hardware"),
                    }
                records[s][t][m] = keep

    out_path = (
        Path(args.out) if args.out
        else ROOT / "runs" / "reports" / f"acceleval_consolidated_{datetime.now():%Y%m%d_%H%M}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = {
        "generated_at": datetime.now().isoformat(),
        "n_active_tasks": len(active),
        "active_tasks": active,
        "models": dict(RUNS_L3),
        "cpu_baselines_ms": cpu_baselines,
        "records": records,
    }

    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_path}")
    print(f"  size: {out_path.stat().st_size / 1024:.1f} KB")
    # Stats
    for s in ("small", "medium", "large"):
        ok = sum(1 for t in active for m in RUNS_L3
                 if records[s][t].get(m, {}).get("compiled")
                 and records[s][t].get(m, {}).get("correct"))
        print(f"  {s:8s}: {ok}/{len(active)*len(RUNS_L3)} (task,model) cells passing")


if __name__ == "__main__":
    main()
