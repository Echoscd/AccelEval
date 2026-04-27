#!/usr/bin/env python3
"""
Stage 3 of the strategy-transfer pipeline.

Compare Stage-1 (no-guidance) and Stage-2 (guidance-augmented) medium-size
speedups per (model, task) pair and emit:
  * per-task, per-model replication ratio RR(M, Q) = y_S2(M,Q) / y_winner(Q)
  * per-model aggregate: mean RR, %discovery (RR>=0.7), %implementation (RR<0.7)
  * LaTeX table fragment for paper §9 Strategy Transfer

Usage:
    python3 scripts/compute_replication_ratio.py \\
        --s1-runs gemini-3.1-pro-preview-openrouter_l3_20260421_1712 ... \\
        --s2-runs gemini-3.1-pro-preview-openrouter_l3s2_20260423_XXXX ... \\
        --best-manifest runs/strategy_transfer/best/manifest.json \\
        --size medium \\
        --out runs/strategy_transfer/results
"""
import argparse
import json
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent


def extract_speedups(run_dir: Path, size: str) -> dict:
    """Returns {task_id: speedup} using the most recent eval record per task."""
    out = {}
    if not run_dir.exists():
        return out
    # Collect (mtime, record) per task across all eval_results_*.json, keep latest
    latest: dict[str, tuple[float, dict]] = {}
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
                prev = latest.get(task_id)
                if prev is None or mt > prev[0]:
                    latest[task_id] = (mt, v)
        except Exception:
            continue
    for task_id, (_, v) in latest.items():
        if not (v.get("compiled") and v.get("correct")):
            out[task_id] = 0.0
            continue
        bm = v.get("benchmark") or {}
        sp = bm.get("speedup_e2e") or 0
        out[task_id] = float(sp) if sp > 0 else 0.0
    return out


def model_from_run_name(run: str) -> str:
    """Strip timestamp/level suffix, keep vendor prefix for clarity."""
    # e.g. "gemini-3.1-pro-preview-openrouter_l3_20260421_1712" → "gemini-3.1-pro-preview-openrouter"
    # "openai/gpt-5.4_l3s2_20260423_1600" → "openai/gpt-5.4"
    core = run.split("_l3")[0]
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-runs", nargs="+", required=True)
    ap.add_argument("--s2-runs", nargs="+", required=True)
    ap.add_argument("--best-manifest", default="runs/strategy_transfer/best/manifest.json")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="RR threshold for discovery vs implementation")
    ap.add_argument("--out", default="runs/strategy_transfer/results")
    args = ap.parse_args()

    best = json.loads((ROOT / args.best_manifest).read_text())
    winner_sp = {t: info["winner_speedup"] for t, info in best["tasks"].items()}
    winner_model = {t: info["winner_model"] for t, info in best["tasks"].items()}

    # Build model -> {s1: {task:sp}, s2: {task:sp}}
    per_model = defaultdict(lambda: {"s1": {}, "s2": {}})
    for run in args.s1_runs:
        rd = ROOT / "runs" / run
        m = model_from_run_name(os.path.basename(run))
        per_model[m]["s1"] = extract_speedups(rd, args.size)
    for run in args.s2_runs:
        rd = ROOT / "runs" / run
        m = model_from_run_name(os.path.basename(run))
        per_model[m]["s2"] = extract_speedups(rd, args.size)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-model aggregate
    rows = []
    for model in sorted(per_model):
        s1 = per_model[model]["s1"]
        s2 = per_model[model]["s2"]
        eligible = []
        for t, w in winner_sp.items():
            if w < 2.0:
                continue
            if winner_model[t] == model:
                # Skip own-wins when computing RR (we're the ceiling, trivially 1.0)
                continue
            if t not in s2:
                continue
            rr = (s2[t] / w) if w > 0 else 0.0
            eligible.append({
                "task": t,
                "winner_speedup": w,
                "s1_speedup": s1.get(t, 0.0),
                "s2_speedup": s2[t],
                "RR": round(rr, 3),
            })
        if not eligible:
            continue
        mean_rr = mean(e["RR"] for e in eligible)
        median_rr = median(e["RR"] for e in eligible)
        discovery = sum(1 for e in eligible if e["RR"] >= args.threshold) / len(eligible) * 100
        impl = 100.0 - discovery
        # Lift: how many tasks improved vs Stage 1?
        improved = sum(1 for e in eligible if e["s2_speedup"] > e["s1_speedup"] * 1.1)
        regressed = sum(1 for e in eligible if e["s2_speedup"] < e["s1_speedup"] * 0.9)
        rows.append({
            "model": model,
            "n_tasks": len(eligible),
            "mean_RR": round(mean_rr, 3),
            "median_RR": round(median_rr, 3),
            "discovery_pct": round(discovery, 1),
            "impl_pct": round(impl, 1),
            "improved": improved,
            "regressed": regressed,
            "detail": eligible,
        })

    with open(out_dir / "replication_ratio.json", "w") as f:
        json.dump({"threshold": args.threshold, "size": args.size, "rows": rows}, f, indent=2)

    # Console summary
    print(f"\nReplication Ratio at {args.size} (threshold τ={args.threshold})\n")
    print(f"{'Model':38s} {'n':>4s} {'mean':>7s} {'med':>7s} {'Disc%':>7s} {'Impl%':>7s}  improved/regressed")
    print("-" * 98)
    for r in rows:
        print(f"{r['model']:38s} {r['n_tasks']:>4d} {r['mean_RR']:>7.3f} {r['median_RR']:>7.3f}"
              f" {r['discovery_pct']:>6.1f}% {r['impl_pct']:>6.1f}%     "
              f"{r['improved']}/{r['regressed']}")

    # LaTeX fragment
    tex = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Competitor} & \textbf{\# tasks} & \textbf{Mean RR} & "
        r"\textbf{Discovery\%} & \textbf{Impl.\%} & \textbf{Improved / Regr.} \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['model'].split('/')[-1]} & {r['n_tasks']} & {r['mean_RR']:.2f} & "
            f"{r['discovery_pct']:.1f}\\% & {r['impl_pct']:.1f}\\% & "
            f"{r['improved']} / {r['regressed']} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "transfer_table.tex").write_text("\n".join(tex) + "\n")
    print(f"\n  wrote {out_dir / 'replication_ratio.json'}")
    print(f"  wrote {out_dir / 'transfer_table.tex'}")


if __name__ == "__main__":
    main()
