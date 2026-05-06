#!/usr/bin/env python3
"""Compare Stage-2 paired GM lift under three conditions:

  S1            : zero-hint baseline (paper Table 2 source)
  S2-treatment  : standard prompt + per-task winner recipe (paper Table 4 source)
  S2-control    : standard prompt + length-matched generic CUDA tips (this run)

For each recipient model and task, we keep tasks where the model produced a
correct solution in S1, then compare s_S2/s_S1 under treatment vs control.
"""
import os, sys, json, glob, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(ROOT))

# Recipient models (Gemini is the recipe provider, excluded from aggregate but reported for ref.)
RECIPIENTS = [
    "Claude Opus 4.6",
    "DeepSeek V3.2",
    "Qwen 3.6 Plus",
    "GPT-5.4",
    "Kimi K2.5",
    "DeepSeek V4 Pro",
    "GLM 5.1",
]
PROVIDER = "Gemini 3.1 Pro"

# Map paper model name -> S2-control run-dir glob
S2C_GLOB = {
    "Claude Opus 4.6":  "runs/claude-opus-4.6-openrouter_l3s2c_*",
    "DeepSeek V3.2":    "runs/deepseek-v3.2-openrouter_l3s2c_*",
    "DeepSeek V4 Pro":  "runs/deepseek-v4-pro-openrouter_l3s2c_*",
    "Gemini 3.1 Pro":   "runs/gemini-3.1-pro-preview-openrouter_l3s2c_*",
    "GLM 5.1":          "runs/glm-5.1-openrouter_l3s2c_*",
    "Kimi K2.5":        "runs/kimi-k2.5-openrouter_l3s2c_*",
    "GPT-5.4":          "runs/openai/gpt-5.4_l3s2c_*",
    "Qwen 3.6 Plus":    "runs/qwen/qwen3.6-plus_l3s2c_*",
}
# S2-treatment run-dirs — same paths the paper Table 4 export script uses
# (scripts/export_s1_vs_s2_xlsx.py). DO NOT switch to
# build/solutions/stage2_with_guidance/<model>/eval_results.json — that path
# holds a stale early-iteration dump (e.g. claude bellman_ford 1.91x there
# vs the real 267x), and using it yields LIFTs ~5x lower than paper Table 4.
S2T_RUN = {
    "Claude Opus 4.6":  "runs/claude-opus-4.6-openrouter_l3s2_20260424_1534",
    "DeepSeek V3.2":    "runs/deepseek-v3.2-openrouter_l3s2_20260424_1534",
    "DeepSeek V4 Pro":  "runs/deepseek-v4-pro-openrouter_l3s2_20260427_1312",
    "Gemini 3.1 Pro":   "runs/gemini-3.1-pro-preview-openrouter_l3s2_20260424_1534",
    "GLM 5.1":          "runs/glm-5.1-openrouter_l3s2_20260424_1534",
    "Kimi K2.5":        "runs/kimi-k2.5-openrouter_l3s2_20260424_1534",
    "GPT-5.4":          "runs/openai/gpt-5.4_l3s2_20260424_1534",
    "Qwen 3.6 Plus":    "runs/qwen/qwen3.6-plus_l3s2_20260424_1534",
}

ELIGIBLE_TASKS = sorted([Path(p).stem for p in glob.glob("runs/strategy_transfer/guidance/*.md")])
print(f"Strategy-eligible tasks: {len(ELIGIBLE_TASKS)}", file=sys.stderr)


def load_s1_speedups():
    """Load Stage-1 per-(model,task) speedups at medium from consolidated leaderboard."""
    fp = sorted(glob.glob("runs/reports/acceleval_consolidated_*.json"))[-1]
    d = json.load(open(fp))
    out = defaultdict(dict)  # [model][task] = speedup
    for t, recs in d["records"]["medium"].items():
        for m, r in recs.items():
            if not isinstance(r, dict) or not r.get("correct"): continue
            sp = (r.get("benchmark") or {}).get("speedup_e2e")
            if sp is None or sp <= 0: continue
            out[m][t] = sp
    return out, fp


def load_s2t_speedups():
    """Load Stage-2 treatment speedups at medium from runs/<model>_l3s2_*/.

    Same logic as scripts/export_s1_vs_s2_xlsx.py:latest_per_task — read every
    eval_results_*.json in the run-dir and keep the latest entry per task at
    the medium scale. Speedups outside medium are ignored; pass@2/large dumps
    in the same dir do not contaminate medium numbers.
    """
    out = defaultdict(dict)
    for m, rd in S2T_RUN.items():
        if not os.path.isdir(rd): continue
        latest_per_task = {}  # task -> (mtime, speedup)
        for ev in glob.glob(f"{rd}/eval_results_*.json"):
            mt = os.path.getmtime(ev)
            try: d = json.load(open(ev))
            except: continue
            for k, v in d.items():
                if not isinstance(v, dict) or not v.get("correct"): continue
                bm = v.get("benchmark") or {}
                if bm.get("size_name") != "medium": continue
                t = v.get("task_id") or k.rsplit("_sample_", 1)[0]
                sp = bm.get("speedup_e2e")
                if sp is None or sp <= 0: continue
                if t not in latest_per_task or mt > latest_per_task[t][0]:
                    latest_per_task[t] = (mt, sp)
        out[m] = {t: sp for t, (_, sp) in latest_per_task.items()}
    return out


def load_s2c_speedups():
    """Load Stage-2 control speedups from runs/<model>_l3s2c_*/eval_results_*.json."""
    out = defaultdict(dict)
    for m, glob_pat in S2C_GLOB.items():
        run_dirs = sorted(glob.glob(glob_pat))
        for rd in run_dirs:
            evals = sorted(glob.glob(f"{rd}/eval_results_*.json"))
            if not evals: continue
            d = json.load(open(evals[-1]))
            for k, v in d.items():
                if not isinstance(v, dict) or not v.get("correct"): continue
                t = v.get("task_id")
                sp = (v.get("benchmark") or {}).get("speedup_e2e")
                if sp is None or sp <= 0: continue
                # If duplicate task across runs (shouldn't happen), keep best
                cur = out[m].get(t, 0.0)
                if sp > cur: out[m][t] = sp
    return out


def paired_lift(s1, s2, model, eligible_tasks):
    """Paired GM(s2 / s1) over tasks where both stages have correct speedups for this model."""
    pairs = []
    for t in eligible_tasks:
        a = s1.get(model, {}).get(t)
        b = s2.get(model, {}).get(t)
        if a is None or b is None: continue
        pairs.append((t, a, b, b/a))
    if not pairs: return None, 0
    gm = math.exp(sum(math.log(b/a) for _, a, b, _ in pairs) / len(pairs))
    return gm, len(pairs)


def main():
    s1, s1_fp = load_s1_speedups()
    s2t = load_s2t_speedups()
    s2c = load_s2c_speedups()

    print(f"S1 source: {s1_fp}", file=sys.stderr)
    print(f"S1 models: {len(s1)} ({sum(len(v) for v in s1.values())} (m,t) pairs)", file=sys.stderr)
    print(f"S2-treatment models: {len(s2t)} ({sum(len(v) for v in s2t.values())} (m,t) pairs)", file=sys.stderr)
    print(f"S2-control models: {len(s2c)} ({sum(len(v) for v in s2c.values())} (m,t) pairs)", file=sys.stderr)
    print(file=sys.stderr)

    print(f"{'Model':<20} {'LIFT_S2t':>10} {'n_t':>5}   {'LIFT_S2c':>10} {'n_c':>5}   {'t/c':>6}")
    print("-" * 80)
    rows = []
    for m in [PROVIDER] + RECIPIENTS:
        gm_t, n_t = paired_lift(s1, s2t, m, ELIGIBLE_TASKS)
        gm_c, n_c = paired_lift(s1, s2c, m, ELIGIBLE_TASKS)
        gm_t_str = f"{gm_t:.2f}x" if gm_t else " --   "
        gm_c_str = f"{gm_c:.2f}x" if gm_c else " --   "
        ratio_str = f"{gm_t/gm_c:.2f}" if (gm_t and gm_c) else " -- "
        print(f"{m:<20} {gm_t_str:>10} {n_t:>5}   {gm_c_str:>10} {n_c:>5}   {ratio_str:>6}")
        rows.append({"model": m, "lift_s2t": gm_t, "n_t": n_t, "lift_s2c": gm_c, "n_c": n_c})

    # Aggregate over 7 recipients
    print("-" * 80)
    rec_t = [r for r in rows if r["model"] in RECIPIENTS and r["lift_s2t"]]
    rec_c = [r for r in rows if r["model"] in RECIPIENTS and r["lift_s2c"]]
    if rec_t:
        agg_t = math.exp(sum(math.log(r["lift_s2t"]) for r in rec_t) / len(rec_t))
        print(f"Aggregate (7 recipients) LIFT_S2t = {agg_t:.2f}x  (n={len(rec_t)} models)")
    if rec_c:
        agg_c = math.exp(sum(math.log(r["lift_s2c"]) for r in rec_c) / len(rec_c))
        print(f"Aggregate (7 recipients) LIFT_S2c = {agg_c:.2f}x  (n={len(rec_c)} models)")
    if rec_t and rec_c:
        common = [r["model"] for r in rec_t if r["model"] in [s["model"] for s in rec_c]]
        agg_t_c = math.exp(sum(math.log(next(r["lift_s2t"] for r in rec_t if r["model"]==m)) for m in common) / len(common))
        agg_c_c = math.exp(sum(math.log(next(r["lift_s2c"] for r in rec_c if r["model"]==m)) for m in common) / len(common))
        print(f"Treatment / Control aggregate ratio = {agg_t_c/agg_c_c:.2f}x  (over {len(common)} common models)")

    out_fp = "compare/stage2_control_summary.json"
    with open(out_fp, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nSaved: {out_fp}", file=sys.stderr)


if __name__ == "__main__":
    main()
