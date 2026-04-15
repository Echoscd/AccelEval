#!/usr/bin/env python3
"""Visualize AccelEval task categories, sources, and difficulty distribution."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TASKS_DIR = Path(__file__).parent / "tasks"

# ── Collect task metadata ───────────────────────────────────────────
tasks = []
for tj in sorted(TASKS_DIR.glob("*/task.json")):
    task_id = tj.parent.name
    with open(tj) as f:
        d = json.load(f)

    # Detect source from cpu_reference.c header
    ref_file = tj.parent / "cpu_reference.c"
    source = "Original"
    if ref_file.exists():
        header = ref_file.read_text()[:500].lower()
        if "financebench" in header:
            source = "FinanceBench"
        elif "dualsphysics" in header:
            source = "DualSPHysics"
        elif "or-tools" in header or "or_tools" in header:
            source = "Google OR-Tools"
        elif "gromacs" in header:
            source = "GROMACS"
        elif "cudtw" in header:
            source = "cuDTW"
        elif "cuspatial" in header:
            source = "cuSpatial"
        elif "knn-cuda" in header or "knncuda" in header:
            source = "kNN-CUDA"
        elif "fast-cuda-gpu-dbscan" in header or "dbscan.cpp" in header:
            source = "fast-cuda-gpu-dbscan"
        elif "heurigym" in header:
            source = "HeuriGym"

    # 10 tasks ported from OR papers via pipeline (created 2026-03-19 ~ 2026-03-27)
    OR_PAPER_TASKS = {
        "network_rm_dp", "gittins_index", "inventory_replenishment_dp",
        "hawkes_dynamic_pricing_hjb", "self_exciting_pricing_dp",
        "robust_value_iteration_hypercube", "motzkin_straus_blp_eval",
        "nash_flows_over_time", "thompson_sampling", "batched_lhpca_portfolio",
    }
    if source == "Original" and task_id in OR_PAPER_TASKS:
        source = "Operation Research"
    if task_id == "miniWeather":
        source = "miniWeather"

    tasks.append({
        "id": task_id,
        "name": d.get("name", task_id),
        "category": d.get("category", "unknown"),
        "difficulty": d.get("difficulty", 1),
        "source": source,
        "tags": d.get("tags", []),
    })

# ── Normalize categories ────────────────────────────────────────────
CAT_MAP = {
    "computational_finance": "Finance",
    "financial_computing": "Finance",
    "Financial Engineering": "Finance",
    "graph": "Graph",
    "dynamic_programming": "Dynamic Programming",
    "Dynamic Programming": "Dynamic Programming",
    "computational_geometry": "Computational Geometry",
    "optimization": "Optimization",
    "Optimization": "Optimization",
    "parsing": "Parsing & Automata",
    "automata_simulation": "Parsing & Automata",
    "spatial_clustering": "Spatial / Time-Series",
    "time_series_distance": "Spatial / Time-Series",
    "time_series": "Spatial / Time-Series",
    "spatial_distance": "Spatial / Time-Series",
    "bioinformatics": "Spatial / Time-Series",
    "scientific_computing": "Scientific Computing",
    "molecular_dynamics": "Scientific Computing",
    "fluid_simulation": "Scientific Computing",
    "stencil_simulation": "Scientific Computing",
    "linear_algebra": "Linear Algebra / LP",
    "linear_programming": "Linear Algebra / LP",
    "stochastic_simulation": "Stochastic / Bandit",
    "probabilistic_inference": "Stochastic / Bandit",
    "Transportation": "Graph",
}
for t in tasks:
    t["cat_norm"] = CAT_MAP.get(t["category"], t["category"])

# ── Color palettes ──────────────────────────────────────────────────
CAT_COLORS = {
    "Finance": "#FF6B6B",
    "Graph": "#4ECDC4",
    "Dynamic Programming": "#45B7D1",
    "Scientific Computing": "#FFA07A",
    "Spatial / Time-Series": "#BB86FC",
    "Optimization": "#FFD54F",
    "Linear Algebra / LP": "#81C784",
    "Parsing & Automata": "#90CAF9",
    "Stochastic / Bandit": "#F48FB1",
    "Computational Geometry": "#CE93D8",
}

SRC_COLORS = {
    "Original": "#4ECDC4",
    "Operation Research": "#FF9800",
    "FinanceBench": "#FF6B6B",
    "DualSPHysics": "#FFA07A",
    "Google OR-Tools": "#45B7D1",
    "GROMACS": "#81C784",
    "cuDTW": "#BB86FC",
    "cuSpatial": "#FFD54F",
    "kNN-CUDA": "#90CAF9",
    "fast-cuda-gpu-dbscan": "#F48FB1",
    "miniWeather": "#A5D6A7",
    "HeuriGym": "#E0E0E0",
}

DIFF_COLORS = {1: "#4ECDC4", 2: "#45B7D1", 3: "#FFA500", 4: "#FF4444"}

# ── Figure setup ────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16), facecolor="#1A1A2E")
fig.suptitle("AccelEval Task Overview", fontsize=28, color="white",
             fontweight="bold", y=0.97)
fig.text(0.5, 0.945, f"Total: {len(tasks)} tasks across {len(set(t['cat_norm'] for t in tasks))} categories",
         ha="center", fontsize=14, color="#AAAACC")

# ── 1. Category donut (top-left) ────────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor("#1A1A2E")

from collections import Counter
cat_counts = Counter(t["cat_norm"] for t in tasks)
cat_sorted = cat_counts.most_common()
labels, sizes = zip(*cat_sorted)
colors = [CAT_COLORS.get(l, "#888888") for l in labels]

wedges, texts, autotexts = ax1.pie(
    sizes, labels=None, autopct=lambda p: f"{int(round(p*sum(sizes)/100))}",
    colors=colors, startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.45, edgecolor="#1A1A2E", linewidth=2))
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(12)
    at.set_fontweight("bold")

ax1.legend(wedges, [f"{l} ({s})" for l, s in zip(labels, sizes)],
           loc="center left", bbox_to_anchor=(-0.15, 0.5),
           fontsize=10, facecolor="#22223A", edgecolor="#333355",
           labelcolor="white", framealpha=0.9)
ax1.set_title("By Category", color="white", fontsize=18, pad=15)

# ── 2. Source bar chart (top-right) ─────────────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor("#22223A")

src_counts = Counter(t["source"] for t in tasks)
src_sorted = src_counts.most_common()
src_labels, src_sizes = zip(*src_sorted)
src_colors = [SRC_COLORS.get(l, "#888888") for l in src_labels]

y_pos = np.arange(len(src_labels))
bars = ax2.barh(y_pos, src_sizes, color=src_colors, edgecolor="#1A1A2E", height=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(src_labels, color="white", fontsize=12)
ax2.invert_yaxis()
ax2.set_xlabel("Number of Tasks", color="#AAAACC", fontsize=12)
ax2.set_title("By Source / Reference Implementation", color="white", fontsize=18, pad=15)
ax2.tick_params(colors="#AAAACC")
ax2.spines["bottom"].set_color("#333355")
ax2.spines["left"].set_color("#333355")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
for bar, val in zip(bars, src_sizes):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             str(val), va="center", color="white", fontsize=12, fontweight="bold")

# ── 3. Gemini 3.1 Pro speedup (bottom-left) ─────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor("#22223A")

# Load eval results: combine L2 + L3 retry
import json as _json
_gemini_speedups = {}

# L2 results
_l2_path = TASKS_DIR.parent / "runs" / "gemini-3.1-pro-preview-openrouter_l2_20260413" / "eval_results_20260413_155625.json"
if _l2_path.exists():
    _r1 = _json.load(open(_l2_path))
    for _k, _v in _r1.items():
        _tid = _v.get("task_id", _k)
        if _v.get("correct"):
            _bench = _v.get("benchmark") or {}
            _sp = _bench.get("speedup_e2e") or _bench.get("speedup_vs_cpu")
            if _sp:
                _gemini_speedups[_tid] = _sp

# L3 retry results (override if better)
_l3_path = TASKS_DIR.parent / "runs" / "gemini-3.1-pro-preview-openrouter_l3_retry_20260413"
for _ef in sorted(_l3_path.glob("eval_results_*.json")):
    _r2 = _json.load(open(_ef))
    for _k, _v in _r2.items():
        _tid = _v.get("task_id", _k)
        if _v.get("correct"):
            _bench = _v.get("benchmark") or {}
            _sp = _bench.get("speedup_e2e") or _bench.get("speedup_vs_cpu")
            if _sp and _sp > _gemini_speedups.get(_tid, 0):
                _gemini_speedups[_tid] = _sp

# Collect failed tasks (compiled but wrong, or compile fail)
_gemini_failed = {}  # tid -> (status, speedup_or_0)
_all_eval_tasks = set()
if _l2_path.exists():
    for _k, _v in _r1.items():
        _tid = _v.get("task_id", _k)
        _all_eval_tasks.add(_tid)
        if not _v.get("correct") and _tid not in _gemini_speedups:
            _compiled = _v.get("compiled", False)
            _bench = _v.get("benchmark") or {}
            _sp = _bench.get("speedup_e2e") or _bench.get("speedup_vs_cpu") or 0
            _gemini_failed[_tid] = ("WRONG" if _compiled else "COMPILE_FAIL", _sp)

# Override with L3 retry if task got fixed
for _tid in list(_gemini_failed.keys()):
    if _tid in _gemini_speedups:
        del _gemini_failed[_tid]

# Sort correct by speedup descending, then append failed
_sorted_sp = sorted(_gemini_speedups.items(), key=lambda x: x[1], reverse=True)
_sorted_fail = sorted(_gemini_failed.items(), key=lambda x: x[1][1], reverse=True)

_all_tasks_plot = []
_all_vals_plot = []
_all_colors_plot = []
_all_labels_plot = []

import math
for _tid, _sp in _sorted_sp:
    _all_tasks_plot.append(_tid)
    _all_vals_plot.append(_sp)
    src = next((tt["source"] for tt in tasks if tt["id"] == _tid), "Original")
    _all_colors_plot.append(SRC_COLORS.get(src, "#888888"))
    label = f"{_sp:.0f}x" if _sp >= 10 else f"{_sp:.1f}x"
    _all_labels_plot.append(label)

for _tid, (_status, _sp) in _sorted_fail:
    _all_tasks_plot.append(_tid)
    _all_vals_plot.append(max(_sp, 0.01))  # small value for log scale
    _all_colors_plot.append("#444455")  # gray for failed
    _all_labels_plot.append(_status)

n_correct = len(_sorted_sp)
y_pos = np.arange(len(_all_tasks_plot))
bars = ax3.barh(y_pos, _all_vals_plot, color=_all_colors_plot, edgecolor="#1A1A2E", height=0.7)
ax3.set_xscale("log")
ax3.set_yticks(y_pos)
# Color task labels: white for correct, red for failed
_ytick_labels = ax3.set_yticklabels(_all_tasks_plot, fontsize=6.5)
for i, lbl in enumerate(_ytick_labels):
    lbl.set_color("#CCCCDD" if i < n_correct else "#FF6B6B")
ax3.invert_yaxis()
ax3.set_xlabel("Speedup vs CPU (log scale)", color="#AAAACC", fontsize=11)
ax3.set_title(f"Gemini 3.1 Pro — {n_correct}/40 correct, {len(_sorted_fail)} failed",
              color="white", fontsize=16, pad=12)
ax3.tick_params(colors="#AAAACC")
ax3.spines["bottom"].set_color("#333355")
ax3.spines["left"].set_color("#333355")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.axvline(x=1, color="#FF6B6B", linestyle="--", linewidth=0.8, alpha=0.5)
for i, (bar, label) in enumerate(zip(bars, _all_labels_plot)):
    val = _all_vals_plot[i]
    if i < n_correct:
        ax3.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height()/2,
                 label, va="center", color="#AAAACC", fontsize=6)
    else:
        ax3.text(0.015, bar.get_y() + bar.get_height()/2,
                 label, va="center", color="#FF6B6B", fontsize=6.5, fontweight="bold")

# ── 4. CPU baseline lines of code (bottom-right) ─────────────────────
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor("#22223A")

# Count lines in each cpu_reference.c
loc_data = []
for t in tasks:
    ref = TASKS_DIR / t["id"] / "cpu_reference.c"
    if ref.exists():
        lines = len([l for l in ref.read_text().splitlines() if l.strip()])
    else:
        lines = 0
    loc_data.append((t["id"], lines, t["cat_norm"], t["source"]))
    t["loc"] = lines

loc_data.sort(key=lambda x: x[1], reverse=True)

# Short source abbreviations for labels
_SRC_ABBR = {
    "Original": "Orig",
    "Operation Research": "OR Paper",
    "FinanceBench": "FinBench",
    "Google OR-Tools": "OR-Tools",
    "DualSPHysics": "DualSPH",
    "fast-cuda-gpu-dbscan": "DBSCAN",
    "GROMACS": "GROMACS",
    "cuDTW": "cuDTW",
    "cuSpatial": "cuSpatial",
    "kNN-CUDA": "kNN-CUDA",
    "miniWeather": "miniWeather",
    "HeuriGym": "HeuriGym",
}
loc_ids = [f"{x[0]} ({_SRC_ABBR.get(x[3], x[3])})" for x in loc_data]
loc_vals = [x[1] for x in loc_data]
loc_colors = [SRC_COLORS.get(x[3], "#888888") for x in loc_data]

y_pos = np.arange(len(loc_ids))
bars = ax4.barh(y_pos, loc_vals, color=loc_colors, edgecolor="#1A1A2E", height=0.7)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(loc_ids, color="#CCCCDD", fontsize=7.5)
ax4.invert_yaxis()
ax4.set_xlabel("Lines of Code (non-blank)", color="#AAAACC", fontsize=12)
ax4.set_title("CPU Baseline Code Size", color="white", fontsize=18, pad=15)
ax4.tick_params(colors="#AAAACC")
ax4.spines["bottom"].set_color("#333355")
ax4.spines["left"].set_color("#333355")
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
for bar, val in zip(bars, loc_vals):
    ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             str(val), va="center", color="#AAAACC", fontsize=7.5)

# ── 5. Full task table ──────────────────────────────────────────────
fig2 = plt.figure(figsize=(22, 20), facecolor="#1A1A2E")
fig2.suptitle("AccelEval — All Tasks Detail", fontsize=24, color="white",
              fontweight="bold", y=0.98)

ax_table = fig2.add_subplot(111)
ax_table.set_facecolor("#1A1A2E")
ax_table.axis("off")

# Sort by category then difficulty
tasks_sorted = sorted(tasks, key=lambda t: (t["cat_norm"], t["difficulty"]))

col_labels = ["#", "Task ID", "Category", "Diff", "Source", "Name"]
cell_text = []
cell_colors = []
for i, t in enumerate(tasks_sorted):
    diff_str = "★" * t["difficulty"]
    row = [str(i+1), t["id"], t["cat_norm"], diff_str, t["source"],
           t["name"][:50]]
    cell_text.append(row)
    cat_color = CAT_COLORS.get(t["cat_norm"], "#888888")
    # Muted row background
    r, g, b = int(cat_color[1:3], 16), int(cat_color[3:5], 16), int(cat_color[5:7], 16)
    bg = (r/255*0.15 + 0.1, g/255*0.15 + 0.1, b/255*0.15 + 0.1, 0.9)
    cell_colors.append([bg] * 6)

table = ax_table.table(cellText=cell_text, colLabels=col_labels,
                       cellColours=cell_colors,
                       colColours=["#333366"]*6,
                       loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.35)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#444466")
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    else:
        cell.set_text_props(color="#CCCCDD")
    if col == 3 and row > 0:  # difficulty column
        diff_val = len(cell.get_text().get_text())
        cell.set_text_props(color=DIFF_COLORS.get(diff_val, "white"))

table.auto_set_column_width([0, 1, 2, 3, 4, 5])

# ── Save ────────────────────────────────────────────────────────────
out1 = Path(__file__).parent / "task_overview.png"
out2 = Path(__file__).parent / "task_detail_table.png"
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(out1, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
fig2.savefig(out2, dpi=150, facecolor=fig2.get_facecolor(), bbox_inches="tight")
print(f"Saved: {out1}")
print(f"Saved: {out2}")

# ── Print summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Total tasks: {len(tasks)}")
print(f"\nBy Category:")
for cat, cnt in cat_sorted:
    print(f"  {cat:<25s} {cnt}")
print(f"\nBy Source:")
for src, cnt in src_sorted:
    print(f"  {src:<25s} {cnt}")
print(f"\nBy Difficulty:")
for d in diff_levels:
    print(f"  {'★'*d:<6s} ({d})  {diff_counts.get(d,0)}")
