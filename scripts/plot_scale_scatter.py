#!/usr/bin/env python3
"""
plot_scale_scatter.py — companion scatter for §6.2.

x = CPU baseline time (s, log)
y = end-to-end speedup (log)
one dot per (model, task, size) where the solution compiled & passed.
Colored by model (same palette as the trajectory plot).
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "nips" / "figs" / "scale_scatter.pdf"

f = sorted(glob.glob(str(ROOT / "runs" / "reports" / "acceleval_consolidated_*.json")))[-1]
d = json.loads(Path(f).read_text())
records = d["records"]
MODELS = list(d["models"])

PALETTE = {
    "Gemini 3.1 Pro":  "#1f77b4",
    "Claude Opus 4.6": "#d62728",
    "DeepSeek V4 Pro": "#2ca02c",
    "Qwen 3.6 Plus":   "#ff7f0e",
    "Kimi K2.5":       "#9467bd",
    "GLM 5.1":         "#8c564b",
    "GPT-5.4":         "#e377c2",
    "DeepSeek V3.2":   "#7f7f7f",
}

# Collect all (cpu_s, speedup) per model
points: dict[str, list[tuple[float, float]]] = {m: [] for m in MODELS}
for sz in ["small", "medium", "large"]:
    for t, pm in records[sz].items():
        for m, rec in pm.items():
            if not (rec.get("compiled") and rec.get("correct")):
                continue
            bm = rec.get("benchmark") or {}
            sp = bm.get("speedup_e2e")
            cpu_ms = bm.get("cpu_baseline_ms")
            if sp and sp > 0 and cpu_ms and cpu_ms > 0:
                points[m].append((cpu_ms / 1000.0, sp))   # ms -> seconds

print(f"Per-model points: {sum(len(v) for v in points.values())}")
for m, pts in points.items():
    print(f"  {m:<18} n={len(pts):3}")

# Order legend by descending Large-scale GM (consistent with companion plot)
def m_score(m):
    pts = points[m]
    if not pts: return 0
    return sum(math.log(p[1]) for p in pts) / len(pts)
order = sorted(MODELS, key=lambda m: -m_score(m))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(5.0, 3.0))

for m in order:
    xs = [p[0] for p in points[m]]
    ys = [p[1] for p in points[m]]
    ax.scatter(xs, ys, color=PALETTE.get(m, "#333"),
               s=11, alpha=0.62, edgecolors="white", linewidths=0.25,
               label=m, zorder=3)

ax.axhline(1.0, color="#888", linestyle="--", linewidth=0.7, zorder=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("CPU baseline time (s)")
ax.set_ylabel(r"End-to-end speedup ($\times$)")

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(False)
ax.tick_params(direction="out", length=3, width=0.5, which="both")

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT)
print(f"\n[saved] {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
