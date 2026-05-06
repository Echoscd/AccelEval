#!/usr/bin/env python3
"""
plot_scale_trajectories.py — §6.2 figure.

Per-model GM speedup at three input scales (Small / Medium / Large),
linear y-scale, 8 lines (one per model). Numeric labels at each scale
point of the strongest model for orientation.
"""
from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "nips" / "figs" / "scale_trajectories.pdf"

f = sorted(glob.glob(str(ROOT / "runs" / "reports" / "acceleval_consolidated_*.json")))[-1]
d = json.loads(Path(f).read_text())
records = d["records"]
MODELS = list(d["models"])

def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0

# Per-model GM speedup at each scale, restricted to passing tasks
SIZES = ["small", "medium", "large"]
gm_per_model: dict[str, list[float]] = {}
for m in MODELS:
    row = []
    for sz in SIZES:
        sps = []
        for t, pm in records[sz].items():
            rec = pm.get(m)
            if rec and rec.get("compiled") and rec.get("correct"):
                sp = (rec.get("benchmark") or {}).get("speedup_e2e")
                if sp and sp > 0:
                    sps.append(sp)
        row.append(gmean(sps))
    gm_per_model[m] = row

print("Per-model GM speedup:")
for m, row in sorted(gm_per_model.items(), key=lambda x: -x[1][1]):
    print(f"  {m:<18}  {row[0]:5.1f}× / {row[1]:5.1f}× / {row[2]:5.1f}×")

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

# Distinct, print-friendly palette ordered to match the leaderboard rank
PALETTE = {
    "Gemini 3.1 Pro":  "#1f77b4",  # blue
    "Claude Opus 4.6": "#d62728",  # red
    "DeepSeek V4 Pro": "#2ca02c",  # green
    "Qwen 3.6 Plus":   "#ff7f0e",  # orange
    "Kimi K2.5":       "#9467bd",  # purple
    "GLM 5.1":         "#8c564b",  # brown
    "GPT-5.4":         "#e377c2",  # pink
    "DeepSeek V3.2":   "#7f7f7f",  # gray
}

fig, ax = plt.subplots(figsize=(5.0, 3.0))
xs = [0, 1, 2]
labels = ["Small", "Medium", "Large"]

# Order by Large-scale GM descending, so legend reads from highest-final to lowest
order = sorted(MODELS, key=lambda m: -gm_per_model[m][2])
for m in order:
    row = gm_per_model[m]
    ax.plot(xs, row, color=PALETTE.get(m, "#333"), linewidth=1.6,
            marker="o", markersize=4.5, label=m, zorder=3)

ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_xlabel("Input scale")
ax.set_ylabel(r"End-to-end speedup ($\times$)")
ax.set_xlim(-0.15, 2.4)
ax.set_ylim(0, max(max(r) for r in gm_per_model.values()) * 1.08)

ax.legend(loc="upper left", frameon=False, fontsize=8.0, ncol=2,
          handlelength=1.4, columnspacing=1.0, labelspacing=0.25)

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(False)
ax.tick_params(direction="out", length=3, width=0.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT)
print(f"\n[saved] {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
