#!/usr/bin/env python3
"""
plot_scale_beta.py — for paper §6.2 Scale Sensitivity.

For each (model, task) where all three input scales pass, fit a log-log
OLS slope:
   log s_{m,t}(scale) = β_{m,t} · log T_cpu_baseline(scale) + α
β quantifies how much GPU advantage compounds with problem size.

Outputs nips/figs/scale_beta_hist.pdf — a single histogram of β with
median + zero markers. Used in §6.2 as the headline figure for the
"speedup grows with size" study.
"""
from __future__ import annotations

import json
import glob
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "nips" / "figs" / "scale_beta_hist.pdf"

f = sorted(glob.glob(str(ROOT / "runs" / "reports" / "acceleval_consolidated_*.json")))[-1]
print(f"[load] {Path(f).relative_to(ROOT)}")
d = json.loads(Path(f).read_text())
records = d["records"]
models = list(d["models"])

from collections import defaultdict
triples: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
for sz in ["small", "medium", "large"]:
    for t, pm in records[sz].items():
        for m, rec in pm.items():
            if not (rec.get("compiled") and rec.get("correct")):
                continue
            bm = rec.get("benchmark") or {}
            sp = bm.get("speedup_e2e")
            cpu = bm.get("cpu_baseline_ms")
            if sp and sp > 0 and cpu and cpu > 0:
                triples[(m, t)][sz] = (cpu, sp)

SIZES = ["small", "medium", "large"]
betas: list[float] = []
for (m, t), s2cs in triples.items():
    if not all(sz in s2cs for sz in SIZES):
        continue
    xs = [math.log10(s2cs[sz][0]) for sz in SIZES]   # log cpu_ms
    ys = [math.log10(s2cs[sz][1]) for sz in SIZES]   # log speedup
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        continue
    betas.append(num / den)

print(f"[fit] {len(betas)} (model, task) triples")

import statistics
b_med = statistics.median(betas)
b_mean = statistics.mean(betas)
b_std = statistics.stdev(betas)
frac_pos = sum(1 for b in betas if b > 0) / len(betas)

print(f"  median β = {b_med:+.3f}")
print(f"  mean β   = {b_mean:+.3f}  (stdev {b_std:.3f})")
print(f"  β > 0    = {frac_pos*100:.1f}%")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(4.6, 2.4))
bins = [x * 0.05 for x in range(-15, 26)]    # -0.75 to 1.25 step 0.05
ax.hist(betas, bins=bins, color="#4878A8", edgecolor="white", linewidth=0.4)
ax.axvline(0.0, color="#888", linestyle="--", linewidth=0.9, zorder=1)
ax.axvline(b_med, color="#C44E4E", linestyle="-", linewidth=1.4, zorder=3,
           label=f"median = {b_med:+.2f}")
ax.set_xlabel(r"Scaling exponent  $\beta = d\,\log s\,/\,d\,\log T_\mathrm{CPU}$")
ax.set_ylabel("# (model, task) pairs")
ax.set_xlim(-0.8, 1.3)
ax.text(0.02, 0.94,
        f"$n={len(betas)}$  triples\n"
        f"$\\beta>0$:  {frac_pos*100:.0f}%\n"
        f"mean = {b_mean:+.2f},  sd = {b_std:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.5))
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(direction="out", length=3, width=0.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT)
print(f"[saved] {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.1f} KB)")
