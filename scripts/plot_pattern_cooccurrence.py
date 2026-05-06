#!/usr/bin/env python3
"""Compute pattern co-occurrence across (model, task) cells and plot a
hierarchically-clustered heatmap. Produces nips/figs/pattern_cooccurrence.pdf
for the appendix.

Co-occurrence definition: for each ordered pair (P, Q), count the number of
(model, task) cells in which BOTH P and Q appear in the LLM analyzer's
pattern_summaries, divided by the number of cells in which P appears
(conditional probability P(Q | P)). Diagonal = 1 by construction.
The matrix is asymmetric; we plot it directly so readers can see e.g.
P(reduction | tile) versus P(tile | reduction).
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(ROOT))

PAPER_RUNS = {
    "Gemini":  "runs/gemini-3.1-pro-preview-openrouter_l3_20260421_1712/agent_pattern_analysis.json",
    "Claude":  "runs/claude-opus-4.6-openrouter_l3_20260421_1905/agent_pattern_analysis.json",
    "GPT":     "runs/openai/gpt-5.4_l3_20260421_1905/agent_pattern_analysis.json",
    "Qwen":    "runs/qwen/qwen3.6-plus_l3_20260421_1905/agent_pattern_analysis.json",
    "Kimi":    "runs/kimi-k2.5-openrouter_l3_20260421_1905/agent_pattern_analysis.json",
    "GLM":     "runs/glm-5.1-openrouter_l3_20260421_1905/agent_pattern_analysis.json",
    "Dpsk3.2": "runs/deepseek-v3.2-openrouter_l3_20260421_1905/agent_pattern_analysis.json",
    "Dpsk4":   "runs/deepseek-v4-pro-openrouter_l3_20260424_1706/agent_pattern_analysis.json",
}

# Build (model, task) -> set(pattern_id)
cell_patterns = {}
for model, fp in PAPER_RUNS.items():
    if not os.path.exists(fp):
        print(f"  missing: {fp}", file=sys.stderr)
        continue
    d = json.load(open(fp))
    for task, payload in d.items():
        if not isinstance(payload, dict): continue
        pids = set()
        for ps in payload.get("pattern_summaries", []):
            pid = ps.get("pattern_id")
            if pid: pids.add(pid)
        # Match LIFT-analysis filter: include only cells with >=1 detected pattern
        # so that co-occurrence uses the exact same 231-cell denominator as §5.2.
        if pids:
            cell_patterns[(model, task)] = pids

print(f"Total (model, task) cells with pattern data: {len(cell_patterns)}", file=sys.stderr)

# Build co-occurrence matrix
pattern_count = defaultdict(int)
pair_count = defaultdict(int)
for cell, pids in cell_patterns.items():
    for p in pids:
        pattern_count[p] += 1
    for p in pids:
        for q in pids:
            pair_count[(p, q)] += 1

# Keep only patterns observed in >= MIN_USE cells (filter rare ones for readable heatmap)
MIN_USE = 8
patterns = sorted([p for p, c in pattern_count.items() if c >= MIN_USE])
n = len(patterns)
print(f"Patterns observed in >= {MIN_USE} cells: {n}", file=sys.stderr)

# Conditional probability matrix M[i,j] = P(j | i) = count(i AND j) / count(i)
M = np.zeros((n, n))
for i, p in enumerate(patterns):
    base = pattern_count[p]
    for j, q in enumerate(patterns):
        M[i, j] = pair_count[(p, q)] / base if base > 0 else 0.0

# Hierarchical cluster on rows (and apply same order to cols, since this is roughly symmetric in usage)
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    # Use 1 - Jaccard similarity for clustering: use symmetric dist matrix
    sym = (M + M.T) / 2
    np.fill_diagonal(sym, 1.0)
    dist = 1.0 - sym
    np.fill_diagonal(dist, 0.0)
    # Make sure it's a valid distance matrix (symmetric, non-negative)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    cond = squareform(dist, checks=False)
    link = linkage(cond, method="average")
    order = list(leaves_list(link))
except Exception as e:
    print(f"Clustering failed ({e}); using alphabetical order", file=sys.stderr)
    order = list(range(n))

patterns_sorted = [patterns[i] for i in order]
M_sorted = M[np.ix_(order, order)]

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.0, 7.0))
im = ax.imshow(M_sorted, cmap="viridis", vmin=0, vmax=1, aspect="equal")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(patterns_sorted, rotation=90, fontsize=6)
ax.set_yticklabels(patterns_sorted, fontsize=6)
ax.set_xlabel("pattern $Q$ (column)", fontsize=8)
ax.set_ylabel("pattern $P$ (row)", fontsize=8)
ax.set_title("Pattern co-occurrence: $P(Q\\,|\\,P)$ across (model, task) cells\n"
             f"hierarchical-clustered, $n_{{cells}}={len(cell_patterns)}$, "
             f"patterns with $\\geq {MIN_USE}$ uses", fontsize=8)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("$P(Q\\,|\\,P)$", fontsize=8)
cbar.ax.tick_params(labelsize=6)

plt.tight_layout()
out = ROOT / "nips" / "figs" / "pattern_cooccurrence.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out), bbox_inches="tight", dpi=200)
print(f"Saved: {out}", file=sys.stderr)

# Also dump top co-occurring pattern pairs (high P(Q|P) but P != Q)
print("\nTop conditional co-occurrences (P -> Q, P(Q|P)):", file=sys.stderr)
pairs = []
for i, p in enumerate(patterns_sorted):
    for j, q in enumerate(patterns_sorted):
        if p == q: continue
        pairs.append((M_sorted[i, j], p, q, pattern_count[p]))
pairs.sort(reverse=True)
for r, p, q, base in pairs[:15]:
    print(f"  {p} -> {q}: {r:.2f}   (base count {p}={base})", file=sys.stderr)
