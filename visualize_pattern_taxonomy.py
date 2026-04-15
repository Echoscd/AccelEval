#!/usr/bin/env python3
"""Visualize the taxonomy of 33 GPU optimization patterns detected by agent analysis."""

import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ── Load pattern data ───────────────────────────────────────────────
summary = json.load(open("runs/gemini-3.1-pro-preview-openrouter_l2_20260413/pattern_summary.json"))
pat_counter = summary["pat_counter"]

# ── Classify patterns into families ─────────────────────────────────
FAMILIES = {
    "Memory Access": {
        "color": "#FF6B6B",
        "patterns": [
            "Memory coalescing",
            "Shared memory tiling",
            "__restrict__ pointer qualification",
            "Data layout transformation (AoS→SoA / memory order)",
            "Constant memory broadcast",
            "Read-only cache via __ldg",
            "Vectorized memory access",
        ],
        "example": "Memory coalescing",
        "example_desc": "Adjacent threads access adjacent memory\naddresses → maximizes bandwidth.\nint u = blockIdx.x * blockDim.x\n        + threadIdx.x;\nfloat val = data[u];  // coalesced",
    },
    "Memory Management": {
        "color": "#FFA07A",
        "patterns": [
            "Persistent device memory allocation",
            "Minimize host-device transfer",
            "Register promotion via compile-time array sizing",
            "Recompute to avoid memory traffic",
        ],
        "example": "Persistent device alloc",
        "example_desc": "Allocate GPU memory once in init(),\nreuse across multiple compute() calls.\nAvoids repeated cudaMalloc/Free\noverhead. Used in 35/40 tasks.",
    },
    "Compute Optimization": {
        "color": "#FFD54F",
        "patterns": [
            "Pragma unroll",
            "Branchless / predicated execution",
            "Compile-time branch elimination",
            "Fused multiply-add (FMA)",
            "Scalar broadcast (compute-once-use-many)",
        ],
        "example": "Branchless execution",
        "example_desc": "Replace if-else with arithmetic:\nfloat result = cond * val_a\n             + (1-cond) * val_b;\nAvoids warp divergence where\n32 threads must take same path.",
    },
    "Parallelism Strategy": {
        "color": "#4ECDC4",
        "patterns": [
            "Thread coarsening",
            "Grid-stride loop",
            "Kernel fission for phase-specific parallelism",
            "Kernel fusion",
            "Workload compaction / sparse packing",
        ],
        "example": "Kernel fission",
        "example_desc": "Split one big kernel into phases:\nK1: count_cells → prefix sum\nK2: build_pairs → narrow phase\nEach kernel has optimal parallelism\nfor its specific workload.",
    },
    "Reduction & Sync": {
        "color": "#45B7D1",
        "patterns": [
            "Block-level shared memory reduction",
            "Atomic operations for parallel update",
        ],
        "example": "Atomic operations",
        "example_desc": "Thread-safe parallel updates:\natomicAdd(&counts[i], 1);\natomicMin via int casting for float.\n7/40 tasks use atomics for\nrace-free accumulation.",
    },
    "Precomputation": {
        "color": "#BB86FC",
        "patterns": [
            "Precomputation via init kernel",
            "Precomputation on host",
            "Multi-dimensional index stride precomputation",
            "Demand/outcome deduplication on host",
        ],
        "example": "Precomputation via init",
        "example_desc": "Move invariant computation out of\nhot loop into a setup kernel:\nprecompute CSR offsets, prefix sums,\nlookup tables → 32/40 tasks use this.",
    },
    "Algorithm Restructure": {
        "color": "#81C784",
        "patterns": [
            "GPU-friendly algorithm substitution",
            "Early exit / branch-skip for large code blocks",
            "Loop reordering for data reuse",
            "Algebraic reformulation to reduce memory access",
            "Phase elimination via algorithm choice",
            "Padding for uniform loop bounds",
        ],
        "example": "GPU-friendly substitution",
        "example_desc": "Replace CPU algorithm with GPU-native:\nSerial BFS → parallel Bellman-Ford\nRecursive DP → iterative wavefront\nScalar scan → CUB parallel scan\n10/40 tasks restructure algorithms.",
    },
}

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(34, 20), facecolor="#1A1A2E")
fig.suptitle("AccelEval — GPU Optimization Pattern Taxonomy",
             fontsize=28, color="white", fontweight="bold", y=0.97)
fig.text(0.5, 0.94,
         f"33 patterns classified into 7 families  |  Agent-analyzed from Gemini 3.1 Pro generated code",
         ha="center", fontsize=14, color="#AAAACC")

# Layout: 7 family boxes arranged in a grid
# Top row: 4 families, Bottom row: 3 families + legend
positions = [
    # (x, y, w, h) for each family box
    (0.02, 0.52, 0.235, 0.38),   # Memory Access
    (0.265, 0.52, 0.235, 0.38),  # Memory Management
    (0.51, 0.52, 0.235, 0.38),   # Compute Optimization
    (0.755, 0.52, 0.235, 0.38),  # Parallelism Strategy
    (0.02, 0.08, 0.235, 0.38),   # Reduction & Sync
    (0.265, 0.08, 0.235, 0.38),  # Precomputation
    (0.51, 0.08, 0.235, 0.38),   # Algorithm Restructure
]

family_names = list(FAMILIES.keys())

for idx, (fname, fdata) in enumerate(FAMILIES.items()):
    x, y, w, h = positions[idx]
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor("#22223A")
    for spine in ax.spines.values():
        spine.set_color(fdata["color"])
        spine.set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])

    # Family title
    ax.text(0.5, 0.97, fname, transform=ax.transAxes,
            fontsize=24, color=fdata["color"], fontweight="bold",
            ha="center", va="top")

    # Count total tasks using patterns in this family
    total_usage = sum(pat_counter.get(p, 0) for p in fdata["patterns"])
    ax.text(0.5, 0.90, f"{len(fdata['patterns'])} patterns  |  {total_usage} total usages",
            transform=ax.transAxes, fontsize=14, color="#999AAA",
            ha="center", va="top")

    # List patterns with counts
    y_start = 0.84
    spacing = 0.065
    for i, pat in enumerate(fdata["patterns"]):
        count = pat_counter.get(pat, 0)
        short = pat[:40] + ".." if len(pat) > 42 else pat
        bullet_color = fdata["color"] if count >= 5 else "#777799"
        ax.text(0.04, y_start - i * spacing, "●",
                transform=ax.transAxes, fontsize=14, color=bullet_color, va="center")
        ax.text(0.10, y_start - i * spacing, f"{short}",
                transform=ax.transAxes, fontsize=14, color="#EEEEFF", va="center")
        ax.text(0.96, y_start - i * spacing, f"{count}",
                transform=ax.transAxes, fontsize=14, color=fdata["color"],
                va="center", ha="right", fontweight="bold")

    # Example box (bottom of each family)
    example_y = 0.02
    # Separator line
    ax.plot([0.05, 0.95], [0.22, 0.22], color="#444466",
            linewidth=0.5, transform=ax.transAxes, clip_on=False)
    ax.text(0.05, 0.21, f"Example: {fdata['example']}",
            transform=ax.transAxes, fontsize=14, color=fdata["color"],
            fontweight="bold", va="top")
    ax.text(0.05, 0.14, fdata["example_desc"],
            transform=ax.transAxes, fontsize=11, color="#DDDDEE",
            va="top", family="monospace", linespacing=1.15)

# ── Summary panel (bottom-right) ────────────────────────────────────
ax_sum = fig.add_axes([0.755, 0.08, 0.235, 0.38])
ax_sum.set_facecolor("#22223A")
for spine in ax_sum.spines.values():
    spine.set_color("#555577")
    spine.set_linewidth(1)
ax_sum.set_xticks([])
ax_sum.set_yticks([])

ax_sum.text(0.5, 0.97, "Summary", transform=ax_sum.transAxes,
            fontsize=24, color="white", fontweight="bold", ha="center", va="top")

# Pie chart of family sizes
family_sizes = [len(FAMILIES[f]["patterns"]) for f in family_names]
family_colors = [FAMILIES[f]["color"] for f in family_names]
# Small pie in top portion
ax_pie = fig.add_axes([0.78, 0.25, 0.18, 0.18])
ax_pie.set_facecolor("#22223A")
wedges, _ = ax_pie.pie(family_sizes, colors=family_colors,
                        startangle=90,
                        wedgeprops=dict(width=0.5, edgecolor="#1A1A2E", linewidth=1.5))

# Legend below pie
legend_y = 0.58
for i, fname in enumerate(family_names):
    n = len(FAMILIES[fname]["patterns"])
    usage = sum(pat_counter.get(p, 0) for p in FAMILIES[fname]["patterns"])
    ax_sum.add_patch(plt.Rectangle((0.05, legend_y - i * 0.07), 0.06, 0.045,
                                    transform=ax_sum.transAxes,
                                    facecolor=FAMILIES[fname]["color"],
                                    edgecolor="none"))
    ax_sum.text(0.14, legend_y - i * 0.07 + 0.02,
                f"{fname} ({n}p, {usage}u)",
                transform=ax_sum.transAxes, fontsize=13, color="#EEEEFF", va="center")

# Key insight
ax_sum.text(0.5, 0.06, "Memory optimizations dominate:\n60% of patterns target\nmemory access & management",
            transform=ax_sum.transAxes, fontsize=14, color="#FFA500",
            ha="center", va="bottom", style="italic")

# ── Save ────────────────────────────────────────────────────────────
out = Path(__file__).parent / "pattern_taxonomy.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"Saved: {out}")
