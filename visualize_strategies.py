#!/usr/bin/env python3
"""Visualize GPU optimization strategies across all AccelEval tasks."""

import json, os, re
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

TASKS_DIR = Path(__file__).parent / "tasks"

# ── Collect raw strategies ──────────────────────────────────────────
raw_strategies = {}
for tj in sorted(TASKS_DIR.glob("*/task.json")):
    task_id = tj.parent.name
    with open(tj) as f:
        d = json.load(f)
    raw_strategies[task_id] = d.get("gpu_optimization_points", [])

# ── Normalize strategies into canonical categories ──────────────────
# Map raw strategy strings to canonical categories
STRATEGY_CATEGORIES = {
    # Memory optimization
    "shared_memory": "Shared Memory",
    "shared_mem": "Shared Memory",
    "tiling": "Shared Memory Tiling",
    "shared_memory_tiling": "Shared Memory Tiling",
    "register": "Register Optimization",
    "register_blocking": "Register Optimization",
    "register_accumulation": "Register Optimization",
    "register_pressure": "Register Optimization",
    "coalesced": "Coalesced Memory Access",
    "memory_coalescing": "Coalesced Memory Access",
    "memory_bandwidth": "Memory Bandwidth Opt",
    "read_only": "Read-Only / Constant Memory",
    "constant_memory": "Read-Only / Constant Memory",
    "cache": "Read-Only / Constant Memory",
    # Parallelism patterns
    "wavefront": "Wavefront / Diagonal",
    "anti_diagonal": "Wavefront / Diagonal",
    "diagonal": "Wavefront / Diagonal",
    "batch": "Batch Parallelism",
    "one_thread_per": "Thread-per-Element",
    "thread_per": "Thread-per-Element",
    "parallel_per": "Thread-per-Element",
    "embarrassingly": "Thread-per-Element",
    # Reduction & synchronization
    "warp": "Warp-Level Primitives",
    "warp_level": "Warp-Level Primitives",
    "warp_shuffle": "Warp-Level Primitives",
    "warp_cooperative": "Warp-Level Primitives",
    "reduction": "Parallel Reduction",
    "atomic": "Atomic Operations",
    # Kernel design
    "persistent_kernel": "Persistent Kernel",
    "persistent": "Persistent Kernel",
    "kernel_fusion": "Kernel Fusion",
    "fused": "Kernel Fusion",
    "fusing": "Kernel Fusion",
    "double_buffer": "Double Buffering",
    "ping_pong": "Double Buffering",
    "rolling_buffer": "Double Buffering",
    # Compute
    "fast_math": "Fast Math Intrinsics",
    "fast_rsqrt": "Fast Math Intrinsics",
    "fma": "Fast Math Intrinsics",
    "intrinsic": "Fast Math Intrinsics",
    # Data structure
    "sort": "Parallel Sort / Scan",
    "scan": "Parallel Sort / Scan",
    "spatial": "Spatial Indexing",
    "grid": "Spatial Indexing",
    "neighbor": "Neighbor Search",
    "neighbor_list": "Neighbor Search",
    # Library
    "cublas": "cuBLAS/cuSPARSE",
    "cusparse": "cuBLAS/cuSPARSE",
    "cusolver": "cuBLAS/cuSPARSE",
    # Scheduling
    "load_balance": "Load Balancing",
    "load_balanced": "Load Balancing",
    "bucketing": "Load Balancing",
    "cuda_graph": "CUDA Graphs",
    "cooperative_groups": "CUDA Graphs",
    # Algorithm-specific
    "dp_buffer": "DP Buffer Management",
    "backward_induction": "Backward Induction DP",
    "bisection": "Binary Search on GPU",
    "binary_search": "Binary Search on GPU",
    "early_exit": "Early Termination",
    "stencil": "Stencil / Halo Exchange",
    "halo": "Stencil / Halo Exchange",
    "state_propagation": "State Propagation",
    "bitmask": "Bitmask Parallelism",
    # Additional mappings to reduce "Other"
    "precomput": "Precomputation",
    "precomput": "Precomputation",
    "rng": "Device-Side RNG",
    "device_side_rng": "Device-Side RNG",
    "beta_sampling": "Device-Side RNG",
    "device_side_date": "Device-Side Compute",
    "iterative_solver": "Device-Side Compute",
    "simple_arithmetic": "Device-Side Compute",
    "cost_evaluation": "Device-Side Compute",
    "greedy": "Algorithm-Specific Logic",
    "constraint": "Algorithm-Specific Logic",
    "seed_list": "Algorithm-Specific Logic",
    "newton_third": "Algorithm-Specific Logic",
    "csr_traversal": "Sparse Traversal (CSR)",
    "sparse_matrix": "Sparse Traversal (CSR)",
    "column_per": "Sparse Traversal (CSR)",
    "frontier": "Frontier / Phase Sync",
    "phase_sync": "Frontier / Phase Sync",
    "synchronize": "Frontier / Phase Sync",
    "k_blocking": "Blocking / Tiling",
    "boundary": "Boundary Handling",
    "avoid_realloc": "Memory Reuse",
    "workspace_reuse": "Memory Reuse",
    "cardinality": "Load Balancing",
}

def categorize_strategy(raw: str) -> list:
    """Map a raw strategy string to one or more canonical categories."""
    raw_lower = raw.lower().replace("-", "_")
    cats = set()
    for keyword, category in STRATEGY_CATEGORIES.items():
        if keyword in raw_lower:
            cats.add(category)
    if not cats:
        # Fallback: try to extract a meaningful label
        if "parallel" in raw_lower:
            cats.add("Data Parallelism")
        elif "memory" in raw_lower:
            cats.add("Memory Optimization")
        else:
            cats.add("Other")
    return list(cats)

# ── Load agent-detected patterns (from Gemini analysis) ─────────────
_agent_path = Path(__file__).parent / "runs" / "gemini-3.1-pro-preview-openrouter_l2_20260413" / "pattern_summary.json"
_use_agent = _agent_path.exists()

if _use_agent:
    _agent_data = json.load(open(_agent_path))
    cat_counter = Counter(_agent_data["pat_counter"])
    task_cats = {k: set(v) for k, v in _agent_data["task_patterns"].items()}
    cat_tasks = defaultdict(set)
    for task_id, pats in task_cats.items():
        for p in pats:
            cat_tasks[p].add(task_id)
else:
    # Fallback: keyword-based categorization
    task_cats = {}
    cat_counter = Counter()
    cat_tasks = defaultdict(set)
    for task_id, strategies in raw_strategies.items():
        cats = set()
        for s in strategies:
            for c in categorize_strategy(s):
                cats.add(c)
                cat_tasks[c].add(task_id)
        task_cats[task_id] = cats
        for c in cats:
            cat_counter[c] += 1

# ── Also get task categories for coloring ───────────────────────────
CAT_MAP = {
    "computational_finance": "Finance",
    "financial_computing": "Finance",
    "Financial Engineering": "Finance",
    "graph": "Graph",
    "dynamic_programming": "Dynamic Programming",
    "Dynamic Programming": "Dynamic Programming",
    "computational_geometry": "Geometry",
    "optimization": "Optimization",
    "Optimization": "Optimization",
    "parsing": "Parsing",
    "automata_simulation": "Parsing",
    "spatial_clustering": "Spatial/TimeSeries",
    "time_series_distance": "Spatial/TimeSeries",
    "time_series": "Spatial/TimeSeries",
    "spatial_distance": "Spatial/TimeSeries",
    "bioinformatics": "Spatial/TimeSeries",
    "scientific_computing": "Scientific",
    "molecular_dynamics": "Scientific",
    "fluid_simulation": "Scientific",
    "stencil_simulation": "Scientific",
    "linear_algebra": "LA/LP",
    "linear_programming": "LA/LP",
    "stochastic_simulation": "Stochastic",
    "probabilistic_inference": "Stochastic",
    "Transportation": "Graph",
}

DOMAIN_COLORS = {
    "Finance": "#FF6B6B",
    "Graph": "#4ECDC4",
    "Dynamic Programming": "#45B7D1",
    "Scientific": "#FFA07A",
    "Spatial/TimeSeries": "#BB86FC",
    "Optimization": "#FFD54F",
    "LA/LP": "#81C784",
    "Parsing": "#90CAF9",
    "Stochastic": "#F48FB1",
    "Geometry": "#CE93D8",
}

task_domain = {}
for tj in sorted(TASKS_DIR.glob("*/task.json")):
    tid = tj.parent.name
    with open(tj) as f:
        d = json.load(f)
    task_domain[tid] = CAT_MAP.get(d.get("category", ""), "Other")

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 18), facecolor="#1A1A2E")
fig.suptitle("AccelEval — GPU Optimization Strategy Landscape",
             fontsize=26, color="white", fontweight="bold", y=0.97)
_subtitle = f"{len(raw_strategies)} tasks  |  {len(cat_counter)} optimization patterns detected" + ("  (Agent-analyzed by Gemini)" if _use_agent else "")
fig.text(0.5, 0.94, _subtitle, ha="center", fontsize=13, color="#AAAACC")

# ── Panel 1: Strategy frequency bar chart (left) ────────────────────
ax1 = fig.add_axes([0.04, 0.08, 0.28, 0.82])
ax1.set_facecolor("#22223A")

top_cats = cat_counter.most_common()
labels, counts = zip(*top_cats)
y_pos = np.arange(len(labels))

# Color bars by strategy family
FAMILY_COLORS = {
    "Memory": "#FF6B6B",
    "Parallelism": "#4ECDC4",
    "Reduction": "#45B7D1",
    "Kernel": "#FFA07A",
    "Compute": "#FFD54F",
    "Data": "#BB86FC",
    "Library": "#81C784",
    "Algorithm": "#90CAF9",
    "Other": "#888888",
}
def get_family(cat):
    cat_lower = cat.lower()
    # Memory
    if any(k in cat_lower for k in ["memory", "coalescing", "cache", "__ldg",
            "layout", "aos", "soa", "__restrict__", "register"]):
        return "Memory"
    # Parallelism
    if any(k in cat_lower for k in ["thread", "parallelism", "wavefront",
            "batch", "coarsening", "element"]):
        return "Parallelism"
    # Reduction & sync
    if any(k in cat_lower for k in ["reduction", "atomic", "warp", "shuffle"]):
        return "Reduction"
    # Kernel design
    if any(k in cat_lower for k in ["persistent", "kernel f", "double buffer",
            "fission", "fusion", "cuda graph"]):
        return "Kernel"
    # Compute
    if any(k in cat_lower for k in ["fast math", "fma", "intrinsic", "pragma",
            "unroll", "branchless", "predicated", "compile-time"]):
        return "Compute"
    # Data & algorithm
    if any(k in cat_lower for k in ["sort", "scan", "spatial", "neighbor",
            "substitution", "algorithm", "reorder"]):
        return "Data"
    # Host-device
    if any(k in cat_lower for k in ["host", "transfer", "precomput", "broadcast",
            "scalar", "early exit", "minimize"]):
        return "Algorithm"
    return "Other"

bar_colors = [FAMILY_COLORS[get_family(l)] for l in labels]
bars = ax1.barh(y_pos, counts, color=bar_colors, edgecolor="#1A1A2E", height=0.75)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, color="white", fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("Number of Tasks Using Strategy", color="#AAAACC", fontsize=11)
ax1.set_title(f"All {len(labels)} Optimization Strategies", color="white", fontsize=16, pad=12)
ax1.tick_params(colors="#AAAACC")
for spine in ax1.spines.values():
    spine.set_color("#333355")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
for bar, val in zip(bars, counts):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             str(val), va="center", color="white", fontsize=10, fontweight="bold")

# Legend for families
from matplotlib.patches import Patch
family_patches = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items() if f != "Other"]
ax1.legend(handles=family_patches, loc="lower right", fontsize=8,
           facecolor="#22223A", edgecolor="#444466", labelcolor="white")

# ── Panel 2: Task × Strategy heatmap (right) ────────────────────────
ax2 = fig.add_axes([0.38, 0.08, 0.60, 0.82])
ax2.set_facecolor("#22223A")

# Use ALL strategies
top_strategy_names = [s for s, _ in cat_counter.most_common()]

# Sort tasks by domain
sorted_tasks = sorted(raw_strategies.keys(), key=lambda t: (task_domain.get(t, "ZZ"), t))

# Load intensity data if available
_intensity_path = Path(__file__).parent / "runs" / "gemini-3.1-pro-preview-openrouter_l2_20260413" / "pattern_summary.json"
_intensity_data = {}
if _intensity_path.exists():
    _idata = json.load(open(_intensity_path))
    _intensity_data = _idata.get("task_pattern_intensity", {})

matrix = np.zeros((len(sorted_tasks), len(top_strategy_names)), dtype=float)
for i, task in enumerate(sorted_tasks):
    for j, strat in enumerate(top_strategy_names):
        if _intensity_data and task in _intensity_data:
            val = _intensity_data[task].get(strat, 0)
            matrix[i, j] = val
        elif strat in task_cats.get(task, set()):
            matrix[i, j] = 1

# Continuous colormap: clear separation between 0 and non-zero
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
# Custom discrete colormap: 0=background, 1=light, 2-3=medium, 4+=bright
cmap = LinearSegmentedColormap.from_list("intensity", [
    "#22223A",   # 0: not used (matches background)
    "#163545",   # boundary
    "#1A5A6E",   # low
    "#238B9E",   # medium-low
    "#2CB5C8",   # medium
    "#45CCE0",   # medium-high
    "#6DE0EF",   # high
    "#95ECF5",   # very high
])
# Normalize each strategy (row in transposed matrix) by its max usage
matrix_norm = np.zeros_like(matrix, dtype=float)
for j in range(matrix.shape[1]):
    col_max = matrix[:, j].max()
    if col_max > 0:
        matrix_norm[:, j] = matrix[:, j] / col_max

# Push zeros below colormap start for clear separation
matrix_display = matrix_norm.copy()
matrix_display[matrix_display == 0] = -0.15

im = ax2.imshow(matrix_display.T, cmap=cmap, aspect="auto", interpolation="nearest",
                vmin=-0.15, vmax=1.0)

ax2.set_xticks(range(len(sorted_tasks)))
# Color task labels by domain
task_labels = []
for t in sorted_tasks:
    dom = task_domain.get(t, "Other")
    short = t[:18] + ".." if len(t) > 20 else t
    task_labels.append(short)
ax2.set_xticklabels(task_labels, color="white", fontsize=6.5, rotation=75, ha="right")

ax2.set_yticks(range(len(top_strategy_names)))
ax2.set_yticklabels(top_strategy_names, color="white", fontsize=7)
n_strats = len(top_strategy_names)
ax2.set_title(f"Strategy Intensity Matrix ({n_strats} strategies × {len(sorted_tasks)} tasks)",
              color="white", fontsize=16, pad=12)

# Add colorbar for intensity
cbar = fig.colorbar(im, ax=ax2, shrink=0.4, pad=0.01, aspect=20)
cbar.ax.tick_params(colors="#AAAACC", labelsize=8)
cbar.set_label("Usage Intensity", color="#AAAACC", fontsize=9)

# Color x-tick labels by domain
for i, t in enumerate(sorted_tasks):
    dom = task_domain.get(t, "Other")
    color = DOMAIN_COLORS.get(dom, "#888888")
    ax2.get_xticklabels()[i].set_color(color)

# Add domain color bar at bottom
for i, t in enumerate(sorted_tasks):
    dom = task_domain.get(t, "Other")
    color = DOMAIN_COLORS.get(dom, "#888888")
    ax2.add_patch(plt.Rectangle((i - 0.5, len(top_strategy_names) - 0.5),
                                 1, 0.3, color=color, clip_on=False))

# Grid lines
ax2.set_xticks([x - 0.5 for x in range(len(sorted_tasks) + 1)], minor=True)
ax2.set_yticks([y - 0.5 for y in range(len(top_strategy_names) + 1)], minor=True)
ax2.grid(which="minor", color="#444466", linewidth=0.5)
ax2.tick_params(which="minor", size=0)

# Add intensity labels in cells for high relative values
for i in range(len(sorted_tasks)):
    for j in range(len(top_strategy_names)):
        val = matrix_norm[i, j]
        if val >= 0.8:
            ax2.text(i, j, str(int(matrix[i, j])), ha="center", va="center",
                     color="white", fontsize=5.5, fontweight="bold")

# Domain legend
domain_patches = [Patch(facecolor=c, label=d) for d, c in DOMAIN_COLORS.items()]
ax2.legend(handles=domain_patches, loc="upper right", fontsize=7,
           facecolor="#22223A", edgecolor="#444466", labelcolor="white",
           title="Domain", title_fontsize=8)

# ── Save ────────────────────────────────────────────────────────────
out = Path(__file__).parent / "strategy_landscape.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"Saved: {out}")

# Print summary
print(f"\nTotal strategies annotated: {sum(len(v) for v in raw_strategies.values())}")
print(f"Canonical categories: {len(cat_counter)}")
print(f"\nTop 20:")
for s, c in cat_counter.most_common(20):
    family = get_family(s)
    print(f"  {s:<30s} {c:>3d} tasks  [{family}]")
