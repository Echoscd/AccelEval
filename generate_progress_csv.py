#!/usr/bin/env python3
"""
generate_progress_csv.py — Generate a CSV summary of all ORBench tasks.

Scans tasks/, data/, and runs/ to produce a progress overview.

Usage:
  python3 generate_progress_csv.py              # prints to stdout
  python3 generate_progress_csv.py -o status.csv # writes to file
"""

import os
import sys
import json
import glob
import csv
import re
from pathlib import Path

ORBENCH_ROOT = Path(__file__).resolve().parent
TASKS_DIR = ORBENCH_ROOT / "tasks"
RUNS_DIR = ORBENCH_ROOT / "runs"

# ── Known problem sources (manually curated) ─────────────────────────────
PROBLEM_SOURCES = {
    "bellman_ford": "Bellman (1958). Classic graph SSSP algorithm. Textbook.",
    "collision_detection": "Separating Axis Theorem. Computational geometry standard.",
    "regex_match": "Thompson (1968). NFA-based regex matching. Classic CS.",
    "network_rm_dp": "Jasin (2014, Operations Research). Network revenue management DP.",
    "thompson_sampling": "Agrawal & Goyal (2012, COLT/JMLR 2017). Thompson Sampling regret analysis.",
    "miniWeather": "Matt Norman, ORNL. https://github.com/mrnorman/miniWeather",
    "crew_pairing": "Airline crew scheduling. Set partitioning formulation.",
    "gittins_index": "Gittins (1979). Optimal allocation index for bandits.",
    "hawkes_dynamic_pricing_hjb": "Hawkes process dynamic pricing. HJB PDE solver.",
    "inventory_replenishment_dp": "Multi-period inventory control. Stochastic DP.",
    "motzkin_straus_blp_eval": "Motzkin & Straus (1965). Clique relaxation via bilevel programming.",
    "self_exciting_pricing_dp": "Self-exciting point process pricing. Finite difference DP.",
    "nash_flows_over_time": "Nash (1950) / Ford-Fulkerson. Dynamic traffic assignment.",
    "robust_value_iteration_hypercube": "Iyengar (2005). Robust MDP with hypercube uncertainty.",
    "batched_lhpca_portfolio": "Long-history PCA for portfolio optimization. Financial engineering.",
    "euclidean_distance_matrix": "Garcia, Debreuve & Barlaud, ICIP 2010. kNN-CUDA compute_distances kernel (vincentfpgarcia/kNN-CUDA).",
    "dtw_distance":              "Schmidt & Hundt, Euro-Par 2020. cuDTW++ SHFL_FULLDTW_1023 length-specialized warp-shuffle wavefront kernel (asbschmidt/cuDTW).",
    "hausdorff_distance":        "cuSpatial directed_hausdorff_distance (NVIDIA RAPIDS, rapidsai/cuspatial). Thread-per-LHS-point with atomicMax.",
    "pdlp":                      "Applegate et al., NeurIPS 2021 (PDLP). Google OR-Tools PDHG LP solver (ortools/pdlp/primal_dual_hybrid_gradient.cc).",
    "spmv_csr":                  "Google OR-Tools PDLP sharder.cc TransposedMatrixVectorProduct. CSR row-gather equivalent of CSC column-gather SpMV.",
    "held_karp_tsp":             "Held & Karp (1962). Google OR-Tools hamiltonian_path.h with LatticeMemoryManager and Gosper's hack.",
    "max_flow_push_relabel":     "Goldberg & Tarjan (1986). Google OR-Tools GenericMaxFlow (ortools/graph/generic_max_flow.h). Push-relabel with GlobalUpdate BFS.",
}

# ── Input/output type descriptions ────────────────────────────────────────
IO_TYPES = {
    "bellman_ford":       {"input": "CSR graph + (src,dst) queries", "output": "shortest distances (float)"},
    "collision_detection": {"input": "2D convex polygons + AABBs", "output": "collision counts per polygon (int)"},
    "regex_match":        {"input": "NFA automaton + string batch", "output": "match results per string (int)"},
    "network_rm_dp":      {"input": "RM instance (capacity, menus, demand)", "output": "value function V[1][s] (float)"},
    "thompson_sampling":  {"input": "arm means + seed", "output": "avg regret + avg pull counts (float)"},
    "miniWeather":        {"input": "simulation params (no data)", "output": "conservation metrics + L2 norms (double)"},
    "crew_pairing":       {"input": "flight legs + pairings", "output": "optimal crew schedule cost (float)"},
    "gittins_index":      {"input": "Beta grid params", "output": "Gittins index table (float)"},
    "hawkes_dynamic_pricing_hjb": {"input": "HJB PDE grid params", "output": "value function grid (float)"},
    "inventory_replenishment_dp": {"input": "inventory DP params", "output": "optimal cost table (float)"},
    "motzkin_straus_blp_eval":    {"input": "graph adjacency + trial points", "output": "BLP objective values (float)"},
    "self_exciting_pricing_dp":   {"input": "pricing DP grid params", "output": "value function grid (float)"},
    "nash_flows_over_time":       {"input": "time-expanded network graph", "output": "flow assignments (float)"},
    "robust_value_iteration_hypercube": {"input": "MDP transition + reward", "output": "robust value function (float)"},
    "batched_lhpca_portfolio":    {"input": "return history matrices", "output": "PCA weights (float)"},
    "euclidean_distance_matrix": {"input": "column-major ref [dim,ref_nb] + query [dim,query_nb] (float)", "output": "squared L2 distance matrix [ref_nb,query_nb] (float)"},
    "dtw_distance":              {"input": "subjects [num_entries,1023] + query [1023] in __constant__ (float)", "output": "DTW distance per subject (float)"},
    "hausdorff_distance":        {"input": "2D points + per-space offsets (float/int32)", "output": "directed Hausdorff matrix [num_spaces,num_spaces] (float)"},
    "pdlp":                      {"input": "sparse LP (CSC matrix A, obj, bounds)", "output": "averaged primal solution after K PDHG iters (float)"},
    "spmv_csr":                  {"input": "CSR sparse matrix (row_ptr, col_idx, vals) + dense x", "output": "dense y = A*x (float)"},
    "held_karp_tsp":             {"input": "B batches of n*n cost matrices (int)", "output": "TSP tour cost per batch (int)"},
    "max_flow_push_relabel":     {"input": "directed graph (tails, heads, caps) + source/sink", "output": "maximum flow value (int)"},
}


def read_file_float(path):
    """Read a single float from a file, return None if missing."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except Exception:
        return None


def format_time(ms):
    """Format milliseconds into human-readable string."""
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    elif ms < 3600000:
        return f"{ms/60000:.1f}min"
    else:
        return f"{ms/3600000:.1f}h"


def scan_best_speedup(task_id):
    """Scan runs/ for the best speedup achieved on this task (any size)."""
    best_speedup = None
    best_info = ""

    # Method 1: eval_results.json
    for ej in glob.glob(str(RUNS_DIR / "*/eval_results.json")):
        try:
            with open(ej) as f:
                results = json.load(f)
            for r in results:
                if r.get("task_id") != task_id:
                    continue
                if not r.get("correct"):
                    continue
                bm = r.get("benchmark")
                if not bm:
                    continue
                sp = bm.get("speedup_e2e")
                if sp and (best_speedup is None or sp > best_speedup):
                    best_speedup = sp
                    run_name = Path(ej).parent.name
                    best_info = f"{sp:.1f}x ({run_name})"
        except Exception:
            continue

    # Method 2: agent progress / timing.json in runs
    for tj in glob.glob(str(RUNS_DIR / f"*/{task_id}/*/timing.json")):
        try:
            with open(tj) as f:
                timing = json.load(f)
            mean_ms = timing.get("mean_ms")
            if mean_ms and mean_ms > 0:
                # Try to find which size this is
                for size in ["small", "medium", "large"]:
                    cpu_f = TASKS_DIR / task_id / "data" / size / "cpu_time_ms.txt"
                    cpu_ms = read_file_float(cpu_f)
                    if cpu_ms:
                        sp = cpu_ms / mean_ms
                        if sp > 1 and (best_speedup is None or sp > best_speedup):
                            best_speedup = sp
                            run_name = Path(tj).parents[1].name
                            best_info = f"{sp:.1f}x ({run_name})"
        except Exception:
            continue

    # Method 3: scan timing.json directly in data dirs (from manual runs)
    for size in ["small", "medium", "large"]:
        tj = TASKS_DIR / task_id / "data" / size / "timing.json"
        cpu_f = TASKS_DIR / task_id / "data" / size / "cpu_time_ms.txt"
        if tj.exists() and cpu_f.exists():
            try:
                with open(tj) as f:
                    timing = json.load(f)
                mean_ms = timing.get("mean_ms")
                cpu_ms = read_file_float(cpu_f)
                if mean_ms and cpu_ms and mean_ms > 0:
                    sp = cpu_ms / mean_ms
                    if sp > 1 and (best_speedup is None or sp > best_speedup):
                        best_speedup = sp
                        best_info = f"{sp:.1f}x ({size})"
            except Exception:
                continue

    return best_info


def scan_gpu_baselines(task_id):
    """Find GPU baseline timings (e.g. YAKL)."""
    baselines = []
    for size in ["small", "medium", "large"]:
        data_dir = TASKS_DIR / task_id / "data" / size
        for f in sorted(data_dir.glob("*_time_ms.txt")):
            name = f.stem.replace("_time_ms", "")
            if name == "cpu":
                continue
            ms = read_file_float(f)
            if ms is not None:
                baselines.append(f"{name}({size})={format_time(ms)}")
    return "; ".join(baselines) if baselines else ""


def generate_csv(output_path=None):
    # Discover all tasks
    task_ids = sorted([
        p.parent.name for p in TASKS_DIR.glob("*/task.json")
    ])

    rows = []
    for task_id in task_ids:
        tj_path = TASKS_DIR / task_id / "task.json"
        with open(tj_path) as f:
            tj = json.load(f)

        # CPU times
        cpu_small  = read_file_float(TASKS_DIR / task_id / "data/small/cpu_time_ms.txt")
        cpu_medium = read_file_float(TASKS_DIR / task_id / "data/medium/cpu_time_ms.txt")
        cpu_large  = read_file_float(TASKS_DIR / task_id / "data/large/cpu_time_ms.txt")

        # Best speedup from experiments
        best_speedup = scan_best_speedup(task_id)

        # GPU baselines
        gpu_baselines = scan_gpu_baselines(task_id)

        # Problem source
        source = PROBLEM_SOURCES.get(task_id, "")

        # IO types
        io = IO_TYPES.get(task_id, {"input": "", "output": ""})

        row = {
            "Task ID": task_id,
            "Name": tj.get("name", ""),
            "Domain": tj.get("category", ""),
            "Difficulty": tj.get("difficulty", ""),
            "Interface": tj.get("interface_mode", "init_compute"),
            "CPU Small": format_time(cpu_small),
            "CPU Medium": format_time(cpu_medium),
            "CPU Large": format_time(cpu_large),
            "Best Speedup": best_speedup,
            "GPU Baseline": gpu_baselines,
            "Input Type": io["input"],
            "Output Type": io["output"],
            "Problem Source": source,
        }
        rows.append(row)

    # Write CSV
    fieldnames = [
        "Task ID", "Name", "Domain", "Difficulty", "Interface",
        "CPU Small", "CPU Medium", "CPU Large",
        "Best Speedup", "GPU Baseline",
        "Input Type", "Output Type", "Problem Source",
    ]

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} tasks to {output_path}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate ORBench progress CSV")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: stdout)")
    args = parser.parse_args()
    generate_csv(args.output)
