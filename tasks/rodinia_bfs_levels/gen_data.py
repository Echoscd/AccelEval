#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"num_nodes": 20_000,  "avg_degree": 6, "seed": 11},
    "medium": {"num_nodes": 100_000, "avg_degree": 8, "seed": 11},
    "large":  {"num_nodes": 250_000, "avg_degree": 8, "seed": 11},
}


def generate_layered_graph(num_nodes: int, avg_degree: int, seed: int):
    rng = np.random.default_rng(seed)
    source = 0

    # layered partition for broad frontiers
    target_layers = max(16, int(np.sqrt(num_nodes) // 2))
    probs = rng.random(target_layers)
    probs /= probs.sum()
    sizes = np.maximum(1, (probs * num_nodes).astype(np.int64))
    sizes[0] = max(1, min(sizes[0], 8))
    diff = num_nodes - int(sizes.sum())
    sizes[-1] += diff
    if sizes[-1] <= 0:
        sizes[-1] = 1
        sizes[-2] -= 1

    starts = np.cumsum(np.concatenate([[0], sizes[:-1]])).astype(np.int64)
    layers = [np.arange(starts[i], starts[i] + sizes[i], dtype=np.int32) for i in range(target_layers)]

    adj = [[] for _ in range(num_nodes)]

    for li in range(target_layers):
        cur = layers[li]
        nxt = layers[min(li + 1, target_layers - 1)]
        for u in cur:
            deg = max(1, int(rng.poisson(avg_degree)))
            forward = max(1, deg // 2)
            same = max(0, deg - forward - 1)
            back = deg - forward - same

            if li + 1 < target_layers:
                vs = rng.choice(nxt, size=min(forward, len(nxt)), replace=len(nxt) < forward)
                adj[u].extend(int(v) for v in np.atleast_1d(vs))
            else:
                vs = rng.choice(cur, size=min(forward, len(cur)), replace=len(cur) < forward)
                adj[u].extend(int(v) for v in np.atleast_1d(vs) if int(v) != int(u))

            if same > 0 and len(cur) > 1:
                vs = rng.choice(cur, size=min(same, len(cur) - 1), replace=(len(cur) - 1) < same)
                adj[u].extend(int(v) for v in np.atleast_1d(vs) if int(v) != int(u))

            if back > 0 and li > 0:
                prev = layers[li - 1]
                vs = rng.choice(prev, size=min(back, len(prev)), replace=len(prev) < back)
                adj[u].extend(int(v) for v in np.atleast_1d(vs))

    # ensure reachability chain from source across layers
    rep = [int(layer[0]) for layer in layers]
    rep[0] = source
    for i in range(target_layers - 1):
        if rep[i + 1] not in adj[rep[i]]:
            adj[rep[i]].append(rep[i + 1])

    # deduplicate and build Rodinia-style arrays
    node_start = np.empty(num_nodes, dtype=np.int32)
    node_degree = np.empty(num_nodes, dtype=np.int32)
    edge_list = []
    cursor = 0
    for u in range(num_nodes):
        nbrs = sorted(set(v for v in adj[u] if 0 <= v < num_nodes and v != u))
        node_start[u] = cursor
        node_degree[u] = len(nbrs)
        edge_list.extend(nbrs)
        cursor += len(nbrs)

    edge_dst = np.asarray(edge_list, dtype=np.int32)
    return node_start, node_degree, edge_dst, source


def bfs_reference(node_start: np.ndarray, node_degree: np.ndarray, edge_dst: np.ndarray, source: int) -> np.ndarray:
    n = int(node_start.shape[0])
    dist = np.full(n, -1, dtype=np.int32)
    q = deque([source])
    dist[source] = 0
    while q:
        u = q.popleft()
        s = int(node_start[u])
        e = s + int(node_degree[u])
        du = int(dist[u])
        for v in edge_dst[s:e]:
            vv = int(v)
            if dist[vv] == -1:
                dist[vv] = du + 1
                q.append(vv)
    return dist


def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "rodinia_bfs_levels"
    exe = task_dir / "solution_cpu"
    src = task_dir / "cpu_reference.c"
    task_io_cpu = task_dir / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    sources = [src, task_io_cpu, harness]
    if exe.exists():
        try:
            exe_m = exe.stat().st_mtime
            if all(exe_m >= s.stat().st_mtime for s in sources):
                return exe
        except Exception:
            pass
    cmd = ["gcc", "-O2", "-I", str(orbench_root / "framework"), str(harness), str(task_io_cpu), str(src), "-o", str(exe), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    shutil.copy2(out_txt, data_dir / "expected_output.txt")


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)
    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SIZES[size_name]
    num_nodes = int(cfg["num_nodes"])
    avg_degree = int(cfg["avg_degree"])
    seed = int(cfg["seed"])

    node_start, node_degree, edge_dst, source = generate_layered_graph(num_nodes, avg_degree, seed)
    num_edges = int(edge_dst.shape[0])

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("node_start", "int32", node_start),
            ("node_degree", "int32", node_degree),
            ("edge_dst", "int32", edge_dst),
        ],
        params={
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "source": source,
            "seed": seed,
        },
    )

    if with_expected:
        dist = bfs_reference(node_start, node_degree, edge_dst, source)
        with open(out_dir / "expected_output.txt", "w", encoding="utf-8") as f:
            for x in dist:
                f.write(f"{int(x)}\n")

        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        cpu_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w", encoding="utf-8") as f:
            f.write(f"{cpu_ms:.6f}\n")
        run_cpu_expected_output(exe, out_dir)
    else:
        dist = bfs_reference(node_start, node_degree, edge_dst, source)
        with open(out_dir / "expected_output.txt", "w", encoding="utf-8") as f:
            for x in dist:
                f.write(f"{int(x)}\n")

    print(f"Generated {size_name} in {out_dir} (nodes={num_nodes}, edges={num_edges})")


if __name__ == "__main__":
    main()
