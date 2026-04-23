#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) — Generate random flow network for max-flow benchmark.

Generates a random directed graph with a layered structure ensuring
connectivity from source (0) to sink (num_nodes-1), with random capacities.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import sys
import re as re_mod
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small": {"num_nodes": 1000, "num_arcs": 5000, "seed": 42},
    "medium": {"num_nodes": 5000, "num_arcs": 25000, "seed": 42},
    "large": {"num_nodes": 20000, "num_arcs": 100000, "seed": 42},
}


def generate_flow_network(num_nodes, num_arcs, seed):
    """Generate a random flow network with source=0 and sink=num_nodes-1."""
    rng = np.random.RandomState(seed)
    source = 0
    sink = num_nodes - 1

    tails = []
    heads = []
    caps = []

    # Ensure connectivity: create a path source → ... → sink
    perm = np.arange(num_nodes)
    rng.shuffle(perm[1:-1])  # keep source=0, sink=N-1 in place
    for i in range(num_nodes - 1):
        tails.append(int(perm[i]))
        heads.append(int(perm[i + 1]))
        caps.append(int(rng.randint(1, 100)))

    # Fill remaining arcs randomly (avoid self-loops and duplicate arcs)
    existing = set(zip(tails, heads))
    remaining = num_arcs - len(tails)
    attempts = 0
    while remaining > 0 and attempts < remaining * 10:
        t = int(rng.randint(0, num_nodes))
        h = int(rng.randint(0, num_nodes))
        if t != h and (t, h) not in existing:
            tails.append(t)
            heads.append(h)
            caps.append(int(rng.randint(1, 100)))
            existing.add((t, h))
            remaining -= 1
        attempts += 1

    actual_arcs = len(tails)
    return (np.array(tails, dtype=np.int32),
            np.array(heads, dtype=np.int32),
            np.array(caps, dtype=np.int32),
            source, sink, actual_arcs)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "max_flow_push_relabel"
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

    cmd = [
        "gcc", "-O2",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re_mod.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}. Available: {list(SIZES.keys())}")

    cfg = SIZES[size_name]
    num_nodes = cfg["num_nodes"]
    num_arcs  = cfg["num_arcs"]

    print(f"[gen_data] Generating {size_name}: "
          f"num_nodes={num_nodes}, num_arcs={num_arcs}")

    tails, heads, caps, source, sink, actual_arcs = generate_flow_network(
        num_nodes, num_arcs, cfg["seed"])

    print(f"  actual_arcs={actual_arcs}, source={source}, sink={sink}")

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("tails", "int32", tails),
            ("heads", "int32", heads),
            ("caps",  "int32", caps),
        ],
        params={
            "num_nodes": num_nodes,
            "num_arcs":  actual_arcs,
            "source":    source,
            "sink":      sink,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write(f"{num_nodes}\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"  CPU time: {time_ms:.3f} ms")

    print(f"[gen_data] {size_name}: wrote files in {out_dir}")


if __name__ == "__main__":
    main()
