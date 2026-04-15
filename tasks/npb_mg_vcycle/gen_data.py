#!/usr/bin/env python3
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"interior_n": 64,  "num_cycles": 4, "pre_smooth": 2, "post_smooth": 2, "coarse_iters": 20, "seed": 42},
    "medium": {"interior_n": 128, "num_cycles": 5, "pre_smooth": 2, "post_smooth": 2, "coarse_iters": 24, "seed": 42},
    "large":  {"interior_n": 192, "num_cycles": 6, "pre_smooth": 2, "post_smooth": 2, "coarse_iters": 28, "seed": 42},
}

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "npb_mg_vcycle" / "solution_cpu"
    src = orbench_root / "tasks" / "npb_mg_vcycle" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "npb_mg_vcycle" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    sources = [src, task_io_cpu, harness]
    if exe.exists():
        exe_m = exe.stat().st_mtime
        if all(exe_m >= s.stat().st_mtime for s in sources):
            return exe
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
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}")
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
        raise RuntimeError(f"CPU baseline validate run failed:\n{r.stderr}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    shutil.copy2(out_txt, data_dir / "expected_output.txt")

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)
    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SIZES[size_name]
    n = int(cfg["interior_n"])
    total_n = n + 2
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    rhs = np.zeros((total_n, total_n, total_n), dtype=np.float64)
    x = np.linspace(0.0, 1.0, total_n, dtype=np.float64)
    y = np.linspace(0.0, 1.0, total_n, dtype=np.float64)
    z = np.linspace(0.0, 1.0, total_n, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    base = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    noise = rng.normal(0.0, 0.02, size=base.shape)
    rhs[1:-1, 1:-1, 1:-1] = (base + noise)[1:-1, 1:-1, 1:-1]

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("rhs", "float64", rhs.reshape(-1))],
        params={
            "interior_n": n,
            "num_cycles": int(cfg["num_cycles"]),
            "pre_smooth": int(cfg["pre_smooth"]),
            "post_smooth": int(cfg["post_smooth"]),
            "coarse_iters": int(cfg["coarse_iters"]),
        },
    )

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote input.bin/expected_output.txt/cpu_time_ms.txt in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")

if __name__ == "__main__":
    main()
