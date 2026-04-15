#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) - Generate miniWeather instances.

miniWeather has no external input data — all parameters are simulation constants.
This script creates a minimal input.bin with params and optionally runs the CPU
baseline to generate expected_output.txt.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin


SIZES = {
    "small":  {"NX": 100,  "NZ": 50,   "SIM_TIME": 400, "DATA_SPEC": 2},
    "medium": {"NX": 800,  "NZ": 400,  "SIM_TIME": 200, "DATA_SPEC": 2},
    "large":  {"NX": 2000, "NZ": 1000, "SIM_TIME": 100, "DATA_SPEC": 2},
}


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "miniWeather" / "solution_cpu"
    src = orbench_root / "tasks" / "miniWeather" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "miniWeather" / "task_io_cpu.c"
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
        "gcc", "-O2", "-DORBENCH_COMPUTE_ONLY",
        "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src),
        "-o", str(exe), "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline compile failed:\n{r.stderr}")
    return exe


def run_cpu_time(exe: Path, data_dir: Path, timeout: int = 600) -> float:
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


def run_cpu_expected_output(exe: Path, data_dir: Path, timeout: int = 600) -> None:
    out_txt = data_dir / "output.txt"
    if out_txt.exists():
        out_txt.unlink()
    r = subprocess.run([str(exe), str(data_dir), "--validate"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
    if not out_txt.exists():
        raise RuntimeError("output.txt not produced by CPU baseline")
    expected = data_dir / "expected_output.txt"
    shutil.copy2(out_txt, expected)


def estimate_cpu_time(cfg):
    """Rough estimate of CPU time in seconds."""
    nx, nz = cfg["NX"], cfg["NZ"]
    sim_time = cfg["SIM_TIME"]
    dx = 2e4 / nx
    dz = 1e4 / nz
    dt = min(dx, dz) / 450 * 1.5
    nsteps = int(sim_time / dt) + 1
    cells = nx * nz
    # ~6 semi_discrete_step per timestep, ~100 FLOPs per cell per step
    total_ops = nsteps * 6 * cells * 100
    return total_ops / 1e9  # rough: 1 GFLOP/s on single core


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
    NX = cfg["NX"]
    NZ = cfg["NZ"]
    SIM_TIME = cfg["SIM_TIME"]
    DATA_SPEC = cfg["DATA_SPEC"]

    dx = 2e4 / NX
    dz = 1e4 / NZ
    dt = min(dx, dz) / 450 * 1.5
    nsteps = int(SIM_TIME / dt) + 1

    print(f"[gen_data] {size_name}: NX={NX}, NZ={NZ}, SIM_TIME={SIM_TIME}, DATA_SPEC={DATA_SPEC}")
    print(f"[gen_data] dx={dx:.2f}m, dz={dz:.2f}m, dt={dt:.4f}s, ~{nsteps} steps")
    print(f"[gen_data] Grid cells: {NX*NZ:,}, state size: {NX*NZ*4*8/1e6:.1f} MB")
    est = estimate_cpu_time(cfg)
    print(f"[gen_data] Estimated CPU time: {est:.0f}s")

    # Write minimal input.bin (no tensor data, just params)
    # Use a dummy 1-element tensor so orbench_io doesn't complain about 0 tensors
    dummy = np.zeros(1, dtype=np.float32)
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("dummy", "float32", dummy),
        ],
        params={
            "NX": NX,
            "NZ": NZ,
            "SIM_TIME": SIM_TIME,
            "DATA_SPEC": DATA_SPEC,
        },
    )

    print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        est_timeout = max(120, int(est * 2))
        print(f"[gen_data] Running CPU baseline (timeout={est_timeout}s)...")
        try:
            time_ms = run_cpu_time(exe, out_dir, timeout=est_timeout)
            with open(out_dir / "cpu_time_ms.txt", "w") as f:
                f.write(f"{time_ms:.3f}\n")
            run_cpu_expected_output(exe, out_dir, timeout=est_timeout)
            print(f"[gen_data] {size_name}: CPU time={time_ms:.1f}ms, wrote expected_output.txt")
        except subprocess.TimeoutExpired:
            print(f"[gen_data] CPU baseline timed out ({est_timeout}s). Run manually.")
        except Exception as e:
            print(f"[gen_data] CPU baseline error: {e}")


if __name__ == "__main__":
    main()
