#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"rows": 256,  "cols": 256,  "iters": 50,  "seed": 7},
    "medium": {"rows": 768,  "cols": 768,  "iters": 100, "seed": 7},
    "large":  {"rows": 1536, "cols": 1536, "iters": 150, "seed": 7},
}

MAX_PD = np.float32(3.0e6)
PRECISION = np.float32(0.001)
SPEC_HEAT_SI = np.float32(1.75e6)
K_SI = np.float32(100.0)
FACTOR_CHIP = np.float32(0.5)
t_chip = np.float32(0.0005)
chip_height = np.float32(0.016)
chip_width = np.float32(0.016)
amb_temp = np.float32(80.0)


def generate_inputs(rows: int, cols: int, seed: int):
    rng = np.random.default_rng(seed)
    temp0 = rng.uniform(50.0, 90.0, size=(rows, cols)).astype(np.float32)
    power = rng.uniform(0.0, 2.0e6, size=(rows, cols)).astype(np.float32)
    return temp0.reshape(-1), power.reshape(-1)


def hotspot_reference(temp0: np.ndarray, power: np.ndarray, rows: int, cols: int, iters: int) -> np.ndarray:
    src = temp0.astype(np.float32).reshape(rows, cols).copy()
    dst = np.empty_like(src)

    grid_height = chip_height / np.float32(rows)
    grid_width = chip_width / np.float32(cols)
    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height
    Rx = grid_width / (np.float32(2.0) * K_SI * t_chip * grid_height)
    Ry = grid_height / (np.float32(2.0) * K_SI * t_chip * grid_width)
    Rz = t_chip / (K_SI * grid_height * grid_width)
    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    step = PRECISION / max_slope / np.float32(1000.0)
    Rx_1 = np.float32(1.0) / Rx
    Ry_1 = np.float32(1.0) / Ry
    Rz_1 = np.float32(1.0) / Rz
    Cap_1 = step / Cap
    p = power.reshape(rows, cols)

    for _ in range(iters):
        center = src
        north = np.empty_like(src); north[0, :] = src[0, :]; north[1:, :] = src[:-1, :]
        south = np.empty_like(src); south[-1, :] = src[-1, :]; south[:-1, :] = src[1:, :]
        west = np.empty_like(src); west[:, 0] = src[:, 0]; west[:, 1:] = src[:, :-1]
        east = np.empty_like(src); east[:, -1] = src[:, -1]; east[:, :-1] = src[:, 1:]

        dst[:, :] = center + Cap_1 * (
            p + (south + north - np.float32(2.0) * center) * Ry_1
              + (east + west - np.float32(2.0) * center) * Rx_1
              + (amb_temp - center) * Rz_1
        )

        # top row
        dst[0, 1:-1] = center[0, 1:-1] + Cap_1 * (
            p[0, 1:-1] + (east[0, 1:-1] + west[0, 1:-1] - np.float32(2.0) * center[0, 1:-1]) * Rx_1
            + (south[0, 1:-1] - center[0, 1:-1]) * Ry_1
            + (amb_temp - center[0, 1:-1]) * Rz_1)
        # bottom row
        dst[-1, 1:-1] = center[-1, 1:-1] + Cap_1 * (
            p[-1, 1:-1] + (east[-1, 1:-1] + west[-1, 1:-1] - np.float32(2.0) * center[-1, 1:-1]) * Rx_1
            + (north[-1, 1:-1] - center[-1, 1:-1]) * Ry_1
            + (amb_temp - center[-1, 1:-1]) * Rz_1)
        # left col
        dst[1:-1, 0] = center[1:-1, 0] + Cap_1 * (
            p[1:-1, 0] + (south[1:-1, 0] + north[1:-1, 0] - np.float32(2.0) * center[1:-1, 0]) * Ry_1
            + (east[1:-1, 0] - center[1:-1, 0]) * Rx_1
            + (amb_temp - center[1:-1, 0]) * Rz_1)
        # right col
        dst[1:-1, -1] = center[1:-1, -1] + Cap_1 * (
            p[1:-1, -1] + (south[1:-1, -1] + north[1:-1, -1] - np.float32(2.0) * center[1:-1, -1]) * Ry_1
            + (west[1:-1, -1] - center[1:-1, -1]) * Rx_1
            + (amb_temp - center[1:-1, -1]) * Rz_1)

        # corners
        dst[0, 0] = center[0, 0] + Cap_1 * (p[0, 0] + (center[0, 1] - center[0, 0]) * Rx_1 + (center[1, 0] - center[0, 0]) * Ry_1 + (amb_temp - center[0, 0]) * Rz_1)
        dst[0, -1] = center[0, -1] + Cap_1 * (p[0, -1] + (center[0, -2] - center[0, -1]) * Rx_1 + (center[1, -1] - center[0, -1]) * Ry_1 + (amb_temp - center[0, -1]) * Rz_1)
        dst[-1, 0] = center[-1, 0] + Cap_1 * (p[-1, 0] + (center[-1, 1] - center[-1, 0]) * Rx_1 + (center[-2, 0] - center[-1, 0]) * Ry_1 + (amb_temp - center[-1, 0]) * Rz_1)
        dst[-1, -1] = center[-1, -1] + Cap_1 * (p[-1, -1] + (center[-1, -2] - center[-1, -1]) * Rx_1 + (center[-2, -1] - center[-1, -1]) * Ry_1 + (amb_temp - center[-1, -1]) * Rz_1)

        src, dst = dst, src

    return src.reshape(-1)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "hotspot_2d"
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
    rows = int(cfg["rows"]); cols = int(cfg["cols"]); iters = int(cfg["iters"]); seed = int(cfg["seed"])
    temp0, power = generate_inputs(rows, cols, seed)
    write_input_bin(str(out_dir / "input.bin"),
                    tensors=[("temp0", "float32", temp0), ("power", "float32", power)],
                    params={"rows": rows, "cols": cols, "iters": iters, "seed": seed})
    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"rows={rows}\ncols={cols}\niters={iters}\nseed={seed}\n")
    if with_expected:
        expected = hotspot_reference(temp0, power, rows, cols, iters)
        with open(out_dir / "expected_output.txt", "w", encoding="utf-8") as f:
            for x in expected:
                f.write(f"{float(x):.6e}\n")
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        # sanity check against CPU baseline
        out_txt = out_dir / "output.txt"
        if out_txt.exists():
            out_txt.unlink()
        r = subprocess.run([str(exe), str(out_dir), "--validate"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"CPU baseline validate failed:\n{r.stderr}\n{r.stdout}")
        cpu_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w", encoding="utf-8") as f:
            f.write(f"{cpu_ms:.6f}\n")
    print(f"[gen_data] Done: {size_name} -> {out_dir}")

if __name__ == "__main__":
    main()
