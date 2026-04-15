#!/usr/bin/env python3
"""
Generate ORBench input.bin + expected_output.txt + cpu_time_ms.txt for rodinia_lud_blocked.

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

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
    "small":  {"n": 256,  "seed": 42, "block_size": 16},
    "medium": {"n": 1024, "seed": 42, "block_size": 16},
    "large":  {"n": 1536, "seed": 42, "block_size": 16},
}


def generate_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n), dtype=np.float32)
    a = (x @ x.T).astype(np.float32)
    a += np.float32(n) * np.eye(n, dtype=np.float32)
    return a.astype(np.float32, copy=False)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "rodinia_lud_blocked" / "solution_cpu"
    src = orbench_root / "tasks" / "rodinia_lud_blocked" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "rodinia_lud_blocked" / "task_io_cpu.c"
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
        str(harness),
        str(task_io_cpu),
        str(src),
        "-o", str(exe),
        "-lm",
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


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)

    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    if size_name not in SIZES:
        raise ValueError(f"Unknown size: {size_name}")

    n = int(SIZES[size_name]["n"])
    seed = int(SIZES[size_name]["seed"])
    block_size = int(SIZES[size_name]["block_size"])
    if n % block_size != 0:
        raise ValueError("n must be a multiple of block_size")

    A0 = generate_matrix(n, seed)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("A0", "float32", A0.reshape(-1))],
        params={"n": n, "block_size": block_size},
    )

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote input.bin/expected_output.txt/cpu_time_ms.txt in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir} (expected/cpu_time skipped; pass --with-expected)")


if __name__ == "__main__":
    main()
