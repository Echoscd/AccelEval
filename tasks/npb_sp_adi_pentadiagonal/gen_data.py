#!/usr/bin/env python3
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
    "small":  {"n": 24, "iters": 6,  "omega_milli": 900, "seed": 42},
    "medium": {"n": 40, "iters": 8,  "omega_milli": 900, "seed": 42},
    "large":  {"n": 56, "iters": 10, "omega_milli": 900, "seed": 42},
}


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "npb_sp_adi_pentadiagonal" / "solution_cpu"
    src = orbench_root / "tasks" / "npb_sp_adi_pentadiagonal" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "npb_sp_adi_pentadiagonal" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    sources = [src, task_io_cpu, harness]
    if exe.exists():
        exe_m = exe.stat().st_mtime
        if all(exe_m >= s.stat().st_mtime for s in sources):
            return exe
    cmd = ["gcc", "-O2", "-I", str(orbench_root / "framework"), str(harness), str(task_io_cpu), str(src), "-o", str(exe), "-lm"]
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


def make_inputs(n: int, seed: int):
    rng = np.random.default_rng(seed)
    shape = (n, n, n, 5)
    u0 = rng.normal(loc=0.0, scale=0.03, size=shape).astype(np.float64)
    rhs = rng.normal(loc=0.0, scale=0.08, size=shape).astype(np.float64)

    coords = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    forcing = np.stack([
        0.20 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y),
        0.18 * np.cos(2.0 * np.pi * Y) * np.sin(2.0 * np.pi * Z),
        0.16 * np.sin(2.0 * np.pi * Z) * np.cos(2.0 * np.pi * X),
        0.12 * (X - 0.5) * (Y - 0.5),
        0.10 * (Y - 0.5) * (Z - 0.5),
    ], axis=-1)
    rhs += forcing
    return np.ascontiguousarray(u0.reshape(-1)), np.ascontiguousarray(rhs.reshape(-1))


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)
    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SIZES[size_name]
    n = int(cfg["n"])
    iters = int(cfg["iters"])
    omega_milli = int(cfg["omega_milli"])
    seed = int(cfg["seed"])

    u0, rhs = make_inputs(n, seed)
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("u0", "float64", u0), ("rhs", "float64", rhs)],
        params={"n": n, "iters": iters, "omega_milli": omega_milli},
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
