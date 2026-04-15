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
    "small":  {"ni": 256, "nj": 256, "nk": 256, "alpha_milli": 1500, "beta_milli": 1200},
    "medium": {"ni": 512, "nj": 512, "nk": 512, "alpha_milli": 1500, "beta_milli": 1200},
    "large":  {"ni": 768, "nj": 768, "nk": 768, "alpha_milli": 1500, "beta_milli": 1200},
}


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "polybench_gemm" / "solution_cpu"
    src = orbench_root / "tasks" / "polybench_gemm" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "polybench_gemm" / "task_io_cpu.c"
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


def make_inputs(ni: int, nj: int, nk: int):
    # Adapted from PolyBench/C GEMM init_array in gemm.c.
    C0 = np.empty((ni, nj), dtype=np.float32)
    A = np.empty((ni, nk), dtype=np.float32)
    B = np.empty((nk, nj), dtype=np.float32)

    inv_ni = np.float32(1.0 / float(ni))
    inv_nk = np.float32(1.0 / float(nk))
    inv_nj = np.float32(1.0 / float(nj))

    for i in range(ni):
        for j in range(nj):
            C0[i, j] = np.float32(((i * j + 1) % ni) * inv_ni)
    for i in range(ni):
        for j in range(nk):
            A[i, j] = np.float32((i * (j + 1) % nk) * inv_nk)
    for i in range(nk):
        for j in range(nj):
            B[i, j] = np.float32((i * (j + 2) % nj) * inv_nj)

    return np.ascontiguousarray(A.reshape(-1)), np.ascontiguousarray(B.reshape(-1)), np.ascontiguousarray(C0.reshape(-1))


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)
    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = (len(sys.argv) == 4 and sys.argv[3] == "--with-expected")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SIZES[size_name]
    ni = int(cfg["ni"])
    nj = int(cfg["nj"])
    nk = int(cfg["nk"])
    alpha_milli = int(cfg["alpha_milli"])
    beta_milli = int(cfg["beta_milli"])

    A, B, C0 = make_inputs(ni, nj, nk)
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("A", "float32", A), ("B", "float32", B), ("C0", "float32", C0)],
        params={"ni": ni, "nj": nj, "nk": nk, "alpha_milli": alpha_milli, "beta_milli": beta_milli},
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
