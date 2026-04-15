#!/usr/bin/env python3
"""
Generate ORBench v2 inputs for TransposedMatrixVectorProduct (CSC SpMV).

Computes answer = A^T * vector where A is stored CSC (ColMajor), matching
Google OR-Tools PDLP sharder.cc::TransposedMatrixVectorProduct().

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import sys
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import sparse

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"num_rows": 4096,   "num_cols": 4096,   "nnz_per_col": 16, "seed": 42},
    "medium": {"num_rows": 32768,  "num_cols": 32768,  "nnz_per_col": 24, "seed": 42},
    "large":  {"num_rows": 131072, "num_cols": 131072, "nnz_per_col": 32, "seed": 42},
}


def make_csc_instance(num_rows, num_cols, nnz_per_col, seed):
    """Generate a random CSC sparse matrix and dense vector."""
    rng = np.random.default_rng(seed)

    # Build CSC by generating each column's nonzero entries
    col_ptrs = np.zeros(num_cols + 1, dtype=np.int32)
    all_rows = np.arange(num_rows, dtype=np.int32)
    row_lists = []
    val_lists = []

    for j in range(num_cols):
        rows = rng.choice(all_rows, size=nnz_per_col, replace=False)
        rows.sort()
        vals = rng.uniform(-1.0, 1.0, size=nnz_per_col).astype(np.float32)
        # Diagonal bias for stability
        diag_loc = np.where(rows == min(j, num_rows - 1))[0]
        if diag_loc.size > 0:
            vals[diag_loc[0]] += np.float32(1.5 + 0.2 * rng.random())
        row_lists.append(rows)
        val_lists.append(vals)
        col_ptrs[j + 1] = col_ptrs[j] + nnz_per_col

    row_indices = np.concatenate(row_lists).astype(np.int32)
    values = np.concatenate(val_lists).astype(np.float32)

    # Dense input vector (matching or-tools VectorXd)
    vector = rng.normal(0.0, 1.0, size=num_rows).astype(np.float32)
    vector += np.linspace(-0.5, 0.5, num_rows, dtype=np.float32)

    return col_ptrs, row_indices, values, vector


def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "spmv_csr"
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


def run_cpu(exe: Path, data_dir: Path, validate=False):
    cmd = [str(exe), str(data_dir)]
    if validate:
        cmd.append("--validate")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CPU run failed:\n{r.stderr}\n{r.stdout}")
    m = re.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"TIME_MS not found in stdout:\n{r.stdout}")
    return float(m.group(1))


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
    num_rows    = cfg["num_rows"]
    num_cols    = cfg["num_cols"]
    nnz_per_col = cfg["nnz_per_col"]

    print(f"[gen_data] Generating {size_name}: "
          f"num_rows={num_rows}, num_cols={num_cols}, nnz_per_col={nnz_per_col}")

    col_ptrs, row_indices, values, vector = make_csc_instance(
        num_rows, num_cols, nnz_per_col, cfg["seed"])
    nnz = len(values)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("col_ptrs",    "int32",   col_ptrs),
            ("row_indices", "int32",   row_indices),
            ("values",      "float32", values),
            ("vector",      "float32", vector),
        ],
        params={
            "num_rows": num_rows,
            "num_cols": num_cols,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write("solve\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        cpu_ms = run_cpu(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{cpu_ms:.3f}\n")
        run_cpu(exe, out_dir, validate=True)
        shutil.copy2(out_dir / "output.txt", out_dir / "expected_output.txt")
        print(f"  nnz={nnz}, CPU baseline: {cpu_ms:.3f} ms")

    print(f"[gen_data] {size_name}: wrote files in {out_dir}")


if __name__ == "__main__":
    main()
