#!/usr/bin/env python3
"""
gen_data.py (ORBench v2) — Generate random sparse LP data for PDLP benchmark.

Generates a random feasible LP in standard form:
  min c^T x  s.t.  l_c <= Ax <= u_c,  l_x <= x <= u_x
The constraint matrix A is stored in CSC (Compressed Sparse Column) format.
Step size is set to 1/||A||_2 (estimated via power iteration), matching
PDLP's default initialization (primal_dual_hybrid_gradient.cc).

Usage:
  python3 gen_data.py <size_name> <output_dir> [--with-expected]
"""

import os
import sys
import re as re_mod
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))

from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"num_vars": 500,   "num_constraints": 200,   "nnz_per_col": 4, "num_iters": 500,  "seed": 42},
    "medium": {"num_vars": 5000,  "num_constraints": 2000,  "nnz_per_col": 4, "num_iters": 1000, "seed": 42},
    "large":  {"num_vars": 50000, "num_constraints": 20000, "nnz_per_col": 4, "num_iters": 2000, "seed": 42},
}


def generate_lp(num_vars, num_constraints, nnz_per_col, seed):
    """Generate a random sparse LP with known feasible point."""
    rng = np.random.RandomState(seed)

    # Random sparse A in CSC format
    nnz = num_vars * nnz_per_col
    rows = rng.randint(0, num_constraints, size=nnz)
    cols = np.repeat(np.arange(num_vars), nnz_per_col)
    vals = rng.randn(nnz).astype(np.float32)
    A = sparse.csc_matrix((vals, (rows, cols)),
                          shape=(num_constraints, num_vars))
    # Ensure no duplicate entries
    A.sum_duplicates()

    # Feasible point
    x_feas = rng.uniform(0.0, 1.0, size=num_vars).astype(np.float32)
    Ax_feas = (A @ x_feas).astype(np.float32)

    # Variable bounds: [0, 2] (feasible point is in [0,1])
    var_lb = np.zeros(num_vars, dtype=np.float32)
    var_ub = np.full(num_vars, 2.0, dtype=np.float32)

    # Constraint bounds around the feasible Ax
    slack = 0.5
    con_lb = (Ax_feas - slack).astype(np.float32)
    con_ub = (Ax_feas + slack).astype(np.float32)

    # Random objective
    obj = rng.randn(num_vars).astype(np.float32) * 0.1

    # Estimate ||A||_2 for step size (matching PDLP initialization)
    try:
        sigma = svds(A.astype(np.float64), k=1, return_singular_vectors=False)[0]
    except Exception:
        sigma = float(sparse.linalg.norm(A))
    step_size = np.float32(1.0 / max(sigma, 1e-6))
    primal_weight = np.float32(1.0)

    return {
        "A": A,
        "obj": obj,
        "var_lb": var_lb,
        "var_ub": var_ub,
        "con_lb": con_lb,
        "con_ub": con_ub,
        "step_size": step_size,
        "primal_weight": primal_weight,
    }


# ---------------------------------------------------------------------------
# CPU baseline compile/run
# ---------------------------------------------------------------------------

def compile_cpu_baseline(orbench_root: Path) -> Path:
    task_dir = orbench_root / "tasks" / "pdlp"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    num_vars        = cfg["num_vars"]
    num_constraints = cfg["num_constraints"]
    nnz_per_col     = cfg["nnz_per_col"]
    num_iters       = cfg["num_iters"]

    print(f"[gen_data] Generating {size_name}: "
          f"vars={num_vars}, constraints={num_constraints}, "
          f"nnz_per_col={nnz_per_col}, iters={num_iters}")

    lp = generate_lp(num_vars, num_constraints, nnz_per_col, cfg["seed"])
    A = lp["A"]

    # Extract CSC arrays
    col_ptrs    = A.indptr.astype(np.int32)
    row_indices = A.indices.astype(np.int32)
    values      = A.data.astype(np.float32)
    nnz         = len(values)

    print(f"  nnz={nnz}, step_size={lp['step_size']:.6f}, "
          f"primal_weight={lp['primal_weight']:.6f}")

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("obj",            "float32", lp["obj"]),
            ("var_lb",         "float32", lp["var_lb"]),
            ("var_ub",         "float32", lp["var_ub"]),
            ("con_lb",         "float32", lp["con_lb"]),
            ("con_ub",         "float32", lp["con_ub"]),
            ("col_ptrs",       "int32",   col_ptrs),
            ("row_indices",    "int32",   row_indices),
            ("values",         "float32", values),
            ("step_size",      "float32", np.array([lp["step_size"]], dtype=np.float32)),
            ("primal_weight",  "float32", np.array([lp["primal_weight"]], dtype=np.float32)),
        ],
        params={
            "num_vars":        num_vars,
            "num_constraints": num_constraints,
            "nnz":             nnz,
            "num_iters":       num_iters,
        },
    )

    with open(out_dir / "requests.txt", "w") as f:
        f.write(f"{num_vars}\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)
        print(f"[gen_data] {size_name}: wrote all files in {out_dir}")
    else:
        print(f"[gen_data] {size_name}: wrote input.bin in {out_dir}")


if __name__ == "__main__":
    main()
