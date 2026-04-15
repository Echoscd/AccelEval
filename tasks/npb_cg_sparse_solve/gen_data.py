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
    "small":  {"n": 8192,   "neighbors_per_row": 12, "max_iters": 40, "tol_exp": 8, "seed": 42},
    "medium": {"n": 32768,  "neighbors_per_row": 16, "max_iters": 50, "tol_exp": 8, "seed": 42},
    "large":  {"n": 131072, "neighbors_per_row": 20, "max_iters": 60, "tol_exp": 8, "seed": 42},
}

def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "npb_cg_sparse_solve" / "solution_cpu"
    src = orbench_root / "tasks" / "npb_cg_sparse_solve" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "npb_cg_sparse_solve" / "task_io_cpu.c"
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

def build_spd_csr(n: int, neighbors_per_row: int, seed: int):
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for delta in (1, 2):
            j = (i + delta) % n
            adj[i].add(j)
            adj[j].add(i)
    target = max(neighbors_per_row, 4)
    for i in range(n):
        while len(adj[i]) < target:
            j = int(rng.integers(0, n))
            if j == i:
                continue
            adj[i].add(j)
            adj[j].add(i)
    row_ptr = [0]
    col_idx = []
    values = []
    for i in range(n):
        nbrs = sorted(adj[i])
        off_vals = -rng.uniform(0.01, 0.08, size=len(nbrs))
        diag = float(np.sum(np.abs(off_vals)) + 1.0)
        col_idx.append(i)
        values.append(diag)
        for j, v in zip(nbrs, off_vals):
            col_idx.append(j)
            values.append(float(v))
        row_ptr.append(len(col_idx))
    return (np.asarray(row_ptr, dtype=np.int32),
            np.asarray(col_idx, dtype=np.int32),
            np.asarray(values, dtype=np.float64))

def csr_matvec(row_ptr, col_idx, values, x):
    n = row_ptr.shape[0] - 1
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s, e = row_ptr[i], row_ptr[i + 1]
        y[i] = np.dot(values[s:e], x[col_idx[s:e]])
    return y

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
    neighbors_per_row = int(cfg["neighbors_per_row"])
    max_iters = int(cfg["max_iters"])
    tol_exp = int(cfg["tol_exp"])
    seed = int(cfg["seed"])

    row_ptr, col_idx, values = build_spd_csr(n, neighbors_per_row, seed)
    nnz = int(values.shape[0])
    rng = np.random.default_rng(seed + 1)
    x_true = rng.normal(0.0, 1.0, size=n).astype(np.float64)
    b = csr_matvec(row_ptr, col_idx, values, x_true)

    write_input_bin(str(out_dir / "input.bin"),
        tensors=[("row_ptr", "int32", row_ptr), ("col_idx", "int32", col_idx), ("values", "float64", values), ("b", "float64", b)],
        params={"n": n, "nnz": nnz, "max_iters": max_iters, "tol_exp": tol_exp})

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
