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
    "small": {"nx": 32, "ny": 32, "nz": 32, "coarse_levels": 3},
    "medium": {"nx": 56, "ny": 56, "nz": 56, "coarse_levels": 3},
    "large": {"nx": 72, "ny": 72, "nz": 72, "coarse_levels": 3},
}


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "hpcg_mg_vcycle" / "solution_cpu"
    src = orbench_root / "tasks" / "hpcg_mg_vcycle" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "hpcg_mg_vcycle" / "task_io_cpu.c"
    harness = orbench_root / "framework" / "harness_cpu.c"
    sources = [src, task_io_cpu, harness]
    if exe.exists():
        exe_m = exe.stat().st_mtime
        if all(exe_m >= s.stat().st_mtime for s in sources):
            return exe
    cmd = [
        "gcc", "-O2", "-I", str(orbench_root / "framework"),
        str(harness), str(task_io_cpu), str(src), "-o", str(exe), "-lm"
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


def generate_hpcg_27pt_csr(nx: int, ny: int, nz: int):
    n = nx * ny * nz
    row_ptr = np.empty(n + 1, dtype=np.int32)
    col_idx = np.empty(n * 27, dtype=np.int32)
    values = np.empty(n * 27, dtype=np.float64)
    diag_idx = np.empty(n, dtype=np.int32)

    ptr = 0
    row_ptr[0] = 0
    row = 0
    plane = nx * ny
    for iz in range(nz):
        for iy in range(ny):
            base_row = iz * plane + iy * nx
            for ix in range(nx):
                current = base_row + ix
                dpos = -1
                for sz in (-1, 0, 1):
                    z2 = iz + sz
                    if z2 < 0 or z2 >= nz:
                        continue
                    zoff = z2 * plane
                    for sy in (-1, 0, 1):
                        y2 = iy + sy
                        if y2 < 0 or y2 >= ny:
                            continue
                        yoff = zoff + y2 * nx
                        for sx in (-1, 0, 1):
                            x2 = ix + sx
                            if x2 < 0 or x2 >= nx:
                                continue
                            col = yoff + x2
                            col_idx[ptr] = col
                            if col == current:
                                values[ptr] = 26.0
                                dpos = ptr
                            else:
                                values[ptr] = -1.0
                            ptr += 1
                diag_idx[row] = dpos
                row += 1
                row_ptr[row] = ptr

    col_idx = np.ascontiguousarray(col_idx[:ptr])
    values = np.ascontiguousarray(values[:ptr])
    x_exact = np.ones(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rhs[i] = np.dot(values[row_ptr[i]:row_ptr[i+1]], x_exact[col_idx[row_ptr[i]:row_ptr[i+1]]])
    x_init = np.zeros(n, dtype=np.float64)
    return (
        np.ascontiguousarray(row_ptr),
        col_idx,
        values,
        np.ascontiguousarray(diag_idx),
        np.ascontiguousarray(rhs),
        x_init,
    )


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 gen_data.py <size_name> <output_dir> [--with-expected]")
        sys.exit(1)
    size_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    with_expected = len(sys.argv) == 4 and sys.argv[3] == "--with-expected"
    cfg = SIZES[size_name]
    out_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = int(cfg["nx"]), int(cfg["ny"]), int(cfg["nz"])
    coarse_levels = int(cfg["coarse_levels"])
    n = nx * ny * nz
    row_ptr, col_idx, values, diag_idx, rhs, x_init = generate_hpcg_27pt_csr(nx, ny, nz)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("row_ptr", "int32", row_ptr),
            ("col_idx", "int32", col_idx),
            ("values", "float64", values),
            ("diag_idx", "int32", diag_idx),
            ("rhs", "float64", rhs),
            ("x_init", "float64", x_init),
        ],
        params={"nx": nx, "ny": ny, "nz": nz, "n": n, "nnz": int(values.size), "coarse_levels": coarse_levels},
    )

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        t_ms = run_cpu_time(exe, out_dir)
        (out_dir / "cpu_time_ms.txt").write_text(f"{t_ms:.6f}\n", encoding="utf-8")
        run_cpu_expected_output(exe, out_dir)


if __name__ == "__main__":
    main()
