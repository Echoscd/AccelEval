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
    "small":  {"n": 32768,  "num_offsets": 16},
    "medium": {"n": 131072, "num_offsets": 24},
    "large":  {"n": 262144, "num_offsets": 28},
}


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "gapbs_triangle_orderedcount" / "solution_cpu"
    src = orbench_root / "tasks" / "gapbs_triangle_orderedcount" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "gapbs_triangle_orderedcount" / "task_io_cpu.c"
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


def make_graph(n: int, num_offsets: int):
    if 2 * num_offsets >= n:
        raise ValueError("num_offsets must be less than n/2")
    u = np.arange(n, dtype=np.int64)[:, None]
    offs = np.arange(1, num_offsets + 1, dtype=np.int64)[None, :]
    plus = (u + offs) % n
    minus = (u - offs) % n
    nbrs = np.concatenate([minus, plus], axis=1)
    nbrs.sort(axis=1)
    col_idx = np.ascontiguousarray(nbrs.astype(np.int32, copy=False).reshape(-1))
    degree = 2 * num_offsets
    row_ptr = np.arange(0, (n + 1) * degree, degree, dtype=np.int64).astype(np.int32)
    return row_ptr, col_idx


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
    num_offsets = int(cfg["num_offsets"])

    row_ptr, col_idx = make_graph(n, num_offsets)
    m = int(col_idx.size)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("row_ptr", "int32", row_ptr),
            ("col_idx", "int32", col_idx),
        ],
        params={"n": n, "m": m},
    )

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        (out_dir / "cpu_time_ms.txt").write_text(f"{time_ms:.6f}\n", encoding="utf-8")
        run_cpu_expected_output(exe, out_dir)


if __name__ == "__main__":
    main()
