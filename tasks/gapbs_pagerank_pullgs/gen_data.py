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
    "small":  {"n": 32768,  "out_deg": 8,  "max_iters": 20},
    "medium": {"n": 131072, "out_deg": 10, "max_iters": 20},
    "large":  {"n": 262144, "out_deg": 12, "max_iters": 20},
}

MULTS = np.array([
    2654435761, 2246822519, 3266489917, 668265263,
    374761393, 1103515245, 122949829, 214013,
    134775813, 362437, 69069, 48271, 69621, 9301,
    1664525, 22695477
], dtype=np.uint64)
ADDS = np.array([
    1013904223, 3266489917, 374761393, 668265263,
    2147483647, 12345, 271828183, 314159265,
    161803399, 362436069, 521288629, 88675123,
    5783321, 6615241, 1234567, 7654321
], dtype=np.uint64)


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "gapbs_pagerank_pullgs" / "solution_cpu"
    src = orbench_root / "tasks" / "gapbs_pagerank_pullgs" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "gapbs_pagerank_pullgs" / "task_io_cpu.c"
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


def make_graph(n: int, out_deg: int):
    u = np.arange(n, dtype=np.uint64)
    dst_cols = []
    hub_cap = max(1, n // 128)
    for j in range(out_deg):
        if j == 0:
            v = (u + 1) % np.uint64(n)
        elif j == 1:
            v = (u + 7) % np.uint64(n)
        else:
            h = (u * MULTS[j % len(MULTS)] + ADDS[j % len(ADDS)]) % np.uint64(n)
            mod = j % 3
            if mod == 0:
                v = h % np.uint64(hub_cap)
            elif mod == 1:
                v = h
            else:
                v = (h + (u >> 2)) % np.uint64(n)
        same = (v == u)
        if np.any(same):
            v = v.copy()
            v[same] = (v[same] + 1) % np.uint64(n)
        dst_cols.append(v.astype(np.int32, copy=False))

    dst = np.stack(dst_cols, axis=1).reshape(-1)
    src = np.repeat(np.arange(n, dtype=np.int32), out_deg)

    order = np.argsort(dst, kind="mergesort")
    dst_sorted = dst[order]
    src_sorted = src[order]
    counts = np.bincount(dst_sorted, minlength=n).astype(np.int32)
    in_row_ptr = np.empty(n + 1, dtype=np.int32)
    in_row_ptr[0] = 0
    np.cumsum(counts, out=in_row_ptr[1:])
    in_col_idx = np.ascontiguousarray(src_sorted.astype(np.int32, copy=False))
    out_degree = np.full(n, out_deg, dtype=np.int32)
    return in_row_ptr, in_col_idx, out_degree


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
    out_deg = int(cfg["out_deg"])
    max_iters = int(cfg["max_iters"])

    in_row_ptr, in_col_idx, out_degree = make_graph(n, out_deg)
    m = int(in_col_idx.size)

    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[
            ("in_row_ptr", "int32", np.ascontiguousarray(in_row_ptr)),
            ("in_col_idx", "int32", np.ascontiguousarray(in_col_idx)),
            ("out_degree", "int32", np.ascontiguousarray(out_degree)),
        ],
        params={"n": n, "m": m, "max_iters": max_iters},
    )

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        (out_dir / "cpu_time_ms.txt").write_text(f"{time_ms:.6f}\n", encoding="utf-8")
        run_cpu_expected_output(exe, out_dir)


if __name__ == "__main__":
    main()
