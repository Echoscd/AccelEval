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
    "small":  {"ni": 192, "nj": 192, "nk": 192, "nl": 192, "alpha_milli": 1500, "beta_milli": 1200},
    "medium": {"ni": 384, "nj": 384, "nk": 384, "nl": 384, "alpha_milli": 1500, "beta_milli": 1200},
    "large":  {"ni": 640, "nj": 640, "nk": 640, "nl": 640, "alpha_milli": 1500, "beta_milli": 1200},
}


def compile_cpu_baseline(orbench_root: Path) -> Path:
    exe = orbench_root / "tasks" / "polybench_2mm" / "solution_cpu"
    src = orbench_root / "tasks" / "polybench_2mm" / "cpu_reference.c"
    task_io_cpu = orbench_root / "tasks" / "polybench_2mm" / "task_io_cpu.c"
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


def make_inputs(ni: int, nj: int, nk: int, nl: int):
    # Adapted from PolyBench/C 2MM init_array in 2mm.c.
    A = np.empty((ni, nk), dtype=np.float32)
    B = np.empty((nk, nj), dtype=np.float32)
    C = np.empty((nj, nl), dtype=np.float32)
    D0 = np.empty((ni, nl), dtype=np.float32)

    inv_ni = np.float32(1.0 / float(ni))
    inv_nj = np.float32(1.0 / float(nj))
    inv_nl = np.float32(1.0 / float(nl))
    inv_nk = np.float32(1.0 / float(nk))

    for i in range(ni):
        for j in range(nk):
            A[i, j] = np.float32(((i * j + 1) % ni) * inv_ni)
    for i in range(nk):
        for j in range(nj):
            B[i, j] = np.float32((i * (j + 1) % nj) * inv_nj)
    for i in range(nj):
        for j in range(nl):
            C[i, j] = np.float32(((i * (j + 3) + 1) % nl) * inv_nl)
    for i in range(ni):
        for j in range(nl):
            D0[i, j] = np.float32((i * (j + 2) % nk) * inv_nk)

    return (
        np.ascontiguousarray(A.reshape(-1)),
        np.ascontiguousarray(B.reshape(-1)),
        np.ascontiguousarray(C.reshape(-1)),
        np.ascontiguousarray(D0.reshape(-1)),
    )


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
    nl = int(cfg["nl"])
    alpha_milli = int(cfg["alpha_milli"])
    beta_milli = int(cfg["beta_milli"])

    A, B, C, D0 = make_inputs(ni, nj, nk, nl)
    write_input_bin(
        str(out_dir / "input.bin"),
        tensors=[("A", "float32", A), ("B", "float32", B), ("C", "float32", C), ("D0", "float32", D0)],
        params={"ni": ni, "nj": nj, "nk": nk, "nl": nl, "alpha_milli": alpha_milli, "beta_milli": beta_milli},
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
