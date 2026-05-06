# ORBench Task Porting Guide — Extracting an Upstream Repo into the ORBench v2 Format

## Overview

Porting an existing CPU/GPU benchmark (e.g.\ FinanceBench, GROMACS) into an
ORBench v2 task requires producing seven files. This guide consolidates
lessons learnt from porting four FinanceBench tasks and the GROMACS
`nbnxm` kernel.

---

## 1. File manifest

```
tasks/<task_id>/
├── task.json              # Task metadata
├── cpu_reference.c        # CPU baseline (pure C)
├── gpu_baseline.cu        # GPU baseline (CUDA, expert-tuned)
├── task_io_cpu.c          # CPU I/O adapter
├── task_io.cu             # GPU I/O adapter
├── gen_data.py            # Data generator
├── prompt_template.yaml   # LLM prompt template
└── data/{small,medium,large}/  # Generated input / output / timing
```

---

## 2. Porting principles

### 2.1 Fidelity first

**Core principle: code structure and naming must stay faithful to the
upstream repo.**

| Must keep | May change |
|---|---|
| Original function names (e.g.\ `interestRateCompoundFactorCpu`) | Strip the `Cpu`/`Gpu` suffix and mark `static` |
| Original struct names (e.g.\ `bondsDateStruct`, `monteCarloOptionStruct`) | Field declaration order |
| Original algorithmic flow and call chain | Add a `static` qualifier where appropriate |
| Original constant names and values (e.g.\ `ERROR_FUNCT_pp0`) | Replace non-deterministic RNG with a deterministic one |
| Key explanatory comments from the original | File I/O and `main()` |

### 2.3 GPU baseline naming convention

Upstream code typically uses `Cpu`/`Gpu` suffixes to disambiguate the two
versions:

- CPU version: `interestRateCompoundFactorCpu` → keep the original name
  and mark it `static`.
- GPU version: `interestRateCompoundFactor` (suffix dropped) or
  `interestRateCompoundFactorGpu`.
- `__device__` helpers retain the original GPU naming.

---

## 3. Per-file porting steps

### 3.1 Step 1: Inspect the upstream code

```bash
# 1. List source files.
ls original_repo/CPU/

# 2. Enumerate function definitions.
grep "^[a-zA-Z].*(" original_repo/CPU/*.c | head -40

# 3. Enumerate structs.
grep "typedef struct" original_repo/CPU/*.h

# 4. Locate the data-generation logic (initialisation block in main()).
grep -n "rand\|srand\|for.*num\|malloc" original_repo/CPU/*Engine.c

# 5. Enumerate constants.
grep "#define" original_repo/CPU/*.h
```

### 3.2 Step 2: Author `task.json`

```json
{
  "task_id": "<task_name>",
  "name": "<human-readable name>",
  "category": "<category>",
  "difficulty": 1-4,
  "tags": ["tag1", "tag2"],

  "input_sizes": {
    "small":  {"N": 100000,   "seed": 42},
    "medium": {"N": 1000000,  "seed": 42},
    "large":  {"N": 10000000, "seed": 42}
  },

  "correctness": {
    "mode": "numerical",     // or "exact"
    "atol": 0.01,
    "rtol": 1e-4
  },

  "timing": {
    "warmup": 3,
    "trials": 10,
    "timeout": 60
  }
}
```

### 3.3 Step 3: Author `cpu_reference.c`

**Template structure:**

```c
// cpu_reference.c — <task name> CPU baseline.
//
// Faithfully ported from <path inside upstream repo>.
// Preserves original function names, struct definitions, and computation flow.
// NO file I/O, NO main(). All I/O is handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>  // include only if the original code uses bool

// ===== Original constants (from <upstream header>) =====
#define CONSTANT_NAME value

// ===== Original structs (from <upstream header>) =====
typedef float dataType;  // include only if the original code uses this typedef

typedef struct {
    // Keep original field names and order.
} originalStructName;

// ===== Module-level static state =====
static int g_N;
static const float* g_param1;
// ...

// ===== Functions ported from upstream (keep names, mark static) =====

static float originalFunctionNameCpu(float arg1, originalStruct arg2)
{
    // Copy the upstream logic line-for-line.
    // Adjust only: & → *, new → malloc, drop I/O.
}

// ... all upstream functions ...

// ===== Public interface (only these three are non-static) =====

void solution_init(int N, /* flattened parameters */)
{
    g_N = N;
    // Stash pointers into static variables.
}

void solution_compute(int N, float* output)
{
    // Reconstruct the upstream struct from flat arrays.
    // Call the upstream main compute routine.
    // Extract the result into `output`.
}

void solution_free(void)
{
    // Free anything allocated in solution_init (if applicable).
}
```

**Checklist:**
- [ ] All upstream function names preserved
- [ ] All upstream struct names preserved
- [ ] All upstream constant names and values preserved
- [ ] `typedef`s match upstream (e.g.\ `dataType`)
- [ ] No `#include` of upstream headers (the file is self-contained)
- [ ] No file I/O, no `printf`, no `main()`
- [ ] Pure C (no C++ features)
- [ ] Function-overloaded names disambiguated by renaming
- [ ] `rand()` replaced with a deterministic RNG (e.g.\ `xorshift32`)

### 3.4 Step 4: Author `task_io_cpu.c`

```c
// task_io_cpu.c — <task_id> CPU I/O adapter.

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

// Forward-declare the three solution functions from cpu_reference.c.
extern void solution_init(int N, ...);
extern void solution_compute(int N, float* output);
extern void solution_free(void);

typedef struct {
    int N;
    float* output;  // Output buffer.
} TaskContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    TaskContext* ctx = (TaskContext*)calloc(1, sizeof(TaskContext));
    ctx->N = (int)get_param(data, "N");

    // Pull every tensor out of input.bin.
    const float* param1 = get_tensor_float(data, "param1");
    const int*   param2 = get_tensor_int(data, "param2");
    // ...

    solution_init(ctx->N, param1, param2, ...);
    ctx->output = (float*)calloc(ctx->N, sizeof(float));
    return ctx;
}

void task_run(void* test_data) {
    TaskContext* ctx = (TaskContext*)test_data;
    solution_compute(ctx->N, ctx->output);
}

void task_write_output(void* test_data, const char* output_path) {
    TaskContext* ctx = (TaskContext*)test_data;
    FILE* fp = fopen(output_path, "w");
    for (int i = 0; i < ctx->N; i++)
        fprintf(fp, "%.6f\n", ctx->output[i]);  // or "%.6e\n"
    fclose(fp);
}

void task_cleanup(void* test_data) {
    TaskContext* ctx = (TaskContext*)test_data;
    solution_free();
    free(ctx->output);
    free(ctx);
}
```

### 3.5 Step 5: Author `task_io.cu`

Almost identical to `task_io_cpu.c`, plus:

```c
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
// extern declarations + the four task_* functions
#ifdef __cplusplus
}
#endif
```

### 3.6 Step 6: Author `gen_data.py`

```python
#!/usr/bin/env python3
"""gen_data.py — Generate <task_name> data for ORBench."""

import os, sys, subprocess, shutil, re as re_mod
from pathlib import Path
import numpy as np

_ORBENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORBENCH_ROOT))
from framework.orbench_io_py import write_input_bin

SIZES = {
    "small":  {"N": 100000,   "seed": 42},
    "medium": {"N": 1000000,  "seed": 42},
    "large":  {"N": 10000000, "seed": 42},
}

def generate_data(N, seed):
    """Generate the input tensors, mirroring the upstream initialisation logic."""
    rng = np.random.RandomState(seed)
    # ... build the data ...
    return tensors_dict

def compile_cpu_baseline(orbench_root: Path) -> Path:
    """Compile the CPU reference into an executable."""
    task_dir = orbench_root / "tasks" / "<task_id>"
    exe = task_dir / "solution_cpu"
    cmd = ["gcc", "-O2", "-I", str(orbench_root / "framework"),
           str(orbench_root / "framework" / "harness_cpu.c"),
           str(task_dir / "task_io_cpu.c"),
           str(task_dir / "cpu_reference.c"),
           "-o", str(exe), "-lm"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return exe

def run_cpu_time(exe: Path, data_dir: Path) -> float:
    """Run the CPU baseline and parse the wall-clock time."""
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    m = re_mod.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    return float(m.group(1))

def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    """Run the baseline with --validate to materialise expected_output.txt."""
    subprocess.run([str(exe), str(data_dir), "--validate"], check=True,
                   capture_output=True, text=True)
    shutil.copy2(data_dir / "output.txt", data_dir / "expected_output.txt")

def main():
    size_name, out_dir = sys.argv[1], Path(sys.argv[2])
    with_expected = "--with-expected" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SIZES[size_name]
    N = cfg["N"]

    data = generate_data(N, cfg["seed"])

    write_input_bin(str(out_dir / "input.bin"),
        tensors=[(name, dtype, arr) for name, (dtype, arr) in data.items()],
        params={"N": N})

    with open(out_dir / "requests.txt", "w") as f:
        f.write(f"{N}\n")

    if with_expected:
        exe = compile_cpu_baseline(_ORBENCH_ROOT)
        time_ms = run_cpu_time(exe, out_dir)
        with open(out_dir / "cpu_time_ms.txt", "w") as f:
            f.write(f"{time_ms:.3f}\n")
        run_cpu_expected_output(exe, out_dir)

if __name__ == "__main__":
    main()
```

### 3.7 Step 7: Author `gpu_baseline.cu`

```cuda
// gpu_baseline.cu — <task_name> GPU baseline.
//
// Faithfully ported from <upstream CUDA path>.
// __device__ functions keep their upstream GPU names (suffix-stripped or with a Gpu suffix).

#include <cuda_runtime.h>

// ===== Original structs (mirror cpu_reference.c) =====

// ===== __device__ helpers (preserve upstream GPU function names) =====
__device__ float originalFunctionName(...)  // GPU version: usually no suffix or a Gpu suffix
{
    // Same logic as the CPU helper, but fast-math intrinsics are allowed:
    //   __expf(), __logf(), erff(), rsqrtf(), ...
}

// ===== Main kernel =====
__launch_bounds__(256)
__global__ void mainKernel(int N, /* device pointers */, float* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // One thread per output sample.
    // Walk the __device__ helper chain.
    output[i] = result;
}

// ===== Persistent device memory =====
static float* d_input1 = nullptr;
static float* d_output = nullptr;

// ===== Public interface (signatures must match cpu_reference.c) =====
extern "C" void solution_init(int N, ...) {
    cudaMalloc(&d_input1, N * sizeof(float));
    cudaMemcpy(d_input1, host_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, N * sizeof(float));
}

extern "C" void solution_compute(int N, float* output) {
    mainKernel<<<(N+255)/256, 256>>>(N, d_input1, d_output);
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" void solution_free(void) {
    cudaFree(d_input1); d_input1 = nullptr;
    cudaFree(d_output); d_output = nullptr;
}
```

### 3.8 Step 8: Author `prompt_template.yaml`

```yaml
task_description: |
  <Problem statement: background, inputs/outputs, objective.>

interface: |
  ```c
  extern "C" void solution_init(int N, ...);
  extern "C" void solution_compute(int N, float* output);
  ```

input_size_notes: |
  | Size | N | Notes |
  |------|---|-------|

output_contract: |
  - output[i] = ...
  - Tolerance: atol=..., rtol=...

algorithm_background: |
  <Algorithm details, including pseudocode.>

hints_l2: |
  <Mid-level hints: parallelisation strategy, key optimisation directions.>

hints_l1: |
  <Detailed hints: kernel structure, specific optimisations, code snippets.>
```

---

## 4. Verification checklist

### 4.1 Compile

```bash
# CPU baseline
gcc -O2 -I framework framework/harness_cpu.c tasks/<task>/task_io_cpu.c \
    tasks/<task>/cpu_reference.c -o tasks/<task>/solution_cpu -lm

# GPU baseline
nvcc -O2 -arch=sm_89 -I framework framework/harness_gpu.cu \
    tasks/<task>/task_io.cu tasks/<task>/gpu_baseline.cu \
    -o tasks/<task>/gpu_baseline_exe
```

### 4.2 Generate data

```bash
python3 tasks/<task>/gen_data.py small tasks/<task>/data/small --with-expected
```

### 4.3 Numerical sanity check (against the upstream baseline)

```bash
# Compile and run the upstream code.
cd tasks/OriginalRepo/CPU && g++ -O2 -o orig *.c -lm && ./orig

# Compare key numbers.
python3 -c "
orig_val = ...  # value reported by the upstream binary
our_lines = open('tasks/<task>/data/small/expected_output.txt').readlines()
our_val = float(our_lines[0])
print(f'Original: {orig_val}, Ours: {our_val}, Diff: {abs(orig_val-our_val):.6f}')
"
```

### 4.4 GPU correctness check

```bash
./tasks/<task>/gpu_baseline_exe tasks/<task>/data/small --validate
# Compare output.txt against expected_output.txt.
python3 -c "
g = [float(l) for l in open('tasks/<task>/data/small/output.txt')]
e = [float(l) for l in open('tasks/<task>/data/small/expected_output.txt')]
max_err = max(abs(a-b) for a,b in zip(g,e))
print(f'Max abs error: {max_err:.6e}')
"
```

### 4.5 End-to-end LLM smoke test

```bash
python3 run.py generate-batch --models gemini-3-flash --tasks <task> --levels 2 --samples 1 --yes
python3 run.py eval --run <run_name> --sizes small
```

---

## 5. Common pitfalls and fixes

| Symptom | Cause | Fix |
|---|---|---|
| CPU compile error: `expected ';'` | Code uses C++ references `&` | Replace with pointer `*` |
| CPU compile error: `redefinition of 'func'` | Function overloading in upstream | Rename to `func` + `funcN` |
| Large GPU vs CPU divergence | Precision difference between `exp()` and `__expf()` | Use `expf()` instead of `__expf()`, or relax tolerance |
| Non-reproducible `rand()` | Upstream uses `srand(time(NULL))` | Replace with `xorshift32` + a fixed seed |
| `gen_data` output disagrees with the CPU baseline | Tensor name/type mismatch with the I/O adapter | Audit the `name` argument to every `get_tensor_*` call |
| GPU baseline segfaults | `cudaMalloc` lives inside `solution_compute` (re-allocates every call) | Move the allocation into `solution_init` |
| `task_io.cu` warning: `unused function` | A CPU-only helper exists in `task_io_cpu.c` | Delete from `task_io.cu` or guard with `#ifndef __CUDACC__` |

---

## 6. Porting case studies

| Upstream repo | ORBench task | Key challenge |
|---|---|---|
| FinanceBench / Black-Scholes | `black_scholes` | Polynomial `erf` → `erff()` intrinsic on GPU |
| FinanceBench / Monte-Carlo | `monte_carlo` | `rand()` → `xorshift32` deterministic RNG |
| FinanceBench / Bonds | `bonds_pricing` | Function overloading: `closeCpu` → `closeCpu` / `closeNcpu` |
| FinanceBench / Repo | `repo_pricing` | Reuses every Bonds helper plus repo-specific logic |
| GROMACS `nbnxm` | `nbnxm_forces` | Super-cluster data structure, warp-shuffle reduction |
