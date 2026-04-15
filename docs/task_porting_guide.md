# ORBench Task Porting Guide — 从原始 Repo 提取代码到 ORBench v2 格式

## 概览

将一个已有的 CPU/GPU benchmark（如 FinanceBench、GROMACS）移植为 ORBench v2 task，
需要生成 7 个文件。本指南总结了从 FinanceBench 四个任务和 GROMACS nbnxm 的移植经验。

---

## 1. 文件清单

```
tasks/<task_id>/
├── task.json              # 任务元数据
├── cpu_reference.c        # CPU baseline（纯 C）
├── gpu_baseline.cu        # GPU baseline（CUDA，人类专家水平）
├── task_io_cpu.c          # CPU I/O 适配器
├── task_io.cu             # GPU I/O 适配器
├── gen_data.py            # 数据生成脚本
├── prompt_template.yaml   # LLM 提示模板
└── data/{small,medium,large}/  # 生成的数据目录
```

---

## 2. 移植原则

### 2.1 忠实度优先

**核心原则：代码结构和命名必须忠实于原始 repo。**

| 必须保留 | 允许修改 |
|---------|---------|
| 原版函数名（如 `interestRateCompoundFactorCpu`） | 去掉 `Cpu`/`Gpu` 后缀改为 `static` |
| 原版结构体名（如 `bondsDateStruct`, `monteCarloOptionStruct`） | 结构体字段顺序 |
| 原版算法流程和调用链 | 添加 `static` 关键字 |
| 原版常量名和值（如 `ERROR_FUNCT_pp0`） | 非确定性 RNG 替换为确定性 RNG |
| 原版注释中的关键说明 | 文件 I/O、main() 函数 |



### 2.3 GPU baseline 命名规则

原版通常用 `Cpu`/`Gpu` 后缀区分 CPU 和 GPU 版本：

- CPU 版本：`interestRateCompoundFactorCpu` → 保留原名，加 `static`
- GPU 版本：`interestRateCompoundFactor`（去掉后缀）或 `interestRateCompoundFactorGpu`
- `__device__` 函数保持原版 GPU 命名

---

## 3. 逐文件移植步骤

### 3.1 Step 1: 分析原始代码

```bash
# 1. 列出所有源文件
ls original_repo/CPU/

# 2. 找出所有函数名
grep "^[a-zA-Z].*(" original_repo/CPU/*.c | head -40

# 3. 找出所有结构体
grep "typedef struct" original_repo/CPU/*.h

# 4. 找出数据生成逻辑（main函数中的初始化代码）
grep -n "rand\|srand\|for.*num\|malloc" original_repo/CPU/*Engine.c

# 5. 找出常量定义
grep "#define" original_repo/CPU/*.h
```

### 3.2 Step 2: 创建 task.json

```json
{
  "task_id": "<task_name>",
  "name": "<人类可读名称>",
  "category": "<类别>",
  "difficulty": 1-4,
  "tags": ["tag1", "tag2"],

  "input_sizes": {
    "small":  {"N": 100000,   "seed": 42},
    "medium": {"N": 1000000,  "seed": 42},
    "large":  {"N": 10000000, "seed": 42}
  },

  "correctness": {
    "mode": "numerical",     // 或 "exact"
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

### 3.3 Step 3: 创建 cpu_reference.c

**模板结构：**

```c
// cpu_reference.c — <任务名> CPU baseline
//
// Faithfully ported from <原始repo路径>
// Preserves original function names, struct definitions, and computation flow.
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>  // 如果原版用了 bool

// ===== 原版常量 (从 <原版头文件>) =====
#define CONSTANT_NAME value

// ===== 原版结构体 (从 <原版头文件>) =====
typedef float dataType;  // 如果原版用了 typedef

typedef struct {
    // 保留原版字段名和顺序
} originalStructName;

// ===== 模块级静态状态 =====
static int g_N;
static const float* g_param1;
// ...

// ===== 从原版移植的函数（保留原名，加 static） =====

static float originalFunctionNameCpu(float arg1, originalStruct arg2)
{
    // 逐行复制原版逻辑
    // 仅修改：& → *, new → malloc, 移除 I/O
}

// ... 所有原版函数 ...

// ===== 公开接口（仅这三个函数不加 static） =====

void solution_init(int N, /* 扁平化的参数 */)
{
    g_N = N;
    // 存储指针到静态变量
}

void solution_compute(int N, float* output)
{
    // 从扁平数组构建原版结构体
    // 调用原版的主计算函数
    // 提取结果到 output
}

void solution_free(void)
{
    // 释放 solution_init 中分配的内存（如有）
}
```

**关键检查清单：**
- [ ] 所有原版函数名保留
- [ ] 所有原版结构体名保留
- [ ] 所有原版常量名和值保留
- [ ] `typedef` 匹配原版（如 `dataType`）
- [ ] 无 `#include` 引用原版头文件（所有内容自包含）
- [ ] 无文件 I/O、无 `printf`、无 `main()`
- [ ] 纯 C（无 C++ 特性）
- [ ] 函数重载已转为不同名称
- [ ] `rand()` 已替换为确定性 RNG（如 xorshift32）

### 3.4 Step 4: 创建 task_io_cpu.c

```c
// task_io_cpu.c — <task_id> CPU I/O adapter

#include "orbench_io.h"
#include <stdio.h>
#include <stdlib.h>

// 声明 cpu_reference.c 的三个接口函数
extern void solution_init(int N, ...);
extern void solution_compute(int N, float* output);
extern void solution_free(void);

typedef struct {
    int N;
    float* output;  // 输出缓冲区
} TaskContext;

void* task_setup(const TaskData* data, const char* data_dir) {
    TaskContext* ctx = (TaskContext*)calloc(1, sizeof(TaskContext));
    ctx->N = (int)get_param(data, "N");

    // 从 input.bin 提取所有 tensor
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
        fprintf(fp, "%.6f\n", ctx->output[i]);  // 或 "%.6e\n"
    fclose(fp);
}

void task_cleanup(void* test_data) {
    TaskContext* ctx = (TaskContext*)test_data;
    solution_free();
    free(ctx->output);
    free(ctx);
}
```

### 3.5 Step 5: 创建 task_io.cu

与 `task_io_cpu.c` 几乎完全相同，仅增加：
```c
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
// 所有 extern 声明和四个 task_* 函数
#ifdef __cplusplus
}
#endif
```

### 3.6 Step 6: 创建 gen_data.py

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
    """生成输入数据，匹配原版 repo 的数据生成逻辑。"""
    rng = np.random.RandomState(seed)
    # ... 生成数据 ...
    return tensors_dict

def compile_cpu_baseline(orbench_root: Path) -> Path:
    """编译 CPU reference 为可执行文件。"""
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
    """运行 CPU baseline 获取计时。"""
    r = subprocess.run([str(exe), str(data_dir)], capture_output=True, text=True)
    m = re_mod.search(r"TIME_MS:\s*([0-9.]+)", r.stdout)
    return float(m.group(1))

def run_cpu_expected_output(exe: Path, data_dir: Path) -> None:
    """运行 --validate 生成 expected_output.txt。"""
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

### 3.7 Step 7: 创建 gpu_baseline.cu

```cuda
// gpu_baseline.cu — <task_name> GPU baseline
//
// Faithfully ported from <原版CUDA路径>
// __device__ 函数保留原版 GPU 命名（去掉 Cpu 后缀或用 Gpu 后缀）

#include <cuda_runtime.h>

// ===== 原版结构体（同 cpu_reference.c） =====

// ===== __device__ 函数（保留原版 GPU 函数名） =====
__device__ float originalFunctionName(...)  // 注意：GPU 版本通常无后缀或用 Gpu 后缀
{
    // 逻辑同 CPU，但可用 fast math: __expf(), __logf(), erff(), rsqrtf()
}

// ===== 主 kernel =====
__launch_bounds__(256)
__global__ void mainKernel(int N, /* device pointers */, float* output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // 每个线程处理一个样本
    // 调用 __device__ 函数链
    output[i] = result;
}

// ===== 持久化 GPU 内存 =====
static float* d_input1 = nullptr;
static float* d_output = nullptr;

// ===== 公开接口（必须与 cpu_reference.c 签名完全一致） =====
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

### 3.8 Step 8: 创建 prompt_template.yaml

```yaml
task_description: |
  <问题描述，包括背景、输入输出、目标>

interface: |
  ```c
  extern "C" void solution_init(int N, ...);
  extern "C" void solution_compute(int N, float* output);
  ```

input_size_notes: |
  | Size | N | 说明 |
  |------|---|------|

output_contract: |
  - output[i] = ...
  - Tolerance: atol=..., rtol=...

algorithm_background: |
  <算法详细说明，含伪代码>

hints_l2: |
  <中等提示：并行策略、关键优化方向>

hints_l1: |
  <详细提示：kernel 结构、具体优化技巧、代码片段>
```

---

## 4. 验证检查清单

### 4.1 编译验证

```bash
# CPU baseline
gcc -O2 -I framework framework/harness_cpu.c tasks/<task>/task_io_cpu.c \
    tasks/<task>/cpu_reference.c -o tasks/<task>/solution_cpu -lm

# GPU baseline
nvcc -O2 -arch=sm_89 -I framework framework/harness_gpu.cu \
    tasks/<task>/task_io.cu tasks/<task>/gpu_baseline.cu \
    -o tasks/<task>/gpu_baseline_exe
```

### 4.2 数据生成

```bash
python3 tasks/<task>/gen_data.py small tasks/<task>/data/small --with-expected
```

### 4.3 数值验证（与原版对比）

```bash
# 编译运行原版
cd tasks/OriginalRepo/CPU && g++ -O2 -o orig *.c -lm && ./orig

# 对比关键数值
python3 -c "
orig_val = ...  # 原版输出
our_lines = open('tasks/<task>/data/small/expected_output.txt').readlines()
our_val = float(our_lines[0])
print(f'Original: {orig_val}, Ours: {our_val}, Diff: {abs(orig_val-our_val):.6f}')
"
```

### 4.4 GPU 正确性验证

```bash
./tasks/<task>/gpu_baseline_exe tasks/<task>/data/small --validate
# 对比 output.txt 和 expected_output.txt
python3 -c "
g = [float(l) for l in open('tasks/<task>/data/small/output.txt')]
e = [float(l) for l in open('tasks/<task>/data/small/expected_output.txt')]
max_err = max(abs(a-b) for a,b in zip(g,e))
print(f'Max abs error: {max_err:.6e}')
"
```

### 4.5 LLM 端到端测试

```bash
python3 run.py generate-batch --models gemini-3-flash --tasks <task> --levels 2 --samples 1 --yes
python3 run.py eval --run <run_name> --sizes small
```

---

## 5. 常见坑和解决方案

| 问题 | 原因 | 解决 |
|------|------|------|
| CPU 编译报 `error: expected ';'` | 用了 C++ 引用 `&` | 改为指针 `*` |
| CPU 编译报 `error: redefinition of 'func'` | 原版函数重载 | 重命名为 `func` + `funcN` |
| GPU vs CPU 结果差异大 | `exp()` vs `__expf()` 精度不同 | 用 `expf()` 代替 `__expf()`，或接受 tolerance |
| `rand()` 导致不可重复 | 原版用 `srand(time(NULL))` | 替换为 xorshift32 + 固定 seed |
| gen_data 输出和 CPU 不一致 | tensor 名字/类型不匹配 task_io | 检查 `get_tensor_*` 的 name 参数 |
| GPU baseline segfault | `cudaMalloc` 在 `solution_compute` 内（每次调用都分配） | 移到 `solution_init` |
| task_io.cu 编译警告 `unused function` | task_io_cpu.c 中有 CPU-only 辅助函数 | 在 task_io.cu 中删除或 `#ifndef __CUDACC__` 包裹 |

---

## 6. 移植案例索引

| 原始 Repo | ORBench Task | 关键难点 |
|-----------|-------------|---------|
| FinanceBench/Black-Scholes | `black_scholes` | erf 多项式 → GPU 用 `erff()` 内建 |
| FinanceBench/Monte-Carlo | `monte_carlo` | `rand()` → `xorshift32` 确定性 RNG |
| FinanceBench/Bonds | `bonds_pricing` | 函数重载 `closeCpu` → `closeCpu`/`closeNcpu` |
| FinanceBench/Repo | `repo_pricing` | 依赖 bonds 全部函数 + 自身 repo 逻辑 |
| GROMACS nbnxm | `nbnxm_forces` | 超集群数据结构、warp-shuffle 归约 |


