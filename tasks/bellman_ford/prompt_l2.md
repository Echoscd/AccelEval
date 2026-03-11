# ORBench v2 任务：Bellman-Ford SSSP（Request-Based GPU Service）

你要实现的是 **GPU 版本的“服务式”最短路**：图只加载一次并常驻显存，然后处理多次查询（requests）。

本任务不再使用 `LLM_input.cu` 模板，也不需要写 `main()`。你只需要提供 3 个函数（见下）。

---

## 输入数据布局（每个 size 一个目录）

`tasks/bellman_ford/data/<size>/`

- `input.bin`: 大输入（CSR 图结构），只在 `solution_setup()` 读入并 H2D（不计时）
- `requests.txt`: N 行，每行一个查询（格式：`s t`，两个整数，空格分隔）
- `expected_output.txt`: N 行，每行一个浮点数（s 到 t 的最短距离）
- `cpu_time_ms.txt`: CPU baseline 处理全部 requests 的时间（毫秒）

---

## 你需要实现的接口（只写这 3 个函数）

在你的 `solution.cu` 中实现：

```c
extern void solution_setup(const TaskData* data);

extern void solution_run(int num_requests, const char** requests,
                         char** responses);

extern void solution_cleanup();
```

其中 `TaskData` / `Tensor` 与辅助函数定义在 `framework/orbench_io.h`（C 头文件）：

- `get_param(data, "V")`, `get_param(data, "E")`
- `get_tensor_int(data, "row_offsets")`, `get_tensor_int(data, "col_indices")`
- `get_tensor_float(data, "weights")`

**关键约束：**
- `solution_setup/cleanup` 不计时；计时区间只包含 `solution_run`。
- `distances[i]` 是单个浮点数，表示从 `s` 到 `t` 的最短距离。
- `solution_run` 会被 warmup 与 timed trials 多次调用，必须避免重复分配/重复 H2D。

---

## request / response 格式

### requests.txt（输入）
每行一个 (s, t) 对，格式：`s t`（两个整数，空格分隔）

```
0 42
10 99
20 5
```

**说明**：s 分 10 组，每组 10 个不同的 s，每个 s 配 10 个不同的 t，共 100 条请求。

### distances[i]（输出）
**单个距离值**：s 到 t 的最短距离。

- `distances[i]` 是一个 `float` 值
- 表示从 `s` 到 `t` 的最短距离
- 如果不可达，返回 `INF_VAL`（约 `1e30f`）
- harness 会自动格式化为文本：`output.txt` 每行一个浮点数

**重要**：不要计算整个距离数组、top-k、checksum 或手动格式化文本，这些会引入额外开销。只计算并返回 s 到 t 的距离值即可。

校验模式由 `task.json` 定义：
- `correctness.mode = "checksum"`
- `correctness.field = "cs"`
- `correctness.atol = 0.1`

---

## 编译方式（供你本地自测）

GPU solution：

```bash
nvcc -O2 -arch=sm_89 -I framework/ framework/harness_gpu.cu solution.cu -o solution_gpu
./solution_gpu tasks/bellman_ford/data/large --validate
```

---

## 性能建议（L2，不给实现提示）

目标是把 `solution_run()` 做到尽量薄：复用显存、批处理 requests、避免频繁 cudaMalloc/cudaFree、减少同步与 host-device 往返。
