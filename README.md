# AccelEval

**A benchmark for evaluating LLMs on CPU-to-CUDA code acceleration.**

AccelEval measures whether LLMs can translate sequential CPU programs into
efficient CUDA kernels. Unlike prior benchmarks that focus on isolated GPU
primitives or pure syntax, AccelEval stresses **end-to-end acceleration**
across **42 production-style workloads** drawn from established HPC suites
(HPCG, NAS Parallel Benchmarks, Rodinia, GAPBS, miniWeather), domain-specific
scientific codes (DualSPHysics, Box2D), and industrial financial /
operations-research workloads (FinanceBench, OR-Tools, recent OR papers).

- 📦 **Test data**:
  [`Accel-Eval/AccelEval-data`](https://huggingface.co/datasets/Accel-Eval/AccelEval-data)
  on Hugging Face — input binaries, expected outputs, CPU baseline times,
  plus a `tasks.parquet` manifest with the full `cpu_reference.c` and
  `prompt_template.yaml` for every task.
- 🔬 **Tasks**: 42 tasks × 3 input scales (small / medium / large)
- 🧪 **Interface**: every task is a single `solution_compute(...)` function,
  timed end-to-end (allocation, H↔D copy, kernel, library, cleanup) so there
  is no place to hide work in an untimed setup phase.
- 🧠 **Decomposition pipeline**: every passing solution is labelled against
  a 43-pattern CUDA optimization catalog and fed back as natural-language
  guidance for cross-model strategy transfer.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Pull the benchmark data from Hugging Face
python3 scripts/download_data.py small      # smoke (~50 MB)
python3 scripts/download_data.py medium     # leaderboard (~1.8 GB)
python3 scripts/download_data.py large      # stress (~2.5 GB)

# 3. Configure API keys (.env): OPENROUTER_API_KEY, ANTHROPIC_API_KEY, ...
cp .env.example .env

# 4. Generate + evaluate + analyze for one model
python3 scripts/run_all_tasks.py \
    --models gemini-3.1-pro-preview-openrouter \
    --levels 3 --samples 1 --sizes small --yes
```

Solutions and per-cell results are written to
`runs/<model>_l<level>_<timestamp>/`; consolidated leaderboards land in
`runs/reports/`.

## Task categories

42 tasks across six top-level categories:

| Category | Count | Examples |
|---|---:|---|
| HPC reference kernels    | 7  | `hpcg_spmv_27pt`, `hpcg_symgs_sweep`, `hpcg_mg_vcycle`, `npb_cg_sparse_solve`, `npb_lu_ssor_structured`, `npb_sp_adi_pentadiagonal`, `miniWeather` |
| Scientific simulation    | 4  | `sph_cell_index`, `sph_forces`, `sph_position`, `hotspot_2d` |
| Graph algorithms         | 7  | `bellman_ford`, `held_karp_tsp`, `max_flow_push_relabel`, `rodinia_bfs_levels`, `gapbs_cc_afforest`, `gapbs_pagerank_pullgs`, `gapbs_triangle_orderedcount` |
| Spatial–temporal         | 7  | `dbscan`, `dtw_distance`, `euclidean_distance_matrix`, `hausdorff_distance`, `collision_detection`, `smith_waterman`, `regex_match` |
| Financial computing      | 5  | `black_scholes`, `bonds_pricing`, `monte_carlo`, `repo_pricing`, `batched_lhpca_portfolio` |
| Operations research      | 12 | `crew_pairing`, `gittins_index`, `hawkes_dynamic_pricing_hjb`, `inventory_replenishment_dp`, `motzkin_straus_blp_eval`, `nash_flows_over_time`, `network_rm_dp`, `pathfinder_grid_dp`, `pdlp`, `robust_value_iteration_hypercube`, `self_exciting_pricing_dp`, `thompson_sampling` |

The full per-task manifest (source repo, brief description) is in
`tasks/<task_id>/task.json`.

## Unified `solution_compute` interface

Every task exposes a single function:

```c
extern "C" void solution_compute(
    /* inputs */  int N, const float* xs, const float* ys, float eps, int minPts,
    /* output */  int* labels);
```

The harness passes full host-side inputs every call; the LLM-generated CUDA
must do H2D copy + kernel launch + D2H copy and synchronise before
returning. `solution_compute` is called with three warmups and five timed
trials; the **full wall time of every call is measured via CUDA Events**, so
allocation cost cannot be hidden in an untimed `init()` phase.

An automated audit (`solution_compute` is called repeatedly with cleared
device state) detects timing-loophole exploits such as static device
pointers that survive across calls.

## Prompt levels

| Level | Includes | Purpose |
|-------|----------|---------|
| L1 | Task + interface + CPU code + **full optimization guide** | Ceiling with scaffolding |
| L2 | Task + interface + CPU code + brief hints                | Optimization selection |
| L3 | Task + interface + CPU code only                          | Autonomous capability (default) |

Prompts assemble from `tasks/<id>/prompt_template.yaml` via
`framework/generate_prompt.py`.

## Directory layout

```
AccelEval/
├── run.py                  # CLI entry-point (single model / single task)
├── framework/
│   ├── benchmark.py        # CUDA-Event end-to-end timing
│   ├── compile.py          # nvcc compile (auto-injects weak solution_free)
│   ├── validate.py         # Output comparison (per-task tolerance)
│   ├── generate.py         # LLM dispatcher
│   ├── generate_prompt.py  # L1 / L2 / L3 prompt assembly
│   ├── run_all_tasks.py    # Generate → eval → analyze pipeline
│   ├── llm/                # Provider clients (OpenAI, Anthropic, Google, OpenRouter)
│   ├── knowledge/          # Pattern decomposition + LLM analyzer
│   └── harness_{gpu,cpu}.{cu,c}   # Timing + validation skeleton
├── tasks/<task_id>/
│   ├── task.json               # Metadata: category, difficulty, sizes, tolerance
│   ├── prompt_template.yaml    # Task description + interface + hints
│   ├── cpu_reference.c         # CPU baseline
│   ├── task_io.{cu,c}          # I/O adapter
│   ├── gen_data.py             # Generate input.bin + expected_output.txt
│   └── data/{small,medium,large}/   # ← `python3 scripts/download_data.py`
├── scripts/
│   ├── download_data.py            # Pull benchmark data from Hugging Face
│   ├── upload_to_hf.py             # Maintainer: push data to HF
│   ├── gen_all_data.sh / gen_data.sh
│   ├── run_all_tasks.py / run_dual_gpu_clean_eval.sh / clean_eval_*.sh
│   ├── consolidate_eval_data.py    # Build per-(model, task, size) leaderboard JSON
│   ├── export_xlsx.py / export_xlsx_from_consolidated.py
│   ├── analyze_pattern_impact.py   # Per-pattern within-task LIFT
│   ├── analyze_s2_control.py       # Strategy-transfer ablation (treatment vs control)
│   ├── compute_passk.py            # pass@k aggregates from a single k-sample run
│   ├── plot_pattern_cooccurrence.py / plot_scale_*.py
│   └── run_human_baselines.sh
├── compare/                # Snapshots of related-work benchmark code (KernelBench, ParEval, ...)
├── docs/                   # REPRODUCE.md, task_porting_guide.md
└── runs/                   # Generated solutions + eval results (gitignored)
```

## Common workflows

```bash
# Eval already-generated .cu files against a fresh data download (no API calls)
python3 run.py eval --run runs/<model>_<config>_<date> --sizes medium

# Cross-model summary
python3 scripts/consolidate_eval_data.py
python3 scripts/export_xlsx_from_consolidated.py

# Best-of-k pass@k leaderboard (after generating k samples per task in ONE run)
python3 scripts/compute_passk.py --runs runs/<model>_<config>_<date> --k 3

# Decomposition pipeline: pattern attribution + LIFT analysis
python3 scripts/analyze_pattern_impact.py
python3 scripts/plot_pattern_cooccurrence.py

# Strategy-transfer Stage-2 (treatment + length-matched control)
python3 scripts/analyze_s2_control.py
```

The end-to-end re-run that produced the public leaderboard is in
`docs/REPRODUCE.md`.

## Adding a new task

1. Create `tasks/<task_id>/`
2. Write `task.json` — metadata including `"interface_mode": "compute_only"`
3. Write `prompt_template.yaml` — description + single `solution_compute` signature + L1 / L2 hints
4. Write `cpu_reference.c` — pure computation, one `solution_compute(...)` function, no I/O
5. Write `task_io.cu` and `task_io_cpu.c` — read `input.bin` into `ctx`, call `solution_compute`
6. Write `gen_data.py` — produce `input.bin` and the expected output via the CPU baseline
7. Run `python3 tasks/<task_id>/gen_data.py small tasks/<task_id>/data/small --with-expected`

The full porting workflow is documented in `docs/task_porting_guide.md`.

## Environment requirements

- Python 3.10+
- CUDA Toolkit 12.0+ (`nvcc` on `PATH`)
- NVIDIA GPU; targets `sm_80` and newer (default `sm_89` — H200 / RTX 4090)
- `nsys` (optional, for kernel-level profiling)
- `pip install huggingface_hub` for data download

## Contributing

Bug reports, new tasks, and new LLM-provider integrations are welcome.
For large task additions, please include a `gen_data.py` that produces
deterministic output and keeps the *medium* size under three minutes of
single-thread CPU baseline time.

## License

Apache 2.0 — see [LICENSE](LICENSE).
