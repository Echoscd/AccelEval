# AccelEval (ORBench)

**A benchmark for evaluating LLMs on CPU-to-CUDA code acceleration**

AccelEval measures whether LLMs can translate sequential CPU programs into
efficient CUDA kernels. Unlike prior benchmarks that focus on isolated GPU
primitives (KernelBench) or pure syntax (ComputeEval), AccelEval stresses
**end-to-end acceleration** across 43 production-style workloads — graph
analytics, scientific computing, financial engineering, dynamic programming,
spatial indexing, and more.

- 📦 **Test data**: [Cosmoscd/AccelEval](https://huggingface.co/datasets/Cosmoscd/AccelEval) on HuggingFace
- 📄 **Paper**: `docs/paper.tex` (arXiv) · `nips/neurips_2026.tex` (NeurIPS)
- 🔬 **Tasks**: 43 tasks × 3 sizes (small/medium/large)
- 🧪 **Interface**: single `solution_compute` that is **timed end-to-end**

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download benchmark data (auto-fetches from HuggingFace)
python3 scripts/download_data.py small      # smoke (≈50 MB)
python3 scripts/download_data.py medium     # leaderboard (≈1.8 GB)
python3 scripts/download_data.py large      # stress (≈2.5 GB)

# 3. Configure API keys
cp .env.example .env    # edit with OPENROUTER_API_KEY / ANTHROPIC_API_KEY etc.

# 4. One-shot: generate + evaluate + analyze
python3 -m framework.run_all_tasks \
    --models gemini-3.1-pro-preview-openrouter \
    --levels 3 --samples 1 --sizes small --yes
```

Results land in `runs/<model>_l<level>_<timestamp>/` and a consolidated
JSON in `runs/reports/all_results_<timestamp>.json`.

## Current leaderboard (L3, 44 tasks, **medium** size)

| Rank | Model | Pass | Median Speedup | fast@10× |
|:-:|---|---:|---:|---:|
| 🥇 | **Gemini 3.1 Pro** | **94.9%** | **33×** | **64.1%** |
| 🥈 | GLM 5.1 | 91.7% | 10× | 47.2% |
| 🥉 | DeepSeek V3.2 | 89.7% | 7× | 27.6% |
| 4 | Claude Opus 4.6 | 87.2% | 18× | 46.2% |
| 5 | Kimi K2.5 | 84.8% | 14× | 45.5% |
| 6 | Qwen 3.6 Plus | 84.2% | 17× | 47.4% |
| 7 | GPT-5.4 | 73.5% | 19× | 44.1% |

Regenerate with `python3 scripts/export_xlsx.py` after new runs.

## Task categories

| Category | Tasks |
|----------|------|
| Dynamic programming | `bellman_ford`, `held_karp_tsp`, `inventory_replenishment_dp`, `pathfinder_grid_dp`, `network_rm_dp`, `hawkes_dynamic_pricing_hjb`, `robust_value_iteration_hypercube`, `self_exciting_pricing_dp`, `gittins_index` |
| Graph analytics | `gapbs_pagerank_pullgs`, `gapbs_triangle_orderedcount`, `gapbs_cc_afforest`, `rodinia_bfs_levels`, `max_flow_push_relabel` |
| Sparse linear algebra | `hpcg_spmv_27pt`, `hpcg_symgs_sweep`, `hpcg_mg_vcycle`, `npb_cg_sparse_solve`, `spmv_csr` |
| CFD / Stencil | `hotspot_2d`, `miniWeather`, `npb_lu_ssor_structured`, `npb_sp_adi_pentadiagonal` |
| Financial computing | `black_scholes`, `bonds_pricing`, `monte_carlo`, `repo_pricing`, `batched_lhpca_portfolio` |
| Fluid simulation | `sph_position`, `sph_cell_index`, `sph_forces` |
| Distance / clustering | `euclidean_distance_matrix`, `hausdorff_distance`, `dtw_distance`, `dbscan` |
| Optimization | `crew_pairing`, `pdlp`, `motzkin_straus_blp_eval` |
| Stochastic / automata | `thompson_sampling`, `regex_match` |
| Bioinformatics | `smith_waterman` |
| Transportation | `nash_flows_over_time` |
| Geometry | `collision_detection` |

## Unified `compute_only` interface

Every task exposes a **single** function:

```c
extern "C" void solution_compute(
    /* inputs */ int N, const float* xs, const float* ys, float eps, int minPts,
    /* output */ int* labels);
```

The harness passes full host-side inputs every call; the LLM's code must
do H2D copy + kernel launch + D2H copy and synchronize before returning.
`solution_compute` is called with warmup + timed trials — it must be
idempotent. The full wall time of each call is measured via CUDA Events,
so there is no way to hide work in an untimed init phase.

## Prompt levels

| Level | Includes | Purpose |
|-------|----------|------|
| L1 | Task + interface + CPU code + **full optimization guide** | Ceiling with scaffolding |
| L2 | Task + interface + CPU code + brief hints | Optimization selection |
| L3 | Task + interface + CPU code only | Autonomous capability |

Prompts assemble from `tasks/<id>/prompt_template.yaml` via
`framework/generate_prompt.py`.

## Directory layout

```
AccelEval/
├── run.py                  # Legacy CLI (single-model / single-task)
├── framework/
│   ├── run_all_tasks.py    # Recommended: generate → eval → analyze
│   ├── compile.py          # nvcc compilation (auto-injects weak solution_free)
│   ├── benchmark.py        # CUDA Event timing
│   ├── validate.py         # Output comparison
│   ├── generate_prompt.py  # L1/L2/L3 prompt assembly
│   ├── llm/                # Multi-provider clients (OpenAI/Anthropic/OpenRouter/...)
│   └── harness_{gpu,cpu}.{cu,c}   # Timing + validation skeleton
├── tasks/
│   └── <task>/
│       ├── task.json               # Metadata: category, difficulty, sizes, tolerance
│       ├── prompt_template.yaml    # Task description + interface + hints
│       ├── cpu_reference.c         # CPU baseline
│       ├── task_io.{cu,c}          # I/O adapter (framework-internal)
│       ├── gen_data.py             # Generate input.bin + expected_output.txt
│       └── data/{small,medium,large}/   # ← download via scripts/download_data.py
├── scripts/
│   ├── download_data.py            # Pull test data from HuggingFace
│   ├── upload_to_hf.py             # Maintainer: push data to HF
│   ├── migrate_to_compute_only.py  # One-off migration tool
│   └── export_xlsx.py              # Build leaderboard xlsx from eval_results
├── docs/                           # Paper drafts
├── nips/                           # NeurIPS 2026 submission template
└── runs/                           # Generated code + eval results (gitignored)
```

## Running specific workflows

```bash
# Eval existing generated .cu files (no API calls)
python3 run.py eval --run <run_dir> --sizes medium

# Cross-model comparison
python3 run.py compare --runs gemini-*_l3_* claude-*_l3_*

# Analyze one run
python3 run.py analyze --run <run_dir>

# Multi-turn agent loop (generate → eval → feedback → refine)
python3 run.py agent-multiturn --model gemini-3.1-pro-preview \
    --task network_rm_dp --level 2 --turns 10
```

## Adding a new task

1. Create `tasks/<task_id>/`
2. Write `task.json` — metadata including `"interface_mode": "compute_only"`
3. Write `prompt_template.yaml` — description + single `solution_compute` signature + L1/L2 hints
4. Write `cpu_reference.c` — pure computation, one `solution_compute(...)` function, no I/O
5. Write `task_io.cu` and `task_io_cpu.c` — read input.bin into ctx, call `solution_compute`
6. Write `gen_data.py` — generate input.bin + compute expected output via CPU baseline
7. Run `python3 tasks/<task_id>/gen_data.py small tasks/<task_id>/data/small --with-expected`

## Environment requirements

- Python 3.10+
- CUDA Toolkit 12.0+ (`nvcc`)
- NVIDIA GPU (sm_80 or newer recommended; default is `sm_89`)
- `nsys` (optional — for kernel-level profiling)
- `pip install huggingface_hub` for data download

## Contributing

Bug reports, new tasks, and new LLM integrations welcome. For large task
additions, please include `gen_data.py` that produces deterministic output
and keeps the medium size under 3 minutes of CPU baseline time.

## Citation

```bibtex
@misc{acceleval2026,
  title={AccelEval: A Benchmark for Evaluating LLMs on CPU-to-CUDA Code Acceleration},
  author={Chen Dong},
  year={2026},
  url={https://github.com/Cosmoscd/AccelEval}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
