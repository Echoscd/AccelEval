---
license: apache-2.0
tags:
- benchmark
- cuda
- gpu
- code-generation
- orbench
size_categories:
- 1B<n<10B
---

# ORBench Test Data

Test data for **ORBench** — a benchmark for evaluating LLMs on CPU-to-CUDA
code acceleration.

This repo hosts only the **input data + expected outputs**. The benchmark
harness (CPU references, task_io adapters, evaluation scripts) lives at
[github.com/YOURNAME/ORBench](https://github.com/YOURNAME/ORBench).

## Contents

| File | Size | Description |
|------|------|------|
| `small.tar.gz`  | ~100 MB | Small inputs — for smoke tests (43 tasks) |
| `medium.tar.gz` | ~3.5 GB | Medium inputs — main leaderboard |
| `large.tar.gz`  | ~8 GB   | Large inputs — stress test |
| `manifest.json` | —     | SHA256 + task metadata |

Each tarball extracts to `<task>/data/<size>/` containing:

- `input.bin` — binary input tensors (ORBench v2 format)
- `expected_output.txt` — reference output from CPU baseline
- `cpu_time_ms.txt` — CPU baseline wall time
- `requests.txt` — per-call queries (if applicable)

## Usage

```bash
# Clone the harness repo
git clone https://github.com/YOURNAME/ORBench.git
cd ORBench

# Install HF client
pip install huggingface_hub

# Download data
python3 scripts/download_data.py small       # smoke
python3 scripts/download_data.py medium      # main leaderboard
python3 scripts/download_data.py large       # stress test
python3 scripts/download_data.py all         # everything

# Run the benchmark
python3 -m framework.run_all_tasks \
    --models gemini-3.1-pro-preview-openrouter \
    --levels 3 --sizes medium --yes
```

## Verification

```bash
python3 scripts/download_data.py medium --verify
```

## Citation

If you use ORBench, please cite:

```bibtex
@misc{orbench2026,
  title={ORBench: Evaluating LLMs on CPU-to-CUDA Code Acceleration},
  author={...},
  year={2026},
  url={https://github.com/YOURNAME/ORBench}
}
```
