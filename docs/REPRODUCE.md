# Reproducing the AccelEval leaderboard

This guide walks through running AccelEval end-to-end on a fresh machine and
producing the consolidated leaderboard xlsx.

## 0. Prerequisites

| | Minimum | Recommended |
|---|---|---|
| OS | Linux (kernel ≥ 5.4) | Ubuntu 22.04 / RHEL 9 |
| GPU | NVIDIA with CUDA 12.0 | L40S / H100 / A100 (`sm_80+`) |
| CUDA toolkit | 12.0 (`nvcc --version`) | 12.4 |
| GCC | 9 | 11 |
| Python | 3.9 | 3.11 |
| Disk | 15 GB free | 30 GB |
| RAM | 16 GB | 32 GB |
| Network | required for HF download + LLM API calls | |

Verify the GPU is healthy:
```bash
nvidia-smi
nvcc --version
```

## 1. Clone & install

```bash
git clone https://github.com/Echoscd/ORbench.git
cd ORbench
pip install -r requirements.txt huggingface_hub openpyxl
```

## 2. Download benchmark data

The 43 tasks' input data + CPU-reference expected outputs + CPU baselines
live on HuggingFace as
[`Cosmoscd/AccelEval`](https://huggingface.co/datasets/Cosmoscd/AccelEval).
Pull the size you want into `tasks/<task>/data/<size>/`:

```bash
python3 scripts/download_data.py small      # ~30 MB,   smoke test
python3 scripts/download_data.py medium     # ~1.2 GB,  the leaderboard size
python3 scripts/download_data.py large      # ~3.4 GB,  scale-stress
# or all three:
python3 scripts/download_data.py all
```

After this, every active task has `tasks/<task>/data/<size>/{input.bin,
expected_output.txt, cpu_time_ms.txt, requests.txt}` populated.

Optionally, download the model-generated solutions and current
leaderboard/strategy-transfer xlsx for inspection:
```bash
huggingface-cli download Cosmoscd/AccelEval solutions.tar.gz \
    leaderboard.xlsx stage1_vs_stage2.xlsx --repo-type dataset \
    --local-dir reference_artifacts/
```

## 3. Configure LLM API keys

Set whichever providers you intend to query (env vars or a `.env` file in
the repo root):

```bash
export OPENROUTER_API_KEY=sk-or-...   # access: Gemini, Claude via OR, GLM, Kimi, Qwen, GPT-5.4, DeepSeek V3.2/V4 Pro
export ANTHROPIC_API_KEY=sk-ant-...   # native Anthropic (optional)
export OPENAI_API_KEY=sk-...          # native OpenAI (optional)
export GOOGLE_API_KEY=...             # native Gemini (optional)
```

We recommend OpenRouter for everything except whichever provider you have
direct access to. Model IDs are listed in [`models.yaml`](../models.yaml).

## 4. Smoke test (single model, 1 task, small size)

Verify the toolchain end-to-end before launching full runs:

```bash
python3 -m framework.run_all_tasks \
    --models gemini-3.1-pro-preview-openrouter \
    --tasks black_scholes \
    --levels 3 --samples 1 --sizes small \
    --workers 2 --yes
```

Expected output:
```
[gen]  black_scholes/sample_0  OK ($0.003)
[eval] black_scholes/small  compiled | correct | speedup=8.7x
```

A run dir like `runs/gemini-3.1-pro-preview-openrouter_l3_<timestamp>/` is
created with the generated `.cu`, plus an `eval_results_<timestamp>.json`.

## 5. Full leaderboard generation

This is the headline experiment — 8 models × 42 tasks × L3 prompt × 1 sample,
generated and then evaluated at the medium scale.

```bash
# (a) Generate. Skip eval here so we can run it sequentially under the GPU.
python3 -m framework.run_all_tasks \
    --models gemini-3.1-pro-preview-openrouter \
             claude-opus-4.6-openrouter \
             openai/gpt-5.4 \
             qwen/qwen3.6-plus \
             kimi-k2.5-openrouter \
             glm-5.1-openrouter \
             deepseek-v3.2-openrouter \
             deepseek-v4-pro-openrouter \
    --levels 3 --samples 1 --sizes medium \
    --workers 10 --skip-eval --yes
```
Generation runs in parallel; expect 30–60 min wall time and roughly $5–10
in API spend (cost depends on model pricing in `models.yaml`).

```bash
# (b) Evaluate sequentially per run dir, at all three sizes.
for rd in $(ls -d runs/*_l3_* | grep -v _archive); do
    python3 run.py eval --run "${rd#runs/}" --sizes small --timeout 600
    python3 run.py eval --run "${rd#runs/}" --sizes medium --timeout 600
    python3 run.py eval --run "${rd#runs/}" --sizes large --timeout 600
done
```
Each model × size pass takes 5–15 min on an L40S.

## 6. Consolidate and export the leaderboard xlsx

This is the step that produces the goal artifact — a single xlsx that
aggregates all 8 models × 3 sizes into a per-task matrix plus a Summary
sheet with PAR-GM as the headline metric.

```bash
# (a) Wire up the run dirs you just produced.
$EDITOR scripts/consolidate_eval_data.py
# Edit the RUNS_L3 dict at the top of the file:
#   RUNS_L3 = {
#       "Gemini 3.1 Pro":  "<your gemini run dir name>",
#       "Claude Opus 4.6": "<your claude run dir>",
#       ...
#   }
```

```bash
# (b) Build the canonical JSON (latest record per task,size,model).
python3 scripts/consolidate_eval_data.py
# → runs/reports/acceleval_consolidated_<timestamp>.json (~600 KB)
```

```bash
# (c) Export the multi-sheet leaderboard xlsx.
python3 scripts/export_xlsx_from_consolidated.py
# → runs/reports/acceleval_task_level_<timestamp>.xlsx
```

Open the xlsx — it should contain 5 sheets:

| Sheet | Contents |
|---|---|
| `README` | data source + metric definitions + colour legend |
| `Small`, `Medium`, `Large` | task × model speedup matrix per size, with the bottom 4 rows showing `# pass / 42`, `GM× (pass)`, `PAR-GM (pass)`, `fast@10×` for each model |
| `Summary` | 8 models × 3 sizes × 4 metrics in a single grid (the headline view) |

The `Summary` sheet is the cross-size leaderboard and is the artifact most
papers / blog posts will quote.

## 7. (Optional) Strategy-transfer comparison

If you also ran Stage 2 (with-guidance generation), regenerate the side-by-side
xlsx:
```bash
$EDITOR scripts/export_s1_vs_s2_xlsx.py   # update S1 and S2 dicts to your run dirs
python3 scripts/export_s1_vs_s2_xlsx.py
# → runs/reports/acceleval_s1_vs_s2_gm_<timestamp>.xlsx
```
This produces a 3-sheet xlsx (`README`, `Summary`, `Per-task`) with PAR-GM
delta and per-task speedup ratios for the guidance experiment.

## Common pitfalls

- **`nvcc: command not found`** → CUDA toolkit not on PATH. Add
  `export PATH=/usr/local/cuda/bin:$PATH`.
- **`undefined reference to cublasCreate_v2`** during evaluation → the model
  reached for cuBLAS/cuSOLVER. The default build command does not link
  these (this is intentional and documented in the bad-case analysis,
  Appendix C of the paper). Either ignore the failure or extend
  `framework/compile.py`'s `cmd` to include `-lcublas -lcusolver`.
- **`CUDA_ERROR_OUT_OF_MEMORY`** on the large size → some tasks need >24 GB.
  Run those at medium only, or use `--sizes medium`.
- **Stale eval records after data regeneration** → if you re-run
  `gen_data.py` for any task, also re-run the eval for that task
  (`run.py eval --tasks <id> ...`). The consolidator picks the latest
  record by the timestamp embedded in the eval-results filename.
- **Cost runaway** on full runs → start with `--sizes small` to validate
  the pipeline, then scale up.

## Where the numbers live

After step 6 you will have:
```
runs/reports/
├── acceleval_consolidated_<ts>.json   # canonical, machine-readable
├── acceleval_task_level_<ts>.xlsx     # 5-sheet leaderboard (the goal)
└── acceleval_s1_vs_s2_gm_<ts>.xlsx    # only if step 7 was run
```
The `consolidated_<ts>.json` is the source of truth — both xlsx exports
read from it.
