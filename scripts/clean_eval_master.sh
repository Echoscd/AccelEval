#!/bin/bash
# clean_eval_master.sh
#
# Launches sweeper + 2 GPU workers (each via setsid + own session for clean
# kill semantics). Workers are scripts/clean_eval_worker.sh — no nested bash -c
# tricks (those were buggy: only the first iteration of an inner for-loop ran).

set -u
cd /root/chendong/ORbench/ORBench

TS=$(date +%Y%m%d_%H%M)
export TS
MASTER_LOG=logs/dual_eval_master_${TS}.log
SWEEPER_LOG=logs/dual_eval_sweeper_${TS}.log
mkdir -p logs
exec >>"$MASTER_LOG" 2>&1

echo "[$(date)] === DUAL-GPU CLEAN EVAL PIPELINE START ==="
echo "[$(date)] master pid=$$"

# ---------- Run lists ----------
PASS2_RUNS=(
  "claude-opus-4.6-openrouter_l3_20260503_2312"
  "deepseek-v3.2-openrouter_l3_20260503_2312"
  "deepseek-v4-pro-openrouter_l3_20260503_2312"
  "gemini-3.1-pro-preview-openrouter_l3_20260503_2312"
  "glm-5.1-openrouter_l3_20260503_2312"
  "kimi-k2.5-openrouter_l3_20260503_2312"
  "openai/gpt-5.4_l3_20260503_2312"
  "qwen/qwen3.6-plus_l3_20260503_2312"
)
S2C_RUNS=(
  "claude-opus-4.6-openrouter_l3s2c_20260504_1334"
  "deepseek-v3.2-openrouter_l3s2c_20260504_1334"
  "deepseek-v4-pro-openrouter_l3s2c_20260504_1334"
  "gemini-3.1-pro-preview-openrouter_l3s2c_20260504_1334"
  "glm-5.1-openrouter_l3s2c_20260504_1334"
  "kimi-k2.5-openrouter_l3s2c_20260504_1334"
  "openai/gpt-5.4_l3s2c_20260504_1334"
  "qwen/qwen3.6-plus_l3s2c_20260504_1334"
)

# Helper: join array elements with newlines (for the worker's mapfile parsing).
join_nl() { printf '%s\n' "$@"; }

# Split 8 models 4-4 across two GPUs (model order kept consistent for both phases).
GPU0_PASS2_NL=$(join_nl "${PASS2_RUNS[@]:0:4}")
GPU1_PASS2_NL=$(join_nl "${PASS2_RUNS[@]:4:4}")
GPU0_S2C_NL=$(join_nl "${S2C_RUNS[@]:0:4}")
GPU1_S2C_NL=$(join_nl "${S2C_RUNS[@]:4:4}")

# ---------- Layer 3: background sweeper ----------
sweep_orphans() {
  while true; do
    sleep 30
    for p in $(ps -eo pid,ppid,cmd 2>/dev/null | awk '$2==1 && /solution_gpu/ {print $1}'); do
      echo "[$(date)] sweeper: SIGKILL orphan solution_gpu PID=$p" >>"$SWEEPER_LOG"
      kill -9 "$p" 2>/dev/null
    done
  done
}
sweep_orphans &
SWEEPER_PID=$!
echo "[$(date)] sweeper PID=$SWEEPER_PID"

# ---------- Layer 2: trap cleanup (idempotent) ----------
WORKER0_PID=""
WORKER1_PID=""
cleanup() {
  echo "[$(date)] cleanup: shutting down"
  for pid in "$SWEEPER_PID" "$WORKER0_PID" "$WORKER1_PID"; do
    [[ -n "$pid" ]] && kill -- -"$pid" 2>/dev/null
    [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null
  done
  pkill -9 -f "python3 run.py eval" 2>/dev/null || true
  pkill -9 -f "solution_gpu" 2>/dev/null || true
  echo "[$(date)] cleanup: done"
}
trap cleanup EXIT INT TERM

# ---------- Launch workers (each in own session) ----------
QUEUE_PASS2="$GPU0_PASS2_NL" QUEUE_S2C="$GPU0_S2C_NL" \
  setsid bash scripts/clean_eval_worker.sh 0 &
WORKER0_PID=$!

QUEUE_PASS2="$GPU1_PASS2_NL" QUEUE_S2C="$GPU1_S2C_NL" \
  setsid bash scripts/clean_eval_worker.sh 1 &
WORKER1_PID=$!

echo "[$(date)] worker0 PID=$WORKER0_PID  worker1 PID=$WORKER1_PID"

# Wait for both workers to drain their queues independently (no per-size barrier).
wait "$WORKER0_PID"
echo "[$(date)] worker0 DONE"
wait "$WORKER1_PID"
echo "[$(date)] worker1 DONE"

echo "[$(date)] === ALL DONE ==="
