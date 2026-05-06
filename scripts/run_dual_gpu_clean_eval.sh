#!/bin/bash
# Dual-GPU clean re-eval pipeline with 4-layer zombie protection.
#
# Layer 1: framework/benchmark.py:_run_exe uses Popen + start_new_session +
#          os.killpg(SIGKILL) on timeout (kills CUDA child + entire group).
# Layer 2: This script's trap kills both worker process groups + orphan
#          solution_gpu on EXIT/INT/TERM.
# Layer 3: Background sweeper kills PPID=1 solution_gpu every 30s.
# Layer 4: Each worker runs in its own session (setsid) so we can kill the
#          full tree by pgid even if the worker shell dies abruptly.

set -u
cd /root/chendong/AccelEval/AccelEval

TS=$(date +%Y%m%d_%H%M)
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
SIZES=(small medium large)

# Split 8 models 4-4 across two GPUs.
GPU0_PASS2=( "${PASS2_RUNS[@]:0:4}" )
GPU1_PASS2=( "${PASS2_RUNS[@]:4:4}" )
GPU0_S2C=( "${S2C_RUNS[@]:0:4}" )
GPU1_S2C=( "${S2C_RUNS[@]:4:4}" )

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

# ---------- Layer 2: trap cleanup ----------
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

# ---------- Worker function (Layer 4: own session) ----------
# $1 = gpu_index ; $2... = run names (one phase, one size at a time).
run_worker() {
  local gpu=$1
  local size=$2
  local phase=$3
  shift 3
  local runs=("$@")
  for run in "${runs[@]}"; do
    local safe="${run//\//_}"
    local log="logs/dual_${phase}_g${gpu}_${size}_${safe}_${TS}.log"
    echo "[$(date)] [GPU${gpu}] >>> ${phase} ${run} --sizes ${size}"
    CUDA_VISIBLE_DEVICES=$gpu \
      python3 run.py eval --run "$run" --sizes "$size" --gpus 1 --no-nsys \
      >"$log" 2>&1
    local rc=$?
    echo "[$(date)] [GPU${gpu}] <<< ${phase} ${run} --sizes ${size} (rc=${rc})"
  done
}

# ---------- Phase 1: pass@2 (small/medium/large) ----------
for sz in "${SIZES[@]}"; do
  echo "[$(date)] === PHASE 1 size=${sz} ==="
  setsid bash -c "$(declare -f run_worker); run_worker 0 $sz pass2 ${GPU0_PASS2[@]@Q}" &
  WORKER0_PID=$!
  setsid bash -c "$(declare -f run_worker); run_worker 1 $sz pass2 ${GPU1_PASS2[@]@Q}" &
  WORKER1_PID=$!
  wait "$WORKER0_PID"
  wait "$WORKER1_PID"
  WORKER0_PID=""; WORKER1_PID=""
  echo "[$(date)] === PHASE 1 size=${sz} BOTH WORKERS DONE ==="
done

# ---------- Phase 2: stage2-control (medium only) ----------
echo "[$(date)] === PHASE 2 stage2-control ==="
setsid bash -c "$(declare -f run_worker); run_worker 0 medium s2c ${GPU0_S2C[@]@Q}" &
WORKER0_PID=$!
setsid bash -c "$(declare -f run_worker); run_worker 1 medium s2c ${GPU1_S2C[@]@Q}" &
WORKER1_PID=$!
wait "$WORKER0_PID"
wait "$WORKER1_PID"
WORKER0_PID=""; WORKER1_PID=""
echo "[$(date)] === PHASE 2 BOTH WORKERS DONE ==="

echo "[$(date)] === ALL DONE ==="
