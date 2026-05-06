#!/bin/bash
# clean_eval_worker.sh GPU_INDEX
#
# Self-contained worker that processes its hardcoded queue on one physical
# GPU. Reads its queue from environment variables QUEUE_PASS2 and QUEUE_S2C
# (newline-separated). The model partition is decided by the master script.

set -u
cd /root/chendong/ORbench/ORBench

GPU=$1
TS=${TS:-$(date +%Y%m%d_%H%M)}

# Read queues from env (newline-separated)
mapfile -t PASS2 <<<"$QUEUE_PASS2"
mapfile -t S2C   <<<"$QUEUE_S2C"

run_one() {
  local phase=$1
  local size=$2
  local run=$3
  local safe="${run//\//_}"
  local log="logs/dual_${phase}_g${GPU}_${size}_${safe}_${TS}.log"
  echo "[$(date)] [GPU${GPU}] >>> ${phase} ${run} --sizes ${size}"
  CUDA_VISIBLE_DEVICES=$GPU \
    python3 run.py eval --run "$run" --sizes "$size" --gpus 1 --no-nsys \
    >"$log" 2>&1
  local rc=$?
  echo "[$(date)] [GPU${GPU}] <<< ${phase} ${run} --sizes ${size} (rc=${rc})"
}

echo "[$(date)] [GPU${GPU}] worker start, pid=$$, pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
echo "[$(date)] [GPU${GPU}] PASS2 queue (${#PASS2[@]}): ${PASS2[*]}"
echo "[$(date)] [GPU${GPU}] S2C queue (${#S2C[@]}): ${S2C[*]}"

# Phase 1: pass@2 — small/medium/large per model. Iterate sizes outer so that
# both workers progress through small first (still balanced even without barrier).
for size in small medium large; do
  for run in "${PASS2[@]}"; do
    [[ -z "$run" ]] && continue
    run_one pass2 "$size" "$run"
  done
done

# Phase 2: stage2-control (medium only)
for run in "${S2C[@]}"; do
  [[ -z "$run" ]] && continue
  run_one s2c medium "$run"
done

echo "[$(date)] [GPU${GPU}] worker DONE"
