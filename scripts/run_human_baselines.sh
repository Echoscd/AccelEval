#!/bin/bash
# run_human_baselines.sh — time all task gpu_baseline_exe binaries on all 3 sizes,
# collect to a JSON for later "human ceiling" analysis.
set -u
cd "$(dirname "$0")/.."
OUT="compare/human_baseline_timings.json"
mkdir -p compare

TASKS=()
for d in tasks/*/gpu_baseline_exe; do
  t=$(basename "$(dirname "$d")")
  TASKS+=("$t")
done
echo "Tasks with gpu_baseline_exe: ${#TASKS[@]}"

echo "{" > "$OUT"
first_task=true
for t in "${TASKS[@]}"; do
  $first_task || echo "," >> "$OUT"
  first_task=false
  echo -n "  \"$t\": {" >> "$OUT"
  first_sz=true
  for sz in small medium large; do
    $first_sz || echo -n "," >> "$OUT"
    first_sz=false
    data_dir="tasks/$t/data/$sz"
    if [ ! -f "$data_dir/input.bin" ]; then
      echo -n " \"$sz\": null" >> "$OUT"
      continue
    fi
    echo "[$(date +%H:%M:%S)] $t / $sz" >&2
    out=$(timeout 300 ./tasks/$t/gpu_baseline_exe "$data_dir" --warmup 3 --trials 5 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then
      echo -n " \"$sz\": {\"status\":\"timeout_or_err\", \"rc\": $rc}" >> "$OUT"
      continue
    fi
    time_ms=$(echo "$out" | grep -E "^TIME_MS:" | awk '{print $2}')
    init_ms=$(echo "$out" | grep -E "^INIT_MS:" | awk '{print $2}')
    cpu_ms=$(cat "$data_dir/cpu_time_ms.txt" 2>/dev/null)
    echo -n " \"$sz\": {\"time_ms\": ${time_ms:-null}, \"init_ms\": ${init_ms:-null}, \"cpu_ms\": ${cpu_ms:-null}}" >> "$OUT"
  done
  echo -n "}" >> "$OUT"
done
echo >> "$OUT"
echo "}" >> "$OUT"
echo "Saved: $OUT"
