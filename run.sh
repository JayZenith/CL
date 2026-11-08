#!/usr/bin/env bash
# Runs each arm for seeds 0..n-1, split evenly across GPUs. Arms are ';'-separated train.py args:
#   ./run.sh 6 "--swr-k 0;--swr-k 1e-2" --lr 0.001 --tasks 300
#   SCRIPT=cil.py ./run.sh 3 "--swr-k 0;--swr-k 1e-2 --replay-tasks 20"
#   SCRIPT=cil_multi.py ./run.sh 6 "..."   # One process per arm, all seeds stacked.
# Rerun to resume.
set -euo pipefail
cd "$(dirname "$0")"

n=$1
IFS=';' read -ra arms <<< "$2"
shift 2

./setup.sh

# Clear leftover workers so a restart can't OOM.
pkill -f '^.venv/bin/python ' || true
for _ in {1..30}; do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  sleep 1
done

script=${SCRIPT:-train.py}  # SCRIPT=cil.py for class-incremental, which writes to runs_cil/.
out=$([ "$script" = train.py ] && echo runs || echo runs_cil)
gpus=$(nvidia-smi -L | wc -l)
mkdir -p "$out"
export OMP_NUM_THREADS=1  # Each run is CPU-bound on one core.
i=0
if [ "$script" = cil_multi.py ]; then
  for a in "${!arms[@]}"; do
    gpu=$((i++ % gpus))
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python "$script" --seeds $(seq 0 $((n - 1))) ${arms[a]} "$@" \
      > "$out/$(echo "${arms[a]}" | tr -d ' -')_multi.log" 2>&1 &
  done
  n=0
fi
for ((s = 0; s < n; s++)); do
  for a in "${!arms[@]}"; do
    gpu=$((i++ % gpus))  # Outside the backgrounded command, so i updates here.
    # shellcheck disable=SC2086  # Arm args are meant to split into words.
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python "$script" --seed "$s" ${arms[a]} "$@" \
      > "$out/$(echo "${arms[a]}" | tr -d ' -')_s$s.log" 2>&1 &
  done
done
wait
.venv/bin/python "$([ "$script" = train.py ] && echo analyze.py || echo analyze_cil.py)" "$out"/*/ \
  | tee "$out/results.csv"
