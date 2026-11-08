#!/usr/bin/env bash
# Tests tasks 0-19 on each run once it reaches task 300.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
i=0
for d in runs/*_s[0-9]/; do
  ( until grep -q "^299," "$d/log.csv" 2>/dev/null && [ -f "$d/latest.pt" ] && .venv/bin/python -c "import torch,sys; sys.exit(torch.load(\"$d/latest.pt\",weights_only=True)[\"next_task\"]!=300)"; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$((i % 2)) .venv/bin/python savings.py "$d" --old 0 20 > "$d/savings.log" 2>&1 ) &
  i=$((i + 1))
done
wait
