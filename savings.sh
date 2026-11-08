#!/bin/bash
# Runs savings.py on every finished run without a savings.csv, split across GPUs.
export OMP_NUM_THREADS=1
i=0
for d in runs/*/; do
  [ -f "$d/savings.csv" ] && continue
  gpu=$((i++ % 2))
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python savings.py "$d" > "$d/savings.log" 2>&1 &
done
wait
