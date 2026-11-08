#!/bin/bash
# SWR strength sweep: 100 tasks, seeds 0-1.
export OMP_NUM_THREADS=1
i=0
for k in 1e-5 1e-4 1e-3 1e-2; do for s in 0 1; do
  gpu=$((i++ % 2))
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python train.py --seed $s --lr 0.001 --tasks 100 --swr-k $k --output-dir sweep > sweep_k${k}_s$s.log 2>&1 &
done; done
wait
