"""Mean test accuracy in bins of 50 tasks, as in the paper: python analyze.py runs/*/"""
import csv
import sys
from collections import defaultdict
from statistics import mean

BIN = 50

curves = defaultdict(lambda: defaultdict(list))
for run_dir in sys.argv[1:]:
    arm = run_dir.rstrip("/").rsplit("/", 1)[-1].rsplit("_s", 1)[0]
    with open(f"{run_dir}/log.csv") as f:
        runs = defaultdict(list)
        for row in csv.DictReader(f):
            runs[int(row["task"]) // BIN].append(float(row["test_acc"]))
    for b, values in runs.items():
        if len(values) == BIN:
            curves[arm][b].append(mean(values))

print("arm,first_task,runs,test_acc")
for arm, bins in sorted(curves.items()):
    for b, values in sorted(bins.items()):
        print(f"{arm},{b * BIN},{len(values)},{100 * mean(values):.2f}")
