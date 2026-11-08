"""Class-incremental results per arm: python analyze_cil.py runs_cil/*/

Per 50-task bin, test accuracy of each new task picking between its own 2 classes (plasticity) and among all
seen classes. Then, at each eval, accuracy on the old tasks and on all seen classes.
"""
import csv
import sys
from collections import defaultdict
from statistics import mean

BIN = 50

bins = defaultdict(lambda: defaultdict(list))
evals = defaultdict(lambda: defaultdict(list))
for run_dir in sys.argv[1:]:
    arm = run_dir.rstrip("/").rsplit("/", 1)[-1].rsplit("_s", 1)[0]
    rows = defaultdict(list)
    for row in csv.DictReader(open(f"{run_dir}/log.csv")):
        rows[int(row["task"]) // BIN].append(row)
    for b, rs in rows.items():
        if len(rs) == BIN:
            bins[arm][b].append([mean(float(r[k]) for r in rs) for k in ("test_acc_pair", "test_acc_seen")])
    for row in csv.DictReader(open(f"{run_dir}/eval.csv")):
        evals[arm][int(row["after_task"]) + 1].append(
            [float(row[k]) for k in ("old_acc_seen", "old_acc_old", "all_acc_seen")])

print("arm,first_task,runs,new_acc_pair,new_acc_seen")
for arm, bs in sorted(bins.items()):
    for b, values in sorted(bs.items()):
        print(arm, b * BIN, len(values), *(f"{100 * mean(v):.2f}" for v in zip(*values)), sep=",")
print("arm,tasks_seen,runs,old_acc_seen,old_acc_old,all_acc_seen")
for arm, es in sorted(evals.items()):
    for t, values in sorted(es.items()):
        print(arm, t, len(values), *(f"{100 * mean(v):.2f}" for v in zip(*values)), sep=",")
