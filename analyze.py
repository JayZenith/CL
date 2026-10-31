"""Summarizes runs: python analyze.py runs/*  (writes summary.csv next to the runs)

For each metric, higher is better (nll is negated):
  gain       post-training minus pre-training score on a new task  (plasticity)
  gain_slope change in gain per task; negative means plasticity loss
  speed      mean improvement over pre-training across the learning curve  (learning speed)
  forgetting best score on a task minus its final score, excluding the last task
  final      final mean score over all tasks
early/late are the first and last 10 tasks.
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

WINDOW = 10


def summarize(rows, field):
    sign = -1 if field == "nll" else 1
    curves, history = defaultdict(dict), defaultdict(dict)
    for r in rows:
        trained, step, task, value = int(r["trained_task"]), int(r["task_step"]), int(r["eval_task"]), sign * float(r[field])
        if trained == task:
            curves[task][step] = value
        history[task][trained] = value  # Keeps the post-training value for the current task.
    tasks = sorted(curves)
    pre = [curves[t][0] for t in tasks]
    post = [curves[t][max(curves[t])] for t in tasks]
    gain = [b - a for a, b in zip(pre, post)]
    speed = []
    for t, base in zip(tasks, pre):
        steps = sorted(curves[t])
        area = sum((b - a) * ((curves[t][a] + curves[t][b]) / 2 - base) for a, b in zip(steps, steps[1:]))
        speed.append(area / steps[-1])
    last = tasks[-1]
    final = [history[t][last] for t in tasks]
    forgetting = [max(history[t].values()) - history[t][last] for t in tasks[:-1]]
    result = {
        "gain": st.mean(gain), "gain_early": st.mean(gain[:WINDOW]), "gain_late": st.mean(gain[-WINDOW:]),
        "gain_slope": st.linear_regression(tasks, gain).slope,
        "speed": st.mean(speed), "speed_early": st.mean(speed[:WINDOW]), "speed_late": st.mean(speed[-WINDOW:]),
        "forgetting": st.mean(forgetting), "final": st.mean(final),
    }
    return {f"{field}_{k}": v for k, v in result.items()}


def main():
    runs = [Path(p) for p in sys.argv[1:] if (Path(p) / "eval.csv").exists()]
    results = []
    for run in runs:
        with open(run / "eval.csv") as f:
            rows = list(csv.DictReader(f))
        arm, seed = run.name.rsplit("_s", 1)
        results.append({"arm": arm, "seed": seed, **summarize(rows, "exact_match"), **summarize(rows, "nll")})

    with open(runs[0].parent / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    arms = sorted({r["arm"] for r in results})
    print("metric," + ",".join(arms))
    for key in list(results[0])[2:]:
        cells = []
        for arm in arms:
            values = [r[key] for r in results if r["arm"] == arm]
            std = st.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{st.mean(values):.5f}±{std:.5f}")
        print(f"{key}," + ",".join(cells))


if __name__ == "__main__":
    main()
