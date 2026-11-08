"""README figures from the pulled logs: python plots.py (writes results/figures/*.png)

Color is the method (plain blue, SWR orange); a dashed line means replay of tasks 0-19.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "figures"
BIN = 50
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARMS = [  # (dir prefix, label, color, dashed)
    ("bp_lr", "Plain", "#2a78d6", False),
    ("swr_k0.01_f2048_lr", "SWR", "#eb6834", False),
    ("bp_rp", "Plain + replay", "#2a78d6", True),
    ("swr_k0.01_f2048_rp", "SWR + replay", "#eb6834", True),
]


def arm_dirs(folder, prefix):
    return sorted(d for d in folder.glob(f"{prefix}*_s[0-9]") if d.is_dir())


def binned(dirs, field):
    """Mean over seeds of the per-run mean of field in each full 50-task bin."""
    per_bin = defaultdict(list)
    for d in dirs:
        rows = list(csv.DictReader(open(d / "log.csv")))
        for b in range(len(rows) // BIN):
            per_bin[b].append(mean(float(r[field]) for r in rows[b * BIN:(b + 1) * BIN]))
    return [b * BIN + BIN / 2 for b in sorted(per_bin)], [100 * mean(per_bin[b]) for b in sorted(per_bin)]


def evals(dirs, field):
    per_t = defaultdict(list)
    for d in dirs:
        for r in csv.DictReader(open(d / "eval.csv")):
            per_t[int(r["after_task"]) + 1].append(float(r[field]))
    return sorted(per_t), [100 * mean(per_t[t]) for t in sorted(per_t)]


def figure(title, ylabel, xlabel, series, path, note=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ends = defaultdict(list)  # Series that end at the same value share one label.
    for label, color, dashed, (x, y) in series:
        ax.plot(x, y, color=color, lw=2, ls=(0, (5, 3)) if dashed else "-", marker="o", ms=4,
                markeredgewidth=0, label=label)
        ends[(x[-1], round(y[-1], 1))].append(label)
    for (x, y), labels in ends.items():
        ax.annotate(f"{', '.join(labels)} {y:.1f}%", (x, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=12)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best", handlelength=3.5)
    ax.set_xlim(right=ax.get_xlim()[1] + 75)  # Room for the end labels.
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


OUT.mkdir(exist_ok=True)
til, cil = RESULTS / "lop_rp600", RESULTS / "lop_cil"
figure("Task-incremental: accuracy on each new task", "Test accuracy (%)", "Task (mean of each 50-task bin)",
       [(l, c, dash, binned(arm_dirs(til, p), "test_acc")) for p, l, c, dash in ARMS],
       OUT / "task_incremental_plasticity.png", "3 seeds. Replay: all train images of tasks 0-19.")
figure("Class-incremental: new task, picking between its own 2 classes", "Test accuracy (%)",
       "Task (mean of each 50-task bin)",
       [(l, c, dash, binned(arm_dirs(cil, p), "test_acc_pair")) for p, l, c, dash in ARMS],
       OUT / "class_incremental_plasticity.png", "6 seeds. Measures plasticity.")
figure("Class-incremental: tasks 0-19, picking among all seen classes", "Test accuracy (%)", "Tasks learned",
       [(l, c, dash, evals(arm_dirs(cil, p), "old_acc_seen")) for p, l, c, dash in ARMS],
       OUT / "class_incremental_memory.png", "6 seeds. Without replay both are at 0%.")
figure("Class-incremental: new task, picking among all seen classes", "Test accuracy (%)",
       "Task (mean of each 50-task bin)",
       [(l, c, dash, binned(arm_dirs(cil, p), "test_acc_seen")) for p, l, c, dash in ARMS],
       OUT / "class_incremental_new_vs_old.png", "6 seeds. Replay arms mislabel some new images as old classes.")
