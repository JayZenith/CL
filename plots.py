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


def pair_means(path):
    per_arm = defaultdict(list)
    for r in csv.DictReader(open(path)):
        per_arm[r["run"].rsplit("_s", 1)[0]].append(100 * float(r["recent_pair"]))
    return {a: mean(v) for a, v in per_arm.items()}


pairs = {**pair_means(RESULTS / "lop_cil" / "pair.csv"), **pair_means(RESULTS / "lop_cil_nohead" / "pair.csv")}
bars = [("SWR + replay", "swr_k0.01_f2048_rpt20b50_lr0.001", "#eb6834"),
        ("Plain + replay", "bp_rpt20b50_lr0.001", "#2a78d6"),
        ("SWR + replay,\nno head", "swr_k0.01_f2048_nohead_rpt20b50_lr0.001", "#1baf7a")]
fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ys = range(len(bars))
values = [pairs[k] for _, k, _ in bars]
ax.barh(ys, values, height=0.6, color=[c for *_, c in bars], edgecolor=SURFACE, linewidth=2)
for y, v in zip(ys, values):
    ax.annotate(f"{v:.1f}%", (v, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
ax.axvline(50, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.annotate("chance", (50, len(bars) - 0.45), xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED)
ax.set_yticks(list(ys), [b[0] for b in bars], fontsize=9, color=INK)
ax.set_xlim(0, 100)
ax.set_xlabel("Test accuracy (%)", fontsize=9, color=MUTED)
ax.set_title("Recent tasks 250-299 (not replayed), own 2 classes, at task 300", loc="left", fontsize=11, color=INK,
             pad=12)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=8, length=0)
for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color(GRID)
fig.text(0.01, 0.01, "6 seeds. SWR's head resets erase classes it stops training; leaving out the head fixes it.",
         fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "class_incremental_head_resets.png", facecolor=SURFACE)
plt.close(fig)
