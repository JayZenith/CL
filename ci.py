"""Paired 95% confidence intervals over the 6 class-incremental seeds.

Seeds are paired across arms: seed s has the same class order in every arm.
"""
import csv
import math
from pathlib import Path
from statistics import mean, stdev

R = Path(__file__).parent / "results"
CIL, NOHEAD, X50 = R / "lop_cil", R / "lop_cil_nohead", R / "lop_cil_x50"
SEEDS = range(6)
T95 = 2.5706  # t, 5 degrees of freedom

SWR = "swr_k0.01_f2048"
NH = "swr_k0.01_f2048_nohead"


def plasticity(folder, arm):
    """Mean 2-way test accuracy on new tasks 250-299."""
    out = []
    for s in SEEDS:
        rows = list(csv.DictReader(open(folder / f"{arm}_lr0.001_s{s}" / "log.csv")))[250:300]
        assert len(rows) == 50
        out.append(100 * mean(float(r["test_acc_pair"]) for r in rows))
    return out


def old(folder, arm):
    """Tasks 0-19 among all seen classes, at task 300."""
    out = []
    for s in SEEDS:
        r = list(csv.DictReader(open(folder / f"{arm}_lr0.001_s{s}" / "eval.csv")))[-1]
        assert r["after_task"] == "299"
        out.append(100 * float(r["old_acc_seen"]))
    return out


def recent_pair(folder, arm):
    """Tasks 250-299, own 2 classes, at task 300."""
    rows = {r["run"]: r for r in csv.DictReader(open(folder / "pair.csv"))}
    return [100 * float(rows[f"{arm}_lr0.001_s{s}"]["recent_pair"]) for s in SEEDS]


COMPARISONS = [
    ("New tasks 250-299: SWR vs plain", plasticity(CIL, SWR), plasticity(CIL, "bp")),
    ("New tasks 250-299: SWR + extra images vs SWR + replay",
     plasticity(X50, f"{SWR}_x50"), plasticity(CIL, f"{SWR}_rpt20b50")),
    ("Tasks 0-19 at task 300: SWR + replay vs plain + replay",
     old(CIL, f"{SWR}_rpt20b50"), old(CIL, "bp_rpt20b50")),
    ("New tasks 250-299: SWR + replay (no head) vs SWR + replay",
     plasticity(NOHEAD, f"{NH}_rpt20b50"), plasticity(CIL, f"{SWR}_rpt20b50")),
    ("Tasks 0-19 at task 300: SWR + replay (no head) vs SWR + replay",
     old(NOHEAD, f"{NH}_rpt20b50"), old(CIL, f"{SWR}_rpt20b50")),
    ("Tasks 250-299 pairs at task 300: SWR + replay (no head) vs SWR + replay",
     recent_pair(NOHEAD, f"{NH}_rpt20b50"), recent_pair(CIL, f"{SWR}_rpt20b50")),
    ("Tasks 250-299 pairs at task 300: SWR + replay (no head) vs plain + replay",
     recent_pair(NOHEAD, f"{NH}_rpt20b50"), recent_pair(CIL, "bp_rpt20b50")),
]

print("comparison,a,b,diff,ci_low,ci_high,a_wins")
for name, a, b in COMPARISONS:
    d = [x - y for x, y in zip(a, b)]
    m, h = mean(d), T95 * stdev(d) / math.sqrt(len(d))
    print(f"{name},{mean(a):.1f},{mean(b):.1f},{m:+.1f},{m - h:+.1f},{m + h:+.1f},{sum(x > 0 for x in d)}/6")
