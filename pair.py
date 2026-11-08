"""2-way memory check on finished class-incremental runs: python pair.py runs_cil/*/ > pair.csv

On the final checkpoint, accuracy on each task's test images picking between only that task's own 2 classes, for
tasks 0-19 and the last 50 tasks. This asks whether the features still separate a pair, apart from which classes
the head favors.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from cil import accuracy, load
from train import build_net

device = torch.device("cuda")
writer = csv.writer(sys.stdout)
writer.writerow(["run", "old_pair", "recent_pair"])
for run_dir in map(Path, sys.argv[1:]):
    ckpt = torch.load(run_dir / "latest.pt", map_location=device, weights_only=True)
    tasks = ckpt["next_task"]
    net = build_net(ckpt["net"]["14.weight"].shape[0]).to(device)
    net.load_state_dict(ckpt["net"])
    order = np.load("data/class_order.npy")[int(run_dir.resolve().name.rsplit("_s", 1)[1])]
    accs = []
    for tasks_in in (range(20), range(tasks - 50, tasks)):
        x_y = [load(order, [t], device, test_only=True) for t in tasks_in]
        accs.append(np.mean([accuracy(net, x, y, 2 * t, 2 * t + 2) for t, (x, y) in zip(tasks_in, x_y)]))
    writer.writerow([run_dir.resolve().name, *(f"{a:.5f}" for a in accs)])
    sys.stdout.flush()
