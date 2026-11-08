"""Head-bias check on finished class-incremental runs: python wa.py runs_cil/*/ > runs_cil/wa.csv

Replayed classes end up with much larger head rows, so they win too often. Weight aligning (Zhao et al. 2020),
in its simplest form: scale every class's head row to the mean row norm, then test again. All on the final
checkpoint, before and after, picking among all seen classes.
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
writer.writerow(["run", "aligned", "old_acc", "recent_acc", "all_acc"])
for run_dir in map(Path, sys.argv[1:]):
    ckpt = torch.load(run_dir / "latest.pt", map_location=device, weights_only=True)
    tasks = ckpt["next_task"]
    seen = 2 * tasks
    net = build_net(ckpt["net"]["14.weight"].shape[0]).to(device)
    net.load_state_dict(ckpt["net"])
    order = np.load("data/class_order.npy")[int(run_dir.resolve().name.rsplit("_s", 1)[1])]
    sets = {"old": range(20), "recent": range(tasks - 50, tasks), "all": range(tasks)}
    for aligned in (False, True):
        if aligned:
            with torch.no_grad():
                w = net[-1].weight[:seen]
                norms = w.norm(dim=1, keepdim=True)
                w.mul_(norms.mean() / norms)
        accs = []
        for name, tasks_in in sets.items():
            hits = n = 0
            for lo in range(0, len(tasks_in), 25):  # Chunks bound GPU memory.
                x, y = load(order, tasks_in[lo:lo + 25], device, test_only=True)
                hits += accuracy(net, x, y, 0, seen) * len(x)
                n += len(x)
            accs.append(f"{hits / n:.5f}")
        writer.writerow([run_dir.resolve().name, int(aligned), *accs])
        sys.stdout.flush()
