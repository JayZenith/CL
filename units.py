"""Plasticity diagnostics from Dohare et al. 2024: python units.py runs/*/ > units.csv

For each run's latest checkpoint, on the train images of the next 10 never-seen tasks:
dead units (zero on every image), mean |weight|, and effective rank of the last hidden layer.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train import build_net, load_task

parser = argparse.ArgumentParser()
parser.add_argument("run_dirs", type=Path, nargs="+")
parser.add_argument("--tasks", type=int, default=10, help="Never-seen tasks whose images are used")
cli = parser.parse_args()
device = torch.device("cuda")

writer = csv.writer(sys.stdout)
writer.writerow(["run", "next_task", "dead_frac", "dead_conv", "dead_fc", "mean_abs_weight", "effective_rank"])
for run_dir in cli.run_dirs:
    ckpt = torch.load(run_dir / "latest.pt", map_location=device, weights_only=True)
    net = build_net().to(device)
    net.load_state_dict(ckpt["net"])
    seed = int(run_dir.resolve().name.rsplit("_s", 1)[1])
    order = np.load("data/class_order.npy")[seed]
    start = ckpt["next_task"]
    x = torch.cat([load_task([order[(2 * t + i) % len(order)] for i in range(2)], device)[0]
                   for t in range(start, start + cli.tasks)])

    # Per ReLU, whether each unit (a channel for conv) was ever positive.
    alive, outputs = {}, {}
    relus = [m for m in net if isinstance(m, nn.ReLU)]
    for m in relus:
        m.register_forward_hook(lambda m, i, o: outputs.__setitem__(m, o))
    features = []
    with torch.no_grad():
        for batch in x.split(1000):
            net(batch)
            for m in relus:
                o = outputs[m]
                on = (o > 0).flatten(2).any(2).any(0) if o.dim() == 4 else (o > 0).any(0)
                alive[m] = alive[m] | on if m in alive else on
            features.append(outputs[relus[-1]])
    dead = [(~alive[m]).float() for m in relus]
    sv = torch.linalg.svdvals(torch.cat(features))
    p = sv / sv.sum()
    rank = torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item()
    weights = torch.cat([m.weight.abs().flatten() for m in net if isinstance(m, (nn.Conv2d, nn.Linear))])
    writer.writerow([run_dir.resolve().name, start, f"{torch.cat(dead).mean().item():.4f}",
                     f"{torch.cat(dead[:3]).mean().item():.4f}", f"{torch.cat(dead[3:]).mean().item():.4f}",
                     f"{weights.mean().item():.5f}", f"{rank:.2f}"])
