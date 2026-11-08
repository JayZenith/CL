"""Memory vs plasticity from a finished run: python savings.py runs/<run>/

From the final checkpoint, on separate copies, relearns old tasks and learns never-seen ones.
Old tasks relearned faster than new ones means the network kept something; equal means it is only plastic.
"""
import argparse
import copy
import csv
import re
from pathlib import Path

import numpy as np
import torch

from train import SPEED, THRESHOLDS, build_net, build_swr, epochs_to, load_task, test, train_task

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--old", type=int, nargs=2, default=(250, 270), help="Range of learned tasks to relearn")
parser.add_argument("--new", type=int, nargs=2, default=(300, 320), help="Range of never-seen tasks to learn")
cli = parser.parse_args()

# Settings come from the run name, e.g. swr_k0.01_f2048_lr0.001_s0 or bp_lr0.001_s0.
name = cli.run_dir.resolve().name
match = re.fullmatch(r"(?:bp|swr_k(?P<k>[^_]+)_f(?P<f>\d+))(?:_rp\d+b\d+(?:t\d+)?)?(?:_x\d+)?_lr(?P<lr>[^_]+)_s(?P<seed>\d+)", name)
args = argparse.Namespace(lr=float(match["lr"]), seed=int(match["seed"]), swr_k=float(match["k"] or 0),
                          swr_freq=int(match["f"] or 0), extra_batch=0, epochs=250, batch_size=100, momentum=0.9)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

ckpt = torch.load(cli.run_dir / "latest.pt", map_location=device, weights_only=True)
assert ckpt["next_task"] >= cli.old[1] and ckpt["next_task"] <= cli.new[0], ckpt["next_task"]
net = build_net().to(device)
net.load_state_dict(ckpt["net"])
order = np.load("data/class_order.npy")[args.seed]

with open(cli.run_dir / "savings.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["kind", "task", "test_acc_now", *SPEED])
    writer.writeheader()
    for kind, (lo, hi) in (("old", cli.old), ("new", cli.new)):
        for t in range(lo, hi):
            x_train, y_train, x_test, y_test = load_task([order[(2 * t + i) % len(order)] for i in range(2)], device)
            test_now = test(net, x_test, y_test, ckpt["heads"][t]) if kind == "old" else ""
            copy_net = copy.deepcopy(net)
            optimizer = torch.optim.SGD(copy_net.parameters(), lr=args.lr, momentum=args.momentum)
            swr = build_swr(copy_net, args) if args.swr_k else None
            accs, _, _ = train_task(copy_net, optimizer, swr, x_train, y_train, args, stop_at=THRESHOLDS[-1])
            writer.writerow({"kind": kind, "task": t, "test_acc_now": f"{test_now:.5f}" if kind == "old" else "",
                             **dict(zip(SPEED, epochs_to(accs)))})
            print(f"{name} {kind} task {t} epochs_95 {epochs_to(accs)[1]}", flush=True)
