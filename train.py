"""Backprop or SWR, optionally with replay, on Continual ImageNet (Dohare et al. 2024):
python train.py --seed 0 [--swr-k 1e-2] [--replay-per-class 10]"""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from swr import SelectiveWeightReinitialization

TRAIN_PER_CLASS = 600
THRESHOLDS = (0.9, 0.95, 0.99)
SPEED = [f"epochs_{int(100 * t)}" for t in THRESHOLDS]
LOG_FIELDS = ["task", "class_a", "class_b", "train_loss", "train_acc", "test_acc", *SPEED, "num_reinit"]
PROBE_FIELDS = ["after_task", "probe_task", "test_acc_learned", "test_acc_now",
                *[f"first_{s}" for s in SPEED], *[f"relearn_{s}" for s in SPEED]]


def build_net(outputs=2):
    """lop/nets/conv_net.py: 177,218 parameters with 2 outputs."""
    def conv(in_channels, out_channels, size):
        return [nn.Conv2d(in_channels, out_channels, size), nn.ReLU(), nn.MaxPool2d(2, 2)]
    return nn.Sequential(*conv(3, 32, 5), *conv(32, 64, 3), *conv(64, 128, 3), nn.Flatten(),
                         nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, outputs))


def build_swr(net, args):
    """SWR on every parameter. Resamples from a normal with the std of PyTorch's default init,
    U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases alike."""
    params, stds = [], []
    for layer in net:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            fan_in = layer.weight[0].numel()
            for p in (layer.weight, layer.bias):
                params.append(p)
                stds.append((1 / fan_in) ** 0.5 / 3 ** 0.5)
    return SelectiveWeightReinitialization(params, "gradient", "threshold", [0.0] * len(params), stds,
                                           args.swr_freq, args.swr_k)


def load_task(classes, device):
    x_train, y_train, x_test, y_test = [], [], [], []
    for label, name in enumerate(classes):
        x = torch.from_numpy(np.load(f"data/classes/{name}.npy")).float().to(device)
        x_train.append(x[:TRAIN_PER_CLASS])
        x_test.append(x[TRAIN_PER_CLASS:])
        y_train.append(torch.full((TRAIN_PER_CLASS,), label, device=device))
        y_test.append(torch.full((len(x) - TRAIN_PER_CLASS,), label, device=device))
    return torch.cat(x_train), torch.cat(y_train), torch.cat(x_test), torch.cat(y_test)


def train_task(net, optimizer, swr, x, y, args, stop_at=None, replay=None):
    """Trains a zeroed head plus the body. Returns per-epoch train accuracy, last-epoch loss and reinit count.

    replay is (x, y, head weights, head biases) with one frozen head per stored example. Each step adds
    the loss of args.replay_batch stored examples, each scored by the head of the task it came from.
    args.extra_batch instead adds the loss of that many random current-task examples, the control for replay's
    extra images per step.
    """
    body, head = net[:-1], net[-1]
    head.weight.data.zero_()
    head.bias.data.zero_()
    accs, num_reinit = [], 0
    for _ in range(args.epochs):
        loss_sum = hits = 0
        for batch in torch.randperm(len(x), device=x.device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            if replay is None:
                logits = net(x[batch])
                loss = F.cross_entropy(logits, y[batch])
                if args.extra_batch:
                    extra = torch.randint(len(x), (args.extra_batch,), device=x.device)
                    loss = loss + F.cross_entropy(net(x[extra]), y[extra])
            else:
                rx, ry, rw, rb = replay
                old = torch.randint(len(rx), (args.replay_batch,), device=x.device)
                features = body(torch.cat([x[batch], rx[old]]))
                logits = head(features[:len(batch)])
                old_logits = torch.einsum("nf,nkf->nk", features[len(batch):], rw[old]) + rb[old]
                loss = F.cross_entropy(logits, y[batch]) + F.cross_entropy(old_logits, ry[old])
            loss.backward()
            optimizer.step()
            if swr:  # After the step and before the next zero_grad, as gradient utility reads p.grad.
                swr.step()
                num_reinit += swr.num_replaced
                swr.reset_reinit_indicator()
            loss_sum += loss.detach() * len(batch)
            hits += (logits.detach().argmax(1) == y[batch]).sum()
        accs.append(hits.item() / len(x))
        if stop_at is not None and accs[-1] >= stop_at:
            break
    return accs, loss_sum.item() / len(x), num_reinit


def epochs_to(accs):
    """First epoch (1-based) whose train accuracy reaches each threshold; blank if never."""
    return [next((i + 1 for i, a in enumerate(accs) if a >= t), "") for t in THRESHOLDS]


@torch.no_grad()
def test(net, x, y, head=None):
    if head is not None:
        net = copy.deepcopy(net)
        net[-1].load_state_dict(head)
    return (net(x).argmax(1) == y).float().mean().item()


def open_csv(path, fields, keep):
    """Rewrites the log with only the rows that the checkpoint covers."""
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    writer = csv.DictWriter(open(path, "w", newline="", buffering=1), fieldnames=fields)
    writer.writeheader()
    writer.writerows(r for r in rows if keep(r))
    return writer, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Also picks the class order; 0-29 as in the paper")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--swr-k", type=float, default=0.0, help="SWR threshold factor; 0 = plain backprop")
    parser.add_argument("--swr-freq", type=int, default=2048, help="Steps between SWR resets")
    parser.add_argument("--replay-per-class", type=int, default=0, help="Stored train images per class; 0 = no replay")
    parser.add_argument("--replay-batch", type=int, default=50, help="Stored images replayed per step")
    parser.add_argument("--replay-tasks", type=int, default=0, help="Store only tasks 0 to this - 1; 0 = all")
    parser.add_argument("--extra-batch", type=int, default=0,
                        help="Random current-task images added per step, as in replay; 0 = none")
    parser.add_argument("--probe-every", type=int, default=50, help="Relearn the task this many tasks back")
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()

    arm = f"swr_k{args.swr_k}_f{args.swr_freq}" if args.swr_k else "bp"
    if args.replay_per_class:
        arm += f"_rp{args.replay_per_class}b{args.replay_batch}"
        if args.replay_tasks:
            arm += f"t{args.replay_tasks}"
    if args.extra_batch:
        arm += f"_x{args.extra_batch}"
    run_dir = Path(args.output_dir) / f"{arm}_lr{args.lr}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    order = np.load("data/class_order.npy")[args.seed]
    net = build_net().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    swr = build_swr(net, args) if args.swr_k else None
    heads = {}  # Head of each task when it was learned, to test it later with the current body.

    start = 0
    ckpt_path = run_dir / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if swr:
            swr.load_state_dict(ckpt["swr"])
        heads = {t: {k: v.to(device) for k, v in h.items()} for t, h in ckpt["heads"].items()}
        torch.set_rng_state(ckpt["rng"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        start = ckpt["next_task"]
        print(f"Resumed at task {start}", flush=True)

    log, rows = open_csv(run_dir / "log.csv", LOG_FIELDS, lambda r: int(r["task"]) < start)
    probes, _ = open_csv(run_dir / "probe.csv", PROBE_FIELDS, lambda r: int(r["after_task"]) < start)
    learned = {int(r["task"]): r for r in rows if int(r["task"]) < start}

    def classes_of(t):
        # The paper's order is a list of 14,000 classes, read two at a time, and wraps as in its code.
        return [order[(2 * t + i) % len(order)] for i in range(2)]

    # Replay buffer: the first replay_per_class train images of each class of each stored task.
    keep = [i for c in range(2) for i in range(c * TRAIN_PER_CLASS, c * TRAIN_PER_CLASS + args.replay_per_class)]
    stored = []

    def store(t, x_train, y_train):
        if args.replay_per_class and (not args.replay_tasks or t < args.replay_tasks):
            stored.append((x_train[keep], y_train[keep], t))

    def replay_set():
        if not stored:
            return None
        rx, ry, tasks = torch.cat([s[0] for s in stored]), torch.cat([s[1] for s in stored]), [s[2] for s in stored]
        per_task = len(keep)
        rw = torch.stack([heads[t]["weight"] for t in tasks]).repeat_interleave(per_task, 0)
        rb = torch.stack([heads[t]["bias"] for t in tasks]).repeat_interleave(per_task, 0)
        return rx, ry, rw, rb

    for t in range(min(start, args.replay_tasks or start) if args.replay_per_class else 0):
        store(t, *load_task(classes_of(t), device)[:2])

    for t in range(start, args.tasks):
        classes = classes_of(t)
        x_train, y_train, x_test, y_test = load_task(classes, device)
        accs, loss, num_reinit = train_task(net, optimizer, swr, x_train, y_train, args, replay=replay_set())
        # The paper's plots use only the accuracy after the last epoch.
        test_acc = test(net, x_test, y_test)
        heads[t] = {k: v.clone() for k, v in net[-1].state_dict().items()}
        store(t, x_train, y_train)
        learned[t] = {"task": t, "class_a": classes[0], "class_b": classes[1], "train_loss": f"{loss:.5f}",
                      "train_acc": f"{accs[-1]:.5f}", "test_acc": f"{test_acc:.5f}",
                      **dict(zip(SPEED, epochs_to(accs))), "num_reinit": num_reinit}
        log.writerow(learned[t])

        # Forgetting and relearning, on a copy so the main run is untouched.
        p = t + 1 - args.probe_every
        if args.probe_every and (t + 1) % args.probe_every == 0 and p >= 0:
            x_train, y_train, x_test, y_test = load_task(classes_of(p), device)
            test_now = test(net, x_test, y_test, heads[p])
            copy_net = copy.deepcopy(net)
            copy_opt = torch.optim.SGD(copy_net.parameters(), lr=args.lr, momentum=args.momentum)
            relearn, _, _ = train_task(copy_net, copy_opt, build_swr(copy_net, args) if swr else None,
                                       x_train, y_train, args, stop_at=THRESHOLDS[-1])
            probes.writerow({"after_task": t, "probe_task": p, "test_acc_learned": learned[p]["test_acc"],
                             "test_acc_now": f"{test_now:.5f}",
                             **{f"first_{s}": learned[p][s] for s in SPEED},
                             **{f"relearn_{s}": e for s, e in zip(SPEED, epochs_to(relearn))}})
            del copy_net, copy_opt

        if (t + 1) % args.ckpt_every == 0 or t == args.tasks - 1:
            torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "swr": swr.state_dict() if swr else None, "heads": heads, "next_task": t + 1,
                        "rng": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()},
                       run_dir / "latest.tmp")
            (run_dir / "latest.tmp").replace(ckpt_path)
        print(f"task {t + 1}/{args.tasks} test_acc {test_acc:.3f} reinit {num_reinit}", flush=True)


if __name__ == "__main__":
    main()
