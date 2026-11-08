"""Class-incremental Continual ImageNet: python cil.py --seed 0 [--swr-k 1e-2] [--replay-tasks 20]

One head for all classes. Task t adds classes 2t and 2t+1, and the model always picks from every class seen
so far, with no hint which task an image is from. Replay keeps all train images of tasks 0 to
replay_tasks - 1 and adds args.replay_batch of them to each step.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train import SPEED, TRAIN_PER_CLASS, build_net, build_swr, epochs_to, open_csv

LOG_FIELDS = ["task", "class_a", "class_b", "train_loss", "train_acc", "test_acc_pair", "test_acc_seen", *SPEED,
              "num_reinit"]
EVAL_FIELDS = ["after_task", "old_acc_seen", "old_acc_old", "all_acc_seen"]


def load(order, tasks, device, test_only=False):
    """Images and global labels (2t, 2t + 1) of the given tasks."""
    xs, ys = [], []
    for t in tasks:
        for i in range(2):
            x = np.load(f"data/classes/{order[2 * t + i]}.npy", mmap_mode="r")
            x = x[TRAIN_PER_CLASS:] if test_only else x[:TRAIN_PER_CLASS]
            xs.append(torch.from_numpy(np.ascontiguousarray(x)).float().to(device))
            ys.append(torch.full((len(x),), 2 * t + i, device=device))
    return torch.cat(xs), torch.cat(ys)


def train_task(net, optimizer, swr, x, y, seen, args, replay=None):
    """Trains on one task over the logits of the seen classes. Returns per-epoch train accuracy, loss, reinits."""
    accs, num_reinit = [], 0
    for _ in range(args.epochs):
        loss_sum = hits = 0
        for batch in torch.randperm(len(x), device=x.device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = net(x[batch])[:, :seen]
            loss = F.cross_entropy(logits, y[batch])
            if replay is not None:
                old = torch.randint(len(replay[0]), (args.replay_batch,), device=x.device)
                loss = loss + F.cross_entropy(net(replay[0][old])[:, :seen], replay[1][old])
            if args.extra_batch:  # Control: as many extra images per step as replay adds, from the current task.
                extra = torch.randint(len(x), (args.extra_batch,), device=x.device)
                loss = loss + F.cross_entropy(net(x[extra])[:, :seen], y[extra])
            loss.backward()
            optimizer.step()
            if swr:  # After the step and before the next zero_grad, as gradient utility reads p.grad.
                swr.step()
                num_reinit += swr.num_replaced
                swr.reset_reinit_indicator()
            loss_sum += loss.detach() * len(batch)
            hits += (logits.detach().argmax(1) == y[batch]).sum()
        accs.append(hits.item() / len(x))
    return accs, loss_sum.item() / len(x), num_reinit


@torch.no_grad()
def accuracy(net, x, y, lo, hi):
    """Accuracy when picking among classes lo to hi - 1."""
    return torch.cat([(net(b)[:, lo:hi].argmax(1) + lo == yb) for b, yb in zip(x.split(500), y.split(500))]
                     ).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tasks", type=int, default=300, help="At most 500, where the class order repeats")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--swr-k", type=float, default=0.0, help="SWR threshold factor; 0 = plain backprop")
    parser.add_argument("--swr-freq", type=int, default=2048)
    parser.add_argument("--replay-tasks", type=int, default=0, help="Replay tasks 0 to this - 1; 0 = no replay")
    parser.add_argument("--replay-batch", type=int, default=50)
    parser.add_argument("--extra-batch", type=int, default=0, help="Extra current-task images per step; 0 = none")
    parser.add_argument("--old-tasks", type=int, default=20, help="Tasks 0 to this - 1 are tracked as old")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--output-dir", default="runs_cil")
    args = parser.parse_args()
    assert args.tasks <= 500

    arm = f"swr_k{args.swr_k}_f{args.swr_freq}" if args.swr_k else "bp"
    if args.replay_tasks:
        arm += f"_rpt{args.replay_tasks}b{args.replay_batch}"
    if args.extra_batch:
        arm += f"_x{args.extra_batch}"
    run_dir = Path(args.output_dir) / f"{arm}_lr{args.lr}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    order = np.load("data/class_order.npy")[args.seed]
    net = build_net(2 * args.tasks).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    swr = build_swr(net, args) if args.swr_k else None

    start = 0
    ckpt_path = run_dir / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if swr:
            swr.load_state_dict(ckpt["swr"])
        torch.set_rng_state(ckpt["rng"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        start = ckpt["next_task"]
        print(f"Resumed at task {start}", flush=True)

    log, _ = open_csv(run_dir / "log.csv", LOG_FIELDS, lambda r: int(r["task"]) < start)
    evals, _ = open_csv(run_dir / "eval.csv", EVAL_FIELDS, lambda r: int(r["after_task"]) < start)
    replay = load(order, range(min(start, args.replay_tasks)), device) if start and args.replay_tasks else None
    old_x, old_y = load(order, range(args.old_tasks), device, test_only=True)

    for t in range(start, args.tasks):
        seen = 2 * (t + 1)
        x_train, y_train = load(order, [t], device)
        x_test, y_test = load(order, [t], device, test_only=True)
        accs, loss, num_reinit = train_task(net, optimizer, swr, x_train, y_train, seen, args, replay)
        if t < args.replay_tasks:
            replay = (x_train, y_train) if replay is None else (torch.cat([replay[0], x_train]),
                                                                torch.cat([replay[1], y_train]))
        log.writerow({"task": t, "class_a": order[2 * t], "class_b": order[2 * t + 1], "train_loss": f"{loss:.5f}",
                      "train_acc": f"{accs[-1]:.5f}",
                      "test_acc_pair": f"{accuracy(net, x_test, y_test, 2 * t, seen):.5f}",
                      "test_acc_seen": f"{accuracy(net, x_test, y_test, 0, seen):.5f}",
                      **dict(zip(SPEED, epochs_to(accs))), "num_reinit": num_reinit})

        if (t + 1) % args.eval_every == 0 or t == args.tasks - 1:
            hits = n = 0
            for lo in range(0, t + 1, 25):  # All seen classes, in chunks to bound memory.
                x, y = load(order, range(lo, min(lo + 25, t + 1)), device, test_only=True)
                hits += accuracy(net, x, y, 0, seen) * len(x)
                n += len(x)
            old = min(t + 1, args.old_tasks)
            mask = old_y < 2 * old
            evals.writerow({"after_task": t, "old_acc_seen": f"{accuracy(net, old_x[mask], old_y[mask], 0, seen):.5f}",
                            "old_acc_old": f"{accuracy(net, old_x[mask], old_y[mask], 0, 2 * old):.5f}",
                            "all_acc_seen": f"{hits / n:.5f}"})
            del x, y
            torch.cuda.empty_cache()  # Twelve runs share two GPUs, so return the eval memory.
            torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "swr": swr.state_dict() if swr else None, "next_task": t + 1,
                        "rng": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()},
                       run_dir / "latest.tmp")
            (run_dir / "latest.tmp").replace(ckpt_path)
        print(f"task {t + 1}/{args.tasks} train_acc {accs[-1]:.3f} reinit {num_reinit}", flush=True)


if __name__ == "__main__":
    main()
