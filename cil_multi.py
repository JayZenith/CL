"""cil.py for several seeds in one process: python cil_multi.py --seeds 0 1 2 3 4 5 [cil.py args]

Same training, per seed, as cil.py. The seeds' nets are stacked into one model (grouped convs, batched matmuls), so
each step is one set of GPU calls for all seeds instead of one per seed. Each seed keeps its own weights, class
order, batches, SGD momentum and SWR threshold, and the summed loss gives each seed exactly its own gradient. Only
the random streams differ from separate cil.py runs. Writes the same per-seed run dirs, logs and latest.pt.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cil import EVAL_FIELDS, LOG_FIELDS, accuracy, load
from train import SPEED, build_net, epochs_to, open_csv

LAYERS = (0, 3, 6, 10, 12, 14)  # Indices of build_net's conv and linear layers.


class StackedNet(nn.Module):
    """S copies of build_net. Parameters are [S, *shape] and inputs [S, B, 3, 32, 32]."""

    def __init__(self, nets):
        super().__init__()
        self.s = len(nets)
        self.weights = nn.ParameterList(torch.stack([n[i].weight.detach() for n in nets]).clone() for i in LAYERS)
        self.biases = nn.ParameterList(torch.stack([n[i].bias.detach() for n in nets]).clone() for i in LAYERS)

    def forward(self, x):
        s, b = x.shape[:2]
        h = x.transpose(0, 1).reshape(b, s * 3, 32, 32)
        for w, bias in zip(self.weights[:3], self.biases[:3]):
            h = F.max_pool2d(F.relu(F.conv2d(h, w.flatten(0, 1), bias.flatten(), groups=s)), 2, 2)
        h = h.reshape(b, s, -1).transpose(0, 1)  # [S, B, 512], flattened as nn.Flatten does.
        for i, (w, bias) in enumerate(zip(self.weights[3:], self.biases[3:])):
            h = torch.baddbmm(bias.unsqueeze(1), h, w.transpose(1, 2))
            if i < 2:
                h = F.relu(h)
        return h

    def state_dict_of(self, s):
        """Seed s's weights under build_net's keys."""
        sd = {}
        for i, w, b in zip(LAYERS, self.weights, self.biases):
            sd[f"{i}.weight"], sd[f"{i}.bias"] = w[s].detach().clone(), b[s].detach().clone()
        return sd


class StackedSWR:
    """SWR (gradient utility, threshold pruning, resample) with each seed's threshold over its own slice of each
    tensor, as in a separate run. Same rule as swr.SelectiveWeightReinitialization."""

    def __init__(self, params, stds, freq, k):
        self.params, self.stds, self.freq, self.k = list(params), stds, freq, k
        self.current_step = 0
        self.num_replaced = torch.zeros(len(self.params[0]), dtype=torch.long)

    @torch.no_grad()
    def step(self):
        self.current_step += 1
        if self.current_step % self.freq:
            return
        for p, std in zip(self.params, self.stds):
            u = (p.grad * p).abs().flatten(1)
            mask = u <= self.k * u.mean(1, keepdim=True)
            p.view(len(p), -1)[mask] = torch.randn(int(mask.sum()), device=p.device, dtype=p.dtype) * std
            self.num_replaced += mask.sum(1).cpu()

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, sd):
        self.current_step = sd["current_step"]


def stds_of(net):
    """Resample std per parameter, as train.build_swr: std of U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    return [(1 / net[i].weight[0].numel()) ** 0.5 / 3 ** 0.5 for i in LAYERS for _ in range(2)]


def ce(logits, y):
    """Each seed's mean cross-entropy, [S]. Summed over seeds, each seed gets exactly its own gradient."""
    return F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(y.shape).mean(1)


@torch.compile
def step_loss(net, unseen, x, y, xr=None, yr=None, xe=None, ye=None):
    """Loss over the seen classes. Masking unseen logits to -inf is the same softmax as slicing them off, and keeps
    shapes fixed so this compiles once instead of once per task."""
    logits = net(x).masked_fill(unseen, float("-inf"))
    per_seed = ce(logits, y)
    loss = per_seed.sum()
    if xr is not None:
        loss = loss + ce(net(xr).masked_fill(unseen, float("-inf")), yr).sum()
    if xe is not None:
        loss = loss + ce(net(xe).masked_fill(unseen, float("-inf")), ye).sum()
    return loss, per_seed, logits


def train_task(net, optimizer, swr, x, y, seen, args, replay=None):
    """x [S, N, ...], y [S, N]. Returns per-seed lists of per-epoch train accuracy, losses, reinit counts."""
    s, n = y.shape
    rows = torch.arange(s, device=x.device)[:, None]
    accs = [[] for _ in range(s)]
    if swr:
        swr.num_replaced.zero_()
    unseen = torch.arange(net.weights[-1].shape[1], device=x.device) >= seen
    for _ in range(args.epochs):
        loss_sum = torch.zeros(s, device=x.device)
        hits = torch.zeros(s, device=x.device)
        perm = torch.rand(s, n, device=x.device).argsort(1)
        for i in range(0, n, args.batch_size):
            idx = perm[:, i:i + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            yb = y[rows, idx]
            more = {}
            if replay is not None:
                old = torch.randint(replay[1].shape[1], (s, args.replay_batch), device=x.device)
                more.update(xr=replay[0][rows, old], yr=replay[1][rows, old])
            if args.extra_batch:
                extra = torch.randint(n, (s, args.extra_batch), device=x.device)
                more.update(xe=x[rows, extra], ye=y[rows, extra])
            loss, per_seed, logits = step_loss(net, unseen, x[rows, idx], yb, **more)
            loss.backward()
            optimizer.step()
            if swr:
                swr.step()
            loss_sum += per_seed.detach() * idx.shape[1]
            hits += (logits.detach().argmax(-1) == yb).sum(1)
        for j, h in enumerate((hits / n).tolist()):
            accs[j].append(h)
    return accs, (loss_sum / n).tolist(), swr.num_replaced.tolist() if swr else [0] * s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tasks", type=int, default=300, help="At most 500, where the class order repeats")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--swr-k", type=float, default=0.0, help="SWR threshold factor; 0 = plain backprop")
    parser.add_argument("--swr-freq", type=int, default=2048)
    parser.add_argument("--swr-no-head", action="store_true", help="SWR on every layer except the head")
    parser.add_argument("--replay-tasks", type=int, default=0, help="Replay tasks 0 to this - 1; 0 = no replay")
    parser.add_argument("--replay-batch", type=int, default=50)
    parser.add_argument("--extra-batch", type=int, default=0, help="Extra current-task images per step; 0 = none")
    parser.add_argument("--old-tasks", type=int, default=20, help="Tasks 0 to this - 1 are tracked as old")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--output-dir", default="runs_cil")
    args = parser.parse_args()
    assert args.tasks <= 500

    arm = f"swr_k{args.swr_k}_f{args.swr_freq}{'_nohead' if args.swr_no_head else ''}" if args.swr_k else "bp"
    if args.replay_tasks:
        arm += f"_rpt{args.replay_tasks}b{args.replay_batch}"
    if args.extra_batch:
        arm += f"_x{args.extra_batch}"
    run_dirs = [Path(args.output_dir) / f"{arm}_lr{args.lr}_s{s}" for s in args.seeds]
    for d in run_dirs:
        d.mkdir(parents=True, exist_ok=True)
    group_ckpt = Path(args.output_dir) / f"{arm}_lr{args.lr}_seeds{'-'.join(map(str, args.seeds))}.pt"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    nets = []
    for s in args.seeds:  # Each seed's init is what cil.py would draw.
        torch.manual_seed(s)
        nets.append(build_net(2 * args.tasks))
    net = StackedNet(nets).to(device)
    eval_net = build_net(2 * args.tasks).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    params = [p for pair in zip(net.weights, net.biases) for p in pair]
    stds = stds_of(nets[0])
    if args.swr_no_head:  # The head is the last weight and bias.
        params, stds = params[:-2], stds[:-2]
    swr = StackedSWR(params, stds, args.swr_freq, args.swr_k) if args.swr_k else None
    torch.manual_seed(args.seeds[0])
    orders = [np.load("data/class_order.npy")[s] for s in args.seeds]

    start = 0
    if group_ckpt.exists():
        ckpt = torch.load(group_ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if swr:
            swr.load_state_dict(ckpt["swr"])
        torch.set_rng_state(ckpt["rng"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        start = ckpt["next_task"]
        print(f"Resumed at task {start}", flush=True)

    logs = [open_csv(d / "log.csv", LOG_FIELDS, lambda r: int(r["task"]) < start)[0] for d in run_dirs]
    evals = [open_csv(d / "eval.csv", EVAL_FIELDS, lambda r: int(r["after_task"]) < start)[0] for d in run_dirs]

    def stacked(tasks, test_only=False):
        xs, ys = zip(*(load(o, tasks, device, test_only) for o in orders))
        return torch.stack(xs), torch.stack(ys)

    replay = stacked(range(min(start, args.replay_tasks))) if start and args.replay_tasks else None
    old = [load(o, range(args.old_tasks), device, test_only=True) for o in orders]

    for t in range(start, args.tasks):
        seen = 2 * (t + 1)
        x_train, y_train = stacked([t])
        accs, losses, reinits = train_task(net, optimizer, swr, x_train, y_train, seen, args, replay)
        if t < args.replay_tasks:
            replay = (x_train, y_train) if replay is None else (torch.cat([replay[0], x_train], 1),
                                                                torch.cat([replay[1], y_train], 1))
        do_eval = (t + 1) % args.eval_every == 0 or t == args.tasks - 1
        for j, (order, d) in enumerate(zip(orders, run_dirs)):
            eval_net.load_state_dict(net.state_dict_of(j))
            x_test, y_test = load(order, [t], device, test_only=True)
            logs[j].writerow({"task": t, "class_a": order[2 * t], "class_b": order[2 * t + 1],
                              "train_loss": f"{losses[j]:.5f}", "train_acc": f"{accs[j][-1]:.5f}",
                              "test_acc_pair": f"{accuracy(eval_net, x_test, y_test, 2 * t, seen):.5f}",
                              "test_acc_seen": f"{accuracy(eval_net, x_test, y_test, 0, seen):.5f}",
                              **dict(zip(SPEED, epochs_to(accs[j]))), "num_reinit": reinits[j]})
            if do_eval:
                hits = n = 0
                for lo in range(0, t + 1, 25):  # All seen classes, in chunks to bound memory.
                    x, y = load(order, range(lo, min(lo + 25, t + 1)), device, test_only=True)
                    hits += accuracy(eval_net, x, y, 0, seen) * len(x)
                    n += len(x)
                n_old = min(t + 1, args.old_tasks)
                mask = old[j][1] < 2 * n_old
                ox, oy = old[j][0][mask], old[j][1][mask]
                evals[j].writerow({"after_task": t, "old_acc_seen": f"{accuracy(eval_net, ox, oy, 0, seen):.5f}",
                                   "old_acc_old": f"{accuracy(eval_net, ox, oy, 0, 2 * n_old):.5f}",
                                   "all_acc_seen": f"{hits / n:.5f}"})
                del x, y
                torch.save({"net": net.state_dict_of(j), "next_task": t + 1}, d / "latest.pt")
        if do_eval:
            torch.cuda.empty_cache()
            torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "swr": swr.state_dict() if swr else None, "next_task": t + 1,
                        "rng": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()},
                       group_ckpt.with_suffix(".tmp"))
            group_ckpt.with_suffix(".tmp").replace(group_ckpt)
        print(f"task {t + 1}/{args.tasks} train_acc {np.mean([a[-1] for a in accs]):.3f} "
              f"reinit {sum(reinits)}", flush=True)


if __name__ == "__main__":
    main()
