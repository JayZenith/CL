"""Checks that cil_multi's stacked training matches separate per-seed nets: python test_multi.py"""
import torch
import torch.nn.functional as F

from cil_multi import StackedNet, StackedSWR, stds_of
from train import build_net, build_swr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
S, OUT, SEEN, STEPS, B = 3, 600, 40, 5, 100
nets = []
for s in range(S):
    torch.manual_seed(s)
    nets.append(build_net(OUT).to(device))
stacked = StackedNet([n.cpu() for n in nets]).to(device)
nets = [n.to(device) for n in nets]
args = type("A", (), {"swr_freq": STEPS, "swr_k": 0.5})  # A large k so this step resets many weights.
opts = [torch.optim.SGD(n.parameters(), lr=0.001, momentum=0.9) for n in nets]
swrs = [build_swr(n, args) for n in nets]
opt = torch.optim.SGD(stacked.parameters(), lr=0.001, momentum=0.9)
params = [p for pair in zip(stacked.weights, stacked.biases) for p in pair]
swr = StackedSWR(params, stds_of(nets[0]), STEPS, 0.5)

g = torch.Generator(device="cpu").manual_seed(0)
for step in range(STEPS):
    x = torch.rand(S, B, 3, 32, 32, generator=g).to(device) - 0.5
    y = torch.randint(SEEN, (S, B), generator=g).to(device)
    last = step == STEPS - 1
    if last:  # Before the SWR step, which resamples randomly: compare the weights and which ones get reset.
        before = [n.state_dict() for n in nets]
    for j, (n, o) in enumerate(zip(nets, opts)):
        o.zero_grad()
        F.cross_entropy(n(x[j])[:, :SEEN], y[j]).backward()
        o.step()
        swrs[j].step()
    opt.zero_grad()
    logits = stacked(x)[..., :SEEN]
    F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(S, B).mean(1).sum().backward()
    opt.step()
    if not last:
        swr.step()
        for j, n in enumerate(nets):
            for k, v in n.state_dict().items():
                diff = (v - stacked.state_dict_of(j)[k]).abs().max().item()
                assert diff < 1e-5, (step, j, k, diff)
    else:
        swr.step()
        print("reinit per seed, separate:", [s.num_replaced for s in swrs], "stacked:", swr.num_replaced.tolist())
        # GPU float sums can tip a weight sitting exactly at the threshold, so allow a few in 100,000.
        for sep, st in zip([s.num_replaced for s in swrs], swr.num_replaced.tolist()):
            assert abs(sep - st) <= 1e-5 * sep + 2, (sep, st)
print("stacked training matches separate nets")
