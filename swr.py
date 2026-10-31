"""Selective Weight Reinitialization, adapted from
https://github.com/JFernando4/collas_2025_swr_paper (MIT, see LICENSE.swr).

Changes from src/swr_functions/selective_weight_reinitialization.py:
- avg_utility is only allocated when decay_rate > 0 (it costs ~1 GB here).
- The fractional binomial draw uses torch's RNG, so checkpoints capture it.
"""
import torch
from torch.optim import Optimizer

MLP_WEIGHTS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


@torch.no_grad()
def compute_utility(p: torch.Tensor, utility_name: str):
    if utility_name == "gradient":
        return torch.abs(p.grad * p).flatten()
    elif utility_name == "magnitude":
        return torch.abs(p).flatten()
    elif utility_name == "random":
        return torch.rand(size=(p.numel(),), dtype=p.dtype, device=p.device)
    else:
        raise ValueError("Utility function not recognized")


@torch.no_grad()
def prune_weights(utility: torch.Tensor, pruning_method: str, reinit_factor: float) -> torch.Tensor:
    """Returns the indices of the weights to be pruned based on the utility and reinit_factor."""
    if pruning_method == "proportional":
        fraction_to_prune = utility.numel() * reinit_factor
        drop_num = int(fraction_to_prune) + int(torch.rand(1).item() < fraction_to_prune % 1)
        if drop_num == 0:
            return torch.empty(0)
        return torch.argsort(utility)[:drop_num]
    elif pruning_method == "threshold":
        prune_threshold = reinit_factor * utility.mean()
        return torch.where(utility <= prune_threshold)[0]
    else:
        raise ValueError("Pruning method not recognized.")


class SelectiveWeightReinitialization(Optimizer):
    def __init__(self, params, utility_function, pruning_method, param_means, param_stds,
                 reinit_freq, reinit_factor, decay_rate=0.0):
        defaults = dict(utility_function=utility_function, pruning_method=pruning_method,
                        means=param_means, stds=param_stds, reinit_freq=reinit_freq,
                        reinit_factor=reinit_factor, decay_rate=decay_rate)
        super().__init__(params, defaults)
        self.reinit_indicator = False
        self.num_replaced = 0
        for group in self.param_groups:
            group.setdefault("current_step", 0)
            if decay_rate > 0.0:
                for p in group["params"]:
                    self.state[p].setdefault("avg_utility", torch.zeros_like(p.flatten()))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            group["current_step"] += 1

            if group["decay_rate"] > 0.0:
                for p in group["params"]:
                    temp_utility = compute_utility(p, group["utility_function"])
                    self.state[p]["avg_utility"].mul_(group["decay_rate"]).add_(temp_utility, alpha=1.0 - group["decay_rate"])

            if group["current_step"] % group["reinit_freq"] == 0:
                for p, m, s in zip(group["params"], group["means"], group["stds"]):
                    parameter_utility = compute_utility(p, group["utility_function"]) if group["decay_rate"] == 0.0 else self.state[p]["avg_utility"]
                    reinit_indices = prune_weights(parameter_utility, group["pruning_method"], group["reinit_factor"])
                    num_reinit = reinit_indices.numel()
                    if num_reinit > 0:
                        new_values = torch.normal(mean=m, std=s, size=(num_reinit,), device=p.device, dtype=p.dtype)
                        if not p.is_contiguous():
                            raise ValueError("Parameter tensor is not contiguous, which may lead to unexpected behavior.")
                        p.view(-1)[reinit_indices] = new_values
                        if group["decay_rate"] > 0.0:
                            self.state[p]["avg_utility"].zero_()
                        self.reinit_indicator = True
                        self.num_replaced += num_reinit

    def reset_reinit_indicator(self):
        self.reinit_indicator = False
        self.num_replaced = 0


def build_swr(model, cfg):
    """SWR over Qwen's MLP weights. Qwen3 initializes Linear weights as N(0, initializer_range)
    and these layers have no bias, so the paper's init-distribution lookup reduces to one std."""
    params = [p for n, p in model.named_parameters() if n.endswith(MLP_WEIGHTS)]
    assert len(params) == 3 * model.config.num_hidden_layers, "MLP parameter names changed"
    std = {"resample": model.config.initializer_range, "mean": 0.0}[cfg["reinit_strat"]]
    return SelectiveWeightReinitialization(
        params, cfg["utility_function"], cfg["pruning_method"], [0.0] * len(params), [std] * len(params),
        cfg["reinit_freq"], cfg["reinit_factor"], cfg["decay_rate"])
