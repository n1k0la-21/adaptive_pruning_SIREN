import src.model.SIREN as si
import torch
import math

def densify(model: si.SIRENSDF, optimizer: torch.optim.Optimizer):
    first = model.hidden[0]
    next_layer = model.hidden[1]
    device = first.linear.weight.device

    with torch.no_grad():
        W = next_layer.linear.weight.data
        col_norms = torch.sum(torch.abs(W), dim=0)
        threshold = torch.quantile(col_norms, 0.7)
        important = col_norms >= threshold

        omegas = first.omega_scale
        frequencies = omegas[important] * 2
        new_freq = torch.cat([frequencies, omegas])
        added = len(frequencies)

        # old params (needed for optimizer state lookup)
        old_first_weight = first.linear.weight
        old_first_bias = first.linear.bias
        old_next_weight = next_layer.linear.weight
        old_next_bias = next_layer.linear.bias

        new_linear = torch.nn.Linear(
            in_features=first.linear.in_features,
            out_features=len(new_freq),
            bias=True
        ).to(device)
        bound = 1 / new_linear.in_features
        new_linear.weight.uniform_(-bound, bound)
        new_linear.bias.uniform_(-bound, bound)
        new_linear.weight.data[added:, :] = first.linear.weight.data.clone()
        new_linear.bias.data[added:] = first.linear.bias.data.clone()
        first.omega_scale = new_freq
        first.linear = new_linear

        new_next_lin = torch.nn.Linear(
            in_features=first.linear.out_features,
            out_features=next_layer.linear.out_features,
            bias=True
        ).to(device)
        bound = math.sqrt(6 / new_next_lin.in_features) / next_layer.omega
        new_next_lin.weight.data.uniform_(-bound, bound)
        new_next_lin.bias.data.uniform_(-bound, bound)
        new_next_lin.weight.data[:, added:] = next_layer.linear.weight.data.clone()
        new_next_lin.bias.data[:] = next_layer.linear.bias.data.clone()
        next_layer.linear = new_next_lin

        def transfer_state(old_param, new_param, dim):
            if old_param not in optimizer.state:
                return
            old_state = optimizer.state.pop(old_param)
            new_state = {}
            for k, v in old_state.items():
                if torch.is_tensor(v) and v.dim() > 0:
                    new_tensor = torch.zeros_like(new_param.data)
                    if dim == 0:
                        new_tensor[added:] = v
                    elif dim == 1:
                        new_tensor[:, added:] = v
                    elif dim == -1: 
                        new_tensor[:] = v
                    new_state[k] = new_tensor
                else:
                    new_state[k] = v
            optimizer.state[new_param] = new_state

        transfer_state(old_first_weight, first.linear.weight, dim=0)
        transfer_state(old_first_bias, first.linear.bias, dim=0)
        transfer_state(old_next_weight, next_layer.linear.weight, dim=1)
        transfer_state(old_next_bias, next_layer.linear.bias, dim=-1)

        # Update optimizer
        new_param_map = {
            id(old_first_weight): first.linear.weight,
            id(old_first_bias): first.linear.bias,
            id(old_next_weight): next_layer.linear.weight,
            id(old_next_bias): next_layer.linear.bias,
        }
        for group in optimizer.param_groups:
            group["params"] = [new_param_map.get(id(p), p) for p in group["params"]]

    return frequencies