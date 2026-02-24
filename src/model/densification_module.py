import src.model.SIREN as si
import torch

def densify(model: si.SIRENSDF):
    first = model.hidden[0]
    next = model.hidden[1]
    device = first.linear.weight.device

    with torch.no_grad():

        W = next.linear.weight.data

        col_norms = torch.sum(torch.abs(W), dim=0) # columns are associated with one frequency in the embedding layer (so we are interested in the input dimension)

        threshold = torch.quantile(col_norms, 0.8)  # top 20%
        important = col_norms >= threshold

        # expand embedding layer with double the frequency of important outputs

        omegas = first.omega_scale
        frequencies = omegas[important] * 2

        new_freq = torch.cat([frequencies, omegas])

        # rebuild weight matrix
        new_linear = torch.nn.Linear(
            in_features=first.linear.in_features,
            out_features=len(new_freq),
            bias=True
        ).to(device)

        first.omega_scale = torch.nn.Parameter(new_freq)

        # initialized using the vanilla SIREN procedure
        bound = 1 / new_linear.in_features
        new_linear.weight.uniform_(-bound, bound)
        new_linear.bias.uniform_(-bound, bound)

        new_linear.weight.data[len(frequencies):, :] = first.linear.weight.data.clone()
        new_linear.bias.data[len(frequencies):] = first.linear.bias.data.clone()

        first.linear = new_linear

        # rebuild next layers weight matrix
        bound = 1e-4

        new_next_lin = torch.nn.Linear(
            in_features=first.linear.out_features,
            out_features=next.linear.out_features,
            bias=True
        ).to(device)

        new_next_lin.weight.data.uniform_(-bound, bound)
        new_next_lin.bias.data.uniform_(-bound, bound)

        new_next_lin.weight.data[:, len(frequencies):] = next.linear.weight.data.clone()
        new_next_lin.bias.data[:] = next.linear.bias.data.clone()

        next.linear = new_next_lin

        return frequencies