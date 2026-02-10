import torch
import torch.nn as nn
import math

class FirstSineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            bound = 1 / self.linear.in_features
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor):
        return torch.sin(self.omega_0 * self.linear(x)) # forward: sin(omega_0 * (W_T * x + bias))
    

    
class DeepSineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            bound = math.sqrt(6 / self.linear.in_features) / self.omega_0 # proposed by paper
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor):
        return torch.sin(self.omega_0 * self.linear(x)) # forward: sin(omega_0 * (W_T * x + bias))
    


class SIRENSDF(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 256, num_hidden_layers: int = 4, omega_0: float = 30.0):
        super().__init__()

        layers = []

        layers.append(FirstSineLayer(in_dim, hidden_dim, omega_0))

        for _ in range(num_hidden_layers):
            layers.append(DeepSineLayer(hidden_dim, hidden_dim, omega_0))

        self.hidden = nn.Sequential(*layers)

        self.final = nn.Linear(hidden_dim, 1) # final layer without sine activation

        self.init_final()

    def init_final(self):
        with torch.no_grad():
            self.final.weight.uniform_(-1e-5, 1e-5)
            self.final.bias.zero_()

    def forward(self, x):
        x = self.hidden(x)
        return self.final(x)
    
    def neuron_counts(model):
        print("Neuron counts per layer:")
        print("-" * 40)

        total_neurons = 0

        # hidden layers
        for i, layer in enumerate(model.hidden):
            if hasattr(layer, "linear"):
                n = layer.linear.out_features
                print(f"Hidden layer {i:2d}: {n:4d} neurons")
                total_neurons += n

        # final layer
        final_neurons = model.final.out_features
        print(f"Final layer    : {final_neurons:4d} neurons")