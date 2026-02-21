import torch
import torch.nn as nn


class SimpleSDF(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, num_layers=4):
        super().__init__()

        layers = []
        last_dim = in_dim

        for i in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden(x)
        return self.final(x)
