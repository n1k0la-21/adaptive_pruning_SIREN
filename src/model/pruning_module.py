import src.model.SIREN as si
import torch

class Pruning_module:
    
    def __init__(self, model: si.SIRENSDF, threshold_percentage: float):
        self.model = model
        self.threshold_percentage = threshold_percentage
    
    def prune(self):

        total = 0
        layers = self.model.hidden
        device = self.model.hidden[0].linear.weight.device

        for i in range(len(layers)):

            current = layers[i]

            if not isinstance(current, si.DeepSineLayer):
                continue

            W = current.linear.weight.data
            b = current.linear.bias.data

            row_norms = torch.sum(torch.abs(W), dim=1)

            threshold = torch.quantile(row_norms, self.threshold_percentage)

            mask = row_norms <= threshold

            if mask.sum() == 0:
                continue

            keep = torch.where(~mask)[0]

        # rebuild current layer
            new_current = torch.nn.Linear(
                in_features=current.linear.in_features,
                out_features=len(keep),
                bias=True
            ).to(device)

            new_current.weight.data = W[keep, :].clone()
            new_current.bias.data = b[keep].clone()

            current.linear = new_current

        # update next layer
            if i < len(layers) - 1:

                next_layer = layers[i+1]

        
                old_next = next_layer.linear

                new_next = torch.nn.Linear(
                    in_features=len(keep),
                    out_features=old_next.out_features,
                    bias=True
                ).to(device)

                new_next.weight.data = old_next.weight.data[:, keep].clone()
                new_next.bias.data = old_next.bias.data.clone()

                next_layer.linear = new_next

        # update final layer if this was last hidden layer
            else:

                final = self.model.final

                new_final = torch.nn.Linear(
                    in_features=len(keep),
                    out_features=final.out_features,
                    bias=True
                ).to(device)

                new_final.weight.data = final.weight.data[:, keep].clone()
                new_final.bias.data = final.bias.data.clone()

                self.model.final = new_final

            total += mask.sum().item()

        return total
