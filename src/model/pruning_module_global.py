import src.model.SIREN as si
import torch
import torch_pruning as tp

class AIRe():
    
    def __init__(self, model, threshold_percentage: float, k):
        self.model = model
        self.threshold_percentage = threshold_percentage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
    
    def prune(self):
        total = 0
        layers = self.model.hidden
        device = self.device
        normalized_norms_list = []

        # compute global threshold on normalized norms
        for i in range(1, len(layers)):
            W = layers[i].linear.weight.data
            row_norms = torch.sum(torch.abs(W), dim=1)
            # unsure about normalization because i want to avoid overpruning single layers
            normalized = row_norms / (row_norms.mean())
            normalized_norms_list.append(normalized)

        normalized_norms_tensor = torch.cat(normalized_norms_list)
        threshold = torch.quantile(normalized_norms_tensor, self.threshold_percentage)

        for i in range(1, len(layers)):
            W = layers[i].linear.weight.data
            row_norms = torch.sum(torch.abs(W), dim=1)
            normalized = row_norms / (row_norms.mean())

            mask = normalized <= threshold

            # could be that a lot of neurons are pruned which messes up the frequency composition for a SIREN
            #max_prune = int(0.7 * len(mask)) 
            #if mask.sum() > max_prune:
            #    _, top_idx = torch.topk(normalized, k=len(normalized) - max_prune, largest=True)
            #    mask = torch.ones_like(mask, dtype=torch.bool)
            #    mask[top_idx] = False

            if mask.sum() == 0:
                continue

            keep = torch.where(~mask)[0]
            new_current, new_next = update(model=self.model, layer_idx=i, keep=keep)
            if new_current is None or new_next is None:
                continue

            new_current = new_current.to(device)
            new_next = new_next.to(device)

            layers[i].linear = new_current
            if i < len(layers) - 1:
                layers[i + 1].linear = new_next
            else:
                self.model.final = new_next

            total += mask.sum().item()
        return total
    
    def reg_term(self):

        penalty = 0.0

        for module in self.model.hidden[1:].modules():

            if isinstance(module, torch.nn.Linear):
                W = module.weight
                col_norms = torch.sum(torch.abs(W), dim=0)
            
                indices = torch.topk(-col_norms, self.k).indices
                penalty += torch.sum(col_norms[indices])

        return penalty

class DepGraph():
    def __init__(self, model, example_inputs, threshold=0.5, alpha=4):
        self.model = model
        self.device = next(model.parameters()).device
        self.threshold = threshold

        self.example_inputs = example_inputs.to(self.device)

        self.DG = tp.DependencyGraph().build_dependency(
            self.model, example_inputs=self.example_inputs
        )

        self.importance = tp.importance.GroupMagnitudeImportance(p=2)

    # (paper eq. 4 & 5)
    def reg_term(self, ignored_layers=None):
        if ignored_layers is None:
            ignored_layers = []

        groups = self.DG.get_all_groups(ignored_layers=ignored_layers)

        importance_list = []

        for group in groups:
            I_gk = self.importance(group)  # shape: [num_channels]
            if I_gk is None or len(I_gk) == 0:
                continue
            importance_list.append(I_gk)

        importance = torch.cat(importance_list)

        k_max = importance.max()
        k_min = importance.min()

        # 4 is suggested by paper
        gamma = 2 ** (4 * (k_max - importance) / (k_max - k_min))

        return torch.sum(gamma * importance)

    def prune(self, ignored_layers=None):
        if ignored_layers is None:
            ignored_layers = []

        pruner = tp.pruner.MagnitudePruner(
            self.model,
            self.example_inputs,
            importance=self.importance,
            pruning_ratio=self.threshold,
            ignored_layers=ignored_layers,
        )

        before = sum(p.numel() for p in self.model.parameters())

        pruner.step()

        after = sum(p.numel() for p in self.model.parameters())

        return before - after

def update(model, layer_idx: int, keep):
            if len(keep) == 0:
                return None, None
            
            layers = model.hidden
            current = layers[layer_idx]
            
            W = current.linear.weight.data
            b = current.linear.bias.data 

            # rebuild current layer
            new_current = torch.nn.Linear(
                in_features=current.linear.in_features,
                out_features=len(keep),
                bias=True
            )

            new_current.weight.data = W[keep, :].clone()
            new_current.bias.data = b[keep].clone()

            new_next = None

            # update next layer
            if layer_idx < len(layers) - 1:

                next_layer = layers[layer_idx + 1]

        
                old_next = next_layer.linear

                new_next = torch.nn.Linear(
                    in_features=len(keep),
                    out_features=old_next.out_features,
                    bias=True
                )

                new_next.weight.data = old_next.weight.data[:, keep].clone()
                new_next.bias.data = old_next.bias.data.clone()

            # update final layer if this was last hidden layer
            else:

                final = model.final

                new_next = torch.nn.Linear(
                    in_features=len(keep),
                    out_features=final.out_features,
                    bias=True
                )

                new_next.weight.data = final.weight.data[:, keep].clone()
                new_next.bias.data = final.bias.data.clone()

            return new_current, new_next
