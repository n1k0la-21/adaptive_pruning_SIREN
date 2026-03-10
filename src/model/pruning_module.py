import src.model.SIREN as si
import torch
import torch_pruning as tp

class AIRe():

    def __init__(self, model, pruning_ratio):
        self.model = model
        self.ratio = pruning_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prune(self):

        layers = self.model.hidden
        total_pruned = 0

        # skip first layer
        for i in range(1, len(layers)):

            layer = layers[i]
            W = layer.linear.weight.data

            row_norms = torch.sum(torch.abs(W), dim=1)

            layer_size = row_norms.shape[0]
            k_layer = int(layer_size * self.ratio)

            if k_layer == 0:
                continue

            prune_idx = torch.topk(
                row_norms, k_layer, largest=False
            ).indices

            keep = torch.tensor(
                [j for j in range(layer_size) if j not in prune_idx],
                device=self.device
            )

            if len(keep) == 0:
                continue

            new_current, new_next = update(self.model, i, keep)

            if new_current is None or new_next is None:
                continue

            new_current = new_current.to(self.device)
            new_next = new_next.to(self.device)

            layers[i].linear = new_current

            if i < len(layers) - 1:
                layers[i + 1].linear = new_next
            else:
                self.model.final = new_next

            total_pruned += k_layer

        return total_pruned


    def reg_term(self):

        layers = self.model.hidden
        penalty = 0.0

        for i in range(1, len(layers)):

            W = layers[i].linear.weight
            row_norms = torch.sum(torch.abs(W), dim=1)

            layer_size = row_norms.shape[0]
            k_layer = int(layer_size * self.ratio)

            if k_layer == 0:
                continue

            weakest = torch.topk(
                row_norms, k_layer, largest=False
            ).values

            penalty += torch.sum(weakest)

        return penalty

class DepGraph():
    def __init__(self, model, threshold=0.5, alpha=4):
        self.model = model
        self.device = torch.device("cuda")
        self.threshold = threshold

        self.example_inputs = torch.randn(1,3).to(self.device)

        self.DG = tp.DependencyGraph().build_dependency(
            self.model, example_inputs=self.example_inputs
        )

        self.importance = tp.importance.GroupMagnitudeImportance(p=2)

    # (paper eq. 4 & 5)
    def reg_term(self, ignored_layers=None):
        if ignored_layers is None:
            ignored_layers = []

        groups = self.DG.get_all_groups(ignored_layers=[self.model.hidden[0].linear])

        importance_list = []

        for group in groups:
            I_gk = self.importance(group)  
            if I_gk is None or len(I_gk) == 0:
                continue
            importance_list.append(I_gk)

        importance = torch.cat(importance_list)

        k_max = importance.max()
        k_min = importance.min()

        # 4 is suggested by paper
        gamma = 2 ** (4 * (k_max - importance) / (k_max - k_min))

        return torch.sum(gamma * importance)

    def prune(self):
        pruner = tp.pruner.MagnitudePruner(
            self.model,
            self.example_inputs,
            importance=self.importance,
            pruning_ratio=self.threshold,
            ignored_layers=[self.model.hidden[0].linear],
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
