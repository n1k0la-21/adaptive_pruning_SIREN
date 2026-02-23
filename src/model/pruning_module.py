import src.model.SIREN as si
import torch

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
            new_current, new_next = update(model=self.model, layer_idx=i, keep=keep)
            new_current = new_current.to(device)
            new_next = new_next.to(device)
        
            current.linear = new_current

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
    def __init__(self, model, threshold_percentage, k):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold_percentage
        self.k = k

    def prune(self):
        device = self.device
        total = 0
        layers = self.model.hidden
        importance, _ = self.importance()

        for i in range(len(layers)):
            current = layers[i]
            if not isinstance(current, torch.nn.Linear):
                current = current.linear

            current_importance = importance[i] # relationship between out_dim of this layer and in_dim of the next
            k = min(self.k, len(current_importance)) # safety measure
            topk_vals = torch.topk(current_importance, k=k).values
            relative_importance = k * current_importance / torch.sum(topk_vals)

            threshold = torch.quantile(relative_importance, self.threshold)
            mask = relative_importance <= threshold

            keep = torch.where(~mask)[0]

            new_current, new_next = update(model=self.model, layer_idx=i, keep=keep)
            new_current = new_current.to(device)
            new_next = new_next.to(device)
        
            current.linear = new_current

            if i < len(layers) - 1:
                layers[i + 1].linear = new_next
            else:
                self.model.final = new_next

            total += mask.sum().item()

        return total
    
    def importance(self):
        layers = self.model.hidden
        importance_list = []

        for l in range(len(layers) - 1):
            W_out = layers[l].linear.weight
            W_in = None  
            if l < len(layers) - 1:
                W_in = layers[l+1].linear.weight
            else:
                W_in = self.model.final.weight 

            # Compute squared norms
            out_dim_norm = torch.sum(W_out**2, dim=0)  # per neuron in current layer
            in_dim_norm = torch.sum(W_in**2, dim=1)    # per neuron in current layer

            # Importance per neuron
            importance = out_dim_norm + in_dim_norm
            importance_list.append(importance)

        return importance_list
    
    def reg_term(self):
        importance_list = self.importance()
        importance = torch.cat(importance_list)

        k_max = torch.max(importance)
        k_min = torch.min(importance)
        lambda_k = 2**(4 * (k_max - importance)/(k_max - k_min))

        return torch.sum(lambda_k * importance)

def update(model, layer_idx: int, keep):
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
