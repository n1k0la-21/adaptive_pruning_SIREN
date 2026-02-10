import src.model.SIREN as si
import math
import torch

class PruningModule:
    
    def __init__(self, model: si.SIRENSDF, threshold_percentage: float):
        self.model = model
        self.threshold_percentage = threshold_percentage
    
    # TODO: implement real pruning by removing the neurons and adjusting the weights of the next layer accordingly 
    def prune(self):
        sum = 0
        for module in self.model.hidden[1:].modules():

            if isinstance(module, torch.nn.Linear):
                W = module.weight
                col_norms = torch.sum(torch.abs(W), dim=0)

                threshold = torch.quantile(col_norms, self.threshold_percentage)
                mask = col_norms <= threshold
                W.data[:, mask] = 0
                sum += torch.sum(mask).item()
            
        return sum