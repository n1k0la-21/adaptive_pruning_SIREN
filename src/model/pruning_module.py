import src.model.SIREN as si
import math
import torch

class PruningModule:
    
    def __init__(self, model: si.SIRENSDF, threshold: float):
        self.model = model
        self.threshold = threshold
        
    def prune(self):
        sum = 0.0
        for module in self.model.hidden[1:].modules():

            if isinstance(module, torch.nn.Linear):
                W = module.weight
                col_norms = torch.sum(torch.abs(W), dim=0)

                mask = col_norms <= self.threshold
                W.data[:, mask] = 0.0
                sum += torch.sum(mask).item()
            
        return sum