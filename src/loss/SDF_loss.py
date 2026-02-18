import numpy as np
import torch
import src.model.SIREN as si

class Loss:
    def __init__(self, lambda_twd:float, lambda_sdf:float, lambda_surface:float, lambda_eikonal:float, lambda_normal:float, model, k: int):
        self.lambda_sdf = lambda_sdf
        self.lambda_eikonal = lambda_eikonal
        self.lambda_normal = lambda_normal
        self.lambda_twd = lambda_twd
        self.lambda_surface = lambda_surface
        self.k = k
        self.model = model
    
    def compute_loss(self, sdf_surface, sdf_true, sdf_pred, grad_all=None, grad_surface=None, normals=None):
        loss = self.lambda_sdf * sdf_loss(sdf_true, sdf_pred) + self.lambda_surface * surface_loss(sdf_surface)# + self.lambda_eikonal * eikonal_loss(grad_all) # + self.lambda_twd * targeted_weight_decay(self.model, self.k)

        return loss
    
def sdf_loss(sdf_true: torch.Tensor, sdf_pred: torch.Tensor):
    return ((sdf_true - sdf_pred) ** 2).mean()
    
def surface_loss(sdf_surface: torch.Tensor):
    return (sdf_surface ** 2).mean()    

def eikonal_loss(gradients: torch.Tensor):
    return ((gradients.norm(dim=-1) - 1) ** 2).mean()
    
def normal_loss(gradients: torch.Tensor, normals: torch.Tensor):
    return (1 - (gradients * normals).sum(dim=-1)).mean()
    
def targeted_weight_decay(model: si.SIRENSDF, k: int):

    penalty = 0.0

    for module in model.hidden[1:].modules():

        if isinstance(module, torch.nn.Linear):
            W = module.weight
            col_norms = torch.sum(torch.abs(W), dim=0)
            
            indices = torch.topk(-col_norms, k).indices
            penalty += torch.sum(col_norms[indices])

    return penalty