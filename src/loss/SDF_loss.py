import numpy as np
import torch

class Loss:
    def __init__(self, lambda_surface:float, lambda_eikonal:float, lambda_normal:float, normal_present:bool):
        self.lambda_surface = lambda_surface
        self.lambda_eikonal = lambda_eikonal
        self.lambda_normal = lambda_normal
        self.normal_present = normal_present
    
    def compute_loss(self, sdf_pred, grad_all, grad_surface, normals):
        loss = self.lambda_surface * surface_loss(sdf_pred) + self.lambda_eikonal * eikonal_loss(grad_all)
        if self.normal_present and normals is not None:
            loss += self.lambda_normal * normal_loss(gradients=grad_surface, normals=normals)

        return loss
    
def surface_loss(sdf_pred: torch.Tensor):
    return (sdf_pred ** 2).mean()
    
def eikonal_loss(gradients: torch.Tensor):
    return ((gradients.norm(dim=-1) - 1) ** 2).mean()
    
def normal_loss(gradients: torch.Tensor, normals: torch.Tensor):
    return (1 - (gradients * normals).sum(dim=-1)).mean()
    
    # optionally add loss definition for off-surface points  