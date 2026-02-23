import numpy as np
import torch
import src.model.SIREN as si

class Loss:
    def __init__(self, lambda_twd:float, lambda_surface:float, lambda_eikonal:float, lambda_normal:float, model, k: int, lambda_inter: float, lambda_sign: float):
        self.lambda_eikonal = lambda_eikonal
        self.lambda_normal = lambda_normal
        self.lambda_twd = lambda_twd
        self.lambda_surface = lambda_surface
        self.lambda_inter = lambda_inter
        self.lambda_sign = lambda_sign
        self.k = k
        self.model = model
    
    #TODO: add regularization terms
    def compute_loss(self, input, pred, pred_surface, pred_inside, pred_outside, pred_off, sdf_grad, normals, surface_mask):
        loss_normal = self.lambda_normal * normal_loss(
            pred_sdf=pred,
            coords=input,
            gt_normals=normals,
            on_surface_mask=surface_mask
        )

        loss_surface = self.lambda_surface * surface_loss(sdf_surface=pred_surface)
        loss_sign = self.lambda_sign * sign_loss(sdf_inside=pred_inside, sdf_outside=pred_outside)
        loss_inter = self.lambda_inter * interior_loss(sdf_off=pred_off)
        loss_eikonal = self.lambda_eikonal * eikonal_loss(gradients=sdf_grad)

        total_loss = loss_normal + loss_surface + loss_sign + loss_inter + loss_eikonal
        return total_loss
    
def surface_loss(sdf_surface: torch.Tensor):
    return (sdf_surface ** 2).mean()    

def eikonal_loss(gradients: torch.Tensor):
    return ((gradients.norm(dim=-1) - 1) ** 2).mean()

def normal_loss(pred_sdf, coords, gt_normals, on_surface_mask):
    # Only compute for on-surface points
    if on_surface_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_sdf.device)

    surface_normals = gt_normals[on_surface_mask]

    # Compute gradient of SDF w.r.t input coordinates
    sdf_grad = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=coords,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True
    )[0]

    # slice gradients only for surface points
    gradients = sdf_grad[on_surface_mask]

    # Normalize gradients
    grad_norm = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-8)
    gt_normals = surface_normals / (surface_normals.norm(dim=-1, keepdim=True) + 1e-8)

    # Cosine similarity: 1 if aligned, -1 if opposite
    cos_sim = (grad_norm * gt_normals).sum(dim=-1, keepdim=True)  # (N,1)

    # Loss: 1 - cos(similarity)
    loss = (1 - cos_sim).mean()

    return loss

def sign_loss(sdf_inside, sdf_outside):
    return torch.relu(sdf_inside).mean() + torch.relu(-sdf_outside).mean()
    
def interior_loss(sdf_off):
    return torch.exp(-100 * torch.abs(sdf_off)).mean()
