import numpy as np
import torch
import src.model.SIREN as si

class Loss:
    def __init__(self, lambda_twd:float, lambda_surface:float, lambda_eikonal:float, lambda_normal:float, model, lambda_inter: float, lambda_off: float, pruning_module=None):
        self.lambda_eikonal = lambda_eikonal
        self.lambda_normal = lambda_normal
        self.lambda_twd = lambda_twd
        self.lambda_surface = lambda_surface
        self.lambda_inter = lambda_inter
        self.lambda_off = lambda_off
        self.pruning_module = pruning_module
        self.model = model
        self.prune = False # can be switched through training
    
    def compute_loss(self, input, pred, pred_surface, pred_inside, pred_outside, pred_off, sdf_grad, normals, surface_mask, true_inside, true_outside):
        loss_normal = self.lambda_normal * normal_loss(
            pred_sdf=pred,
            coords=input,
            gt_normals=normals,
            on_surface_mask=surface_mask
        )

        loss_surface = self.lambda_surface * surface_loss(sdf_surface=pred_surface)
        loss_off = self.lambda_off * off_surface_loss(sdf_inside=pred_inside, sdf_outside=pred_outside, true_inside=true_inside, true_outside=true_outside)
        loss_inter = self.lambda_inter * interior_loss(sdf_off=pred_off)
        loss_eikonal = self.lambda_eikonal * eikonal_loss(gradients=sdf_grad)

        total_loss = loss_normal + loss_surface + loss_off + loss_inter + loss_eikonal

        if self.pruning_module != None and self.prune == True:
            total_loss += self.lambda_twd * self.pruning_module.reg_term()
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

    gradients = sdf_grad[on_surface_mask]

    grad_norm = gradients / (gradients.norm(dim=-1, keepdim=True))
    gt_normals = surface_normals / (surface_normals.norm(dim=-1, keepdim=True))

    # Cosine similarity: 1 if aligned, -1 if opposite
    cos_sim = (grad_norm * gt_normals).sum(dim=-1, keepdim=True)  # (N,1)

    # Loss: 1 - cos(similarity)
    loss = (1 - cos_sim).mean()

    return loss

def off_surface_loss(sdf_inside, sdf_outside, true_inside, true_outside):
    inside = torch.relu(sdf_inside).mean() *2
    outside = torch.relu(-sdf_outside).mean() *2
    return inside + outside
    
def interior_loss(sdf_off):
    # this pushes off surface sdfs away from zero and prevents floating surfaces
    return torch.exp(-100 * torch.abs(sdf_off)).mean()
