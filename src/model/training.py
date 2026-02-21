import src.model.SIREN as si
import src.loss.SDF_loss as loss_module
import torch
import numpy as np
import src.model.pruning_module as pm
from src.model.densification_module import densify
import open3d as o3d
from src.data.dataset import MeshDataset 

def sample_global(num, rng):
    return rng.uniform(-1, 1, size=(num, 3))

def normal_constraint(pred_sdf, coords, gt_normals, on_surface_mask):
    # Only compute for on-surface points
    if on_surface_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_sdf.device)

    surface_coords = coords[on_surface_mask]
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

# TODO: try to make loss such that model actually learns zero crossing (from outside to inside)
def train(epochs: int, data: MeshDataset, no_surface: int, no_off_surface:int, model, loss: loss_module.Loss, optimizer: torch.optim.Adam, scene: o3d.t.geometry.RaycastingScene, prune=False):
    rng = np.random.default_rng(seed=42)
    pruning_module = None
    
    if prune == True:
        pruning_module = pm.Pruning_module(model=model, threshold_percentage=0.2)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for step in range(epochs):
        rng = np.random.default_rng(seed=42 + step)
        x_surface = data.sample_surface_points(no_surface, rng=rng) # not reproducible right now because o3d sampler does not have a fixed seed
        x_off_surface = data.sample_close_to_surface(num_points=int(no_off_surface/2), rng=rng)
        x_global = sample_global(int(no_off_surface/2), rng)
        x_global = np.concatenate((x_global, x_off_surface), axis=0)

        # signed distances for filtering
        tensor = o3d.core.Tensor(x_global , dtype=o3d.core.Dtype.Float32)
        global_distances = scene.compute_signed_distance(tensor).numpy()

        # filter points that lie outside/inside
        x_inside = global_distances < 0
        x_outside = 0 < global_distances

        # torch tensors as input
        x_surface = torch.from_numpy(x_surface).float().to(device)
        x_inside = torch.from_numpy(x_global[x_inside]).float().to(device)
        x_outside = torch.from_numpy(x_global[x_outside]).float().to(device)

        x_all = torch.cat([x_surface, x_inside, x_outside], dim=0)
        x_all.requires_grad_(True)

        # pred
        sdf_pred = model(x_all)
        sdf_surface = sdf_pred[:len(x_surface)]
        sdf_inside = sdf_pred[len(x_surface): len(x_surface) + len(x_inside)]
        sdf_outside = sdf_pred[len(x_surface) + len(x_inside):]
        sdf_off = torch.cat([sdf_inside, sdf_outside], dim=0)
        sdf_grad = torch.autograd.grad(
                    outputs=sdf_pred,
                    inputs=x_all,
                    grad_outputs=torch.ones_like(sdf_pred),
                    create_graph=True,
                    retain_graph=True
                )[0]

        # true
        sdf_true = scene.compute_signed_distance(o3d.core.Tensor(x_all[len(x_surface):].cpu().detach().numpy(), dtype=o3d.core.Dtype.Float32))
        sdf_true = torch.from_numpy(sdf_true.numpy()).float().to(device)

        on_surface_mask = torch.zeros(len(x_all), dtype=torch.bool, device=x_all.device)
        on_surface_mask[:len(x_surface)] = True  # first batch points are surface

        x_surface_np = x_surface.detach().cpu().numpy()
        x_surface_normals = torch.from_numpy(data.sample_surface_normals(x_surface_np)).float().to(device)

        # Dummy normals for off-surface points
        x_outside_normals = torch.zeros_like(x_outside)
        x_inside_normals = torch.zeros_like(x_inside)

        loss_normal = normal_constraint(
            pred_sdf=sdf_pred,
            coords=x_all,
            gt_normals=torch.cat([x_surface_normals, x_outside_normals, x_inside_normals], dim=0),
            on_surface_mask=on_surface_mask
        )

        # compute loss
        loss_surface = ((sdf_surface)**2).mean()

        loss_sign = (
            torch.relu(sdf_inside).mean() +      # inside should be negative
            torch.relu(-sdf_outside).mean()      # outside should be positive
        )

        # Inter constraint (push away from zero off-surface)
        loss_inter = torch.exp(-100 * torch.abs(sdf_off)).mean()

        # eikonal
        loss_eikonal = ((sdf_grad.norm(dim=-1) - 1) ** 2).mean()

        current_loss = 150 * loss_surface + 1.5 * loss_sign + 0.5 * loss_inter + 0.5 * loss_eikonal + 1.5 * loss_normal

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            if(prune == True and step == 100):
                added_frequencies = densify(model=model)
                pruned_neurons = pruning_module.prune()
                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                print(f"Pruned {pruned_neurons} neurons.")
                print(f"Added {len(added_frequencies)} frequencies to the embedding layer.")
            print(f"Step {step} | Loss {current_loss.item()}")
            


        