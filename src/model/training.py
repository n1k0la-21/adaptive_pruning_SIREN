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

def train(epochs: int, data: MeshDataset, no_surface: int, no_off_surface:int, model, loss: loss_module.Loss, optimizer: torch.optim.Adam, scene: o3d.t.geometry.RaycastingScene, pruning_module=None, densification=False):
    rng = np.random.default_rng(seed=42)
    o3d.utility.random.seed(42)

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

        on_surface_mask = torch.zeros(len(x_all), dtype=torch.bool, device=x_all.device)
        on_surface_mask[:len(x_surface)] = True  # first batch points are surface

        x_surface_np = x_surface.detach().cpu().numpy()
        x_surface_normals = torch.from_numpy(data.sample_surface_normals(x_surface_np)).float().to(device)

        # Dummy normals for off-surface points
        x_outside_normals = torch.zeros_like(x_outside)
        x_inside_normals = torch.zeros_like(x_inside)

        normals = torch.cat([x_surface_normals, x_outside_normals, x_inside_normals], dim=0)

        current_loss = loss.compute_loss(input=x_all, pred=sdf_pred, pred_surface=sdf_surface, pred_inside=sdf_inside, 
                                         pred_outside=sdf_outside, pred_off=sdf_off, normals=normals, surface_mask=on_surface_mask, sdf_grad=sdf_grad)

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            if(densification == True and step == 200):
                added_frequencies = densify(model=model)
                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                print(f"Added {len(added_frequencies)} frequencies to the embedding layer.")

            if(pruning_module != None and step == 250):
                pruned_neurons = pruning_module.prune()
                loss.prune = False
                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                print(f"Pruned {pruned_neurons} neurons.")
                
            print(f"Step {step} | Loss {current_loss.item()}")
            


        