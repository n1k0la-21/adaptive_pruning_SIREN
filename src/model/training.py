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
        x_surface = torch.tensor(data.sample_surface_points(no_surface, rng=rng), device=device, dtype=torch.float32)
        x_off_surface = data.sample_close_to_surface(num_points=no_off_surface//2, rng=rng)
        x_global = sample_global(no_off_surface//2, rng)
        x_global = np.concatenate([x_global, x_off_surface], axis=0)
        x_global_tensor = torch.from_numpy(x_global).float().to(device)

        # Compute signed distances
        sd_tensor = o3d.core.Tensor(x_global, dtype=o3d.core.Dtype.Float32)
        global_distances = scene.compute_signed_distance(sd_tensor).numpy()
        global_distances_tensor = torch.tensor(global_distances, device=device, dtype=torch.float32)

        # Split inside / outside
        inside_mask = global_distances_tensor < 0
        outside_mask = global_distances_tensor > 0

        true_inside = global_distances_tensor[inside_mask]
        true_outside = global_distances_tensor[outside_mask]
        x_inside = x_global_tensor[inside_mask]
        x_outside = x_global_tensor[outside_mask]

        x_all = torch.cat([x_surface, x_inside, x_outside])
        x_all.requires_grad_(True)

        sdf_pred = model(x_all)

        len_surface = len(x_surface)
        len_inside = len(x_inside)

        sdf_surface = sdf_pred[:len_surface]
        sdf_inside = sdf_pred[len_surface: len_surface+len_inside]
        sdf_outside = sdf_pred[len_surface+len_inside:]

        # Far-off mask per tensor to extract from return value
        far_inside_mask = torch.abs(true_inside) > 0.05
        far_outside_mask = torch.abs(true_outside) > 0.05
        sdf_off = torch.cat([sdf_inside[far_inside_mask], sdf_outside[far_outside_mask]], dim=0)

        
        sdf_grad = torch.autograd.grad(
            outputs=sdf_pred,
            inputs=x_all,
            grad_outputs=torch.ones_like(sdf_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        surface_normals = torch.from_numpy(data.sample_surface_normals(x_surface.detach().cpu().numpy())).float().to(device)
        dummy_normals = torch.zeros_like(torch.cat([x_inside, x_outside], dim=0))
        normals = torch.cat([surface_normals, dummy_normals], dim=0)

        on_surface_mask = torch.zeros(len(x_all), dtype=torch.bool, device=device)
        on_surface_mask[:len_surface] = True

        current_loss = loss.compute_loss(
            input=x_all, pred=sdf_pred,
            pred_surface=sdf_surface, pred_inside=sdf_inside,
            pred_outside=sdf_outside, pred_off=sdf_off,
            normals=normals, surface_mask=on_surface_mask,
            sdf_grad=sdf_grad,
            true_inside=true_inside,
            true_outside=true_outside
        )

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            if(densification == True and step == 200):
                added_frequencies = densify(model=model)
                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                print(f"Added {len(added_frequencies)} frequencies to the embedding layer.")

            if(pruning_module != None and step == 400):
                pruned_neurons = pruning_module.prune()
                loss.prune = False
                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                print(f"Pruned {pruned_neurons} neurons.")
                
            print(f"Step {step} | Loss {current_loss.item()}")
            


        