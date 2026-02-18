import src.model.SIREN as si
import src.loss.SDF_loss as loss_module
import torch
import numpy as np
import src.model.pruning_module as pm
from src.model.densification_module import densify
import open3d as o3d



def sample_surface(mesh: o3d.t.geometry.TriangleMesh, num: int):
    legacy_mesh = mesh.to_legacy()
    
    pcd = legacy_mesh.sample_points_uniformly(number_of_points=num)
    
    surface_points = np.asarray(pcd.points)
    
    return surface_points

def sample_off_surface(surface_points: np.ndarray, epsilon: float, rng: np.random.Generator):
    # Random perturbation in each axis between -epsilon and +epsilon
    offsets = rng.uniform(-epsilon, epsilon, size=surface_points.shape)
    
    off_surface_points = surface_points + offsets
    
    return off_surface_points

def sample_global(num, rng):
    return rng.uniform(-1, 1, size=(num, 3))

def train(epochs: int, data: np.array, no_surface: int, no_off_surface:int, model, loss: loss_module.Loss, optimizer: torch.optim.Adam, scene: o3d.t.geometry.RaycastingScene, mesh: o3d.t.geometry.TriangleMesh, prune=False):
    rng = np.random.default_rng(seed=42)
    epsilon = 0.05
    pruning_module = None
    
    if prune == True:
        pruning_module = pm.Pruning_module(model=model, threshold_percentage=0.2)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for step in range(epochs):
        x_surface = sample_surface(mesh=mesh, num=no_surface)
        x_off_surface = sample_off_surface(surface_points=x_surface, epsilon=epsilon, rng=rng)
        x_global = sample_global(no_off_surface, rng)

        # create torch tensors
        x_surface = torch.from_numpy(x_surface).float().to(device)
        x_off_surface = torch.from_numpy(x_off_surface).float().to(device)
        x_global = torch.from_numpy(x_global).float().to(device)

        # concatenate all points
        x_all = torch.cat([x_surface, x_off_surface, x_global], dim=0)
        x_all.requires_grad_(True)

        # compute sdf
        sdf_all = model.forward(x_all)
        # extract sdf from points on surface, near and far away from it
        sdf_surface = sdf_all[:no_surface]
        sdf_near = sdf_all[no_surface: 2*no_surface]
        sdf_global = sdf_all[2*no_surface:]

        # compute ground truth sdf
        sdf_true = scene.compute_signed_distance(o3d.core.Tensor(x_all.cpu().detach().numpy(), dtype=o3d.core.Dtype.Float32))
        sdf_true = torch.from_numpy(sdf_true.numpy()).float().to(device)
        
        # precaution
        sdf_true[:no_surface] = torch.zeros(no_surface)
        true_surface = sdf_true[:no_surface]
        true_near = sdf_true[no_surface: 2*no_surface]
        true_global = sdf_true[2*no_surface:]

        # compute loss
        loss_surface = ((sdf_surface - true_surface)**2).mean()
        loss_near = ((sdf_near - true_near)**2).mean()
        loss_global = ((sdf_global - true_global)**2).mean()

        current_loss = 20.0 * loss_surface + 2.0 * loss_near + 0.5 * loss_global

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
            


        