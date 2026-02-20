import src.model.SIREN as si
import src.loss.SDF_loss as loss_module
import torch
import numpy as np
import src.model.pruning_module as pm
from src.model.densification_module import densify
import open3d as o3d



def sample_surface(mesh: o3d.t.geometry.TriangleMesh, num: int):    
    pcd = mesh.sample_points_uniformly(number_of_points=num)
    
    surface_points = np.asarray(pcd.points)
    
    return surface_points

def sample_off_surface(surface_points: np.ndarray, epsilon: float, rng: np.random.Generator):
    # Random perturbation in each axis between -epsilon and +epsilon
    offsets = rng.uniform(-epsilon, epsilon, size=surface_points.shape)
    
    off_surface_points = surface_points + offsets
    
    return off_surface_points

def sample_global(num, rng):
    return rng.uniform(-1, 1, size=(num, 3))

# TODO: try to make loss such that model actually learns zero crossing (from outside to inside)
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
        sdf_grad = sdf_grad[len(x_surface):]

        # true
        sdf_true = scene.compute_signed_distance(o3d.core.Tensor(x_all[len(x_surface):].cpu().detach().numpy(), dtype=o3d.core.Dtype.Float32))
        sdf_true = torch.from_numpy(sdf_true.numpy()).float().to(device)

        true_inside = sdf_true[:len(x_inside)]
        true_inside = torch.clamp(true_inside, -0.5, 0.5)
        true_outside = sdf_true[len(x_inside):]
        true_outside = torch.clamp(true_outside, -0.5, 0.5)

        true_off = torch.cat([true_inside, true_outside], dim=0)



        # compute loss
        loss_surface = ((sdf_surface)**2).mean()
        loss_inside = ((sdf_inside - true_inside)**2).mean()
        loss_outside = ((sdf_outside - true_outside)**2).mean()
        loss_off = ((sdf_off - true_off)**2).mean()

        # eikonal
        loss_eikonal = ((sdf_grad.norm(dim=-1) - 1) ** 2).mean()

        current_loss = 5.0 * loss_surface + loss_inside + loss_outside + 0.1 * loss_eikonal

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
            


        