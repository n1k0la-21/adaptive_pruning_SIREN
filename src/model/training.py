import src.model.SIREN as si
import src.loss.SDF_loss as loss_module
import torch
import numpy as np
import src.model.pruning_module as pm
from src.model.densification_module import densify
import open3d as o3d
from src.data.dataset import MeshDataset 
import matplotlib.pyplot as plt

def train(epochs: int, data: MeshDataset, no_surface: int, no_off_surface:int, model, loss: loss_module.Loss, optimizer: torch.optim.Adam, scene: o3d.t.geometry.RaycastingScene, pruning_module=None, densification=False):
    o3d.utility.random.seed(42)
    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # global pool
    x_surface_global = data.sample_surface_points(1000000, rng)
    x_far = data.sample_global(500000, rng)
    x_off_surface_global = data.sample_close_to_surface(500000, rng)

    # corresponding sdf
    sdf = o3d.core.Tensor(np.concatenate([x_far, x_off_surface_global], axis=0), dtype=o3d.core.Dtype.Float32)
    sdf = scene.compute_signed_distance(sdf).numpy()

    # ground truth grid mask for iou
    with torch.no_grad():
        x = torch.linspace(-1, 1, 64)
        y = torch.linspace(-1, 1, 64)
        z = torch.linspace(-1, 1, 64)

        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1,3)
        grid_o3d = o3d.core.Tensor(grid.numpy(), dtype=o3d.core.Dtype.Float32)
        grid_sdf = scene.compute_signed_distance(grid_o3d).numpy()
        grid_mask = torch.tensor(grid_sdf < 0, dtype=torch.bool)

    loss_history = []       # every 10 steps
    iou_history = []        # every 100 steps
    iou_steps = []

    for step in range(epochs):

        if(densification == True and (step == 100 or step == 750)):
                added_frequencies = densify(model=model, optimizer=optimizer)
                #print(model.hidden[0].omega_scale)
                print(f"Added {len(added_frequencies)} frequencies to the embedding layer.")

        x_surface = torch.tensor(rng.choice(x_surface_global, no_surface), device=device, dtype=torch.float32)
        idx_off = rng.choice(len(x_off_surface_global), no_off_surface//2)
        x_off_surface = x_off_surface_global[idx_off]
        idx_global = rng.choice(len(x_far), no_off_surface//2)
        x_global = x_far[idx_global]
        x_global = np.concatenate([x_global, x_off_surface], axis=0)
        x_global_tensor = torch.from_numpy(x_global).float().to(device)

        # pooling from sdf (idx offset for x_off_surface)
        idx_off += len(x_far)
        global_distances = np.concatenate([sdf[idx_global], sdf[idx_off]], axis=0)
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
        if(loss.prune == True and isinstance(pruning_module, pm.DepGraph)):
            pruning_module.regularize()
        current_loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            loss_history.append(current_loss.item())
            
            if(pruning_module != None and step == 200):
                loss.prune = True
                print(f"TWD is now applied")

            if(pruning_module != None and step == 700):
                pruned_neurons = pruning_module.prune()
                loss.prune = False
                optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
                print(f"Pruned {pruned_neurons} neurons.")

            if step % 100 == 0:
                current_iou = iou(model, grid_mask, grid)
                iou_history.append(current_iou)
                iou_steps.append(step)
                msg = f"Step {step} | IoU {current_iou:.4f} | Loss {current_loss.item():.4f}"
            else:    
                msg = f"Step {step} | Loss {current_loss:.4f}"
            print(msg)

    current_iou = iou(model, grid_mask, grid)
    msg = f"Step {step} | IoU {current_iou:.4f} | Loss {current_loss.item():.4f}"
    print(msg)
    plot_training(loss_history, iou_history, iou_steps)
    return loss_history, iou_history, iou_steps

def plot_training(loss_history, iou_history, iou_steps):
    loss_steps = [i * 10 for i in range(len(loss_history))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(loss_steps, loss_history, color='steelblue', linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(iou_steps, iou_history, color='darkorange', linewidth=1.5, marker='o', markersize=4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("IoU")
    ax2.set_title("IoU (64^3 grid)")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved training_curves.png")
        
def chamfer_hausdorff(path1, path2):
    mesh1 = o3d.io.read_triangle_mesh(path1)
    mesh2 = o3d.io.read_triangle_mesh(path2)

    print(f"mesh1 triangles: {len(mesh1.triangles)}, vertices: {len(mesh1.vertices)}")
    print(f"mesh2 triangles: {len(mesh2.triangles)}, vertices: {len(mesh2.vertices)}")

    pcd1 = mesh1.sample_points_uniformly(number_of_points=100000)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=100000)

    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))

    chamfer = (np.mean(d1) + np.mean(d2)) / 2
    hausdorff = max(np.max(d1), np.max(d2))

    return chamfer, hausdorff

def iou(model, grid_mask, grid, batch_size=65536):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(grid), batch_size):
            batch = grid[i:i+batch_size].to(next(model.parameters()).device)
            sdf_pred = model(batch)
            pred_mask = sdf_pred < 0
            all_preds.append(pred_mask.cpu())
    all_preds = torch.cat(all_preds, dim=0).squeeze()
    
    intersection = (all_preds & grid_mask).sum().float()
    union = (all_preds | grid_mask).sum().float()
    return (intersection / union).item()
        