import torch
import numpy as np
import src.model.SIREN as si
import open3d as o3d

def chamfer_hausdorff(mesh, model: si.SIRENSDF, path, gt_points):
    o3d.utility.random.seed(42)

    # mesh -> inr
    inr_points = project(gt_points, model)

    d1 = np.linalg.norm(gt_points.detach().cpu().numpy() - inr_points.detach().cpu().numpy(), axis=-1)

    # inr -> mesh
    mesh_inr = o3d.io.read_triangle_mesh(path)
    mesh_points = torch.from_numpy(np.asarray(mesh_inr.sample_points_uniformly(number_of_points=50000).points, dtype=np.float32)).to(torch.device("cuda"))
    # couldnt project from random points (gradients take points on a journey outside bounding box since the further youre outside the less accurate they become)
    inr_points = project(mesh_points, model)
    query = o3d.core.Tensor(inr_points.detach().cpu().numpy(), dtype=o3d.core.Dtype.Float32)

    corresponding = mesh.scene.compute_closest_points(query)
    gt_closest_points = corresponding['points'].numpy()

    d2 = np.linalg.norm(inr_points.detach().cpu().numpy() - gt_closest_points, axis=-1)

    chamfer = (np.mean(d1) + np.mean(d2)) / 2
    hausdorff = max(np.max(d1), np.max(d2))

    

    return chamfer, hausdorff

def project(points, model):
    model_zero_level = points.detach().clone()
    max_iter = 1000
    iterations = 0
    while iterations < max_iter:
        sd = model(model_zero_level.requires_grad_(True))
    
        if (torch.abs(sd) > 1e-5).sum() == 0:
            break
        
        sd_grad = torch.autograd.grad(
            outputs=sd,
            inputs=model_zero_level,
            grad_outputs=torch.ones_like(sd),
            create_graph=False
        )[0]
    
        grad_norm_squared = (sd_grad ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        model_zero_level = model_zero_level.detach() - (sd.detach() / grad_norm_squared) * sd_grad.detach()
    
        iterations += 1

    return model_zero_level