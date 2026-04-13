import torch
import numpy as np
import src.model.SIREN as si
import open3d as o3d

def chamfer_hausdorff(mesh, model: si.SIRENSDF, path, gt_points):
    o3d.utility.random.seed(42)

    # mesh -> inr
    inr_points = project(gt_points, model)
    filtered_points, mask = filter_bbox(inr_points, model)
    print(f"mesh -> sdf \n projected points: {inr_points.shape[0]}, after filtering out points outside bbox: {filtered_points.shape[0]}")

    d1 = np.linalg.norm(gt_points[mask].detach().cpu().numpy() - filtered_points.detach().cpu().numpy(), axis=-1)

    # inr -> mesh
    mesh_inr = o3d.io.read_triangle_mesh(path)
    inr_points = torch.from_numpy(np.asarray(mesh_inr.sample_points_uniformly(number_of_points=50000).points, dtype=np.float32)).to(torch.device("cuda"))
    inr_points = project(inr_points, model)
    filtered_points, _ = filter_bbox(inr_points, model)
    query = o3d.core.Tensor(filtered_points.detach().cpu().numpy(), dtype=o3d.core.Dtype.Float32)

    print(f"sdf -> mesh \n projected points: {inr_points.shape[0]}, after filtering out points outside bbox: {filtered_points.shape[0]}")

    corresponding = mesh.scene.compute_closest_points(query)
    gt_closest_points = corresponding['points'].numpy()

    d2 = np.linalg.norm(filtered_points.detach().cpu().numpy() - gt_closest_points, axis=-1)

    chamfer = (np.mean(d1) + np.mean(d2)) / 2
    hausdorff = max(np.max(d1), np.max(d2))

    return chamfer, hausdorff

def chamfer_hausdorff_mesh_based(mesh, model:si.SIRENSDF, path, gt_points):
    o3d.utility.random.seed(42)

    # ---------- load meshes ----------
    gt_mesh = mesh.mesh
    pred_mesh = o3d.io.read_triangle_mesh(path)

    # ---------- GT points (fixed, provided) ----------
    gt_points_np = gt_points.detach().cpu().numpy()

    # ---------- sample pred mesh ----------
    pred_points = np.asarray(
        pred_mesh.sample_points_uniformly(number_of_points=50000).points
    )

    # ---------- build scenes ----------
    gt_tmesh = o3d.t.geometry.TriangleMesh.from_legacy(gt_mesh)
    pred_tmesh = o3d.t.geometry.TriangleMesh.from_legacy(pred_mesh)

    gt_scene = o3d.t.geometry.RaycastingScene()
    gt_scene.add_triangles(gt_tmesh)

    pred_scene = o3d.t.geometry.RaycastingScene()
    pred_scene.add_triangles(pred_tmesh)

    # ---------- GT → Pred ----------
    query_gt = o3d.core.Tensor(gt_points_np, dtype=o3d.core.Dtype.Float32)
    result_gt = pred_scene.compute_closest_points(query_gt)
    closest_pred = result_gt["points"].numpy()

    d1 = np.linalg.norm(gt_points_np - closest_pred, axis=1)

    print(f"GT -> Pred: {len(d1)} samples (fixed)")

    # ---------- Pred → GT ----------
    query_pred = o3d.core.Tensor(pred_points, dtype=o3d.core.Dtype.Float32)
    result_pred = gt_scene.compute_closest_points(query_pred)
    closest_gt = result_pred["points"].numpy()

    d2 = np.linalg.norm(pred_points - closest_gt, axis=1)

    print(f"Pred -> GT: {len(d2)} samples (mesh sampled)")

    # ---------- metrics ----------
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
        model_zero_level = model_zero_level.detach() - (sd.detach() / grad_norm_squared) * sd_grad.detach() * 0.1
    
        iterations += 1

        if iterations == max_iter:
            print(f"model probably breaks here! Interesting for thesis")

    return model_zero_level

def filter_bbox(points, model):
    sd_mask = (torch.abs(model(points)) <= 1e-5).squeeze(-1)
    bb_mask = ((points >= -1) & (points <= 1)).all(dim=-1)
    mask = sd_mask + bb_mask
    return points[mask], mask