import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def compute_surface_distances(scene, query_points):
    """
    Compute distance from query points to mesh in RaycastingScene.
    """
    query = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    result = scene.compute_closest_points(query)

    closest_points = result["points"].numpy()
    distances = np.linalg.norm(closest_points - query_points, axis=1)

    return distances


def color_mesh_from_dataset(
    gt_dataset,
    pred_mesh_path,
    output_path=None,
    colormap="viridis",
    max_dist=0.01
):
    """
    High-quality distance visualization using surface distances.
    """

    # --- load predicted mesh ---
    pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)

    # --- build raycasting scene for predicted mesh ---
    pred_tmesh = o3d.t.geometry.TriangleMesh.from_legacy(pred_mesh)
    pred_scene = o3d.t.geometry.RaycastingScene()
    pred_scene.add_triangles(pred_tmesh)

    # --- GT points ---
    gt_points = gt_dataset.vertices
    pred_points = np.asarray(pred_mesh.vertices)

    # --- GT → Pred (smooth surface distance) ---
    d_gt_to_pred = compute_surface_distances(pred_scene, gt_points)

    # --- Pred → GT (captures floaters) ---
    d_pred_to_gt = compute_surface_distances(gt_dataset.scene, pred_points)

    # --- propagate pred errors back to GT ---
    # nearest GT vertex for each pred point
    from scipy.spatial import cKDTree
    gt_tree = cKDTree(gt_points)

    _, idx = gt_tree.query(pred_points)

    d_pred_propagated = np.zeros(len(gt_points))
    for i, gt_idx in enumerate(idx):
        d_pred_propagated[gt_idx] = max(
            d_pred_propagated[gt_idx],
            d_pred_to_gt[i]
        )

    d_final = np.maximum(d_gt_to_pred, d_pred_propagated)

    d_final = np.clip(d_final, 0, max_dist)

    d_norm = (d_final / max_dist) ** 0.6

    # --- colormap ---
    cmap = plt.get_cmap(colormap)
    colors = cmap(d_norm)[:, :3]

    # --- assign colors ---
    colored_mesh = o3d.geometry.TriangleMesh(gt_dataset.mesh)
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # --- save if requested ---
    if output_path:
        o3d.io.write_triangle_mesh(output_path, colored_mesh)
        print(f"Saved to {output_path}")

    return colored_mesh