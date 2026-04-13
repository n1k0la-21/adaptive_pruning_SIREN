import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap


def get_gray_blue_red_cmap():
    return LinearSegmentedColormap.from_list(
        "gray_blue_red",
        [
            (0.0, "#909090"),
            (0.35, "#057F9E"),
            (0.6, "#FFFF00"),
            (1.0, "#FF0000"),
        ]
    )


def compute_surface_distances(scene, query_points):
    query = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    result = scene.compute_closest_points(query)

    closest_points = result["points"].numpy()
    distances = np.linalg.norm(closest_points - query_points, axis=1)

    return distances


def propagate_distances(
    source_points,
    target_points,
    source_distances,
    radius
):
    """
    Propagate distances from source_points → target_points
    using neighborhood max pooling.
    """
    tree = cKDTree(target_points)
    propagated = np.zeros(len(target_points))

    for i, p in enumerate(source_points):
        d = source_distances[i]

        if d < 1e-8:
            continue

        neighbors = tree.query_ball_point(p, r=radius)

        for n in neighbors:
            propagated[n] = max(propagated[n], d)

    return propagated


def color_mesh_from_dataset(
    gt_dataset,
    pred_mesh_path,
    output_path_gt=None,
    output_path_pred=None,
    max_dist=0.01,
    spread_radius=0.01
):

    # --- Load meshes ---
    pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)

    pred_tmesh = o3d.t.geometry.TriangleMesh.from_legacy(pred_mesh)
    pred_scene = o3d.t.geometry.RaycastingScene()
    pred_scene.add_triangles(pred_tmesh)

    gt_points = gt_dataset.vertices
    pred_points = np.asarray(pred_mesh.vertices)

    # --- Distance computations ---
    d_gt_to_pred = compute_surface_distances(pred_scene, gt_points)
    d_pred_to_gt = compute_surface_distances(gt_dataset.scene, pred_points)

    # --- Symmetric propagation ---
    d_pred_propagated_on_gt = propagate_distances(
        pred_points,
        gt_points,
        d_pred_to_gt,
        spread_radius
    )

    d_gt_propagated_on_pred = propagate_distances(
        gt_points,
        pred_points,
        d_gt_to_pred,
        spread_radius
    )

    # --- Combine distances ---
    d_final_gt = np.maximum(d_gt_to_pred, d_pred_propagated_on_gt)
    d_final_pred = np.maximum(d_pred_to_gt, d_gt_propagated_on_pred)

    # --- Normalize ---
    d_final_gt = np.clip(d_final_gt, 0, max_dist)
    d_final_pred = np.clip(d_final_pred, 0, max_dist)

    d_norm_gt = (d_final_gt / max_dist) ** 0.9
    d_norm_pred = (d_final_pred / max_dist) ** 0.9

    # --- Colormap ---
    cmap = get_gray_blue_red_cmap()

    # --- Color GT mesh ---
    gt_colors = cmap(d_norm_gt)[:, :3]
    colored_gt_mesh = o3d.geometry.TriangleMesh(gt_dataset.mesh)
    colored_gt_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_colors)

    # --- Color Pred mesh ---
    pred_colors = cmap(d_norm_pred)[:, :3]
    pred_mesh.vertex_colors = o3d.utility.Vector3dVector(pred_colors)

    # --- Save ---
    if output_path_gt:
        o3d.io.write_triangle_mesh(output_path_gt, colored_gt_mesh)
        print(f"Saved GT mesh to {output_path_gt}")

    if output_path_pred:
        o3d.io.write_triangle_mesh(output_path_pred, pred_mesh)
        print(f"Saved Pred mesh to {output_path_pred}")

    return colored_gt_mesh, pred_mesh