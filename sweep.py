"""
sweep.py  –  Full hyperparameter / seed sweep for SIREN SDF experiments.
 
For every mesh in {lucy, armadillo, dragon, bunny} this script trains:
  1. One large un-pruned model (256-256-256)
  2. One densified model       (256-256-256)
  3. For each pruning ratio in [0.3, 0.95]
       a. AIRe               (256 → keep hidden)
       b. DepGraph            (256 → keep hidden)
       c. AIRe  + densification
       d. DepGraph + densification
 
Each run is repeated for seeds {42, 43, 44}.  Results (weights + history) are
stored under  {mesh}_weights/seed_{seed}/  so different seeds never collide.
 
Folder structure produced (example for bunny, seed 42, ratio 0.30):
  bunny_weights/seed_42/large_unpruned.pth
  bunny_weights/seed_42/history/large_unpruned_history.npz
  bunny_weights/seed_42/densified.pth
  bunny_weights/seed_42/history/densified_history.npz
  bunny_weights/seed_42/AIRe_0.3.pth
  bunny_weights/seed_42/history/AIRe_0.3_history.npz
  ...
 
Mesh (.obj) files used for chamfer/hausdorff are written to the same
seed sub-directory so they are unambiguous across runs.
"""
 
import os
import math
import random
 
import numpy as np
import torch
import open3d as o3d
 
import src.model.SIREN as si
from src.model.training import train 
import src.loss.SDF_loss as loss
import src.model.pruning_module as pm
import src.data.dataset as data
import src.mesh_extraction.marching_cubes_gpu as marching_cubes
from src.model.metrics import chamfer_hausdorff
 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  
 
MESHES = ["bunny", "armadillo", "dragon", "lucy"]
 
SEEDS = [42, 43, 44]
 
# 30 % → 95 % in somewhat logarithmic steps (5 total)
PRUNE_RATIOS = [0.30, 0.50, 0.70, 0.85, 0.95]
 
EPOCHS       = 1000
NO_SURFACE   = 10000
NO_OFF_SURFACE = 15000
LR           = 1e-4 * 2
 
LOSS_KWARGS = dict(
    lambda_surface=175,
    lambda_eikonal=20,
    lambda_normal=15,
    lambda_inter=10,
    lambda_off=15,
)
 
DEVICE = torch.device("cuda")
MC_RESOLUTION = 256 
MC_LEVEL      = 0.0
 
 
# helpers 
 
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
 
def compute_iou(model, mesh_dataset) -> float:
    """Compute IoU on a 256^3 grid against the ground-truth SDF."""
    with torch.no_grad(): 
        x = torch.linspace(-1, 1, 256)
        y = torch.linspace(-1, 1, 256)
        z = torch.linspace(-1, 1, 256)
        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        grid_o3d = o3d.core.Tensor(grid.numpy(), dtype=o3d.core.Dtype.Float32)
        grid_sdf = mesh_dataset.scene.compute_signed_distance(grid_o3d).numpy()
        grid_mask = torch.tensor(grid_sdf < 0, dtype=torch.bool)

    from src.model.training_copy import iou
    return iou(model, grid_mask, grid)
 
def hidden_size_after_prune(ratio: float, base: int = 256) -> int:
    """Return the number of neurons kept after pruning *ratio* fraction."""
    return math.ceil(base * (1.0 - ratio))
 
 
def make_dirs(mesh_name: str, seed: int) -> tuple[str, str]:
    """Create weight / history directories and return their paths."""
    weight_dir  = os.path.join(f"{mesh_name}_data", f"seed_{seed}")
    history_dir = os.path.join(weight_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    return weight_dir, history_dir
 
 
def save_history(
    history_dir: str,
    name: str,
    loss_hist,
    iou_hist,
    iou_steps,
    iou: float = None,
) -> None:
    kwargs = dict(
        loss=np.array(loss_hist),
        iou=np.array(iou_hist),
        steps=np.array(iou_steps),
    )
    if iou is not None:
        kwargs["iou"] = np.array([iou])
    np.savez(os.path.join(history_dir, f"{name}_history.npz"), **kwargs)
 
 
def extract_mesh_and_metrics(
    model,
    mesh_dataset,
    gt_points,
    obj_path: str,
) -> tuple[float, float]:
    """
    Write the reconstructed mesh to *obj_path* then compute chamfer / hausdorff.
    The mesh extraction is mandatory for the metric computation to work.
    """
    marching_cubes.write_obj(obj_path, model=model, resolution=MC_RESOLUTION, level=MC_LEVEL)
    chamfer, hausdorff = chamfer_hausdorff(mesh_dataset, model, obj_path, gt_points)
    return chamfer, hausdorff
 
 
def load_model(pth_path: str) -> si.SIRENSDF:
    state_dict = torch.load(pth_path, map_location=DEVICE, weights_only=True)

    hidden_1_out = state_dict["hidden.1.linear.weight"].shape[0]
    hidden_2_out = state_dict["hidden.2.linear.weight"].shape[0]

    model = si.SIRENSDF(hidden_dims=[256, hidden_1_out, hidden_2_out])
    
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    
    return model
 
  
def run_large_unpruned(mesh_dataset, gt_points, weight_dir, history_dir, seed):
    tag = "large_unpruned"
    pth = os.path.join(weight_dir, f"{tag}.pth")
    obj = os.path.join(weight_dir, f"{tag}.obj")
 
    print(f"    [{tag}] training …")
    set_seeds(seed)
    model      = si.SIRENSDF(hidden_dims=[256, 256, 256])
    model_loss = loss.Loss(**LOSS_KWARGS, model=model)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
 
    loss_hist, iou_hist, iou_steps = train(
        epochs=EPOCHS, data=mesh_dataset,
        no_surface=NO_SURFACE, no_off_surface=NO_OFF_SURFACE,
        model=model, loss=model_loss, optimizer=optimizer,
        scene=mesh_dataset.scene,
    )
    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps)
    torch.save(model.state_dict(), pth)
 
    print(f"    [{tag}] extracting mesh + metrics …")
    model_eval = load_model(pth)
    chamfer, hausdorff = extract_mesh_and_metrics(model_eval, mesh_dataset, gt_points, obj)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f}\n")
    
    print(f"    [{tag}] computing 256^3 IoU …")
    iou = compute_iou(model_eval, mesh_dataset)

    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps, iou)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f} | iou={iou:.4f}\n")
    return chamfer, hausdorff, iou
 
 
def run_densified(mesh_dataset, gt_points, weight_dir, history_dir, seed):
    tag = "densified"
    pth = os.path.join(weight_dir, f"{tag}.pth")
    obj = os.path.join(weight_dir, f"{tag}.obj")
 
    print(f"    [{tag}] training …")
    set_seeds(seed)
    model      = si.SIRENSDF(hidden_dims=[151, 256, 256])
    model_loss = loss.Loss(**LOSS_KWARGS, model=model)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
 
    loss_hist, iou_hist, iou_steps = train(
        epochs=EPOCHS, data=mesh_dataset,
        no_surface=NO_SURFACE, no_off_surface=NO_OFF_SURFACE,
        model=model, loss=model_loss, optimizer=optimizer,
        scene=mesh_dataset.scene, densification=True,
    )
    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps)
    torch.save(model.state_dict(), pth)
 
    print(f"    [{tag}] extracting mesh + metrics …")
    model_eval = load_model(pth)
    chamfer, hausdorff = extract_mesh_and_metrics(model_eval, mesh_dataset, gt_points, obj)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f}\n")
    
    print(f"    [{tag}] computing 256^3 IoU …")
    iou = compute_iou(model_eval, mesh_dataset)

    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps, iou)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f} | iou={iou:.4f}\n")
    return chamfer, hausdorff, iou
 
 
def run_aire(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio):
    keep = hidden_size_after_prune(ratio)
    ratio_str = str(ratio)
    tag = f"AIRe_{ratio_str}"
    pth = os.path.join(weight_dir, f"{tag}.pth")
    obj = os.path.join(weight_dir, f"{tag}.obj")
 
    print(f"    [{tag}] training  (keep={keep}) …")
    set_seeds(seed)
    model       = si.SIRENSDF(hidden_dims=[256, 256, 256])
    prune_AIRe  = pm.AIRe(model, ratio)
    model_loss  = loss.Loss(**LOSS_KWARGS, model=model, pruning_module=prune_AIRe)
    optimizer   = torch.optim.Adam(model.parameters(), lr=LR)
 
    loss_hist, iou_hist, iou_steps = train(
        epochs=EPOCHS, data=mesh_dataset,
        no_surface=NO_SURFACE, no_off_surface=NO_OFF_SURFACE,
        model=model, loss=model_loss, optimizer=optimizer,
        scene=mesh_dataset.scene, pruning_module=prune_AIRe,
    )
    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps)
    torch.save(model.state_dict(), pth)
 
    print(f"    [{tag}] extracting mesh + metrics …")
    model_eval = load_model(pth)
    chamfer, hausdorff = extract_mesh_and_metrics(model_eval, mesh_dataset, gt_points, obj)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f}\n")
    
    print(f"    [{tag}] computing 256^3 IoU …")
    iou = compute_iou(model_eval, mesh_dataset)

    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps, iou)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f} | iou={iou:.4f}\n")
    return chamfer, hausdorff, iou
 
 
def run_depgraph(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio):
    keep = hidden_size_after_prune(ratio) - 1   # DepGraph keeps one fewer neuron
    
    ratio_str = str(ratio)
    tag = f"DepGraph_{ratio_str}"
    pth = os.path.join(weight_dir, f"{tag}.pth")
    obj = os.path.join(weight_dir, f"{tag}.obj")
 
    print(f"    [{tag}] training  (keep={keep}) …")
    set_seeds(seed)
    model          = si.SIRENSDF(hidden_dims=[256, 256, 256]).to(DEVICE)
    prune_DepGraph = pm.DepGraph(model, ratio)
    model_loss     = loss.Loss(**LOSS_KWARGS, model=model, pruning_module=prune_DepGraph)
    optimizer      = torch.optim.Adam(model.parameters(), lr=LR)
 
    loss_hist, iou_hist, iou_steps = train(
        epochs=EPOCHS, data=mesh_dataset,
        no_surface=NO_SURFACE, no_off_surface=NO_OFF_SURFACE,
        model=model, loss=model_loss, optimizer=optimizer,
        scene=mesh_dataset.scene, pruning_module=prune_DepGraph,
    )
    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps)
    torch.save(model.state_dict(), pth)
 
    print(f"    [{tag}] extracting mesh + metrics …")
    model_eval = load_model(pth)
    chamfer, hausdorff = extract_mesh_and_metrics(model_eval, mesh_dataset, gt_points, obj)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f}\n")
    
    print(f"    [{tag}] computing 256^3 IoU …")
    iou = compute_iou(model_eval, mesh_dataset)

    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps, iou)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f} | iou={iou:.4f}\n")
    return chamfer, hausdorff, iou
 
 
def run_aire_densified(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio):
    keep = hidden_size_after_prune(ratio)
    ratio_str = str(ratio)
    tag = f"AIRe_{ratio_str}_densified"
    pth = os.path.join(weight_dir, f"{tag}.pth")
    obj = os.path.join(weight_dir, f"{tag}.obj")
 
    dense_first = 151  
 
    print(f"    [{tag}] training  (dense_first={dense_first}, keep={keep}) …")
    set_seeds(seed)
    model      = si.SIRENSDF(hidden_dims=[dense_first, 256, 256])
    prune_AIRe = pm.AIRe(model, ratio)
    model_loss = loss.Loss(**LOSS_KWARGS, model=model, pruning_module=prune_AIRe)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
 
    loss_hist, iou_hist, iou_steps = train(
        epochs=EPOCHS, data=mesh_dataset,
        no_surface=NO_SURFACE, no_off_surface=NO_OFF_SURFACE,
        model=model, loss=model_loss, optimizer=optimizer,
        scene=mesh_dataset.scene, pruning_module=prune_AIRe, densification=True,
    )
    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps)
    torch.save(model.state_dict(), pth)
 
    print(f"    [{tag}] extracting mesh + metrics …")
    model_eval = load_model(pth)
    chamfer, hausdorff = extract_mesh_and_metrics(model_eval, mesh_dataset, gt_points, obj)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f}\n")
    
    print(f"    [{tag}] computing 256^3 IoU …")
    iou = compute_iou(model_eval, mesh_dataset)

    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps, iou)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f} | iou={iou:.4f}\n")
    return chamfer, hausdorff, iou
 
 
def run_depgraph_densified(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio):
    keep = hidden_size_after_prune(ratio) - 1

    ratio_str = str(ratio)
    tag = f"DepGraph_{ratio_str}_densified"
    pth = os.path.join(weight_dir, f"{tag}.pth")
    obj = os.path.join(weight_dir, f"{tag}.obj")
 
    dense_first = 151 
 
    print(f"    [{tag}] training  (dense_first={dense_first}, keep={keep}) …")
    set_seeds(seed)
    model          = si.SIRENSDF(hidden_dims=[dense_first, 256, 256]).to(DEVICE)
    prune_DepGraph = pm.DepGraph(model, ratio)
    model_loss     = loss.Loss(**LOSS_KWARGS, model=model, pruning_module=prune_DepGraph)
    optimizer      = torch.optim.Adam(model.parameters(), lr=LR)
 
    loss_hist, iou_hist, iou_steps = train(
        epochs=EPOCHS, data=mesh_dataset,
        no_surface=NO_SURFACE, no_off_surface=NO_OFF_SURFACE,
        model=model, loss=model_loss, optimizer=optimizer,
        scene=mesh_dataset.scene, pruning_module=prune_DepGraph, densification=True,
    )
    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps)
    torch.save(model.state_dict(), pth)
 
    print(f"    [{tag}] extracting mesh + metrics …")
    model_eval = load_model(pth)
    chamfer, hausdorff = extract_mesh_and_metrics(model_eval, mesh_dataset, gt_points, obj)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f}\n")
    
    print(f"    [{tag}] computing 256^3 IoU …")
    iou = compute_iou(model_eval, mesh_dataset)

    save_history(history_dir, tag, loss_hist, iou_hist, iou_steps, iou)
    print(f"    [{tag}] chamfer={chamfer:.4f} | hausdorff={hausdorff:.4f} | iou={iou:.4f}\n")
    return chamfer, hausdorff, iou
 
  
def main():
    all_results = {}   # {(mesh, seed, tag): (chamfer, hausdorff)}
 
    for mesh_name in MESHES:
        ply_path = f"data/pointclouds/{mesh_name}/Stanford_{mesh_name}.ply"
        print(f"\n{'='*60}")
        print(f"MESH: {mesh_name}   ({ply_path})")
        print(f"{'='*60}")
 
        mesh_dataset = data.MeshDataset(ply_path)
 
        gt_points = (
            torch.from_numpy(
                np.asarray(
                    mesh_dataset.mesh.sample_points_uniformly(
                        number_of_points=50_000
                    ).points
                ).astype(np.float32)
            )
            .to(DEVICE)
            .requires_grad_(True)
        )
 
        for seed in SEEDS:
            print(f"\n  ── seed {seed} ──")
            weight_dir, history_dir = make_dirs(mesh_name, seed)
 
            c, h, iou = run_large_unpruned(mesh_dataset, gt_points, weight_dir, history_dir, seed)
            all_results[(mesh_name, seed, "large_unpruned")] = (c, h, iou)

            c, h, iou = run_densified(mesh_dataset, gt_points, weight_dir, history_dir, seed)
            all_results[(mesh_name, seed, "densified")] = (c, h, iou)

 
            # Pruning
            for ratio in PRUNE_RATIOS:
                c, h, iou = run_aire(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio)
                all_results[(mesh_name, seed, f"AIRe_{ratio}")] = (c, h, iou)
 
                c, h, iou = run_depgraph(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio)
                all_results[(mesh_name, seed, f"DepGraph_{ratio}")] = (c, h, iou)
 
                c, h, iou = run_aire_densified(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio)
                all_results[(mesh_name, seed, f"AIRe_{ratio}_densified")] = (c, h, iou)
 
                c, h, iou = run_depgraph_densified(mesh_dataset, gt_points, weight_dir, history_dir, seed, ratio)
                all_results[(mesh_name, seed, f"DepGraph_{ratio}_densified")] = (c, h, iou)
 
    # print summary table 
    print(f"\n\n{'='*80}")
    print("SWEEP COMPLETE – SUMMARY")
    print(f"{'='*80}")
    header = f"{'mesh':<12} {'seed':>4}  {'variant':<40}  {'chamfer':>10}  {'hausdorff':>10}  {'iou':>10}"
    print(header)
    print("-" * len(header))
    for (mesh_name, seed, tag), (chamfer, hausdorff, iou) in sorted(all_results.items()):
        print(f"{mesh_name:<12} {seed:>4}  {tag:<40}  {chamfer:>10.4f}  {hausdorff:>10.4f}  {iou:>10.4f}")

    keys = [f"{m}|{s}|{t}" for (m, s, t) in all_results]
    vals = np.array(list(all_results.values()))   # shape (N, 3)
    np.savez("sweep_results.npz", keys=np.array(keys), chamfer=vals[:, 0], hausdorff=vals[:, 1], iou=vals[:, 2])
    print("\nResults saved to sweep_results.npz")
 
 
if __name__ == "__main__":
    main()
