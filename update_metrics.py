import os
import numpy as np
import torch
import open3d as o3d
from datetime import datetime

import src.model.SIREN as si
import src.data.dataset as data
from src.model.metrics import chamfer_hausdorff_mesh_based  
DEVICE = torch.device("cuda")

MESHES = ["bunny", "armadillo", "dragon", "lucy"]
SEEDS = [42, 43, 44]


# ---------- utils ----------

def load_model(pth_path: str) -> si.SIRENSDF:
    state_dict = torch.load(pth_path, map_location=DEVICE, weights_only=True)

    hidden_1_out = state_dict["hidden.1.linear.weight"].shape[0]
    hidden_2_out = state_dict["hidden.2.linear.weight"].shape[0]

    model = si.SIRENSDF(hidden_dims=[256, hidden_1_out, hidden_2_out])
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)

    return model


# ---------- main ----------

def main():
    results = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = f"recomputed_metrics_{timestamp}.txt"

    with open(txt_path, "w") as f:

        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log(f"{'='*80}")
        log("RECOMPUTED METRICS (USING EXISTING MESHES)")
        log(f"Timestamp: {timestamp}")
        log(f"{'='*80}\n")

        for mesh_name in MESHES:
            log(f"\n{'='*60}")
            log(f"MESH: {mesh_name}")
            log(f"{'='*60}")

            ply_path = f"data/pointclouds/{mesh_name}/Stanford_{mesh_name}.ply"
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
            )

            for seed in SEEDS:
                log(f"\n  ── seed {seed} ──")

                weight_dir = os.path.join(f"{mesh_name}_data", f"seed_{seed}")

                for file in sorted(os.listdir(weight_dir)):
                    if not file.endswith(".pth"):
                        continue

                    tag = file.replace(".pth", "")
                    pth_path = os.path.join(weight_dir, file)

                    # IMPORTANT: use existing mesh
                    obj_path = os.path.join(weight_dir, f"{tag}.obj")

                    if not os.path.exists(obj_path):
                        log(f"    [{tag}] WARNING: mesh not found → skipping")
                        continue

                    log(f"    [{tag}] loading model...")
                    model = load_model(pth_path)

                    log(f"    [{tag}] computing metrics (existing mesh)...")
                    chamfer, hausdorff = chamfer_hausdorff_mesh_based(
                        mesh_dataset,
                        model,
                        obj_path,
                        gt_points
                    )

                    log(f"    [{tag}] chamfer={chamfer:.6f} | hausdorff={hausdorff:.6f}")

                    results[(mesh_name, seed, tag)] = (chamfer, hausdorff)

        # ---------- summary ----------
        log(f"\n\n{'='*80}")
        log("SUMMARY TABLE")
        log(f"{'='*80}")

        header = f"{'mesh':<12} {'seed':>4}  {'variant':<40}  {'chamfer':>12}  {'hausdorff':>12}"
        log(header)
        log("-" * len(header))

        for (mesh_name, seed, tag), (c, h) in sorted(results.items()):
            log(f"{mesh_name:<12} {seed:>4}  {tag:<40}  {c:>12.6f}  {h:>12.6f}")

        # ---------- save npz ----------
        keys = [f"{m}|{s}|{t}" for (m, s, t) in results]
        vals = np.array(list(results.values()))

        np.savez(
            "recomputed_metrics_mesh_based.npz",
            keys=np.array(keys),
            chamfer=vals[:, 0],
            hausdorff=vals[:, 1],
        )

        log("\nSaved to recomputed_metrics.npz")
        log(f"Saved log to {txt_path}")


if __name__ == "__main__":
    main()