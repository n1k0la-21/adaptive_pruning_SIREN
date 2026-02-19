import torch
import numpy as np
from src.mesh_extraction.lookup_table import tri_table
import src.model.SIREN as si

def interpolation(a, b, level):
    """Linear interpolation along edge for the isosurface crossing."""
    a = a - level
    b = b - level
    return a / (a - b + 1e-8)  # add epsilon to avoid division by zero

def marching(model: si.SIRENSDF, res: int, level=0.0, chunk_size=65536):
    """
    Fully vectorized Marching Cubes using Paul Bourke's numbering.
    Grid is generated as xxx, yyy, zzz for vectorization.
    """
    device = next(model.parameters()).device

    # 1️⃣ Generate coordinate grids
    lin = torch.linspace(-1, 1, steps=res, device=device)
    xxx, yyy, zzz = torch.meshgrid(lin, lin, lin, indexing='ij')  # shape (res,res,res)
    
    # Flatten the grid points for SDF evaluation
    grid_points = torch.stack([xxx, yyy, zzz], dim=-1).reshape(-1,3)  # (res^3, 3)

    # 2️⃣ Precompute corner offsets for Paul Bourke numbering
    corner_offsets = torch.tensor([
        0,                 # 0
        1,                 # 1
        1 + res,           # 2
        0 + res,           # 3
        res*res,           # 4
        res*res + 1,       # 5
        res*res + 1 + res, # 6
        res*res + 0 + res  # 7
    ], device=device, dtype=torch.long)

    # 3️⃣ Edge pairs
    edge_pairs = torch.tensor([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ], device=device, dtype=torch.long)

    # 4️⃣ Generate voxel base indices (skip last along each axis)
    xs = torch.arange(res-1, device=device)
    ys = torch.arange(res-1, device=device)
    zs = torch.arange(res-1, device=device)
    base_x, base_y, base_z = torch.meshgrid(xs, ys, zs, indexing='ij')
    voxel_bases = (base_z * res * res + base_y * res + base_x).flatten()  # (num_voxels,)

    triangles = []
    vertices = []

    # 5️⃣ Process voxels in chunks
    num_voxels = len(voxel_bases)
    for start in range(0, num_voxels, chunk_size):
        end = min(start + chunk_size, num_voxels)
        batch = voxel_bases[start:end]  # (chunk,)

        # 6️⃣ Get 8 corner indices per voxel
        voxel_corner_indices = batch[:, None] + corner_offsets[None, :]  # (chunk, 8)
        voxel_corner_points = grid_points[voxel_corner_indices]  # (chunk, 8, 3)

        # 7️⃣ Evaluate SDF in one forward pass
        flat_points = voxel_corner_points.reshape(-1, 3)
        sdf_flat = model(flat_points).flatten()
        voxel_sdf = sdf_flat.reshape(voxel_corner_points.shape[:2])  # (chunk,8)

        # 8️⃣ Cube index
        mask = (voxel_sdf < level).long()
        bits = (1 << torch.arange(8, device=device, dtype=torch.long))
        cube_index = (mask * bits).sum(dim=1)  # (chunk,)

        # 9️⃣ Edge interpolation
        v1 = voxel_sdf[:, edge_pairs[:,0]]  # (chunk, 12)
        v2 = voxel_sdf[:, edge_pairs[:,1]]  # (chunk, 12)
        p1 = voxel_corner_points[:, edge_pairs[:,0]]  # (chunk,12,3)
        p2 = voxel_corner_points[:, edge_pairs[:,1]]  # (chunk,12,3)

        mask_cross = ((v1 < level) & (v2 > level)) | ((v1 > level) & (v2 < level))
        delta = interpolation(v1, v2, level)
        delta[~mask_cross] = 0.0
        edge_vert = p1 + delta.unsqueeze(-1)*(p2-p1)  # (chunk,12,3)

        # 10️⃣ Assemble triangles for this batch
        # This part still requires some Python iteration per voxel, but no nested loops
        for i in range(len(batch)):
            idx = cube_index[i].item()
            edge_indices = tri_table[idx]
            for t in range(0, len(edge_indices), 3):
                if edge_indices[t] == -1:
                    break
                verts = []
                for e in edge_indices[t:t+3]:
                    v = edge_vert[i,e].detach().cpu().numpy()
                    try:
                        vi = vertices.index(tuple(v))
                    except ValueError:
                        vertices.append(tuple(v))
                        vi = len(vertices)-1
                    verts.append(vi)
                triangles.append(verts)

    return triangles, vertices

def write_obj(filename, model: si.SIRENSDF, resolution: int, level=0.0):
    triangles, vertices = marching(model=model, res=resolution, level=level)
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in triangles:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
