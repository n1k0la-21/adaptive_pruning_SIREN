import torch
import numpy as np
from src.mesh_extraction.lookup_table import tri_table
import src.model.SIREN as si

def interpolation(a, b, level):
    a = a - level
    b = b - level
    return a / (a - b + 1e-8)  # add epsilon to avoid division by zero

def marching(model, res: int, level=0.0, chunk_size=65536):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1️⃣ Generate coordinate grid
    lin = torch.linspace(-1, 1, steps=res, device=device)
    xxx, yyy, zzz = torch.meshgrid(lin, lin, lin, indexing='ij')
    grid_points = torch.stack([xxx, yyy, zzz], dim=-1).reshape(-1,3)

    # 2️⃣ Corner offsets and edge pairs
    corner_offsets = torch.tensor([
        0, 1, 1+res, 0+res,
        res*res, res*res+1, res*res+1+res, res*res+0+res
    ], device=device, dtype=torch.long)
    edge_pairs = torch.tensor([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ], device=device, dtype=torch.long)

    # 3️⃣ Voxel base indices
    xs = torch.arange(res-1, device=device)
    ys = torch.arange(res-1, device=device)
    zs = torch.arange(res-1, device=device)
    base_x, base_y, base_z = torch.meshgrid(xs, ys, zs, indexing='ij')
    voxel_bases = (base_z * res * res + base_y * res + base_x).flatten()

    vertices = []
    vertex_map = {}  # dict for deduplication
    triangles = []

    num_voxels = len(voxel_bases)
    for start in range(0, num_voxels, chunk_size):
        end = min(start + chunk_size, num_voxels)
        batch = voxel_bases[start:end]

        # Corner indices & positions
        voxel_corner_indices = batch[:, None] + corner_offsets[None, :]
        voxel_corner_points = grid_points[voxel_corner_indices]

        # Evaluate SDF
        flat_points = voxel_corner_points.reshape(-1,3).to(device)
        sdf_flat = model(flat_points)
        voxel_sdf = sdf_flat.reshape(voxel_corner_points.shape[:2]).to(device)

        # Cube index
        mask = (voxel_sdf < level).long()
        bits = (1 << torch.arange(8, device=device, dtype=torch.long))
        cube_index = (mask * bits).sum(dim=1)

        # Edge interpolation
        v1 = voxel_sdf[:, edge_pairs[:,0]]
        v2 = voxel_sdf[:, edge_pairs[:,1]]
        p1 = voxel_corner_points[:, edge_pairs[:,0]]
        p2 = voxel_corner_points[:, edge_pairs[:,1]]

        mask_cross = ((v1 < level) & (v2 > level)) | ((v1 > level) & (v2 < level))
        delta = interpolation(v1, v2, level)
        delta[~mask_cross] = 0.0
        edge_vert = p1 + delta.unsqueeze(-1)*(p2-p1)

        # Assemble triangles using dict deduplication
        for i in range(len(batch)):
            idx = cube_index[i].item()
            edge_indices = tri_table[idx]
            for t in range(0, len(edge_indices), 3):
                if edge_indices[t] == -1:
                    break
                verts = []
                for e in edge_indices[t:t+3]:
                    v = tuple(edge_vert[i,e].detach().cpu().numpy())
                    if v not in vertex_map:
                        vertex_map[v] = len(vertices)
                        vertices.append(v)
                    verts.append(vertex_map[v])
                triangles.append(verts)

    return triangles, vertices

def write_obj(filename, model, resolution: int, level=0.0):
    triangles, vertices = marching(model=model, res=resolution, level=level)
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in triangles:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
