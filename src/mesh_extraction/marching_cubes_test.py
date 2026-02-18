import numpy as np
import torch
from src.mesh_extraction.lookup_table import tri_table
import open3d as o3d

def interpolation(a, b, level):
    a = a - level
    b = b - level
    return a / (a - b)

def marching(scene: o3d.t.geometry.RaycastingScene, res: int, level=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triangles = list()
    vertices = list()

    step_size = 2.0 / res   # because domain is [-1, 1]

    for zi in range(res):
        z = -1 + zi * step_size
        for yi in range(res):
            y = -1 + yi * step_size
            for xi in range(res):
                x = -1 + xi * step_size

                edge_vert = [None] * 12 # empty list with 12 slots
                
                dots = np.asarray([[x, y, z], # 0. edges: 0-1, 0-2, 0-4
                     [x + step_size, y, z], # 1. edges: 1-5, 1-3
                     [x + step_size, y + step_size, z], # 2. edges: 2-6, 2-3
                     [x, y + step_size, z], # 3: edges: 3-7
                     [x, y, z + step_size], # 4. edges: 4-6, 4-5
                     [x + step_size, y, z + step_size], # 5. edges: 5-6
                     [x + step_size, y + step_size, z + step_size], # 6. edges: 6-7
                     [x, y + step_size, z + step_size]] # 7.
                )
                
                edges = [
                    (0,1),(1,2),(2,3),(3,0),
                    (4,5),(5,6),(6,7),(7,4),
                    (0,4),(1,5),(2,6),(3,7)
                    ]
                
                tensor = o3d.core.Tensor(dots, dtype=o3d.core.Dtype.Float32)
                sdf_result = scene.compute_signed_distance(tensor)
                cube_index = 0

                # build index for triTable
                for i in range(8):
                    if sdf_result[i] < level:
                        cube_index |= (1 << i)

                # find exact location of vertices
                for i in range(len(edges)):
                    a, b = edges[i]
                    v1 = sdf_result[a].item()
                    v2 = sdf_result[b].item()

                    if (v1 < level and v2 > level) or (v1 > level and v2 < level):
                        p1 = dots[a]
                        p2 = dots[b]
                        delta = interpolation(v1, v2, level)

                        v = p1 + delta * (p2 - p1)
                        edge_vert[i] = v

                # find triangles using cube_index and append to triangles aswell as vertices
                edges = tri_table[cube_index]
                index = 0
                
                for index in range(0, len(edges), 3):
                    if(edges[index] == -1):
                        break
                    
                    e0 = edges[index]
                    e1 = edges[index + 1]
                    e2 = edges[index + 2]

                    v0 = edge_vert[e0]
                    v1 = edge_vert[e1]
                    v2 = edge_vert[e2]

                    if v0 is None or v1 is None or v2 is None:
                        continue  # skip invalid triangle

                    vert_list = [v0, v1, v2]
                    triangle = []

                    for v in vert_list:
                        if v is None:
                            continue  # skip invalid vertex
                        
                        v_tuple = tuple(v)
                        
                        try:
                            i = vertices.index(v_tuple)
                            triangle.append(i)
                        except ValueError:
                            vertices.append(v_tuple)
                            triangle.append(vertices.index(v_tuple))

                    triangles.append(triangle)

    return triangles, vertices

def write_obj(filename, scene: o3d.t.geometry.RaycastingScene, resolution: int, level=0.0):
    triangles, vertices = marching(scene=scene, res=resolution, level=level)
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for tri in triangles:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")