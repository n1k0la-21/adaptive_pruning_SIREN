import numpy as np
import torch
import src.model.SIREN as si
from lookup_table import tri_table

def interpolation(a, b, level):
    a = a - level
    b = b - level
    return a / (a - b)

def marching(model: si.SIRENSDF, resolution: float, level=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triangles = list()

    step_size = 1/resolution

    x, y, z = -1

    while z <= 1:
        y = -1
        while y <= 1:
            x = -1
            while x <= 1:
                edge_vert = [None] * 12 # empty list with 12 slots
                
                dots = np.array(
                    [[x, y, z], # 0. edges: 0-1, 0-2, 0-4
                     [x + step_size, y, z], # 1. edges: 1-5, 1-3
                     [x, y + step_size, z], # 2. edges: 2-6, 2-3
                     [x + step_size, y + step_size, z], # 3: edges: 3-7
                     [x, y, z + step_size], # 4. edges: 4-6, 4-5
                     [x + step_size, y, z + step_size], # 5. edges: 5-6
                     [x, y + step_size, z + step_size], # 6. edges: 6-7
                     [x + step_size, y + step_size, z + step_size]] # 7.
                     )
                
                edges = [
                    (0,1),(1,3),(3,2),(2,0),
                    (4,5),(5,7),(7,6),(6,4),
                    (0,4),(1,5),(2,6),(3,7)
                    ]
                
                tensor = torch.from_numpy(dots).float().to(device)
                sdf_result = model(tensor)
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

                    if (v1 < level and v2 > level) or (v1 > 0 and v2 < 0):
                        p1 = dots[a]
                        p2 = dots[b]
                        delta = interpolation(v1, v2, level)

                        v = p1 + delta * (p2 - p1)
                        edge_vert[i] = v

                # find triangles using cube_index and append to triangles aswell as vertices
                edges = tri_table[cube_index]
                index = 0
                temp = list()
                while index in range(len(edges)-1):
                    if(edges[index] == -1):
                        break
                    
                    edge_id = edges[index]
                    temp.append(edge_vert[edge_id])
                    
                    if index % 3 == 2:
                        triangles.append(temp)
                        temp = list()
                    index += 1

                x += step_size
            y += step_size
        z += step_size

    return triangles