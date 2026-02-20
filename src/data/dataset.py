from networkx import center
import open3d as o3d
import numpy as np
from sklearn.preprocessing import scale

class MeshDataset:
    def __init__(self, mesh_path):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        # Convert vertices to numpy
        v = np.asarray(self.mesh.vertices)

        vmin = v.min(0)
        vmax = v.max(0)

        center = (vmin + vmax) / 2.0
        extent = (vmax - vmin)
        scale = extent.max() / 2.0

        # normalize
        v = (v - center) / scale

        # assign back
        self.mesh.vertices = o3d.utility.Vector3dVector(v)

        # Convert to tensor mesh for raycasting
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tmesh)

        self.vertices = v
        self.triangles = np.asarray(self.mesh.triangles)
    
    def sample_surface_points(self, num_points):
        # Sample points uniformly on the mesh surface
        pcd = self.mesh.sample_points_uniformly(number_of_points=num_points)
        return np.asarray(pcd.points)
    
    def sample_close_to_surface(self, num_points):
        return
    
    def sample_sharp(self, num_points):
        return
        
    def laplace_magnitude(self):

        mag = []
        for v in self.mesh.vertices:
            magnitude = (v - v.neighbours().mean()).norm()
            mag.append(magnitude)
            
                