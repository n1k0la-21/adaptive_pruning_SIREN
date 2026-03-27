import open3d as o3d
import numpy as np

class MeshDataset:
    def __init__(self, mesh_path):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        v = np.asarray(self.mesh.vertices)

        vmin = v.min(axis=0)
        vmax = v.max(axis=0)

        center = (vmin + vmax) / 2.0
        extent = (vmax - vmin)
        scale = extent.max() / 2.0 + 0.02

        v = (v - center) / scale

        # assign back
        self.mesh.vertices = o3d.utility.Vector3dVector(v)

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tmesh)

        self.vertices = v
        self.triangles = np.asarray(self.mesh.triangles)

        # detect sharper regions 
        self.l_mag = self.laplace_magnitude()
        mask = self.l_mag > np.percentile(self.l_mag, 70) 
        self.biased = self.vertices[mask] # extract top 30%

        if not hasattr(self, 'vertex_normals'):
            self.mesh.compute_vertex_normals()
            self.vertex_normals = np.asarray(self.mesh.vertex_normals)

        pcd = o3d.geometry.PointCloud()
        pcd.points = self.mesh.vertices
        pcd.normals = self.mesh.vertex_normals
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)


    def sample_surface_points(self, num_points, rng: np.random.Generator) -> np.ndarray:
        # Sample points uniformly on the mesh surface
        pcd = np.asarray(self.mesh.sample_points_uniformly(number_of_points=int(2*num_points/3)).points)
        idx = rng.choice(len(self.biased), size=int(num_points/3))
        samples = np.concatenate([pcd, self.biased[idx]], axis=0)
        return samples
    
    def sample_close_to_surface(self, num_points, rng: np.random.Generator) -> np.ndarray:
        sharp_sample = self.sample_sharp(rng)

        surface_sample = np.asarray(self.mesh.sample_points_uniformly(number_of_points=num_points).points)
        surface_sample += rng.uniform(-0.005, 0.005, size=surface_sample.shape)

        k = int(num_points/3)

        if len(sharp_sample) <= k:
            biased = sharp_sample
        else:
            idx = rng.choice(len(sharp_sample), size=k, replace=False)
            biased = sharp_sample[idx]

        remaining = num_points - len(biased)
        idx = rng.choice(len(surface_sample), size=remaining, replace=False)
        unbiased = surface_sample[idx]

        return np.concatenate([biased, unbiased], axis=0)
    
    def sample_sharp(self, rng: np.random.Generator) -> np.ndarray:
        return self.biased + rng.uniform(low=-0.005, high=0.005, size=(len(self.biased), 3)) # add noise (forces zero crossing)
        
    def laplace_magnitude(self) -> np.ndarray:
        adj_list = self.mesh.compute_adjacency_list().adjacency_list
        means = np.asarray([np.mean(self.vertices[np.asarray(list(neighbours))], axis=0) for neighbours in adj_list])
        laplace_mag = np.linalg.norm(self.mesh.vertices - means, axis=1)
        return laplace_mag

    def sample_surface_normals(self, points: np.ndarray) -> np.ndarray:
        # Compute vertex normals if not already computed
        
        sampled_normals = []
        for pt in points:
            [_, idx, _] = self.kdtree.search_knn_vector_3d(o3d.utility.Vector3dVector([pt])[0], 1)
            sampled_normals.append(self.mesh.vertex_normals[idx[0]])

        return np.array(sampled_normals)
    
    def sample_global(self, num, rng):
        return rng.uniform(-1, 1, size=(num, 3))
            
                