import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("data/pointclouds/Armadillo/Armadillo.ply")

points = np.asarray(pcd.points)   # (N, 3)
normals = np.asarray(pcd.normals) # may be empty

