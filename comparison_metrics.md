## Comparison metrics to use when comparing pruning methods

#### Important: keeping training arguments equal, fixing seed, applying deterministic sampling, pruning after same amount of epochs

- Intersection over Union (IoU) / Volumetric IoU: Measures the overlap between the generated 3D shape and the target shape, often computed on a voxelized grid.

- Chamfer Distance (CD): Measures the distance between a set of points sampled from the generated mesh and a set of points sampled from the target mesh. Lower is better.