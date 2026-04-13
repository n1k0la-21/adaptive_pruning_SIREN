## Usage

A notebook is provided to give an insight into how I used the framework for my thesis. The results used have been extracted using 
'sweep.py'.

To run `sweep.py` (script for training, extracting meshes, and saving metrics), create an environment either through:

### Using `venv`

```bash
python -m venv pruning_sdf_nikola
source pruning_sdf_nikola/bin/activate
pip install -r requirements.txt
```

---

### Using `conda`

```bash
conda create -n pruning_sdf_nikola python=3.11
conda activate pruning_sdf_nikola
pip install -r requirements.txt
```

Furthermore, the script requires the ground truth meshes, which are provided through this Google Drive link: 
https://drive.google.com/file/d/1Oq6JcS22T3NBxRTXRuSi_EkyywBtehEQ/view?usp=drive_link

So what is left is to just download the data, extract it, and place it in the directory as follows:

```
adaptive_pruning_SIREN/
└── **data/**
|      └── pointclouds/
|            ├── lucy/
|            │   └── Stanford_lucy.ply
|            ├── armadillo/
|            │   └── Stanford_armadillo.ply
|            ├── bunny/
|            │   └── Stanford_bunny.ply
|            └── dragon/
|                └── Stanford_dragon.ply
└── notebooks/
└── src/
etc. 
```

After running the script, you will end up with each models weights, extracted meshes (256^3 in resolution) and a 
summary table of all the computed metrics post-training. Training is done with 3 seeds, with the loss history being tracked
during training. All files are hierarchically structured by {mesh}_data/seed_{seed}/ with the loss histories being saved in the
history folder. 

Such a folder structure (example for bunny, seed 42, pruning ratio 0.30) will look as follows:
```
bunny_data/
└── seed_42/
|      └── history/
|      |     ├── large_unpruned_history.npz
|      |     ├── densified_history.npz
|      |     ├── AIRe_0.3_history.npz
|      |     └── ...
|      └── large_unpruned.pth
|      └── large_unpruned.obj // mesh file
|      └── densified.pth
|      └── densified.obj
|      └── ...
└── seed_43/
└── seed_44/
```

The pruning ratios are [0.30, 0.50, 0.70, 0.85, 0.95] so the script entails 4x3x(2 + 4x5) = 264 runs.
