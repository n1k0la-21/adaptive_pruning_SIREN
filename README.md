## Usage

To run `sweep.py` (script for training, extracting meshes and saving metrics), create an environment either through:

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

Furthermore, the script requires the ground truth meshes, they are provided through this google drive link: 
https://drive.google.com/file/d/1Oq6JcS22T3NBxRTXRuSi_EkyywBtehEQ/view?usp=drive_link

So what is left is to just download the data, extract it and place it in the directory as follows:

adaptive_pruning_SIREN/
└── **data/**
|      └── pointclouds/
|            ├── lucy/
|            │     └── Stanford_lucy.ply
|            ├── armadillo/
|            │   └── Stanford_armadillo.ply
|            ├── bunny/
|            │   └── Stanford_bunny.ply
|            └── dragon/
|            └── Stanford_dragon.ply
└── notebooks/
└── src/
etc. 

