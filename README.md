## PfHT1
Title of Manuscript.

---

## Overview

This repository accompanies our PfHT1 molecular mechanism study and contains:
- Graph Attention Network (GAT) model
- Transition-path theory (TPT), pathway fingerprinting, and VAE model scripts
- preprocessing scripts for graph and transition-path generation
- pretrained model weights

Large datasets including molecular dynamics trajectories, topology files, MSM data, and processed simulation datasets are hosted separately in the associated data repository.

---

<p align="center">
  <img src="figures/overview_pfht1.png" width="850">
</p>

---

## Repository Contents

```text
PfHT1/
├── preprocessing/
│   ├── make_input_graphs.py
│   └── tpt_fingerprints.py
├── GAT_Model/
├── VAE_Model/
│   ├── analysis_config.example.json
│   ├── train_vae.py
│   └── cluster_pathways.py
├── requirements.txt
└── figures/
```

---

## Workflow

Generate graph inputs:

```bash
python make_input_graphs.py \
    --trajectory traj.xtc \
    --topology system.prmtop
```

The notebook `gat_model.ipynb` contains:
- model architecture
- training workflow
- inference and evaluation

Pretrained weights:
```text
pfht1_gat_weights.pt
```

### TPT--VAE pathway analysis

Copy `VAE_Model/analysis_config.example.json`, supply the apo and holo MSM,
discrete-trajectory, feature, and tICA input paths, and run:

```bash
python preprocessing/tpt_fingerprints.py \
    --config VAE_Model/analysis_config.json

python VAE_Model/train_vae.py \
    --system apo --seed 167 \
    --fingerprints outputs/fingerprints/apo.npz \
    --output outputs/models/apo

python VAE_Model/train_vae.py \
    --system holo --seed 239 \
    --fingerprints outputs/fingerprints/holo.npz \
    --output outputs/models/holo

python VAE_Model/cluster_pathways.py \
    --embedding outputs/models/apo/embedding.npz \
    --fingerprints outputs/fingerprints/apo.npz \
    --output outputs/clusters/apo
```

Repeat the clustering command with the holo files. Generated TPT paths,
fingerprints, trained models, and clustering outputs are not tracked in Git.

---

## Additional Data

The following files are hosted separately due to size limitations:
- MD trajectories
- topology (`.prmtop`) files
- MSM datasets
- processed graph datasets
- framewise feature and tICA datasets used by the TPT--VAE analysis

Data repository:
[DATA REPOSITORY LINK]

---

## Citation

If you use this repository, please cite:

```text
[PAPER CITATION]
```

---
