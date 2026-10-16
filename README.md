## PfHT1
Substrate binding reorganizes the energetic landscape of Plasmodium falciparum hexose transporter PfHT1

---

## Overview

This repository accompanies our PfHT1 molecular mechanism study and contains:
- Graph Attention Network (GAT) model
- Transition-path theory (TPT), pathway fingerprinting, and VAE model scripts
- custom MSM, free-energy, residue-contact, binding-pocket/TM7b, and yeast-growth analyses
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
├── custom_analysis_codes/
│   ├── msm_kinetics/
│   ├── free_energy_landscapes/
│   ├── binding_pocket_tm7b/
│   ├── residue_contacts/
│   └── yeast_growth/
├── requirements.txt
└── figures/
```

---

## Workflow

Generate graph inputs:

```bash
export PFHT1_DATA_DIR=/path/to/shared_data

python preprocessing/make_input_graphs.py \
    --trajectory "$PFHT1_DATA_DIR/GAT/filtered_trajectory.xtc" \
    --topology "$PFHT1_DATA_DIR/topology/PfHT1_OF_with_glc_mix_memb-strip.prmtop" \
    --output "$PFHT1_DATA_DIR/GAT/processed_graphs/ligand_residue_allosteric_graphs_proteinCOM.pt"
```

The notebook `GAT_Model/gat_interpretable-residue_importance.ipynb` reads the
same data location from `PFHT1_DATA_DIR` and contains:
- model architecture
- training workflow
- inference and evaluation

Pretrained weights:
```text
GAT_Model/best_model.pt
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

### Custom molecular-simulation and experimental analyses

The `custom_analysis_codes/` directory contains the remaining custom analysis
used in the manuscript. Apo and glucose-bound calculations share the same
configurable scripts instead of maintaining condition-specific copies. The
included workflows cover:

- transport-coordinate featurization, tICA, MSM construction, bootstrapping,
  and macrostate MFPTs
- MSM-weighted free-energy landscapes
- fresh MSM-weighted sampling and apo-reference clustering of binding-pocket
  geometries
- TM7b hydrogen-bond cracking and the corresponding two-state equilibrium
  free-energy difference
- residue-residue contact-score differences
- pooled yeast-growth analysis

See [`custom_analysis_codes/README.md`](custom_analysis_codes/README.md) for
inputs, analysis definitions, parameter values, and execution order.

---

## Additional Data

The following files are hosted separately due to size limitations:
- filtered glucose-transport trajectory used for GAT graph preprocessing
- topology (`.prmtop`) files
- MSM datasets
- processed graph datasets
- framewise feature and tICA datasets used by the TPT--VAE analysis
- trajectories, MSM models, frame-provenance tables, and intermediate arrays used by the custom analyses

Data repository:
(https://uofi.box.com/s/i5l1juaejny6m4bet8ybkmpyp5xqrdfs)

---

## Citation

If you use this repository, please cite:
Paul, A., Xu, L., & Shukla, D. (2026). Substrate binding reorganizes the energetic landscape of Plasmodium falciparum hexose transporter PfHT1. bioRxiv, 2026-09

---
