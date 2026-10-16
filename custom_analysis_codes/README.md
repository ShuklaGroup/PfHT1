# PfHT1 custom analysis code

This folder complements the existing PfHT1 GitHub repository. It contains the
custom analysis used for the MSM, free-energy, residue-contact, yeast-growth,
and binding-pocket/TM7b results that is not already present in that repository.
The existing GAT, VAE, transition-path, and graph-preprocessing programs are
therefore not duplicated here.

Each workflow has one condition-independent implementation. Apo and
glucose-bound data are supplied through command-line inputs or a shared project
layout; there are no separate apo and holo copies of the same featurization,
MSM, or plotting code.

## Contents

- `msm_kinetics/`: common transport-coordinate featurization, tICA,
  clustering, MSM estimation, trajectory bootstrap, and macrostate MFPTs.
- `free_energy_landscapes/`: MSM-weighted two-dimensional free-energy
  calculation and plotting.
- `binding_pocket_tm7b/`: fresh MSM-weighted frame sampling; pocket
  featurization; apo-reference PCA and clustering; gate, TM7b cracking, and
  two-state equilibrium energetic analyses; final quantitative plotting.
- `residue_contacts/`: one implementation for RRCS matrices and the
  difference between two conformations.
- `yeast_growth/`: pooled-control normalization summaries and growth plots.

The scripts do not bundle trajectories, topology files, fitted models, or
intermediate arrays. Those inputs remain in the associated shared-data
directories.

## Environment

The analyses used Python 3 with the packages listed in `requirements.txt`.
The saved MSM objects require the same `deeptime` version used to create them.

For the pocket workflow, set the project root if the scripts cannot infer it:

```bash
export PFHT1_PROJECT_ROOT=/path/to/PfHT1
```

The expected default result directory is
`$PFHT1_PROJECT_ROOT/analysis/binding_pocket`. It can be overridden with
`PFHT1_BINDING_POCKET_DIR`.

## MSM workflow

The transport feature script is shared by apo and holo data. The optional
ligand argument adds the glucose C2 z coordinate and C2-C5 orientation for
centered holo trajectories:

```bash
python msm_kinetics/featurize_transport_coordinates.py \
  --topology system.parm7 --trajectories traj1.xtc traj2.xtc \
  --output features

python msm_kinetics/featurize_transport_coordinates.py \
  --topology holo.prmtop --trajectories holo1.xtc holo2.xtc \
  --ligand-resname GLC --output holo_features
```

The final apo and glucose-bound MSMs used the same model-building script with
different selected hyperparameters:

```bash
# Apo: 65% retained tICA variance and 1700 microstate clusters
python msm_kinetics/build_msm.py --features apo_features.pkl --output apo_msm \
  --label apo --variance-cutoff 0.65 --clusters 1700 --tica-lag 200 --msm-lag 200

# Glucose-bound: 80% retained tICA variance and 900 microstate clusters
python msm_kinetics/build_msm.py --features holo_features.pkl --output holo_msm \
  --label holo --variance-cutoff 0.80 --clusters 900 --tica-lag 200 --msm-lag 200
```

`macrostate_mfpt.py` accepts the transport-coordinate column numbers and a
JSON file of state ranges. Verify these against the feature ordering used for
the fitted model; `state_ranges.example.json` records the manuscript ranges.

## Binding-pocket and TM7b workflow

The retained manuscript analysis used 10,000 freshly drawn apo frames and
10,000 contact-qualified glucose-bound frames. MSM states were drawn according
to their stationary probabilities, with frames sampled within states using the
saved microstate populations. The exact original trajectory and frame are
recorded for every draw.

Pocket geometry is defined once in `pocket_features.py` using residues 48,
168, 169, 305, 306, 311, 341, 412, 416, 432, and 435. The descriptor contains
55 pairwise side-chain heavy-atom-center distances and 55 pairwise C-alpha
distances. Gate distances and the TM7b angle are annotations rather than
clustering inputs. Glucose-bound frames require contacts with at least three
of the eight core contact residues and a glucose-to-pocket center distance
below 0.85 nm.

Run the scripts from `binding_pocket_tm7b/` in this order:

1. `map_msm_feature_rows.py apo` and `map_msm_feature_rows.py holo`
2. `draw_compact_msm_manifest.py`
3. `extract_fresh_pockets.py apo` and `extract_fresh_pockets.py holo`
4. `save_full_selected_xtcs.py apo` and `save_full_selected_xtcs.py holo`
5. `cluster_fresh_20000.py`
6. `apo_reference_pca.py`
7. `cluster_apo_reference_pcs.py`
8. `analyze_gate_by_apo_pc_cluster.py`
9. `analyze_tm7b_cracking_all_frames.py`
10. `analyze_tm7b_cracking_energetics.py`
11. `make_manuscript_pocket_figure.py`

`cluster_fresh_20000.py` prepares the common 20,000-frame arrays and the joint
diagnostic clustering. The manuscript result is produced by the following
apo-reference PCA and clustering steps, which fit scaling, PCA, and clusters
to apo pockets and only then project and assign glucose-bound pockets.

TM7b cracking is classified from the central backbone i,i+4 hydrogen-bond
network over residues 313–319. A frame is cracked when at least two of the
three central hydrogen bonds are absent. The energetic script reports
`DeltaG(cracked - intact)` from the MSM-weighted two-state populations at
310 K; it does not estimate an activation barrier.

## Reproducibility notes

- Random seeds and analysis constants are recorded in the scripts and output
  metadata.
- Sampling outputs preserve original trajectory/frame provenance.
- `MANIFEST.sha256` records the contents of this code folder.
- No license is assigned in this archival folder; apply the repository license
  when these files are added to the public code repository.
