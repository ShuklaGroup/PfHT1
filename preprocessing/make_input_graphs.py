import mdtraj as md
import numpy as np
import networkx as nx
import os
import math
import pickle
import random
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from tqdm import tqdm
from scipy.stats import spearmanr

res_lig_cutoff = 0.8   # nm cutoff for ligand–residue edges
res_res_cutoff = 0.8   # nm cutoff for residue–residue edges
lambda_val = 5         # Å decay parameter
epsilon = 1e-8
graphs = []


traj = md.load(
    "/media/arnav_ds04/PfHT1/holo_pfht1/extract_states/filtered_trajectory.xtc",
    top="/media/arnav_ds04/PfHT1/holo_pfht1/PfHT1_OF_with_glc_mix_memb-strip.prmtop"
)

protein_atoms = traj.top.select("protein and not element H")
ligand_atoms = traj.top.select("resname GLC and not element H")
ca_atoms = traj.top.select("protein and name CA")

protein_residues = [
    res for res in traj.top.residues if any(a.index in protein_atoms for a in res.atoms)
]

for frame_idx, frame in enumerate(tqdm(traj, desc="Building graphs")):
    residue_positions = {}

    # Compute residue COM positions
    for res in protein_residues:
        res_atoms = [a.index for a in res.atoms if a.index in protein_atoms]
        if res_atoms:
            residue_positions[res.index] = frame.xyz[0, res_atoms, :].mean(axis=0)

    # Protein COM (for centering)
    protein_com = np.mean(frame.xyz[0, protein_atoms, :], axis=0)

    # Ligand COM (only for edge building, NOT features)
    ligand_pos = np.mean(frame.xyz[0, ligand_atoms, :], axis=0)

    # z-coordinate target
    ligand_z = ligand_pos[2]

    # node mapping: residues + ligand
    node_mapping = {res.index: idx for idx, res in enumerate(protein_residues)}
    ligand_node_idx = len(protein_residues)
    node_mapping["ligand"] = ligand_node_idx

    edge_list = []
    edge_weights = []

    # ---- Ligand–residue edges (Cα cutoff) ----
    for res in protein_residues:
        ca_index = [a.index for a in res.atoms if a.index in ca_atoms][0]
        dist_nm = np.linalg.norm(frame.xyz[0, ca_index, :] - ligand_pos)
        if dist_nm < res_lig_cutoff:
            weight = np.exp(-(dist_nm * 10) / lambda_val)  # Å
            res_idx = node_mapping[res.index]
            edge_list.extend([(res_idx, ligand_node_idx), (ligand_node_idx, res_idx)])
            edge_weights.extend([weight, weight])

    # ---- Residue–residue edges (Cα–Cα cutoff) ----
    ca_positions = frame.xyz[0, ca_atoms, :]
    for i, res_i in enumerate(protein_residues):
        for j, res_j in enumerate(protein_residues):
            if j <= i:
                continue
            ca_i = [a.index for a in res_i.atoms if a.index in ca_atoms][0]
            ca_j = [a.index for a in res_j.atoms if a.index in ca_atoms][0]
            dist_nm = np.linalg.norm(frame.xyz[0, ca_i, :] - frame.xyz[0, ca_j, :])
            if dist_nm < res_res_cutoff:
                weight = np.exp(-(dist_nm * 10) / lambda_val)
                idx_i = node_mapping[res_i.index]
                idx_j = node_mapping[res_j.index]
                edge_list.extend([(idx_i, idx_j), (idx_j, idx_i)])
                edge_weights.extend([weight, weight])

    # ---- Node features (protein COM-centered absolute positions) ----
    x_res = torch.tensor(
        [residue_positions[res.index] - protein_com for res in protein_residues],
        dtype=torch.float32
    )
    x_lig = torch.zeros((1, 3), dtype=torch.float32)  # Ligand node has no info
    x = torch.cat([x_res, x_lig], dim=0)

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1),
        y=torch.tensor([ligand_z], dtype=torch.float32),
        residue_ids=torch.tensor([res.index for res in protein_residues] + [-1], dtype=torch.long),
        protein_com=torch.tensor(protein_com, dtype=torch.float32),
        ligand_position=torch.tensor(ligand_pos, dtype=torch.float32)
    )
    graphs.append(data)

# ===================== SAVE =====================
out_file = "ligand_residue_allosteric_graphs_proteinCOM.pt"
torch.save(graphs, out_file)
print(f"Saved {len(graphs)} graphs to {out_file}")

