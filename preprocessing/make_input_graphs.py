"""Build the processed PyTorch-Geometric dataset used by the PfHT1 GAT."""

from __future__ import annotations

import argparse
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def build_graphs(
    trajectory: Path,
    topology: Path,
    residue_ligand_cutoff_nm: float = 0.8,
    residue_residue_cutoff_nm: float = 0.8,
    decay_length_angstrom: float = 5.0,
) -> list[Data]:
    """Convert trajectory frames into residue--ligand graphs."""
    traj = md.load(str(trajectory), top=str(topology))

    protein_atoms = traj.top.select("protein and not element H")
    ligand_atoms = traj.top.select("resname GLC and not element H")
    ca_atoms = traj.top.select("protein and name CA")
    protein_atom_set = set(map(int, protein_atoms))
    ca_atom_set = set(map(int, ca_atoms))

    protein_residues = [
        residue
        for residue in traj.top.residues
        if any(atom.index in protein_atom_set for atom in residue.atoms)
    ]
    residue_atoms = {
        residue.index: [
            atom.index for atom in residue.atoms if atom.index in protein_atom_set
        ]
        for residue in protein_residues
    }
    residue_ca = {
        residue.index: next(
            atom.index for atom in residue.atoms if atom.index in ca_atom_set
        )
        for residue in protein_residues
    }
    node_mapping = {
        residue.index: index for index, residue in enumerate(protein_residues)
    }
    ligand_node = len(protein_residues)

    graphs: list[Data] = []
    for frame in tqdm(traj, desc="Building graphs"):
        coordinates = frame.xyz[0]
        residue_positions = {
            residue.index: coordinates[residue_atoms[residue.index]].mean(axis=0)
            for residue in protein_residues
        }
        protein_com = coordinates[protein_atoms].mean(axis=0)
        ligand_position = coordinates[ligand_atoms].mean(axis=0)

        edge_list: list[tuple[int, int]] = []
        edge_weights: list[float] = []

        for residue in protein_residues:
            distance_nm = np.linalg.norm(
                coordinates[residue_ca[residue.index]] - ligand_position
            )
            if distance_nm < residue_ligand_cutoff_nm:
                weight = np.exp(-(distance_nm * 10.0) / decay_length_angstrom)
                residue_node = node_mapping[residue.index]
                edge_list.extend(
                    [(residue_node, ligand_node), (ligand_node, residue_node)]
                )
                edge_weights.extend([weight, weight])

        for index, residue_i in enumerate(protein_residues):
            ca_i = residue_ca[residue_i.index]
            for residue_j in protein_residues[index + 1 :]:
                ca_j = residue_ca[residue_j.index]
                distance_nm = np.linalg.norm(coordinates[ca_i] - coordinates[ca_j])
                if distance_nm < residue_residue_cutoff_nm:
                    weight = np.exp(-(distance_nm * 10.0) / decay_length_angstrom)
                    node_i = node_mapping[residue_i.index]
                    node_j = node_mapping[residue_j.index]
                    edge_list.extend([(node_i, node_j), (node_j, node_i)])
                    edge_weights.extend([weight, weight])

        residue_features = torch.tensor(
            [
                residue_positions[residue.index] - protein_com
                for residue in protein_residues
            ],
            dtype=torch.float32,
        )
        node_features = torch.cat(
            [residue_features, torch.zeros((1, 3), dtype=torch.float32)], dim=0
        )
        graphs.append(
            Data(
                x=node_features,
                edge_index=torch.tensor(edge_list, dtype=torch.long)
                .t()
                .contiguous(),
                edge_attr=torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1),
                y=torch.tensor([ligand_position[2]], dtype=torch.float32),
                residue_ids=torch.tensor(
                    [residue.index for residue in protein_residues] + [-1],
                    dtype=torch.long,
                ),
                protein_com=torch.tensor(protein_com, dtype=torch.float32),
                ligand_position=torch.tensor(ligand_position, dtype=torch.float32),
            )
        )
    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ligand_residue_allosteric_graphs_proteinCOM.pt"),
    )
    parser.add_argument("--residue-ligand-cutoff-nm", type=float, default=0.8)
    parser.add_argument("--residue-residue-cutoff-nm", type=float, default=0.8)
    parser.add_argument("--decay-length-angstrom", type=float, default=5.0)
    args = parser.parse_args()

    graphs = build_graphs(
        args.trajectory,
        args.topology,
        args.residue_ligand_cutoff_nm,
        args.residue_residue_cutoff_nm,
        args.decay_length_angstrom,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, args.output)
    print(f"Saved {len(graphs)} graphs to {args.output}")


if __name__ == "__main__":
    main()
