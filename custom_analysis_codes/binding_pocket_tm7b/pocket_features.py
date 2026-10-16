"""Shared PfHT1 pocket-feature and glucose-contact definitions."""
from itertools import combinations

import mdtraj as md
import numpy as np


CORE_RESSEQ = (48, 168, 169, 305, 306, 311, 341, 412, 416, 432, 435)
CONTACT_RESSEQ = (168, 169, 305, 306, 311, 341, 412, 435)
GATE_RESSEQ = (45, 49, 145, 315, 316, 317, 412, 416)
CONTACT_CUTOFF_NM = 0.45
MAX_GLC_CENTER_TO_CORE_NM = 0.85
MIN_CONTACT_RESIDUES = 3


def protein_residue(top: md.Topology, resseq: int):
    return top.residue(resseq - 1)


def atom_map(top: md.Topology):
    groups = {}
    keep = set()
    needed = CORE_RESSEQ + CONTACT_RESSEQ + GATE_RESSEQ + (
        47, 152, 307, 308, 309, 314, 322, 409, 419,
    )
    for number in sorted(set(needed)):
        residue = protein_residue(top, number)
        ca = [a.index for a in residue.atoms if a.name == "CA"]
        side = [
            a.index for a in residue.atoms
            if a.element.symbol not in ("H", "D")
            and a.name not in ("N", "CA", "C", "O", "OXT")
        ]
        if len(ca) != 1:
            raise ValueError(f"Missing CA: {residue}")
        if number in CORE_RESSEQ and not side:
            raise ValueError(f"Missing side-chain atoms: {residue}")
        groups[number] = {"ca": ca, "side": side}
        keep.update(ca + side)
    ligand = []
    glucose_residues = [r for r in top.residues if r.name == "GLC"]
    if glucose_residues:
        if len(glucose_residues) != 1:
            raise ValueError(f"Expected one GLC residue, got {len(glucose_residues)}")
        ligand = [
            a.index for a in glucose_residues[0].atoms
            if a.element.symbol not in ("H", "D")
        ]
        keep.update(ligand)
    selected = np.asarray(sorted(keep), dtype=int)
    lookup = {int(atom): i for i, atom in enumerate(selected)}
    groups = {
        number: {
            kind: np.asarray([lookup[atom] for atom in atoms], dtype=int)
            for kind, atoms in group.items()
        }
        for number, group in groups.items()
    }
    return selected, groups, np.asarray([lookup[a] for a in ligand], dtype=int)


def extract_features(chunk, groups, ligand):
    xyz = chunk.xyz.astype(np.float32)
    n_frames = len(xyz)
    ca = {r: xyz[:, g["ca"], :].mean(axis=1) for r, g in groups.items()}
    side = {
        r: xyz[:, g["side"], :].mean(axis=1)
        for r, g in groups.items() if len(g["side"])
    }
    columns, names = [], []
    for kind, coordinates in (("side", side), ("ca", ca)):
        for first, second in combinations(CORE_RESSEQ, 2):
            columns.append(np.linalg.norm(coordinates[first] - coordinates[second], axis=1))
            names.append(f"{kind}_{first}_{second}")
    features = np.column_stack(columns).astype(np.float32)
    gate_out = np.linalg.norm(
        np.mean([ca[r] for r in (45, 47, 49)], axis=0)
        - np.mean([ca[r] for r in (315, 316, 317)], axis=0), axis=1,
    )
    gate_in = np.linalg.norm(
        np.mean([ca[r] for r in (145, 152)], axis=0)
        - np.mean([ca[r] for r in (409, 419)], axis=0), axis=1,
    )
    first = np.mean([ca[r] for r in (307, 308, 309)], axis=0)
    vertex = ca[314]
    third = ca[322]
    va, vc = first - vertex, third - vertex
    denominator = np.linalg.norm(va, axis=1) * np.linalg.norm(vc, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.sum(va * vc, axis=1) / denominator, -1, 1)))
    gates = np.column_stack((gate_out, gate_in, angle)).astype(np.float32)
    if not len(ligand):
        return (features, names, gates, np.ones(n_frames, dtype=bool),
                np.zeros(n_frames, dtype=int), np.full(n_frames, np.nan))
    glucose = xyz[:, ligand, :]
    center_distance = np.linalg.norm(
        glucose.mean(axis=1) - np.mean([side[r] for r in CONTACT_RESSEQ], axis=0), axis=1,
    )
    contact_count = np.zeros(n_frames, dtype=np.int16)
    for residue in CONTACT_RESSEQ:
        atoms = xyz[:, groups[residue]["side"], :]
        distances = np.linalg.norm(
            glucose[:, :, None, :] - atoms[:, None, :, :], axis=-1,
        )
        contact_count += distances.min(axis=(1, 2)) < CONTACT_CUTOFF_NM
    bound = ((contact_count >= MIN_CONTACT_RESIDUES)
             & (center_distance < MAX_GLC_CENTER_TO_CORE_NM))
    return features, names, gates, bound, contact_count, center_distance
