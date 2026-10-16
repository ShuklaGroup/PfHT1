#!/usr/bin/env python3
"""Calculate the transport coordinates used for the PfHT1 MSMs.

One implementation handles apo and glucose-bound trajectories. Passing
``--ligand-resname GLC`` adds the glucose C2 z coordinate and C2-C5
orientation used for the bound data; omit it for apo trajectories.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mdtraj as md
import numpy as np


# MDTraj residue indices are zero based; add one for manuscript residue numbers.
CONTACT_PAIRS = (
    (181, 335), (74, 451), (89, 433), (86, 440), (177, 339),
    (79, 444), (75, 444), (82, 440), (357, 417), (165, 413),
    (314, 400), (314, 176), (150, 428), (47, 314), (152, 416),
)
ANGLE_GROUPS = ((306, 307, 308), (313, 314, 315), (321, 322, 323))


def center_of_mass(xyz: np.ndarray, topology: md.Topology,
                   residue_indices: tuple[int, ...]) -> np.ndarray:
    atoms = [atom for residue_index in residue_indices
             for atom in topology.residue(residue_index).atoms]
    indices = np.asarray([atom.index for atom in atoms], dtype=int)
    masses = np.asarray([atom.element.mass for atom in atoms], dtype=float)
    return np.average(xyz[:, indices, :], axis=1, weights=masses)


def calculate(chunk: md.Trajectory, ligand_resname: str | None) -> np.ndarray:
    distances, _ = md.compute_contacts(
        chunk, contacts=list(CONTACT_PAIRS), scheme="closest-heavy"
    )
    first, vertex, third = (
        center_of_mass(chunk.xyz, chunk.topology, group) for group in ANGLE_GROUPS
    )
    vector_a = first - vertex
    vector_c = third - vertex
    cosine = np.sum(vector_a * vector_c, axis=1) / (
        np.linalg.norm(vector_a, axis=1) * np.linalg.norm(vector_c, axis=1)
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))[:, None]
    columns = [distances]
    if ligand_resname:
        c2 = chunk.topology.select(f"resname {ligand_resname} and name C2")
        c5 = chunk.topology.select(f"resname {ligand_resname} and name C5")
        if len(c2) != 1 or len(c5) != 1:
            raise ValueError(
                f"Expected one C2 and one C5 atom for {ligand_resname}; "
                f"found {len(c2)} and {len(c5)}"
            )
        c2_xyz = chunk.xyz[:, int(c2[0]), :]
        c5_xyz = chunk.xyz[:, int(c5[0]), :]
        c2_z = c2_xyz[:, 2:3]
        direction = c5_xyz - c2_xyz
        glucose_angle = np.degrees(np.arccos(np.clip(
            direction[:, 2] / np.linalg.norm(direction, axis=1), -1.0, 1.0
        )))[:, None]
        columns.extend((c2_z, angle, glucose_angle))
    else:
        columns.append(angle)
    return np.concatenate(columns, axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--trajectories", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ligand-resname", default=None)
    parser.add_argument("--chunk", type=int, default=1000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = []
    feature_trajectories = []
    for trajectory in args.trajectories:
        parts = [calculate(chunk, args.ligand_resname) for chunk in md.iterload(
            str(trajectory), top=str(args.topology), chunk=args.chunk
        )]
        values = np.concatenate(parts)
        feature_trajectories.append(values[np.isfinite(values).all(axis=1)])
        output = args.output / f"{trajectory.stem}_distance_features_new.npy"
        np.save(output, values)
        outputs.append({"trajectory": str(trajectory.resolve()),
                        "feature_file": str(output.resolve()),
                        "frames": int(len(values)),
                        "finite_frames": int(np.isfinite(values).all(axis=1).sum()),
                        "columns": int(values.shape[1])})

    feature_pickle = args.output / "transport_feature_trajectories.pkl"
    with feature_pickle.open("wb") as handle:
        pickle.dump(feature_trajectories, handle)

    (args.output / "feature_metadata.json").write_text(json.dumps({
        "topology": str(args.topology.resolve()),
        "contact_pairs_zero_based": CONTACT_PAIRS,
        "contact_pairs_biological_residue_numbers": [
            [first + 1, second + 1] for first, second in CONTACT_PAIRS
        ],
        "tm7b_angle_groups_zero_based": ANGLE_GROUPS,
        "ligand_resname": args.ligand_resname,
        "feature_pickle": str(feature_pickle.resolve()),
        "column_order": (
            "15 closest-heavy distances, glucose C2 z, TM7b angle, glucose C2-C5 angle"
            if args.ligand_resname else
            "15 closest-heavy distances, TM7b angle"
        ),
        "outputs": outputs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
