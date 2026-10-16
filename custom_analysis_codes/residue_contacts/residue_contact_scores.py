#!/usr/bin/env python3
"""Calculate residue-residue contact scores (RRCS) and their difference.

The same calculation is applied to both input conformations. For residues
separated by four or fewer positions, only side-chain heavy atoms are used;
all heavy atoms are used for longer-range pairs. Pairwise atomic distances are
converted to scores of 1 below 3.23 A, 0 above 4.63 A, and a linear value
between those limits.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import mdtraj as md
import numpy as np


LOW_NM = 0.323
HIGH_NM = 0.463


def residue_atoms(topology: md.Topology, residue_index: int, sidechain: bool) -> np.ndarray:
    query = f"resid {residue_index} and mass > 2"
    if sidechain:
        query += " and not backbone"
    return topology.select(query)


def score_structure(path: Path) -> np.ndarray:
    trajectory = md.load(str(path))
    n_residues = trajectory.topology.n_residues
    matrix = np.zeros((n_residues, n_residues), dtype=np.float32)
    for first, second in combinations(range(n_residues), 2):
        sidechain_only = abs(first - second) <= 4
        atoms_first = residue_atoms(trajectory.topology, first, sidechain_only)
        atoms_second = residue_atoms(trajectory.topology, second, sidechain_only)
        if not len(atoms_first) or not len(atoms_second):
            continue
        pairs = np.asarray(list(product(atoms_first, atoms_second)), dtype=np.int32)
        distances = md.compute_distances(trajectory, pairs)
        atomic_scores = np.where(
            distances < LOW_NM,
            1.0,
            np.where(
                distances < HIGH_NM,
                1.0 - (distances - LOW_NM) / (HIGH_NM - LOW_NM),
                0.0,
            ),
        )
        value = float(atomic_scores.sum(axis=1).mean())
        matrix[first, second] = matrix[second, first] = value
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output NPZ; delta is reference minus comparison")
    args = parser.parse_args()

    reference = score_structure(args.reference)
    comparison = score_structure(args.comparison)
    if reference.shape != comparison.shape:
        raise ValueError("Input structures contain different residue counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        reference_rrcs=reference,
        comparison_rrcs=comparison,
        delta_rrcs=reference - comparison,
    )
    args.output.with_suffix(".json").write_text(json.dumps({
        "reference": str(args.reference.resolve()),
        "comparison": str(args.comparison.resolve()),
        "delta_definition": "reference minus comparison",
        "residue_count": int(reference.shape[0]),
        "distance_thresholds_A": [LOW_NM * 10.0, HIGH_NM * 10.0],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
