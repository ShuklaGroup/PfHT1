#!/usr/bin/env python3
"""Measure central TM7b helix cracking in all retained apo and holo frames."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import mdtraj as md
import numpy as np


from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
SYSTEMS = {
    "apo": (
        OUT / "apo_selected_full.xtc",
        (ROOT / "../../apo_pfht1/PfHT1_Real_1_96x96y122z_HMR_210330-strip.parm7").resolve(),
    ),
    "holo": (
        OUT / "holo_selected_full.xtc",
        (ROOT / "../../holo_pfht1/PfHT1_OF_with_glc_mix_memb-strip.prmtop").resolve(),
    ),
}
PAIR_STARTS = (313, 314, 315)
HO_CUTOFF_NM = 0.25
NHO_ANGLE_CUTOFF_DEG = 120.0
BOOTSTRAPS = 5000
SEED = 29092026


def atom(topology: md.Topology, biological_residue: int, name: str) -> int:
    residue = topology.residue(biological_residue - 1)
    matches = [candidate.index for candidate in residue.atoms if candidate.name == name]
    if len(matches) != 1:
        raise ValueError((biological_residue, name, matches))
    return matches[0]


def calculate(trajectory: Path, topology_path: Path) -> np.ndarray:
    """Return frame x bond Boolean array for the three central i->i+4 bonds."""
    topology = md.load_topology(str(topology_path))
    triplets = []
    keep = set()
    for start in PAIR_STARTS:
        triplet = (
            atom(topology, start + 4, "N"),
            atom(topology, start + 4, "H"),
            atom(topology, start, "O"),
        )
        triplets.append(triplet)
        keep.update(triplet)
    selected = np.asarray(sorted(keep), dtype=np.int32)
    lookup = {int(original): local for local, original in enumerate(selected)}
    local = np.asarray([[lookup[index] for index in triplet] for triplet in triplets])

    parts = []
    for chunk in md.iterload(
        str(trajectory), top=str(topology_path), chunk=1000, atom_indices=selected
    ):
        xyz = chunk.xyz
        donor_n = xyz[:, local[:, 0], :]
        donor_h = xyz[:, local[:, 1], :]
        acceptor_o = xyz[:, local[:, 2], :]
        h_to_o = acceptor_o - donor_h
        h_to_n = donor_n - donor_h
        distance = np.linalg.norm(h_to_o, axis=2)
        cosine = np.sum(h_to_n * h_to_o, axis=2) / (
            np.linalg.norm(h_to_n, axis=2) * np.linalg.norm(h_to_o, axis=2)
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        parts.append((distance < HO_CUTOFF_NM) & (angle > NHO_ANGLE_CUTOFF_DEG))
    return np.concatenate(parts)


def block_indices(records: list[dict[str, str]]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["trajectory_slot"]].append(index)
    return [np.asarray(indices, dtype=int) for indices in groups.values()]


def block_bootstrap_fraction(
    values: np.ndarray, blocks: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    samples = np.empty(BOOTSTRAPS, dtype=float)
    for bootstrap in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        positive = 0
        total = 0
        for block in chosen:
            indices = blocks[int(block)]
            positive += int(values[indices].sum())
            total += len(indices)
        samples[bootstrap] = positive / total
    return samples


def main() -> None:
    records = list(csv.DictReader((OUT / "selected_frame_provenance.csv").open()))
    condition_records = {"apo": records[:10000], "holo": records[10000:]}
    arrays = {}
    bootstrap = {}
    summary = {
        "scope": "all 10000 MSM-drawn frames per condition; no pocket-cluster filter",
        "bond_definition": (
            "An i->i+4 backbone hydrogen bond is formed when H(i+4)-O(i) < 2.5 A "
            "and angle N(i+4)-H(i+4)-O(i) > 120 degrees."
        ),
        "central_pairs": [f"{start}->{start + 4}" for start in PAIR_STARTS],
        "per_frame_metric": "100 times the number of absent central bonds divided by 3",
        "binary_crack_definition": "at least two of the three central bonds are absent",
        "systems": {},
    }
    rng = np.random.default_rng(SEED)
    for condition, (trajectory, topology) in SYSTEMS.items():
        formed = calculate(trajectory, topology)
        if formed.shape != (10000, 3):
            raise ValueError(f"Unexpected {condition} array shape: {formed.shape}")
        absent_count = (~formed).sum(axis=1)
        disrupted_percent = absent_count / 3.0 * 100.0
        cracked = absent_count >= 2
        samples = block_bootstrap_fraction(
            cracked, block_indices(condition_records[condition]), rng
        )
        arrays[f"{condition}_formed"] = formed
        arrays[f"{condition}_disrupted_percent"] = disrupted_percent
        arrays[f"{condition}_cracked"] = cracked
        bootstrap[condition] = samples
        unique, counts = np.unique(disrupted_percent, return_counts=True)
        summary["systems"][condition] = {
            "frames": int(len(formed)),
            "trajectory_slots": int(len(block_indices(condition_records[condition]))),
            "pair_occupancy": formed.mean(axis=0).tolist(),
            "mean_disrupted_percent": float(disrupted_percent.mean()),
            "median_disrupted_percent": float(np.median(disrupted_percent)),
            "disrupted_percent_counts": {
                f"{value:.6g}": int(count) for value, count in zip(unique, counts)
            },
            "cracked_frames": int(cracked.sum()),
            "cracked_fraction": float(cracked.mean()),
            "cracked_percent": float(cracked.mean() * 100.0),
            "trajectory_slot_bootstrap_ci95_percent": (
                np.quantile(samples, (0.025, 0.975)) * 100.0
            ).tolist(),
        }

    difference = (bootstrap["holo"] - bootstrap["apo"]) * 100.0
    summary["holo_minus_apo_cracked_percentage_points"] = {
        "estimate": float(
            (arrays["holo_cracked"].mean() - arrays["apo_cracked"].mean()) * 100.0
        ),
        "bootstrap_ci95": np.quantile(difference, (0.025, 0.975)).tolist(),
    }
    np.savez_compressed(
        OUT / "tm7b_cracking_all_frames.npz",
        pair_starts=np.asarray(PAIR_STARTS, dtype=int),
        **arrays,
    )
    (OUT / "tm7b_cracking_all_frames_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
