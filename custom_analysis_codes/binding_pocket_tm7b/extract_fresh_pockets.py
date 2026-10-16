#!/usr/bin/env python3
"""Measure pocket geometry on fresh MSM draws from their original XTC files."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import time

import mdtraj as md
import numpy as np

from pocket_features import atom_map, extract_features

from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
TOPOLOGIES = {
    "apo": ROOT / "../../apo_pfht1/PfHT1_Real_1_96x96y122z_HMR_210330-strip.parm7",
    "holo": ROOT / "../../holo_pfht1/PfHT1_OF_with_glc_mix_memb.prmtop",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("apo", "holo"))
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sidechain-only", action="store_true")
    args = parser.parse_args()
    condition = args.condition
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    draw = list(csv.DictReader((out / f"{condition}_draw.csv").open()))
    if args.candidates is not None:
        draw = draw[:args.candidates]
    top = md.load_topology(str(TOPOLOGIES[condition]))
    atoms, groups, ligand = atom_map(top)
    n = len(draw)
    n_features = 55 if args.sidechain_only else 110
    features = np.empty((n, n_features), dtype=np.float32)
    gates = np.empty((n, 3), dtype=np.float32)
    bound = np.empty(n, dtype=bool)
    contacts = np.empty(n, dtype=np.int16)
    centers = np.empty(n, dtype=np.float32)
    sources = defaultdict(list)
    for i, rec in enumerate(draw):
        sources[rec["source"]].append((int(rec["frame"]), i))
    start = time.time()
    done = 0
    for file_num, (source, selections) in enumerate(sources.items(), 1):
        selections.sort()
        # A single XTC handle avoids repeated topology parsing and random-access
        # index construction for every sampled frame from the same source.
        with md.formats.XTCTrajectoryFile(source) as xtc:
            unique_frames = sorted({f for f, _ in selections})
            xyz = np.empty((len(unique_frames), len(atoms), 3), dtype=np.float32)
            for j, frame in enumerate(unique_frames):
                xtc.seek(frame)
                one, _, _, _ = xtc.read(n_frames=1, atom_indices=atoms)
                if len(one) != 1:
                    raise ValueError(f"Missing source frame {source}:{frame}")
                xyz[j] = one[0]
        f, _, g, b, c, d = extract_features(type("Chunk", (), {"xyz": xyz})(), groups, ligand)
        by_frame = {frame: j for j, frame in enumerate(unique_frames)}
        for frame, i in selections:
            j = by_frame[frame]
            features[i] = f[j, :n_features]
            gates[i] = g[j]
            bound[i] = b[j]
            contacts[i] = c[j]
            centers[i] = d[j]
        done += len(selections)
        if file_num % 20 == 0 or file_num == len(sources):
            print(f"{condition}: {done}/{n} draws, {file_num}/{len(sources)} files, "
                  f"elapsed {time.time()-start:.0f}s", flush=True)
    if condition == "holo":
        selected = np.flatnonzero(bound)[:10000]
        if len(selected) != 10000:
            raise RuntimeError(f"Only {len(selected)} pocket-bound holo frames among {n} candidates; increase --candidates")
    else:
        selected = np.arange(n)
    for i in selected:
        draw[int(i)]["contact_count"] = int(contacts[i]) if condition == "holo" else ""
        draw[int(i)]["glc_center_nm"] = float(centers[i]) if condition == "holo" else ""
    with (out / f"{condition}_selected.csv").open("w", newline="") as fh:
        fields = list(draw[0]) + ["contact_count", "glc_center_nm"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(draw[int(i)] for i in selected)
    np.savez_compressed(out / f"{condition}_pockets.npz",
                        pocket_distances_nm=features[selected], gate=gates[selected],
                        source_draw_rank=selected)
    history_fields = ("draw_rank", "condition", "source", "frame", "msm_feature_row",
                      "msm_state", "state_stationary_probability", "frame_equilibrium_weight",
                      "feature_file", "trajectory_slot", "trajectory_equilibrium_mass")
    with (out / f"{condition}_original_trajectory_history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=history_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(draw[int(i)] for i in selected)
    summary = {"condition": condition, "candidates_scanned": n,
               "bound_candidates": int(bound.sum()), "selected": len(selected),
               "distinct_source_files": len(sources), "elapsed_seconds": time.time()-start,
               "full_selected_xtc": str(out / f"{condition}_selected_full.xtc"),
               "original_trajectory_history": str(out / f"{condition}_original_trajectory_history.csv"),
               "selection": "first 10000 contact-qualified candidates in random MSM draw order" if condition == "holo" else "all 10000 MSM draws",
               "pocket_features": (
                   "55 side-chain heavy-atom-center pair distances" if args.sidechain_only
                   else "55 side-chain heavy-atom-center plus 55 C-alpha pair distances"
               )}
    (out / f"{condition}_extraction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
