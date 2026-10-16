#!/usr/bin/env python3
"""Write MSM-selected full PfHT1 protein (+ holo glucose) XTC trajectories."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import tempfile
import time

import mdtraj as md
import numpy as np

from extract_fresh_pockets import OUT, TOPOLOGIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("apo", "holo"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    condition = args.condition
    out = args.out.resolve()
    selected = list(csv.DictReader((out / f"{condition}_selected.csv").open()))
    if len(selected) != 10000:
        raise ValueError("Expected exactly 10000 selected frames")
    top = md.load_topology(str(TOPOLOGIES[condition]))
    protein_atoms = list(top.select("protein"))
    glucose_atoms = ([atom.index for residue in top.residues if residue.name == "GLC"
                      for atom in residue.atoms] if condition == "holo" else [])
    atoms = np.array(sorted(set(protein_atoms + glucose_atoms)), dtype=np.int32)
    reduced_top = top.subset(atoms)
    output = out / f"{condition}_selected_full.xtc"
    topology_output = out / f"{condition}_selected_full_topology.pdb"
    sources = defaultdict(list)
    for target_frame, record in enumerate(selected):
        sources[record["source"]].append((int(record["frame"]), target_frame))
    times = np.empty(len(selected), dtype=np.float32)
    steps = np.empty(len(selected), dtype=np.int32)
    boxes = np.empty((len(selected), 3, 3), dtype=np.float32)
    start = time.time()
    # Recover after interruption between committing the complete XTC and writing
    # its small topology/summary files. Incomplete XTCs are rebuilt below.
    if output.exists():
        with md.formats.XTCTrajectoryFile(str(output)) as existing:
            existing_frames = len(existing)
            first_xyz, _, _, _ = existing.read(n_frames=1)
        if existing_frames == len(selected) and first_xyz.shape[1] == len(atoms):
            md.Trajectory(xyz=first_xyz, topology=reduced_top).save_pdb(str(topology_output))
            report = {"condition": condition, "frames": len(selected),
                      "protein_atoms": len(protein_atoms), "glucose_atoms": len(glucose_atoms),
                      "distinct_source_files": len(sources), "selected_xtc": str(output),
                      "topology_pdb": str(topology_output),
                      "original_trajectory_history": str(out / f"{condition}_original_trajectory_history.csv"),
                      "elapsed_seconds": time.time()-start, "recovered_complete_xtc": True}
            (out / f"{condition}_full_xtc_summary.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            return
        output.unlink()
    with tempfile.NamedTemporaryFile(prefix=f"{condition}_selected_xyz_", suffix=".dat", dir="/tmp") as tmp:
        tmp.truncate(len(selected) * len(atoms) * 3 * 4)
        xyz = np.memmap(tmp.name, dtype=np.float32, mode="r+",
                        shape=(len(selected), len(atoms), 3))
        done = 0
        for file_num, (source, picks) in enumerate(sources.items(), 1):
            picks.sort()
            with md.formats.XTCTrajectoryFile(source) as xtc:
                by_frame = defaultdict(list)
                for frame, target in picks:
                    by_frame[frame].append(target)
                for frame in sorted(by_frame):
                    xtc.seek(frame)
                    one, t, step, box = xtc.read(n_frames=1, atom_indices=atoms)
                    if len(one) != 1:
                        raise ValueError(f"Missing {source}:{frame}")
                    for target in by_frame[frame]:
                        xyz[target] = one[0]
                        times[target] = t[0]
                        steps[target] = step[0]
                        boxes[target] = box[0]
            done += len(picks)
            if file_num % 20 == 0 or file_num == len(sources):
                print(f"{condition} full XTC: {done}/10000, {file_num}/{len(sources)} files, "
                      f"elapsed {time.time()-start:.0f}s", flush=True)
        with md.formats.XTCTrajectoryFile(str(output), mode="w") as xtc:
            for lo in range(0, len(selected), 100):
                hi = min(lo+100, len(selected))
                xtc.write(np.asarray(xyz[lo:hi]), time=times[lo:hi],
                          step=steps[lo:hi], box=boxes[lo:hi])
        md.Trajectory(xyz=np.asarray(xyz[0:1]), topology=reduced_top).save_pdb(str(topology_output))
        xyz.flush()
        del xyz
    report = {"condition": condition, "frames": len(selected), "protein_atoms": len(protein_atoms),
              "glucose_atoms": len(glucose_atoms), "distinct_source_files": len(sources),
              "selected_xtc": str(output),
              "topology_pdb": str(topology_output),
              "original_trajectory_history": str(out / f"{condition}_original_trajectory_history.csv"),
              "elapsed_seconds": time.time()-start}
    (out / f"{condition}_full_xtc_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
