#!/usr/bin/env python3
"""Draw fresh source frames using saved PfHT1 MSM stationary distributions.

No prepared frame trajectory is used. Holo candidates are prescreened only by
the precomputed glucose z coordinate; actual glucose-pocket contacts are checked
from atomic coordinates during pocket-feature extraction.
"""
from __future__ import annotations
import csv
import json
import pickle
import re
from pathlib import Path
import numpy as np

from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
SEED = 47
N_APO = 10000
N_HOLO_CANDIDATES = 50000
HOLO_Z_PREFILTER_NM = 2.0
SETUPS = {
    "apo": {
        "map": ROOT / "msm_feature_row_map_apo.csv",
        "dtraj": ROOT / "../../apo_pfht1/msm/clus_tica_dtrajs/clus_tica_dtrajs_combined_1700_65.pkl",
        "model": ROOT / "../../apo_pfht1/msm/mlmsm_model/PfHT1_mlmsm_lag200_cls1700_ct65.pkl",
    },
    "holo": {
        "map": ROOT / "msm_feature_row_map_holo.csv",
        "dtraj": ROOT / "../../holo_pfht1/msm/clus_tica_dtrajs/clus_tica_dtrajs_combined_900_80.pkl",
        "model": ROOT / "../../holo_pfht1/msm/mlmsm_model/PfHT1_mlmsm_lag200_cls900_ct80.pkl",
    },
}


def original_trajectory(condition, feature_file):
    path = Path(feature_file)
    if condition == "apo":
        name = path.name.replace("_distance_features_new.npy", "-strip.xtc")
        return ROOT / "../../apo_pfht1/strip_trajs" / name
    name = path.name.replace("_distance_features_new.npy", ".xtc")
    match = re.search(r"cMD(\d+)", name)
    if not match:
        raise ValueError(name)
    run = int(match.group(1)) - 1
    return ROOT / f"../../holo_pfht1/RUN{run}" / name


def build_candidate_index(condition, mapping, dtrajs):
    state_parts, row_parts, frame_parts = [], [], []
    total_per_state = {}
    eligible_per_state = {}
    invalid = 0
    for i, (record, labels) in enumerate(zip(mapping, dtrajs)):
        original_len = int(record["original_frames"])
        if len(labels) != int(record["clean_frames"]):
            raise ValueError(f"Row {i}: feature/dtraj length mismatch")
        if condition == "holo":
            data = np.load(record["feature_file"], mmap_mode="r")
            data = data.reshape(-1, data.shape[-1])
            good = np.isfinite(data).all(axis=1)
            if good.sum() != len(labels):
                raise ValueError(f"Row {i}: source finite-mask mismatch")
            frame_ids = np.flatnonzero(good)
            z = data[good, 15]
            eligible = np.abs(z) <= HOLO_Z_PREFILTER_NM
        elif int(record["invalid_frames"]):
            data = np.load(record["feature_file"], mmap_mode="r")
            good = np.isfinite(data.reshape(-1, data.shape[-1])).all(axis=1)
            frame_ids = np.flatnonzero(good)
            eligible = np.ones(len(labels), dtype=bool)
            invalid += int((~good).sum())
        else:
            frame_ids = np.arange(original_len, dtype=np.int32)
            eligible = np.ones(len(labels), dtype=bool)
        states = np.asarray(labels, dtype=np.int32)
        unique, counts = np.unique(states, return_counts=True)
        for s, count in zip(unique, counts):
            total_per_state[int(s)] = total_per_state.get(int(s), 0) + int(count)
        unique, counts = np.unique(states[eligible], return_counts=True)
        for s, count in zip(unique, counts):
            eligible_per_state[int(s)] = eligible_per_state.get(int(s), 0) + int(count)
        state_parts.append(states[eligible])
        row_parts.append(np.full(int(eligible.sum()), i, dtype=np.int32))
        frame_parts.append(frame_ids[eligible].astype(np.int32))
        if (i + 1) % 500 == 0:
            print(f"{condition}: indexed {i+1}/{len(mapping)} feature trajectories", flush=True)
    states = np.concatenate(state_parts)
    rows = np.concatenate(row_parts)
    frames = np.concatenate(frame_parts)
    return states, rows, frames, total_per_state, eligible_per_state, invalid


def draw(condition, n, mapping, states, rows, frames, total, eligible, model, rng):
    symbols = np.asarray(model.state_symbols(), dtype=np.int32)
    pi = np.asarray(model.stationary_distribution, dtype=np.float64)
    if len(symbols) != len(pi):
        raise ValueError("MSM symbols and pi length mismatch")
    probability = {int(s): float(p) for s, p in zip(symbols, pi)}
    masses = {s: probability.get(s, 0.0) * eligible.get(s, 0) / count
              for s, count in total.items() if count}
    selected_states = np.array(sorted(s for s, mass in masses.items() if mass > 0), dtype=np.int32)
    mass_array = np.array([masses[int(s)] for s in selected_states], dtype=np.float64)
    mass_array /= mass_array.sum()
    quotas = rng.multinomial(n, mass_array)
    state_order = np.argsort(states, kind="stable")
    sorted_states = states[state_order]
    selected = []
    for state, quota in zip(selected_states, quotas):
        if not quota:
            continue
        lo = np.searchsorted(sorted_states, state, side="left")
        hi = np.searchsorted(sorted_states, state, side="right")
        options = state_order[lo:hi]
        if quota > len(options):
            raise ValueError(f"State {state}: quota {quota} exceeds {len(options)} eligible frames")
        selected.extend(rng.choice(options, size=quota, replace=False).tolist())
    selected = np.array(selected, dtype=np.int64)
    rng.shuffle(selected)
    records = []
    for rank, idx in enumerate(selected):
        row = int(rows[idx])
        state = int(states[idx])
        source = original_trajectory(condition, mapping[row]["feature_file"])
        records.append({"draw_rank": rank, "condition": condition,
                        "source": str(source), "frame": int(frames[idx]),
                        "msm_feature_row": row, "msm_state": state,
                        "state_stationary_probability": probability[state],
                        "frame_equilibrium_weight": probability[state] / total[state],
                        "feature_file": mapping[row]["feature_file"]})
    missing = [r["source"] for r in records if not Path(r["source"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected trajectory paths absent; first: {missing[0]}")
    return records, {"n": n, "eligible_frames": len(states),
                     "active_msm_states": len(selected_states),
                     "candidate_source_files": len({r["source"] for r in records}),
                     "model": str(SETUPS[condition]["model"]),
                     "dtraj": str(SETUPS[condition]["dtraj"]),
                     "sampling": "state multinomial proportional to pi_s * eligible_fraction_s; unique uniform frames within state"}


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = {"seed": SEED, "holo_z_prefilter_nm": HOLO_Z_PREFILTER_NM,
               "holo_final_contact_rule": "At least 3 of 8 core residues within 0.45 nm of glucose heavy atoms and glucose center within 0.85 nm of core; checked later from coordinates",
               "preselected_frame_sets_used": False}
    for condition, n in (("apo", N_APO), ("holo", N_HOLO_CANDIDATES)):
        cfg = SETUPS[condition]
        mapping = list(csv.DictReader(cfg["map"].open()))
        with cfg["dtraj"].open("rb") as f:
            dtrajs = pickle.load(f)
        with cfg["model"].open("rb") as f:
            model = pickle.load(f)
        if len(mapping) != len(dtrajs):
            raise ValueError(f"{condition}: mapping/dtraj row mismatch")
        states, rows, frames, total, eligible, invalid = build_candidate_index(
            condition, mapping, dtrajs)
        records, info = draw(condition, n, mapping, states, rows, frames,
                             total, eligible, model, rng)
        path = OUT / f"{condition}_draw.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        info.update(invalid_feature_frames=invalid, draw_file=str(path))
        summary[condition] = info
        print(f"{condition}: drew {n} fresh frames from {info['candidate_source_files']} source trajectories", flush=True)
    (OUT / "draw_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
