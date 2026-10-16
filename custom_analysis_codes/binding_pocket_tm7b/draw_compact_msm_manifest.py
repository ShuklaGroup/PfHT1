#!/usr/bin/env python3
"""Fresh hierarchical equilibrium draw with fewer randomly selected source XTCs."""
from __future__ import annotations
import csv
import json
import pickle
from pathlib import Path
import numpy as np

from make_fresh_msm_manifest import (ROOT, OUT, SETUPS, HOLO_Z_PREFILTER_NM,
                                     build_candidate_index, original_trajectory)

SEED = 1081
SLOTS = 800
PER_SLOT = {"apo": 13, "holo": 40}


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = {"seed": SEED, "trajectory_slots_per_condition": SLOTS,
               "holo_z_prefilter_nm": HOLO_Z_PREFILTER_NM,
               "preselected_frame_sets_used": False,
               "sampling": "sample trajectory slots by sum of pi_s/N_s over eligible frames; sample frames within each slot proportional to pi_s/N_s with replacement; exact frame marginal proportional to equilibrium weight"}
    for condition in ("apo", "holo"):
        cfg = SETUPS[condition]
        mapping = list(csv.DictReader(cfg["map"].open()))
        with cfg["dtraj"].open("rb") as f:
            dtrajs = pickle.load(f)
        with cfg["model"].open("rb") as f:
            model = pickle.load(f)
        states, rows, frames, total, eligible, invalid = build_candidate_index(condition, mapping, dtrajs)
        symbols = np.asarray(model.state_symbols(), dtype=np.int32)
        pi = np.asarray(model.stationary_distribution, dtype=np.float64)
        prob = np.zeros(int(symbols.max()) + 1, dtype=np.float64)
        prob[symbols] = pi
        counts = np.zeros_like(prob)
        for state, count in total.items():
            counts[state] = count
        frame_weight = prob[states] / counts[states]
        row_mass = np.bincount(rows, weights=frame_weight, minlength=len(mapping))
        slots = rng.choice(len(mapping), size=SLOTS, replace=True, p=row_mass/row_mass.sum())
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        records = []
        for slot_id, row in enumerate(slots):
            lo = np.searchsorted(sorted_rows, row, side="left")
            hi = np.searchsorted(sorted_rows, row, side="right")
            options = order[lo:hi]
            weights = frame_weight[options]
            choices = rng.choice(options, size=PER_SLOT[condition], replace=True,
                                 p=weights/weights.sum())
            source = original_trajectory(condition, mapping[int(row)]["feature_file"])
            for idx in choices:
                state = int(states[idx])
                records.append({"draw_rank": -1, "condition": condition,
                                "source": str(source), "frame": int(frames[idx]),
                                "msm_feature_row": int(row), "msm_state": state,
                                "state_stationary_probability": float(prob[state]),
                                "frame_equilibrium_weight": float(frame_weight[idx]),
                                "feature_file": mapping[int(row)]["feature_file"],
                                "trajectory_slot": slot_id,
                                "trajectory_equilibrium_mass": float(row_mass[int(row)])})
        rng.shuffle(records)
        if condition == "apo":
            records = records[:10000]
        for rank, record in enumerate(records):
            record["draw_rank"] = rank
        absent = [r["source"] for r in records if not Path(r["source"]).exists()]
        if absent:
            raise FileNotFoundError(absent[0])
        path = OUT / f"{condition}_draw.csv"
        if path.exists():
            path.rename(OUT / f"{condition}_draw_fullsource_archived.csv")
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        summary[condition] = {"draws": len(records), "distinct_source_files": len({r["source"] for r in records}),
                              "eligible_source_frames": len(states), "invalid_feature_frames": invalid,
                              "model": str(cfg["model"]), "dtraj": str(cfg["dtraj"]),
                              "row_map": str(cfg["map"]), "manifest": str(path)}
        print(f"{condition}: {len(records)} draws, {summary[condition]['distinct_source_files']} source files", flush=True)
    (OUT / "compact_draw_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
