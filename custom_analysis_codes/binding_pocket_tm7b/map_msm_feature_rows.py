#!/usr/bin/env python3
"""Recover exact featurized-trajectory order used by existing PfHT1 MSMs.

The original feature pickles were built from unsorted glob() lists. We use a
full cleaned-array SHA256 fingerprint rather than guessing file order.
"""
from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import pickle
from pathlib import Path
import numpy as np

from project_paths import ANALYSIS_ROOT as ROOT
SETUPS = {
    "apo": {
        "feature_pkl": ROOT / "../../apo_pfht1/msm/PfHT1_apo_features_clean_heavy.pkl",
        "dtraj_pkl": ROOT / "../../apo_pfht1/msm/clus_tica_dtrajs/clus_tica_dtrajs_combined_1700_65.pkl",
        "npy_glob": str(ROOT / "../../apo_pfht1/distance_features_heavy/*distance_features_new.npy"),
    },
    "holo": {
        "feature_pkl": ROOT / "../../holo_pfht1/msm/PfHT1_holo_features_clean_heavy.pkl",
        "dtraj_pkl": ROOT / "../../holo_pfht1/msm/clus_tica_dtrajs/clus_tica_dtrajs_combined_900_80.pkl",
        "npy_glob": str(ROOT / "../../holo_pfht1/distance_features_heavy/filtered/*centered_distance_features_new.npy"),
    },
}


def digest(array):
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(x.dtype.str.encode())
    h.update(memoryview(x).cast("B"))
    return h.hexdigest()


def main(condition):
    cfg = SETUPS[condition]
    files = sorted(glob.glob(cfg["npy_glob"]))
    with cfg["feature_pkl"].open("rb") as f:
        feature_rows = pickle.load(f)
    with cfg["dtraj_pkl"].open("rb") as f:
        dtrajs = pickle.load(f)
    if not (len(files) == len(feature_rows) == len(dtrajs)):
        raise ValueError((condition, len(files), len(feature_rows), len(dtrajs)))
    fingerprint_to_rows = {}
    for i, (features, dtraj) in enumerate(zip(feature_rows, dtrajs)):
        if len(features) != len(dtraj):
            raise ValueError(f"MSM row {i}: {len(features)} features versus {len(dtraj)} labels")
        fingerprint_to_rows.setdefault(digest(features), []).append(i)
    records = []
    collisions = 0
    for j, path in enumerate(files):
        original = np.load(path)
        original = original.reshape(-1, original.shape[-1])
        good = np.isfinite(original).all(axis=1)
        cleaned = original[good]
        key = digest(cleaned)
        matches = fingerprint_to_rows.get(key, [])
        if not matches:
            raise ValueError(f"No exact MSM feature-row match for {path}")
        if len(matches) != 1:
            # Exact duplicate feature trajectories do occur. Their MSM labels
            # must also be identical; then a stable bijection is sufficient.
            if not all(np.array_equal(dtrajs[matches[0]], dtrajs[t]) for t in matches[1:]):
                raise ValueError(f"Duplicate features with different labels: {matches}")
            collisions += 1
        i = matches.pop(0)
        records.append({"msm_row": i, "feature_file": path,
                        "original_frames": len(original), "clean_frames": len(cleaned),
                        "invalid_frames": int((~good).sum()), "fingerprint": key})
        if (j + 1) % 500 == 0:
            print(f"{condition}: mapped {j+1}/{len(files)} files", flush=True)
    if any(v for v in fingerprint_to_rows.values()):
        raise ValueError("Some MSM feature rows were not mapped")
    records.sort(key=lambda r: r["msm_row"])
    out = ROOT / f"msm_feature_row_map_{condition}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"{condition}: exact map of {len(records)} rows; {sum(int(r['invalid_frames']) for r in records)} invalid frames; {collisions} duplicate-feature rows resolved by identical MSM labels; wrote {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("condition", choices=SETUPS)
    main(p.parse_args().condition)
