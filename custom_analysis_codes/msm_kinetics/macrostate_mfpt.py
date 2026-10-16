#!/usr/bin/env python3
"""Assign gate-defined PfHT1 macrostates and calculate pairwise MFPTs."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--dtrajs", required=True, type=Path)
    parser.add_argument("--msm", required=True, type=Path)
    parser.add_argument("--state-ranges", required=True, type=Path,
                        help="JSON mapping state names to eGate/iGate ranges")
    parser.add_argument("--egate-column", type=int, default=14)
    parser.add_argument("--igate-column", type=int, default=13)
    parser.add_argument("--time-per-step-us", type=float, default=0.0001)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    features = load_pickle(args.features)
    dtrajs = load_pickle(args.dtrajs)
    msm = load_pickle(args.msm)
    ranges = json.loads(args.state_ranges.read_text())
    if len(features) != len(dtrajs):
        raise ValueError("Feature and discrete-trajectory counts differ")

    sums = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    counts = defaultdict(int)
    for feat, labels in zip(features, dtrajs):
        if len(feat) != len(labels):
            raise ValueError("Feature/dtraj frame count differs")
        values = np.asarray(feat)[:, [args.egate_column, args.igate_column]]
        for state in np.unique(labels):
            mask = np.asarray(labels) == state
            sums[int(state)] += values[mask].sum(axis=0)
            counts[int(state)] += int(mask.sum())
    means = {state: sums[state] / counts[state] for state in counts}

    symbols = np.asarray(msm.state_symbols(), dtype=int)
    pi = np.asarray(msm.stationary_distribution)
    pi_by_symbol = {int(symbol): float(prob) for symbol, prob in zip(symbols, pi)}
    selected = {}
    for name, definition in ranges.items():
        e_lo, e_hi = definition["egate_nm"]
        i_lo, i_hi = definition["igate_nm"]
        candidates = [state for state, (egate, igate) in means.items()
                      if state in pi_by_symbol and e_lo <= egate <= e_hi
                      and i_lo <= igate <= i_hi]
        if not candidates:
            raise ValueError(f"No active microstates in range for {name}")
        selected[name] = max(candidates, key=pi_by_symbol.get)

    # deeptime MSM methods accept the original state symbols exposed by
    # state_symbols(); this preserves cluster numbering after connected-set
    # restriction.
    rows = []
    for source, target in permutations(selected, 2):
        value = float(msm.mfpt([selected[source]], [selected[target]]))
        rows.append({"source": source, "target": target,
                     "source_microstate": selected[source],
                     "target_microstate": selected[target],
                     "mfpt_model_steps": value,
                     "mfpt_us": value * args.time_per_step_us})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".states.json").write_text(json.dumps({
        "ranges": ranges,
        "selected_microstates": selected,
        "microstate_mean_gate_distances_nm": {
            str(k): v.tolist() for k, v in means.items()},
        "time_per_step_us": args.time_per_step_us,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
