#!/usr/bin/env python3
"""Trajectory-level bootstrap of a PfHT1 maximum-likelihood MSM."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtrajs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lag", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--trajectory-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    with args.dtrajs.open("rb") as handle:
        dtrajs = pickle.load(handle)
    n_pick = int(round(args.trajectory_fraction * len(dtrajs)))
    if not 1 <= n_pick <= len(dtrajs):
        raise ValueError("Invalid trajectory fraction")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    records = []
    for replicate in range(args.replicates):
        indices = rng.choice(len(dtrajs), size=n_pick, replace=False)
        selected = [dtrajs[int(i)] for i in indices]
        counts = TransitionCountEstimator(
            args.lag, "sliding").fit_fetch(selected)
        msm = MaximumLikelihoodMSM(lagtime=args.lag).fit_fetch(
            counts.submodel_largest())
        with (args.output / f"msm_bootstrap_{replicate:04d}.pkl").open("wb") as handle:
            pickle.dump(msm, handle)
        np.save(args.output / f"trajectory_indices_{replicate:04d}.npy", indices)
        records.append({"replicate": replicate, "trajectories": n_pick,
                        "active_states": int(msm.n_states)})

    (args.output / "bootstrap_metadata.json").write_text(json.dumps({
        "source_dtrajs": str(args.dtrajs.resolve()),
        "lag_frames": args.lag,
        "replicates": args.replicates,
        "trajectory_fraction": args.trajectory_fraction,
        "sampling": "without replacement within each replicate",
        "seed": args.seed,
        "records": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
