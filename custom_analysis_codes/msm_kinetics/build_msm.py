#!/usr/bin/env python3
"""Fit the PfHT1 tICA, clustering, and maximum-likelihood MSM workflow.

The same implementation is used for apo and glucose-bound data. Condition-
specific model parameters are supplied on the command line; the manuscript
models used 1700 clusters/65% tICA variance for apo and 900 clusters/80% tICA
variance for glucose-bound PfHT1.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from deeptime.clustering import MiniBatchKMeans
from deeptime.decomposition import TICA
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path,
                        help="Pickle containing a list of 2D feature trajectories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True, help="Dataset label, e.g. apo or holo")
    parser.add_argument("--tica-lag", type=int, default=200)
    parser.add_argument("--variance-cutoff", type=float, required=True,
                        help="Retained tICA variance fraction, e.g. 0.65")
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--msm-lag", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    with args.features.open("rb") as handle:
        features = pickle.load(handle)
    features = [np.asarray(x) for x in features]
    if not features or any(x.ndim != 2 for x in features):
        raise ValueError("--features must contain a non-empty list of 2D arrays")

    tica = TICA(lagtime=args.tica_lag,
                var_cutoff=args.variance_cutoff).fit_fetch(features)
    transformed = [tica.transform(x) for x in features]
    clusterer = MiniBatchKMeans(
        n_clusters=args.clusters,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        init_strategy="kmeans++",
        n_jobs=args.jobs,
    ).fit_fetch(np.concatenate(transformed))
    dtrajs = [clusterer.transform(x) for x in transformed]
    counts = TransitionCountEstimator(
        args.msm_lag, "sliding").fit_fetch(dtrajs)
    msm = MaximumLikelihoodMSM(lagtime=args.msm_lag).fit_fetch(
        counts.submodel_largest())
    weights = msm.compute_trajectory_weights(dtrajs)

    prefix = args.output / args.label
    dump(tica, prefix.with_name(prefix.name + "_tica.pkl"))
    dump(transformed, prefix.with_name(prefix.name + "_tica_trajectories.pkl"))
    dump(clusterer, prefix.with_name(prefix.name + "_clusterer.pkl"))
    dump(dtrajs, prefix.with_name(prefix.name + "_discrete_trajectories.pkl"))
    dump(msm, prefix.with_name(prefix.name + "_msm.pkl"))
    dump(weights, prefix.with_name(prefix.name + "_trajectory_weights.pkl"))

    metadata = {
        "label": args.label,
        "feature_file": str(args.features.resolve()),
        "trajectory_count": len(features),
        "frame_count": int(sum(len(x) for x in features)),
        "tica_lag_frames": args.tica_lag,
        "tica_variance_cutoff": args.variance_cutoff,
        "clusters": args.clusters,
        "msm_lag_frames": args.msm_lag,
        "transition_count_mode": "sliding",
        "connected_set": "largest",
    }
    prefix.with_name(prefix.name + "_model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
