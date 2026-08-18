"""Cluster VAE encoder means and summarize flux-weighted pathway classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ROUTE_NAMES = ("egate_first", "igate_first", "concerted")


def canonicalize_by_flux(labels, flux):
    totals = np.bincount(labels, weights=flux)
    order = np.argsort(-totals)
    mapping = np.empty(len(order), dtype=np.int32)
    mapping[order] = np.arange(len(order))
    return mapping[labels]


def state_masses(offsets, states, flux, labels, clusters, number_states):
    masses = np.zeros((clusters, number_states), dtype=np.float64)
    for path_index, cluster in enumerate(labels):
        visited = states[offsets[path_index] : offsets[path_index + 1]]
        np.add.at(masses[cluster], visited, flux[path_index] / len(visited))
    masses /= masses.sum(axis=1, keepdims=True)
    return masses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--fingerprints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clusters", type=int, default=2)
    args = parser.parse_args()

    with np.load(args.embedding) as archive:
        latent = np.asarray(archive["mu"], dtype=np.float64)
    with np.load(args.fingerprints) as archive:
        flux = np.asarray(archive["path_flux"], dtype=np.float64)
        route_code = np.asarray(archive["route_code"], dtype=np.int8)
        offsets = np.asarray(archive["path_state_offsets"], dtype=np.int64)
        states = np.asarray(archive["path_states"], dtype=np.int32)
        number_states = int(archive["number_states"])
        total_reactive_flux = float(archive["total_reactive_flux"])
    if len(latent) != len(flux):
        raise ValueError("Embedding and path-flux lengths differ")

    model = KMeans(
        n_clusters=args.clusters,
        n_init=100,
        random_state=20260804,
        algorithm="lloyd",
    )
    labels = canonicalize_by_flux(model.fit_predict(latent), flux)
    masses = state_masses(
        offsets, states, flux, labels, args.clusters, number_states
    )

    summary = {
        "clusters": args.clusters,
        "silhouette": float(
            silhouette_score(
                latent,
                labels,
                sample_size=min(5000, len(labels)),
                random_state=20260804,
            )
        ),
        "represented_fraction_total_reactive_flux": float(flux.sum() / total_reactive_flux),
        "cluster_summary": [],
    }
    for cluster in range(args.clusters):
        selected = labels == cluster
        cluster_flux = flux[selected].sum()
        route_fractions = {
            name: float(flux[selected & (route_code == code)].sum() / cluster_flux)
            for code, name in enumerate(ROUTE_NAMES)
        }
        summary["cluster_summary"].append(
            {
                "cluster": cluster + 1,
                "path_count": int(selected.sum()),
                "represented_flux_fraction": float(cluster_flux / flux.sum()),
                "route_fractions": route_fractions,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "assignments.npz",
        latent=latent,
        cluster=labels,
        path_flux=flux,
        route_code=route_code,
        cluster_state_masses=masses,
    )
    with (args.output / "cluster_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
