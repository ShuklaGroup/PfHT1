#!/usr/bin/env python3
"""Fit pocket PCA on apo only and project bound-holo observations unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"


def nearest(query, reference, self_query=False, algorithm="auto"):
    model = NearestNeighbors(n_neighbors=2 if self_query else 1,
                             algorithm=algorithm, n_jobs=1).fit(reference)
    distances, _ = model.kneighbors(query)
    return distances[:, 1 if self_query else 0]


def support_summary(apo, holo, algorithm="auto"):
    apo_self = nearest(apo, apo, self_query=True, algorithm=algorithm)
    holo_to_apo = nearest(holo, apo, algorithm=algorithm)
    median, q95 = np.quantile(apo_self, (0.5, 0.95))
    return {
        "apo_self_nearest_neighbor_median": float(median),
        "apo_self_nearest_neighbor_95pct": float(q95),
        "holo_to_apo_nearest_neighbor_median": float(np.median(holo_to_apo)),
        "holo_inside_apo_fraction_at_median_radius": float(np.mean(holo_to_apo <= median)),
        "holo_inside_apo_fraction_at_95pct_radius": float(np.mean(holo_to_apo <= q95)),
    }, apo_self, holo_to_apo


def main():
    raw = np.load(OUT / "joint_pockets.npz")["raw_distances_nm"]
    apo_raw, holo_raw = raw[:10000], raw[10000:]
    scaler = StandardScaler().fit(apo_raw)
    apo_scaled = scaler.transform(apo_raw)
    holo_scaled = scaler.transform(holo_raw)
    pca = PCA(svd_solver="full").fit(apo_scaled)
    apo_pc = pca.transform(apo_scaled)
    holo_pc = pca.transform(holo_scaled)

    summary_2d, apo_nn_2d, holo_nn_2d = support_summary(
        apo_pc[:, :2], holo_pc[:, :2], algorithm="kd_tree")
    summary_full, _, _ = support_summary(apo_scaled, holo_scaled, algorithm="brute")
    summary = {
        "reference_fit": "Scaler and PCA fitted to 10000 apo frames only; 10000 holo frames transformed without refitting.",
        "input_features": "110 pocket distances (55 side-chain-center and 55 C-alpha pair distances)",
        "displayed_components": ["apo PC1", "apo PC2"],
        "explained_variance_ratio": {
            "PC1": float(pca.explained_variance_ratio_[0]),
            "PC2": float(pca.explained_variance_ratio_[1]),
            "PC1_plus_PC2": float(pca.explained_variance_ratio_[:2].sum()),
        },
        "two_pc_nearest_neighbor_support": summary_2d,
        "full_110d_apo_scaled_nearest_neighbor_support": summary_full,
        "interpretation_limit": "The two-PC view is a visualization; full-dimensional support is the stronger subset test.",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(apo_pc[:, 0], apo_pc[:, 1], s=4, alpha=0.16,
                    color="#2d6eaa", rasterized=True, label="apo (PCA reference)")
    axes[0].scatter(holo_pc[:, 0], holo_pc[:, 1], s=4, alpha=0.16,
                    color="#dc8640", rasterized=True, label="bound holo (projected)")
    axes[0].set(
        xlabel=f"apo PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"apo PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        title="Holo projected onto apo-reference pocket PCs",
    )
    axes[0].legend(markerscale=5)

    for values, label, color in ((apo_nn_2d, "apo leave-one-out", "#2d6eaa"),
                                 (holo_nn_2d, "holo to nearest apo", "#dc8640")):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        axes[1].plot(x, y, lw=2, color=color, label=label)
    radius95 = summary_2d["apo_self_nearest_neighbor_95pct"]
    axes[1].axvline(radius95, color="#333333", ls="--", lw=1.5,
                    label="apo 95% self-NN radius")
    axes[1].set(xlabel="Nearest-apo distance in apo PC1–PC2 space",
                ylabel="Cumulative fraction", title="Projected apo-support check",
                xlim=(0, np.quantile(holo_nn_2d, 0.995)), ylim=(0, 1.01))
    axes[1].legend()
    fig.tight_layout()
    figure = OUT / "apo_reference_pca_holo_projection.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    cluster_file = OUT / "cluster_summary.json"
    cluster = json.loads(cluster_file.read_text())
    cluster["apo_reference_pca"] = summary
    cluster_file.write_text(json.dumps(cluster, indent=2) + "\n")

    manifest_file = OUT / "analysis_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["features_and_clustering"]["apo_reference_pca"] = summary
    for name in ("cluster_summary.json", figure.name):
        path = OUT / name
        manifest["integrity"][name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
