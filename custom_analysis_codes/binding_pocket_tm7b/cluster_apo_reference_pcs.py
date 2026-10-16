#!/usr/bin/env python3
"""Cluster apo pocket PCs and assign bound-holo frames to fixed apo clusters."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
SEED = 89
BOOTSTRAPS = 1000


def block_bootstrap(records, labels, k, rng):
    blocks = {}
    for i, record in enumerate(records):
        blocks.setdefault(record["trajectory_slot"], []).append(i)
    arrays = [np.asarray(indices, dtype=int) for indices in blocks.values()]
    values = np.empty((BOOTSTRAPS, k), dtype=np.float32)
    for b in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(arrays), size=len(arrays))
        counts = np.zeros(k, dtype=np.int64)
        total = 0
        for block_id in sampled:
            block_labels = labels[arrays[int(block_id)]]
            counts += np.bincount(block_labels, minlength=k)
            total += len(block_labels)
        values[b] = counts / total
    low, high = np.quantile(values, (0.025, 0.975), axis=0)
    return {"block": "sampled trajectory slot", "blocks": len(arrays),
            "replicates": BOOTSTRAPS, "ci95_low": low.tolist(),
            "ci95_high": high.tolist()}


def main():
    data = np.load(OUT / "joint_pockets.npz")
    raw = data["raw_distances_nm"]
    apo_raw, holo_raw = raw[:10000], raw[10000:]
    scaler = StandardScaler().fit(apo_raw)
    apo_scaled = scaler.transform(apo_raw)
    holo_scaled = scaler.transform(holo_raw)
    pca = PCA(svd_solver="full").fit(apo_scaled)
    n_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1)
    apo_pc = pca.transform(apo_scaled)
    holo_pc = pca.transform(holo_scaled)
    apo_fit, holo_fit = apo_pc[:, :n_pc], holo_pc[:, :n_pc]

    scores, fits = {}, {}
    for k in range(3, 11):
        fit = MiniBatchKMeans(n_clusters=k, n_init=10, random_state=SEED,
                              batch_size=1024, max_iter=300).fit(apo_fit)
        scores[k] = float(silhouette_score(apo_fit, fit.labels_, sample_size=2500,
                                            random_state=SEED))
        fits[k] = fit
    k = max(scores, key=scores.get)
    fit = fits[k]
    apo_labels_old = fit.labels_
    holo_labels_old = fit.predict(holo_fit)

    # Stable cluster numbering from negative to positive apo PC1; it does not
    # depend on holo occupancy.
    old_order = np.argsort(fit.cluster_centers_[:, 0])
    relabel = np.empty(k, dtype=int)
    relabel[old_order] = np.arange(k)
    apo_labels = relabel[apo_labels_old]
    holo_labels = relabel[holo_labels_old]
    centers = fit.cluster_centers_[old_order]
    apo_dist = np.linalg.norm(apo_fit - fit.cluster_centers_[apo_labels_old], axis=1)
    holo_dist = np.linalg.norm(holo_fit - fit.cluster_centers_[holo_labels_old], axis=1)

    provenance = OUT / "selected_frame_provenance.csv"
    records = list(csv.DictReader(provenance.open()))
    if len(records) != 20000:
        raise ValueError("Expected 20000 provenance records")
    labels = np.concatenate((apo_labels, holo_labels))
    distances = np.concatenate((apo_dist, holo_dist))
    for record, label, distance in zip(records, labels, distances):
        record["apo_pc_cluster"] = int(label)
        record["apo_pc_cluster_distance"] = float(distance)
    fields = list(records[0])
    with provenance.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    rng = np.random.default_rng(SEED)
    apo_counts = np.bincount(apo_labels, minlength=k)
    holo_counts = np.bincount(holo_labels, minlength=k)
    uncertainty = {
        "apo": block_bootstrap(records[:10000], apo_labels, k, rng),
        "holo": block_bootstrap(records[10000:], holo_labels, k, rng),
    }
    representatives = []
    for cluster in range(k):
        for condition, condition_labels, condition_dist, offset in (
                ("apo", apo_labels, apo_dist, 0),
                ("holo", holo_labels, holo_dist, 10000)):
            members = np.flatnonzero(condition_labels == cluster)
            if len(members):
                local = int(members[np.argmin(condition_dist[members])])
                record = records[offset + local]
                representatives.append({"cluster": cluster, "condition": condition,
                                        "selected_xtc_frame": int(record["selected_xtc_frame"]),
                                        "original_trajectory": record["original_trajectory"],
                                        "original_frame": int(record["original_frame"]),
                                        "distance_to_apo_cluster_center": float(condition_dist[local])})

    fractions = {"apo": (apo_counts / 10000).tolist(),
                 "holo": (holo_counts / 10000).tolist()}
    summary = {
        "reference_definition": "Scaler, PCA, and MiniBatchKMeans fit to apo only; holo transformed and assigned without refitting.",
        "retained_apo_pcs": n_pc,
        "retained_apo_variance_fraction": float(pca.explained_variance_ratio_[:n_pc].sum()),
        "displayed_apo_pc_variance": {"PC1": float(pca.explained_variance_ratio_[0]),
                                      "PC2": float(pca.explained_variance_ratio_[1])},
        "k_selection": "Maximum sampled silhouette across k=3..10 on apo only.",
        "silhouette_by_k": {str(q): value for q, value in scores.items()},
        "k": k,
        "cluster_numbering": "Apo cluster centers sorted from negative to positive apo PC1.",
        "counts": {"apo": apo_counts.tolist(), "holo": holo_counts.tolist()},
        "fractions": fractions,
        "trajectory_slot_block_bootstrap": uncertainty,
        "representative_frames": representatives,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(apo_pc[:, 0], apo_pc[:, 1], s=4, alpha=0.15,
                    color="#2d6eaa", rasterized=True, label="apo (reference)")
    axes[0].scatter(holo_pc[:, 0], holo_pc[:, 1], s=4, alpha=0.15,
                    color="#dc8640", rasterized=True, label="bound holo (projected)")
    for cluster, center in enumerate(centers):
        axes[0].scatter(center[0], center[1], s=80, facecolor="white",
                        edgecolor="black", linewidth=1.2, zorder=5)
        axes[0].text(center[0], center[1], str(cluster), ha="center", va="center",
                     weight="bold", fontsize=9, zorder=6)
    axes[0].set(xlabel=f"apo PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                ylabel=f"apo PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                title="Apo-reference pocket-shape space")
    axes[0].legend(markerscale=5)

    x = np.arange(k)
    width = 0.38
    for condition, values, color, shift in (
            ("apo", np.asarray(fractions["apo"]), "#2d6eaa", -width / 2),
            ("bound holo", np.asarray(fractions["holo"]), "#dc8640", width / 2)):
        key = "holo" if condition == "bound holo" else "apo"
        low = np.asarray(uncertainty[key]["ci95_low"])
        high = np.asarray(uncertainty[key]["ci95_high"])
        axes[1].bar(x + shift, values * 100, width, color=color, label=condition,
                    yerr=np.vstack(((values-low)*100, (high-values)*100)),
                    capsize=3, error_kw={"elinewidth": 1})
    axes[1].set(xlabel="Apo-defined pocket cluster", ylabel="MSM-drawn occupancy (%)",
                title=f"Occupancy in apo PC clusters ({n_pc} PCs; {pca.explained_variance_ratio_[:n_pc].sum():.1%} apo variance)",
                xticks=x, ylim=(0, 102))
    axes[1].legend()
    fig.tight_layout()
    figure = OUT / "apo_reference_pc_clusters.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    cluster_file = OUT / "cluster_summary.json"
    cluster_summary = json.loads(cluster_file.read_text())
    cluster_summary["apo_reference_pc_clustering"] = summary
    cluster_file.write_text(json.dumps(cluster_summary, indent=2) + "\n")

    manifest_file = OUT / "analysis_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["features_and_clustering"]["apo_reference_pc_clustering"] = summary
    for name in ("cluster_summary.json", "selected_frame_provenance.csv", figure.name):
        path = OUT / name
        manifest["integrity"][name] = {"bytes": path.stat().st_size,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
