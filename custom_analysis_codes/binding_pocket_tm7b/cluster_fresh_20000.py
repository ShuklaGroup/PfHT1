#!/usr/bin/env python3
"""Joint cluster and compare 10k apo plus 10k glucose-bound holo pockets."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pocket_features import CORE_RESSEQ

from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
SEED = 73
TOP = {
    "apo": ROOT / "../../apo_pfht1/PfHT1_Real_1_96x96y122z_HMR_210330-strip.parm7",
    "holo": ROOT / "../../holo_pfht1/PfHT1_OF_with_glc_mix_memb.prmtop",
}


def nearest(query, reference, self_query=False):
    model = NearestNeighbors(n_neighbors=2 if self_query else 1, algorithm="brute", n_jobs=1)
    model.fit(reference)
    dist, _ = model.kneighbors(query)
    return dist[:, 1 if self_query else 0]


def block_bootstrap_cluster_fractions(records, labels, k, condition, offset, rng, replicates=1000):
    local_records = records[offset:offset+10000]
    local_labels = labels[offset:offset+10000]
    blocks = {}
    for i, record in enumerate(local_records):
        blocks.setdefault(record["trajectory_slot"], []).append(i)
    arrays = [np.asarray(index, dtype=int) for index in blocks.values()]
    boot = np.empty((replicates, k), dtype=np.float32)
    for b in range(replicates):
        picked = rng.integers(0, len(arrays), size=len(arrays))
        counts = np.zeros(k, dtype=np.int64)
        total = 0
        for block_id in picked:
            block_labels = local_labels[arrays[int(block_id)]]
            counts += np.bincount(block_labels, minlength=k)
            total += len(block_labels)
        boot[b] = counts / total
    estimate = np.bincount(local_labels, minlength=k) / len(local_labels)
    low, high = np.quantile(boot, (0.025, 0.975), axis=0)
    return {"condition": condition, "block": "sampled trajectory slot",
            "blocks": len(arrays), "replicates": replicates,
            "cluster_fraction": estimate.tolist(),
            "ci95_low": low.tolist(), "ci95_high": high.tolist()}


def selected_holo_z_audit(records):
    cache = {}
    values = np.empty(10000, dtype=np.float32)
    for i, record in enumerate(records[10000:]):
        path = record["feature_file"]
        if path not in cache:
            cache[path] = np.load(path, mmap_mode="r")
        values[i] = cache[path][int(record["frame"]), 15]
    return {"min_nm": float(values.min()), "max_nm": float(values.max()),
            "abs_max_nm": float(np.abs(values).max()),
            "fraction_abs_z_gt_1_5_nm": float(np.mean(np.abs(values) > 1.5)),
            "fraction_abs_z_gt_1_8_nm": float(np.mean(np.abs(values) > 1.8))}


def save_pdb(condition, local_index, name, selected_top):
    with md.formats.XTCTrajectoryFile(str(OUT / f"{condition}_selected_full.xtc")) as xtc:
        xtc.seek(local_index)
        xyz, _, _, _ = xtc.read(n_frames=1)
    traj = md.Trajectory(xyz=xyz, topology=selected_top)
    target = OUT / "representatives" / name
    traj.save_pdb(str(target))
    return str(target)


def main():
    records = []
    blocks, gate_blocks = [], []
    for condition in ("apo", "holo"):
        part = list(csv.DictReader((OUT / f"{condition}_selected.csv").open()))
        data = np.load(OUT / f"{condition}_pockets.npz")
        if len(part) != 10000 or data["pocket_distances_nm"].shape != (10000, 110):
            raise ValueError(f"{condition}: expected exactly 10000 frames and 110 pocket distances")
        records.extend(part)
        blocks.append(data["pocket_distances_nm"])
        gate_blocks.append(data["gate"])
    features = np.concatenate(blocks)
    gates = np.concatenate(gate_blocks)
    scaler = StandardScaler().fit(features)
    z = scaler.transform(features).astype(np.float32)
    rng = np.random.default_rng(SEED)
    # Silhouette is scored on a fixed equal-condition subsample; fitting uses
    # every pocket frame and all 110 three-dimensional distance features.
    sample = np.concatenate((rng.choice(10000, 750, replace=False),
                             10000 + rng.choice(10000, 750, replace=False)))
    scores, fits = {}, {}
    for k in (6, 8, 10, 12):
        fit = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=5,
                              batch_size=1024, max_iter=200).fit(z)
        scores[k] = float(silhouette_score(z[sample], fit.predict(z[sample]),
                                            sample_size=1500, random_state=SEED))
        fits[k] = fit
        print(f"k={k}, silhouette={scores[k]:.3f}", flush=True)
    k = max(scores, key=scores.get)
    fit = fits[k]
    labels = fit.predict(z)
    center_dist = np.linalg.norm(z - fit.cluster_centers_[labels], axis=1)
    representatives = []
    for c in range(k):
        for condition, interval in (("apo", np.arange(10000)),
                                    ("holo", np.arange(10000, 20000))):
            members = interval[labels[interval] == c]
            if len(members):
                i = int(members[np.argmin(center_dist[members])])
                representatives.append((c, condition, i))
    (OUT / "representatives").mkdir(exist_ok=True)
    selected_tops = {condition: md.load_topology(str(OUT / f"{condition}_selected_full_topology.pdb"))
                     for condition in ("apo", "holo")}
    for c, condition, i in representatives:
        name = f"cluster_{c:02d}_{condition}_representative.pdb"
        records[i]["selected_xtc_frame"] = i if condition == "apo" else i - 10000
        records[i]["representative_pdb"] = save_pdb(condition, records[i]["selected_xtc_frame"],
                                                      name, selected_tops[condition])
    for i, (record, label, gate) in enumerate(zip(records, labels, gates)):
        record["selected_xtc_frame"] = i if i < 10000 else i - 10000
        record["cluster"] = int(label)
        record["e_gate_nm"] = float(gate[0])
        record["i_gate_nm"] = float(gate[1])
        record["tm7b_angle_deg"] = float(gate[2])
        record["pocket_cluster_distance_standardized"] = float(center_dist[i])
    with (OUT / "frames_clustered.csv").open("w", newline="") as fh:
        fields = sorted(set().union(*(r.keys() for r in records)))
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(OUT / "joint_pockets.npz", raw_distances_nm=features,
                        standardized=z, gate=gates, cluster=labels)
    apo, holo = z[:10000], z[10000:]
    apo_self = nearest(apo, apo, self_query=True)
    holo_self = nearest(holo, holo, self_query=True)
    holo_to_apo = nearest(holo, apo)
    apo_to_holo = nearest(apo, holo)
    radius50, radius95 = np.quantile(apo_self, (0.5, 0.95))
    dispersion = {name: float(np.sqrt(np.mean(np.sum((part-part.mean(0))**2, axis=1))))
                  for name, part in (("apo", apo), ("holo", holo))}
    cluster_counts = {name: np.bincount(lab, minlength=k).tolist()
                      for name, lab in (("apo", labels[:10000]), ("holo", labels[10000:]))}
    uncertainty = {
        "apo": block_bootstrap_cluster_fractions(records, labels, k, "apo", 0, rng),
        "holo": block_bootstrap_cluster_fractions(records, labels, k, "holo", 10000, rng),
    }
    summary = {
        "apo_frames": 10000, "glucose_bound_holo_frames": 10000,
        "clustering_features": "55 sidechain-center and 55 CA pairwise distances among 11 pocket residues",
        "pocket_residue_sequence_numbers": CORE_RESSEQ,
        "gate_features_are_annotations_not_clustering_inputs": ["TM1-TM7b eGate", "TM4-TM10 iGate", "TM7b CA bend angle"],
        "k": k, "silhouette_by_k": scores, "cluster_counts": cluster_counts,
        "cluster_fraction_block_bootstrap": uncertainty,
        "selected_holo_precomputed_z_audit": selected_holo_z_audit(records),
        "shape_rms_dispersion_standardized": dispersion,
        "apo_self_nearest_neighbor_median": float(radius50),
        "apo_self_nearest_neighbor_95pct": float(radius95),
        "holo_to_apo_nearest_neighbor_median": float(np.median(holo_to_apo)),
        "apo_to_holo_nearest_neighbor_median": float(np.median(apo_to_holo)),
        "holo_inside_apo_fraction_at_apo_self_nn_median": float(np.mean(holo_to_apo <= radius50)),
        "holo_inside_apo_fraction_at_apo_self_nn_95pct": float(np.mean(holo_to_apo <= radius95)),
        "holo_self_nearest_neighbor_median": float(np.median(holo_self)),
        "representatives": [{"cluster": c, "condition": condition, **records[i]}
                            for c, condition, i in representatives],
    }
    (OUT / "cluster_summary.json").write_text(json.dumps(summary, indent=2))
    pca = PCA(n_components=2, random_state=SEED).fit(z)
    xy = pca.transform(z)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for name, inds, color in (("apo", slice(0, 10000), "#2d6eaa"),
                              ("bound holo", slice(10000, 20000), "#dc8640")):
        axs[0].scatter(xy[inds, 0], xy[inds, 1], s=3, alpha=0.14,
                       color=color, rasterized=True, label=name)
    for c, condition, i in representatives:
        if condition == "apo":
            axs[0].text(xy[i, 0], xy[i, 1], str(c), fontsize=8, weight="bold")
    axs[0].set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
               ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
               title="Joint pocket-shape projection")
    axs[0].legend(markerscale=5)
    x = np.arange(k)
    axs[1].bar(x-.19, cluster_counts["apo"], .38, color="#2d6eaa", label="apo")
    axs[1].bar(x+.19, cluster_counts["holo"], .38, color="#dc8640", label="bound holo")
    axs[1].set(xlabel="Joint pocket cluster", ylabel="MSM-drawn frame count")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "apo_holo_pocket_ensemble.png", dpi=220)
    print(json.dumps({q: v for q, v in summary.items() if q != "representatives"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
