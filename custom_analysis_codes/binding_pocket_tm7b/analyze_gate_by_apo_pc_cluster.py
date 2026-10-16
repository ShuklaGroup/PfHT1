#!/usr/bin/env python3
"""Compare gate and TM7b distributions inside apo-defined pocket clusters."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import wasserstein_distance

from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
SEED = 113
BOOTSTRAPS = 1000
FEATURES = {
    "e_gate_nm": {"label": "Extracellular gate distance (nm)",
                  "definition": "Distance between CA centroids of TM1 residues 45/47/49 and TM7b residues 315/316/317."},
    "i_gate_nm": {"label": "Intracellular gate distance (nm)",
                  "definition": "Distance between CA centroids of TM4 residues 145/152 and TM10 residues 409/419."},
    "tm7b_angle_deg": {"label": "TM7b bend angle (degrees)",
                       "definition": "Angle defined by CA centroid 307/308/309, CA 314, and CA 322; CA 314 is the vertex."},
}


def condition_bootstrap(records, condition, cluster, key, rng):
    condition_rows = [r for r in records if r["condition"] == condition]
    all_slots = sorted({r["trajectory_slot"] for r in condition_rows})
    by_slot = {slot: [] for slot in all_slots}
    for row in condition_rows:
        if int(row["apo_pc_cluster"]) == cluster:
            by_slot[row["trajectory_slot"]].append(float(row[key]))
    arrays = [np.asarray(by_slot[slot], dtype=float) for slot in all_slots]
    medians = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(arrays), size=len(arrays))
        nonempty = [arrays[int(i)] for i in sampled if len(arrays[int(i)])]
        if nonempty:
            medians.append(float(np.median(np.concatenate(nonempty))))
    return np.asarray(medians), sum(bool(len(a)) for a in arrays), len(arrays)


def describe(values):
    q = np.quantile(values, (0.025, 0.25, 0.5, 0.75, 0.975))
    return {"n": len(values), "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "q2.5": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q97.5": float(q[4])}


def main():
    records = list(csv.DictReader((OUT / "selected_frame_provenance.csv").open()))
    rng = np.random.default_rng(SEED)
    result = {
        "note": "Gate and TM7b variables are annotations and were not inputs to apo-PC clustering.",
        "definitions": {key: value["definition"] for key, value in FEATURES.items()},
        "bootstrap": "1000 trajectory-slot block-bootstrap replicates; differences are holo median minus apo median.",
        "clusters": {},
    }
    plot_data = {}
    for cluster in range(3):
        cluster_result = {"conditions": {}, "holo_minus_apo": {}}
        plot_data[cluster] = {}
        for condition in ("apo", "holo"):
            rows = [r for r in records if r["condition"] == condition and
                    int(r["apo_pc_cluster"]) == cluster]
            plot_data[cluster][condition] = {
                key: np.asarray([float(r[key]) for r in rows]) for key in FEATURES}
            cluster_result["conditions"][condition] = {
                key: describe(plot_data[cluster][condition][key]) for key in FEATURES}
        for key in FEATURES:
            apo = plot_data[cluster]["apo"][key]
            holo = plot_data[cluster]["holo"][key]
            comparison = {
                "median_difference": float(np.median(holo) - np.median(apo)),
                "wasserstein_distance": float(wasserstein_distance(apo, holo)),
            }
            apo_boot, apo_contributing, apo_total = condition_bootstrap(
                records, "apo", cluster, key, rng)
            holo_boot, holo_contributing, holo_total = condition_bootstrap(
                records, "holo", cluster, key, rng)
            comparison["contributing_trajectory_slots"] = {
                "apo": apo_contributing, "apo_total": apo_total,
                "holo": holo_contributing, "holo_total": holo_total}
            if holo_contributing >= 2 and len(apo_boot) and len(holo_boot):
                n = min(len(apo_boot), len(holo_boot))
                difference = holo_boot[:n] - apo_boot[:n]
                comparison["median_difference_block_bootstrap_ci95"] = [
                    float(np.quantile(difference, 0.025)),
                    float(np.quantile(difference, 0.975))]
            else:
                comparison["median_difference_block_bootstrap_ci95"] = None
                comparison["bootstrap_warning"] = "Fewer than two holo trajectory slots contribute; uncertainty is not estimable."
            cluster_result["holo_minus_apo"][key] = comparison
        result["clusters"][str(cluster)] = cluster_result

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = {"apo": "#2d6eaa", "holo": "#dc8640"}
    shifts = {"apo": -0.18, "holo": 0.18}
    for ax, (key, metadata) in zip(axes, FEATURES.items()):
        for cluster in range(3):
            for condition in ("apo", "holo"):
                values = plot_data[cluster][condition][key]
                position = cluster + shifts[condition]
                if len(values) >= 2:
                    violin = ax.violinplot([values], positions=[position], widths=0.32,
                                           showmeans=False, showmedians=False,
                                           showextrema=False, points=150)
                    for body in violin["bodies"]:
                        body.set_facecolor(colors[condition])
                        body.set_edgecolor(colors[condition])
                        body.set_alpha(0.45)
                    q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
                    ax.vlines(position, q25, q75, color=colors[condition], lw=4, zorder=4)
                    ax.scatter(position, median, color="white", edgecolor=colors[condition],
                               s=28, zorder=5)
                else:
                    ax.scatter(position, values[0], marker="x", color=colors[condition],
                               s=45, linewidth=1.5, zorder=5)
        ax.set(xlabel="Apo-defined pocket cluster", ylabel=metadata["label"],
               xticks=(0, 1, 2))
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(handles=[Patch(facecolor=colors["apo"], alpha=.55, label="apo"),
                            Patch(facecolor=colors["holo"], alpha=.55, label="bound holo")],
                   loc="best")
    fig.suptitle("Gate and TM7b distributions within apo-defined pocket clusters", y=1.01)
    fig.tight_layout()
    figure = OUT / "apo_pc_cluster_gate_distributions.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

    cluster_path = OUT / "cluster_summary.json"
    summary = json.loads(cluster_path.read_text())
    summary["gate_distributions_by_apo_pc_cluster"] = result
    cluster_path.write_text(json.dumps(summary, indent=2) + "\n")
    manifest_path = OUT / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["features_and_clustering"]["gate_distributions_by_apo_pc_cluster"] = result
    for name in ("cluster_summary.json", figure.name):
        path = OUT / name
        manifest["integrity"][name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
