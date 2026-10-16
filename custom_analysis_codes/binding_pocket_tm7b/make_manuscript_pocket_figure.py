#!/usr/bin/env python3
"""Build the manuscript figure for apo pocket breadth and glucose selection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
PANEL_OUT = OUT / "manuscript_pocket_subplots"

# Dark pastel colors already used in the manuscript's TIC/FES palette.
APO_COLOR = "#6247AA"   # muted purple
HOLO_COLOR = "#718355"  # muted sage green

# Internal labels are unchanged in the analysis files. For display, put the
# glucose-dominant old cluster 1 first, followed by old clusters 0 and 2.
DISPLAY_ORDER = (1, 0, 2)
OLD_TO_DISPLAY = {old: display for display, old in enumerate(DISPLAY_ORDER, 1)}


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left")


def plot_projection(ax, apo_pc, holo_pc, apo_labels, *, label_panel=True):
    ax.scatter(apo_pc[:, 0], apo_pc[:, 1], s=3, alpha=0.14,
               color=APO_COLOR, linewidths=0, rasterized=True,
               label="apo (reference)")
    ax.scatter(holo_pc[:, 0], holo_pc[:, 1], s=3, alpha=0.17,
               color=HOLO_COLOR, linewidths=0, rasterized=True,
               label="glucose-bound (projected)")
    for old_cluster in DISPLAY_ORDER:
        center = apo_pc[apo_labels == old_cluster, :2].mean(axis=0)
        display_cluster = OLD_TO_DISPLAY[old_cluster]
        ax.text(
            center[0], center[1], str(display_cluster), ha="center", va="center",
            fontsize=18, fontweight="normal", zorder=6,
            bbox={"boxstyle": "circle,pad=0.34", "facecolor": "white",
                  "edgecolor": "#222222", "linewidth": 1.3},
        )
    ax.set_xlabel("apo PC1 (29.4%)")
    ax.set_ylabel("apo PC2 (10.6%)")
    ax.legend(loc="upper right", markerscale=3, frameon=False,
              handletextpad=0.4, borderaxespad=0.2)
    if label_panel:
        panel_label(ax, "a")


def plot_occupancy(ax, fractions, bootstrap, *, label_panel=True):
    x = np.arange(1, 4)
    width = 0.36
    for condition, color, shift, label in (
            ("apo", APO_COLOR, -width / 2, "apo"),
            ("holo", HOLO_COLOR, width / 2, "glucose-bound")):
        values = fractions[condition][list(DISPLAY_ORDER)]
        low = np.asarray(bootstrap[condition]["ci95_low"])[list(DISPLAY_ORDER)]
        high = np.asarray(bootstrap[condition]["ci95_high"])[list(DISPLAY_ORDER)]
        ax.bar(x + shift, values * 100, width, color=color, label=label,
               yerr=np.vstack(((values - low) * 100, (high - values) * 100)),
               error_kw={"elinewidth": 1, "capthick": 1}, capsize=2.5)
    ax.set_xlabel("Apo-defined cluster")
    ax.set_ylabel("MSM occupancy (%)")
    ax.set_xticks(x)
    ax.set_ylim(0, 103)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4,
              borderaxespad=0.2)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    if label_panel:
        panel_label(ax, "b")


def plot_violin(ax, arrays, ylabel, panel, *, label_panel=True):
    violin = ax.violinplot(arrays, positions=(0, 1), widths=0.72,
                           showmeans=False, showmedians=False,
                           showextrema=False, points=200)
    for body, color in zip(violin["bodies"], (APO_COLOR, HOLO_COLOR)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.52)
    for position, values, color in zip((0, 1), arrays,
                                       (APO_COLOR, HOLO_COLOR)):
        q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
        ax.vlines(position, q25, q75, color=color, linewidth=5.5, zorder=4)
        ax.scatter(position, median, s=38, facecolor="white", edgecolor=color,
                   linewidth=1.4, zorder=5)
    ax.set_xticks((0, 1), ("apo", "glucose-bound"))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    if label_panel:
        panel_label(ax, panel)


def plot_cracking_violin(ax, arrays, cracked_percent, *, label_panel=True):
    """Plot per-frame central TM7b H-bond disruption for complete ensembles."""
    plot_violin(
        ax, arrays, "TM7b disruption (%)", "d",
        label_panel=label_panel,
    )
    ax.set_yticks((0, 100 / 3, 200 / 3, 100), ("0", "33", "67", "100"))
    # Leave headroom for the cracked-population annotations after the panel is
    # reduced to one quarter of the manuscript text width.
    ax.set_ylim(-8, 122)
    for position, percent, color in zip(
        (0, 1), cracked_percent, (APO_COLOR, HOLO_COLOR)
    ):
        ax.text(
            position, 108, f"{percent:.1f}% cracked", ha="center", va="center",
            fontsize=16, fontweight="normal", color=color,
        )


def export_standalone(plotter, filename: str, figsize):
    """Export a clean, unlabeled panel in raster and vector formats."""
    fig, ax = plt.subplots(figsize=figsize)
    plotter(ax, label_panel=False)
    # Fixed margins and no tight bounding box keep every exported panel at the
    # same pixel dimensions for later manuscript assembly.
    fig.subplots_adjust(left=0.235, right=0.98, bottom=0.215, top=0.96)
    stem = PANEL_OUT / Path(filename).stem
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    PANEL_OUT.mkdir(parents=True, exist_ok=True)

    archive = np.load(OUT / "joint_pockets.npz")
    raw = archive["raw_distances_nm"]
    apo_raw, holo_raw = raw[:10000], raw[10000:]
    scaler = StandardScaler().fit(apo_raw)
    pca = PCA(svd_solver="full").fit(scaler.transform(apo_raw))
    apo_pc = pca.transform(scaler.transform(apo_raw))
    holo_pc = pca.transform(scaler.transform(holo_raw))

    records = list(csv.DictReader((OUT / "selected_frame_provenance.csv").open()))
    apo_records, holo_records = records[:10000], records[10000:]
    apo_labels = np.asarray([int(r["apo_pc_cluster"]) for r in apo_records])

    summary = json.loads((OUT / "cluster_summary.json").read_text())
    clustering = summary["apo_reference_pc_clustering"]
    bootstrap = clustering["trajectory_slot_block_bootstrap"]
    fractions = {condition: np.asarray(clustering["fractions"][condition])
                 for condition in ("apo", "holo")}

    # Gate distances are stored in nm and displayed here in angstroms. The
    # final distribution uses all MSM-drawn frames, without a cluster filter.
    all_records = {"apo": apo_records, "holo": holo_records}
    feature_info = (
        ("e_gate_nm", "Extracellular gate (Å)", 10.0, "c"),
    )
    violin_data = {}
    for key, ylabel, scale, panel in feature_info:
        violin_data[panel] = (
            [np.asarray([float(r[key]) * scale for r in all_records[condition]])
             for condition in ("apo", "holo")],
            ylabel,
        )

    cracking_archive = np.load(OUT / "tm7b_cracking_all_frames.npz")
    cracking_summary = json.loads(
        (OUT / "tm7b_cracking_all_frames_summary.json").read_text()
    )
    cracking_arrays = [
        cracking_archive[f"{condition}_disrupted_percent"]
        for condition in ("apo", "holo")
    ]
    cracked_percent = [
        cracking_summary["systems"][condition]["cracked_percent"]
        for condition in ("apo", "holo")
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        # These panels are reduced to approximately one quarter of text width
        # in Figure 6, so the source typography must be substantially larger.
        "font.size": 18,
        "font.weight": "normal",
        "axes.labelsize": 21,
        "axes.labelweight": "normal",
        "axes.titlesize": 21,
        "axes.titleweight": "normal",
        "legend.fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6.5,
        "ytick.major.size": 6.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })

    # Equal-size 2 x 2 organization for final manuscript assembly.
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    plot_projection(ax_a, apo_pc, holo_pc, apo_labels)
    plot_occupancy(ax_b, fractions, bootstrap)
    arrays, ylabel = violin_data["c"]
    plot_violin(ax_c, arrays, ylabel, "c")
    plot_cracking_violin(ax_d, cracking_arrays, cracked_percent)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.09,
                        hspace=0.42, wspace=0.34)
    stem = OUT / "manuscript_pocket_selection_figure"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=300, facecolor="white")
    plt.close(fig)

    # Independent PNGs have no fixed panel letters, making later rearrangement
    # possible without editing the plot itself.
    export_standalone(
        lambda ax, label_panel=False: plot_projection(
            ax, apo_pc, holo_pc, apo_labels, label_panel=label_panel),
        "panel_a_apo_reference_pocket_space.png", (4.8, 4.1))
    export_standalone(
        lambda ax, label_panel=False: plot_occupancy(
            ax, fractions, bootstrap, label_panel=label_panel),
        "panel_b_cluster_occupancy.png", (4.8, 4.1))
    arrays, ylabel = violin_data["c"]
    export_standalone(
        lambda ax, label_panel=False, arrays=arrays, ylabel=ylabel:
            plot_violin(ax, arrays, ylabel, "c", label_panel=label_panel),
        "panel_c_all_frames_egate_violin.png", (4.8, 4.1))
    export_standalone(
        lambda ax, label_panel=False: plot_cracking_violin(
            ax, cracking_arrays, cracked_percent, label_panel=label_panel
        ),
        "panel_d_all_frames_tm7b_cracking_violin.png", (4.8, 4.1),
    )

    mapping = {
        "display_cluster_mapping": {
            "1": {"internal_apo_pc_cluster": 1,
                  "note": "glucose-dominant cluster; 95.0% of holo frames"},
            "2": {"internal_apo_pc_cluster": 0},
            "3": {"internal_apo_pc_cluster": 2},
        },
        "colors": {"apo": APO_COLOR, "glucose_bound": HOLO_COLOR},
        "units": {"eGate": "angstrom",
                  "TM7b_cracking": "percent central i-to-i+4 bonds disrupted per frame",
                  "TM7b_cracking_free_energy": "kcal per mol"},
        "panel_scope": {
            "c_eGate": "all 10000 frames per condition; no pocket-cluster filter",
            "d_TM7b_cracking": "all 10000 frames per condition; no pocket-cluster filter",
            "e_TM7b_cracking_free_energy": (
                "DeltaG(cracked-intact) from the same all-frame MSM-weighted "
                "populations; trajectory-slot block-bootstrap uncertainty"
            ),
        },
        "TM7b_crack_definition": cracking_summary["binary_crack_definition"],
        "TM7b_central_pairs": cracking_summary["central_pairs"],
        "standalone_panels_have_fixed_letters": False,
    }
    mapping_path = PANEL_OUT / "panel_metadata.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    (PANEL_OUT / "README.md").write_text(
        "# Manuscript pocket-ensemble subplots\n\n"
        "The four primary subplots have identical canvas dimensions and are "
        "exported as PNG, PDF, and SVG without fixed panel letters or titles "
        "so they can be rearranged during manuscript assembly. Typography is "
        "sized for four panels placed across the manuscript text width. The "
        "composite figure uses "
        "panels a–d. Display cluster 1 is the glucose-dominant state and "
        "corresponds to internal apo-PC cluster 1; display clusters 2 and 3 "
        "correspond to internal clusters 0 and 2. eGate is shown in angstroms "
        "for all frames. Panel d uses all 10,000 frames per condition "
        "and shows the percentage of the three central TM7b i-to-i+4 hydrogen "
        "bonds disrupted in each frame. Apo is muted purple (#6247AA), and "
        "glucose-bound PfHT1 is "
        "muted sage (#718355), matching colors used in the manuscript TIC/FES "
        "palette. See `panel_metadata.json` for the machine-readable mapping.\n"
        "`panel_e_tm7b_cracking_free_energy.{png,pdf,svg}` is generated by "
        "`analyze_tm7b_cracking_energetics.py` and reports "
        "DeltaG(cracked-intact) at 310 K with trajectory-slot block-bootstrap "
        "95% confidence intervals.\n"
    )

    manifest_path = OUT / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["integrity"] = {
        name: value for name, value in manifest["integrity"].items()
        if not name.startswith("manuscript_pocket_subplots/")
        and not name.startswith("manuscript_pocket_selection_figure_kink_angle")
    }
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        manifest["integrity"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for path in sorted(PANEL_OUT.iterdir()):
        if path.is_file():
            relative = str(path.relative_to(OUT))
            manifest["integrity"][relative] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    for name in (
        "tm7b_cracking_all_frames.npz",
        "tm7b_cracking_all_frames_summary.json",
    ):
        path = OUT / name
        manifest["integrity"][name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
