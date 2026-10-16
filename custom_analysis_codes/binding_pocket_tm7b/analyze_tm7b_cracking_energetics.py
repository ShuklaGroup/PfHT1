#!/usr/bin/env python3
"""Convert MSM-weighted TM7b crack populations into two-state free energies."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from project_paths import ANALYSIS_ROOT as ROOT
OUT = ROOT / "fresh_msm_20000"
PANEL_OUT = OUT / "manuscript_pocket_subplots"
TEMPERATURE_K = 310.0
R_KCAL_MOL_K = 0.00198720425864083
KBT = R_KCAL_MOL_K * TEMPERATURE_K
BOOTSTRAPS = 10000
SEED = 30092026
COLORS = {"apo": "#6247AA", "holo": "#718355"}


def trajectory_blocks(records: list[dict[str, str]]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["trajectory_slot"]].append(index)
    return [np.asarray(indices, dtype=int) for indices in groups.values()]


def bootstrap_fraction(
    cracked: np.ndarray, blocks: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    result = np.empty(BOOTSTRAPS, dtype=float)
    for replicate in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        positive = 0
        total = 0
        for block_index in chosen:
            indices = blocks[int(block_index)]
            positive += int(cracked[indices].sum())
            total += len(indices)
        result[replicate] = positive / total
    return result


def population_to_dg(cracked_fraction: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(cracked_fraction)
    # Clipping only guards pathological bootstrap replicates; no observed or
    # bootstrap population is expected at exactly zero or one.
    p = np.clip(p, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    dg = -KBT * np.log(p / (1.0 - p))
    return float(dg) if dg.ndim == 0 else dg


def save_panel(
    values: dict[str, float], cis: dict[str, list[float]],
    cracked_populations: dict[str, float],
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "font.weight": "normal",
        "axes.labelsize": 21,
        "axes.labelweight": "normal",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6.5,
        "ytick.major.size": 6.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })
    conditions = ("apo", "holo")
    x = np.arange(2)
    y = np.asarray([values[c] for c in conditions])
    low = np.asarray([cis[c][0] for c in conditions])
    high = np.asarray([cis[c][1] for c in conditions])
    errors = np.vstack((y - low, high - y))

    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    ax.bar(
        x, y, width=0.58, color=[COLORS[c] for c in conditions],
        yerr=errors, capsize=5,
        error_kw={"elinewidth": 1.7, "capthick": 1.7, "ecolor": "#303030"},
    )
    ax.axhline(0, color="#303030", linewidth=1.25)
    ax.set_xticks(x, ("apo", "glucose-bound"))
    ax.set_ylabel(r"$\Delta G$ (kcal mol$^{-1}$)")
    ax.set_xlim(-0.68, 1.68)
    ax.set_ylim(-2.05, 0.58)
    ax.grid(axis="y", alpha=0.16, linewidth=0.7)
    ax.set_axisbelow(True)
    # Put each numerical free energy below its lower confidence-limit cap so
    # the value remains centered on its bar without crossing the error bar.
    for position, value, lower_limit in zip(x, y, low):
        ax.text(
            position, lower_limit - 0.07, f"{value:.2f}",
            ha="center", va="top", fontsize=16, color="#303030",
        )
    # Population and free energy are mathematically linked, so the population
    # is reported as an annotation rather than repeated in a second panel.
    for position, condition in zip(x, conditions):
        ax.text(
            position, 0.34,
            f"{100.0 * cracked_populations[condition]:.1f}%\ncracked",
            ha="center", va="center", fontsize=14, linespacing=0.95,
            color=COLORS[condition],
        )
    fig.subplots_adjust(left=0.235, right=0.98, bottom=0.215, top=0.96)
    stem = PANEL_OUT / "panel_e_tm7b_cracking_free_energy"
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    archive = np.load(OUT / "tm7b_cracking_all_frames.npz")
    records = list(csv.DictReader((OUT / "selected_frame_provenance.csv").open()))
    condition_records = {"apo": records[:10000], "holo": records[10000:]}
    rng = np.random.default_rng(SEED)
    point = {}
    ci = {}
    population = {}
    fraction_bootstrap = {}
    dg_bootstrap = {}
    report = {
        "temperature_K": TEMPERATURE_K,
        "kBT_kcal_per_mol": KBT,
        "definition": (
            "DeltaG_cracked-intact = -kBT ln(P_cracked/P_intact); negative values "
            "mean that cracked TM7b is favored."
        ),
        "scope": (
            "10,000 MSM-weighted frames per condition; holo is conditional on "
            "glucose remaining in the binding pocket."
        ),
        "uncertainty": (
            f"{BOOTSTRAPS} trajectory-slot block-bootstrap replicates, 95% interval"
        ),
        "systems": {},
    }
    for condition in ("apo", "holo"):
        cracked = archive[f"{condition}_cracked"].astype(bool)
        p = float(cracked.mean())
        population[condition] = p
        fractions = bootstrap_fraction(
            cracked, trajectory_blocks(condition_records[condition]), rng
        )
        dgs = population_to_dg(fractions)
        dg = population_to_dg(p)
        interval = np.quantile(dgs, (0.025, 0.975)).tolist()
        point[condition] = dg
        ci[condition] = interval
        fraction_bootstrap[condition] = fractions
        dg_bootstrap[condition] = dgs
        report["systems"][condition] = {
            "frames": int(len(cracked)),
            "trajectory_slots": int(
                len(trajectory_blocks(condition_records[condition]))
            ),
            "cracked_fraction": p,
            "intact_fraction": 1.0 - p,
            "deltaG_cracked_minus_intact_kcal_per_mol": dg,
            "deltaG_ci95_kcal_per_mol": interval,
        }
    delta_delta = dg_bootstrap["holo"] - dg_bootstrap["apo"]
    report["holo_minus_apo_delta_deltaG_kcal_per_mol"] = {
        "estimate": point["holo"] - point["apo"],
        "ci95": np.quantile(delta_delta, (0.025, 0.975)).tolist(),
        "interpretation": (
            "Negative values indicate stabilization of cracked relative to intact "
            "TM7b in the glucose-bound ensemble."
        ),
    }
    np.savez_compressed(
        OUT / "tm7b_cracking_free_energy_bootstrap.npz",
        apo_cracked_fraction=fraction_bootstrap["apo"],
        holo_cracked_fraction=fraction_bootstrap["holo"],
        apo_deltaG=dg_bootstrap["apo"],
        holo_deltaG=dg_bootstrap["holo"],
        holo_minus_apo_delta_deltaG=delta_delta,
    )
    summary_path = OUT / "tm7b_cracking_free_energy_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    save_panel(point, ci, population)

    manifest_path = OUT / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in (
        summary_path,
        OUT / "tm7b_cracking_free_energy_bootstrap.npz",
        *sorted(PANEL_OUT.glob("panel_e_tm7b_cracking_free_energy.*")),
    ):
        relative = str(path.relative_to(OUT))
        manifest["integrity"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
