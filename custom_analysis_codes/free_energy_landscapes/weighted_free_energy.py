#!/usr/bin/env python3
"""Calculate a two-dimensional MSM-weighted free-energy landscape."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KB_KCAL_PER_MOL_K = 0.00198720425864083


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="NPZ containing x, y, and weight arrays")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=310.0)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--xlabel", default="Extracellular gate (nm)")
    parser.add_argument("--ylabel", default="Intracellular gate (nm)")
    args = parser.parse_args()

    data = np.load(args.input)
    x, y, weight = (np.asarray(data[key], dtype=float)
                    for key in ("x", "y", "weight"))
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight >= 0)
    x, y, weight = x[valid], y[valid], weight[valid]
    weight = weight / weight.sum()
    hist, xedge, yedge = np.histogram2d(x, y, bins=args.bins, weights=weight)
    with np.errstate(divide="ignore"):
        free_energy = -KB_KCAL_PER_MOL_K * args.temperature * np.log(hist)
    free_energy -= np.nanmin(free_energy[np.isfinite(free_energy)])
    free_energy[~np.isfinite(free_energy)] = np.nan

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_prefix.with_suffix(".npz"),
                        free_energy_kcal_mol=free_energy,
                        x_edges=xedge, y_edges=yedge)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    mesh = ax.pcolormesh(xedge, yedge, free_energy.T, shading="auto",
                         cmap="viridis")
    ax.set(xlabel=args.xlabel, ylabel=args.ylabel)
    fig.colorbar(mesh, ax=ax, label=r"$\Delta G$ (kcal mol$^{-1}$)")
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300)


if __name__ == "__main__":
    main()
