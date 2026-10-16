#!/usr/bin/env python3
"""Plot pooled-control PfHT1 growth rates in ascending mutation order.

The normalization reproduces the original combined figure:
    normalized_mu = (mu - pooled_empty) / (pooled_WT - pooled_empty)
calculated separately for glucose and fructose. Therefore, pooled empty = 0
and pooled WT = 1 for each substrate.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"glucose": "#4C78A8", "fructose": "#E45756"}
CONTROLS = ["PfHT1_WT", "empty"]


def mutation_label(transformant: str) -> str:
    """Return only the amino-acid mutation, without plate-position labels."""
    if transformant == "PfHT1_WT":
        return "WT"
    if transformant == "empty":
        return "empty"
    return transformant.rsplit("_", 1)[-1]


def controls_then_mutations_ascending(
    summary: pd.DataFrame, value: str
) -> pd.DataFrame:
    """Keep controls first and rank mutations from lowest to highest."""
    control_rows = []
    for control in CONTROLS:
        control_rows.append(summary.loc[summary["transformant"].eq(control)])
    mutations = summary.loc[~summary["transformant"].isin(CONTROLS)]
    mutations = mutations.sort_values(value, ascending=True, kind="stable")
    ranked = pd.concat([*control_rows, mutations], ignore_index=True)
    ranked.insert(0, "plot_order", np.arange(1, len(ranked) + 1))
    ranked.insert(2, "plot_label", ranked["transformant"].map(mutation_label))
    return ranked


def summarize(data: pd.DataFrame, value: str, prefix: str) -> pd.DataFrame:
    return (
        data.groupby(["condition", "transformant"], as_index=False)
        .agg(
            **{
                f"{prefix}_mean": (value, "mean"),
                f"{prefix}_sd": (value, "std"),
                "n": (value, "size"),
            }
        )
    )


def draw_single(
    replicates: pd.DataFrame,
    summary: pd.DataFrame,
    condition: str,
    value: str,
    mean_col: str,
    sd_col: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(18, 7.5), constrained_layout=True)
    ax.bar(
        x,
        summary[mean_col],
        yerr=summary[sd_col].fillna(0),
        width=0.72,
        color=COLORS[condition],
        edgecolor="#333333",
        linewidth=0.7,
        capsize=3,
        error_kw={"elinewidth": 1.1},
        alpha=0.9,
        zorder=2,
    )

    positions = dict(zip(summary["transformant"], x))
    for transformant, group in replicates.groupby("transformant", sort=False):
        offsets = (
            np.array([0.0])
            if len(group) == 1
            else np.linspace(-0.16, 0.16, len(group))
        )
        ax.scatter(
            positions[transformant] + offsets,
            group[value],
            s=20,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.65,
            alpha=0.85,
            zorder=3,
        )

    ax.axhline(0, color="#555555", linewidth=0.9, zorder=1)
    ax.set_xticks(x, summary["plot_label"], rotation=55, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.margins(x=0.01)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def draw_combined(
    normalized: pd.DataFrame,
    glucose_order: list[str],
    output: Path,
) -> pd.DataFrame:
    summary = summarize(normalized, "mu_norm", "mu_norm")
    lookup = summary.set_index(["condition", "transformant"])
    rows = []
    for order, transformant in enumerate(glucose_order, start=1):
        row = {
            "plot_order": order,
            "transformant": transformant,
            "plot_label": mutation_label(transformant),
        }
        for condition in ("glucose", "fructose"):
            values = lookup.loc[(condition, transformant)]
            row[f"{condition}_mean"] = values["mu_norm_mean"]
            row[f"{condition}_sd"] = values["mu_norm_sd"]
            row[f"{condition}_n"] = int(values["n"])
        rows.append(row)
    combined = pd.DataFrame(rows)

    x = np.arange(len(combined))
    width = 0.38
    fig, ax = plt.subplots(figsize=(24, 10), constrained_layout=True)
    for offset, condition in ((-width / 2, "glucose"), (width / 2, "fructose")):
        ax.bar(
            x + offset,
            combined[f"{condition}_mean"],
            yerr=combined[f"{condition}_sd"].fillna(0),
            width=width,
            color=COLORS[condition],
            edgecolor="#333333",
            linewidth=0.65,
            capsize=2.5,
            error_kw={"elinewidth": 1.0},
            alpha=0.92,
            label=condition.capitalize(),
            zorder=2,
        )

    ax.axhline(0, color="#555555", linewidth=0.9, zorder=1)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xticks(x, combined["plot_label"], rotation=55, ha="right")
    ax.tick_params(axis="x", labelsize=21)
    ax.tick_params(axis="y", labelsize=21)
    ax.set_ylabel(
        "Normalized growth rate", fontsize=26
    )
    ax.legend(frameon=False, fontsize=23)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.margins(x=0.01)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return combined


def main() -> None:
    project = Path(os.environ.get("PFHT1_PROJECT_ROOT", ".")).expanduser().resolve()
    default_data = project / "experiments"
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=Path,
                        default=default_data / "growth_rates_0_20h_replicates.csv")
    parser.add_argument("--normalized", type=Path,
                        default=default_data / "growth_rates_0_20h_normalized_replicates.csv")
    parser.add_argument("--output", type=Path,
                        default=default_data / "pooled_normalized_growth_plots")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rates = pd.read_csv(args.rates)
    normalized = pd.read_csv(args.normalized)

    normalized_summary = summarize(normalized, "mu_norm", "mu_norm")

    pooled_wt = (
        rates.loc[rates["transformant"].eq("PfHT1_WT")]
        .groupby("condition")["mu_0_20"]
        .mean()
        .rename("pooled_wt_mean")
    )
    delta = rates.join(pooled_wt, on="condition")
    delta["delta_mu_from_pooled_wt"] = delta["mu_0_20"] - delta["pooled_wt_mean"]
    delta_summary = summarize(delta, "delta_mu_from_pooled_wt", "delta_mu")

    glucose_normalized_order = None
    for condition in ("glucose", "fructose"):
        condition_norm = normalized.loc[normalized["condition"].eq(condition)].copy()
        norm_ranked = controls_then_mutations_ascending(
            normalized_summary.loc[
                normalized_summary["condition"].eq(condition)
            ].copy(),
            "mu_norm_mean",
        )
        norm_ranked.to_csv(
            args.output / f"normalized_growth_{condition}_ascending.csv", index=False
        )
        draw_single(
            replicates=condition_norm,
            summary=norm_ranked,
            condition=condition,
            value="mu_norm",
            mean_col="mu_norm_mean",
            sd_col="mu_norm_sd",
            ylabel="Normalized growth rate",
            title=(
                f"{condition.capitalize()}: pooled-control normalized growth ± SD\n"
                "WT and empty first; mutations ordered from lowest to highest growth"
            ),
            output=args.output / f"normalized_growth_{condition}_ascending.png",
        )
        if condition == "glucose":
            glucose_normalized_order = norm_ranked["transformant"].tolist()

        condition_delta = delta.loc[delta["condition"].eq(condition)].copy()
        delta_ranked = controls_then_mutations_ascending(
            delta_summary.loc[delta_summary["condition"].eq(condition)].copy(),
            "delta_mu_mean",
        )
        delta_ranked.to_csv(
            args.output / f"difference_from_pooled_wt_{condition}_ascending.csv",
            index=False,
        )
        draw_single(
            replicates=condition_delta,
            summary=delta_ranked,
            condition=condition,
            value="delta_mu_from_pooled_wt",
            mean_col="delta_mu_mean",
            sd_col="delta_mu_sd",
            ylabel=r"$\Delta\mu$ from pooled WT (h$^{-1}$)",
            title=(
                f"{condition.capitalize()}: difference from pooled WT ± SD\n"
                "WT and empty first; mutations ordered from lowest to highest growth"
            ),
            output=args.output / f"difference_from_pooled_wt_{condition}_ascending.png",
        )

    combined = draw_combined(
        normalized,
        glucose_normalized_order,
        args.output / "normalized_growth_glucose_fructose_glucose_order.png",
    )
    combined.to_csv(
        args.output / "normalized_growth_glucose_fructose_glucose_order.csv", index=False
    )


if __name__ == "__main__":
    main()
