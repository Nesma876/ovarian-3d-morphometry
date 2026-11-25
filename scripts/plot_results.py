"""Build manuscript-style plots from locally generated analysis tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def read(name):
        return pd.read_csv(args.inputs / (name + ".csv"))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(args.output / (name + ".pdf"))
        plt.close(fig)

    count = read("connectivity")
    delta = count.abs_count_diff_6_26
    by_person = count.assign(affected=delta > 0).groupby("participant_id").affected.agg(["size", "mean"])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].bar(delta.value_counts().sort_index().index, delta.value_counts().sort_index().values)
    ax[0].set(xlabel="Absolute count difference", ylabel="Acquisitions")
    ax[1].scatter(by_person["size"], by_person["mean"], alpha=.6)
    ax[1].set(xlabel="Acquisitions per participant", ylabel="Affected fraction")
    save(fig, "connectivity")

    cats = ["[0,2)", "[2,5)", "[5,10)", "[10,inf)"]
    transitions = read("morphometric_transitions").pivot(
        index="source_category", columns="group_category", values="source_components"
    ).reindex(index=cats, columns=cats).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(transitions, cmap="Blues")
    for (y, x), n in np.ndenumerate(transitions.to_numpy()):
        if n:
            ax.text(x, y, str(int(n)), ha="center", va="center",
                    color="white" if n > transitions.max().max() / 2 else "black")
    ax.set_xticks(range(4), cats)
    ax.set_yticks(range(4), cats)
    ax.set(xlabel="26-connected group diameter (mm)", ylabel="Six-connected source diameter (mm)")
    fig.colorbar(image, ax=ax, label="Source components")
    save(fig, "morphometric_redistribution")

    size = read("size_sensitivity_summary")
    size = size[size.protocol == "B_common_6_reference_mask"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a, metric, label in zip(ax, [size.affected_fraction, size.category_crossing_groups,
                                    100 * size.mean_common_volume_retained],
                                ["Affected fraction", "Category-crossing groups", "Foreground retained (%)"]):
        a.plot(size.threshold_mm, metric, marker="o")
        a.set(xlabel="Minimum diameter (mm)", ylabel=label)
    save(fig, "size_sensitivity")

    paired = read("representation_matched_results")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    rng = np.random.default_rng(20261120)
    for a, col, label in zip(ax, ["delta_dice_linear_minus_nearest", "delta_count_error_linear_minus_nearest"],
                            ["Dice difference (linear minus nearest)", "Count-error difference (linear minus nearest)"]):
        for factor, rows in paired.groupby("spacing_factor"):
            a.scatter(factor + rng.uniform(-.05, .05, len(rows)), rows[col], alpha=.5)
        a.axhline(0, color="grey", linewidth=.8)
        a.set(xticks=[1.5, 2.0], xlabel="Round-trip factor", ylabel=label)
    save(fig, "paired_resampling")


if __name__ == "__main__":
    main()
