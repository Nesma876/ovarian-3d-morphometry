#!/usr/bin/env python3
"""Build the object-matching sensitivity figure from a local result table."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"nearest": "#0072B2", "linear": "#D55E00"}
ORDER = [
    "any_overlap",
    "intersection_ge_2",
    "intersection_ge_5",
    "original_fraction_ge_0.01",
    "original_fraction_ge_0.05",
    "pair_iou_ge_0.05",
    "pair_iou_ge_0.10",
]
REQUIRED = {
    "criterion", "factor", "method", "merger_events", "fragmentation_events",
    "fraction_original_matched", "mean_abs_diameter_error_mm",
}


def summarise(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    unknown = set(data["criterion"].dropna()).difference(ORDER)
    if unknown:
        raise ValueError(f"Unknown matching criteria: {sorted(unknown)}")
    return data.groupby(["criterion", "factor", "method"], as_index=False).agg(
        mergers=("merger_events", "sum"),
        fragments=("fragmentation_events", "sum"),
        matched=("fraction_original_matched", "mean"),
        diameter=("mean_abs_diameter_error_mm", "mean"),
    )


def build_figure(data: pd.DataFrame, output: Path) -> None:
    aggregate = summarise(data)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for (factor, method), group in aggregate.groupby(["factor", "method"]):
        group = group.set_index("criterion").reindex(ORDER)
        style = "-" if factor == 1.5 else "--"
        kwargs = {"marker": "o", "color": COLORS[method], "ls": style,
                  "label": f"{method}, x{factor:g}"}
        axes[0, 0].plot(range(1, 8), group.mergers, **kwargs)
        axes[0, 1].plot(range(1, 8), group.fragments, **kwargs)
        axes[1, 0].plot(range(1, 8), group.matched, **kwargs)
        axes[1, 1].plot(range(1, 8), group.diameter, **kwargs)
    titles = ["Graph-defined merger events", "Graph-defined fragmentation events",
              "Fraction of original objects matched", "Mean absolute diameter error"]
    ylabels = ["Events", "Events", "Fraction", "mm"]
    for panel, title, ylabel, letter in zip(axes.ravel(), titles, ylabels, "ABCD"):
        panel.set(xticks=range(1, 8), xlabel="Matching criterion", ylabel=ylabel, title=title)
        panel.grid(axis="y", alpha=0.2)
        panel.text(0.01, 0.98, letter, transform=panel.transAxes,
                   va="top", fontweight="bold")
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path, help="Local object-matching result table")
    parser.add_argument("output", type=Path, help="Output PDF or PNG")
    args = parser.parse_args()
    build_figure(pd.read_csv(args.input_csv), args.output)


if __name__ == "__main__":
    main()
