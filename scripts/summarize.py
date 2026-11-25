"""Recompute the reviewer contrasts from archived component and resampling tables."""
import csv
import json
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np


SEED = 20261124
REPLICATES = 20000


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def default_label_checks(packages):
    """Test the installed library defaults on minimal binary contact fixtures."""
    from importlib.metadata import version
    results = {}
    for package in packages:
        match package:
            case "scipy":
                from scipy import ndimage

                label = lambda mask: ndimage.label(mask)[0]
            case "scikit-image":
                from skimage import measure

                label = measure.label
            case "SimpleITK":
                import SimpleITK as sitk

                label = lambda mask: sitk.GetArrayFromImage(
                    sitk.ConnectedComponent(sitk.GetImageFromArray(mask))
                )
            case "connected-components-3d":
                import cc3d

                label = cc3d.connected_components
        counts = []
        for contact in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
            mask = np.zeros((2, 2, 2), dtype=np.uint8)
            mask[0, 0, 0] = mask[contact] = 1
            counts.append(int(label(mask).max()))
        assert counts == ([1, 2, 2] if package in {"scipy", "SimpleITK"} else [1, 1, 1])
        results[package] = {"version": version(package), "face_edge_vertex_counts": counts}
    return results


def spearman(x, y):
    def ranks(values):
        _, index, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts + 1) / 2)[index]

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main(inputs, output):
    connectivity = rows(inputs / "connectivity.csv")
    geometry = {
        row["acquisition_id"]: row
        for row in rows(inputs / "connectivity.csv")
    }
    components = rows(inputs / "component_correspondence.csv")
    source_counts, target_counts = Counter(), Counter()
    source_totals, target_totals = Counter(), Counter()
    targets = {}
    for row in components:
        aid = row["acquisition_id"]
        source_totals[aid] += 1
        source_counts[aid] += 2 <= float(row["source_diameter_mm"]) <= 10
        key = (aid, row["group_26"])
        diameter = float(row["group_diameter_mm"])
        assert targets.setdefault(key, diameter) == diameter
    for (aid, _), diameter in targets.items():
        target_totals[aid] += 1
        target_counts[aid] += 2 <= diameter <= 10

    people = sorted({row["participant_id"] for row in connectivity})
    aids = [row["acquisition_id"] for row in connectivity]
    assert len(people) == 29 and len(aids) == len(set(aids)) == 533
    assert set(aids) == set(geometry) and set(source_totals) == set(target_counts) <= set(aids)
    assert len(components) == 13775 and len(targets) == 13247
    n6 = np.array([source_totals[aid] for aid in aids])
    d18, d26 = (
        np.array([int(row[f"abs_count_diff_6_{adjacency}"]) for row in connectivity])
        for adjacency in (18, 26)
    )
    assert np.all(d18 <= d26) and d26.sum() == 528
    assert np.array_equal(n6 - d26, [target_totals[aid] for aid in aids])
    afc6, afc26 = (np.array([counts[aid] for aid in aids]) for counts in (source_counts, target_counts))
    spacing = np.array([
        [float(geometry[aid][f"spacing_{axis}_mm"]) for axis in "zyx"] for aid in aids
    ])
    assert np.allclose(spacing, spacing[:, :1]) and np.all(spacing > 0)
    groups = [np.flatnonzero([row["participant_id"] == person for row in connectivity]) for person in people]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(groups), size=(REPLICATES, len(groups)))
    totals = np.array([len(group) for group in groups])[draws].sum(axis=1)

    def interval(values):
        event_totals = np.array([values[group].sum() for group in groups])
        return [float(x) for x in np.quantile(event_totals[draws].sum(axis=1) / totals, [.025, .975])]

    def summarize(delta):
        return {
            "changed": int(np.count_nonzero(delta)),
            "changed_ci": interval(delta != 0),
            "absolute_quantiles_50_75_90_100": np.quantile(abs(delta), [.5, .75, .9, 1]).tolist(),
        }

    merged = rows(inputs / "merger_groups.csv")
    diameter_increase = [
        float(row["group_diameter_mm"]) - max(map(float, row["source_diameters_mm"].split(";")))
        for row in merged if int(row["multiplicity"]) > 1
    ]
    assert len(diameter_increase) == 433
    paired = rows(inputs / "representation_matched_results.csv")
    thresholds = rows(inputs / "threshold_sensitivity.csv")
    resampling = {}
    for factor in (1.5, 2.0):
        diffs = [
            float(row["delta_count_error_linear_minus_nearest"])
            for row in paired if float(row["spacing_factor"]) == factor
        ]
        better, tied, worse = sum(x < 0 for x in diffs), diffs.count(0), sum(x > 0 for x in diffs)
        assert len(diffs) == 29
        grid = {}
        for threshold in (.3, .4, .5, .6, .7):
            subset = [
                row for row in thresholds if row["method"] == "linear"
                and float(row["spacing_factor"]) == factor and float(row["threshold"]) == threshold
            ]
            assert len(subset) == len({row["participant_id"] for row in subset}) == 29
            grid[str(threshold)] = {
                metric: float(np.mean([float(row[metric]) for row in subset]))
                for metric in ("dice", "iou", "absolute_relative_volume_error")
            }
            grid[str(threshold)]["changed"] = sum(
                int(row["absolute_count_change"]) > 0 for row in subset
            )
        resampling[str(factor)] = {
            "linear_better_tied_worse": [better, tied, worse],
            "two_sided_sign_test_p": min(
                1., 2 * sum(comb(better + worse, k) for k in range(min(better, worse) + 1))
                / 2 ** (better + worse)
            ),
            "threshold_grid": grid,
        }

    afc_delta = afc26 - afc6
    spacing_rho = spearman(spacing[:, 0], d26)
    spacing_boot = []
    for draw in draws[:5000]:
        idx = np.concatenate([groups[i] for i in draw])
        if np.ptp(d26[idx]) and np.ptp(spacing[idx, 0]):
            spacing_boot.append(spearman(spacing[idx, 0], d26[idx]))
    assert np.all(d26[n6 == 0] == 0)
    result = {
        "acquisitions": len(aids),
        "participants": len(people),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "all_components_6_to_18": summarize(d18),
        "all_components_6_to_26": summarize(d26),
        "nonempty_acquisitions": int(np.count_nonzero(n6)),
        "relative_6_to_26_quantiles_50_75_90_100":
            np.quantile(d26[n6 > 0] / n6[n6 > 0], [.5, .75, .9, 1]).tolist(),
        "equivalent_diameter_2_to_10_mm": {
            **summarize(afc_delta),
            "decreased": int(np.count_nonzero(afc_delta < 0)),
            "increased": int(np.count_nonzero(afc_delta > 0)),
            "illustrative_cutoff_disagreements": {
                "lt5": int(np.count_nonzero((afc6 < 5) != (afc26 < 5))),
                "gt15": int(np.count_nonzero((afc6 > 15) != (afc26 > 15))),
                "gt20": int(np.count_nonzero((afc6 > 20) != (afc26 > 20))),
            },
        },
        "merger_diameter_gain_vs_largest_source_mm_quantiles_50_90_100":
            np.quantile(diameter_increase, [.5, .9, 1]).tolist(),
        "spacing_count_difference_spearman_rho": float(spacing_rho),
        "spacing_bootstrap_replicates": len(spacing_boot),
        "spacing_count_difference_cluster_ci": np.quantile(spacing_boot, [.025, .975]).tolist(),
        "round_trip_resampling": resampling,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-library-defaults", nargs="+",
        choices=("scipy", "scikit-image", "SimpleITK", "connected-components-3d"),
    )
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packages = args.check_library_defaults
    if packages:
        print(json.dumps(default_label_checks(packages), indent=2))
    else:
        if args.inputs is None or args.output is None:
            parser.error("--inputs and --output are required")
        main(args.inputs, args.output)
