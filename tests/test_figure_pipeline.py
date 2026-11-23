from pathlib import Path
import importlib.util

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_matching_sensitivity_figure.py"
spec = importlib.util.spec_from_file_location("matching_figure", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic_table():
    rows = []
    for criterion in module.ORDER:
        for factor in (1.5, 2.0):
            for method in ("nearest", "linear"):
                rows.append({
                    "criterion": criterion,
                    "factor": factor,
                    "method": method,
                    "merger_events": 1,
                    "fragmentation_events": 0,
                    "fraction_original_matched": 0.99,
                    "mean_abs_diameter_error_mm": 0.04,
                })
    return pd.DataFrame(rows)


def test_complete_grid_is_summarised():
    summary = module.summarise(synthetic_table())
    assert len(summary) == 7 * 2 * 2
    assert summary["criterion"].nunique() == 7


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        module.summarise(pd.DataFrame({"criterion": ["any_overlap"]}))


def test_unknown_criterion_is_rejected():
    data = synthetic_table()
    data.loc[0, "criterion"] = "unregistered_rule"
    with pytest.raises(ValueError, match="Unknown matching criteria"):
        module.summarise(data)


def test_figure_is_created(tmp_path):
    output = tmp_path / "figure.pdf"
    module.build_figure(synthetic_table(), output)
    assert output.stat().st_size > 1_000
