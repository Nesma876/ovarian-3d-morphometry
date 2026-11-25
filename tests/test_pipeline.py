"""Synthetic integration test for the published analysis stages."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_analyses as analysis


class PipelineTest(unittest.TestCase):
    def test_fixed_mask_size_resampling_and_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "synthetic.zip"
            archive.touch()
            manifest = root / "manifest.csv"
            selection = root / "selection.csv"
            output = root / "outputs"
            row = {"acquisition_id": "synthetic", "participant_id": "synthetic-participant"}
            pd.DataFrame([row]).to_csv(selection, index=False)
            pd.DataFrame([{**row, "segmentation_zip_member": "synthetic-member"}]).to_csv(manifest, index=False)
            mask = np.zeros((20, 20, 20), dtype=np.uint8)
            mask[4:8, 4:8, 4:8] = 3
            mask[8:12, 8:12, 8:12] = 3
            with patch.object(analysis, "read_segmentation_from_zip", return_value=(mask, (1., 1., 1.), None)), patch.object(
                sys, "argv", ["run_analyses.py", "--archive", str(archive), "--manifest", str(manifest),
                              "--selection", str(selection), "--output", str(output), "all"]
            ):
                analysis.main()
            counts = pd.read_csv(output / "connectivity.csv").iloc[0]
            self.assertEqual((counts.abs_count_diff_6_18, counts.abs_count_diff_6_26), (0, 1))
            components = pd.read_csv(output / "component_correspondence.csv")
            self.assertEqual(len(components), 2)
            self.assertEqual(components.group_26.nunique(), 1)
            sizes = pd.read_csv(output / "size_sensitivity_summary.csv")
            self.assertEqual(len(sizes), 12)
            self.assertEqual(len(pd.read_csv(output / "threshold_sensitivity.csv")), 12)
            self.assertEqual(len(pd.read_csv(output / "object_matching_sensitivity.csv")), 28)
