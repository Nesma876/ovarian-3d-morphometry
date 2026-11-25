# From Semantic Masks to Object Measurements: Connectivity-Dependent Morphometric Variability in 3D Ovarian Ultrasound

This repository contains the analysis code for the final manuscript by Nesma Settouti and Mohammed El Amine Bechar. It computes connected-component counts, correspondence and contacts, equivalent diameters, size-filtering sensitivity, round-trip resampling and matching sensitivity from locally accessed FollicleFinder masks.

## Reproduce locally

Use Python 3.12 in a virtual environment. Obtain FollicleFinder separately as described in [Data access](DATASET_ACCESS.md). Run the following commands from the repository root.

```sh
python -m pip install -r requirements.txt
python scripts/summarize.py --check-library-defaults scipy scikit-image
python scripts/prepare_folliclefinder.py --zip data/Data.zip --init-key
python scripts/run_analyses.py --archive data/Data.zip --manifest artifacts/internal/uid_manifest.csv --metadata artifacts/shareable/folliclefinder_metadata.csv --output local-output all
python scripts/summarize.py --inputs local-output --output local-output/connectivity_summary.json
python scripts/plot_results.py --inputs local-output --output local-output/figures
```

The preparation step writes a private linkage manifest and a local pseudonym key. Keep both local along with all per-acquisition, per-participant and component-level outputs. The manuscript and its supplement provide aggregate reference values. The code does not download the archive or upload results.

The selection algorithm orders participants by local pseudonym, then applies the image-quality and spacing rule in the manuscript. A new random pseudonym key can change the participant ordering, tied acquisition ordering and finite-bootstrap samples. To repeat the exact historical subgroup and intervals, use the original private preparation context (or supply the original local study selection with `--selection` alongside its matching manifest). The original key, linkage and selection are excluded from the public repository. Fixed-mask cohort point estimates are independent of the key. The paper's real ultrasound illustration is excluded from this repository.

## Outputs

| Analysis | Output under local-output |
|---|---|
| Fixed-mask connectivity and volume-equivalent morphometry | connectivity.csv, component_correspondence.csv, merger_groups.csv, diagonal_contacts.csv, morphometric_transitions.csv |
| Size sensitivity | size_sensitivity_summary.csv and local per-acquisition table |
| Round-trip interpolation, overlap and count changes | threshold_sensitivity.csv, representation_matched_results.csv and label_propagation_results.csv |
| Object matching | object_matching_sensitivity.csv, fixed_common_correspondences.csv and object_matching_paired_statistics.csv |
| Main aggregate contrasts | connectivity_summary.json |
| Cohort, morphometry, filtering and paired resampling plots | figures/ |
```sh
python -m unittest discover -s tests -v
```

The archive-level workflow requires the 533 paired acquisitions. The synthetic test runs all four stages on a fabricated mask. Compare generated aggregates with the manuscript and its supplement.
