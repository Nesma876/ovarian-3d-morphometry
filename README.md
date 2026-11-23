# Ovarian 3D morphometry

Reproducibility material for the study of connectivity-dependent morphometric variability in semantic 3D ovarian ultrasound masks.

This repository contains analysis code, documentation, synthetic unit tests, and one pseudonymised illustrative figure. It intentionally excludes original ultrasound/DICOM data, segmentation files, derived result tables, private identifiers, linkage tables, and local manifests.

## Reproduction

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest tests -q
python scripts/build_matching_sensitivity_figure.py /path/to/local_results.csv output.pdf
```

The input CSV remains local and is not distributed in this repository. The script validates its required schema before plotting.

## Repository contents

- `figures/`: one pseudonymised real-acquisition illustration of the study's central contribution.
- `scripts/`: portable figure generation from a user-supplied local result table.
- `tests/`: synthetic schema, aggregation, and rendering checks.
- `DATASET_ACCESS.md`: source-data access and non-redistribution information.

## Data and licence

Original FollicleFinder images and segmentations are not redistributed. See `DATASET_ACCESS.md`. No software licence has yet been granted; unless a licence is added, copyright remains with the repository owner and reuse requires permission.
