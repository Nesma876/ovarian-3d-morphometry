#!/usr/bin/env python3
"""Create privacy-preserving metadata without extracting or modifying Data.zip."""
import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from follicle_trust.privacy import ALLOWED_SHAREABLE_FIELDS, pseudonym


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=ROOT / "data" / "Data.zip")
    p.add_argument("--key-file", type=Path, default=ROOT / "private" / "pseudonym.key")
    p.add_argument("--init-key", action="store_true", help="Create a local ignored key if absent")
    return p.parse_args()


def load_key(path, init):
    if not path.exists():
        if not init:
            raise SystemExit(f"missing pseudonym key: {path}; rerun once with --init-key")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(secrets.token_bytes(32))
    key = path.read_bytes()
    if len(key) < 32:
        raise SystemExit("pseudonym key must contain at least 32 bytes")
    return key


def main():
    args = parse_args()
    key = load_key(args.key_file, args.init_key)
    internal_dir = ROOT / "artifacts" / "internal"
    shareable_dir = ROOT / "artifacts" / "shareable"
    internal_dir.mkdir(parents=True, exist_ok=True)
    shareable_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        metadata_names = [n for n in archive.namelist() if n.endswith("image_metadata.csv")]
        if len(metadata_names) != 1:
            raise SystemExit(f"expected one image_metadata.csv, found {len(metadata_names)}")
        source = csv.DictReader(io.TextIOWrapper(archive.open(metadata_names[0]), encoding="utf-8-sig"))
        rows = list(source)
        members = set(archive.namelist())

    dates_by_patient = defaultdict(set)
    for row in rows:
        dates_by_patient[row["PatientID"]].add(row["Date"])
    visit_ord = {p: {d: i + 1 for i, d in enumerate(sorted(ds))} for p, ds in dates_by_patient.items()}
    clean, internal = [], []
    missing = []
    for row in rows:
        patient = row["PatientID"]
        sop = row["SOPInstanceUID"]
        participant_id = pseudonym("pt", patient, key)
        acquisition_id = pseudonym("acq", sop, key)
        spacing = [float(v) for v in row["spacing"].split(",")]
        if len(spacing) != 3 or any(v <= 0 for v in spacing):
            raise ValueError(f"invalid spacing for {acquisition_id}")
        filename = f"{sop}.dcm"
        raw_member = f"Data/BiCycleStudy/raw_ultrasound_images/{filename}"
        seg_member = f"Data/BiCycleStudy/validated_segmentations/{filename}"
        status = "paired" if raw_member in members and seg_member in members else "missing_member"
        if status != "paired":
            missing.append(acquisition_id)
        clean.append({
            "acquisition_id": acquisition_id,
            "participant_id": participant_id,
            "cycle_id": f"{participant_id}-cycle-{int(row['cycle_name']):02d}",
            "visit_id": f"{participant_id}-visit-{visit_ord[patient][row['Date']]:02d}",
            "side": row["Location"], "fold": row["fold"],
            "spacing_z_mm": spacing[0], "spacing_y_mm": spacing[1], "spacing_x_mm": spacing[2],
            "image_quality": row["ImageQuality"] or "Missing",
            "complete_structure": row["CompleteStructure"] or "Missing",
            "best_series_organ_instance": row["BestSeriesOrganInstance"] or "Missing",
            "status": status,
        })
        internal.append({
            "acquisition_id": acquisition_id, "participant_id": participant_id,
            "original_patient_id": patient, "sop_instance_uid": sop,
            "raw_zip_member": raw_member, "segmentation_zip_member": seg_member,
        })
    clean.sort(key=lambda r: r["acquisition_id"])
    clean_path = shareable_dir / "folliclefinder_metadata.csv"
    with clean_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(ALLOWED_SHAREABLE_FIELDS))
        writer.writeheader(); writer.writerows(clean)
    internal_path = internal_dir / "uid_manifest.csv"
    with internal_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(internal[0]))
        writer.writeheader(); writer.writerows(internal)
    hasher = hashlib.sha256()
    with args.zip.open("rb") as source_file:
        for block in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    checksum = hasher.hexdigest()
    report = {"source_sha256": checksum, "rows": len(clean), "participants": len(dates_by_patient),
              "missing_pairs": len(missing), "shareable": str(clean_path), "internal": str(internal_path)}
    (internal_dir / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "internal"}, indent=2))


if __name__ == "__main__":
    main()
