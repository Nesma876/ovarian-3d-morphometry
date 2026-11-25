import csv
import hashlib
import hmac
import re
from pathlib import Path

ALLOWED_SHAREABLE_FIELDS = {
    "acquisition_id", "participant_id", "cycle_id", "visit_id", "side", "fold",
    "spacing_z_mm", "spacing_y_mm", "spacing_x_mm", "image_quality",
    "complete_structure", "best_series_organ_instance", "status",
    "external_fold", "model_id", "code_version", "checkpoint_sha256",
    "provenance_status", "prediction_path", "preprocessing", "postprocessing",
    "n_reference", "n_predicted", "signed_count_error", "absolute_count_error",
    "relative_count_error", "false_positives", "false_negatives", "fusions",
    "fragmentations", "n_acquisitions", "mean_absolute_count_error",
    "total_fusions", "total_fragmentations",
}

FORBIDDEN_FIELD_NAMES = {
    "imagecomments", "patientid", "patientname", "studydate", "studytime",
    "studyinstanceuid", "seriesinstanceuid", "sopinstanceuid", "frameofreferenceuid",
    "acquisitiondatetime", "accessionnumber", "operatorsname", "deviceserialnumber",
    "raw_dcm_path", "segmentation_dcm_path", "source_path", "date", "cycle_start_date",
}

UID_PATTERN = re.compile(r"(?<![\d.])(?:\d+\.){3,}\d+(?![\d.])")


def pseudonym(kind: str, value: str, key: bytes, length: int = 16) -> str:
    digest = hmac.new(key, f"{kind}\0{value}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{kind}-{digest[:length]}"


def audit_csv(path: Path, allowed_fields=ALLOWED_SHAREABLE_FIELDS) -> list[str]:
    issues = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for field in fields:
            if field.lower() in FORBIDDEN_FIELD_NAMES:
                issues.append(f"forbidden field: {field}")
            if field not in allowed_fields:
                issues.append(f"field not whitelisted: {field}")
        for row_number, row in enumerate(reader, start=2):
            for field, value in row.items():
                if value and UID_PATTERN.search(value):
                    issues.append(f"technical UID-like value at row {row_number}, field {field}")
    return issues
