"""Privacy-preserving access to semantic DICOM segmentations stored in Data.zip."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pydicom


def read_segmentation_from_zip(zip_path: Path, member: str):
    with zipfile.ZipFile(zip_path) as archive:
        ds = pydicom.dcmread(io.BytesIO(archive.read(member)))
    shape = (int(ds.NumberOfFrames), int(ds.Rows), int(ds.Columns))
    expected = int(np.prod(shape))
    raw = np.frombuffer(ds.PixelData, dtype=np.uint8)
    if raw.size < expected:
        raise ValueError(f"truncated PixelData: expected {expected}, found {raw.size}")
    # DICOM values have even byte length; an odd number of uint8 voxels gains one pad byte.
    pixels = raw[:expected].reshape(shape)
    spacing = (
        float(getattr(ds, "SpacingBetweenSlices", ds.PixelSpacing[0])),
        float(ds.PixelSpacing[0]),
        float(ds.PixelSpacing[1]),
    )
    return pixels, spacing, ds


def read_dicom_pixels_from_zip(zip_path: Path, member: str):
    """Read a local DICOM volume; caller must not export source metadata."""
    with zipfile.ZipFile(zip_path) as archive:
        ds = pydicom.dcmread(io.BytesIO(archive.read(member)))
    return np.asarray(ds.pixel_array), ds


def reconstruct_instances(semantic: np.ndarray, follicle_label: int = 3, connectivity: int = 1):
    """Reconstruct instances as 3-D connected components of semantic label 3.

    This is an explicit dataset-specific assumption, not an expert instance label.
    connectivity=1 means face connectivity (6-neighbourhood), chosen to avoid joining
    objects that merely touch at an edge or corner.
    """
    from scipy import ndimage as ndi

    structure = ndi.generate_binary_structure(3, connectivity)
    labels, count = ndi.label(np.asarray(semantic) == follicle_label, structure=structure)
    return labels.astype(np.int32, copy=False), int(count)
