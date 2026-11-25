"""Morphometry in physical (millimetre) space."""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def equivalent_diameter_mm(volume_mm3: float) -> float:
    return float((6.0 * volume_mm3 / math.pi) ** (1.0 / 3.0)) if volume_mm3 > 0 else 0.0


def _surface_area_faces(mask: np.ndarray, spacing) -> float:
    """Anisotropy-aware exposed-voxel-face area (not a smooth mesh estimate)."""
    z, y, x = map(float, spacing)
    padded = np.pad(mask.astype(bool), 1)
    area = 0.0
    for axis, face_area in ((0, y * x), (1, z * x), (2, z * y)):
        area += np.count_nonzero(np.diff(padded.astype(np.int8), axis=axis)) * face_area
    return float(area)


def _principal_dimensions(coords: np.ndarray, spacing):
    if len(coords) < 4:
        return (float("nan"),) * 4
    xyz = coords * np.asarray(spacing, dtype=float)
    covariance = np.cov(xyz, rowvar=False)
    values = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    # For a uniform solid ellipsoid, covariance eigenvalue = semi-axis^2 / 5.
    dims = 2.0 * np.sqrt(5.0 * values)
    elongation = float(dims[0] / dims[-1]) if dims[-1] > 0 else float("nan")
    return float(dims[0]), float(dims[1]), float(dims[2]), elongation


def extract_morphometry(labels: np.ndarray, spacing_mm, ovarian_mask=None):
    labels = np.asarray(labels)
    spacing = tuple(map(float, spacing_mm))
    if labels.ndim != 3 or len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError("expected a 3-D label image and three positive spacings")
    ids = np.unique(labels)
    ids = ids[ids > 0]
    voxel_volume = float(np.prod(spacing))
    objects = ndi.find_objects(labels)
    rows = []
    boundaries = {}
    boundary_coords = {}
    structure = ndi.generate_binary_structure(3, 1)
    for instance_id in ids:
        slc = objects[int(instance_id) - 1]
        local = labels[slc] == instance_id
        count = int(local.sum())
        volume = count * voxel_volume
        area = _surface_area_faces(local, spacing)
        coords = np.argwhere(labels == instance_id)
        major, intermediate, minor, elongation = _principal_dimensions(coords, spacing)
        sphericity = (math.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area) if area else float("nan")
        boundary = np.logical_xor(labels == instance_id, ndi.binary_erosion(labels == instance_id, structure=structure))
        boundaries[int(instance_id)] = boundary
        boundary_coords[int(instance_id)] = np.argwhere(boundary) * np.asarray(spacing)
        rows.append({
            "instance_id": int(instance_id), "voxel_count": count,
            "volume_mm3": volume, "equivalent_diameter_mm": equivalent_diameter_mm(volume),
            "surface_area_mm2_voxel_faces": area, "sphericity_voxel_faces": float(sphericity),
            "principal_diameter_major_mm": major, "principal_diameter_intermediate_mm": intermediate,
            "principal_diameter_minor_mm": minor, "elongation": elongation,
            "component_count": 1,
        })
    nearest = {int(i): float("inf") for i in ids}
    trees = {iid: cKDTree(coords) for iid, coords in boundary_coords.items() if len(coords)}
    ordered = list(trees)
    for pos, left in enumerate(ordered):
        for right in ordered[pos + 1:]:
            # Exact closest pair, querying the smaller boundary against the larger tree.
            lc, rc = boundary_coords[left], boundary_coords[right]
            if len(lc) <= len(rc): distance = float(trees[right].query(lc, k=1)[0].min())
            else: distance = float(trees[left].query(rc, k=1)[0].min())
            nearest[left] = min(nearest[left], distance)
            nearest[right] = min(nearest[right], distance)
    ovary_distance = (ndi.distance_transform_edt(np.asarray(ovarian_mask, bool), sampling=spacing)
                      if ovarian_mask is not None else None)
    for row in rows:
        iid = row["instance_id"]
        row["nearest_instance_boundary_mm"] = nearest[iid] if np.isfinite(nearest[iid]) else float("nan")
        if ovary_distance is not None:
            row["distance_to_ovary_boundary_mm"] = float(ovary_distance[boundaries[iid]].min())
        else:
            row["distance_to_ovary_boundary_mm"] = float("nan")
    return rows
