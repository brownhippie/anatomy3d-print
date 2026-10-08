"""Carve a 3D body mesh from 2+ silhouettes shot from different angles
around the subject — the classical visual hull / space carving method
(Laurentini 1994; Kutulakos & Seitz 2000). Pure geometry, no learned model,
no license.

Each silhouette is a cone of space the subject must lie inside; the true
shape is bounded by the intersection of all those cones. We approximate
each view as orthographic (no calibrated camera, just an assumed rotation
angle around the vertical axis — accurate enough for a turntable-style
capture at roughly constant distance) and carve a voxel grid down to the
intersection.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from .mesh_types import BodyMesh


@dataclass
class SilhouetteView:
    mask: np.ndarray  # (H, W) bool
    angle_deg: float  # rotation around the vertical axis, 0 = front


def _view_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return row_min, row_max, col_min, col_max


def carve_visual_hull(
    views: list,
    target_height_mm: float = 150.0,
    voxel_resolution: int = 110,
    smooth_sigma: float = 0.8,
) -> BodyMesh:
    if len(views) < 2:
        raise ValueError("Visual hull carving needs at least 2 views.")

    bboxes = [_view_bbox(v.mask) for v in views]

    # Shared vertical extent: the tallest view's own row span sets the
    # world Y range, so views photographed slightly closer/further don't
    # each rescale the figure's height independently.
    row_spans = [rb[1] - rb[0] for rb in bboxes]
    ref_idx = int(np.argmax(row_spans))
    row_min_ref, row_max_ref = bboxes[ref_idx][0], bboxes[ref_idx][1]
    row_span_ref = row_max_ref - row_min_ref

    # One shared pixels-per-world-unit scale across every view (same
    # camera distance/zoom assumed for all shots, the standard turntable
    # setup). Deriving each view's horizontal scale from its OWN mask
    # width instead would silently cancel out the exact information
    # carving depends on: a narrower silhouette must map to fewer world
    # units than a wider one, not be re-stretched to match it.
    px_per_unit = row_span_ref / 2.0

    # The grid must extend past the carved content on every side, or the
    # isosurface gets clipped flat at the grid boundary (most visibly: an
    # open hole at the top of the head and bottom of the feet, since the
    # reference view's silhouette touches rows 0/±1 exactly). procedural_body.py
    # has the same requirement for the same reason.
    margin = 0.15
    n = voxel_resolution
    coords = np.linspace(-1 - margin, 1 + margin, n)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")

    inside = np.ones((n, n, n), dtype=bool)
    for view, (row_min, row_max, col_min, col_max) in zip(views, bboxes):
        a = np.radians(view.angle_deg)
        # Undo this view's turntable rotation to get into its own
        # camera-facing frame; under an orthographic assumption the visible
        # plane is just (local_x, Y).
        local_x = gx * np.cos(-a) - gz * np.sin(-a)
        col_center = (col_min + col_max) / 2.0

        h, w = view.mask.shape
        cols = col_center + local_x * px_per_unit
        rows = row_min_ref + (1 - (gy + 1) / 2) * row_span_ref

        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        rows_c = np.clip(rows, 0, h - 1).astype(np.int32)
        cols_c = np.clip(cols, 0, w - 1).astype(np.int32)
        sampled = view.mask[rows_c, cols_c]
        inside &= valid & sampled

    if not inside.any():
        raise RuntimeError(
            "Visual hull carving produced an empty volume — check that the "
            "silhouettes actually overlap (same subject, consistent framing)."
        )

    field = ndimage.gaussian_filter(inside.astype(np.float64), sigma=smooth_sigma)
    if field.min() >= 0.5 or field.max() <= 0.5:
        raise RuntimeError("Could not extract a closed surface from the carved volume.")

    grid_extent = 2.0 + 2.0 * margin
    spacing = (grid_extent / (n - 1),) * 3
    verts, faces, _normals, _values = marching_cubes(field, level=0.5, spacing=spacing)
    verts = verts - (1.0 + margin)  # back to world coordinates

    height = verts[:, 1].max() - verts[:, 1].min()
    scale = target_height_mm / height if height > 1e-6 else 1.0
    verts = verts * scale

    return BodyMesh(vertices=verts, faces=faces)
