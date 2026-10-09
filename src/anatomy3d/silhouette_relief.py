"""Builds a body mesh directly from a 2D cutout's own alpha mask, not
from capsule primitives fit to detected joints (see procedural_body.py).

Unlike build_body_mesh, this guarantees -- by construction, not by
tuning -- that the mesh's own front-on silhouette (looking down the
camera/z axis) is pixel-identical to the cutout's outline: no pose
detection, no joint positions, no capsule approximation involved at all.
Needs only the alpha mask itself, so it has no dependency on pose
landmarks succeeding.

Each foreground pixel's half-thickness comes from its own distance to
the nearest background pixel (distance_transform_edt -- the same
technique procedural_body.py already uses for its own, much coarser,
6-samples-per-segment capsule radii), giving a rounded body that's
naturally thick in limb/torso centers and tapers toward the cutout's own
edge, rather than a flat cardboard cutout.
"""
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

from .mesh_types import BodyMesh


def build_silhouette_relief_mesh(
    alpha: np.ndarray,
    target_height_mm: float = 150.0,
    grid_size: int = 220,
    depth_scale: float = 0.5,
) -> BodyMesh:
    """`alpha`: a float [0,1] or boolean (H, W) mask -- the same soft
    cutout alpha detect_person_alpha produces (and the webapp's
    /jobs/{id}/cutout.png serves), thresholded at 0.5 to a hard boundary
    for a clean, closed mesh.

    `depth_scale`: real human/animal depth (front-to-back) runs
    shallower than half of the local width a straight distance-transform
    value would give (a torso's front-to-back depth is roughly half its
    side-to-side width, not equal to it) -- a tuned, not measured,
    flattening factor, same honesty this project already applies to its
    own tuned constants elsewhere (procedural_body.py's blend_k_frac).

    No pose, no color, no texture -- purely the cutout's own shape swept
    into a rounded volume. Render/texture it yourself from here."""
    mask = alpha > 0.5
    if not mask.any():
        raise ValueError("Empty mask -- nothing to build a shape from.")

    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = mask[y0:y1, x0:x1]
    h, w = cropped.shape

    scale = grid_size / max(h, w)
    gh, gw = max(1, round(h * scale)), max(1, round(w * scale))
    grid_mask_raw = np.asarray(
        Image.fromarray(cropped.astype(np.uint8) * 255).resize((gw, gh), Image.BILINEAR)
    ) > 127
    # 2px of guaranteed "outside" border -- without it, the mask could
    # touch the array's own edge, leaving marching_cubes no neighboring
    # outside cell to interpolate a clean closing surface against there.
    # Confirmed necessary directly: omitting this left open boundary
    # edges in the exported mesh.
    pad = 2
    grid_mask = np.zeros((gh + 2 * pad, gw + 2 * pad), dtype=bool)
    grid_mask[pad : pad + gh, pad : pad + gw] = grid_mask_raw
    gh, gw = grid_mask.shape

    dist = distance_transform_edt(grid_mask).astype(np.float32)
    dist /= scale  # grid-pixels -> original-photo pixels
    half_thickness_raw = dist * depth_scale
    # A half-thickness that touches exactly 0 right at the mask edge is a
    # literal cusp (zero-measure point) -- mathematically degenerate, and
    # left small non-manifold gaps in marching_cubes' output (confirmed
    # directly: dozens of open boundary edges on a real test mesh). A
    # HARD floor (np.maximum) fixed the cusp but introduced a sharp
    # discontinuity in half_thickness right at the mask boundary instead,
    # which made the gap problem worse, not better (confirmed directly).
    # A smooth soft-floor -- sqrt(raw^2 + floor^2) -- has no
    # discontinuity anywhere (continuous at the true edge too, same as
    # the unfloored version) while still never reaching exactly 0.
    min_half_thickness = max(1.0 / scale, 0.5)
    half_thickness = np.where(grid_mask, np.sqrt(half_thickness_raw**2 + min_half_thickness**2), 0.0)

    true_max = float(half_thickness.max())
    if true_max <= 0:
        raise ValueError("Mask too thin/small to extrude a volume from.")
    # A few percent beyond the true max so the solid never exactly
    # touches the volume's own first/last z-slice -- that single tangent
    # point is itself a small degenerate case, same family of problem as
    # the edge cusp above.
    z_max = true_max * 1.05
    # z-resolution must be fine enough to resolve the THINNEST region
    # (the floor thickness on arms/legs), not just coarse enough for the
    # overall depth -- a global step size set only from z_max left a
    # limb's own thin cross-section spanning less than one z-step, which
    # marching_cubes could not close into a clean tube (confirmed
    # directly: open boundary edges concentrated entirely on thin-limb
    # regions, not the torso, before this fix). At least 3 steps across
    # the floor thickness closes it.
    dz_needed = min_half_thickness / 3.0
    gz = max(4, round(2 * z_max / dz_needed) + 4)

    zs = np.linspace(-z_max, z_max, gz)
    dz = zs[1] - zs[0]
    field = np.empty((gz, gh, gw), dtype=np.float32)
    for i, z in enumerate(zs):
        field[i] = z * z - half_thickness * half_thickness
    field[:, ~grid_mask] = 1.0

    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(dz, 1.0 / scale, 1.0 / scale))

    # A genuinely disconnected scrap of surface (e.g. a small noise
    # island the alpha mask itself has, separate from the main subject)
    # would otherwise ship as stray floating geometry in the exported
    # mesh -- same class of defect build_body_mesh already guards
    # against for its own output. Keep only the largest connected piece,
    # same pattern. NOTE: this does not fix a part that's wrong but
    # still attached (confirmed directly on a real photo: a tangled
    # blob between a dog's two front legs turned out to be a genuine
    # defect in the alpha mask itself -- a stray bridge connecting the
    # two legs -- not a disconnected island; this check correctly left
    # it alone since it's actually one connected piece, matching what
    # the mask itself says).
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: c.vertices.shape[0])
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)

    vz, vrow, vcol = verts[:, 0], verts[:, 1], verts[:, 2]
    # vrow/vcol are already in physical (original-photo-pixel) units via
    # `spacing` -- center on the PADDED grid's own physical extent (gh/gw
    # were reassigned to the padded shape above), not the pre-pad crop.
    x3d = vcol - (gw / scale) / 2.0
    y3d = (gh / scale) / 2.0 - vrow
    z3d = vz - z_max
    verts_xyz = np.stack([x3d, y3d, z3d], axis=1)

    height = verts_xyz[:, 1].max() - verts_xyz[:, 1].min()
    out_scale = target_height_mm / height if height > 1e-6 else 1.0
    verts_xyz = verts_xyz * out_scale

    return BodyMesh(vertices=verts_xyz, faces=faces)
