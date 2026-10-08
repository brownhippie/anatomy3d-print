"""From-scratch photo -> body mesh, with no licensed model weights involved.

Builds a stylized humanoid mesh by:
  1. placing a skeleton from the detected 2D keypoints,
  2. wrapping each bone in a tapered capsule whose radius is set from
     generic, widely-published figure-proportion ratios (the kind used in
     figure drawing / character modeling references, not a licensed
     dataset) scaled by that bone's own measured length,
  3. blending overlapping capsules with a smooth-minimum of their signed
     distance fields (a standard CG technique) so joints merge smoothly
     instead of leaving hard seams,
  4. extracting the zero level set with marching cubes into a closed,
     watertight mesh.

This trades realism for being entirely free of third-party model weights
or training data licensing — everything here is either measured directly
from the input photo or a well-known, unencumbered formula.
"""
from typing import Optional

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

from .face_features import FaceKeypoints
from .landmarks import DetectedKeypoints
from .mesh_types import BodyMesh
from .safety import mad_margin_above_minimum

# Of 11 possible (head, chest, hips, 2 upper arms, 2 forearms, 2 thighs, 2
# shins): the hip/knee/ankle gates below guarantee chest+hips+2 thighs+2
# shins (6) always form — the torso is now two guaranteed segments (chest,
# hips) instead of one, since neck/waist/pelvis are always computed once
# shoulders+hips are present — so 6 is the true floor, not a guess, and no
# longer reachable as a failure path below 6.
MIN_CAPSULES = 6

# Head radius and limb radii as a fraction of that limb's own measured
# length — generic figure-proportion ratios, not a licensed anthropometric
# dataset. Deliberately approximate; this produces a stylized mannequin,
# not an anatomically precise reconstruction.
HEAD_RADIUS_FRAC_OF_SHOULDER_WIDTH = 0.23
# Torso/hip radii must reach out to the shoulder/hip landmarks themselves
# (half the point-to-point width) or the limb capsules attached there end
# up floating outside the torso's own surface with a visible gap.
TORSO_RADIUS_FRAC_OF_SHOULDER_WIDTH = 0.46
HIP_RADIUS_FRAC_OF_HIP_WIDTH = 0.46
# Waist sits narrower than both chest and hips — splitting the torso here
# (instead of one neck-to-pelvis capsule) is what makes "section by
# section" proportion real rather than a single linear taper.
WAIST_RADIUS_FRAC_OF_SHOULDER_WIDTH = 0.34
WAIST_HEIGHT_FRAC = 0.55  # fraction of the way from neck to pelvis
UPPER_ARM_RADIUS_FRAC = 0.14
FOREARM_RADIUS_FRAC = 0.11
THIGH_RADIUS_FRAC = 0.17
SHIN_RADIUS_FRAC = 0.12

# Section-by-section depth:width ratios (front-to-back vs side-to-side) —
# people are not round in cross-section. These are generic, widely-cited
# anthropometric proportions (the kind figure-drawing and character-
# modeling references use), not a licensed dataset, and deliberately
# approximate: a stylized mannequin's cross-section, not a medical cast.
# 1.0 = circular (this project's previous, cruder default everywhere).
DEPTH_RATIO_HEAD = 1.15  # heads measure slightly longer front-to-back than wide
DEPTH_RATIO_CHEST = 0.62  # chest: markedly wider side-to-side than deep
DEPTH_RATIO_WAIST = 0.72  # waist: rounder than the chest, still not circular
DEPTH_RATIO_HIPS = 0.78
DEPTH_RATIO_LIMB = 0.92  # limbs are closer to round, but still slightly flattened

# Face feature sizing, as a fraction of that face's own measured
# interocular distance (inner eye corner to inner eye corner) — the
# standard figure-drawing unit for face proportions ("one eye-width"
# spacing), not a licensed dataset, same approach as the body radii
# above. Unlike the body, these drive real per-photo measured points
# (face_features.py), not just generic proportions scaled by a bone
# length — the nose/chin/eye *positions* are the photo's own, only their
# *thickness* around those points is a generic stylized guess.
NOSE_BASE_RADIUS_FRAC = 0.26
NOSE_TIP_RADIUS_FRAC = 0.16
CHIN_RADIUS_FRAC = 0.20
EYE_SOCKET_RADIUS_FRAC = 0.32
EYE_SOCKET_RECESS_FRAC = 0.35  # how far back from the eye-corner plane the socket center sits

# How finely the marching-cubes grid must resolve the smallest face
# feature to show up at all. Calibrated directly against measured output
# on a synthetic full-body test (see build_body_mesh), not guessed: a
# first guess of needing several voxels across a feature's radius was
# checked against real numbers and was wrong — a voxel only modestly
# *larger* than the radius (res=260, voxel/radius~1.4) still rounded the
# nose away completely (measured diff exactly 0.0mm with vs without it),
# while a voxel roughly *matching* the radius (res=320, voxel/radius~1.15)
# measured a 0.60mm bump, within ~10% of the exact analytic SDF
# prediction (0.67mm). FACE_VOXEL_FRAC is set from that measured
# crossover with a small safety margin, not the original guess. Grid
# resolution is raised (only as needed) to hit this, capped by
# MAX_GRID_RESOLUTION_WITH_FACE to bound runtime cost — tested directly
# up to that cap at ~12s for a full-body figure versus ~2s with no face
# detail, an acceptable tradeoff for an opt-in detail feature.
# FACE_SKIP_FACTOR: if even the capped resolution would still be more
# than 2x too coarse, there's no point paying the extra cost for detail
# that still wouldn't survive — skip it and say why instead.
FACE_VOXEL_FRAC = 1.1
MAX_GRID_RESOLUTION_WITH_FACE = 340
FACE_SKIP_FACTOR = 2.0


def _to_local_xy(keypoints: DetectedKeypoints) -> dict:
    w, h = keypoints.image_width, keypoints.image_height
    out = {}
    for name, (px, py, _vis) in keypoints.joints.items():
        out[name] = np.array([px - w / 2.0, h / 2.0 - py, 0.0])
    return out


def _capsule_sdf(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ra: float,
    rb: float,
    depth_ratio_a: float = 1.0,
    depth_ratio_b: float = 1.0,
) -> np.ndarray:
    """`depth_ratio` < 1 flattens the cross-section front-to-back (the
    global Z axis) relative to side-to-side — every capsule's segment
    lies in the z=0 plane (see _to_local_xy), so Z is always exactly the
    "depth" axis here, never a segment-relative direction, which is what
    makes scaling just the z-offset correct for an arbitrarily-oriented
    segment: scaling the already-orthogonal offset's z-component shrinks
    or grows the surface in the depth direction only, independent of the
    segment's own direction within the XY plane. This is the zero level
    set of a true ellipse in cross-section; it's not an exact Euclidean
    SDF away from that surface (the gradient magnitude isn't 1 off-axis),
    which doesn't matter for marching cubes or smooth-min blending — both
    only need a correctly-signed, reasonably smooth field near zero.

    `depth_ratio_a`/`depth_ratio_b` interpolate along the segment the same
    way `ra`/`rb` do for radius, so a capsule can flatten gradually from
    one end's ratio to the other's instead of jumping at the joint."""
    pa = points - a
    ba = b - a
    ba_dot = np.dot(ba, ba)
    h = np.clip((pa @ ba) / ba_dot, 0.0, 1.0) if ba_dot > 1e-9 else np.zeros(len(points))
    closest = a + h[:, None] * ba
    offset = points - closest
    if depth_ratio_a != 1.0 or depth_ratio_b != 1.0:
        depth_ratio = depth_ratio_a + h * (depth_ratio_b - depth_ratio_a)
        offset = offset.copy()
        offset[:, 2] = offset[:, 2] / depth_ratio
    dist = np.linalg.norm(offset, axis=1)
    radius = ra + h * (rb - ra)
    return dist - radius


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    hh = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - hh * hh * k * 0.25


def _segment(joints, name_a, name_b, radius_a, radius_b, depth_ratio_a=1.0, depth_ratio_b=None):
    if name_a not in joints or name_b not in joints:
        return None
    if depth_ratio_b is None:
        depth_ratio_b = depth_ratio_a
    return (joints[name_a], joints[name_b], radius_a, radius_b, depth_ratio_a, depth_ratio_b)


def _build_capsules(joints: dict) -> list:
    if "left_shoulder" not in joints or "right_shoulder" not in joints:
        raise RuntimeError("Shoulders not detected — need a front-facing upper-body view at least.")
    # Found by calibration, not guessed: without both hips there's no
    # "pelvis" joint, so no torso capsule forms at all. Whatever's left
    # (e.g. head + arm stumps) still gets scaled to fill the full
    # target_height_mm, which inflates it wildly — a measured case showed
    # ~4x the volume of a complete figure. Raw capsule count doesn't catch
    # this (5+ capsules can still be present), so it needs its own gate.
    if "left_hip" not in joints or "right_hip" not in joints:
        raise RuntimeError(
            "Hips not detected — need a photo showing the torso down to at "
            "least the hips, or the figure's proportions come out badly wrong."
        )
    # Same mechanism, found the same way: target_height_mm scaling trusts
    # the mesh's own lowest point to mean "feet." The thigh and shin
    # segments both need the knee as their shared endpoint (hip->knee,
    # knee->ankle), so ankle presence alone doesn't guarantee a leg
    # capsule actually reaches it — tested directly: hips+ankles present
    # but knees missing still inflated the figure to ~222% of a complete
    # figure's volume, same failure as the hips/ankles cases. Needs the
    # full chain, both sides, not just the endpoints.
    if any(j not in joints for j in ("left_knee", "right_knee", "left_ankle", "right_ankle")):
        raise RuntimeError(
            "Legs not fully detected — need a photo showing both legs down "
            "to at least the ankles (hips, knees, and ankles all visible), "
            "or the figure's proportions come out badly wrong."
        )

    shoulder_width = np.linalg.norm(joints["left_shoulder"] - joints["right_shoulder"])
    hip_width = np.linalg.norm(joints["left_hip"] - joints["right_hip"])

    neck = (joints["left_shoulder"] + joints["right_shoulder"]) / 2.0
    joints = dict(joints)
    joints["neck"] = neck
    if "left_hip" in joints and "right_hip" in joints:
        joints["pelvis"] = (joints["left_hip"] + joints["right_hip"]) / 2.0
    if "nose" in joints:
        joints["head_top"] = joints["nose"] + (joints["nose"] - neck) * 1.1
    joints["waist"] = neck + (joints["pelvis"] - neck) * WAIST_HEIGHT_FRAC

    # Pull limb attachment points inward from the raw landmark toward the
    # torso centerline. The torso capsule's own radius can't be stretched
    # all the way out to the shoulder/hip landmarks without also widening
    # the whole torso, so without this, limb capsules start just outside
    # the torso's surface — a gap that's geometrically real, not something
    # blending can reliably close (it only smooths where surfaces already
    # overlap).
    def pulled(name, toward):
        if name not in joints or toward not in joints:
            return None
        return joints[name] + (joints[toward] - joints[name]) * 0.35

    for side in ("left", "right"):
        p = pulled(f"{side}_shoulder", "neck")
        if p is not None:
            joints[f"{side}_shoulder_attach"] = p
        p = pulled(f"{side}_hip", "pelvis")
        if p is not None:
            joints[f"{side}_hip_attach"] = p

    head_r = HEAD_RADIUS_FRAC_OF_SHOULDER_WIDTH * shoulder_width
    torso_r_top = TORSO_RADIUS_FRAC_OF_SHOULDER_WIDTH * shoulder_width
    waist_r = WAIST_RADIUS_FRAC_OF_SHOULDER_WIDTH * shoulder_width
    torso_r_bot = HIP_RADIUS_FRAC_OF_HIP_WIDTH * hip_width

    capsules = []
    # Chest and hips get their own depth ratio either side of the narrower
    # waist, instead of one linear-taper torso capsule — this is the
    # "section by section" proportion split the figure is built from. Each
    # torso segment's depth ratio is itself interpolated end-to-end (chest
    # ratio -> waist ratio -> hip ratio) so the flattening changes smoothly
    # along the torso instead of jumping where the two segments meet.
    segs = [
        ("head_top", "neck", head_r, head_r * 0.6, DEPTH_RATIO_HEAD, DEPTH_RATIO_HEAD),
        ("neck", "waist", torso_r_top, waist_r, DEPTH_RATIO_CHEST, DEPTH_RATIO_WAIST),
        ("waist", "pelvis", waist_r, torso_r_bot, DEPTH_RATIO_WAIST, DEPTH_RATIO_HIPS),
        ("left_shoulder_attach", "left_elbow", shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.5, shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.4, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_shoulder_attach", "right_elbow", shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.5, shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.4, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("left_elbow", "left_wrist", shoulder_width * FOREARM_RADIUS_FRAC * 0.4, shoulder_width * FOREARM_RADIUS_FRAC * 0.3, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_elbow", "right_wrist", shoulder_width * FOREARM_RADIUS_FRAC * 0.4, shoulder_width * FOREARM_RADIUS_FRAC * 0.3, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("left_hip_attach", "left_knee", hip_width * THIGH_RADIUS_FRAC * 0.6, hip_width * THIGH_RADIUS_FRAC * 0.4, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_hip_attach", "right_knee", hip_width * THIGH_RADIUS_FRAC * 0.6, hip_width * THIGH_RADIUS_FRAC * 0.4, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("left_knee", "left_ankle", hip_width * SHIN_RADIUS_FRAC * 0.4, hip_width * SHIN_RADIUS_FRAC * 0.3, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_knee", "right_ankle", hip_width * SHIN_RADIUS_FRAC * 0.4, hip_width * SHIN_RADIUS_FRAC * 0.3, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
    ]
    for a, b, ra, rb, dra, drb in segs:
        seg = _segment(joints, a, b, ra, rb, dra, drb)
        if seg:
            capsules.append(seg)

    if len(capsules) < MIN_CAPSULES:
        raise RuntimeError(
            f"Not enough body parts detected to build a figure ({len(capsules)}, "
            f"need at least {MIN_CAPSULES}) — use a clear, front-facing photo "
            "showing most of the body."
        )

    result = mad_margin_above_minimum(len(capsules), MIN_CAPSULES, min_margin=0.3)
    if not result.ok:
        print(
            f"Warning: only {len(capsules)} body parts detected, {result.margin:.0%} "
            f"above the {MIN_CAPSULES}-part minimum — the figure will be missing "
            "limbs or look rough. A clearer full-body photo will give a better result."
        )

    return capsules


def _face_to_local_xyz(face_kp: FaceKeypoints) -> dict:
    """Same local-coordinate convention as _to_local_xy (origin at image
    center, y flipped so up is positive), extended with a real measured z
    for each face point instead of the flat z=0 every pose joint gets.
    MediaPipe's face-mesh z is normalized on roughly the same scale as x
    (i.e. already in "pixel-equivalent" units once multiplied by image
    width), smaller/more negative meaning closer to the camera — the
    opposite sign of this project's own z convention (z > 0 is the
    camera-facing front, see build_body_mesh/_sculpt_front_surface_from_
    depth), so it's negated here. Verified against a real detected face
    (see face_features.py's module docstring): the nose tip comes out
    with the largest local z (most forward) of any point checked, the
    jaw corners the smallest (furthest back), matching real face shape."""
    w, h = face_kp.image_width, face_kp.image_height
    out = {}
    for idx, (px, py, z_raw) in face_kp.points.items():
        out[idx] = np.array([px - w / 2.0, h / 2.0 - py, -z_raw * w])
    return out


def _build_face_additions(face_points: dict) -> tuple:
    """Builds small additive capsules (nose, chin) and subtractive eye
    sockets from real detected face-landmark positions, to overlay on the
    generic head capsule. Returns ([] , []) if the needed landmarks
    aren't present — face detail is a bonus, not a requirement, so a
    missing/unusable face should silently add nothing rather than fail
    the body build."""
    from .face_features import CHIN, LEFT_EYE, NOSE_BRIDGE, NOSE_TIP, RIGHT_EYE

    needed = {NOSE_TIP, NOSE_BRIDGE, CHIN, *RIGHT_EYE, *LEFT_EYE}
    if not needed.issubset(face_points.keys()):
        return [], []

    right_eye_pts = np.array([face_points[i] for i in RIGHT_EYE])
    left_eye_pts = np.array([face_points[i] for i in LEFT_EYE])
    right_eye_center = right_eye_pts.mean(axis=0)
    left_eye_center = left_eye_pts.mean(axis=0)

    # Inner-corner-to-inner-corner distance — the classic figure-drawing
    # "one eye-width" unit everything else here scales from.
    interocular = np.linalg.norm(right_eye_pts[1] - left_eye_pts[0])
    if interocular < 1e-6:
        return [], []

    nose_tip = face_points[NOSE_TIP]
    nose_bridge = face_points[NOSE_BRIDGE]
    chin = face_points[CHIN]

    extra_capsules = [
        (
            nose_bridge,
            nose_tip,
            NOSE_BASE_RADIUS_FRAC * interocular,
            NOSE_TIP_RADIUS_FRAC * interocular,
            1.0,
            1.0,
        ),
        # A degenerate (same-point) capsule is just a sphere — enough for
        # a small chin protrusion without needing a second chin landmark.
        (chin, chin, CHIN_RADIUS_FRAC * interocular, CHIN_RADIUS_FRAC * interocular, 1.0, 1.0),
    ]

    # Eye sockets are carved in (subtracted), not added, so they're kept
    # separate from extra_capsules: subtraction needs to happen after the
    # main smooth-min union, not chained into it (see build_body_mesh).
    eye_sockets = []
    for center in (right_eye_center, left_eye_center):
        recessed_center = center.copy()
        recessed_center[2] -= EYE_SOCKET_RECESS_FRAC * interocular
        eye_sockets.append((recessed_center, EYE_SOCKET_RADIUS_FRAC * interocular))

    return extra_capsules, eye_sockets


def _sculpt_front_surface_from_depth(
    verts: np.ndarray, image_width: int, image_height: int, rgb: np.ndarray, strength: float
) -> np.ndarray:
    """Optional refinement: push the front half-surface (z > 0, in this
    module's convention — see build_body_mesh) in or out based on a real
    per-pixel depth estimate, instead of leaving it at the capsule's flat
    symmetric guess. The back half (z < 0) is left untouched — we have no
    photo information about it either way.

    Requires `anatomy3d.depth_source` (optional extra, see
    requirements-depth.txt); raises a clear error if it's not installed
    rather than silently no-op'ing, since a caller that explicitly asked
    for depth sculpting should know it didn't happen.
    """
    from .depth_source import estimate_relative_depth

    depth = estimate_relative_depth(rgb)
    depth_h, depth_w = depth.shape

    front = verts[:, 2] > 0
    if not front.any():
        return verts

    px = verts[front, 0] + image_width / 2.0
    py = image_height / 2.0 - verts[front, 1]
    # map_coordinates wants (row, col) = (py, px), scaled to the depth
    # map's own resolution if it differs from the source photo's.
    row = np.clip(py * (depth_h / image_height), 0, depth_h - 1)
    col = np.clip(px * (depth_w / image_width), 0, depth_w - 1)
    sampled = map_coordinates(depth, [row, col], order=1, mode="nearest")

    mean, std = sampled.mean(), sampled.std()
    if std < 1e-9:
        return verts  # flat/uninformative depth map — leave the guess alone
    d_norm = (sampled - mean) / std

    # Displacement scales with each vertex's own distance from the
    # centerline (a proxy for local body-part radius): a thin forearm
    # shifts subtly, a thick torso can shift more, and the effect tapers
    # to ~0 at the silhouette edge (z near 0) instead of cutting off
    # sharply. ASSUMES higher depth value = closer to camera — not yet
    # checked against a real photo in this project (torch wasn't
    # available to test with at the time this was written). If sculpted
    # figures come out concave where they should be convex (e.g. a nose
    # pushed in instead of out), this sign is inverted; flip it here.
    verts = verts.copy()
    disp = d_norm * strength * verts[front, 2]
    new_z = verts[front, 2] + disp
    verts[front, 2] = np.maximum(new_z, verts[front, 2] * 0.15)  # never cross the centerline
    return verts


def build_body_mesh(
    keypoints: DetectedKeypoints,
    target_height_mm: float = 150.0,
    grid_resolution: int = 190,
    blend_k_frac: float = 0.35,
    depth_rgb: Optional[np.ndarray] = None,
    depth_strength: float = 0.5,
    face_keypoints: Optional[FaceKeypoints] = None,
) -> BodyMesh:
    """`depth_rgb`: the same preprocessed photo passed to pose detection.
    When given, sculpts the front surface using real per-pixel depth
    (anatomy3d.depth_source, optional extra) instead of the flat
    symmetric-thickness guess. Leave as None for the default, dependency-
    free behavior.

    `face_keypoints`: real detected face landmarks (face_features.py).
    When given and usable, overlays a nose, chin, and eye sockets built
    from the photo's own measured positions on top of the generic head
    capsule. Optional — a missing or unusable face silently adds nothing."""
    joints = _to_local_xy(keypoints)
    capsules = _build_capsules(joints)

    all_radii = [r for cap in capsules for r in cap[2:4]]
    margin = max(all_radii) * 2.5
    pts_for_bounds = np.array([p for cap in capsules for p in cap[:2]])

    face_capsules, eye_sockets = [], []
    if face_keypoints is not None:
        face_local = _face_to_local_xyz(face_keypoints)
        face_capsules, eye_sockets = _build_face_additions(face_local)
    if face_capsules:
        # Bounds must cover the face additions too, even though they're
        # already within the head capsule's own margin in practice — kept
        # as its own min/max rather than folded into `margin`/`all_radii`
        # above, so face parts (small) don't shrink the main blend width
        # (see blend_k below, computed from body-only radii).
        face_pts = np.array([p for cap in face_capsules for p in cap[:2]] + [c for c, _r in eye_sockets])
        pts_for_bounds = np.vstack([pts_for_bounds, face_pts])

    mins = pts_for_bounds.min(axis=0) - margin
    maxs = pts_for_bounds.max(axis=0) + margin
    # Give the figure real thickness front-to-back even though the input is
    # a single frontal photo — no depth data exists, so this is a flat
    # stylization choice, not a measurement.
    depth_half = max(all_radii) * 3.0
    mins[2], maxs[2] = -depth_half, depth_half

    dims = maxs - mins

    if face_capsules:
        # Measured directly, not assumed: a nose/chin sized right for a
        # real face is tiny next to a whole body's bounding box, and at
        # this function's normal grid_resolution the marching-cubes voxel
        # size is bigger than the feature itself — the added geometry is
        # mathematically there (see _build_face_additions) but gets
        # rounded away before it ever reaches the output mesh. Confirmed
        # directly: at grid_resolution=190 on a synthetic full-body test,
        # the resulting mesh was byte-for-byte identical with and without
        # the face capsules; raising resolution until voxel size dropped
        # below the nose's own radius made the same test start measuring
        # the expected bump (within ~10% of the exact analytic SDF
        # prediction once 2-3 voxels span the smallest feature radius).
        # So: sharpen the grid only as far as needed to resolve the
        # smallest face feature, capped to bound runtime cost (uniform
        # global refinement this coarse already costs ~5x at the cap
        # tested here) — and if even the cap can't get there (a face
        # that's a very small fraction of the frame), skip adding the
        # face geometry rather than silently shipping a change too small
        # for any viewer or slicer to ever see.
        face_radii = [r for cap in face_capsules for r in cap[2:4]] + [r for _c, r in eye_sockets]
        required_resolution = dims.max() / (min(face_radii) * FACE_VOXEL_FRAC)
        if required_resolution > MAX_GRID_RESOLUTION_WITH_FACE * FACE_SKIP_FACTOR:
            print(
                "Note: a face was detected, but it's too small relative to the "
                "whole-body photo to render fine facial detail at a practical "
                "resolution — the generic head shape will be used instead. A "
                "closer, more face-filling photo (or a face/half-body shot) "
                "would let this feature actually show up."
            )
            face_capsules, eye_sockets = [], []
        else:
            grid_resolution = int(np.clip(required_resolution, grid_resolution, MAX_GRID_RESOLUTION_WITH_FACE))

    res = np.maximum((dims / dims.max() * grid_resolution).astype(int), 8)
    xs = np.linspace(mins[0], maxs[0], res[0])
    ys = np.linspace(mins[1], maxs[1], res[1])
    zs = np.linspace(mins[2], maxs[2], res[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Base the blend width on the *thinnest* capsules (forearms/shins), not
    # the thickest (head/torso) — otherwise a wide blend relative to a thin
    # limb's own radius distorts and can fragment that limb.
    blend_k = min(all_radii) * blend_k_frac * 5.0
    field = None
    for a, b, ra, rb, dra, drb in capsules:
        sdf = _capsule_sdf(points, a, b, ra, rb, dra, drb)
        field = sdf if field is None else _smooth_min(field, sdf, blend_k)

    if face_capsules:
        # Face parts are much smaller than any body capsule — blending
        # them with the main body's own (comparatively huge) blend_k would
        # either barely affect anything or wash the small features out
        # entirely, so they get their own, face-scaled blend width.
        face_radii = [r for cap in face_capsules for r in cap[2:4]]
        face_blend_k = min(face_radii) * blend_k_frac * 5.0
        for a, b, ra, rb, dra, drb in face_capsules:
            sdf = _capsule_sdf(points, a, b, ra, rb, dra, drb)
            field = _smooth_min(field, sdf, face_blend_k)

    for center, radius in eye_sockets:
        # Subtracted (carved in), not unioned: max(field, -socket_sdf)
        # keeps the result everywhere the socket sphere ISN'T, which is
        # exactly a boolean subtraction — this is why eye sockets are kept
        # out of the smooth-min chain above rather than passed through it.
        socket_sdf = np.linalg.norm(points - center, axis=1) - radius
        field = np.maximum(field, -socket_sdf)

    field = field.reshape(res)
    spacing = (
        (maxs[0] - mins[0]) / (res[0] - 1),
        (maxs[1] - mins[1]) / (res[1] - 1),
        (maxs[2] - mins[2]) / (res[2] - 1),
    )

    if field.min() >= 0 or field.max() <= 0:
        raise RuntimeError("Could not form a closed body surface from this photo's keypoints.")

    verts, faces, _normals, _values = marching_cubes(field, level=0.0, spacing=spacing)
    verts = verts + mins

    # Chaining pairwise smooth-min across 3+ overlapping capsules near a
    # joint (see print_prep.py's docstring) can leave debris disconnected
    # from the main body — and critically, that debris can sit beyond the
    # main body's own top/bottom, which would otherwise make the height
    # measurement below count space that print_prep later discards,
    # silently undershooting the requested target_height_mm. Drop debris
    # here, before measuring height, not after scaling.
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: c.vertices.shape[0])
    verts, faces = mesh.vertices, mesh.faces

    if depth_rgb is not None:
        # Before scaling: the pixel<->local-coordinate mapping this uses
        # is only valid in this unscaled space (see _to_local_xy).
        verts = _sculpt_front_surface_from_depth(
            verts, keypoints.image_width, keypoints.image_height, depth_rgb, depth_strength
        )

    height = verts[:, 1].max() - verts[:, 1].min()
    scale = target_height_mm / height if height > 1e-6 else 1.0
    verts = verts * scale

    return BodyMesh(vertices=verts, faces=faces)
