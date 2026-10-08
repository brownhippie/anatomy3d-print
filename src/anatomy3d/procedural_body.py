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
import numpy as np
from skimage.measure import marching_cubes

from .landmarks import DetectedKeypoints
from .mesh_types import BodyMesh

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
UPPER_ARM_RADIUS_FRAC = 0.14
FOREARM_RADIUS_FRAC = 0.11
THIGH_RADIUS_FRAC = 0.17
SHIN_RADIUS_FRAC = 0.12


def _to_local_xy(keypoints: DetectedKeypoints) -> dict:
    w, h = keypoints.image_width, keypoints.image_height
    out = {}
    for name, (px, py, _vis) in keypoints.joints.items():
        out[name] = np.array([px - w / 2.0, h / 2.0 - py, 0.0])
    return out


def _capsule_sdf(points: np.ndarray, a: np.ndarray, b: np.ndarray, ra: float, rb: float) -> np.ndarray:
    pa = points - a
    ba = b - a
    ba_dot = np.dot(ba, ba)
    h = np.clip((pa @ ba) / ba_dot, 0.0, 1.0) if ba_dot > 1e-9 else np.zeros(len(points))
    closest = a + h[:, None] * ba
    dist = np.linalg.norm(points - closest, axis=1)
    radius = ra + h * (rb - ra)
    return dist - radius


def _smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    hh = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - hh * hh * k * 0.25


def _segment(joints, name_a, name_b, radius_a, radius_b):
    if name_a not in joints or name_b not in joints:
        return None
    return (joints[name_a], joints[name_b], radius_a, radius_b)


def _build_capsules(joints: dict) -> list:
    if "left_shoulder" not in joints or "right_shoulder" not in joints:
        raise RuntimeError("Shoulders not detected — need a front-facing upper-body view at least.")

    shoulder_width = np.linalg.norm(joints["left_shoulder"] - joints["right_shoulder"])
    hip_width = (
        np.linalg.norm(joints["left_hip"] - joints["right_hip"])
        if "left_hip" in joints and "right_hip" in joints
        else shoulder_width * 0.9
    )

    neck = (joints["left_shoulder"] + joints["right_shoulder"]) / 2.0
    joints = dict(joints)
    joints["neck"] = neck
    if "left_hip" in joints and "right_hip" in joints:
        joints["pelvis"] = (joints["left_hip"] + joints["right_hip"]) / 2.0
    if "nose" in joints:
        joints["head_top"] = joints["nose"] + (joints["nose"] - neck) * 1.1

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
    torso_r_bot = HIP_RADIUS_FRAC_OF_HIP_WIDTH * hip_width

    capsules = []
    segs = [
        ("head_top", "neck", head_r, head_r * 0.6),
        ("neck", "pelvis", torso_r_top, torso_r_bot),
        ("left_shoulder_attach", "left_elbow", shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.5, shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.4),
        ("right_shoulder_attach", "right_elbow", shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.5, shoulder_width * UPPER_ARM_RADIUS_FRAC * 0.4),
        ("left_elbow", "left_wrist", shoulder_width * FOREARM_RADIUS_FRAC * 0.4, shoulder_width * FOREARM_RADIUS_FRAC * 0.3),
        ("right_elbow", "right_wrist", shoulder_width * FOREARM_RADIUS_FRAC * 0.4, shoulder_width * FOREARM_RADIUS_FRAC * 0.3),
        ("left_hip_attach", "left_knee", hip_width * THIGH_RADIUS_FRAC * 0.6, hip_width * THIGH_RADIUS_FRAC * 0.4),
        ("right_hip_attach", "right_knee", hip_width * THIGH_RADIUS_FRAC * 0.6, hip_width * THIGH_RADIUS_FRAC * 0.4),
        ("left_knee", "left_ankle", hip_width * SHIN_RADIUS_FRAC * 0.4, hip_width * SHIN_RADIUS_FRAC * 0.3),
        ("right_knee", "right_ankle", hip_width * SHIN_RADIUS_FRAC * 0.4, hip_width * SHIN_RADIUS_FRAC * 0.3),
    ]
    for a, b, ra, rb in segs:
        seg = _segment(joints, a, b, ra, rb)
        if seg:
            capsules.append(seg)

    if len(capsules) < 3:
        raise RuntimeError(
            "Not enough body parts detected to build a figure — use a clear, "
            "front-facing photo showing most of the body."
        )
    return capsules


def build_body_mesh(
    keypoints: DetectedKeypoints,
    target_height_mm: float = 150.0,
    grid_resolution: int = 190,
    blend_k_frac: float = 0.35,
) -> BodyMesh:
    joints = _to_local_xy(keypoints)
    capsules = _build_capsules(joints)

    all_radii = [r for cap in capsules for r in cap[2:4]]
    margin = max(all_radii) * 2.5
    pts_for_bounds = np.array([p for cap in capsules for p in cap[:2]])
    mins = pts_for_bounds.min(axis=0) - margin
    maxs = pts_for_bounds.max(axis=0) + margin
    # Give the figure real thickness front-to-back even though the input is
    # a single frontal photo — no depth data exists, so this is a flat
    # stylization choice, not a measurement.
    depth_half = max(all_radii) * 3.0
    mins[2], maxs[2] = -depth_half, depth_half

    dims = maxs - mins
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
    for a, b, ra, rb in capsules:
        sdf = _capsule_sdf(points, a, b, ra, rb)
        field = sdf if field is None else _smooth_min(field, sdf, blend_k)

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

    height = verts[:, 1].max() - verts[:, 1].min()
    scale = target_height_mm / height if height > 1e-6 else 1.0
    verts = verts * scale

    return BodyMesh(vertices=verts, faces=faces)
