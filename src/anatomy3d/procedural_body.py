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
# Recalibrated against real output, not left at the original guess: a
# real test photo (MediaPipe's own `pose.jpg`, a dynamic side lunge)
# measured these out at 2-4x thinner than generic circumference-to-height
# anthropometric ratios (adult upper-arm/forearm/thigh/calf circumference
# as a fraction of height, the same kind of generic figure reference used
# throughout this file) predict — e.g. a thigh radius of ~2.5mm on a
# 150mm figure versus the ~7mm these ratios suggest. The segs list below
# used to compound each of these with its own extra ad hoc 0.3-0.6
# end-multiplier on top; that's now folded into the constants themselves
# (base = this fraction, tip = this fraction * the segment's own taper
# below) so there's one place these are set, not two.
UPPER_ARM_RADIUS_FRAC = 0.115
FOREARM_RADIUS_FRAC = 0.095
THIGH_RADIUS_FRAC = 0.30
SHIN_RADIUS_FRAC = 0.21
ARM_TAPER = 0.85  # tip radius = base radius * this
LEG_TAPER = 0.75  # legs narrow more from hip/knee to knee/ankle than arms do

# Hand: previously the arm chain just dead-ended at the wrist with a
# tapered capsule cap — no hand at all, confirmed directly (no "hand" or
# "finger" reference anywhere in this file before this). MediaPipe Pose's
# 33-point topology already detects index/pinky/thumb landmarks (indices
# 17-22) alongside the wrist; they just weren't read.
#
# The palm's LENGTH and ORIENTATION come from the actual detected
# index/pinky/thumb positions for that hand in that photo (a position is
# a position, regardless of how the model arrived at it). Its WIDTH does
# not: tried using the measured index-to-pinky distance directly first,
# and it came out to 1.3-1.5 units against a ~7.8-unit wrist radius on
# the test photo — physically implausible (a palm narrower than the
# wrist). These 6 points are BlazePose's approximate hand-orientation
# stubs, not the dedicated 21-point hand landmark model; confirmed here
# that their mutual spacing isn't a trustworthy width signal even though
# their position is a fine direction/length signal. So width instead
# scales off the wrist radius itself by a fixed anatomical ratio, same
# standard every other segment in this file already uses.
PALM_RADIUS_FRAC_OF_WRIST = 1.3  # palm is measurably wider than the wrist, not narrower
PALM_DEPTH_RATIO = 0.45  # flatter than a limb: a hand is close to planar compared to a forearm
THUMB_DEPTH_RATIO = 0.70
THUMB_RADIUS_FRAC_OF_PALM = 0.45  # thumb is visibly thinner than the palm's own half-width

# Hair: a radial profile of small bumps around the head, sized from the
# photo's own hair mask (hair_features.py) instead of a generic cap —
# stylized (a volume, not individual strands), same "deliberately
# approximate" standard as everything else here.
HAIR_SAMPLES = 24  # angular samples around the head
HAIR_MAX_RADIUS_FRAC = 2.5  # how far out from the head center to search, as a multiple of head_r
HAIR_MIN_RADIUS_FRAC = 0.15  # ignore a direction whose hair barely pokes past the head surface — noise, not a real bump
HAIR_DEPTH_RATIO = 0.95  # close to round; a single front photo gives no real front-to-back hair shape to measure
# How tightly hair bumps blend into the head and each other. Checked
# visually, not left at the body's own blend_k_frac (0.35): rendered side
# by side, 0.35 produced a blend_k comparable to the bumps' own radius,
# which smoothed the hairstyle's actual shape (an asymmetric side-swept
# cut in the test photo) down to a barely-visible shading difference from
# a bare head. 0.08 kept the same bumps distinct enough to actually read
# as a hairstyle while still blending seamlessly (still watertight, no
# separate shells) — tight blending reads as "volume with some shape to
# it", the loose default read as "almost nothing changed".
HAIR_BLEND_FRAC = 0.08

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
# Same purpose, applied to ordinary body limbs instead of face features —
# there's no skip option here (a forearm isn't optional the way fine face
# detail is), only a cost cap. Kept slightly below the face cap since this
# path can trigger on any pose with limbs spread wide, not just when a
# face happens to be detected, so it needs to stay cheap more often.
MAX_GRID_RESOLUTION_BODY = 300

# Single-photo mode used to size every limb/torso segment purely from
# generic proportions (shoulder/hip-width fractions) — it never actually
# looked at the photo's own body outline the way multi-photo visual-hull
# mode does, which is why it always came out as a generic mannequin no
# matter whose photo went in. This measures the photo's own silhouette
# (silhouette.py — already built for multi-photo mode, works on a single
# photo too) at several points along each segment instead of trusting a
# straight-line taper between two joints, so an actual narrower waist, a
# visibly flexed arm, etc. come from the photo rather than a formula.
SILHOUETTE_SAMPLES_PER_SEGMENT = 6
# How far the clamp allows a measured radius to move from the generic
# one it would otherwise have been. Checked against a real adversarial
# case, not picked blind: on a real test photo with a complex (non-studio)
# background, the silhouette extractor correctly separated the person
# from the sky but misclassified a visually similar foreground (sand) as
# part of the body near the legs, which would otherwise balloon those
# segments' measured width to several times the real figure. This clamp
# catches exactly that: a corrupted sample gets pulled back to within a
# sane multiple of the generic estimate instead of propagating a torn
# silhouette into the mesh, while a real, plausible measurement (the
# usual case on a clean background) passes through close to unchanged.
SILHOUETTE_RADIUS_MIN_FRAC = 0.6
SILHOUETTE_RADIUS_MAX_FRAC = 1.35
# How far outward (as a multiple of the generic radius) to search for the
# silhouette's edge before giving up and treating the point as unmeasured
# (falls back to the generic radius for that sample only, not the whole
# segment).
SILHOUETTE_SEARCH_RADIUS_FRAC = 6.0


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


def _segment(joints, name_a, name_b, radius_a, radius_b, depth_ratio_a=1.0, depth_ratio_b=None, group=0):
    if name_a not in joints or name_b not in joints:
        return None
    if depth_ratio_b is None:
        depth_ratio_b = depth_ratio_a
    return (joints[name_a], joints[name_b], radius_a, radius_b, depth_ratio_a, depth_ratio_b, group)


def _measure_half_widths(mask, px, py, perp_dx, perp_dy, max_search_px):
    """Marches outward from (px, py) along +/-(perp_dx, perp_dy) (pixel
    space) until stepping off the silhouette mask, returning the two
    distances (positive side, negative side). Returns None if the center
    point itself isn't inside the mask (a torn/offset silhouette at that
    exact point) OR if either march never finds an edge within
    `max_search_px` — checked against a real failure, not a hypothetical:
    on a real test photo, a leg crossing in front of sand (misclassified
    as foreground by silhouette.py's background-color threshold, a known
    limitation documented there) produced a "measurement" on EVERY single
    sample down the whole thigh, each one landing almost exactly at the
    clamp ceiling that used to apply here — not an occasional outlier a
    generous clamp could absorb, but the search exhausting itself against
    a large connected blob that isn't the limb at all. A search that
    never finds an edge is a sign the measurement is meaningless, not
    that the limb is merely wide, so this now reports that honestly
    (None) and the caller falls back to the pure generic radius instead
    of a clamped-but-still-wrong one."""
    h, w = mask.shape
    cx, cy = int(round(px)), int(round(py))
    if not (0 <= cx < w and 0 <= cy < h) or not mask[cy, cx]:
        return None

    def march(sign):
        dist = 0.0
        while dist < max_search_px:
            x = int(round(px + sign * perp_dx * dist))
            y = int(round(py + sign * perp_dy * dist))
            if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
                return dist
            dist += 1.0
        return None  # never found an edge — not a measurement, a search failure

    d_pos, d_neg = march(1.0), march(-1.0)
    if d_pos is None or d_neg is None:
        return None
    return d_pos, d_neg


def _chain_from_silhouette(
    mask, image_width, image_height, a, b, ra, rb, dra, drb, n_samples, group
):
    """Replaces one straight-taper capsule with a chain of `n_samples`
    shorter ones, each sized from the photo's own silhouette width at
    that point instead of a pure a-to-b interpolation. See
    SILHOUETTE_SAMPLES_PER_SEGMENT's comment for why, and
    SILHOUETTE_RADIUS_MIN_FRAC/MAX_FRAC's for the safety clamp on each
    measurement. Every piece shares `group` so build_body_mesh's SDF
    accumulation hard-unions them together (they already meet exactly at
    shared endpoints with matching radius — no seam to smooth over) instead
    of smooth-min'ing each of the n_samples joints, which would otherwise
    compound a small fillet-bulge at every one of them into a visibly
    fatter limb than any individual measurement called for."""
    ba = b - a
    length = np.linalg.norm(ba)
    if length < 1e-6:
        return None
    direction = ba / length
    # Perpendicular within the local XY plane (joints have z=0 — see
    # _to_local_xy); local x maps directly to pixel x, but local y is
    # pixel-flipped (h/2 - py), so the pixel-space step for a local
    # perpendicular move has its y component negated.
    perp_local = np.array([-direction[1], direction[0], 0.0])
    perp_px = np.array([perp_local[0], -perp_local[1]])

    points, radii = [], []
    for i in range(n_samples + 1):
        t = i / n_samples
        pt = a + ba * t
        generic_r = ra + (rb - ra) * t
        px = pt[0] + image_width / 2.0
        py = image_height / 2.0 - pt[1]
        half = _measure_half_widths(
            mask, px, py, perp_px[0], perp_px[1], generic_r * SILHOUETTE_SEARCH_RADIUS_FRAC
        )
        if half is None:
            measured_r = generic_r
        else:
            d_pos, d_neg = half
            measured_r = (d_pos + d_neg) / 2.0
            lo, hi = generic_r * SILHOUETTE_RADIUS_MIN_FRAC, generic_r * SILHOUETTE_RADIUS_MAX_FRAC
            measured_r = float(np.clip(measured_r, lo, hi))
        points.append(pt)
        radii.append(measured_r)

    chain = []
    for i in range(n_samples):
        t0, t1 = i / n_samples, (i + 1) / n_samples
        dra_i = dra + (drb - dra) * t0
        drb_i = dra + (drb - dra) * t1
        chain.append((points[i], points[i + 1], radii[i], radii[i + 1], dra_i, drb_i, group))
    return chain


def _hair_bumps(hair_mask, image_width, image_height, head_center, head_r, group_start):
    """Samples the photo's hair mask radially around the head center,
    placing a small sphere wherever hair extends meaningfully past the
    head surface in that direction — a short, close-cropped cut produces
    a thin halo close to the head; long or voluminous hair produces a
    bigger one; a bald direction (or a hat the segmenter doesn't read as
    hair) produces none. Each bump gets its own group id (continuing from
    `group_start`) so it blends normally (smooth-min) with the head and
    its neighbors rather than hard-unioning — these are discrete radial
    samples, not a connected chain the way a limb's silhouette samples
    are."""
    max_r = head_r * HAIR_MAX_RADIUS_FRAC
    step = max(1.0, head_r * 0.05)
    bumps = []
    group = group_start
    for i in range(HAIR_SAMPLES):
        angle = 2.0 * np.pi * i / HAIR_SAMPLES
        dx, dy = np.cos(angle), np.sin(angle)
        # Skip directions pointing mostly downward (toward the neck/chest,
        # local y is "up" — see _to_local_xy). Found by running on a real
        # adversarial photo, not a guess: without this, a radial sample
        # pointing straight down lands inside the already-dense neck/torso
        # capsules (verified directly: body field was -21.9 there, far
        # deeper than the bump's own -7.7, so the bump changed nothing —
        # a correct but silently wasted sample). Hair doesn't grow pointing
        # down through the neck, so there's nothing legitimate lost here.
        if dy < -0.3:
            continue

        found_any = False
        hit_cap = False
        outer = 0.0
        r = 0.0
        while r <= max_r:
            local_pt = head_center + np.array([dx, dy, 0.0]) * r
            px = local_pt[0] + image_width / 2.0
            py = image_height / 2.0 - local_pt[1]
            cx, cy = int(round(px)), int(round(py))
            inside = 0 <= cx < image_width and 0 <= cy < image_height and hair_mask[cy, cx]
            if inside:
                found_any = True
                outer = r
            elif found_any:
                break  # left the mask after having been in it — stop at the last True
            r += step
        else:
            hit_cap = found_any  # loop ran to completion without ever exiting the mask

        if not found_any or hit_cap:
            # hit_cap: the mask stayed "True" all the way to max_r without
            # a real edge ever being found — also checked against the
            # same real photo: four separate directions did exactly this,
            # all landing on an identical radius (the search cap itself),
            # which is what a large contiguous mask region unrelated to
            # hair looks like (there, most likely dark clothing the
            # segmenter mistook for hair), not four coincidentally
            # identical real hairstyle measurements. A search that never
            # finds its own edge isn't a measurement, same principle as
            # the silhouette width fix above.
            continue
        if outer < head_r * (1.0 + HAIR_MIN_RADIUS_FRAC):
            continue

        inner = head_r * 0.6  # start the bump a bit inside the head surface, for overlap
        bump_r = max((outer - inner) / 2.0, head_r * 0.08)
        center_dist = inner + bump_r
        center = head_center + np.array([dx, dy, 0.0]) * center_dist
        bumps.append((center, center, bump_r, bump_r, HAIR_DEPTH_RATIO, HAIR_DEPTH_RATIO, group))
        group -= 1
    return bumps


def _build_capsules(
    joints: dict, silhouette_mask=None, image_width=None, image_height=None, hair_mask=None
) -> list:
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
        idx_pt = joints.get(f"{side}_index")
        pinky_pt = joints.get(f"{side}_pinky")
        if idx_pt is not None and pinky_pt is not None:
            joints[f"{side}_hand_mid"] = (idx_pt + pinky_pt) / 2.0

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
        ("left_shoulder_attach", "left_elbow", shoulder_width * UPPER_ARM_RADIUS_FRAC, shoulder_width * UPPER_ARM_RADIUS_FRAC * ARM_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_shoulder_attach", "right_elbow", shoulder_width * UPPER_ARM_RADIUS_FRAC, shoulder_width * UPPER_ARM_RADIUS_FRAC * ARM_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("left_elbow", "left_wrist", shoulder_width * FOREARM_RADIUS_FRAC, shoulder_width * FOREARM_RADIUS_FRAC * ARM_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_elbow", "right_wrist", shoulder_width * FOREARM_RADIUS_FRAC, shoulder_width * FOREARM_RADIUS_FRAC * ARM_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("left_hip_attach", "left_knee", hip_width * THIGH_RADIUS_FRAC, hip_width * THIGH_RADIUS_FRAC * LEG_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_hip_attach", "right_knee", hip_width * THIGH_RADIUS_FRAC, hip_width * THIGH_RADIUS_FRAC * LEG_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("left_knee", "left_ankle", hip_width * SHIN_RADIUS_FRAC, hip_width * SHIN_RADIUS_FRAC * LEG_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
        ("right_knee", "right_ankle", hip_width * SHIN_RADIUS_FRAC, hip_width * SHIN_RADIUS_FRAC * LEG_TAPER, DEPTH_RATIO_LIMB, DEPTH_RATIO_LIMB),
    ]

    # Hand capsules: a palm from wrist to the index/pinky midpoint, plus a
    # thumb, each its own group so they blend smoothly into the forearm
    # at the wrist the same way the forearm blends into the upper arm —
    # not appended as a dead-end cap. Skipped per-side (not a hard
    # requirement like legs above) when that hand's landmarks weren't
    # confidently detected, same fallback as every other optional joint
    # in this function.
    wrist_r = shoulder_width * FOREARM_RADIUS_FRAC * ARM_TAPER
    for side in ("left", "right"):
        hand_mid_name = f"{side}_hand_mid"
        if hand_mid_name not in joints:
            continue
        palm_r = wrist_r * PALM_RADIUS_FRAC_OF_WRIST
        segs.append((f"{side}_wrist", hand_mid_name, wrist_r, palm_r, DEPTH_RATIO_LIMB, PALM_DEPTH_RATIO))
        thumb_name = f"{side}_thumb"
        if thumb_name in joints:
            thumb_r = palm_r * THUMB_RADIUS_FRAC_OF_PALM
            segs.append((f"{side}_wrist", thumb_name, wrist_r * 0.6, thumb_r, DEPTH_RATIO_LIMB, THUMB_DEPTH_RATIO))
    use_silhouette = silhouette_mask is not None and image_width and image_height
    logical_count = 0
    for group_id, (a, b, ra, rb, dra, drb) in enumerate(segs):
        seg = _segment(joints, a, b, ra, rb, dra, drb, group=group_id)
        if not seg:
            continue
        logical_count += 1
        # The head keeps its single generic capsule — real shape there
        # already comes from face_features.py's landmark-driven nose/chin/
        # eye additions, and a hairline/hat can make a perpendicular
        # silhouette width at the scalp read as noise rather than signal.
        if use_silhouette and a != "head_top":
            ja, jb = seg[0], seg[1]
            chain = _chain_from_silhouette(
                silhouette_mask, image_width, image_height, ja, jb, ra, rb, dra, drb,
                SILHOUETTE_SAMPLES_PER_SEGMENT, group_id,
            )
            if chain:
                capsules.extend(chain)
                continue
        capsules.append(seg)

    if logical_count < MIN_CAPSULES:
        raise RuntimeError(
            f"Not enough body parts detected to build a figure ({logical_count}, "
            f"need at least {MIN_CAPSULES}) — use a clear, front-facing photo "
            "showing most of the body."
        )

    result = mad_margin_above_minimum(logical_count, MIN_CAPSULES, min_margin=0.3)
    if not result.ok:
        print(
            f"Warning: only {logical_count} body parts detected, {result.margin:.0%} "
            f"above the {MIN_CAPSULES}-part minimum — the figure will be missing "
            "limbs or look rough. A clearer full-body photo will give a better result."
        )

    if hair_mask is not None and "head_top" in joints:
        head_center = (joints["head_top"] + neck) / 2.0
        # Negative group ids, not continuing the body's own numbering —
        # a simple, unambiguous marker build_body_mesh can filter on
        # without needing to know how many body segments there were, to
        # keep hair bumps (tiny next to any body capsule) out of the
        # body's own blend-width calculation the same way face parts
        # already are (see build_body_mesh).
        capsules.extend(
            _hair_bumps(hair_mask, image_width, image_height, head_center, head_r, group_start=-1)
        )

    return capsules


def _face_to_local_xyz(face_kp: FaceKeypoints, target_image_width: Optional[float] = None) -> dict:
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
    jaw corners the smallest (furthest back), matching real face shape.

    `target_image_width`: face_features.py deliberately detects at the
    photo's full resolution rather than inheriting pose detection's
    1280px cap (a face is often small in a full-body frame and benefits
    from the extra pixels — see that module). That means `face_kp`'s own
    image dimensions usually differ from the pose keypoints' — same
    photo, different pixel density — so local coordinates computed
    straight from face_kp's own width/height would be a different
    *scale* than the body's, not just a different origin, and face parts
    would land at the wrong size/position relative to the body for any
    photo actually larger than 1280px (untested by every synthetic photo
    used so far in this project, which all happened to be smaller than
    that cap, so this a real bug this project would otherwise have
    shipped). Passing the body's own image_width here rescales face-local
    coordinates into the same pixel-unit space the body skeleton uses,
    correcting for that density difference — both describe the same
    physical photo, just sampled at different resolutions."""
    w, h = face_kp.image_width, face_kp.image_height
    scale = (target_image_width / w) if target_image_width else 1.0
    out = {}
    for idx, (px, py, z_raw) in face_kp.points.items():
        out[idx] = np.array([(px - w / 2.0) * scale, (h / 2.0 - py) * scale, -z_raw * w * scale])
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


def _bake_photo_colors(verts: np.ndarray, image_width: int, image_height: int, rgb: np.ndarray) -> np.ndarray:
    """Per-vertex color sampled from the source photo itself, not a flat
    material guess. The mesh is built directly in the photo's own
    pixel-centered coordinate frame (see _to_local_xy), so a camera-facing
    vertex (z > 0, this module's convention — see build_body_mesh) maps
    straight back to the exact pixel the photo showed there; no UV
    unwrapping needed.
    The back half (z <= 0) was never photographed, so there is no real
    color to sample. Copying the nearest camera-facing vertex's color is
    an honest fallback — it reuses real, physically adjacent evidence —
    instead of either inventing a color or mirroring front-to-back, which
    would paste a face onto someone's back."""
    from scipy.spatial import cKDTree

    front = verts[:, 2] > 0
    colors = np.zeros((len(verts), 3), dtype=np.uint8)
    if not front.any():
        colors[:] = (190, 160, 140)  # no photo evidence anywhere on this mesh
        return colors

    px = verts[front, 0] + image_width / 2.0
    py = image_height / 2.0 - verts[front, 1]
    row = np.clip(py, 0, image_height - 1)
    col = np.clip(px, 0, image_width - 1)
    sampled = np.stack(
        [map_coordinates(rgb[..., c].astype(np.float64), [row, col], order=1, mode="nearest") for c in range(3)],
        axis=-1,
    )
    colors[front] = np.clip(sampled, 0, 255).astype(np.uint8)

    back = ~front
    if back.any():
        tree = cKDTree(verts[front])
        _, nearest = tree.query(verts[back])
        colors[back] = colors[front][nearest]

    return colors


def build_body_mesh(
    keypoints: DetectedKeypoints,
    target_height_mm: float = 150.0,
    grid_resolution: int = 190,
    blend_k_frac: float = 0.35,
    depth_rgb: Optional[np.ndarray] = None,
    depth_strength: float = 0.5,
    face_keypoints: Optional[FaceKeypoints] = None,
    silhouette_mask: Optional[np.ndarray] = None,
    hair_mask: Optional[np.ndarray] = None,
    texture_rgb: Optional[np.ndarray] = None,
) -> BodyMesh:
    """`depth_rgb`: the same preprocessed photo passed to pose detection.
    When given, sculpts the front surface using real per-pixel depth
    (anatomy3d.depth_source, optional extra) instead of the flat
    symmetric-thickness guess. Leave as None for the default, dependency-
    free behavior.

    `texture_rgb`: the same preprocessed photo, same coordinate system as
    `keypoints` — same requirement as `silhouette_mask`. When given,
    colors every camera-facing vertex with that exact pixel's real photo
    color (see _bake_photo_colors); the un-photographed back is filled
    from the nearest camera-facing vertex's color. Unlike `depth_rgb`,
    needs no optional extra — only the hard numpy/scipy dependencies this
    module already has. Leave as None to keep the mesh a flat material
    color (the previous, and still the default, behavior).

    `face_keypoints`: real detected face landmarks (face_features.py).
    When given and usable, overlays a nose, chin, and eye sockets built
    from the photo's own measured positions on top of the generic head
    capsule. Optional — a missing or unusable face silently adds nothing.

    `silhouette_mask`: the same photo's own silhouette (silhouette.py),
    in the same pixel coordinate system as `keypoints` (same image
    dimensions). When given, replaces each torso/limb segment's straight
    generic taper with a chain measured from the photo's actual outline —
    see SILHOUETTE_SAMPLES_PER_SEGMENT's comment. Optional — without it,
    every segment falls back to the previous generic-proportion capsule.

    `hair_mask`: the same photo's hair mask (hair_features.py), same
    pixel coordinate system as `keypoints`. When given, adds a stylized
    hair volume shaped from the photo's own hairline (see HAIR_SAMPLES's
    comment). Optional — without it, the head stays bare."""
    joints = _to_local_xy(keypoints)
    all_capsules = _build_capsules(
        joints,
        silhouette_mask=silhouette_mask,
        image_width=keypoints.image_width,
        image_height=keypoints.image_height,
        hair_mask=hair_mask,
    )
    # Hair bumps are tagged with negative group ids (see _hair_bumps) so
    # they can be split out here and given their own blend width, the
    # same reason face parts already are below — a tiny hair bump's
    # radius would otherwise shrink the body's own blend_k for no benefit.
    capsules = [cap for cap in all_capsules if cap[6] >= 0]
    hair_bumps = [cap for cap in all_capsules if cap[6] < 0]

    all_radii = [r for cap in capsules for r in cap[2:4]]
    margin = max(all_radii) * 2.5
    pts_for_bounds = np.array([p for cap in capsules for p in cap[:2]])

    face_capsules, eye_sockets = [], []
    if face_keypoints is not None:
        face_local = _face_to_local_xyz(face_keypoints, target_image_width=keypoints.image_width)
        face_capsules, eye_sockets = _build_face_additions(face_local)
    if face_capsules:
        # Bounds must cover the face additions too, even though they're
        # already within the head capsule's own margin in practice — kept
        # as its own min/max rather than folded into `margin`/`all_radii`
        # above, so face parts (small) don't shrink the main blend width
        # (see blend_k below, computed from body-only radii).
        face_pts = np.array([p for cap in face_capsules for p in cap[:2]] + [c for c, _r in eye_sockets])
        pts_for_bounds = np.vstack([pts_for_bounds, face_pts])

    if hair_bumps:
        hair_pts = np.array([p for cap in hair_bumps for p in cap[:2]])
        pts_for_bounds = np.vstack([pts_for_bounds, hair_pts])

    mins = pts_for_bounds.min(axis=0) - margin
    maxs = pts_for_bounds.max(axis=0) + margin
    # Give the figure real thickness front-to-back even though the input is
    # a single frontal photo — no depth data exists, so this is a flat
    # stylization choice, not a measurement.
    depth_half = max(all_radii) * 3.0
    mins[2], maxs[2] = -depth_half, depth_half

    dims = maxs - mins

    # Not just a face-detail concern: ANY capsule thinner than the grid's
    # own voxel size gets rounded away by marching cubes, same failure
    # mode either way (see the face case below, where this was first
    # measured and calibrated). It just took a real photo to surface it
    # for ordinary body limbs — every synthetic test built so far used a
    # compact standing pose, but a photo with limbs spread wide (arms out,
    # a lunging leg) blows up dims.max() without the limbs themselves
    # getting any thicker, which can push the default resolution's voxel
    # size past even a forearm's or shin's own radius. Measured directly
    # on MediaPipe's own real `pose.jpg` sample (a side lunge, arms out):
    # voxel size 3.26 against a thinnest-limb radius of 2.11 (ratio 1.54,
    # already past the 1.1-ish ratio this project's own calibration below
    # found insufficient) rendered the forearms and shins as near-invisible
    # hairlines instead of tapered capsules. Unlike face detail, limbs
    # aren't optional — there's no "skip it" here, only "resolve it",
    # capped to bound cost.
    required_resolution_body = dims.max() / (min(all_radii) * FACE_VOXEL_FRAC)
    grid_resolution = int(np.clip(required_resolution_body, grid_resolution, MAX_GRID_RESOLUTION_BODY))

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
        # prediction once voxel size roughly matched the smallest feature's
        # own radius — not several voxels across it, an earlier guess that
        # was checked against real numbers and found to demand a far finer
        # grid than necessary). So: sharpen the grid only as far as needed
        # to resolve the smallest face feature, capped to bound runtime
        # cost — and if even the cap can't get there (a face that's a very
        # small fraction of the frame), skip adding the face geometry
        # rather than silently shipping a change too small for any viewer
        # or slicer to ever see.
        face_radii = [r for cap in face_capsules for r in cap[2:4]] + [r for _c, r in eye_sockets]
        required_resolution_face = dims.max() / (min(face_radii) * FACE_VOXEL_FRAC)
        if required_resolution_face > MAX_GRID_RESOLUTION_WITH_FACE * FACE_SKIP_FACTOR:
            print(
                "Note: a face was detected, but it's too small relative to the "
                "whole-body photo to render fine facial detail at a practical "
                "resolution — the generic head shape will be used instead. A "
                "closer, more face-filling photo (or a face/half-body shot) "
                "would let this feature actually show up."
            )
            face_capsules, eye_sockets = [], []
        else:
            grid_resolution = int(
                np.clip(required_resolution_face, grid_resolution, MAX_GRID_RESOLUTION_WITH_FACE)
            )

    if hair_bumps:
        # Same reasoning and same thresholds as the face case just above —
        # a hair bump can be smaller than the grid can resolve too, and
        # there's no reason to invent a second set of constants for an
        # identical problem.
        hair_radii = [r for cap in hair_bumps for r in cap[2:4]]
        required_resolution_hair = dims.max() / (min(hair_radii) * FACE_VOXEL_FRAC)
        if required_resolution_hair > MAX_GRID_RESOLUTION_WITH_FACE * FACE_SKIP_FACTOR:
            print(
                "Note: hair was detected, but it's too small relative to the "
                "whole-body photo to render at a practical resolution — the "
                "head will stay bare. A closer, more face-filling photo "
                "would let this feature actually show up."
            )
            hair_bumps = []
        else:
            grid_resolution = int(
                np.clip(required_resolution_hair, grid_resolution, MAX_GRID_RESOLUTION_WITH_FACE)
            )

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
    group_field = None
    current_group = None
    for a, b, ra, rb, dra, drb, grp in capsules:
        sdf = _capsule_sdf(points, a, b, ra, rb, dra, drb)
        if grp == current_group:
            # Same logical segment (a silhouette chain's own pieces, or a
            # single-capsule segment on its own): these already meet
            # exactly at a shared endpoint with matching radius, so a
            # plain hard union is already seamless — smooth-min would only
            # add an unwanted fillet-bulge at every one of these joints,
            # which compounds across a 6-piece chain into a visibly fatter
            # limb than any individual measurement called for (measured
            # directly: this was the actual cause of limbs ballooning and
            # fusing into the torso on a real test photo before this fix).
            group_field = np.minimum(group_field, sdf)
        else:
            if group_field is not None:
                field = group_field if field is None else _smooth_min(field, group_field, blend_k)
            group_field = sdf
            current_group = grp
    if group_field is not None:
        field = group_field if field is None else _smooth_min(field, group_field, blend_k)

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

    if hair_bumps:
        # Same reasoning as face_blend_k just above: hair bumps need their
        # own, hair-scaled blend width, not the body's.
        hair_radii = [r for cap in hair_bumps for r in cap[2:4]]
        hair_blend_k = min(hair_radii) * HAIR_BLEND_FRAC * 5.0
        for a, b, ra, rb, dra, drb, _grp in hair_bumps:
            sdf = _capsule_sdf(points, a, b, ra, rb, dra, drb)
            field = _smooth_min(field, sdf, hair_blend_k)

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

    colors = None
    if texture_rgb is not None:
        # Before scaling, same reasoning as the depth-sculpt call above:
        # the pixel<->local-coordinate mapping only holds in this unscaled
        # space.
        colors = _bake_photo_colors(verts, keypoints.image_width, keypoints.image_height, texture_rgb)

    height = verts[:, 1].max() - verts[:, 1].min()
    scale = target_height_mm / height if height > 1e-6 else 1.0
    verts = verts * scale

    return BodyMesh(vertices=verts, faces=faces, colors=colors)
