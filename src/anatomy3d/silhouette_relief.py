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
from typing import Optional

import numpy as np
import trimesh
from PIL import Image
from scipy import sparse
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.sparse.linalg import spsolve
from skimage.measure import marching_cubes
from skimage.morphology import medial_axis

from .mesh_types import BodyMesh
from .procedural_body import smooth_grid_terracing

# A rough, illustrative figure-proportion convention (legs roughly as
# long as the subject's own visible height, times this factor) -- NOT
# a cited anthropometric measurement like procedural_body.py's ANSUR-
# derived ratios above it. This is explicitly a generic placeholder for
# a photo that was framed to cut the subject off, not an attempt at a
# precise estimate -- there is no way to know a cropped photo's subject's
# real leg length from the photo alone. Tied to visible height rather
# than the cut's own width: using width was a real bug (see below).
_CROP_LEG_LENGTH_TO_VISIBLE_HEIGHT = 0.9
_CROP_UNDIVIDED_FRAC = 0.12  # hip/upper-thigh taper before the legs separate
_CROP_LEG_GAP_FRAC = 0.12  # crotch gap between the two legs, as a fraction of the cut width
_CROP_ANKLE_WIDTH_FRAC = 0.4  # each leg's width at its far end, relative to its own width at the split


_CROP_MAX_ANIMAL_FRAC = 0.15  # above this, trust the animal channel over a bottom-edge width heuristic
_CROP_TOP_BAND_FRAC = 0.18  # fraction of the mask's own height searched for separate digit-like prongs
_CROP_MAX_TOP_COMPONENTS = 2  # 3+ separate pieces up there reads as fingers (or several distinct subjects)
_CROP_MIN_COMPONENT_PX = 20  # ignore specks this small when counting top-band pieces
_CROP_MIN_COMPONENT_FRAC_OF_LARGEST = 0.05  # ALSO ignore anything under this fraction of
# the band's largest piece -- confirmed directly as a real, not hypothetical, failure of the
# absolute floor above on its own: a genuine single-subject torso/hat photo produced two
# ~35px flecks of BiRefNet's own edge-matting noise next to its real ~4400px head/hat blob,
# clearing the 20px floor and pushing the count to 3, which incorrectly blocked leg extension
# on a photo that should have gotten it. A stray matting fleck is a tiny fraction of the real
# subject's own piece; a second finger or a second animal/person is not.


def _looks_like_digits_or_multi_subject(mask: np.ndarray) -> bool:
    """A hand's fingers (or several separate animals/people in one
    frame) show up as multiple separate pieces near the top of the
    mask's own bounding box -- a real human torso or a single animal's
    body doesn't. Confirmed directly against real photos, not assumed:
    a hand came back with 3-4 separate pieces there, a multi-animal
    photo with 10, every single-subject torso/animal photo tried with
    1-2 (an animal's two ears sometimes separate briefly near the very
    top). Adding a human leg extension below a hand or a group of
    animals would be absurd regardless of how the bottom edge looks, so
    this check runs independently of, and before, the bottom-edge
    signals below."""
    from scipy import ndimage

    ys, xs = np.nonzero(mask)
    y0 = ys.min()
    band_h = max(1, int(round((ys.max() - y0 + 1) * _CROP_TOP_BAND_FRAC)))
    band = mask[y0 : y0 + band_h, :]
    labeled, n = ndimage.label(band)
    if n == 0:
        return False
    sizes = ndimage.sum(band, labeled, index=range(1, n + 1))
    min_size = max(_CROP_MIN_COMPONENT_PX, max(sizes) * _CROP_MIN_COMPONENT_FRAC_OF_LARGEST)
    real_components = sum(1 for s in sizes if s > min_size)
    return real_components > _CROP_MAX_TOP_COMPONENTS


def _detect_crop_and_extend(mask: np.ndarray, animal_frac: float = 0.0) -> np.ndarray:
    """If the mask was cut off by the photo's own frame rather than
    ending at a real body boundary (a foot, a hand, an animal's tail),
    extends it downward with a generic tapered leg shape instead of
    leaving a flat stump at the cut -- confirmed directly as a real,
    not hypothetical, problem: a torso-only photo (a man in a hat,
    cropped at the waist) produced exactly that flat-bottomed mesh.

    Multiple independent signals have to agree before adding legs, not
    just one -- the first version of this function used only the three
    bottom-edge checks below and, tested directly against every cached
    real mask from this project's own test photos, fired on roughly
    half of them, including a cat portrait, a multi-animal photo, and a
    hand close-up (confirmed directly it would have grafted human legs
    onto all three). Two more checks, run first:
      0a. animal_frac (from person_segmenter.detect_person_alpha_with_source)
          is low -- a real animal subject gets excluded even though its
          own bottom-edge width profile can look exactly like a cropped
          human torso (a standing dog's legs don't taper to a point at
          ground level the way the detector originally expected).
      0b. the mask doesn't look like a hand or several separate subjects
          (see _looks_like_digits_or_multi_subject) -- a hand's own
          wrist is wide and un-tapered at the crop edge, same shape
          signature as a real torso crop, so the bottom-edge checks
          alone can't tell them apart; this can.
    Then, same as before, three bottom-edge signals:
      1. the mask actually touches the bottom row at all;
      2. the cross-section width right at that row is still a large
         fraction of the subject's own overall max width -- a real
         extremity (foot, hand, tail) tapers toward a point, a cropped
         torso/limb does not;
      3. that width isn't still shrinking as it approaches the edge --
         a real taper is still narrowing in its last few rows; a crop
         is cut off mid-shape, usually roughly flat or still widening.
    Only the bottom edge is handled -- by far the common real case
    (a photo framed from the waist/chest up) -- not all four edges."""
    if animal_frac > _CROP_MAX_ANIMAL_FRAC:
        return mask  # signal 0a: the animal channel itself identified this subject
    if _looks_like_digits_or_multi_subject(mask):
        return mask  # signal 0b: a hand, or more than one subject, not a torso

    h, w = mask.shape
    if not mask[h - 1, :].any():
        return mask  # signal 1: doesn't even touch the bottom edge

    row_widths = np.array([np.count_nonzero(mask[y, :]) for y in range(h)])
    max_width = row_widths.max()
    if max_width == 0:
        return mask
    edge_width = row_widths[h - 1]
    if edge_width < max_width * 0.35:
        return mask  # signal 2: tapering toward a point, a real extremity

    trend_window = max(3, h // 50)
    recent = row_widths[max(0, h - trend_window) :]
    if recent[-1] < recent[0] * 0.85:
        return mask  # signal 3: still visibly narrowing right up to the edge

    xs_edge = np.nonzero(mask[h - 1, :])[0]
    x_left, x_right = xs_edge.min(), xs_edge.max() + 1
    cut_width = x_right - x_left
    center_x = (x_left + x_right) / 2.0

    # Length basis is the photo's own VISIBLE height, not the cut's
    # width -- using width was a real bug, caught only by actually
    # rendering the result, not by the numbers alone: at the tuned
    # 3.2x-of-width multiplier, a real torso photo came out as a tiny
    # head+shoulders sliver on top of absurdly long, thread-thin legs,
    # several times taller than the visible torso itself. Visible
    # height is at least anchored to the same photo's own real
    # proportions, even though exactly where on the body the crop falls
    # (chest? waist?) still isn't known.
    visible_top = int(np.nonzero(mask.any(axis=1))[0].min())
    visible_height = h - visible_top
    leg_length = int(round(visible_height * _CROP_LEG_LENGTH_TO_VISIBLE_HEIGHT))
    undivided_rows = max(1, int(round(leg_length * _CROP_UNDIVIDED_FRAC)))
    split_rows = leg_length - undivided_rows

    extension = np.zeros((leg_length, w), dtype=bool)
    half_width = cut_width / 2.0
    for i in range(undivided_rows):
        t = i / max(1, undivided_rows - 1)
        row_half = half_width * (1.0 - 0.15 * t)
        lo = max(0, int(round(center_x - row_half)))
        hi = min(w, int(round(center_x + row_half)))
        extension[i, lo:hi] = True

    gap = cut_width * _CROP_LEG_GAP_FRAC
    leg_width_at_split = (cut_width - gap) / 2.0
    left_center = center_x - gap / 2.0 - leg_width_at_split / 2.0
    right_center = center_x + gap / 2.0 + leg_width_at_split / 2.0
    for i in range(split_rows):
        t = i / max(1, split_rows - 1)
        leg_half = (leg_width_at_split / 2.0) * (1.0 - (1.0 - _CROP_ANKLE_WIDTH_FRAC) * t)
        row = undivided_rows + i
        for cx in (left_center, right_center):
            lo = max(0, int(round(cx - leg_half)))
            hi = min(w, int(round(cx + leg_half)))
            extension[row, lo:hi] = True

    return np.concatenate([mask, extension], axis=0)


_LAYER_RESCUE_MIN_ANIMAL_FRAC = _CROP_MAX_ANIMAL_FRAC  # reuse the same "confidently an
# animal subject" boundary already established above, rather than inventing a second one
_LAYER_RESCUE_N_LAYERS = 4
_LAYER_RESCUE_SMOOTH_SIGMA_PX = 2.0  # suppresses single-pixel depth-model noise before clustering
_LAYER_RESCUE_STRENGTH = 0.5
_LAYER_RESCUE_MAX_BOOST = 4.0


def _kmeans_1d(vals: np.ndarray, k: int, iters: int = 25):
    """Plain 1D Lloyd's-algorithm k-means (no sklearn dependency for
    one small clustering step). Returns (centroids sorted far->near,
    per-value cluster index into that sorted order)."""
    centroids = np.quantile(vals, np.linspace(0.1, 0.9, k))
    assign = np.zeros(vals.shape[0], dtype=int)
    for it in range(iters):
        d = np.abs(vals[:, None] - centroids[None, :])
        new_assign = d.argmin(axis=1)
        if it > 0 and np.array_equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        centroids = np.array([
            vals[assign == i].mean() if np.any(assign == i) else centroids[i]
            for i in range(k)
        ])
    order = np.argsort(centroids)
    rank = np.empty_like(order)
    rank[order] = np.arange(k)
    return centroids, rank[assign]


def _layered_detail_rescue(sampled, grid_mask, real_grid_mask, global_std):
    """An ANIMAL-ONLY addition to the plain global z-score depth bias
    (see build_silhouette_relief_mesh's depth_rgb branch), gated behind
    animal_frac. Confirmed directly, not assumed, why it's gated this
    way and not applied universally:

    A subject like a dog has real, physically-distinct depth layers
    (ear, head/snout, body, legs) -- but a single whole-subject z-score
    compresses the SMALL internal relief within any one of those layers
    (e.g. the snout's own shape) into a tiny sliver of the range the
    big ear-vs-legs difference dominates. Clustering the subject's own
    depth values into discrete layers (k-means) and rescuing each
    layer's own internal detail -- but ONLY where that layer's own std
    is demonstrably smaller than the subject's global std, i.e. where
    the global term is provably under-representing it -- recovers real
    structure there (confirmed: the dog's head/ear region went from an
    almost-flat crease to a genuinely contoured shape).

    Tried applying the exact same thing to a human portrait and it was
    a clear regression (confirmed directly, not assumed): a human face
    is much closer to one continuous surface, so per-layer clustering
    mostly carves up smooth lighting/skin gradients into arbitrary
    bands and "rescues" what is actually noise, not real geometry --
    turned a clean, recognizable face into a noisy, mask-like one even
    after several rounds of tuning (smoothing first, flooring the
    per-layer normalization, capping the boost). None of those fixed
    it, which is itself the evidence that it's the wrong tool for a
    human face rather than a tuning problem -- so this is gated to
    animal_frac instead of searching further for one strength value
    that works everywhere.

    Returns an ADDITIVE term for d_norm (same units: fraction of
    half_thickness), zero outside the real (non-synthetic-leg) subject
    mask."""
    smoothed = gaussian_filter(sampled, sigma=_LAYER_RESCUE_SMOOTH_SIGMA_PX)
    subj_vals = smoothed[real_grid_mask]
    if subj_vals.size < _LAYER_RESCUE_N_LAYERS * 4 or global_std <= 1e-9:
        return np.zeros_like(sampled)

    centroids, layer_of_subj = _kmeans_1d(subj_vals, _LAYER_RESCUE_N_LAYERS)
    layer_map = np.abs(smoothed[..., None] - centroids[None, None, :]).argmin(axis=-1)
    layer_centroid_map = centroids[layer_map]
    detail = smoothed - layer_centroid_map

    subj_layer_map = np.full(smoothed.shape, -1)
    subj_layer_map[np.nonzero(real_grid_mask)] = layer_of_subj

    rescue = np.zeros_like(smoothed)
    for i in range(_LAYER_RESCUE_N_LAYERS):
        layer_mask_full = (layer_map == i) & real_grid_mask
        layer_vals = detail[(subj_layer_map == i) & real_grid_mask]
        layer_std = layer_vals.std() if layer_vals.size > 1 else 0.0
        if layer_std <= 1e-9:
            continue
        # a layer whose own std already matches/exceeds the global std needs no rescue --
        # the global z-score term already represents it; boost=1 leaves it untouched below
        boost = np.clip(global_std / layer_std, 1.0, _LAYER_RESCUE_MAX_BOOST)
        # Clipped to +/-3 "layer-detail sigma" -- confirmed directly as a necessary,
        # not cosmetic, fix: a single noisy outlier pixel divided by a small layer_std
        # can produce a huge ratio, and that single spike was enough to blow out z_max
        # (see below) even after heavily smoothing the field, since a blur softens a
        # spike's edges but barely lowers its peak.
        layer_detail_norm = np.clip(detail / layer_std, -3.0, 3.0)
        rescue = np.where(
            layer_mask_full,
            layer_detail_norm * (boost - 1.0) / max(_LAYER_RESCUE_MAX_BOOST - 1.0, 1e-6),
            rescue,
        )
    # The per-layer assignment is a hard nearest-centroid pick, so `rescue` has a sharp
    # step everywhere two layers meet. Smoothing it here turns that sharp terracing into
    # continuous relief, which is also more anatomically honest -- a real head doesn't
    # have literal shelf-like steps.
    #
    # NOTE on mesh size, confirmed directly rather than assumed: this grid_size (360) and
    # a real high-resolution photo already produce a large raw marching_cubes mesh
    # (300k+ vertices) even with NO depth bias and NO rescue at all -- that's pre-existing,
    # not caused by this function. What IS caused by this function: on at least one real
    # test photo, export_stl's quadric decimation to the target face count, which
    # succeeds and stays watertight on the plain/no-rescue mesh at a near-identical raw
    # vertex count, started failing to stay watertight once this rescue term was added --
    # despite trying heavier smoothing (confirmed up to sigma=20px, ~10x this default,
    # with no improvement) and clipping outlier spikes (the fix above, confirmed
    # necessary but not sufficient on its own). The likely cause is some specific local
    # curvature/degenerate-triangle pattern this term introduces that trips up
    # decimation, not raw size or smoothness -- not yet root-caused further than that.
    # export_stl's own fallback (see print_prep.simplify_for_output) already handles this
    # safely: if decimation isn't watertight, it ships the larger un-decimated mesh
    # instead, so this is a real file-size cost on such photos, not a correctness bug.
    rescue = gaussian_filter(rescue, sigma=_LAYER_RESCUE_SMOOTH_SIGMA_PX * 2.0)
    return rescue * _LAYER_RESCUE_STRENGTH


# Features fade to nothing over this fraction of the face's width at its
# outer edge, so the relief doesn't end in a step at the hairline or jaw.
_FACE_RELIEF_FADE_FRAC = 0.2
_FACE_RELIEF_SMOOTH_FRAC = 0.018
_FACE_SURFACE_POINTS = 468
_FACE_RELIEF_WITH_DEPTH = 0.45


# Front-to-back : side-to-side ratio for the head. ANSUR II puts head
# length : head breadth at 1.29 (procedural_body.DEPTH_RATIO_HEAD), but
# the photo's head outline also carries hair and ears, so it's measured
# wider than ANSUR's breadth; 1.15 lands the depth near the real one.
_HEAD_DEPTH_RATIO = 1.15
_THICKNESS_BLUR_FRAC = 0.015  # smoothing of the width/gain fields, as a fraction of the grid


def _masked_blur(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    weight = gaussian_filter(mask.astype(np.float32), sigma)
    blurred = gaussian_filter(np.where(mask, values, 0.0).astype(np.float32), sigma)
    return np.where(mask, blurred / np.maximum(weight, 1e-6), 0.0)


def _head_weight(face_mesh: np.ndarray, grid_shape, y0: int, x0: int, scale: float, pad: int) -> np.ndarray:
    """1 over the head (an ellipse around the face landmarks, extended up
    past the forehead to the crown), fading to 0 just outside it."""
    pts = face_mesh[:_FACE_SURFACE_POINTS]
    rows = (pts[:, 1] - y0) * scale + pad
    cols = (pts[:, 0] - x0) * scale + pad
    face_h = float(rows.max() - rows.min())
    crown = float(rows.min()) - 0.35 * face_h
    chin = float(rows.max())
    cy, ry = (crown + chin) / 2.0, max((chin - crown) / 2.0, 1.0)
    cx, rx = float(cols.mean()), max(0.7 * float(np.ptp(cols)), 1.0)
    gr, gc = np.mgrid[0 : grid_shape[0], 0 : grid_shape[1]]
    rn = np.sqrt(((gr - cy) / ry) ** 2 + ((gc - cx) / rx) ** 2)
    return np.clip((1.25 - rn) / 0.35, 0.0, 1.0).astype(np.float32)


def _proportioned_thickness(
    grid_mask: np.ndarray,
    inflation: np.ndarray,
    torso_ratio: float,
    limb_ratio: float,
    head_weight: Optional[np.ndarray],
) -> np.ndarray:
    """Half-thickness (grid pixels) with real body proportions.

    The inflation alone only reaches a part's own half-width on long thin
    shapes; on compact ones (head, torso) it stops at about 0.7 of it, so
    with the ratio on top every head came out a thin blade and every
    torso an almond from the side. Here the inflation is rescaled along
    the outline's medial axis so its peak equals the local half-width,
    then multiplied by the front-to-back : side-to-side ratio for that
    part: limbs where the outline is narrowest, torso where it's widest,
    head over the face."""
    dist = distance_transform_edt(grid_mask).astype(np.float32)
    skel = medial_axis(grid_mask, rng=0) & (inflation > 1e-6)  # seeded: it breaks ties at random, and the same photo should give the same model
    if not skel.any():
        return inflation * torso_ratio
    gain_on_axis = np.zeros_like(dist)
    gain_on_axis[skel] = np.clip(dist[skel] / inflation[skel], 0.8, 1.6)
    width_on_axis = np.where(skel, dist, 0.0)
    _, (ri, ci) = distance_transform_edt(~skel, return_indices=True)
    sigma = max(_THICKNESS_BLUR_FRAC * max(grid_mask.shape), 1.0)
    gain = _masked_blur(gain_on_axis[ri, ci], grid_mask, sigma)
    half_width = _masked_blur(width_on_axis[ri, ci], grid_mask, sigma)
    inside = half_width[grid_mask]
    lo, hi = float(inside.min()), float(inside.max())
    width_frac = np.clip((half_width - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    ratio = limb_ratio * (1.0 - width_frac) + torso_ratio * width_frac
    if head_weight is not None:
        ratio = ratio * (1.0 - head_weight) + _HEAD_DEPTH_RATIO * head_weight
    return inflation * gain * ratio


# Arms vs body. The front outline alone can't tell an arm crossed in front of
# the chest from the chest itself: inflating the whole outline gave the arms
# the same depth BEHIND the body as in front -- a bulge on the back of every
# crossed-arm build. With pose points the arms are traced as their own tubes
# and the back is built from the torso alone. Sizes are ANSUR II male means
# relative to the shoulder-joint distance (biacromial breadth 415.7 mm):
# biceps circumference / 2pi = 0.137, forearm 0.119, wrist 0.067, hand
# breadth / 2 = 0.106, chest breadth 0.697, waist breadth 0.785.
_ARM_R = {"shoulder": 0.137, "elbow": 0.125, "wrist": 0.067, "hand": 0.090}
_TORSO_BREADTH = (0.697, 0.785)  # chest -> waist, x shoulder distance
_CLOTHING = 1.12  # sleeves and jacket over the bare-body measurements
_ARM_IN_FRONT_FRAC = 0.3  # share of the forearm lying across the torso that marks it as crossed in front


# Side widths from height. ANSUR II male means, each a fraction of stature
# (4,082 men; spread in brackets): head length 0.1137 (4%), neck
# circumference/pi 0.0721 (7%), chest depth 0.1446 (10%), waist depth
# 0.1355 (14%), buttock depth 0.1401 (11%), thigh circumference/pi 0.1134,
# lower-thigh (knee) 0.0742, calf 0.0711, ankle 0.0416, biceps (flexed)
# 0.0650, forearm (flexed) 0.0562, wrist 0.0319.
_DEPTH_OF_STATURE = {
    "head": 0.1137, "neck": 0.0721, "chest": 0.1446, "waist": 0.1355, "buttock": 0.1401,
    "thigh": 0.1134, "knee": 0.0742, "calf": 0.0711, "ankle": 0.0416,
    "upperarm": 0.0650, "forearm": 0.0562, "wrist": 0.0319,
}
# Scale anchors (fraction of stature): shoulder joints (biacromial), the
# outline across the shoulders (bideltoid), and three face measures. A
# portrait's head usually looks bigger than average next to the shoulders
# (it's nearer the camera), so the head and the body each get their own
# scale; measured on the Obama photo the face put stature 30% above what
# the shoulders did, the three face measures within 6% of each other.
_ANCHOR_OF_STATURE = {"biacromial": 0.23682, "bideltoid": 0.29085, "bizygomatic": 0.08131,
                      "menton_sellion": 0.0699, "interpupillary": 0.0365, "ankle_height": 0.0415}
_CLOTHES = 1.06  # ANSUR measured minimal clothing; the photo shows clothes


def _body_scales(pose: dict, face_mesh: Optional[np.ndarray], mask: np.ndarray) -> dict:
    """Stature in photo pixels for the body and for the head."""
    import statistics

    A = _ANCHOR_OF_STATURE
    body = []
    ls, rs = np.array(pose["left_shoulder"]), np.array(pose["right_shoulder"])
    body.append(float(np.linalg.norm(ls - rs)) / A["biacromial"])
    sy = int(round((ls[1] + rs[1]) / 2))
    rows = [r for r in range(max(sy, 0), min(sy + 40, mask.shape[0])) if mask[r].any()]
    if rows:
        w = max(float(np.ptp(np.flatnonzero(mask[r]))) for r in rows)
        body.append(w / 1.05 / A["bideltoid"])  # jacket or sleeves over the deltoids
    if "left_ankle" in pose and "right_ankle" in pose:
        top = int(np.flatnonzero(mask.any(axis=1))[0])
        full = (max(pose["left_ankle"][1], pose["right_ankle"][1]) - top) / (1.0 - A["ankle_height"])
        body = [full]  # the whole figure is in the photo: that IS the height
    s_body = float(statistics.median(body))
    s_head = s_body
    if face_mesh is not None:
        fm = face_mesh
        d = lambda a, b: float(np.linalg.norm(fm[a, :2] - fm[b, :2]))  # noqa: E731
        head = [d(234, 454) / A["bizygomatic"], d(152, 168) / A["menton_sellion"]]
        if len(fm) >= 474:
            head.append(d(468, 473) / A["interpupillary"])
        s_head = float(statistics.median(head))
    return {"body": s_body, "head": s_head}


def _arm_layout(pose: Optional[dict], grid_mask: np.ndarray, y0: int, x0: int, scale: float, pad: int,
                face_mesh: Optional[np.ndarray] = None, photo_mask: Optional[np.ndarray] = None) -> Optional[dict]:
    """Arm tubes and the torso band from pose points (photo pixels).
    Returns None when there aren't two shoulders to size things from."""
    if not pose or "left_shoulder" not in pose or "right_shoulder" not in pose:
        return None
    gh, gw = grid_mask.shape

    def g(name):
        x, y = pose[name][0], pose[name][1]
        return np.array([(y - y0) * scale + pad, (x - x0) * scale + pad], dtype=np.float64)

    ls, rs = g("left_shoulder"), g("right_shoulder")
    sw = float(np.linalg.norm(ls - rs))
    if sw < 8.0:
        return None
    rows, cols = np.mgrid[0:gh, 0:gw].astype(np.float64)
    scales = _body_scales(pose, face_mesh, photo_mask) if photo_mask is not None else None
    # arm radius (grid px) at shoulder / elbow / wrist / hand: from height
    # when known, else from the shoulder-joint distance as before
    if scales is not None:
        sb = scales["body"] * scale * _CLOTHES / 2.0
        r_of = {"shoulder": _DEPTH_OF_STATURE["upperarm"] * sb, "elbow": 0.5 * (_DEPTH_OF_STATURE["upperarm"] + _DEPTH_OF_STATURE["forearm"]) * sb,
                "wrist": _DEPTH_OF_STATURE["wrist"] * sb, "hand": _ARM_R["hand"] * sw * _CLOTHING}
    else:
        r_of = {k: _ARM_R[k] * sw * _CLOTHING for k in _ARM_R}

    def sleeve_half_width(a, b, r_guess):
        """Half the arm's visible width across its axis, from the outline
        where only one side runs into the body (a hanging arm). None where
        both sides do (an arm lying across the chest)."""
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 4:
            return None
        n = np.array([-ab[1], ab[0]]) / L
        got = []
        for t in np.linspace(0.25, 0.75, 9):
            c = a + t * ab
            dists = []
            for sgn in (1, -1):
                k = 0
                lim = int(3 * r_guess) + 2
                while k < lim:
                    q = c + sgn * (k + 1) * n
                    qi, qj = int(round(q[0])), int(round(q[1]))
                    if not (0 <= qi < gh and 0 <= qj < gw) or not grid_mask[qi, qj]:
                        break
                    k += 1
                dists.append(k if k < lim else None)
            seen = [x for x in dists if x is not None]
            if seen:
                got.append(min(seen))
        if len(got) < 4:
            return None
        return float(np.median(got))

    # torso band: shoulders down, centred between shoulders (and hips if seen)
    sy, scx = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    if "left_hip" in pose and "right_hip" in pose:
        lh, rh = g("left_hip"), g("right_hip")
        hy, hcx = (lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0
        torso_len = max(hy - sy, sw * 0.5)
    else:
        hcx, torso_len = scx, _torso_profile()["torso_len_over_shoulders"] * sw
    frac = np.clip((rows[:, 0] - sy) / torso_len, 0.0, 1.0)
    cx_row = scx + (hcx - scx) * frac
    hw_row = 0.5 * sw * _CLOTHING * (_TORSO_BREADTH[0] + (_TORSO_BREADTH[1] - _TORSO_BREADTH[0]) * frac)
    band = np.abs(cols - cx_row[:, None]) <= hw_row[:, None]
    band[rows[:, 0] < sy] = True  # head, neck and shoulder tops are all body

    best = np.full((gh, gw), -np.inf)
    r_px = np.zeros((gh, gw))
    d_px = np.zeros((gh, gw))
    w_on = np.zeros((gh, gw))
    for side in ("left", "right"):
        chain = [("shoulder", g(f"{side}_shoulder"))]
        for joint in ("elbow", "wrist"):
            if f"{side}_{joint}" not in pose:
                break
            chain.append((joint, g(f"{side}_{joint}")))
        if len(chain) == 3:
            tips = [g(f"{side}_{k}") for k in ("index", "pinky", "thumb") if f"{side}_{k}" in pose]
            if tips:
                chain.append(("hand", np.mean(tips, axis=0)))
        if len(chain) < 2:
            continue
        in_front = False
        if len(chain) >= 3:
            e, wr = chain[1][1], chain[2][1]
            pts = e + np.linspace(0, 1, 21)[:, None] * (wr - e)
            ri = np.clip(np.round(pts[:, 0]).astype(int), 0, gh - 1)
            ci = np.clip(np.round(pts[:, 1]).astype(int), 0, gw - 1)
            in_front = float(np.mean(band[ri, ci] & (ri > sy))) >= _ARM_IN_FRONT_FRAC
        # how much of the tube rests on the torso's front, per chain point
        on = {"shoulder": 0.0, "elbow": 0.5 if in_front else 0.0, "wrist": 1.0 if in_front else 0.0, "hand": 1.0 if in_front else 0.0}
        for (na, a), (nb, b) in zip(chain[:-1], chain[1:]):
            ra, rb = r_of[na], r_of[nb]
            if na in ("shoulder", "elbow") and nb in ("elbow", "wrist"):
                # An arm is close to round, so its depth should follow the
                # width it shows in the photo (a sleeve is wider than the
                # bare arm the height-based size describes) -- where that
                # width can be read and is plausible.
                hw = sleeve_half_width(a, b, 0.5 * (ra + rb))
                if hw is not None and 0.7 * 0.5 * (ra + rb) <= hw <= 2.0 * 0.5 * (ra + rb):
                    k = hw / (0.5 * (ra + rb))
                    ra, rb = ra * k, rb * k
            ab = b - a
            L2 = float(ab @ ab) or 1.0
            t = np.clip(((rows - a[0]) * ab[0] + (cols - a[1]) * ab[1]) / L2, 0.0, 1.0)
            d = np.hypot(rows - (a[0] + t * ab[0]), cols - (a[1] + t * ab[1]))
            r = ra + (rb - ra) * t
            inside = r - d
            upd = inside > best
            best = np.where(upd, inside, best)
            r_px = np.where(upd, r, r_px)
            d_px = np.where(upd, d, d_px)
            w_on = np.where(upd, on[na] + (on[nb] - on[na]) * t, w_on)
    if not np.isfinite(best).any():
        return None
    arm = grid_mask & (best > -0.25 * r_px)  # a little past the tube: sleeves are looser than the arm
    arm_only = arm & ~band
    return {"arm": arm, "arm_only": arm_only, "r": r_px / scale, "d": d_px / scale, "w": w_on,
            "shoulder_row": sy, "hip_row": sy + torso_len, "centre_col": cx_row, "scales": scales,
            "grid": (y0, x0, scale, pad)}


_TORSO_PROFILE = {}


def _torso_profile() -> dict:
    """Human torso cross-section shape (data/torso_profile.npz, measured
    from the CC0 MakeHuman base mesh -- see data/README.md)."""
    if not _TORSO_PROFILE:
        from pathlib import Path

        z = np.load(Path(__file__).with_name("data") / "torso_profile.npz")
        _TORSO_PROFILE.update({k: z[k] for k in z.files})
        _TORSO_PROFILE["torso_len_over_shoulders"] = float(z["torso_len_over_shoulders"])
    return _TORSO_PROFILE


# Template levels that are torso alone: above u=0.2 the base mesh's
# A-posed arms are still joined to the shoulders, below u=1.1 the legs part.
_TORSO_U = (0.2, 1.1)


def _scale_to_height(torso_half: np.ndarray, body_mask: np.ndarray, layout: dict, face_mesh: Optional[np.ndarray],
                     pose: dict) -> np.ndarray:
    """Rescale the body's half-thickness so its depth through the centre
    line at each height is a man's depth there for this person's height
    (_DEPTH_OF_STATURE), instead of following the outline's front width --
    which clothes, arms and pose all distort. The head is scaled by the
    head's own size, the body by the body's; each leg separately. The
    shape across each row is kept, only its depth changes."""
    from scipy.ndimage import gaussian_filter1d

    sc = layout.get("scales")
    if not sc:
        return torso_half
    y0, x0, scale, pad = layout["grid"]
    to_row = lambda y: (y - y0) * scale + pad  # noqa: E731
    D = _DEPTH_OF_STATURE
    sb, sh = sc["body"] * _CLOTHES, sc["head"] * 1.04  # hair over the head
    sy, hy = layout["shoulder_row"], layout["hip_row"]
    anchors = []  # (grid row, centre-line depth in photo px)
    if face_mesh is not None:
        eye = to_row((face_mesh[33, 1] + face_mesh[263, 1]) / 2.0)
        chin = to_row(float(face_mesh[:468, 1].max()))
        anchors += [(eye, D["head"] * sh), (chin + 0.45 * (sy - chin), D["neck"] * np.sqrt(sh * sb))]
    anchors += [(sy, 0.9 * D["chest"] * sb), (sy + 0.25 * (hy - sy), D["chest"] * sb),
                (sy + 0.65 * (hy - sy), D["waist"] * sb), (hy, D["buttock"] * sb)]
    anchors.sort()
    ar = np.array([a[0] for a in anchors])
    ad = np.array([a[1] for a in anchors])
    gh, gw = body_mask.shape
    out = torso_half.copy()
    g_rows, g_vals = [], []
    for r in range(gh):
        if r > hy:
            break
        row = body_mask[r]
        on = np.flatnonzero(row)
        if not len(on):
            continue
        c = int(round(layout["centre_col"][r]))
        c = c if 0 <= c < gw and row[c] else int(on[np.argmin(np.abs(on - c))])
        cur = 2.0 * float(torso_half[r, c])
        if cur <= 0:
            continue
        g_rows.append(r)
        # above the top anchor (eye level) the head narrows to the crown:
        # scale it by the eye-level factor so it keeps its own rounding
        g_vals.append(np.nan if r < ar[0] else float(np.interp(r, ar, ad)) / cur)
    if g_rows:
        gv = np.array(g_vals)
        known = np.flatnonzero(np.isfinite(gv))
        if len(known):
            gv[: known[0]] = gv[known[0]]
            g = np.clip(gaussian_filter1d(gv, max(3.0, 0.02 * gh), mode="nearest"), 0.4, 2.5)
            for r, k in zip(g_rows, g):
                out[r] *= k
    # legs: each leg's own run, below the hips
    kn = [pose[k][1] for k in ("left_knee", "right_knee") if k in pose]
    an = [pose[k][1] for k in ("left_ankle", "right_ankle") if k in pose]
    if kn:
        kr = to_row(float(np.mean(kn)))
        ak = to_row(float(np.mean(an))) if an else kr + (kr - hy)
        lr = np.array([hy, hy + 0.2 * (kr - hy), kr, kr + 0.3 * (ak - kr), ak])
        ld = np.array([D["buttock"] * sb * 0.8, D["thigh"] * sb, D["knee"] * sb, D["calf"] * sb, D["ankle"] * sb])
        gmap = np.zeros((gh, gw))
        wmap = np.zeros((gh, gw), dtype=bool)
        for r in range(int(np.ceil(hy)) + 1, gh):
            row = body_mask[r]
            edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
            for a, b in zip(edges[0::2], edges[1::2]):
                cc = (a + b - 1) // 2
                cur = 2.0 * float(torso_half[r, cc])
                if cur > 0:
                    gmap[r, a:b] = np.clip(float(np.interp(r, lr, ld)) / cur, 0.4, 2.5)
                    wmap[r, a:b] = True
        if wmap.any():
            gs = _masked_blur(gmap, wmap, max(3.0, 0.015 * gh))
            out = np.where(wmap, torso_half * gs, out)
    return out


def _human_torso(torso_half: np.ndarray, body_mask: np.ndarray, layout: dict):
    """Torso front and back surfaces with a human cross-section instead of a
    mirror-image lens: at each height the depth through the torso's centre
    line (from the outline and the ANSUR ratios) is kept, and the front and
    back take the template's shape across the width -- chest standing ~70%
    in front of the side seams with a flat upper back, a balanced waist,
    the seat reaching behind. Between the shoulder joints and the neck, and
    below the hips, it fades back to the plain lens."""
    prof = _torso_profile()
    U, S, F, B = prof["u"], prof["s"], prof["front"], prof["back"]
    ok = (U >= _TORSO_U[0]) & (U <= _TORSO_U[1])
    U, F, B = U[ok], F[ok], B[ok]
    # fill the template's few unmeasured edge samples from their neighbours,
    # then smooth along the height: its level-to-level variation (a few
    # percent) showed as horizontal bands in the shading of the back
    from scipy.ndimage import gaussian_filter1d

    F, B = F.copy(), B.copy()
    for A in (F, B):
        for i in range(len(A)):
            good = np.isfinite(A[i])
            A[i] = np.interp(S, S[good], A[i][good])
    F = gaussian_filter1d(F, 1.5, axis=0, mode="nearest")
    B = gaussian_filter1d(B, 1.5, axis=0, mode="nearest")
    front = torso_half.copy()
    back = -torso_half
    sy, hy = layout["shoulder_row"], layout["hip_row"]
    gh, gw = body_mask.shape
    # Measure each row's torso run first, then smooth centre, half-width and
    # depth down the body: measured row by row they jitter, and the jitter
    # showed as horizontal bands across the back.
    rows, us, wts, lefts, rights, ccs, hws, d0s = [], [], [], [], [], [], [], []
    for r in range(gh):
        u = (r - sy) / max(hy - sy, 1.0)
        wt = float(np.clip(u / _TORSO_U[0], 0.0, 1.0) * np.clip((1.25 - u) / (1.25 - _TORSO_U[1]), 0.0, 1.0))
        if wt <= 0.0:
            continue
        row = body_mask[r]
        on = np.flatnonzero(row)
        if not len(on):
            continue
        c = int(round(layout["centre_col"][r]))
        c = c if 0 <= c < gw and row[c] else int(on[np.argmin(np.abs(on - c))])
        left, right = c, c
        while left > 0 and row[left - 1]:
            left -= 1
        while right < gw - 1 and row[right + 1]:
            right += 1
        hw = (right - left + 1) / 2.0
        cc = (left + right) / 2.0
        d0 = 2.0 * float(torso_half[r, int(round(cc))])
        if hw < 2 or d0 <= 0:
            continue
        rows.append(r); us.append(u); wts.append(wt); lefts.append(left); rights.append(right)
        ccs.append(cc); hws.append(hw); d0s.append(d0)
    if not rows:
        return front, back
    sig = max(3.0, 0.025 * gh)
    ccs, hws, d0s = (gaussian_filter1d(np.array(v, dtype=np.float64), sig, mode="nearest") for v in (ccs, hws, d0s))
    for r, u, wt, left, right, cc, hw, d0 in zip(rows, us, wts, lefts, rights, ccs, hws, d0s):
        uu = float(np.clip(u, U[0], U[-1]))
        f_s = np.array([np.interp(uu, U, F[:, j]) for j in range(len(S))])
        b_s = np.array([np.interp(uu, U, B[:, j]) for j in range(len(S))])
        cols = np.arange(left, right + 1)
        sv = np.clip((cols - cc) / hw, -1.0, 1.0)
        front[r, cols] = wt * d0 * np.interp(sv, S, f_s) + (1 - wt) * torso_half[r, cols]
        back[r, cols] = wt * d0 * np.interp(sv, S, b_s) - (1 - wt) * torso_half[r, cols]
    front = np.where(body_mask, _masked_blur(front, body_mask, 1.5), 0.0)
    back = np.where(body_mask, _masked_blur(back, body_mask, 1.5), 0.0)
    return front, np.minimum(back, front)


def _with_arms(layout: dict, grid_mask: np.ndarray, torso_front: np.ndarray, torso_back: np.ndarray, arm_ratio: float, dist_out: np.ndarray):
    """Front and back surfaces (photo px, z toward camera) of the torso plus
    the arm tubes. Arms crossed in front rest on the torso's front surface;
    arms at the sides stay centred. The back is the torso's own back.
    `dist_out`: distance to the photo outline's edge (photo px)."""
    torso = grid_mask & ~layout["arm_only"]
    tf = np.where(torso, torso_front, 0.0)
    r, d, w = layout["r"], layout["d"], layout["w"]
    # Depth into the tube from its nearest edge. Where the arm is outside
    # the body, its outer edge is the photo's own outline (a sleeve is often
    # wider than the bare arm the tube is sized from -- using the tube's
    # edge there left a paper-thin fin along the sleeve); over the body the
    # tube's own edge applies too.
    e = np.where(layout["arm_only"], np.minimum(dist_out, r), np.minimum(np.minimum(dist_out, r - d), r))
    e = np.clip(e, 0.0, None)
    p = arm_ratio * np.sqrt(np.clip(e * (2.0 * r - e), 0.0, None))
    zc = w * (tf + arm_ratio * r)
    af, ab = zc + p, zc - p
    arm = layout["arm"]
    front = np.where(torso, tf, -np.inf)
    back = np.where(torso, torso_back, np.inf)
    front = np.where(arm, np.maximum(front, af), front)
    back = np.where(arm, np.minimum(back, ab), back)
    front = np.where(grid_mask, front, 0.0)
    back = np.where(grid_mask, back, 0.0)
    # soften the step where a tube meets the body without moving the outline
    front = _masked_blur(front, grid_mask, 1.5)
    back = _masked_blur(back, grid_mask, 1.5)
    return front, np.minimum(back, front)


# A human body posed to the photo (body_reference.py) knows what the outline
# can't: where the head sits over the torso, the back of the skull and the
# nape, the curve of the back, forearms resting on the belly. Its front and
# back become the base surfaces; the photo still gives the outline, the fine
# relief and the face. Each part is grown by its own measured gap to the
# outline first (suit over the torso and arms, hair over the head).
_REF_CLOTH_RANGE = (-0.025, 0.05)  # most a part shrinks (slimmer than the reference) / grows (clothes, hair), metres
# A photo confirms widths, never depths: where depth can't be confirmed the
# model errs bigger, never smaller. Clothing adds its full side gap front and
# back too, a part narrower than the body model shrinks only across (its
# depth stays), and every front-to-back thickness gets a margin on top.
_REF_CLOTH_DEPTH = 1.0
_REF_DEPTH_MARGIN = 1.10
_REF_HEAD_MARGIN = 1.05
_REF_SIDE_SMOOTH = 30  # extra smoothing passes on surfaces facing sideways
_REF_EDGE_SLACK = 0.015  # the body's own edge may pass the photo outline by this much, metres
_REF_HAND_SHRINK = 0.3  # hands shrunk to this share about the wrist
# a forearm (and its hand) lying across the body is a solid of its own,
# joined on top: the body's own front runs on under it (one front and one
# back per point had merged arm and chest into a single slab)
_LIMB_SOLIDS = 1  # 1: on, 0: merged into the body as before (a study can try both)
_LIMB_DETAIL = 1.0  # share of the photo's detail (sleeve folds) carried onto a crossed forearm
_LIMB_CLOTH_DETAIL = 1.0
_HAND_THICK = 0.035  # a hand lying on the body stands this thick (a loose fist; flat hand ~0.03), metres
_HAND_SINK = 0.002  # ... sunk this far into what it rests on, so the two join
_REF_HAIR = 0.012  # most the head grows front to back (hair), metres
# How much body to build (see _person_extent): a photo reaching this share
# of the way from the hip joints to the knees counts as showing the legs.
_KNEE_REACH = 0.9
_REF_BARE_PARTS = ("hand.L", "hand.R")
_REF_RELIEF_MAX = 0.015  # photo-depth detail on top of the reference, metres
_HAIR_SHELL_MAX = 0.05  # thickest hair (or hat) lying round the skull, metres (more: a mass of its own)
_HAIR_FEATHER = 0.006  # where hair over the skin ends: feathered over this, metres
_HAIR_SMOOTH = 0.004  # hair and hat surfaces smoothed across their edges by this, metres
_HAIR_THIN = 0.012  # hair thinner than this is the head's own outline (the skull grows to it), metres
_HAIR_SHORT = 0.004  # ... but stands at least this proud of the scalp in front, its hairline a step (metres)
_HAIR_FACE_MEET = 0.012  # hair this near the face comes down to the cheek's level, metres
_HAIR_NECK_BEHIND = 0.008  # hair beside the neck lies this far behind its front, metres
_HAIR_NAPE_EDGE = 0.004  # the hairline at the nape: hair ends over this, metres
_HAIR_BACK_FROM_SIDES = 1.0  # round the back, this share of how far the hair reaches past the skull at the sides
_NECK_COLLAR = 0.009  # a collar (with its fold) stands this far off the neck, metres
_NECK_LACE = 0.003  # a necklace stands this proud, metres
_NECK_WEAR_MAX = 0.025  # a scarf or roll neck stands at most this far off the neck, metres
_NECK_COLLAR_BELOW = 0.02  # ... and reaches this far below the neck's base, metres
_NECK_COLLAR_RISE = 0.012  # round the back it rides this much higher than at the sides, metres
_NECK_DEBUG = None
_HAIR_DEBUG = None  # a dict to fill with _hair_and_hat's working, when looking into it
_HAIR_FRINGE = 0.012  # a fringe over the forehead, at most, metres
_HAIR_LAYER = 0.008  # long hair lying over the shoulders or chest, metres
_HAIR_HANG = 0.012  # long hair hanging beside the neck: at least this half-thick, metres
_HAT_PEAK = 0.3  # a cap's peak reaches this far forward, x the face's width
_MASK_SPAN = 0.04  # a face mask spans hollows narrower than a ball this size, metres
_GLASSES_SPAN = 0.02  # glasses' lenses: the same over the eye sockets
_GLASSES_FRAME = 0.002  # their frame stands this proud, metres
_REF_RELIEF_FACING = (0.35, 0.7)  # ... fading in as the surface turns from side-on to facing the camera
_REF_BACK_SMOOTH = 0.02  # clothing over the back: smoothing radius, metres
# A shirt or jacket over a man's chest spans the hollow between the
# pectorals and the crease under them: the torso's front is lifted onto a
# ball this size (metres) rolled over it. (Bare, the body model's chest read
# as breasts under a suit.)
_REF_DRAPE = 0.08
_REF_HANG = 0.15  # ... and hangs from it: falling back at most this much per unit down
# Ears the photo shows are built in 3D (see _ear_solid): the ear itself
# (weight >= _REF_EAR_ROOT; the head is sampled with smooth skin across its
# root) and a skirt of skin round it (>= _REF_EAR_BASE) sunk up to
# _REF_EAR_SINK (metres) under the head's surface, with a plug
# _REF_EAR_PLUG into the head, never thinner than _REF_EAR_MIN_MM on the
# print (a real ear is about _REF_EAR_THICK thick)
_REF_EAR_HAIR = 0.015  # an ear whose base grew out further than this (metres) is under hair: not built
_REF_EAR_ROOT = 0.95
_REF_EAR_BASE = 0.8
_REF_EAR_SINK = 0.005
_REF_EAR_PLUG = 0.02
_REF_EAR_MIN_MM = 1.2
_REF_EAR_THICK = 0.004
# The hair's pattern, read from the photo (hair_pattern.py) and pressed into
# the hair's surfaces (_hair_pattern), all in metres on the person:
PRINT_GROOVE_DEPTH_MM = 0.3  # a groove the figure is to show: at least this deep on the print (mm) ...
PRINT_GROOVE_W_MM = 0.5  # ... and this wide (a printer's nozzle, a resin print's own blur)
_HAIR_STRAND_W = 0.005  # a sculpted strand (a lock of strands) is this wide
_HAIR_STRAND_LEN = 0.05  # ... and runs on this long before it fades into the next
_HAIR_STRAND_DEPTH = 0.0012  # grooves between strands: this deep either way
_HAIR_WAVE_DEPTH = 0.003  # a wave's crest stands this proud
_HAIR_WAVE_LEN = 0.12  # a wave's length when the photo didn't measure one
_HAIR_CURL = 0.012  # a curl is this across ...
_HAIR_CURL_DEPTH = 0.0025  # ... and stands this proud
_HAIR_COIL = 0.006  # tight coils: this across ...
_HAIR_COIL_DEPTH = 0.0015  # ... this proud
_HAIR_ROPE_W = 0.015  # braids, locs: a rope is this wide when the photo didn't measure it ...
_HAIR_ROPE_DEPTH = 0.003  # ... and the grooves between ropes this deep either way
_HAIRLINE_EDGE = 0.004  # the pattern ends over this at the hairline
_PARTING_DEPTH = 0.0015  # a parting's groove
# Each part of the face, read from the photo (face_parts.py) and shaped by
# _face_parts_relief, metres on the person:
_EYE_INSET = 0.0025  # an open eye's ball sits this far behind its lids, rounding up to them (shown up for a figure)
_IRIS_DEPTH = 0.0015  # the iris, carved this deep (a sculptor's eye; 0.4 mm read as nothing on a figure)
_LID_CREASE = 0.0012  # the crease above the upper lid
_LID_LINE = 0.001  # a closed eye's lid line
_BROW_HEIGHT = 0.003  # eyebrow hair stands this proud (shown up: ~1 mm on a person reads as nothing on a figure)
_BROW_GRAIN = 0.35  # ... with a hair grain this deep (share of its height)
_BROW_GRAIN_STEP = 0.0015  # ... this far apart (m)
_BROW_GRAIN_ANGLE = 25.0  # ... the hairs running this many degrees up off the brow's line
_NOSTRIL_DEPTH = 0.003  # a nostril's hollow
_LIP_RIM = 0.0004  # the lips' edge stands this proud
_LIP_FULL = 0.0  # the lips' own fullness, a dome over their outline (a study sets it)
_LID_RIM = 0.0012  # the upper eyelid's rim over the eyeball (none, the eye's opening had no edge on a figure)
_CHIN_PROUD = 0.0  # the chin's pad, with half as deep a groove above it (a study sets it)
_MOUTH_LINE = 0.0008  # a closed mouth's line between the lips
_MOUTH_DEPTH = 0.008  # an open mouth's dark, this far in
_TEETH_INSET = 0.002  # teeth sit this far behind the lips
_TOOTH_LINE = 0.0006  # the lines between teeth this deep
_TONGUE_INSET = 0.004  # the tongue this far
_SCAR_DEPTH = 0.0005  # a scar's groove
_BEARD_THICK = 0.004  # a beard or moustache stands this proud
_STUBBLE_DEPTH = 0.0003  # stubble's grain
# Clothes (clothes.py): where one garment meets another, the outer one
# stands a step over the inner (layers in clothes.LAYER), metres
_CLOTH_LAYER = 0.0025  # one layer's step
_CLOTH_STEP_SPAN = 0.015  # the step eases back to the garment's own surface over this
_TIE_THICK = 0.002  # a tie stands this proud of the shirt (on top of its layer step)
_BELT_THICK = 0.002
_WATCH_THICK = 0.006  # a watch's case and strap stand this proud of the wrist (a real case is ~10 mm deep)
_BRACELET_THICK = 0.003
# Bones that show under bare skin (_bony_landmarks), metres on the person:
_CLAVICLE = 0.003  # the collarbones' ridge
_STERNAL_NOTCH = 0.003  # the hollow between them at the neck's base
_PATELLA = 0.004  # a kneecap
_MALLEOLUS = 0.003  # the ankle's knobs either side
_KNUCKLE = 0.0015  # the knuckles
_C7 = 0.003  # the knob at the base of the nape
_EARRING_PROUD = 0.004  # an earring stands this proud
# how much of each a kind of hair gets: strands, waves, curls, coils
# (strands, waves, curls, coils, ropes)
_HAIR_KIND_MIX = {"sleek": (0.35, 0, 0, 0, 0), "straight": (1, 0, 0, 0, 0), "wavy": (0.8, 1, 0, 0, 0),
                  "curly": (0.25, 0, 1, 0, 0), "coily": (0, 0, 0, 1, 0), "corded": (0.3, 0, 0, 0, 1)}


def _raster_depth(gx, gy, depth, faces, shape):
    """Nearest and farthest surface depth at each grid cell's centre (cells
    no triangle covers stay +/-inf). Sampling at the centre, not over the
    cell, keeps a round edge's thickness falling smoothly to zero inside
    the cell it ends in."""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    xa, xb, xc, ya, yb, yc = gx[a], gx[b], gx[c], gy[a], gy[b], gy[c]
    cx0 = np.ceil(np.minimum(np.minimum(xa, xb), xc)).astype(np.int64)
    cx1 = np.floor(np.maximum(np.maximum(xa, xb), xc)).astype(np.int64)
    cy0 = np.ceil(np.minimum(np.minimum(ya, yb), yc)).astype(np.int64)
    cy1 = np.floor(np.maximum(np.maximum(ya, yb), yc)).astype(np.int64)
    den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
    k = np.maximum(cx1 - cx0, cy1 - cy0) + 1
    use = (cx1 >= cx0) & (cy1 >= cy0) & (np.abs(den) > 1e-12)
    near = np.full(shape, np.inf)
    far = np.full(shape, -np.inf)
    for kk in np.unique(k[use]):
        sel = np.flatnonzero(use & (k == kk))
        oy, ox = np.divmod(np.arange(kk * kk), kk)
        for part in np.array_split(sel, max(1, len(sel) * kk * kk // 4_000_000 + 1)):
            px = cx0[part, None] + ox[None, :]
            py = cy0[part, None] + oy[None, :]
            l1 = ((yb - yc)[part, None] * (px - xc[part, None]) + (xc - xb)[part, None] * (py - yc[part, None])) / den[part, None]
            l2 = ((yc - ya)[part, None] * (px - xc[part, None]) + (xa - xc)[part, None] * (py - yc[part, None])) / den[part, None]
            l3 = 1.0 - l1 - l2
            ok = ((l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9) & (px <= cx1[part, None]) & (py <= cy1[part, None])
                  & (px >= 0) & (px < shape[1]) & (py >= 0) & (py < shape[0]))
            d = l1 * depth[a[part], None] + l2 * depth[b[part], None] + l3 * depth[c[part], None]
            np.minimum.at(near, (py[ok], px[ok]), d[ok])
            np.maximum.at(far, (py[ok], px[ok]), d[ok])
    return near, far


def _vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for j in range(3):
        np.add.at(vn, F[:, j], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def _mesh_laplacian(F, n):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    adj = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.float64)
    return sparse.diags(np.asarray(adj.sum(axis=1)).ravel()) - adj


def _mesh_smooth(lap, x, iterations):
    """Average each vertex's value with its neighbours', `iterations` times."""
    deg = np.asarray(lap.diagonal()).ravel()
    adj = sparse.diags(deg) - lap
    inv = 1.0 / np.maximum(deg, 1.0)
    for _ in range(iterations):
        x = 0.5 * x + 0.5 * inv * (adj @ x)
    return x


def _harmonic_fill(lap, known, values):
    """Values at `known` vertices spread smoothly over the rest of the mesh."""
    n = lap.shape[0]
    x = np.zeros(n)
    x[known] = values
    unk = np.setdiff1d(np.arange(n), known)
    if len(unk) and len(known):
        L = lap.tocsr()
        sub = (L[unk][:, unk] + sparse.identity(len(unk)) * 1e-6).tocsc()
        x[unk] = spsolve(sub, -(L[unk][:, known] @ x[known]))
    return x


def _drape_chest(V: np.ndarray, F: np.ndarray, torso_w: np.ndarray, px_per_m: float, joints: dict,
                 frontmost: bool = False, drape: float = _REF_DRAPE) -> np.ndarray:
    """Cloth over a man's chest (photo px, depth toward the back): the front
    of the chest lifted toward the camera onto a ball of _REF_DRAPE rolled
    over it, and hanging from the chest down instead of tucking in under it.
    Only the chest between the arms, collarbones to lower ribs: past it the
    arms, armpits and neck keep their own shape."""
    from scipy.ndimage import grey_closing

    t = torso_w > 0.5
    tf = F[t[F].all(axis=1)]
    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if len(tf) < 50 or not all(k in joints for k in need):
        return V
    cell = 0.01 * px_per_m  # 1 cm cells
    ox, oy = V[t, 0].min() - 2 * cell, V[t, 1].min() - 2 * cell
    gx, gy = (V[:, 0] - ox) / cell, (V[:, 1] - oy) / cell
    shape = (int(np.ceil(gy[t].max())) + 3, int(np.ceil(gx[t].max())) + 3)
    near, _ = _raster_depth(gx, gy, V[:, 2], tf, shape)
    have = np.isfinite(near)
    if have.sum() < 20:
        return V
    rise = np.where(have, -near, 0.0)  # toward the camera
    n = int(round(drape * 100))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    ball = (yy**2 + xx**2) <= n * n
    R = drape * px_per_m
    cap = np.where(ball, np.sqrt(np.maximum(R * R - (yy**2 + xx**2) * cell * cell, 0.0)) - R, 0.0)
    floor = float(rise[have].min()) - 4.0 * R
    cloth = grey_closing(np.where(have, rise, floor), footprint=ball, structure=cap)
    # hanging: going down, falls back toward the body by at most _REF_HANG per cell
    for r in range(1, shape[0]):
        cloth[r] = np.where(have[r] & have[r - 1], np.maximum(cloth[r], cloth[r - 1] - _REF_HANG * cell), cloth[r])
    # the chest only
    J = {k: np.asarray(joints[k], dtype=np.float64) for k in need}
    sh = (J["left_shoulder"] + J["right_shoulder"]) / 2.0
    hp = (J["left_hip"] + J["right_hip"]) / 2.0
    sw = float(np.linalg.norm(J["left_shoulder"][:2] - J["right_shoulder"][:2]))
    tl = float(hp[1] - sh[1])
    if sw < 4 * cell or tl < 4 * cell:
        return V
    cy, cx = np.mgrid[0:shape[0], 0:shape[1]]
    px, py = cx * cell + ox, cy * cell + oy

    def band(v, a, b, soft):
        return np.clip((v - a) / soft + 0.5, 0.0, 1.0) * np.clip((b - v) / soft + 0.5, 0.0, 1.0)

    zone = band(px, sh[0] - 0.42 * sw, sh[0] + 0.42 * sw, 0.15 * sw) * band(py, sh[1] + 0.05 * tl, sh[1] + 0.6 * tl, 0.15 * tl)
    lift = np.where(have, np.maximum(cloth - rise, 0.0) * zone, 0.0)
    lift = _masked_blur(lift, have, 1.0)
    at = map_coordinates(lift, [gy, gx], order=1, mode="constant", cval=0.0)
    if frontmost:
        # what the front view sees, facing the camera or not: the underside
        # of a bust faces down, and is just what the cloth hangs over
        near_v = map_coordinates(np.where(have, near, 1e9), [gy, gx], order=0, mode="nearest")
        facing = (V[:, 2] <= near_v + cell).astype(np.float64)
    else:
        facing = np.clip((-_vertex_normals(V, F)[:, 2] - 0.2) / 0.3, 0.0, 1.0)  # front-facing (depth points back)
    V = V.copy()
    V[:, 2] -= at * facing * np.clip((torso_w - 0.5) / 0.3, 0.0, 1.0)
    return V


def _person_extent(mask: np.ndarray, hip_y: Optional[float], knee_y: Optional[float], bust_mode: bool):
    """How much of a person to build: never legs the photo doesn't show. A
    photo that ends between the hips and the knees is cut back to the hip
    line -- a torso, nothing half-shown below it; one that ends above the
    hips, or reaches the knees and below, is built exactly as far as it
    goes. A bust by request is cut at the hip line too. Returns the outline
    and "torso" (ends at or above the hip line) or "photo" (goes past the
    knees)."""
    bottom = int(np.flatnonzero(mask.any(axis=1))[-1])
    if hip_y is None or bottom <= hip_y:
        return mask, "torso"
    if knee_y is not None and not bust_mode and bottom >= hip_y + _KNEE_REACH * (knee_y - hip_y):
        return mask, "photo"
    out = mask.copy()
    out[int(round(hip_y)) + 1:] = False
    return out, "torso"


def _grow_to_outline(V: np.ndarray, F: np.ndarray, reference: dict, part_w: np.ndarray, grid_mask: np.ndarray,
                     grid_xy, shape, d_in: np.ndarray, d_out: np.ndarray, scale: float, photo_mask: Optional[np.ndarray],
                     normals: Optional[np.ndarray] = None, keep: Optional[np.ndarray] = None):
    """The posed body grown along its normals to the photo's outline (the
    clothes' and hair's room round it): (vertices, part names, how far each
    ear's base moved)."""
    from scipy.ndimage import binary_closing, binary_dilation, binary_erosion

    # Each rim vertex (on the body's outline as the camera sees it)
    # measures its own gap to the photo's outline; the gaps spread
    # smoothly over the whole surface, hands stay bare, and the body
    # grows along its normals. Repeated so the grown rim is re-measured.
    names = [str(n) for n in reference["part_names"]]
    lo, hi = (v * float(reference["px_per_m"]) for v in _REF_CLOTH_RANGE)
    hair = _REF_HAIR * float(reference["px_per_m"])
    V0, nrm0 = V.copy(), (_vertex_normals(V, F) if normals is None else normals)
    bare = np.zeros(len(V), dtype=bool)
    for name in _REF_BARE_PARTS:
        if name in names:
            bare |= part_w[:, names.index(name)] > 0.5
    if keep is not None:
        bare |= keep  # (held where it is: the face, in the body-first build)
    head = part_w[:, names.index("head")] if "head" in names else np.zeros(len(V))
    depth_share = 1.0 - (1.0 - _REF_CLOTH_DEPTH) * (1.0 - head)
    # the ears don't measure the outline (they're placed to it after),
    # they ride along with the skin they sit on
    ear_w = reference.get("ear_w")
    ear_w = np.zeros((len(V), 0)) if ear_w is None else np.asarray(ear_w, dtype=np.float64)
    on_ear = ear_w.max(axis=1) > 0.05 if ear_w.shape[1] else np.zeros(len(V), dtype=bool)
    ear_grew = np.zeros(ear_w.shape[1])
    # the skull doesn't grow into hair or a hat: those get their own
    # volume (_hair_volume); its rim under them doesn't measure the outline
    under_hair = np.zeros(len(V), dtype=bool)
    parts = reference.get("parts")
    if parts is not None and parts.get("labels") is not None:
        from .head_parts import HAIR as _HAIR, HAT as _HAT

        lab = parts["labels"]
        ry = np.clip(np.rint(V[:, 1]).astype(int), 0, lab.shape[0] - 1)
        rx = np.clip(np.rint(V[:, 0]).astype(int), 0, lab.shape[1] - 1)
        under_hair = (head > 0.5) & np.isin(lab[ry, rx], (_HAIR, _HAT))
    lap = _mesh_laplacian(F, len(V))
    # distance to the photo's outline (photo px, + inside), measured at the
    # photo's own resolution and smoothed: the grid's pixel steps, read
    # straight into each vertex's growth, made ridges along every side
    signed_px = None
    if photo_mask is not None:
        signed_px = gaussian_filter((distance_transform_edt(photo_mask) - distance_transform_edt(~photo_mask)).astype(np.float32), 1.5)
    total = np.zeros(len(V))
    for _ in range(3):
        gx, gy = grid_xy(V)
        near, _ = _raster_depth(gx, gy, V[:, 2], F, shape)
        cov = binary_closing(np.isfinite(near), iterations=1)
        rim = binary_dilation(cov & ~binary_erosion(cov), iterations=1)
        r = np.clip(np.rint(gy).astype(int), 0, shape[0] - 1)
        c = np.clip(np.rint(gx).astype(int), 0, shape[1] - 1)
        inb = (gy >= 0) & (gy < shape[0]) & (gx >= 0) & (gx < shape[1])
        if signed_px is not None:
            gap = map_coordinates(signed_px, [V[:, 1], V[:, 0]], order=1, mode="nearest")
        else:
            gap = np.where(grid_mask[r, c], d_in[r, c], -d_out[r, c]) / scale
        # (through thin hair it still grows: a short cut is the head's own outline)
        thick_hair = under_hair & (gap > _HAIR_THIN * float(reference["px_per_m"]))
        on_rim = inb & rim[r, c] & (np.abs(_vertex_normals(V, F)[:, 2]) < 0.35) & ~bare & ~on_ear & ~thick_hair
        known = np.flatnonzero(on_rim | bare)
        want = np.where(bare[known], 0.0, np.clip(total[known] + gap[known], lo, hi))
        total = _harmonic_fill(lap, known, want)
        # smooth it over the surface: what's left of the outline's pixel
        # steps in the measured gaps showed as ridges along every side
        total = _mesh_smooth(lap, total, 12)
        step = nrm0 * total[:, None]
        # depth: never shrinks; on the head it grows by hair at most (the
        # outline's ears and hair say nothing about the skull's length)
        grow_z = np.minimum(np.maximum(total, 0.0), head * hair + (1.0 - head) * hi)
        step[:, 2] = nrm0[:, 2] * grow_z * depth_share
        for i in range(ear_w.shape[1]):
            w = ear_w[:, i]
            base = (w > 0.02) & (w < 0.5)
            if base.any():
                ear_grew[i] = float(np.linalg.norm(step[base].mean(axis=0)))
                step = step + w[:, None] * (step[base].mean(axis=0) - step)
        V = V0 + step
    return V, names, ear_grew


def _reference_surfaces(reference: dict, grid_mask: np.ndarray, y0: int, x0: int, scale: float, pad: int,
                        face_mesh: Optional[np.ndarray], photo_mask: Optional[np.ndarray] = None):
    """Front and back (photo px, z toward camera) of the posed reference body
    on the build grid, grown to the outline, and the cells it covers."""
    from scipy.ndimage import binary_closing, binary_dilation, binary_erosion

    V = np.asarray(reference["vertices"], dtype=np.float64).copy()  # x right, y down, depth toward the back
    F = np.asarray(reference["faces"], dtype=np.int64)
    shape = grid_mask.shape
    d_in = distance_transform_edt(grid_mask)
    d_out = distance_transform_edt(~grid_mask)

    def grid_xy(P):
        return (P[:, 0] - x0) * scale + pad, (P[:, 1] - y0) * scale + pad

    part_w = reference.get("part_w")
    if part_w is not None:
        V, names, ear_grew = _grow_to_outline(V, F, reference, part_w, grid_mask, grid_xy, shape, d_in, d_out, scale, photo_mask)

    # Depth margin, in 3D: every point's front-to-back offset from its own
    # bones grows (torso about the spine, each arm about its bones, the head
    # about its joint -- the head less: its length follows from its width,
    # which the photo shows). Widths, which the photo confirms, don't move.
    core = reference.get("core_depth")
    if core is not None and part_w is not None:
        hw = part_w[:, names.index("head")] if "head" in names else 0.0
        m = hw * _REF_HEAD_MARGIN + (1.0 - hw) * _REF_DEPTH_MARGIN
        core = np.asarray(core, dtype=np.float64)
        V[:, 2] = core + m * (V[:, 2] - core)
    report = reference.get("report") or {}
    if part_w is not None and report.get("gender") == "male" and "torso" in names:
        V = _drape_chest(V, F, part_w[:, names.index("torso")], float(reference["px_per_m"]), reference.get("joints_px") or {})
    # the ears as the photo shows them: where they start and end, how far out
    if reference.get("ears") and reference.get("ear_w") is not None and reference.get("head_axes") is not None:
        from .body_reference import place_ears

        V = place_ears(V, np.asarray(reference["ear_w"], dtype=np.float64), reference["head_axes"], reference["ears"],
                       faces=F, px_per_m=float(reference["px_per_m"]))

    # Its hands are open with the fingers spread -- rarely how a photo shows
    # them -- and stood out as shelves and fins: shrink them into the wrist
    # and leave the hands to the outline. (Cutting them off left the forearm
    # open at the wrist, and sampling saw into it: a crack.)
    J = reference.get("joints_px")
    if part_w is not None and J is not None:
        for name, kp in (("hand.L", "left_wrist"), ("hand.R", "right_wrist")):
            if name in names and kp in J:
                w = part_w[:, names.index(name)][:, None]
                c = np.asarray(J[kp], dtype=np.float64)[None]
                V = V + w * ((c + _REF_HAND_SHRINK * (V - c)) - V)
    # the body mesh is coarse (13.7k vertices): smooth it before sampling so
    # its triangles don't show as flat facets
    from trimesh.remesh import subdivide_loop

    # A real ear stands off the head, cupped in front and open behind it:
    # one front and one back surface per point can't hold that (they filled
    # it in solid, a rounded lump on the side of the head). Where the photo
    # shows the ears, each is kept as a 3D piece -- the ear and the skin at
    # its base -- joined to the finished head (_ear_solid); the head itself
    # is sampled with smooth skin spanning the ear's base.
    # Not seen in the photo, an ear is still built -- the body model's own,
    # proportioned to its head -- unless its base had to grow out to the
    # outline (hair over it: the ear stays under the hair).
    ears3d = None
    EW = reference.get("ear_w")
    build3d = []
    if EW is not None and reference.get("head_axes") is not None and part_w is not None:
        EW = np.asarray(EW, dtype=np.float64)
        seen = reference.get("ears") or {}
        build3d = [(side in seen and not seen[side].get("hidden"))
                   or (not seen and ear_grew[i] <= _REF_EAR_HAIR * float(reference["px_per_m"]))
                   for i, side in enumerate("LR")]
    if any(build3d):
        lap_e = _mesh_laplacian(F, len(V))
        flat = V.copy()
        for i in range(EW.shape[1]):
            if not build3d[i]:
                continue
            pf = (EW[F, i] >= _REF_EAR_ROOT).all(axis=1)
            if pf.sum() < 20:
                continue
            e = np.sort(np.vstack([F[pf][:, [0, 1]], F[pf][:, [1, 2]], F[pf][:, [2, 0]]]), axis=1)
            uq, cnt = np.unique(e, axis=0, return_counts=True)
            inner = np.setdiff1d(np.unique(F[pf]), np.unique(uq[cnt == 1]))
            known = np.setdiff1d(np.arange(len(V)), inner)
            for k in range(3):
                flat[:, k] = _harmonic_fill(lap_e, known, flat[known, k])
        # (subdivided as the rest is, but only round the ears: well past the
        # part kept, so the cut's own edge doesn't change it)
        near_ear = (EW[F][:, :, build3d].max(axis=(1, 2)) > 0.05)
        used, inv = np.unique(F[near_ear], return_inverse=True)
        Vs, Fs = subdivide_loop(V[used], inv.reshape(-1, 3), iterations=2)
        Ws, _ = subdivide_loop(np.column_stack([EW[used], np.zeros(len(used))]), inv.reshape(-1, 3), iterations=2)
        A = np.asarray(reference["head_axes"], dtype=np.float64)
        sgx, sgy = grid_xy(Vs)
        ear_cov = np.zeros(shape, dtype=bool)
        patches = []
        for i in range(EW.shape[1]):
            if not build3d[i]:
                continue
            pf = (Ws[Fs, i] >= _REF_EAR_BASE).all(axis=1)
            if pf.sum() < 20:
                continue
            en, _ = _raster_depth(sgx, sgy, Vs[:, 2], Fs[pf], shape)
            ear_cov |= np.isfinite(en)
            used, inv = np.unique(Fs[pf], return_inverse=True)
            out = A[:, 0] * (1.0 if i == 0 else -1.0)  # subject's left ear first
            skirt = np.clip((_REF_EAR_ROOT - Ws[used, i]) / (_REF_EAR_ROOT - _REF_EAR_BASE), 0.0, 1.0)
            P = Vs[used] - out * (skirt * _REF_EAR_SINK * float(reference["px_per_m"]))[:, None]
            patches.append({"vertices": P, "faces": inv.reshape(-1, 3), "inward": -out})
        if patches:
            ears3d = {"patches": patches, "cov": ear_cov, "plug": _REF_EAR_PLUG * float(reference["px_per_m"]),
                      "thick": _REF_EAR_THICK * float(reference["px_per_m"])}
            V = flat
    # forearms lying across the torso, in front of it
    cross = np.zeros(len(F), dtype=bool)
    if _LIMB_SOLIDS and part_w is not None and "torso" in names:
        dom = np.asarray(part_w).argmax(1)
        g0x, g0y = grid_xy(V)
        torso_f = (dom[F] == names.index("torso")).all(axis=1)
        tn, _ = _raster_depth(g0x, g0y, V[:, 2], F[torso_f], shape)
        for s_ in "LR":
            # (the forearm only: hands are the photo's -- a hand tucked out
            # of sight stays in the body; as part of the solid it poked out
            # behind the other elbow)
            ids = [names.index(n) for n in (f"forearm.{s_}",) if n in names]
            af = np.isin(dom[F], ids).all(axis=1)
            if af.sum() < 20:
                continue
            an, _ = _raster_depth(g0x, g0y, V[:, 2], F[af], shape)
            acov = np.isfinite(an)
            over = acov & np.isfinite(tn)
            # (depth runs toward the back: smaller is nearer the camera)
            if acov.any() and over.sum() >= _ARM_IN_FRONT_FRAC * acov.sum() and float(np.mean(an[over] < tn[over])) > 0.8:
                cross |= af
    V_all, F_all = V, F

    def _sub(faces):
        used, inv = np.unique(faces, return_inverse=True)  # only its own vertices (unused ones came out NaN)
        return subdivide_loop(V_all[used], inv.reshape(-1, 3), iterations=2)

    V, F = _sub(F_all[~cross])
    gx, gy = grid_xy(V)
    near, far = _raster_depth(gx, gy, V[:, 2], F, shape)
    limbs3d = None
    if cross.any():
        Va, Fa = _sub(F_all[cross])
        ax_, ay_ = grid_xy(Va)
        an, af_ = _raster_depth(ax_, ay_, Va[:, 2], Fa, shape)
        # (inside the photo's outline: the body's elbow reaching past it stood out as fins)
        acov = np.isfinite(an) & np.isfinite(af_) & (d_out <= 1.5)
        body_cov = np.isfinite(near) & np.isfinite(far)
        # past the torso the arm is the body's surface, as before (its elbow
        # stays one smooth piece with the upper arm); over the torso the
        # forearm is a solid of its own, running a little past where it
        # crosses so it carries on into the arm's own surface
        # (only over the torso itself: by the upper arm the two are combined
        # as before -- switching from one to the other stepped the elbow)
        over_t = acov & np.isfinite(tn)
        rest = acov & ~over_t
        near = np.where(rest, np.fmin(near, an), near)
        far = np.where(rest, np.fmax(far, af_), far)
        body_cov = np.isfinite(near) & np.isfinite(far)
        on_body = over_t & body_cov
        cells = acov & binary_dilation(on_body, iterations=4)
        # (its back smoothed as the body's is)
        sig_l = _REF_BACK_SMOOTH * float(reference["px_per_m"]) * scale
        back_l = _masked_blur(np.where(acov, -af_, 0.0), acov, sig_l)
        limbs3d = {"front": np.where(cells, -an, 0.0), "back": np.where(cells, back_l, 0.0), "cells": cells, "over": on_body}
    cov = np.isfinite(near) & np.isfinite(far)
    if not cov.any():
        return None
    front, back = np.where(cov, -near, 0.0), np.where(cov, -far, 0.0)
    # the photo shows clothes, the body is bare: clothing doesn't follow the
    # back's small folds and creases. Smooth the back, fading out toward the
    # edge so the rim stays where it is.
    # (where an arm's back and the body's back cross, the rearmost of the two
    # left a sharp valley a jacket would hang over.) Not on the head: the
    # back of the skull and the nape are the head's own shape.
    sig = _REF_BACK_SMOOTH * float(reference["px_per_m"]) * scale
    fade = np.clip(distance_transform_edt(cov) / sig, 0.0, 1.0)
    if face_mesh is not None:
        chin_row = (float(np.asarray(face_mesh)[152, 1]) - y0) * scale + pad
        fade = fade * np.clip((np.arange(shape[0]) - chin_row) / max(sig, 1.0), 0.0, 1.0)[:, None]
    back = fade * _masked_blur(back, cov, sig) + (1.0 - fade) * back
    if face_mesh is not None:
        # the reference's own face is a generic one: over the inner face keep
        # only its broad curve, the photo's face relief goes on top later.
        # Fades out well inside the head's edge, which stays the reference's.
        fm = np.asarray(face_mesh)[:, :2]
        fx, fy = (fm[:, 0] - x0) * scale + pad, (fm[:, 1] - y0) * scale + pad
        face_w = max(float(np.ptp(fx)), 1.0)
        nx, ny = fx[1], fy[1]  # nose tip
        ix, iy = nx + 0.7 * (fx - nx), ny + 0.7 * (fy - ny)
        zone = np.zeros(shape, dtype=np.float32)
        zone[np.clip(np.rint(iy).astype(int), 0, shape[0] - 1), np.clip(np.rint(ix).astype(int), 0, shape[1] - 1)] = 1.0
        zone = gaussian_filter(zone, 0.08 * face_w)
        zone = np.clip(2.0 * zone / max(float(zone.max()), 1e-6), 0.0, 1.0)
        zone *= np.clip(distance_transform_edt(cov) / (0.12 * face_w), 0.0, 1.0)
        broad = _masked_blur(front, cov, 0.12 * face_w)
        front = zone * broad + (1.0 - zone) * front
    return front, back, cov, ears3d, limbs3d


def _hands_on_body(parts: Optional[dict], pose: Optional[dict], grid_mask: np.ndarray, y0: int, x0: int, scale: float,
                   pad: int) -> Optional[np.ndarray]:
    """Grid cells of a hand lying over the body: skin (the clothes reading's)
    round the photo's own hand points, well inside the outline (a hand at
    the side is the outline's own)."""
    cl = (parts or {}).get("clothes")
    if not cl or not pose or "left_shoulder" not in pose or "right_shoulder" not in pose:
        return None
    from .clothes import NAMES

    gh, gw = grid_mask.shape
    lab_ = cl["labels"]
    r = np.clip(np.rint(y0 + (np.arange(gh) - pad) / scale).astype(int), 0, lab_.shape[0] - 1)
    c = np.clip(np.rint(x0 + (np.arange(gw) - pad) / scale).astype(int), 0, lab_.shape[1] - 1)
    skin = (lab_[r[:, None], c[None, :]] == NAMES.index("skin") + 1) & grid_mask
    sw = float(np.linalg.norm(np.subtract(pose["left_shoulder"], pose["right_shoulder"]))) * scale
    rows, cols = np.mgrid[0:gh, 0:gw]
    out = np.zeros((gh, gw), dtype=bool)
    for sd in ("left", "right"):
        tips = [pose[f"{sd}_{k}"] for k in ("index", "pinky", "thumb") if f"{sd}_{k}" in pose]
        if not tips:
            continue
        hx, hy = np.mean(np.asarray(tips, dtype=np.float64)[:, :2], axis=0)
        hc, hr = (hx - x0) * scale + pad, (hy - y0) * scale + pad
        out |= skin & ((cols - hc) ** 2 + (rows - hr) ** 2 < (0.14 * sw) ** 2)
    deep = distance_transform_edt(grid_mask) > 0.06 * sw
    out &= deep
    from scipy.ndimage import binary_opening

    out = binary_opening(out, iterations=1)
    return out if out.sum() >= 12 else None


def _layer_solid(front: np.ndarray, back: np.ndarray, cells: np.ndarray, zs: np.ndarray, dz: float, scale: float,
                 smooth: bool):
    """A closed solid between `front` and `back` over `cells` (the build's
    own grid and depth steps), its edge rounded: (vertices as (depth, row,
    col) in the build's units, faces), or None."""
    if cells.sum() < 6:
        return None
    rr, cc = np.nonzero(cells)
    m = 4
    r0, r1 = max(rr.min() - m, 0), min(rr.max() + m + 1, cells.shape[0])
    c0, c1 = max(cc.min() - m, 0), min(cc.max() + m + 1, cells.shape[1])
    cl = cells[r0:r1, c0:c1]
    mid = ((front + back) / 2.0)[r0:r1, c0:c1]
    half = np.maximum((front - back) / 2.0, 0.5 / scale)[r0:r1, c0:c1]
    G = np.where(cl, half ** 2, 0.0)
    d_out, (iy, ix) = distance_transform_edt(~cl, return_indices=True)
    slope = float(np.median(G[cl])) if cl.any() else 1.0
    G = np.where(cl, G, -d_out * max(slope, 1e-6))
    mid = np.where(cl, mid, mid[iy, ix])
    field = np.empty((len(zs), cl.shape[0], cl.shape[1]), dtype=np.float32)
    for i, z in enumerate(zs):
        field[i] = (z - mid) ** 2 - G
    if field.min() >= 0 or field.max() <= 0:
        return None
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(dz, 1.0 / scale, 1.0 / scale))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    parts_ = mesh.split(only_watertight=False)
    if len(parts_) > 1:
        mesh = max(parts_, key=lambda p: p.vertices.shape[0])
    if smooth:
        smooth_grid_terracing(mesh)
    v = np.asarray(mesh.vertices).copy()
    v[:, 1] += r0 / scale
    v[:, 2] += c0 / scale
    return v, np.asarray(mesh.faces)


def _ear_solid(P: np.ndarray, T: np.ndarray, inward: np.ndarray, plug: float, thicken: float):
    """A closed solid from an ear's surface patch (the ear and the skin at its
    base, outward-facing): thickened along its normals, its rim carried
    `plug` along `inward` into the head and capped. None if the patch isn't
    a single disc."""
    if thicken > 0:
        P = P + trimesh.Trimesh(P, T, process=False).vertex_normals * thicken
    edges = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    have = set(map(tuple, edges.tolist()))
    rim = [(a, b) for a, b in edges.tolist() if (b, a) not in have]
    nxt = dict(rim)
    if not rim or len(nxt) != len(rim):
        return None
    loop = [rim[0][0]]
    while len(loop) <= len(rim):
        n = nxt.get(loop[-1])
        if n is None:
            return None
        if n == loop[0]:
            break
        loop.append(n)
    if len(loop) != len(rim):
        return None  # more than one rim: not a disc
    k, n0 = len(loop), len(P)
    ring = P[loop] + inward * plug
    V = np.vstack([P, ring, ring.mean(axis=0)])
    j = np.arange(k)
    a, b = np.asarray(loop), np.asarray(loop)[(j + 1) % k]
    ia, ib = n0 + j, n0 + (j + 1) % k
    walls = np.vstack([np.column_stack([b, a, ia]), np.column_stack([b, ia, ib])])
    cap = np.column_stack([ib, ia, np.full(k, n0 + k)])
    return V, np.vstack([T, walls, cap])


def _join_solids(V: np.ndarray, F: np.ndarray, parts: list):
    """Boolean union (manifold3d) of a closed mesh with closed parts; parts
    that aren't valid solids are left out. The largest resulting piece (a
    union can leave slivers)."""
    from manifold3d import Manifold, Mesh

    def solid(v, f):
        return Manifold(mesh=Mesh(vert_properties=np.asarray(v, dtype=np.float32), tri_verts=np.asarray(f, dtype=np.uint32)))

    out = solid(V, F)
    if out.is_empty():
        raise ValueError("the head is not a closed solid")
    for v, f in parts:
        m = solid(v, f)
        if not m.is_empty() and m.volume() > 0:
            out = out + m
    pieces = out.decompose()
    if len(pieces) > 1:
        out = max(pieces, key=lambda c: c.num_tri())
    mesh = out.to_mesh()
    return np.asarray(mesh.vert_properties[:, :3], dtype=np.float64), np.asarray(mesh.tri_verts, dtype=np.int64)


def _offset_surface(F, B, solid, radius_px, scale):
    """The 3D offset of a front/back solid by `radius_px` (photo px): every
    point of it grown into a ball of that radius (grey dilation of the front,
    erosion of the back, with a ball). Returns front, back and the cells it
    covers."""
    from scipy.ndimage import distance_transform_edt, grey_dilation

    n = max(int(np.ceil(radius_px * scale)), 1)
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(yy, xx) / scale
    ball = d <= radius_px
    cap = np.where(ball, np.sqrt(np.maximum(radius_px ** 2 - d ** 2, 0.0)), 0.0)
    low = -1e9
    Fo = grey_dilation(np.where(solid, F, low), footprint=ball, structure=cap)
    Bo = -grey_dilation(np.where(solid, -B, low), footprint=ball, structure=cap)
    covers = distance_transform_edt(~solid) / scale <= radius_px
    return Fo, Bo, covers & (Fo > low / 2) & (Bo < -low / 2)


def _offset_variable(F, B, solid, radius_px, scale):
    """The 3D offset of a front/back solid where each cell's own radius
    (photo px) is radius_px there: every point of the solid within that
    radius of the cell, as a ball. Returns front, back (-inf/+inf where
    nothing reaches)."""
    gh, gw = F.shape
    n = max(int(np.ceil(float(np.nanmax(radius_px)) * scale)), 1)
    Fo = np.full((gh, gw), -np.inf)
    Bo = np.full((gh, gw), np.inf)
    Fs = np.where(solid, F, -np.inf)
    Bs = np.where(solid, B, np.inf)
    R2 = radius_px ** 2
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            d2 = (dy * dy + dx * dx) / (scale * scale)
            if d2 > n * n / (scale * scale):
                continue
            # the solid's cell at (y + dy, x + dx), seen from (y, x)
            ys0, ys1 = max(0, -dy), min(gh, gh - dy)
            xs0, xs1 = max(0, -dx), min(gw, gw - dx)
            src = (slice(ys0 + dy, ys1 + dy), slice(xs0 + dx, xs1 + dx))
            dst = (slice(ys0, ys1), slice(xs0, xs1))
            h = np.sqrt(np.maximum(R2[dst] - d2, 0.0))
            ok = R2[dst] >= d2
            np.maximum(Fo[dst], np.where(ok, Fs[src] + h, -np.inf), out=Fo[dst])
            np.minimum(Bo[dst], np.where(ok, Bs[src] - h, np.inf), out=Bo[dst])
    return Fo, Bo


def _hair_and_hat(front, back, grid_mask, photo_cells, parts_g, cov, face_mesh, y0, x0, scale, pad, px_per_m):
    """Hair (and a cap or hat) as their own volume round the skull, in the
    build grid (depth in photo px, z toward the camera).

    How thick the hair lies is how far it reaches past the skull at the
    sides in the photo. The skull -- the reference body's head -- is grown by
    that much in 3D (every point of it into a ball), so the hair follows the
    skull's own curve over the top, down the sides and round the back. Hair
    that reaches further than that (a bun, big hair, long hair beside the
    neck) is a rounded mass of its own, as deep as it is wide, joined to it.
    Seen over the skull from the front (the crown, a fringe) the hair lies
    on it, a fringe never thicker than _HAIR_FRINGE, its edges feathered into
    the skin; nothing goes on the face below the brows. Below the chin long
    hair lies over the shoulders and falls straight down the back from the
    back of the head. A cap whose lower edge comes down to the brows gets a
    peak. Returns front, back and the cells to build."""
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter1d, label as cc_label

    from .head_parts import ACCESSORY, HAIR, HAT

    gh, gw = grid_mask.shape
    fm = np.asarray(face_mesh, dtype=np.float64)[:, :2]
    to_r = lambda y: (y - y0) * scale + pad  # noqa: E731
    to_c = lambda x: (x - x0) * scale + pad  # noqa: E731
    chin_r = int(round(to_r(fm[152, 1])))
    eye_r = to_r((fm[33, 1] + fm[263, 1]) / 2.0)
    brow_r = to_r((fm[105, 1] + fm[334, 1]) / 2.0)
    base_r = to_r(fm[2, 1])
    ruler = max(base_r - brow_r, 2.0)  # brow line to nose base, cells
    c_face = int(round(to_c((fm[234, 0] + fm[454, 0]) / 2.0)))
    face_w = abs(float(fm[454, 0] - fm[234, 0]))  # photo px
    rows = np.arange(gh)[:, None]
    hairish = np.isin(parts_g, (HAIR, HAT)) & photo_cells
    # a flower or clip in the hair is part of its volume
    hairish |= (parts_g == ACCESSORY) & photo_cells & binary_dilation(hairish, iterations=2) & (rows < eye_r)
    # nothing on the face below the brows (a stray label on the nose)
    from PIL import Image as _Image, ImageDraw as _Draw

    oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
            149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    face_img = _Image.new("1", (gw, gh), 0)
    _Draw.Draw(face_img).polygon([(float(to_c(fm[i, 0])), float(to_r(fm[i, 1]))) for i in oval], fill=1)
    hairish &= ~(np.asarray(face_img, dtype=bool) & (rows > brow_r))
    if hairish.sum() < 20 or not 0 <= c_face < gw or not 0 < chin_r < gh:
        return front, back, grid_mask
    hat = (parts_g == HAT) & hairish
    # the outline's own edge round the hair, where the parts map called it
    # background (its pixels are coarser): hair too, not a fin of outline
    hairish |= photo_cells & (parts_g == 0) & binary_dilation(hairish, iterations=3) & (rows <= chin_r)
    # the skull: the reference's head above the chin, the piece round the eyes
    lab, _ = cc_label(cov & (rows <= chin_r))
    er = int(np.clip(round(eye_r), 0, gh - 1))
    keep = lab[er, max(c_face - 2, 0):c_face + 3]
    keep = keep[keep > 0]
    if not len(keep):
        return front, back, grid_mask
    skull = lab == np.bincount(keep).argmax()
    # the hair (and hat) touching the head; long hair hanging beside the neck
    lab2, _ = cc_label(skull | (hairish & ~(cov & (rows > chin_r))))
    keep2 = np.unique(lab2[skull])
    region = np.isin(lab2, keep2[keep2 > 0])
    hair_here = region & hairish & ~skull
    if hair_here.sum() < 10 and not (hairish & skull).any():
        return front, back, grid_mask
    # how thick it lies, area by area: how far the hair's outline reaches
    # past the skull there (4-5 cm on top of an updo, a few mm on a short
    # cut), spread smoothly inward from the outline and over the skull; up to
    # _HAIR_SHELL_MAX. (Hair is that thick all over the area -- thinning it to
    # nothing toward the outline in the photo's plane made a wall on top.)
    d_skull = distance_transform_edt(~skull) / scale  # photo px
    tmax = _HAIR_SHELL_MAX * px_per_m
    edge = hair_here & ~binary_erosion(region)
    if edge.sum() < 5:
        return front, back, grid_mask
    iy, ix = distance_transform_edt(~edge, return_distances=False, return_indices=True)
    tloc = np.minimum(d_skull, tmax)[iy, ix]
    tloc = _masked_blur(tloc, np.ones_like(region), max(0.12 * face_w * scale, 1.0))
    # ... and comes down to nothing right at the outline (a hair edge with
    # any thickness left there closed in a cut: a seam over the crown)
    d_edge = distance_transform_edt(region) / scale
    tloc = np.where(hair_here, np.minimum(tloc, d_skull + 3.0 * d_edge), tloc)
    tloc = np.clip(tloc, 0.15 * _HAIR_FRINGE * px_per_m, tmax)
    # the skull grown by it in 3D, each point by its own thickness
    ys, xs = np.nonzero(region | skull)
    m = int(np.ceil(tmax * scale)) + 3
    r0, r1, c0, c1 = max(ys.min() - m, 0), min(ys.max() + m + 1, gh), max(xs.min() - m, 0), min(xs.max() + m + 1, gw)
    crop = (slice(r0, r1), slice(c0, c1))
    fo, bo = _offset_variable(front[crop], back[crop], skull[crop], tloc[crop], scale)
    # round the back the hair is as thick as the front photo shows it
    # reaching past the skull at the sides, at the same height -- the same
    # hair going round -- not as thin as the nearest outline (mostly the
    # crown, where hair lies flat: measured on long full hair, 2 mm at the
    # back against 21 mm at the sides)
    reach = np.full(gh, np.nan)
    for r in np.flatnonzero(skull.any(axis=1)):
        if r > chin_r:
            continue
        sk = np.flatnonzero(skull[r])
        rg = np.flatnonzero(region[r] | skull[r])
        vals = []
        for out_c, sk_c, d_ in ((rg.min(), sk.min(), sk.min() - rg.min()), (rg.max(), sk.max(), rg.max() - sk.max())):
            if d_ > 0 and hairish[r, out_c]:
                vals.append(d_ / scale)
            elif hairish[r, sk_c]:
                vals.append(0.0)
        if vals:
            reach[r] = float(np.mean(vals))
    tback = tloc
    known_r = np.isfinite(reach)
    if known_r.sum() >= 3:
        rr_ = np.arange(gh)
        reach = np.interp(rr_, rr_[known_r], reach[known_r])
        reach = gaussian_filter1d(reach, max(0.15 * face_w * scale, 1.0))  # row to row it stepped: stripes round the back
        side = np.clip(_HAIR_BACK_FROM_SIDES * reach, 0.0, tmax)[:, None]
        tback = np.where(skull, np.maximum(tloc, side), tloc)
    _, bo2 = _offset_variable(front[crop], back[crop], skull[crop], tback[crop], scale)
    Fo = np.full((gh, gw), -np.inf)
    Bo = np.full((gh, gw), np.inf)
    Fo[crop], Bo[crop] = fo, bo
    Bb = np.full((gh, gw), np.inf)
    Bb[crop] = bo2
    shell = np.isfinite(Fo) & np.isfinite(Bo) & (Fo > Bo)
    # a front photo can't say whether hair piled high sits forward or back;
    # it goes up and back: never more than a fringe in front of the forehead
    br = int(np.clip(round(brow_r - 0.3 * ruler), 0, gh - 1))
    fcols = np.arange(max(c_face - 3, 0), min(c_face + 4, gw))
    if skull[br, fcols].any():
        brow_front = float(front[br, fcols][skull[br, fcols]].max())
        Fo = np.minimum(Fo, brow_front + _HAIR_FRINGE * px_per_m)
    if _HAIR_DEBUG is not None:
        _HAIR_DEBUG.update(tloc=tloc.copy(), Fo=Fo.copy(), Bo=Bo.copy(), skull=skull.copy(), region=region.copy(),
                           hair_here=hair_here.copy(), d_skull=d_skull.copy(), edge=edge.copy())
    F, B = front.copy(), back.copy()
    # how much it's hair rather than the head's own outline: none for a
    # short cut, all of it from twice that, smoothly in between (a hard
    # cut-off, mixed with which rows the outline happened to reach, stamped
    # bars above every ear)
    thick = np.clip((tloc - _HAIR_THIN * px_per_m) / (_HAIR_THIN * px_per_m), 0.0, 1.0)
    on_hair = hair_here & shell & np.isfinite(Fo) & np.isfinite(Bo)
    blend = on_hair & grid_mask & (thick > 0)
    F = np.where(blend, F + thick * (np.where(blend, Fo, F) - F), F)
    B = np.where(blend, B + thick * (np.where(blend, Bo, B) - B), B)
    new_cells = on_hair & ~grid_mask & (thick >= 0.5)
    F[new_cells], B[new_cells] = Fo[new_cells], Bo[new_cells]
    built = blend | new_cells
    # further than that (long hair, a big bun): a rounded mass of its own,
    # centred where the shell is
    more = hair_here & ~shell
    if more.any():
        blob = more | (binary_dilation(more, iterations=2) & (shell | skull))
        blob[:1], blob[-1:], blob[:, :1], blob[:, -1:] = False, False, False, False
        hb = _inflation_height(blob) / scale
        known = shell & np.isfinite(Fo) & np.isfinite(Bo)
        if known.any():
            iy, ix = distance_transform_edt(~known, return_distances=False, return_indices=True)
            mid = np.where(known, (np.where(known, Fo, 0.0) + np.where(known, Bo, 0.0)) / 2.0, 0.0)
            mm = _masked_blur(mid[iy, ix], blob, max(0.05 * face_w * scale, 1.0))
            F[more], B[more] = mm[more] + hb[more], mm[more] - hb[more]
            built |= more
    # hair beside the face lies behind the cheek next to it, meeting it
    # (standing proud of the face's own falling edge it left a crack)
    from .head_parts import BODY_SKIN, FACE_SKIN

    skin = skull & np.isin(parts_g, (FACE_SKIN, BODY_SKIN))
    if skin.any():
        dist_s, (sy, sx) = distance_transform_edt(~skin, return_indices=True)
        reach_ = max(_HAIR_FACE_MEET * px_per_m * scale, 1.0)
        near_s = built & (dist_s <= reach_)
        cap_f = front[sy, sx]
        k_ = np.clip(1.0 - dist_s / reach_, 0.0, 1.0)
        F = np.where(near_s, F + k_ * (np.minimum(F, cap_f) - F), F)
    # hair hanging beside the neck falls behind it: the neck shows as a neck
    neck = np.isin(parts_g, (BODY_SKIN,)) & (rows > chin_r) & grid_mask
    if neck.any():
        dist_n, (ny, nx) = distance_transform_edt(~neck, return_indices=True)
        reach_n = max(_HAIR_FACE_MEET * px_per_m * scale, 1.0)
        near_n = (built | (hairish & (rows > chin_r))) & (dist_n <= reach_n) & ~neck
        cap_n = front[ny, nx] - _HAIR_NECK_BEHIND * px_per_m
        F = np.where(near_n, np.minimum(F, cap_n), F)
    # round the back of the head, down to the nape (hair at the back reaches
    # about the ear lobes even when the sides are short)
    nape = max(base_r, max((r for r in np.flatnonzero(hair_here.any(axis=1)) if r <= chin_r), default=0))
    w = np.clip((nape - rows) / max(_HAIR_NAPE_EDGE * px_per_m * scale, 1.0) + 0.5, 0.0, 1.0)  # a hairline, not a fade
    on = skull & np.isfinite(Bb)
    w = w * np.clip((tback - _HAIR_THIN * px_per_m) / (_HAIR_THIN * px_per_m) + 1.0, 0.0, 1.0)
    B = np.where(on, np.minimum(B, B + w * (np.where(on, Bb, B) - B)), B)
    # seen over the skull from the front: on it, a fringe thin, edges feathered
    over = hairish & skull
    lay = np.where(over & shell, np.maximum(Fo - F, 0.0), 0.0)
    near_brow = np.clip((rows - (brow_r - 0.6 * ruler)) / max(0.3 * ruler, 1.0), 0.0, 1.0)
    lay = np.minimum(lay, (1.0 - near_brow) * lay + near_brow * _HAIR_FRINGE * px_per_m)
    # (feathered only where it meets skin: where it meets the hair beside
    # the skull it carries straight on -- feathering there left a seam)
    feather = np.clip(distance_transform_edt(over | hair_here) / max(_HAIR_FEATHER * px_per_m * scale, 1.0), 0.0, 1.0)
    # short hair (thinner than _HAIR_THIN: its outline is the head's own) is
    # still a layer on the scalp, with a hairline across the forehead --
    # without it a short cut read as a bare head
    short = np.where(over, _HAIR_SHORT * px_per_m * (1.0 - thick), 0.0)  # (thicker hair has its own layer)
    F = F + np.maximum(lay * thick, short) * feather
    # long hair over the shoulders and chest: a layer on them
    on_body = hairish & cov & (rows > chin_r)
    F[on_body] = F[on_body] + _HAIR_LAYER * px_per_m * np.clip(distance_transform_edt(on_body)[on_body] / 2.0, 0.0, 1.0)
    # falling down the back from the back of the head, where it hangs on
    # both sides of the neck
    below = hairish & (rows > chin_r)
    if below.any():
        occ = np.where(skull | built, B, np.inf).min(axis=0)
        for r in np.flatnonzero(below.any(axis=1)):
            hc = np.flatnonzero(below[r])
            if hc.min() < c_face < hc.max():
                cols = np.arange(hc.min(), hc.max() + 1)
                cols = cols[(grid_mask[r, cols] | built[r, cols]) & np.isfinite(occ[cols])]
                B[r, cols] = np.minimum(B[r, cols], occ[cols])
    # a cap's peak: along the hat's lower edge across the face, if it comes
    # down near the brows; one smooth curve, thin as a peak is
    if hat.any():
        half_w = 0.45 * face_w * scale
        fc = np.arange(int(round(c_face - half_w)), int(round(c_face + half_w)) + 1)
        fc = fc[(fc >= 0) & (fc < gw)]
        bot = np.array([float(np.flatnonzero(hat[:, cc]).max()) if hat[:, cc].any() else np.nan for cc in fc])
        good = np.isfinite(bot)
        if good.mean() > 0.6 and brow_r - 0.8 * ruler <= np.median(bot[good]) <= eye_r:
            # one smooth arc across the brow (the label's edge is ragged)
            coef = np.polyfit(fc[good] - c_face, bot[good], 2)
            bot = np.polyval(coef, fc - c_face)
            band = max(0.1 * ruler, 1.5)
            peak = _HAT_PEAK * face_w
            base = np.array([F[min(int(round(rb_)) + 1, gh - 1), cc] for cc, rb_ in zip(fc, bot)])
            base = gaussian_filter1d(base, max(0.15 * half_w, 1.0))  # the brow's bumps aren't the peak's
            for cc, rb_, b0 in zip(fc, bot, base):
                k = np.sqrt(max(1.0 - ((cc - c_face) / half_w) ** 2, 0.0))
                r1_ = int(round(rb_))
                rr = np.arange(max(int(round(rb_ - band)), 0), min(r1_, gh - 1) + 1)
                F[rr, cc] = np.maximum(F[rr, cc], b0 + peak * k)
    # smooth across every boundary it changed (label edges are pixel-ragged:
    # left as they are they were cracks along the hairline, the cap's edge)
    cells = grid_mask | built
    changed = cells & ((np.abs(F - front) > 0.5) | (np.abs(B - back) > 0.5) | built)
    if changed.any():
        # (not at the outline's own edge: smoothing there stepped the edge
        # row by row -- ridges down the sides)
        zone = binary_dilation(changed, iterations=3) & cells & (distance_transform_edt(cells) > 3)
        sig = max(_HAIR_SMOOTH * px_per_m * scale, 1.0)
        F = np.where(zone, _masked_blur(F, cells, sig), F)
        B = np.where(zone, _masked_blur(B, cells, sig), B)
        # and wider where the hair meets the face's side (a cliff there read
        # as a crack between the hair and the cheek)
        seam = binary_dilation(built, iterations=2) & binary_dilation(skull & ~built, iterations=2) & cells & (distance_transform_edt(cells) > 3)
        if seam.any():
            w2 = np.clip(gaussian_filter(seam.astype(np.float32), 2.0) * 2.0, 0.0, 1.0)
            F = F + w2 * (_masked_blur(F, cells, 2.5 * sig) - F)
    B = np.minimum(B, F - 1.0 / scale)
    return F, B, cells


def _flow_on_grid(reading: dict, grid_shape, y0: int, x0: int, scale: float, pad: int):
    """The photo's strand direction (cos 2θ, sin 2θ) and coherence at each
    build-grid cell; None outside the reading's box."""
    gh, gw = grid_shape
    bx0, by0, bx1, by1 = reading["box"]
    rr, cc = np.mgrid[0:gh, 0:gw].astype(np.float32)
    py = y0 + (rr - pad) / scale - by0
    px = x0 + (cc - pad) / scale - bx0
    th, coh = reading["theta"], reading["coherence"]
    inside = (py >= 0) & (py < th.shape[0] - 1) & (px >= 0) & (px < th.shape[1] - 1)
    c2 = map_coordinates(np.cos(2 * th), [py, px], order=1, mode="nearest")
    s2 = map_coordinates(np.sin(2 * th), [py, px], order=1, mode="nearest")
    co = map_coordinates(coh, [py, px], order=1, mode="nearest")
    return np.where(inside, c2, 0.0), np.where(inside, s2, 0.0), np.where(inside, co, 0.0), inside


def _lic(noise: np.ndarray, c2: np.ndarray, s2: np.ndarray, half_len: int) -> np.ndarray:
    """Line integral convolution: `noise` averaged along the direction field
    (cos 2θ, sin 2θ), half_len cells each way, Hann-weighted -- streaks that
    follow the strands."""
    gh, gw = noise.shape
    rr, cc = np.mgrid[0:gh, 0:gw].astype(np.float64)
    ang0 = 0.5 * np.arctan2(s2, c2)
    acc = noise.astype(np.float64).copy()
    wsum = np.ones_like(acc)
    for sign in (1.0, -1.0):
        y, x = rr.copy(), cc.copy()
        ty, tx = np.sin(ang0) * sign, np.cos(ang0) * sign
        for k in range(1, half_len + 1):
            cs = map_coordinates(c2, [y, x], order=1, mode="nearest")
            sn = map_coordinates(s2, [y, x], order=1, mode="nearest")
            ang = 0.5 * np.arctan2(sn, cs)
            ny, nx = np.sin(ang), np.cos(ang)
            flip = ny * ty + nx * tx < 0
            ty, tx = np.where(flip, -ny, ny), np.where(flip, -nx, nx)
            y, x = np.clip(y + ty, 0, gh - 1), np.clip(x + tx, 0, gw - 1)
            w = 0.5 * (1.0 + np.cos(np.pi * k / (half_len + 1)))
            acc += w * map_coordinates(noise, [y, x], order=1, mode="nearest")
            wsum += w
    return acc / wsum


def _wave_phase(c2: np.ndarray, s2: np.ndarray, region: np.ndarray, crown) -> np.ndarray:
    """Distance along the strands from the crown (a potential whose gradient
    is the strand direction, pointed away from the crown), for wave crests
    that run across the strands."""
    gh, gw = region.shape
    ang = 0.5 * np.arctan2(s2, c2)
    ty, tx = np.sin(ang), np.cos(ang)
    rr, cc = np.mgrid[0:gh, 0:gw]
    away = (rr - crown[0]) * ty + (cc - crown[1]) * tx < 0
    ty, tx = np.where(away, -ty, ty), np.where(away, -tx, tx)
    idx = -np.ones(region.shape, dtype=np.int64)
    n = int(region.sum())
    idx[region] = np.arange(n)
    rows, cols, vals, rhs = [], [], [], []
    eq = 0
    for dy, dx, t in ((0, 1, tx), (1, 0, ty)):
        a = region[: gh - dy, : gw - dx] & region[dy:, dx:]
        i0 = idx[: gh - dy, : gw - dx][a]
        i1 = idx[dy:, dx:][a]
        tt = 0.5 * (t[: gh - dy, : gw - dx][a] + t[dy:, dx:][a])
        m = len(i0)
        rows += [np.arange(eq, eq + m)] * 2
        cols += [i1, i0]
        vals += [np.ones(m), -np.ones(m)]
        rhs.append(tt)
        eq += m
    # pin the crown's nearest cell at 0
    ci = idx[region][np.argmin((rr[region] - crown[0]) ** 2 + (cc[region] - crown[1]) ** 2)]
    rows.append(np.array([eq]))
    cols.append(np.array([ci]))
    vals.append(np.array([10.0]))
    rhs.append(np.array([0.0]))
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(eq + 1, n))
    # (a little of the identity: hair in separate pieces, each but the
    # crown's unpinned, would leave it singular)
    sol = spsolve((A.T @ A + 1e-6 * sparse.identity(n)).tocsc(), A.T @ np.concatenate(rhs))
    out = np.zeros(region.shape)
    out[region] = sol
    return out


def _hair_pattern(front, back, front_pre, back_pre, cells, parts_g, reading, face_mesh, y0, x0, scale, pad, px_per_m,
                  print_px_per_mm=None):
    """The hair's pattern as offsets (photo px, outward) for its front and
    back surfaces: strands along the photo's own strand directions, the
    waves it measured, curls or coils, the ropes of braids and locs along
    the strands' course -- as much of each as the reading's
    kind of hair takes (_HAIR_KIND_MIX). Ends at the hairline; fades where
    the surface turns side-on (pressed onto a side-facing surface, detail
    stands out as ridges). The back of the head: strands falling down, or
    the same curls."""
    from scipy.ndimage import binary_dilation

    from .head_parts import BODY_SKIN, FACE_SKIN, HAIR

    gh, gw = cells.shape
    zero = np.zeros((gh, gw))
    mix = _HAIR_KIND_MIX.get((reading or {}).get("kind"))
    if mix is None:
        return zero, zero
    per_cell = px_per_m * scale  # grid cells per metre
    # (on the print at least PRINT_GROOVE_DEPTH_MM deep and the strands
    # PRINT_GROOVE_W_MM wide: sized on the person they printed as nothing)
    floor_m = (PRINT_GROOVE_DEPTH_MM * print_px_per_mm / px_per_m) if print_px_per_mm else 0.0
    strand_w = max(_HAIR_STRAND_W, (2.0 * PRINT_GROOVE_W_MM * print_px_per_mm / px_per_m) if print_px_per_mm else 0.0)
    hair_f = (parts_g == HAIR) & cells
    if hair_f.sum() < 30:
        return zero, zero
    grown = cells & (((front - back) - np.where(cells, front_pre - back_pre, 0.0)) * scale > 1.0)
    # the back of the head down to the nape is hair too, however short (its
    # back surface only grows for hair thicker than _HAIR_THIN)
    from scipy.ndimage import label as cc_label

    fm = np.asarray(face_mesh)[:, :2]
    nape_r = (fm[2, 1] - y0) * scale + pad
    face_c = ((fm[234, 0] + fm[454, 0]) / 2.0 - x0) * scale + pad
    half_w = 0.8 * abs(fm[454, 0] - fm[234, 0]) * scale
    rows, cols = np.mgrid[0:gh, 0:gw]
    head = cells & (rows <= nape_r) & (np.abs(cols - face_c) <= half_w)
    lab, _ = cc_label(head)
    keep = np.unique(lab[hair_f & head])
    head = np.isin(lab, keep[keep > 0])
    hair_b = (grown & binary_dilation(hair_f | grown, iterations=2)) | head
    region = hair_f | hair_b
    if region.sum() < 30:
        return zero, zero
    ys, xs = np.nonzero(region)
    m = int(np.ceil(_HAIR_STRAND_LEN * per_cell)) + 4
    r0, r1, c0, c1 = max(ys.min() - m, 0), min(ys.max() + m + 1, gh), max(xs.min() - m, 0), min(xs.max() + m + 1, gw)
    crop = (slice(r0, r1), slice(c0, c1))
    reg = region[crop]
    c2, s2, coh, seen = _flow_on_grid(reading, (gh, gw), y0, x0, scale, pad)
    c2, s2, seen = c2[crop], s2[crop], seen[crop]
    # unseen (the sides past the photo's hair, the back): strands fall down
    down = ~(seen & hair_f[crop])
    c2 = gaussian_filter(np.where(down, -1.0, c2), 1.0)
    s2 = gaussian_filter(np.where(down, 0.0, s2), 1.0)
    shape = reg.shape
    crown = (float(ys.min()) - r0, float((fm[234, 0] + fm[454, 0]) / 2.0 - x0) * scale + pad - c0)

    def unit(a):
        sd = float(a[reg].std()) if reg.any() else 0.0
        return (a - float(a[reg].mean())) / sd if sd > 1e-9 else np.zeros_like(a)

    w_s, w_w, w_c, w_k, w_r = mix
    wl = reading.get("wave_px")
    wl_cells = max(wl * scale if wl else _HAIR_WAVE_LEN * per_cell, 4.0)
    rw = reading.get("rope_px")
    rope_cells = max(rw * scale if rw else _HAIR_ROPE_W * per_cell, 2.0)

    def pattern(fc2, fs2, back_side, seed):
        rng = np.random.default_rng(seed)
        out = np.zeros(shape)
        if w_s:
            noise = gaussian_filter(rng.standard_normal(shape), max(0.5 * strand_w * per_cell, 0.6))
            half = int(np.clip(0.5 * _HAIR_STRAND_LEN * per_cell, 2, 40))
            out += w_s * _HAIR_STRAND_DEPTH * unit(_lic(noise, fc2, fs2, half))
        if w_w:
            # (down the back the strands fall straight: distance down is the phase)
            phase = (np.arange(shape[0])[:, None] - crown[0]) * np.ones(shape) if back_side else _wave_phase(fc2, fs2, reg, crown)
            out += w_w * _HAIR_WAVE_DEPTH * np.where(reg, np.sin(2 * np.pi * phase / wl_cells), 0.0)
        for w_, size_, depth_ in ((w_c, _HAIR_CURL, _HAIR_CURL_DEPTH), (w_k, _HAIR_COIL, _HAIR_COIL_DEPTH)):
            if w_:
                d = max(size_ * per_cell, (2.0 * PRINT_GROOVE_W_MM * print_px_per_mm * scale) if print_px_per_mm else 0.0, 1.5)
                bp = gaussian_filter(rng.standard_normal(shape), d / 4) - gaussian_filter(rng.standard_normal(shape), d / 2)
                out += w_ * depth_ * unit(gaussian_filter(np.abs(unit(bp)), d / 6))
        if w_r:
            # ropes: blobs a rope wide drawn out along the strands, rounded
            # over (tanh) with grooves between
            noise = gaussian_filter(rng.standard_normal(shape), rope_cells / 3)
            half = int(np.clip(6 * rope_cells, 4, 90))
            out += w_r * _HAIR_ROPE_DEPTH * np.tanh(1.5 * unit(_lic(noise, fc2, fs2, half)))
        sd = float(out[reg].std()) if reg.any() else 0.0
        if floor_m > 0 and 1e-12 < sd < floor_m:
            out = out * (floor_m / sd)  # (the pattern as a whole at least the print's groove deep)
        return out * px_per_m  # metres -> photo px

    out_f = pattern(c2, s2, False, 0)
    out_b = pattern(np.full(shape, -1.0), np.zeros(shape), True, 1)
    part = reading.get("parting")
    if part is not None and len(part) >= 2:
        # the parting: a groove along it
        pr = (np.asarray(part)[:, 1] - y0) * scale + pad - r0
        pc = (np.asarray(part)[:, 0] - x0) * scale + pad - c0
        rr_, cc_ = np.mgrid[0:shape[0], 0:shape[1]]
        groove = np.zeros(shape)
        wid = max(0.004 * per_cell, 0.8)
        for a, b_ in zip(np.column_stack([pr, pc])[:-1], np.column_stack([pr, pc])[1:]):
            ab = b_ - a
            t = np.clip(((rr_ - a[0]) * ab[0] + (cc_ - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
            d2 = (rr_ - a[0] - t * ab[0]) ** 2 + (cc_ - a[1] - t * ab[1]) ** 2
            groove = np.maximum(groove, np.exp(-d2 / (2 * wid ** 2)))
        out_f = out_f - _PARTING_DEPTH * px_per_m * groove
    # where it applies: in front, the hair the photo shows, ending at the
    # hairline; behind, the back of the head down to the nape (the photo's
    # face covers those cells -- it says nothing about the back)
    skin = np.isin(parts_g[crop], (FACE_SKIN, BODY_SKIN)) & cells[crop]
    edge_cells = max(_HAIRLINE_EDGE * per_cell, 1.0)
    w_f = gaussian_filter(hair_f[crop].astype(np.float64), 1.0) * np.clip(distance_transform_edt(~skin) / edge_cells, 0.0, 1.0)
    nape = np.clip((nape_r - r0 - np.arange(r1 - r0)[:, None]) / edge_cells + 0.5, 0.0, 1.0)
    w_b = gaussian_filter(hair_b[crop].astype(np.float64), max(1.0, 0.5 * _HAIR_STRAND_W * per_cell))
    w_b = w_b * np.maximum(nape, grown[crop])

    def facing(surface):
        gy, gx = np.gradient(surface)
        f = 1.0 / np.sqrt(1.0 + (np.hypot(gy, gx) * scale) ** 2)
        return np.clip((f - _REF_RELIEF_FACING[0]) / (_REF_RELIEF_FACING[1] - _REF_RELIEF_FACING[0]), 0.0, 1.0)

    rel_f = np.where(hair_f[crop], out_f * w_f * facing(front[crop]), 0.0)
    rel_b = np.where(hair_b[crop] | (w_b > 0.01), out_b * w_b * facing(-back[crop]), 0.0)
    if not (np.isfinite(rel_f).all() and np.isfinite(rel_b).all()):
        return zero, zero
    Rf, Rb = zero.copy(), zero.copy()
    Rf[crop], Rb[crop] = rel_f, rel_b
    return Rf, Rb


def _face_parts_relief(reading: dict, face_mesh: np.ndarray, grid_shape, y0: int, x0: int, scale: float, pad: int,
                       px_per_m: float, print_px_per_mm: Optional[float] = None) -> np.ndarray:
    """Offsets (photo px, toward the camera) for the face's parts on top of
    the landmark relief: eyeballs behind their lids, irises, lid creases or
    a closed lid's line, eyebrow hair, nostrils, the lips' edge, a closed
    mouth's line or an open mouth's dark with its teeth and tongue set in,
    scars, a beard, moustache or stubble."""
    gh, gw = grid_shape
    out = np.zeros((gh, gw))
    if not reading:
        return out
    bx0, by0, bx1, by1 = reading["box"]
    rr, cc = np.mgrid[0:gh, 0:gw]
    py = (y0 + (rr - pad) / scale - by0).astype(np.float64)  # in the reading's box
    px = (x0 + (cc - pad) / scale - bx0).astype(np.float64)
    inbox = (py >= 0) & (py < by1 - by0 - 1) & (px >= 0) & (px < bx1 - bx0 - 1)
    if not inbox.any():
        return out
    r0, r1 = np.flatnonzero(inbox.any(axis=1))[[0, -1]]
    c0, c1 = np.flatnonzero(inbox.any(axis=0))[[0, -1]]
    sub = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    PY, PX = py[sub], px[sub]
    m = lambda v: v * px_per_m  # noqa: E731 - metres -> photo px
    # (a groove a figure is to show is at least so deep and so wide on the
    # print itself: sized on the person, a brow's hairs, the lines between
    # teeth came out a tenth of a millimetre on a 15 cm figure -- nothing)
    pmm = lambda mm: mm * (print_px_per_mm or 0.0)  # noqa: E731 - print mm -> photo px
    cell = 1.0 / scale  # photo px per grid cell

    def sample(mask):
        if mask is None:
            return np.zeros(PY.shape)
        a = map_coordinates(np.asarray(mask, np.float32), [PY, PX], order=1, mode="constant")
        return gaussian_filter(a, 0.7)

    def line(points, width_px):
        """A soft line through `points` (photo px, the reading's box frame)."""
        acc = np.zeros(PY.shape)
        pts = np.asarray(points, np.float64)
        for a, b_ in zip(pts[:-1], pts[1:]):
            n = max(int(np.hypot(*(b_ - a)) / max(cell, 0.5)) + 1, 2)
            for t in np.linspace(0, 1, n):
                x, y = a * (1 - t) + b_ * t
                acc = np.maximum(acc, np.exp(-((PX - x) ** 2 + (PY - y) ** 2) / (2 * max(width_px, 0.5) ** 2)))
        return acc

    f = np.asarray(face_mesh)[:, :2] - [bx0, by0]
    rel = np.zeros(PY.shape)
    # eyes
    from .face_parts import EYE

    for s_, e in reading.get("eyes", {}).items():
        op = sample(e["opening"])
        if e["open"]:
            d = distance_transform_edt(e["opening"]) if e["opening"].any() else None
            if d is not None and d.max() > 0:
                dome = map_coordinates(np.sqrt(np.clip(d / d.max(), 0, 1)).astype(np.float32), [PY, PX], order=1, mode="constant")
                rel -= op * m(_EYE_INSET) * (1.0 - dome)
            ix, iy = e["iris"][0] - bx0, e["iris"][1] - by0
            ri = max(e["iris_r_px"], 1.0)
            rel -= op * m(_IRIS_DEPTH) * np.clip(1.0 - np.hypot(PX - ix, PY - iy) / ri, 0, 1) ** 0.5
            upper = f[EYE[s_][:9]]
            eye_w = float(np.ptp(upper[:, 0]))
            rel -= m(_LID_CREASE) * line(upper - [0, 0.18 * eye_w], 0.02 * eye_w) * (1 - op)
            # the upper lid's rim, standing over the eyeball along the opening's top
            rel += m(_LID_RIM) * line(upper, 0.015 * eye_w) * (1 - op)
        else:
            lid = f[EYE[s_][:9]]
            rel -= m(_LID_LINE) * line(lid, 0.02 * float(np.ptp(lid[:, 0])))
    # eyebrows
    # (stood up as a whole, its edge crisp, shown up for a figure: on a person
    # brows show by their colour, on a one-colour print only by their shape;
    # with a hair grain along them, so they read as brows, not a band)
    for b_ in reading.get("brows", {}).values():
        bm = sample(b_["mask"])
        # (its outline smooth: the mask's pixel staircase stood as a jagged
        # edge along the brow's foot)
        bm_s = gaussian_filter(bm, max(1.0, pmm(0.25) / max(cell, 1e-6)))
        bm = np.clip((bm_s - 0.3) / 0.4, 0.0, 1.0)
        bm = bm * bm * (3.0 - 2.0 * bm)
        ys_, xs_ = np.nonzero(np.asarray(b_["mask"]))
        grain = 1.0
        if len(xs_) > 10:
            P_ = np.column_stack([xs_, ys_]).astype(np.float64)
            _, _, vt = np.linalg.svd(P_ - P_.mean(axis=0), full_matrices=False)
            ax = vt[0] * (1.0 if vt[0][0] >= 0 else -1.0)  # along the brow
            ang = np.deg2rad(_BROW_GRAIN_ANGLE)
            dirn = np.array([ax[0] * np.cos(ang) - ax[1] * np.sin(ang), ax[0] * np.sin(ang) + ax[1] * np.cos(ang)])
            across = PX * -dirn[1] + PY * dirn[0]  # distance across the hairs' run (reading px)
            step_ = max(m(_BROW_GRAIN_STEP), pmm(2.0 * PRINT_GROOVE_W_MM), 1.5)
            g_ = min(max(_BROW_GRAIN, pmm(PRINT_GROOVE_DEPTH_MM) / max(m(_BROW_HEIGHT), 1e-6)), 0.8)
            grain = 1.0 - g_ * 0.5 * (1.0 + np.cos(2.0 * np.pi * across / step_))
        rel += max(m(_BROW_HEIGHT), pmm(PRINT_GROOVE_DEPTH_MM) / 0.8) * gaussian_filter(bm, 0.7) * grain
    # nostrils
    for n_ in reading.get("nostrils", []):
        nx_, ny_ = n_["c"][0] - bx0, n_["c"][1] - by0
        rel -= m(_NOSTRIL_DEPTH) * np.exp(-((PX - nx_) ** 2 + (PY - ny_) ** 2) / (2 * max(n_["r"], 1.0) ** 2))
    # lips and mouth
    from .face_parts import INNER_LIPS, OUTER_LIPS

    mouth = reading.get("mouth", {})
    size = reading.get("size_px", 100.0)
    if not reading.get("masked"):
        rel += m(_LIP_RIM) * line(np.vstack([f[OUTER_LIPS], f[OUTER_LIPS[:1]]]), 0.006 * size)
        # the lips' own fullness: a dome over the photo's outer lip outline
        from skimage.measure import points_in_poly

        lips = points_in_poly(np.column_stack([PX.ravel(), PY.ravel()]), f[OUTER_LIPS]).reshape(PY.shape)
        if lips.any():
            dl = distance_transform_edt(lips)
            rel += m(_LIP_FULL) * gaussian_filter(np.sqrt(np.clip(dl / max(dl.max(), 1e-6), 0, 1)), 0.8)
        # the chin: its pad (round the point between the lower lip and the
        # chin's bottom) and the groove above it, placed by the face points
        chin, sulcus = f[152], f[18]
        pad_c = chin * 0.6 + sulcus * 0.4
        r_c = max(0.35 * float(np.hypot(*(chin - sulcus))), 1.0)
        rel += m(_CHIN_PROUD) * np.exp(-((PX - pad_c[0]) ** 2 + (PY - pad_c[1]) ** 2) / (2 * r_c ** 2))
        rel -= 0.5 * m(_CHIN_PROUD) * np.exp(-((PX - sulcus[0]) ** 2 + (PY - sulcus[1]) ** 2) / (2 * (0.5 * r_c) ** 2))
        if mouth.get("open") and "inside" in mouth:
            inside = sample(mouth["inside"])
            depth = np.full(PY.shape, m(_MOUTH_DEPTH))
            depth = np.where(sample(mouth["tongue"]) > 0.5, m(_TONGUE_INSET), depth)
            depth = np.where(sample(mouth["teeth"]) > 0.5, m(_TEETH_INSET), depth)
            rel -= inside * gaussian_filter(depth, 0.8)
            if "teeth_lines" in mouth:
                # the lines between the teeth: shallow, each tooth its own
                sig_ = max(0.6, pmm(PRINT_GROOVE_W_MM) / 2.355 / max(cell, 1e-6))
                tl = gaussian_filter(sample(mouth["teeth_lines"]), sig_)
                tl = np.clip(tl / max(float(tl.max()), 1e-6), 0.0, 1.0) if tl.max() > 0 else tl
                rel -= inside * max(m(_TOOTH_LINE), pmm(PRINT_GROOVE_DEPTH_MM)) * tl
        else:
            rel -= m(_MOUTH_LINE) * line(f[INNER_LIPS[:11]], 0.006 * size)
    # scars
    for sc in reading.get("scars", []):
        pts = np.asarray(sc["points"]) - [bx0, by0]
        if len(pts) > 1:
            order = np.argsort(pts[:, 0] if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else pts[:, 1])
            rel -= m(_SCAR_DEPTH) * line(pts[order][:: max(1, len(pts) // 40)], max(sc["width_px"] / 2, 1.0))
    # facial hair
    fh = reading.get("facial_hair", {})
    for k in ("beard", "moustache"):
        if fh.get(k) is not None:
            rel += m(_BEARD_THICK) * gaussian_filter(sample(fh[k]), 1.5)
    if fh.get("stubble", 0) > 0:
        from .face_parts import _poly

        jaw = sample(_poly((by1 - by0, bx1 - bx0), f[[172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 291, 0, 61]]))
        grain = gaussian_filter(np.random.default_rng(2).standard_normal(PY.shape), 0.6)
        rel += fh["stubble"] * m(_STUBBLE_DEPTH) * jaw * grain / (grain.std() + 1e-9)
    out[sub] = np.where(inbox[sub], rel, 0.0)
    return out


def _clothes_relief(reading: dict, front: np.ndarray, cells: np.ndarray, y0: int, x0: int, scale: float, pad: int,
                    px_per_m: float) -> np.ndarray:
    """Offsets (photo px, toward the camera) stepping each garment over the
    one under it where they meet (a jacket's edge over the shirt, a tie on
    it, a belt over trousers, a scarf over everything), easing back to the
    garment's own surface away from the edge; fading where the surface
    turns side-on."""
    from .clothes import GARMENTS, NAMES, layer_map, worn_ids

    gh, gw = cells.shape
    labels = reading["labels"]
    r = np.clip(np.rint(y0 + (np.arange(gh) - pad) / scale).astype(int), 0, labels.shape[0] - 1)
    c = np.clip(np.rint(x0 + (np.arange(gw) - pad) / scale).astype(int), 0, labels.shape[1] - 1)
    lab_g = labels[r[:, None], c[None, :]]
    L = layer_map({"labels": lab_g})
    named = np.isfinite(L) & cells
    if not named.any():
        return np.zeros((gh, gw))
    idx = distance_transform_edt(~named, return_distances=False, return_indices=True)
    Lf = np.where(named, L, L[idx[0], idx[1]])
    Lf = np.nan_to_num(Lf, nan=0.0)
    span = max(_CLOTH_STEP_SPAN * px_per_m * scale, 1.0)
    w = gaussian_filter(cells.astype(np.float64), span) + 1e-6
    local = gaussian_filter(np.where(cells, Lf, 0.0), span) / w
    step = _CLOTH_LAYER * px_per_m * (Lf - local)
    garment = np.isin(lab_g, worn_ids())
    for k, t in (("tie", _TIE_THICK), ("belt", _BELT_THICK), ("watch", _WATCH_THICK), ("bracelet", _BRACELET_THICK)):
        m_ = lab_g == NAMES.index(k) + 1
        if m_.any():
            step = step + t * px_per_m * gaussian_filter(m_.astype(np.float64), 1.0)
    gy, gx = np.gradient(front)
    facing = 1.0 / np.sqrt(1.0 + (np.hypot(gy, gx) * scale) ** 2)
    facing = np.clip((facing - _REF_RELIEF_FACING[0]) / (_REF_RELIEF_FACING[1] - _REF_RELIEF_FACING[0]), 0.0, 1.0)
    near = gaussian_filter(garment.astype(np.float64), 1.0)  # garments and their edges, not bare skin far off
    return np.where(cells, gaussian_filter(step, 0.8) * facing * np.clip(near * 3, 0, 1), 0.0)


def _bony_landmarks(reference: dict, pose: Optional[dict], skin: np.ndarray, neck_bare: bool, y0: int, x0: int, scale: float,
                    pad: int):
    """(front, back) offsets (photo px, outward) for the bones that show
    under bare skin: the collarbones' ridges and the notch between them,
    kneecaps, the ankles' knobs, the knuckles, and the knob at the base of
    the nape. Placed from the posed reference body's joints (and the
    photo's own hand points); only where the photo shows skin there."""
    gh, gw = skin.shape
    ppm = float(reference["px_per_m"])
    J = {k: np.asarray(v[:2], np.float64) for k, v in reference.get("joints_px", {}).items()}
    rr, cc = np.mgrid[0:gh, 0:gw]
    py, px = y0 + (rr - pad) / scale, x0 + (cc - pad) / scale  # each cell in photo px
    front, back = np.zeros((gh, gw)), np.zeros((gh, gw))

    def blob(c, r_m):
        return np.exp(-((px - c[0]) ** 2 + (py - c[1]) ** 2) / (2 * (r_m * ppm) ** 2))

    def ridge(a, b, r_m):
        ab = b - a
        t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        d2 = (px - (a[0] + t * ab[0])) ** 2 + (py - (a[1] + t * ab[1])) ** 2
        return np.exp(-d2 / (2 * (r_m * ppm) ** 2)) * np.sin(np.pi * t) ** 0.5

    if "left_shoulder" in J and "right_shoulder" in J:
        ls, rs = J["left_shoulder"], J["right_shoulder"]
        sw = float(np.linalg.norm(ls - rs))
        notch = (ls + rs) / 2 + np.array([0.0, -0.02 * sw])
        for sh in (ls, rs):
            front += _CLAVICLE * ppm * ridge(notch + 0.12 * (sh - notch), notch + 0.85 * (sh - notch), 0.008)
        front -= _STERNAL_NOTCH * ppm * blob(notch + np.array([0.0, 0.03 * sw]), 0.012)
        if neck_bare:
            back += _C7 * ppm * blob(notch + np.array([0.0, -0.08 * sw]), 0.015)
    for sd in ("left", "right"):
        if f"{sd}_knee" in J:
            front += _PATELLA * ppm * blob(J[f"{sd}_knee"], 0.025)
        if f"{sd}_ankle" in J:
            for dx in (-1, 1):
                front += _MALLEOLUS * ppm * blob(J[f"{sd}_ankle"] + np.array([dx * 0.033 * ppm, 0.0]), 0.01)
        if pose and all(f"{sd}_{k}" in pose for k in ("wrist", "index", "pinky")):
            w_, i_, p_ = (np.asarray(pose[f"{sd}_{k}"], np.float64) for k in ("wrist", "index", "pinky"))
            for t in np.linspace(0, 1, 4):  # index to little finger, at the knuckles
                front += _KNUCKLE * ppm * blob(w_ + 0.75 * ((i_ * (1 - t) + p_ * t) - w_), 0.005)
    soft = gaussian_filter(skin.astype(np.float64), 1.0)
    return front * soft, back


def _wear_on_face(front: np.ndarray, grid_mask: np.ndarray, parts_g: np.ndarray, scale: float, px_per_m: float) -> np.ndarray:
    """What's worn on the face, on the front surface (photo px, toward the
    camera): a face mask is cloth stretched from the nose over the mouth to
    the chin (a ball of _MASK_SPAN rolled over the face: no lips or hollows
    under it); glasses are a smooth lens over the eye sockets (a ball of
    _GLASSES_SPAN) with a raised frame round them."""
    from scipy.ndimage import binary_dilation, binary_erosion, grey_closing

    from .head_parts import GLASSES, MASK

    out = front.copy()
    for label, span, rim in ((MASK, _MASK_SPAN, 0.0), (GLASSES, _GLASSES_SPAN, _GLASSES_FRAME)):
        on = (parts_g == label) & grid_mask
        if on.sum() < 6:
            continue
        n = max(int(round(span * px_per_m * scale)), 1)
        ys, xs = np.nonzero(on)
        r0, r1 = max(ys.min() - 2 * n, 0), min(ys.max() + 2 * n + 1, front.shape[0])
        c0, c1 = max(xs.min() - 2 * n, 0), min(xs.max() + 2 * n + 1, front.shape[1])
        crop = (slice(r0, r1), slice(c0, c1))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        d = np.hypot(yy, xx) / scale
        R = span * px_per_m
        ball = d <= R
        cap = np.where(ball, np.sqrt(np.maximum(R * R - d * d, 0.0)) - R, 0.0)
        f = np.where(grid_mask[crop], out[crop], float(out[crop][grid_mask[crop]].min()) - 4 * R)
        closed = grey_closing(f, footprint=ball, structure=cap)
        cloth = _masked_blur(np.maximum(closed, f), grid_mask[crop], 1.0)
        w = np.clip(gaussian_filter(on[crop].astype(np.float32), 1.0) * 1.5, 0.0, 1.0)
        new = out[crop] + w * (np.maximum(cloth, out[crop]) - out[crop])
        if rim > 0:
            band = on[crop] & ~binary_erosion(on[crop], iterations=1)
            band |= binary_dilation(band, iterations=1) & on[crop]
            new = new + gaussian_filter(band.astype(np.float32), 0.7) * rim * px_per_m
        out[crop] = np.where(grid_mask[crop], new, out[crop])
    return out


def _neck_wear(front, back, grid_mask, photo_cells, parts_g, cov, face_mesh, neck, y0, x0, scale, pad, px_per_m):
    """The neck and the collar round it, in the build grid (depth in photo
    px, z toward the camera).

    The neck is the reference body's (`neck`, _neck_geometry): its width,
    and from under the jaw to its base -- where the body model's neck ends,
    not the shoulder joints, which can sit well down the chest, nor the
    outline there, which can already be the shoulders. A collar -- a shirt or polo
    collar, a crew or V neckline, a roll neck, a scarf -- is a band all the
    way round it:
    - at the front, where the photo shows clothes against the neck, it
      stands off by its thickness with a crisp top edge (a shirt collar's
      points lying on the chest);
    - at the sides its top is where the clothes first touch the neck's
      sides in the photo;
    - round the back, which the photo can't show, it carries on behind the
      neck and rides up toward the middle by _NECK_COLLAR_RISE (a collar
      sits higher on the back of the neck), with the same crisp top edge,
      fading into the back of the shirt or jacket below the neck's base.
    It's _NECK_COLLAR thick, or as far out as the photo shows the covering
    past the neck (a scarf, a roll neck: up to _NECK_WEAR_MAX). With no
    clothes against the neck there's no collar. A necklace (an accessory
    on bare neck or chest skin) is a slight raised band."""
    from scipy.ndimage import binary_dilation

    from .head_parts import ACCESSORY, BODY_SKIN, CLOTHES

    gh, gw = grid_mask.shape
    fm = np.asarray(face_mesh, dtype=np.float64)[:, :2]
    to_r = lambda y: (y - y0) * scale + pad  # noqa: E731
    to_c = lambda x: (x - x0) * scale + pad  # noqa: E731
    if neck is None:
        return front, back
    chin_r = int(round(to_r(fm[152, 1])))
    base_r = int(round(to_r(neck["base_y"])))
    below_r = int(round(to_r(neck["base_y"] + _NECK_COLLAR_BELOW * px_per_m)))
    c_face = int(round(to_c((fm[234, 0] + fm[454, 0]) / 2.0)))
    if not 0 < chin_r < below_r < gh or not 0 <= c_face < gw:
        return front, back
    rows = np.arange(gh)[:, None]
    cols = np.arange(gw)[None, :]
    F, B = front.copy(), back.copy()
    # a necklace: an accessory lying on bare skin below the chin
    lace = (parts_g == ACCESSORY) & photo_cells & (rows > chin_r) & binary_dilation(parts_g == BODY_SKIN, iterations=2)
    if lace.any():
        F = F + gaussian_filter(lace.astype(np.float32), 0.8) * _NECK_LACE * px_per_m
    # the neck: the body model's own, across
    cn = to_c(neck["cx"])
    half = max(neck["half"] * scale, 2.0)
    nl, nr = cn - half, cn + half
    # clothes against the neck: in its columns (and a little out, a collar
    # stands off it), from just above the chin line down past its base
    t_c = _NECK_COLLAR * px_per_m
    out_c = t_c * scale + 2
    zone = (np.abs(cols - cn) <= half + out_c) & (rows >= chin_r - half) & (rows <= below_r)
    wear = (parts_g == CLOTHES) & photo_cells & zone
    if wear.sum() < 4:
        return F, B
    # its top at the sides: the first row the clothes touch each side
    tops = []
    for side in (-1, 1):
        edge = (np.abs(cols - (cn + side * half)) <= out_c) & zone
        rr = np.flatnonzero((wear & edge).any(axis=1))
        if len(rr):
            tops.append(rr.min())
    if not tops:
        return F, B
    top_side = float(np.median(tops))
    if _NECK_DEBUG is not None:
        _NECK_DEBUG.update(chin_r=chin_r, base_r=base_r, below_r=below_r, top_side=top_side, half=half, cn=cn, tops=tops)
    # how thick: a collar, or as far past the neck as the photo shows it
    reach = 0.0
    for r in range(int(top_side), min(base_r, gh - 1) + 1):
        w = np.flatnonzero(wear[r])
        if len(w):
            reach = max(reach, (max(nl - w.min(), w.max() - nr, 0) - 1) / scale)
    t = float(np.clip(reach, t_c, _NECK_WEAR_MAX * px_per_m))
    # fade into the torso's clothes below the neck's base
    fade = np.clip((below_r - rows) / max(below_r - base_r, 1.0), 0.0, 1.0)
    # the front: the photo's collar, a crisp top edge where its label starts
    front_c = wear & grid_mask & (np.abs(cols - cn) <= half + out_c)
    F = np.where(front_c, F + t * fade, F)
    # round the back: from the sides' top, riding up toward the middle
    rise = _NECK_COLLAR_RISE * px_per_m * scale
    dx = np.clip((cols - cn) / (half + out_c), -1.0, 1.0)
    top_back = top_side - rise * (1.0 - dx * dx)
    band = (np.abs(cols - cn) <= half + out_c) & grid_mask
    k = np.clip(rows - top_back + 0.5, 0.0, 1.0) * fade * band  # crisp top (half a cell, no steps)
    B = B - t * k
    # its sides: the neck's edge columns stand out by the same thickness
    # front and back (already in the outline where the photo shows it)
    B = np.minimum(B, F - 1.0 / scale)
    # along the band (not across its top edge) the labels' raggedness
    # smoothed out
    ring = binary_dilation(front_c | (k > 0), iterations=1) & grid_mask
    if ring.any():
        F = np.where(ring, _masked_blur(F, ring, 0.8), F)
        B = np.where(ring, _masked_blur(B, ring, 0.8), B)
        B = np.minimum(B, F - 1.0 / scale)
    return F, B


def _neck_geometry(reference: dict) -> Optional[dict]:
    """The reference body's neck in the photo (px): where it ends at the
    front ("base_y"), its middle across ("cx") and half its width ("half")."""
    names = [str(n) for n in reference.get("part_names", [])]
    if "neck" not in names or reference.get("part_w") is None:
        return None
    V = np.asarray(reference["vertices"])
    neck = np.asarray(reference["part_w"])[:, names.index("neck")] > 0.5
    if neck.sum() < 10:
        return None
    x = V[neck, 0]
    return {"base_y": float(np.percentile(V[neck, 1], 90)), "cx": float(np.median(x)),
            "half": float((np.percentile(x, 95) - np.percentile(x, 5)) / 2.0)}


def _parts_on_grid(labels: np.ndarray, grid_shape, y0: int, x0: int, scale: float, pad: int) -> np.ndarray:
    """head_parts labels (photo px) at each build-grid cell (nearest)."""
    gh, gw = grid_shape
    r = np.clip(np.rint(y0 + (np.arange(gh) - pad) / scale).astype(int), 0, labels.shape[0] - 1)
    c = np.clip(np.rint(x0 + (np.arange(gw) - pad) / scale).astype(int), 0, labels.shape[1] - 1)
    return labels[r[:, None], c[None, :]]


def _face_relief(face_mesh: np.ndarray, grid_shape, y0: int, x0: int, scale: float, pad: int) -> np.ndarray:
    """Front-surface offset (photo pixels) carrying the face's own shape --
    brow, eye sockets, nose, lips, cheekbones, jaw -- from the 478 face
    landmarks (face_features.detect_face_mesh). The face's overall
    curvature is removed first with a quadratic fit, since the inflated
    head already supplies a rounded front; only what's left over (the
    features themselves) is added, interpolated across the face and faded
    out toward its edge. Zero outside the face."""
    # The last 10 of MediaPipe's 478 points are irises; their depth isn't
    # reliable and spikes the eyes.
    face_mesh = face_mesh[:_FACE_SURFACE_POINTS]
    rows = (face_mesh[:, 1] - y0) * scale + pad
    cols = (face_mesh[:, 0] - x0) * scale + pad
    toward_camera = -face_mesh[:, 2]
    basis = np.stack([np.ones_like(rows), rows, cols, rows**2, rows * cols, cols**2], axis=1)
    coef, *_ = np.linalg.lstsq(basis, toward_camera, rcond=None)
    features = toward_camera - basis @ coef

    gh, gw = grid_shape
    grid_rows, grid_cols = np.mgrid[0:gh, 0:gw]
    relief = griddata((rows, cols), features, (grid_rows, grid_cols), method="cubic")
    inside = ~np.isnan(relief)
    if not inside.any():
        return np.zeros(grid_shape, dtype=np.float32)
    face_width = float(cols.max() - cols.min())
    # Interpolating ~470 scattered points leaves the surface crumpled
    # between them (and a hard crease where the lips meet); a blur scaled
    # to the face keeps the features and drops that noise. Normalised so
    # the face's edge doesn't get dragged toward zero.
    sigma = max(face_width * _FACE_RELIEF_SMOOTH_FRAC, 0.5)
    filled = np.where(inside, relief, 0.0)
    weight = gaussian_filter(inside.astype(np.float64), sigma)
    relief = np.divide(gaussian_filter(filled, sigma), weight, out=np.zeros_like(filled), where=weight > 1e-6)
    fade = np.clip(distance_transform_edt(inside) / max(face_width * _FACE_RELIEF_FADE_FRAC, 1.0), 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    return np.where(inside, relief * fade, 0.0).astype(np.float32)


def _inflation_height(mask: np.ndarray) -> np.ndarray:
    """Half-thickness (in grid cells) of a smooth, rounded inflation of
    `mask`: solves the Poisson equation lap(h) = -1 inside the outline with
    h = 0 outside, then returns sqrt(2h). For a strip of half-width a that
    is exactly a half-circle cross-section of depth a; for a disc it is a
    flattened dome. The solution is smooth everywhere, so it has no seams
    where one part of the outline (an arm) joins another (the torso).
    `mask` must have at least one False cell on every side (callers pad)."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[ys, xs] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour = index[ys + dy, xs + dx]
        inside = neighbour >= 0
        rows.append(np.arange(n)[inside])
        cols.append(neighbour[inside])
        vals.append(np.full(int(inside.sum()), -1.0))
    laplacian = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    h = np.zeros(mask.shape, dtype=np.float64)
    h[ys, xs] = spsolve(laplacian, np.ones(n))
    return np.sqrt(2.0 * np.maximum(h, 0.0)).astype(np.float32)


def build_silhouette_relief_mesh(
    alpha: np.ndarray,
    target_height_mm: float = 150.0,
    grid_size: int = 360,
    depth_scale: float = 0.75,
    depth_rgb: Optional[np.ndarray] = None,
    depth_strength: float = 0.5,
    animal_frac: float = 0.0,
    limb_depth_scale: float = 0.9,
    bust_mode: bool = False,
    face_mesh: Optional[np.ndarray] = None,
    profile: str = "poisson",
    edge: str = "round",
    depth_map: Optional[np.ndarray] = None,
    layer_rescue: str = "auto",
    face_strength: Optional[float] = None,
    smooth: bool = True,
    pose: Optional[dict] = None,
    reference: Optional[dict] = None,
) -> BodyMesh:
    """Combination Lab switches (defaults are the current production build):
    `profile` -- cross-section shape: "tent" (9 Oct: straight ramp from the
    edge), "row_ellipse" (11 Oct: per-row quarter-ellipse; horizontal
    ripples), "poisson" (12 Oct: one smooth 2D field). `edge` -- "hard"
    (mask cut + thickness floor; seam around the outline) or "round".
    `depth_map` -- a precomputed estimate_relative_depth(rgb) for the same
    photo, used instead of recomputing from depth_rgb. `layer_rescue` --
    "auto" (animals only), "off", or "on" (forced, also on people).
    `face_strength` -- landmark face relief multiplier; None means 0.45
    with depth, 1.0 without. `smooth` -- Taubin pass after marching cubes.

    `bust_mode`: skip _detect_crop_and_extend's synthetic leg
    completion -- for a photo that was always meant to show only the
    upper body (a bust/headshot-style portrait), not a standing figure
    cropped by accident. Without this, a photo cut off at the waist/
    chest gets generic tapered legs grafted on by default (see that
    function's own docstring for why); with it, the mesh ends exactly
    where the photo's own silhouette does -- closed the same way every
    other boundary in this mesh already is (the top of the head, a
    hand, a foot), not a special case. The pipeline still adds a flat
    base disk underneath either way (see mesh_export.add_base_disk),
    so a bust still has something to physically stand on when printed.

    `alpha`: a float [0,1] or boolean (H, W) mask -- the same soft
    cutout alpha detect_person_alpha produces (and the webapp's
    /jobs/{id}/cutout.png serves), thresholded at 0.5 to a hard boundary
    for a clean, closed mesh.

    `animal_frac`: from person_segmenter.detect_person_alpha_with_source
    -- gates _detect_crop_and_extend's leg-completion to subjects the
    animal channel doesn't itself claim (see that function's own
    docstring for why this, not "1 - person channel", is the signal that
    actually works). Leave at the default 0.0 to allow completion
    unconditionally -- e.g. when the caller already knows the subject is
    human some other way.

    `grid_size`: raised from this function's original 220. NOTE: this
    was a wrong diagnosis for the specific problem it was raised to fix
    (a dog's head/ears reading as a featureless blob) -- confirmed
    directly, not assumed: inspecting the alpha mask at full NATIVE
    resolution around the head showed the same smooth, undifferentiated
    boundary already present before any downsampling at all. That head
    was photographed in a 3/4 view, where the eyes/snout/ears are
    conveyed by color and shading, not by any silhouette discontinuity
    against the background -- no grid resolution recovers detail the
    boundary itself never had. (See depth_rgb below for what actually
    helps that case.) Still kept at 360 since it's cheap and does help
    ordinary edge precision generally -- just not a fix for that
    specific failure mode.

    `depth_scale`/`limb_depth_scale`: real front-to-back depth isn't a
    fixed fraction of local side-to-side width everywhere on a body --
    a torso is flattened oval (depth noticeably less than width), but a
    limb is closer to round (depth close to width). A single flat
    multiplier can't represent both at once, and was confirmed directly
    to get the ratio backwards: measuring a real exported mesh's own
    depth/width at different heights (excluding the inter-leg gap and a
    stray gesturing hand from the measurement, both of which silently
    inflated "width" the first time this was checked) showed limbs
    coming out FLATTER than the torso at one flat depth_scale=0.5
    (thigh 0.49, torso 0.39-0.49), backwards from real anatomy where a
    thigh should be rounder than a torso, not flatter.
    `depth_scale` now applies where the local silhouette width (via the
    same distance-transform that sets half_thickness) is at its widest
    for this subject (torso-like), `limb_depth_scale` where it's at its
    thinnest (limb-like), linearly blended by where each pixel's own
    width falls on the subject's own width spectrum -- the same "max/min
    give a spectrum" idea tried earlier for the depth-bias term (where
    it was confirmed NOT to help, capped at sub-mm effects by this very
    half_thickness ceiling) works here instead, because this changes the
    actual depth budget per body part rather than nudging within a fixed
    one. Confirmed directly: on a real photo, thigh depth/width went
    0.49 -> 0.74 (limb_depth_scale=0.9) with the torso essentially
    unchanged (0.39 -> 0.39) -- steeper tunings (limb_depth_scale up to
    1.3 with a power-law remap of the width spectrum) were tried and
    measured WORSE once the gap/hand measurement bug above was fixed, so
    this linear version is the one kept, not a starting point to push
    further.

    `depth_rgb`/`depth_strength`: optional real per-pixel depth (Depth
    Anything V2 Small, see depth_source.py -- same optional extra
    build_body_mesh's own depth_rgb uses). Pure silhouette extrusion has
    no way to know a region is angled toward the camera rather than
    flat-on -- confirmed directly as a real limitation, not a guess: a
    photographed head turned toward the camera came out just as
    symmetric as a straight-on torso, because distance-to-edge alone
    carries no orientation information. When given, biases the FRONT
    surface only (same asymmetric-front/untouched-back design as
    procedural_body._sculpt_front_surface_from_depth, and the same
    sign convention, confirmed directly against a real photo -- a
    known subject in front of a background scored higher raw depth than
    the background did, i.e. higher = closer). Leave None for the
    original pure-silhouette behavior (depth bias is additive on top of
    it, not a replacement).

    Verified directly, not assumed, on two real photos before this was
    trusted to feed a full 3D build: a human portrait's front surface
    came back with genuinely recognizable bulged structure (nose, eye
    sockets, mouth, cheek volume all visible in a shaded render of the
    front surface alone). The SAME labrador photo that originally
    motivated this parameter -- a head turned toward the camera -- did
    NOT, at first: its front surface showed only a faint crease line
    roughly where the mouth/ear shading falls, not an actual protruding
    snout volume. Root-caused, not just retried: the depth map DOES
    correctly know the dog's head is the closest part overall, but
    normalizing against the whole subject's std compressed the much
    smaller internal relief WITHIN the head (snout vs. eye socket vs.
    ear) into a sliver of a range the head-vs-legs difference dominates.

    When animal_frac says this is confidently an animal subject (see
    _layered_detail_rescue), an additional per-layer rescue term kicks
    in: the subject's own depth values are clustered into discrete
    layers (ear, head, body, legs), and each layer's own internal
    detail is rescued relative to its own scale, not swamped by the
    others -- confirmed to turn the labrador's flat crease into a
    genuinely contoured head/ear shape. This is deliberately NOT
    applied to human subjects: tried it there directly and it was a
    clear regression (a clean, recognizable face became noisy and
    mask-like, even after several rounds of tuning) -- a human face is
    closer to one continuous surface, so per-layer clustering mostly
    carves up smooth lighting/skin gradients into arbitrary bands and
    amplifies noise, not real geometry. So: trust the plain depth bias
    for human portraits (it already does real work there); trust the
    animal_frac-gated layered rescue for animal subjects with real
    physically-distinct parts; for either, this is depth-model-derived
    detail layered onto the silhouette's own shape, not a substitute
    for a dedicated landmark/geometry detector.

    No pose, no color, no texture -- purely the cutout's own shape swept
    into a rounded volume. Render/texture it yourself from here."""
    mask = alpha > 0.5
    if not mask.any():
        raise ValueError("Empty mask -- nothing to build a shape from.")
    orig_h, orig_w = mask.shape  # real photo dimensions, saved before any synthetic extension below
    # people only: on an animal the pose detector can invent a human skeleton
    if reference is not None and (not pose or animal_frac > _CROP_MAX_ANIMAL_FRAC):
        reference = None
    extent = None
    if reference is not None:
        J = reference["joints_px"]
        hip_y = (J["left_hip"][1] + J["right_hip"][1]) / 2.0
        knee_y = (J["left_knee"][1] + J["right_knee"][1]) / 2.0
        mask, extent = _person_extent(mask, hip_y, knee_y, bust_mode)
        bust_mode = True  # a person ends where the photo (or the hip line) does
    elif pose and animal_frac <= _CROP_MAX_ANIMAL_FRAC and "left_shoulder" in pose and "right_shoulder" in pose:
        # a person without a matched reference body: the same rule from the
        # photo's own pose points (knees not found: estimated from the trunk)
        hip_y = knee_y = None
        if "left_hip" in pose and "right_hip" in pose:
            hip_y = (pose["left_hip"][1] + pose["right_hip"][1]) / 2.0
            sh_y = (pose["left_shoulder"][1] + pose["right_shoulder"][1]) / 2.0
            knee_y = ((pose["left_knee"][1] + pose["right_knee"][1]) / 2.0 if "left_knee" in pose and "right_knee" in pose
                      else hip_y + 0.8 * (hip_y - sh_y))
        mask, extent = _person_extent(mask, hip_y, knee_y, bust_mode)
        bust_mode = True
    elif not bust_mode:
        mask = _detect_crop_and_extend(mask, animal_frac=animal_frac)

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
    # Per-pixel depth_scale: limb_depth_scale where this subject's own
    # silhouette is at its thinnest (limb-like), depth_scale where it's
    # at its widest (torso-like) -- see this function's own docstring
    # for why a single flat value gets limbs backwards.
    dist_subject = dist[grid_mask]
    dist_min = float(dist_subject.min()) if dist_subject.size else 0.0
    dist_max = float(dist_subject.max()) if dist_subject.size else 0.0
    width_frac = np.clip((dist - dist_min) / max(dist_max - dist_min, 1e-6), 0.0, 1.0)
    depth_scale_field = limb_depth_scale * (1.0 - width_frac) + depth_scale * width_frac

    # Round cross-sections from one smooth 2D field (profile="poisson"). A
    # per-row version (each row scaled to its own widest point) stacked
    # into horizontal ripples, since neighbouring rows' peaks differ, and
    # cut grooves where an arm's row-run merged into the torso's; it and
    # the original straight ramp stay available for the Combination Lab.
    # With pose points the arms become their own tubes and the body's
    # thickness comes from the outline WITHOUT the arm-only parts (see
    # _arm_layout); without them the whole outline is inflated as before.
    # people only: on an animal the pose detector can invent a human skeleton
    layout = (_arm_layout(pose, grid_mask, y0, x0, scale, pad, face_mesh=face_mesh, photo_mask=mask)
              if pose and animal_frac <= _CROP_MAX_ANIMAL_FRAC else None)
    body_mask = grid_mask & ~layout["arm_only"] if layout is not None else grid_mask
    body_dist = distance_transform_edt(body_mask).astype(np.float32) / scale if layout is not None else dist
    if profile == "tent":
        half_thickness_raw = body_dist * depth_scale_field
    elif profile == "row_ellipse":
        row_ridge = np.zeros_like(dist)
        for y in range(dist.shape[0]):
            row_mask = body_mask[y]
            if not row_mask.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(([0], row_mask.astype(np.int8), [0]))))
            for s, e in zip(edges[0::2], edges[1::2]):
                row_ridge[y, s:e] = body_dist[y, s:e].max()
        t = np.clip(np.divide(body_dist, row_ridge, out=np.zeros_like(dist), where=row_ridge > 1e-6), 0.0, 1.0)
        half_thickness_raw = row_ridge * np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0)) * depth_scale_field
    else:
        head_weight = _head_weight(face_mesh, grid_mask.shape, y0, x0, scale, pad) if face_mesh is not None else None
        cut_row = pad + round(h * scale) - 1  # last row of the subject
        if bust_mode and grid_mask[min(cut_row, gh - 1)].any():
            # A bust ends where the photo ends: mirror the outline below
            # the cut so the shape keeps its full thickness down to the
            # base instead of pinching to an edge there.
            upper = body_mask[: cut_row + 1]
            mirrored = np.vstack([upper, upper[::-1]])
            mirrored_head = None
            if head_weight is not None:
                mirrored_head = np.vstack([head_weight[: cut_row + 1], np.zeros_like(head_weight[: cut_row + 1])])
            thick = _proportioned_thickness(
                mirrored, _inflation_height(mirrored), depth_scale, limb_depth_scale, mirrored_head
            )[: cut_row + 1]
            half_thickness_raw = np.zeros_like(dist)
            half_thickness_raw[: cut_row + 1] = thick / scale
        else:
            half_thickness_raw = (
                _proportioned_thickness(body_mask, _inflation_height(body_mask), depth_scale, limb_depth_scale, head_weight)
                / scale
            )
    if layout is not None:
        sized = _scale_to_height(np.where(body_mask, half_thickness_raw, 0.0), body_mask, layout, face_mesh, pose)
        torso_front, torso_back = _human_torso(sized, body_mask, layout)
        front0, back0 = _with_arms(layout, grid_mask, torso_front, torso_back, max(limb_depth_scale, 0.95), dist)
    else:
        front0, back0 = half_thickness_raw, -half_thickness_raw
    ref = (_reference_surfaces(reference, grid_mask, y0, x0, scale, pad, face_mesh, photo_mask=mask)
           if reference is not None else None)
    parts = reference.get("parts") if reference is not None else None
    parts_g = (_parts_on_grid(parts["labels"], grid_mask.shape, y0, x0, scale, pad)
               if parts is not None and parts.get("labels") is not None else None)
    photo_cells = grid_mask
    hair_rel = None
    limbs3d = None
    if ref is not None:
        rf, rb, cov, ears3d, limbs3d = ref
        # where the reference runs just past the pixel outline, its own round
        # edge closes the solid; cutting it at the pixel steps left a wall
        # stepping row by row along the whole outline
        # Slivers of outline the reference falls just short of go too, and
        # only photo-only areas wide enough to matter (a jacket flaring past
        # the body) keep the outline-based thickness.
        from scipy.ndimage import binary_dilation, binary_opening

        # (how far past the outline: its pixel edge is ragged row by row at
        # hair and ears, and cutting the smooth body wherever it poked out
        # 2 cells left a little wall on every row -- grooves down the sides)
        slack = max(2.0, _REF_EDGE_SLACK * float(reference["px_per_m"]) * scale)
        ring = cov & (distance_transform_edt(~grid_mask) <= slack)
        wide_gap = binary_opening(grid_mask & ~cov, iterations=2)
        if ears3d is not None:
            # the photo's ears are the 3D ears' (no outline-based slab there)
            wide_gap &= ~binary_dilation(ears3d["cov"], iterations=3)
        # below the ankles the outline takes in the floor shadow around the
        # feet: there the body's own feet decide (no slabs of shadow)
        Jx = reference["joints_px"]
        ankle_row = (max(Jx["left_ankle"][1], Jx["right_ankle"][1]) - y0) * scale + pad
        wide_gap[np.arange(gh) > ankle_row] = False
        grid_mask = ring | wide_gap

        grid_mask[:pad], grid_mask[-pad:], grid_mask[:, :pad], grid_mask[:, -pad:] = False, False, False, False
        inside = grid_mask & cov
        # keep the outline-based body's depth position; the reference's shape
        shift = float(np.median(((front0 + back0) / 2.0 - (rf + rb) / 2.0)[inside]))
        rf, rb = rf + shift, rb + shift
        ref_shift = shift
        if limbs3d is not None:
            limbs3d["front"], limbs3d["back"] = limbs3d["front"] + shift, limbs3d["back"] + shift
        if extent == "torso":
            # the back hangs straight down from the small of the back: the
            # seat below it isn't part of a torso
            J = reference["joints_px"]
            sh_y = (J["left_shoulder"][1] + J["right_shoulder"][1]) / 2.0
            hp_y = (J["left_hip"][1] + J["right_hip"][1]) / 2.0
            cx = (J["left_hip"][0] + J["right_hip"][0]) / 2.0
            sw = abs(J["left_shoulder"][0] - J["right_shoulder"][0])
            to_r = lambda y: int(round((y - y0) * scale + pad))  # noqa: E731
            to_c = lambda x: int(round((x - x0) * scale + pad))  # noqa: E731
            c0, c1 = max(to_c(cx - 0.15 * sw), 0), min(to_c(cx + 0.15 * sw), gw)
            r0, r1 = max(to_r(sh_y + 0.45 * (hp_y - sh_y)), 0), min(to_r(hp_y), gh - 1)
            if r1 > r0 and c1 > c0:
                band = np.where(cov[r0:r1, c0:c1], rb[r0:r1, c0:c1], np.nan)
                rear = np.nanmin(band, axis=1)  # most rearward back point per row
                if np.isfinite(rear).any():
                    lumbar = r0 + int(np.nanargmax(rear))  # the small of the back
                    # that row's back, column by column (nearest covered column
                    # past its ends), carried straight down: below it nothing
                    # sits further back, blending in over a few rows
                    hipd = abs(J["left_hip"][0] - J["right_hip"][0])
                    span = np.zeros(gw, dtype=bool)
                    span[max(to_c(cx - 1.1 * hipd), 0):min(to_c(cx + 1.1 * hipd), gw)] = True
                    row = np.where(cov[lumbar] & span, rb[lumbar], np.nan)  # the torso's own back, not the arms'
                    have = np.flatnonzero(np.isfinite(row))
                    if len(have):
                        row = np.interp(np.arange(gw), have, row[have])
                        ramp = max(0.06 * gh, 4.0)
                        k = np.clip((np.arange(gh) - lumbar) / ramp, 0.0, 1.0)[:, None]
                        k = k * k * (3.0 - 2.0 * k)  # eased in, no kink
                        # the torso's columns only, fading out at its sides
                        k = k * gaussian_filter(span.astype(np.float32), 2.0)[None, :]
                        rb = np.where(k > 0, np.maximum(rb, rb + k * (row[None, :] - rb)), rb)
        r_mid, r_half = (rf + rb) / 2.0, np.maximum((rf - rb) / 2.0, 0.0)
        # cells it doesn't reach keep the outline-based thickness, centred on
        # the nearest reference depth
        idx = distance_transform_edt(~inside, return_distances=False, return_indices=True)
        gap = grid_mask & ~inside
        r_mid = np.where(gap, _masked_blur(r_mid[idx[0], idx[1]], grid_mask, 3.0), r_mid)
        # (as thick as the body next to them -- a jacket past the body is
        # unconfirmed depth, so bigger -- rounding off only near the outline)
        edge_taper = np.sqrt(np.clip(distance_transform_edt(grid_mask) / 6.0, 0.0, 1.0))
        inner = _masked_blur(r_half, inside & ~binary_dilation(~inside, iterations=4), 6.0)  # away from its own thin edge
        inner = np.where(inner > 0, inner, r_half)
        r_half = np.where(gap, _masked_blur(inner[idx[0], idx[1]], grid_mask, 3.0) * edge_taper, r_half)
        # smooth only where the reference meets the outline-based fill: at
        # the reference's own edge the thickness of the last cell is what
        # places the edge between cells (averaging it with the thicker cells
        # inside stepped the edge row by row), and smoothing the mid-depth
        # alone pushed the front's steps (chin over neck) into the back
        seam = np.clip(4.0 * gaussian_filter(gap.astype(np.float32), 1.5), 0.0, 1.0)
        r_mid = seam * _masked_blur(r_mid, grid_mask, 1.0) + (1.0 - seam) * r_mid
        r_half = seam * _masked_blur(r_half, grid_mask, 1.0) + (1.0 - seam) * r_half
        front0 = np.where(grid_mask, r_mid + r_half, 0.0)
        back0 = np.where(grid_mask, r_mid - r_half, 0.0)
        if parts_g is not None and face_mesh is not None:
            front_pre, back_pre = front0.copy(), back0.copy()
            front0, back0, grid_mask = _hair_and_hat(front0, back0, grid_mask, photo_cells, parts_g, inside | (cov & grid_mask),
                                                     face_mesh, y0, x0, scale, pad, float(reference["px_per_m"]))
            try:
                hair_rel = _hair_pattern(front0, back0, front_pre, back_pre, grid_mask, parts_g, parts.get("hair"), face_mesh,
                                         y0, x0, scale, pad, float(reference["px_per_m"]))
            except Exception as exc:  # noqa: BLE001 - the hair keeps its volume without the pattern
                print(f"Note: hair pattern skipped ({exc}).")
                hair_rel = None
            front0, back0 = _neck_wear(front0, back0, grid_mask, photo_cells, parts_g, inside | (cov & grid_mask), face_mesh,
                                       _neck_geometry(reference), y0, x0, scale, pad, float(reference["px_per_m"]))
            front0, back0 = np.where(grid_mask, front0, 0.0), np.where(grid_mask, back0, 0.0)
    # where a forearm or hand lies over the body: the body's own surface runs
    # on under it (the photo's depth and clothes there are the arm's)
    hands_on = _hands_on_body(parts, pose, grid_mask, y0, x0, scale, pad) if (limbs3d is not None and parts) else None
    limb_soft = None
    limb_detail = 0.0
    if limbs3d is not None:
        under = limbs3d["over"] | (hands_on if hands_on is not None else False)
        limb_soft = np.clip(gaussian_filter(under.astype(np.float64), 1.5) * 1.5, 0.0, 1.0)
    mid0 = np.where(grid_mask, (front0 + back0) / 2.0, 0.0)
    if edge == "hard":
        # Original edge: a soft floor so thickness never reaches 0, with the
        # mask cutting the volume off hard (see the field below).
        min_half_thickness = max(1.0 / scale, 0.5)
        half_thickness = np.where(grid_mask, np.sqrt(((front0 - back0) / 2.0) ** 2 + min_half_thickness**2), 0.0)
    else:
        half_thickness = np.where(grid_mask, (front0 - back0) / 2.0, 0.0)
        min_half_thickness = 1.0 / scale  # one grid cell, used by the depth bias floor below
    front0, back0 = mid0 + half_thickness, mid0 - half_thickness

    # mid/half_span describe each column's front and back surfaces (z
    # toward the camera). Without arms or depth this is the symmetric lens
    # front = +half_thickness, back = -half_thickness.
    mid = mid0
    half_span = half_thickness
    use_depth = depth_map is not None or depth_rgb is not None
    if use_depth:
        if depth_map is not None:
            depth = depth_map
        else:
            from .depth_source import estimate_relative_depth

            depth = estimate_relative_depth(depth_rgb)
        depth_h, depth_w = depth.shape
        # grid (row, col) -> original full-photo pixel coords: undo the
        # pad offset and the crop+downsample scale, then add back the
        # crop's own top-left corner (y0, x0).
        grid_rows, grid_cols = np.mgrid[0:gh, 0:gw]
        orig_row = y0 + (grid_rows - pad) / scale
        orig_col = x0 + (grid_cols - pad) / scale
        # A grid cell landing below orig_h is inside the synthetic leg
        # extension (_detect_crop_and_extend), not a real photographed
        # pixel -- there is no depth to sample there at all, so it's
        # excluded from the depth bias below (clamping its row for the
        # sample call is harmless since real_pixel masks it out either
        # way).
        real_pixel = orig_row < orig_h
        sample_row = np.clip(orig_row * (depth_h / orig_h), 0, depth_h - 1)
        sample_col = np.clip(orig_col * (depth_w / orig_w), 0, depth_w - 1)
        sampled = map_coordinates(depth, [sample_row, sample_col], order=1, mode="nearest")

        # Normalize against the SUBJECT's own depth values, not the whole
        # crop (which includes background at a very different depth scale
        # and would swamp the subject's own, much smaller, internal
        # variation) -- same reasoning procedural_body.py's depth sculpt
        # already applies, by construction there (it only ever samples
        # at front-facing body vertices to begin with). Excludes the
        # synthetic leg extension too -- it was never photographed, so
        # its "sampled" values are just whatever nearest-neighbor pixel
        # estimate_relative_depth happened to clamp to, not real signal.
        real_grid_mask = photo_cells & real_pixel
        subj_vals = sampled[real_grid_mask]
        std = subj_vals.std() if subj_vals.size else 0.0
        if std > 1e-9:
            d_norm = (sampled - subj_vals.mean()) / std
            if layer_rescue == "on" or (layer_rescue == "auto" and animal_frac > _LAYER_RESCUE_MIN_ANIMAL_FRAC):
                d_norm = d_norm + _layered_detail_rescue(sampled, grid_mask, real_grid_mask, std)
            # Front-only bias, proportional to each column's own
            # thickness (tapers to ~0 at the mask edge, same as
            # _sculpt_front_surface_from_depth's displacement does) --
            # never let the front cross back past a small positive
            # floor, matching that function's own "never cross the
            # centerline" safety.
            # Only the fine detail may push the front in: a whole part that
            # sits further back than the body's average (a head behind
            # crossed arms) kept getting thinner instead of just sitting
            # back -- measured, it cut the head's depth from 1.01 to 0.84
            # of its width. Parts nearer than average still bulge forward.
            if profile == "poisson":
                broad = _masked_blur(d_norm, real_grid_mask, max(0.05 * max(gh, gw), 1.0))
                # with a posed reference body the broad shape is its own
                d_norm = (d_norm - broad) if ref is not None else np.maximum(broad, 0.0) + (d_norm - broad)
            d_norm = np.where(photo_cells, d_norm, 0.0)
            relief = half_thickness * d_norm * depth_strength
            if ref is not None:
                # the reference already gives the big shapes (a hand on the
                # belly, the head's forward reach): the photo adds detail only
                cap = _REF_RELIEF_MAX * float(reference["px_per_m"])
                relief = np.clip(relief, -cap, cap)
                # ... on surfaces facing the camera: the depth estimate jumps
                # at every edge it sees (the jaw over the neck, the head
                # against the background), and pressed onto a surface facing
                # sideways that made a ridge down the jaw in front of the ear
                # and ripples over the crown
                sy_, sx_ = np.gradient(front0)
                facing = 1.0 / np.sqrt(1.0 + (np.hypot(sy_, sx_) * scale) ** 2)
                relief = relief * np.clip((facing - _REF_RELIEF_FACING[0]) / (_REF_RELIEF_FACING[1] - _REF_RELIEF_FACING[0]), 0.0, 1.0)
                if limb_soft is not None:
                    limb_detail = relief * limb_soft  # (the arm's own folds: onto the forearm's solid)
                    relief = relief * (1.0 - limb_soft)
            front = np.maximum(front0 + relief, mid0 + min_half_thickness * 0.3)
            back = back0
            # Synthetic extension rows keep the plain symmetric guess
            # (real_pixel False -> front==back==half_thickness mid/span)
            # -- no real depth to bias them with.
            apply_bias = grid_mask & real_pixel
            mid = np.where(apply_bias, (front + back) / 2.0, mid0)
            half_span = np.where(apply_bias, (front - back) / 2.0, half_thickness)

    if face_mesh is not None:
        # Real depth already carries much of the face's shape; full-strength
        # landmark relief on top of it double-counts (deep groove at the mouth).
        if face_strength is not None:
            strength = face_strength
        else:
            strength = _FACE_RELIEF_WITH_DEPTH if use_depth else 1.0
        face_add = np.where(grid_mask, _face_relief(face_mesh, (gh, gw), y0, x0, scale, pad) * strength, 0.0)
        if parts_g is not None:
            # only the face the photo shows: nothing sculpted under a mask,
            # behind sunglasses or under a fringe or a cap's peak (the face
            # landmarks are guessed there, the photo doesn't show them)
            from .head_parts import BODY_SKIN, FACE_SKIN

            face_add = face_add * gaussian_filter(np.isin(parts_g, (FACE_SKIN, BODY_SKIN)).astype(np.float32), 1.5)
        front = mid + half_span + face_add
        back = mid - half_span
        if parts_g is not None and ref is not None:
            front = _wear_on_face(front, grid_mask, parts_g, scale, float(reference["px_per_m"]))
        clothes_read = (parts or {}).get("clothes") if parts is not None else None
        if clothes_read and reference is not None:
            try:
                cr = _clothes_relief(clothes_read, front, grid_mask, y0, x0, scale, pad, float(reference["px_per_m"]))
                if limb_soft is not None:
                    limb_detail = limb_detail + _LIMB_CLOTH_DETAIL * cr * limb_soft
                front = front + (cr if limb_soft is None else cr * (1.0 - limb_soft))
            except Exception as exc:  # noqa: BLE001 - clothes keep their shape without steps
                print(f"Note: clothes steps skipped ({exc}).")
        if reference is not None:
            try:
                from .clothes import NAMES as _CN

                skin_g = np.zeros((gh, gw), bool)
                neck_bare = False
                if clothes_read:
                    lab_c = clothes_read["labels"]
                    rr_ = np.clip(np.rint(y0 + (np.arange(gh) - pad) / scale).astype(int), 0, lab_c.shape[0] - 1)
                    cc_ = np.clip(np.rint(x0 + (np.arange(gw) - pad) / scale).astype(int), 0, lab_c.shape[1] - 1)
                    skin_g = (lab_c[rr_[:, None], cc_[None, :]] == _CN.index("skin") + 1) & grid_mask
                    fm_ = np.asarray(face_mesh)
                    band = (rr_ > fm_[152, 1]) & (rr_ < fm_[152, 1] + 0.3 * np.ptp(fm_[:468, 1]))
                    neck_bare = bool(skin_g[band].mean() > 0.3) if band.any() and skin_g[band].size else False
                bf, bb = _bony_landmarks(reference, pose, skin_g, neck_bare, y0, x0, scale, pad)
                front = np.where(grid_mask, front + bf, front)
                back = np.where(grid_mask, back - bb, back)
                if parts_g is not None:
                    from .head_parts import ACCESSORY

                    fm_ = np.asarray(face_mesh)[:, :2]
                    rows_g = (np.arange(gh)[:, None] - pad) / scale + y0
                    cols_g = (np.arange(gw)[None, :] - pad) / scale + x0
                    fw_ = abs(fm_[454, 0] - fm_[234, 0])
                    ruler_ = fm_[2, 1] - (fm_[105, 1] + fm_[334, 1]) / 2
                    ear_side = (cols_g < fm_[234, 0] + 0.1 * fw_) | (cols_g > fm_[454, 0] - 0.1 * fw_)
                    ear_rows = (rows_g > fm_[2, 1] - 0.5 * ruler_) & (rows_g < fm_[152, 1] + 1.5 * ruler_)
                    ring = (parts_g == ACCESSORY) & ear_side & ear_rows & grid_mask
                    front = front + _EARRING_PROUD * float(reference["px_per_m"]) * gaussian_filter(ring.astype(np.float64), 1.0)
            except Exception as exc:  # noqa: BLE001 - the body keeps its shape without them
                print(f"Note: bony landmarks skipped ({exc}).")
        face_read = (parts or {}).get("face") if parts is not None else None
        if face_read and (face_strength is None or face_strength > 0):
            try:
                ppm = float(reference["px_per_m"]) if reference is not None else (
                    float(np.linalg.norm(face_mesh[10, :2] - face_mesh[152, :2])) / 0.18)
                fp = _face_parts_relief(face_read, face_mesh, (gh, gw), y0, x0, scale, pad, ppm)
                if parts_g is not None:
                    from .head_parts import BODY_SKIN, FACE_SKIN, HAIR

                    fp = fp * gaussian_filter(np.isin(parts_g, (FACE_SKIN, BODY_SKIN, HAIR)).astype(np.float32), 1.0)
                front = np.where(grid_mask, front + fp, front)
            except Exception as exc:  # noqa: BLE001 - the face keeps its landmark shape
                print(f"Note: face parts skipped ({exc}).")
        if hair_rel is not None:
            front = np.where(grid_mask, front + hair_rel[0], front)
            back = np.where(grid_mask, back - hair_rel[1], back)
        mid = (front + back) / 2.0
        # Eye sockets push the front back; never let that thin the head
        # below the floor used elsewhere.
        half_span = np.where(grid_mask, np.maximum((front - back) / 2.0, min_half_thickness * 0.3), half_span)

    layers = []  # (front, back, cells) of solids joined on top: crossed forearms, hands on the body
    if limbs3d is not None and limbs3d["cells"].any():
        # resting on the body: its underside meets the body's front (a gap
        # left it a separate piece, and the join keeps only the body); where
        # it runs on past the torso it sits just inside the arm's own surface
        # (two equal surfaces joined as ragged fringes)
        ppm_l = float(reference["px_per_m"])
        body_front = mid + half_span
        over_s = np.clip(gaussian_filter(limbs3d["over"].astype(np.float64), 1.0) * 2.0, 0.0, 1.0)
        lm = (limbs3d["front"] + limbs3d["back"]) / 2.0
        lh = (limbs3d["front"] - limbs3d["back"]) / 2.0 * (0.9 + 0.1 * over_s)
        # (the sleeve's folds; not where a hand lies on it -- that detail is
        # the hand's outline, and on the forearm it came out a torn patch)
        no_hand = 1.0 - (np.clip(gaussian_filter(hands_on.astype(np.float64), 2.0) * 2.0, 0.0, 1.0) if hands_on is not None else 0.0)
        # (clipped and softened: by the hand and watch the photo's depth jumps,
        # and on the forearm's round surface that came out a torn patch)
        cap_l = 0.5 * _REF_RELIEF_MAX * ppm_l
        det_l = gaussian_filter(np.clip(np.where(limbs3d["cells"], limb_detail, 0.0), -cap_l, cap_l), 1.0)
        lf_ = lm + lh + _LIMB_DETAIL * over_s * no_hand * det_l
        lb_ = np.where(limbs3d["over"], np.minimum(lm - lh, body_front - _HAND_SINK * ppm_l), lm - lh)
        layers.append((lf_, lb_, limbs3d["cells"]))
    if hands_on is not None and hands_on.any():
        base = np.where(grid_mask, mid + half_span, 0.0)
        if limbs3d is not None:
            base = np.where(limbs3d["cells"], np.maximum(base, limbs3d["front"]), base)
        ppm_ = float(reference["px_per_m"])
        d_h = distance_transform_edt(hands_on)
        from scipy.ndimage import label as _cc

        lab_h, n_h = _cc(hands_on)
        dome = np.zeros_like(d_h)
        for i in range(1, n_h + 1):
            m_ = lab_h == i
            dome[m_] = np.sqrt(np.clip(d_h[m_] / max(float(d_h[m_].max()), 1e-6), 0.0, 1.0))
        dome = _masked_blur(dome, hands_on, 1.5)  # (a thin, ragged patch of fingers came out torn)
        base_s = _masked_blur(base, hands_on, 2.0)
        # the photo's own detail on it (the fingers' folds), as on the forearms
        detail_h = np.where(hands_on, limb_detail, 0.0) if limbs3d is not None else 0.0
        detail_h = gaussian_filter(np.clip(detail_h, -0.3 * _HAND_THICK * ppm_, 0.3 * _HAND_THICK * ppm_), 1.0)
        layers.append((base_s + _HAND_THICK * ppm_ * dome + _LIMB_DETAIL * detail_h, base_s - _HAND_SINK * ppm_, hands_on))
    true_max = float((np.abs(mid) + half_span).max())
    for lf, lb, lc in layers:
        if lc.any():
            true_max = max(true_max, float(np.abs(lf[lc]).max()), float(np.abs(lb[lc]).max()))
    if true_max <= 0:
        raise ValueError("Mask too thin/small to extrude a volume from.")
    # A few percent beyond the true max so the solid never exactly
    # touches the volume's own first/last z-slice -- that single tangent
    # point is itself a small degenerate case, same family of problem as
    # the edge cusp above.
    z_max = true_max * 1.05
    if edge == "hard":
        # Original field: depth steps 3x finer than the floor thickness (so
        # thin limbs close), volume cut off hard at the mask.
        dz_target = min_half_thickness / 3.0
        G = np.where(grid_mask, half_span.astype(np.float64) ** 2, 0.0).astype(np.float32)
    else:
        # Depth steps the size of the grid cells (slightly finer), so voxels
        # are about cube-shaped and the rounded edges resolve evenly.
        dz_target = 0.6 / scale
        # The solid is (z - mid)^2 <= G. Inside the outline G = half_span^2,
        # which falls linearly to ~0 at the edge; outside it continues below
        # zero at that same rate instead of being cut off by the hard mask,
        # so the front and back surfaces meet in a rounded edge rather than a
        # seam-like wall around the whole outline.
        span_sq = np.where(grid_mask, half_span.astype(np.float64) ** 2, 0.0)
        dist_grid = distance_transform_edt(grid_mask)
        near_edge = grid_mask & (dist_grid <= 2.0)
        edge_slope = float(np.median(span_sq[near_edge] / dist_grid[near_edge])) if near_edge.any() else 1.0
        # Outside, continue the inside field along its own local slope. With
        # one global slope there, where the surface crossed zero between the
        # last cell inside and the first outside depended on where the true
        # edge fell inside that cell -- different on every row, which showed
        # as grooves down every side. Clamped so the edge stays within a cell
        # of the outline.
        d_out, (iy, ix) = distance_transform_edt(~grid_mask, return_indices=True)
        oy, ox = np.mgrid[0:gh, 0:gw]
        uy, ux = iy - oy, ix - ox
        un = np.maximum(np.hypot(uy, ux), 1e-6)
        jy = np.clip(np.rint(iy + uy / un).astype(int), 0, gh - 1)
        jx = np.clip(np.rint(ix + ux / un).astype(int), 0, gw - 1)
        g1, g2 = span_sq[iy, ix], span_sq[jy, jx]
        slope = np.where(grid_mask[jy, jx] & (g2 > g1), g2 - g1, edge_slope)
        g_out = np.minimum(g1 - slope * d_out, -(d_out - 1.0) * edge_slope)
        G = np.where(grid_mask, span_sq, g_out).astype(np.float32)
    gz = max(4, round(2 * z_max / dz_target) + 4)

    zs = np.linspace(-z_max, z_max, gz)
    dz = zs[1] - zs[0]
    if edge != "hard":
        # Carry each edge cell's mid-depth outward. It was 0 outside the
        # outline, and the edge is placed by interpolating the field between
        # the last cell inside and the first outside: with a jump in mid
        # between them every row's edge landed somewhere different, which
        # showed as grooves down every side.
        near_in = distance_transform_edt(~grid_mask, return_distances=False, return_indices=True)
        mid = np.where(grid_mask, mid, mid[near_in[0], near_in[1]])
    field = np.empty((gz, gh, gw), dtype=np.float32)
    for i, z in enumerate(zs):
        field[i] = (z - mid) ** 2 - G
    if edge == "hard":
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
    if smooth:
        smooth_grid_terracing(mesh)
        if ref is not None:
            # The round sides (the band where front meets back) still step by
            # a fraction of a grid cell from row to row -- below a print layer,
            # but grazing light shows it as grooves. Smooth those harder,
            # fading out toward surfaces facing front or back (face, relief).
            # (Vertices are (depth, row, col) here.)
            side = np.clip((0.75 - np.abs(mesh.vertex_normals[:, 0])) / 0.35, 0.0, 1.0)
            extra = mesh.copy()
            trimesh.smoothing.filter_taubin(extra, lamb=0.5, nu=0.53, iterations=_REF_SIDE_SMOOTH)
            mesh.vertices = mesh.vertices + side[:, None] * (extra.vertices - mesh.vertices)
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
    solids_extra = []
    for lf, lb, lc in layers:
        piece = _layer_solid(lf, lb, lc, zs, dz, scale, smooth)
        if piece is not None:
            pv, pf_ = piece
            pxyz = np.stack([pv[:, 2] - (gw / scale) / 2.0, (gh / scale) / 2.0 - pv[:, 1], pv[:, 0] - z_max], axis=1)
            solids_extra.append((pxyz, pf_))
    if solids_extra:
        try:
            verts_xyz, faces = _join_solids(verts_xyz, faces, solids_extra)
        except Exception as exc:  # noqa: BLE001 - the body without separate limbs is still a model
            print(f"Note: could not join the crossed arms and hands ({exc}).")
    if ref is not None and ref[3] is not None:
        # the 3D ears, joined to the head (photo px -> this frame)
        ears3d = ref[3]
        thicken = max(0.0, 0.5 * (_REF_EAR_MIN_MM / out_scale - ears3d["thick"]))
        solids = []
        for patch in ears3d["patches"]:
            P = patch["vertices"]
            Q = np.column_stack([(P[:, 0] - x0) + pad / scale - (gw / scale) / 2.0,
                                 (gh / scale) / 2.0 - ((P[:, 1] - y0) + pad / scale),
                                 -P[:, 2] + ref_shift])
            inward = np.asarray(patch["inward"], dtype=np.float64) * np.array([1.0, -1.0, -1.0])
            solid = _ear_solid(Q, patch["faces"], inward, ears3d["plug"], thicken)
            if solid is not None:
                solids.append(solid)
        if solids:
            try:
                verts_xyz, faces = _join_solids(verts_xyz, faces, solids)
            except Exception as exc:  # noqa: BLE001 - the head without separate ears is still a model
                print(f"Note: could not join the ears ({exc}).")
    verts_xyz = verts_xyz * out_scale

    return BodyMesh(vertices=verts_xyz, faces=faces)
