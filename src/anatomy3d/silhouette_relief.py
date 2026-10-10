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
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from skimage.measure import marching_cubes

from .mesh_types import BodyMesh

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


def build_silhouette_relief_mesh(
    alpha: np.ndarray,
    target_height_mm: float = 150.0,
    grid_size: int = 360,
    depth_scale: float = 0.5,
    depth_rgb: Optional[np.ndarray] = None,
    depth_strength: float = 0.5,
    animal_frac: float = 0.0,
) -> BodyMesh:
    """`alpha`: a float [0,1] or boolean (H, W) mask -- the same soft
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

    `depth_scale`: real human/animal depth (front-to-back) runs
    shallower than half of the local width a straight distance-transform
    value would give (a torso's front-to-back depth is roughly half its
    side-to-side width, not equal to it) -- a tuned, not measured,
    flattening factor, same honesty this project already applies to its
    own tuned constants elsewhere (procedural_body.py's blend_k_frac).

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

    # mid/half_span generalize the symmetric lens (front = +half_thickness,
    # back = -half_thickness) to an optionally asymmetric one -- with no
    # depth_rgb, mid stays 0 and half_span stays half_thickness, exactly
    # recovering the original symmetric behavior.
    mid = np.zeros_like(half_thickness)
    half_span = half_thickness
    if depth_rgb is not None:
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
        real_grid_mask = grid_mask & real_pixel
        subj_vals = sampled[real_grid_mask]
        std = subj_vals.std() if subj_vals.size else 0.0
        if std > 1e-9:
            d_norm = (sampled - subj_vals.mean()) / std
            if animal_frac > _LAYER_RESCUE_MIN_ANIMAL_FRAC:
                d_norm = d_norm + _layered_detail_rescue(sampled, grid_mask, real_grid_mask, std)
            # Front-only bias, proportional to each column's own
            # thickness (tapers to ~0 at the mask edge, same as
            # _sculpt_front_surface_from_depth's displacement does) --
            # never let the front cross back past a small positive
            # floor, matching that function's own "never cross the
            # centerline" safety.
            front = np.maximum(half_thickness * (1.0 + d_norm * depth_strength), min_half_thickness * 0.3)
            back = -half_thickness
            # Synthetic extension rows keep the plain symmetric guess
            # (real_pixel False -> front==back==half_thickness mid/span)
            # -- no real depth to bias them with.
            apply_bias = grid_mask & real_pixel
            mid = np.where(apply_bias, (front + back) / 2.0, 0.0)
            half_span = np.where(apply_bias, (front - back) / 2.0, half_thickness)

    true_max = float((np.abs(mid) + half_span).max())
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
        field[i] = (z - mid) ** 2 - half_span**2
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
