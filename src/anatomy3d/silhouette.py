"""Extract a subject silhouette from a photo against a roughly plain
background, in two stages that each correct the one before it:

1. Threshold: sample the image border as the background color, keep
   pixels far enough from it, take the largest connected blob. Cheap,
   but a single flat cutoff misclassifies shadowed or near-background-
   colored regions.
2. Two-sided reclassify: cluster BOTH the foreground and background seed
   pixels into several color groups each (not one broad Gaussian per
   side — see _two_sided_reclassify's docstring for why that failed), and
   reclassify every pixel by whichever side's nearest cluster is actually
   closer. Genuinely color-ambiguous pixels (checked, not assumed — see
   AMBIGUITY_MARGIN) fall back to a position check against the subject's
   own detected pose, when given.

A third stage — snapping the boundary to image edges with an active
contour (Kass et al. 1988) — was tried and reverted. On a figure with a
narrow neck and separated legs, the contour's smoothness term pulled
straight across both narrow points, deleting the head and most of the
legs; a coarse area-ratio sanity check didn't catch it because the lost
area wasn't large enough to trip it. Revisit only with a shape-aware check
(e.g. compare the bounding box, not just area) and testing against real
photos, not just clean synthetic edges.
"""
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2

from .safety import mad_margin_above_minimum

# Hard floor: fewer samples than this can't support a stable 3x3 RGB
# covariance fit, so the reclassifier silently keeps the threshold mask
# instead.
MIN_FIT_SAMPLES = 10
# Calibrated against synthetic overlapping-color classes with known ground
# truth, not guessed: mean classification accuracy turned out essentially
# flat across the whole sample range tested (16 to 150k+ pixels) — color
# overlap sets a hard ceiling that more data doesn't lift. What DOES improve
# with more samples is run-to-run variance (noisier reclassification on a
# different unlucky draw of the same scene): std 0.028 at 16 samples, 0.005
# at 225, 0.002 by ~2000, flat after. The original 200 sat in the still-
# noisy part of that curve; 800 is past the knee where returns flatten out.
RECOMMENDED_FIT_SAMPLES = 800


# binary_fill_holes fills every fully-enclosed background region inside
# the blob, no matter its size — which is right for real noise (a few
# misclassified pixels from a highlight or JPEG artifact) but wrong for a
# real, large enclosed background region, like the triangular gap a lunge
# or a hand-on-hip pose leaves between a limb and the torso: confirmed
# directly on a real photo that the per-pixel color classification
# already correctly read that gap as ~85% background (13-15% foreground,
# matching its color sitting well within 2.4 standard deviations of a
# real background cluster) BEFORE fill_holes ran, and blind filling then
# overwrote that correct reading to 98.6% foreground. Real noise holes on
# that same photo topped out at 70px; the real gap was 27,189px — a 388x
# gap in the size distribution, not a borderline case. This fills only
# holes at or below that size, leaving a real enclosed background region
# alone.
MAX_HOLE_FILL_AREA_PX = 1000


def _largest_filled_blob(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        raise RuntimeError(
            "No subject found against the background — use a photo with a "
            "plain, evenly lit backdrop behind the person."
        )
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    largest = 1 + int(np.argmax(sizes))
    blob = labeled == largest

    fully_filled = ndimage.binary_fill_holes(blob)
    holes = fully_filled & ~blob
    hole_labels, n_holes = ndimage.label(holes)
    if n_holes == 0:
        return blob
    hole_sizes = ndimage.sum(holes, hole_labels, index=range(1, n_holes + 1))
    small_hole_ids = [i + 1 for i, s in enumerate(hole_sizes) if s <= MAX_HOLE_FILL_AREA_PX]
    small_holes = np.isin(hole_labels, small_hole_ids)
    return blob | small_holes


# A Mahalanobis distance of this many "standard deviations" from the
# nearest background color cluster — not a raw pixel-distance number like
# the old flat `threshold`, because it needs to mean the same thing
# regardless of how much that particular background region's color
# naturally varies (shadow/texture in sand spreads much wider in color
# space than a clear sky does). Chosen from a real measurement: on a test
# photo's own sand region, real background texture/shading spread to
# roughly 1.5-2 standard deviations from that region's own color-cluster
# mean, so a 32-unit flat Euclidean cutoff (tuned for a clean, low-texture
# background) was both too tight for sand's spread (misreading much of it
# as foreground) and, from the opposite direction, no safer against a
# subject whose color happens to sit near a background cluster. 3.5
# standard deviations comfortably covers real background texture while
# still being a specific, real color difference for anything sitting that
# far out to be flagged foreground — not an unbounded "anything goes".
MAHALANOBIS_THRESHOLD = 3.5

# How many background color tones to look for by default. Was 2 (sky,
# sand) — raised after measuring, not guessing, that a real beach photo
# actually has at least 3 distinct background tones (sky, open water,
# sand), and that 2 clusters left a wide band of water misclassified as
# foreground across whole image rows (see _threshold_mask's docstring for
# the actual before/after numbers). 5 is where that same measurement
# plateaued — more clusters than a photo's real tone count stopped
# changing the result at all, rather than where accuracy was still
# improving, so it isn't just "generously higher than 2". A genuinely
# flat, single-tone background is unaffected regardless: _threshold_mask
# already collapses to k=1 on its own when the border sample's own color
# spread says there's nothing to cluster.
DEFAULT_BACKGROUND_CLUSTERS = 5

# Unlike the background, the foreground has no natural plateau to measure:
# tried k=1..25 on a real photo's own foreground seed pixels and inertia
# kept dropping ~20% at every step the whole way, never flattening like
# the background did at k=5. That's a real, structural difference, not a
# measurement failure — a body's color is a continuous gradient (shading
# across a rounded, lit 3D surface) rather than a few discrete flat
# tones, so there's no small "true" cluster count to find. 5 is chosen to
# match the background default and cover the number of visually distinct
# materials a clothed figure typically has (skin, 1-2 clothing colors, an
# accessory/background prop like a mat, hair) — a reasonable choice, not
# a measured one, unlike DEFAULT_BACKGROUND_CLUSTERS.
DEFAULT_FOREGROUND_CLUSTERS = 5


def _cluster_min_mahalanobis(
    seed_pixels: np.ndarray,
    flat_pixels: np.ndarray,
    k: int,
    max_samples: int = 20000,
    rng_seed: int = 0,
    fallback_scale: float = 32.0,
) -> np.ndarray:
    """Fits `k` color clusters (k-means) to `seed_pixels` and returns, for
    every pixel in `flat_pixels`, its Mahalanobis distance to the NEAREST
    cluster's own mean and spread. Shared by the background clustering in
    _threshold_mask and the two-sided foreground/background clustering in
    _two_sided_reclassify — same reasoning both times: one flat distance
    or one broad covariance can't represent a color population that's
    actually several distinct tones (see DEFAULT_BACKGROUND_CLUSTERS and
    DEFAULT_FOREGROUND_CLUSTERS)."""
    rng = np.random.default_rng(rng_seed)
    sample = seed_pixels[rng.choice(len(seed_pixels), max_samples, replace=False)] if len(seed_pixels) > max_samples else seed_pixels
    k = min(k, len(sample))
    if k > 1 and np.std(sample, axis=0).max() < 1.0:
        k = 1  # nothing to cluster — a near-flat color population
    try:
        centroids, labels = kmeans2(sample, k, seed=rng_seed, minit="++")
    except Exception:
        centroids = np.median(sample, axis=0, keepdims=True)
        labels = np.zeros(len(sample), dtype=int)

    dists = []
    for i in range(len(centroids)):
        cluster = sample[labels == i]
        if len(cluster) < 4:
            dists.append(np.linalg.norm(flat_pixels - centroids[i], axis=-1) / fallback_scale * MAHALANOBIS_THRESHOLD)
            continue
        cov = np.cov(cluster, rowvar=False) + np.eye(3) * 1e-3
        prec = np.linalg.inv(cov)
        diff = flat_pixels - centroids[i]
        dists.append(np.sqrt(np.einsum("ij,jk,ik->i", diff, prec, diff)))
    return np.min(np.stack(dists), axis=0)


def _threshold_mask(
    rgb: np.ndarray, border_width: int = 12, threshold: float = 32.0, background_clusters: int = DEFAULT_BACKGROUND_CLUSTERS
) -> np.ndarray:
    """A single border-sampled color only describes a flat, one-tone
    background, and even a flat Euclidean distance from it doesn't
    account for how much that background's color naturally varies.
    Checked against a real photo with a two-tone, textured background
    (sky above, sand below): a single blended background color, and
    separately a flat per-cluster distance cutoff, both measurably failed
    — sand's natural shading spread wider in color space than a flat
    cutoff tuned for a cleaner backdrop could handle, so the fix isn't
    just "notice there are two background colors" but "account for how
    much each one spreads". Clustering the border sample into
    `background_clusters` colors (k-means) and measuring each pixel's
    Mahalanobis distance (standard deviations, not raw color distance) to
    the *nearest* cluster's own mean and spread fixes this: confirmed
    directly that the clustering step alone recovers the two real
    background colors (sky, sand) from that photo's own border pixels,
    and that per-cluster Mahalanobis distance correctly separates real
    body-color pixels (measured 4-10+ standard deviations out) from real
    background texture (measured within ~2) on that same photo. A
    genuinely flat, low-texture background just produces tight clusters
    and behaves close to the old flat-distance behavior.

    That "two-tone" photo turned out to undersell its own background,
    found later by actually checking per-row mask coverage on it: with
    the default of 2 clusters, a wide band of open water (a third real
    tone — sky, water, and sand are all genuinely different colors, not
    two) came out 29-100% misclassified as foreground across whole image
    rows, not an edge case. Re-measured against several cluster counts
    on that same photo: the bad-row count dropped from 104/200 at k=2 to
    15/200 at k=5, and k=5/6/8/10 then gave IDENTICAL results — a real,
    data-found plateau (more clusters than the photo's actual number of
    distinct background tones just subdivides a tone that's already
    well-fit, not a number picked by feel), not a guess. See
    DEFAULT_BACKGROUND_CLUSTERS."""
    border_pixels = np.concatenate([
        rgb[:border_width].reshape(-1, 3),
        rgb[-border_width:].reshape(-1, 3),
        rgb[:, :border_width].reshape(-1, 3),
        rgb[:, -border_width:].reshape(-1, 3),
    ]).astype(np.float64)

    flat = rgb.astype(np.float64).reshape(-1, 3)
    # Border pixels are already a bounded set (image perimeter, not the
    # whole photo) — a real regression was caught here directly: an
    # earlier version of this refactor applied the same 20000-pixel
    # subsample cap used for the much larger foreground/background body
    # clustering, which halved this photo's ~40,008 border pixels before
    # fitting and nearly doubled the resulting mask's coverage (0.097 ->
    # 0.172) purely from losing cluster fidelity, not any real signal.
    # max_samples=len(border_pixels) keeps every one of them.
    min_maha = _cluster_min_mahalanobis(
        border_pixels, flat, background_clusters, max_samples=len(border_pixels), fallback_scale=threshold
    ).reshape(rgb.shape[:2])
    return min_maha > MAHALANOBIS_THRESHOLD


# How many "standard deviations" apart the foreground- and background-
# cluster distances need to be before a pixel's color alone is trusted to
# decide it. Below this, it's a genuine toss-up, not a confident call —
# measured directly, not guessed: on a real photo, confidently-correct
# spots (clear skin, clear sky, a water patch the clustering alone
# already resolved) all landed past |margin|=3.4, while two real,
# confirmed misclassifications (a foam/wet-sand patch, the background gap
# between two legs in a lunge) both sat under 1 (+0.62 and -0.21) — a
# clean, wide gap in the real distribution, not a borderline choice.
AMBIGUITY_MARGIN = 2.0

# How far (px) a pixel may sit from the nearest bone segment (a straight
# line between two detected, anatomically-adjacent joints) and still be
# trusted as real anatomy, when resolving a color-ambiguous pixel by
# position. Measured, not guessed: real skin at mid-bone (not just near a
# joint) sits at 0px by construction, and even an off-the-straight-line
# point (a shin, which isn't perfectly straight) measured 15.7px; the two
# confirmed-background ambiguous spots measured 51.4px and 74.0px — both
# comfortably beyond either real-anatomy figure. 30 sits with a wide
# margin on both sides of that gap.
POSITION_CORRIDOR_PX = 30.0


def _position_is_on_body(points_xy: np.ndarray, bones: "list[tuple[np.ndarray, np.ndarray]]") -> np.ndarray:
    """For each (x, y) in `points_xy`, the distance to the nearest of
    `bones` (each a (joint_a, joint_b) pixel-coordinate pair) — True where
    that distance is within POSITION_CORRIDOR_PX. See AMBIGUITY_MARGIN's
    docstring for why this only matters for color-ambiguous pixels, not
    as a replacement for the color check."""
    min_dist = None
    for a, b in bones:
        ab = b - a
        length_sq = float(np.dot(ab, ab))
        if length_sq < 1e-9:
            d = np.linalg.norm(points_xy - a, axis=-1)
        else:
            t = np.clip(((points_xy - a) @ ab) / length_sq, 0, 1)
            closest = a + t[:, None] * ab
            d = np.linalg.norm(points_xy - closest, axis=-1)
        min_dist = d if min_dist is None else np.minimum(min_dist, d)
    return min_dist <= POSITION_CORRIDOR_PX


def _two_sided_reclassify(
    rgb: np.ndarray,
    seed_mask: np.ndarray,
    foreground_clusters: int = DEFAULT_FOREGROUND_CLUSTERS,
    background_clusters: int = DEFAULT_BACKGROUND_CLUSTERS,
    bones: "list[tuple[np.ndarray, np.ndarray]] | None" = None,
) -> np.ndarray:
    """Replaces the single-Gaussian-per-side reclassifier this project
    used before: that version fit ONE broad Gaussian to everything stage 1
    called foreground. Checked directly on a real photo, and confirmed
    that's a real bug, not a style choice — a figure's true foreground
    color spans skin, dark clothing, and a mat all at once, so the single
    Gaussian's covariance ends up so wide that it statistically
    out-competed a tighter, more specific background model for an
    unrelated dark water color (-14.2 foreground log-likelihood vs -182.0
    background, confirmed by computing both directly) — exactly backwards.

    This clusters BOTH sides into several tighter color groups (same
    technique DEFAULT_BACKGROUND_CLUSTERS already used, now applied
    symmetrically) and classifies by whichever side's nearest cluster is
    closer. That alone resolved the water-color bug. It does NOT resolve
    every case: some real background colors (pale foam, wet sand) are
    genuinely close, in color alone, to real foreground colors (bright
    skin highlights, light fabric) — confirmed directly, not assumed, by
    checking two known-bad spots sat within 1 standard deviation of a
    toss-up while confidently-correct spots sat past 3.4. For those
    genuinely ambiguous pixels, `bones` (if given) resolves by position
    instead: real anatomy sits on or very near the subject's own detected
    skeleton; background doesn't, confirmed by directly measuring that gap
    too (0-15.7px for real anatomy vs 51.4-74.0px for the two known-bad
    spots). Without `bones`, an ambiguous pixel keeps stage 1's call."""
    rgb64 = rgb.astype(np.float64)
    fg_pixels = rgb64[seed_mask]
    bg_pixels = rgb64[~seed_mask]
    if len(fg_pixels) < MIN_FIT_SAMPLES or len(bg_pixels) < MIN_FIT_SAMPLES:
        return seed_mask  # not enough signal to cluster either side

    smaller = min(len(fg_pixels), len(bg_pixels))
    result = mad_margin_above_minimum(smaller, RECOMMENDED_FIT_SAMPLES, min_margin=0.0)
    if not result.ok:
        print(
            f"Note: color model fit from only {smaller} pixels of the smaller "
            f"class, {-result.margin:.0%} below the {RECOMMENDED_FIT_SAMPLES}-pixel "
            "comfort level — reclassification may be noisy on a small or "
            "tightly-cropped subject."
        )

    flat = rgb64.reshape(-1, 3)
    fg_dist = _cluster_min_mahalanobis(fg_pixels, flat, foreground_clusters)
    bg_dist = _cluster_min_mahalanobis(bg_pixels, flat, background_clusters)
    margin = bg_dist - fg_dist  # positive = confidently foreground

    confident_fg = margin > AMBIGUITY_MARGIN
    confident_bg = margin < -AMBIGUITY_MARGIN
    ambiguous = ~confident_fg & ~confident_bg

    result_mask = confident_fg.copy()
    if bones:
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        points_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
        on_body = _position_is_on_body(points_xy, bones)
        result_mask[ambiguous] = on_body[ambiguous]
    else:
        # No pose to check position against — keep stage 1's own call for
        # the pixels color alone can't confidently decide, rather than
        # guessing either way.
        result_mask[ambiguous] = seed_mask.reshape(-1)[ambiguous]

    return result_mask.reshape(rgb.shape[:2])


def extract_silhouette(
    rgb: np.ndarray,
    border_width: int = 12,
    threshold: float = 32.0,
    refine: bool = True,
    background_clusters: int = DEFAULT_BACKGROUND_CLUSTERS,
    foreground_clusters: int = DEFAULT_FOREGROUND_CLUSTERS,
    bones: "list[tuple[np.ndarray, np.ndarray]] | None" = None,
) -> np.ndarray:
    """`bones`: optional list of (joint_a, joint_b) pixel-coordinate pairs
    from the same photo's own detected pose (e.g. shoulder-to-elbow,
    hip-to-knee) — see _two_sided_reclassify's docstring. Deliberately
    plain coordinate pairs, not a landmarks.py type, so this module stays
    decoupled from pose-detection internals; the caller (pipeline.py)
    builds the list from whatever keypoints it already has."""
    mask = _largest_filled_blob(_threshold_mask(rgb, border_width, threshold, background_clusters))
    if not refine:
        return mask
    return _largest_filled_blob(_two_sided_reclassify(rgb, mask, foreground_clusters, background_clusters, bones))
