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
#
# Like POSITION_CORRIDOR_MIN/MAX_PX before it, this was a flat pixel count
# calibrated from one photo (shoulder width 96.2px) — same disease, caught
# by audit rather than waiting for a third photo to expose it: a hole-fill
# AREA scales with the SQUARE of linear resolution, not linearly, so a
# photo shot at business_person.png's scale (5.5x the beach photo's
# shoulder width) would see both real noise holes and real anatomical gaps
# land at roughly 30x (5.5 squared) this photo's own pixel-area numbers —
# meaning the flat 1000px cutoff could wrongly leave real noise unfilled
# at higher resolction, or wrongly swallow a real gap at lower resolution.
# Expressed as a ratio of (shoulder width)^2 so it scales the same way the
# areas themselves do; see POSITION_CORRIDOR_MIN/MAX_RATIO for the same
# pattern applied to a linear (not area) measurement.
MAX_HOLE_FILL_AREA_PX = 1000
MAX_HOLE_FILL_AREA_RATIO = MAX_HOLE_FILL_AREA_PX / (96.2 ** 2)


def _largest_filled_blob(mask: np.ndarray, body_scale_px: "float | None" = None) -> np.ndarray:
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
    max_area = MAX_HOLE_FILL_AREA_RATIO * (body_scale_px ** 2) if body_scale_px else MAX_HOLE_FILL_AREA_PX
    small_hole_ids = [i + 1 for i, s in enumerate(hole_sizes) if s <= max_area]
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


# How many MADs (median absolute deviations) a border strip's own median
# color may sit from the other three strips' consensus before it's treated
# as contaminated by the subject rather than real background. A real,
# confirmed failure mode, not hypothetical: on a tightly-cropped portrait
# where the subject's dark blazer extends to the bottom edge of the frame,
# the bottom border strip measured 53% dark pixels (vs 0% on the other
# three sides) and its median sat 58-63 MADs from the other strips'
# consensus — nowhere near a real background's natural side-to-side
# variation (the same photo's three honest strips sat within 1.3 MADs of
# each other). That contaminated strip, left in, fit one of the background
# k-means clusters to the subject's own clothing color, which then
# wrongly matched that same clothing color everywhere else in the photo
# too. 4.0 MADs comfortably separates the two: nowhere near the 1.3 a
# clean strip measured, nowhere near the 58+ a contaminated one did.
BORDER_STRIP_CONTAMINATION_MADS = 4.0


def _clean_border_pixels(rgb: np.ndarray, border_width: int) -> np.ndarray:
    """The border-strip background sample assumes all four image edges
    show only background — true for a subject comfortably inside the
    frame, false whenever a photo is cropped tight enough that the
    subject itself touches an edge (common for headshots/portraits). Each
    of the 4 strips is checked against the other three's consensus median
    (robust to one bad strip, since at most 1 of 4 being contaminated
    still leaves a majority); a strip whose own median is a gross outlier
    gets dropped entirely rather than letting it corrupt the shared
    k-means fit. See BORDER_STRIP_CONTAMINATION_MADS for the real
    measurement behind the cutoff."""
    strips = {
        "top": rgb[:border_width].reshape(-1, 3),
        "bottom": rgb[-border_width:].reshape(-1, 3),
        "left": rgb[:, :border_width].reshape(-1, 3),
        "right": rgb[:, -border_width:].reshape(-1, 3),
    }
    medians = {k: np.median(v, axis=0) for k, v in strips.items()}
    all_medians = np.stack(list(medians.values()))
    consensus = np.median(all_medians, axis=0)
    mad = np.median(np.abs(all_medians - consensus), axis=0) + 1e-6
    kept = [v for k, v in strips.items() if not (np.abs(medians[k] - consensus) / mad > BORDER_STRIP_CONTAMINATION_MADS).any()]
    # If 2+ strips disagree this sharply, the border itself isn't a
    # trustworthy background sample at all (not just one contaminated
    # side) — fall back to using all of it rather than guessing further.
    return np.concatenate(kept) if len(kept) >= 2 else np.concatenate(list(strips.values()))


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
    border_pixels = _clean_border_pixels(rgb, border_width).astype(np.float64)

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

# A single flat corridor radius was tried first and measurably wrong: a
# real forearm/upper-arm measured 5.8-10.0px thick (its own confident-mask
# distance transform at mid-bone) on a real photo, while torso/hip bones
# on the SAME photo measured 14.9-17.5px — a wide, real difference a flat
# 30px radius can't fit without being several times too generous for the
# arms. Confirmed as the actual mechanism, not assumed: that excess width
# around the arms was exactly the real sky-blue background visible in a
# cutout image once this was still 30px flat. So the corridor is now
# scaled from each bone's OWN measured thickness, not one shared guess.
POSITION_CORRIDOR_SCALE = 2.0  # multiple of a bone's own measured half-width — covers clothing/edge-blur margin, not just bare skin
POSITION_CORRIDOR_MIN_PX = 10.0  # floor for a degenerate/zero thickness reading (e.g. a bone sampled off the confident mask)
POSITION_CORRIDOR_MAX_PX = 45.0  # ceiling — stays comfortably under the 51.4-74.0px the two known-bad spots measured

# POSITION_CORRIDOR_MIN/MAX_PX and FAR_FOREGROUND_CEILING_PX above were all
# calibrated as flat pixel counts against one real photo (shoulder width
# 96.2px). Confirmed as a real, serious bug on a second, much higher-
# resolution photo (shoulder width 525.9px, 5.5x larger): EVERY bone's
# measured corridor radius hit the flat 45px ceiling and got truncated,
# because 45px was a generous allowance at the first photo's scale but a
# tiny fraction of a limb's real on-screen width at this one — the clip
# chopped real, solid torso fabric into ragged holes that weren't there
# without `bones` at all. Expressing both as ratios of the subject's own
# measured shoulder width (when available) fixes this the same way
# _bone_corridor_radii already fixes per-bone thickness: scale the
# allowance from something measured in THIS photo, not a constant tuned
# on a different one. Ratios below are exactly the flat constants divided
# by the first photo's own 96.2px shoulder width, so a photo at that same
# scale reproduces the original numbers unchanged.
POSITION_CORRIDOR_MIN_RATIO = POSITION_CORRIDOR_MIN_PX / 96.2
POSITION_CORRIDOR_MAX_RATIO = POSITION_CORRIDOR_MAX_PX / 96.2

# A real, confirmed gap the per-bone corridor above doesn't cover: it only
# resolves color-AMBIGUOUS pixels (see AMBIGUITY_MARGIN), but a real bug was
# found where bright wave-foam color near the subject's feet was
# CONFIDENTLY matched to a foreground cluster (margin 2.6-6.3, i.e. not even
# close to ambiguous) because no background cluster fit from the image
# border represented that specific bright tone. Confirmed by distance, not
# guessed: the foam streak's own pixels measured 60-150+ px from the
# nearest bone segment, while every real anatomy pixel on the same photo
# (checked across the whole subject, hair tips included) measured at most
# 60.6px. This ceiling is a position-only sanity check — independent of
# color — applied to EVERY foreground pixel (not just ambiguous ones):
# anything farther than this from every bone segment is rejected regardless
# of how confident the color match was. Deliberately much larger than
# POSITION_CORRIDOR_MAX_PX (45px) so it only catches gross outliers like
# this streak, never real but corridor-exceeding anatomy (hair, loose
# clothing) — 90px was tested directly against the same real photo: zero
# landmarks or real-subject regions lost, while a confirmed 1534px chunk of
# the foam-streak bug was correctly removed. (An alternative fix — enriching
# the background color-cluster fit with distant pixels instead of this flat
# geometric check — was tried first and reverted: it also killed the foam
# streak, but it pulled unrelated dark background tones into the same
# k-means fit, which then started matching real dark shirt fabric too,
# wrongly excluding 3135px of real torso that was correct before. A
# position-only check can't have that failure mode, since it never touches
# the color model.)
FAR_FOREGROUND_CEILING_PX = 90.0
FAR_FOREGROUND_CEILING_RATIO = FAR_FOREGROUND_CEILING_PX / 96.2  # see POSITION_CORRIDOR_MIN/MAX_RATIO


def _bone_corridor_radii(
    bones: "list[tuple[np.ndarray, np.ndarray]]",
    dist_map: np.ndarray,
    n_samples: int = 5,
    body_scale_px: "float | None" = None,
) -> "list[float]":
    """Each bone's own corridor radius, from the CONFIDENT mask's own
    Euclidean distance transform (`dist_map`) sampled at several points
    along that specific bone — not one flat number for every bone. Median
    of several interior samples (excluding the very ends, where a
    nearby joint's own thicker cross-section would bias a thin limb's
    reading) rather than a single point, so one unlucky sample landing in
    a gap doesn't set the whole bone's radius.

    `body_scale_px`: the subject's own measured size in THIS photo (e.g.
    shoulder width), used to scale the [min, max] clip range instead of
    the flat POSITION_CORRIDOR_MIN/MAX_PX — see those constants' own
    docstring for the real photo this was confirmed necessary on. None
    falls back to the flat pixel constants (e.g. no shoulder pair
    detected to measure a scale from)."""
    if body_scale_px:
        min_px = POSITION_CORRIDOR_MIN_RATIO * body_scale_px
        max_px = POSITION_CORRIDOR_MAX_RATIO * body_scale_px
    else:
        min_px, max_px = POSITION_CORRIDOR_MIN_PX, POSITION_CORRIDOR_MAX_PX
    radii = []
    h, w = dist_map.shape
    for a, b in bones:
        samples = []
        for t in np.linspace(0.2, 0.8, n_samples):
            pt = a + (b - a) * t
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                r = dist_map[y, x]
                if r > 0:
                    samples.append(r)
        thickness = float(np.median(samples)) if samples else 0.0
        radii.append(float(np.clip(thickness * POSITION_CORRIDOR_SCALE, min_px, max_px)))
    return radii


def _position_is_on_body(
    points_xy: np.ndarray, bones: "list[tuple[np.ndarray, np.ndarray]]", radii: "list[float]"
) -> np.ndarray:
    """For each (x, y) in `points_xy`, True if it's within that specific
    bone's own corridor radius (see _bone_corridor_radii) of the nearest
    bone segment (a straight line between two detected, anatomically-
    adjacent joints). See AMBIGUITY_MARGIN's docstring for why this only
    matters for color-ambiguous pixels, not as a replacement for the
    color check."""
    on_body = None
    for (a, b), radius in zip(bones, radii):
        ab = b - a
        length_sq = float(np.dot(ab, ab))
        if length_sq < 1e-9:
            d = np.linalg.norm(points_xy - a, axis=-1)
        else:
            t = np.clip(((points_xy - a) @ ab) / length_sq, 0, 1)
            closest = a + t[:, None] * ab
            d = np.linalg.norm(points_xy - closest, axis=-1)
        within = d <= radius
        on_body = within if on_body is None else (on_body | within)
    return on_body if on_body is not None else np.zeros(len(points_xy), dtype=bool)


def _two_sided_reclassify(
    rgb: np.ndarray,
    seed_mask: np.ndarray,
    foreground_clusters: int = DEFAULT_FOREGROUND_CLUSTERS,
    background_clusters: int = DEFAULT_BACKGROUND_CLUSTERS,
    bones: "list[tuple[np.ndarray, np.ndarray]] | None" = None,
    body_scale_px: "float | None" = None,
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
        dist_map = ndimage.distance_transform_edt(seed_mask)
        radii = _bone_corridor_radii(bones, dist_map, body_scale_px=body_scale_px)
        yy, xx = np.mgrid[0:h, 0:w]
        points_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
        on_body = _position_is_on_body(points_xy, bones, radii)
        result_mask[ambiguous] = on_body[ambiguous]
        # Sanity ceiling, not a replacement for the corridor above — see
        # FAR_FOREGROUND_CEILING_PX's docstring. Applied to every foreground
        # pixel, confident or not, since the bug it catches (wave foam
        # confidently matched as foreground) never even reaches `ambiguous`.
        ceiling_px = FAR_FOREGROUND_CEILING_RATIO * body_scale_px if body_scale_px else FAR_FOREGROUND_CEILING_PX
        within_ceiling = _position_is_on_body(
            points_xy, bones, [ceiling_px] * len(bones)
        )
        result_mask &= within_ceiling
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
    body_scale_px: "float | None" = None,
) -> np.ndarray:
    """`bones`: optional list of (joint_a, joint_b) pixel-coordinate pairs
    from the same photo's own detected pose (e.g. shoulder-to-elbow,
    hip-to-knee) — see _two_sided_reclassify's docstring. Deliberately
    plain coordinate pairs, not a landmarks.py type, so this module stays
    decoupled from pose-detection internals; the caller (pipeline.py)
    builds the list from whatever keypoints it already has.

    `body_scale_px`: the subject's own measured size in THIS photo (e.g.
    shoulder width in pixels) — see POSITION_CORRIDOR_MIN/MAX_RATIO's
    docstring for why the position-based corridor/ceiling need this
    instead of a flat pixel count to generalize across photo resolutions.
    None falls back to the flat pixel constants this was originally
    calibrated with."""
    mask = _largest_filled_blob(_threshold_mask(rgb, border_width, threshold, background_clusters), body_scale_px)
    if not refine:
        return mask
    return _largest_filled_blob(
        _two_sided_reclassify(rgb, mask, foreground_clusters, background_clusters, bones, body_scale_px),
        body_scale_px,
    )
