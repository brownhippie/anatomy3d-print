"""Extract a subject silhouette from a photo against a roughly plain
background, in two stages that each correct the one before it:

1. Threshold: sample the image border as the background color, keep
   pixels far enough from it, take the largest connected blob. Cheap,
   but a single flat cutoff misclassifies shadowed or near-background-
   colored regions.
2. Statistical reclassify: fit foreground/background color Gaussians from
   stage 1's result, then reclassify every pixel by which distribution it
   actually fits best. Same core idea as GrabCut's color model, without
   pulling in OpenCV for it.

A third stage — snapping the boundary to image edges with an active
contour (Kass et al. 1988) — was tried and reverted. On a figure with a
narrow neck and separated legs, the contour's smoothness term pulled
straight across both narrow points, deleting the head and most of the
legs; a coarse area-ratio sanity check didn't catch it because the lost
area wasn't large enough to trip it. Revisit only with a shape-aware check
(e.g. compare the bounding box, not just area) and testing against real
photos, not just clean synthetic edges.
"""
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


def _largest_filled_blob(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        raise RuntimeError(
            "No subject found against the background — use a photo with a "
            "plain, evenly lit backdrop behind the person."
        )
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    largest = 1 + int(np.argmax(sizes))
    return ndimage.binary_fill_holes(labeled == largest)


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

    # A near-perfectly flat background (every border pixel almost
    # identical) has nothing for k-means to split — forcing k>1 clusters
    # onto it just produces a degenerate empty-cluster warning for no
    # benefit, so skip straight to the single-cluster case there.
    k = min(background_clusters, len(border_pixels))
    if k > 1 and np.std(border_pixels, axis=0).max() < 1.0:
        k = 1
    try:
        centroids, labels = kmeans2(border_pixels, k, seed=0, minit="++")
    except Exception:
        centroids = np.median(border_pixels, axis=0, keepdims=True)
        labels = np.zeros(len(border_pixels), dtype=int)

    flat = rgb.astype(np.float64).reshape(-1, 3)
    maha_per_cluster = []
    for i in range(len(centroids)):
        cluster_pixels = border_pixels[labels == i]
        if len(cluster_pixels) < 4:
            # Too few samples to fit a covariance — fall back to a plain
            # Euclidean distance (scaled to roughly match Mahalanobis
            # units) for this cluster rather than skip it entirely.
            maha_per_cluster.append(np.linalg.norm(flat - centroids[i], axis=-1) / threshold * MAHALANOBIS_THRESHOLD)
            continue
        cov = np.cov(cluster_pixels, rowvar=False) + np.eye(3) * 1e-3
        prec = np.linalg.inv(cov)
        diff = flat - centroids[i]
        maha = np.sqrt(np.einsum("ij,jk,ik->i", diff, prec, diff))
        maha_per_cluster.append(maha)

    min_maha = np.min(np.stack(maha_per_cluster, axis=0), axis=0).reshape(rgb.shape[:2])
    return min_maha > MAHALANOBIS_THRESHOLD


def _gaussian_reclassify(rgb: np.ndarray, seed_mask: np.ndarray, max_samples: int = 20000) -> np.ndarray:
    """Re-decide every pixel by which of two fitted color distributions
    (foreground, background) it's actually closer to, seeded from
    `seed_mask` — a single-pass, dependency-free approximation of
    GrabCut's color-model step."""
    rgb64 = rgb.astype(np.float64)
    fg_pixels = rgb64[seed_mask]
    bg_pixels = rgb64[~seed_mask]
    if len(fg_pixels) < MIN_FIT_SAMPLES or len(bg_pixels) < MIN_FIT_SAMPLES:
        return seed_mask  # not enough signal to fit two distributions

    smaller = min(len(fg_pixels), len(bg_pixels))
    result = mad_margin_above_minimum(smaller, RECOMMENDED_FIT_SAMPLES, min_margin=0.0)
    if not result.ok:
        print(
            f"Note: color model fit from only {smaller} pixels of the smaller "
            f"class, {-result.margin:.0%} below the {RECOMMENDED_FIT_SAMPLES}-pixel "
            "comfort level — reclassification may be noisy on a small or "
            "tightly-cropped subject."
        )

    rng = np.random.default_rng(0)

    def fit(pixels):
        if len(pixels) > max_samples:
            idx = rng.choice(len(pixels), max_samples, replace=False)
            pixels = pixels[idx]
        mean = pixels.mean(axis=0)
        cov = np.cov(pixels, rowvar=False) + np.eye(3) * 1e-3
        return mean, np.linalg.inv(cov), np.linalg.slogdet(cov)[1]

    fg_mean, fg_prec, fg_logdet = fit(fg_pixels)
    bg_mean, bg_prec, bg_logdet = fit(bg_pixels)

    flat = rgb64.reshape(-1, 3)

    def log_likelihood(pixels, mean, prec, logdet):
        diff = pixels - mean
        maha = np.einsum("ij,jk,ik->i", diff, prec, diff)
        return -0.5 * (maha + logdet)

    fg_ll = log_likelihood(flat, fg_mean, fg_prec, fg_logdet)
    bg_ll = log_likelihood(flat, bg_mean, bg_prec, bg_logdet)
    return (fg_ll > bg_ll).reshape(rgb.shape[:2])


def extract_silhouette(
    rgb: np.ndarray,
    border_width: int = 12,
    threshold: float = 32.0,
    refine: bool = True,
    background_clusters: int = DEFAULT_BACKGROUND_CLUSTERS,
) -> np.ndarray:
    mask = _largest_filled_blob(_threshold_mask(rgb, border_width, threshold, background_clusters))
    if not refine:
        return mask
    return _largest_filled_blob(_gaussian_reclassify(rgb, mask))
