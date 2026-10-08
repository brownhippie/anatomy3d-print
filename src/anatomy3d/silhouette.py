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


def _threshold_mask(rgb: np.ndarray, border_width: int = 12, threshold: float = 32.0) -> np.ndarray:
    border_pixels = np.concatenate([
        rgb[:border_width].reshape(-1, 3),
        rgb[-border_width:].reshape(-1, 3),
        rgb[:, :border_width].reshape(-1, 3),
        rgb[:, -border_width:].reshape(-1, 3),
    ]).astype(np.float64)
    background = np.median(border_pixels, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float64) - background, axis=-1)
    return dist > threshold


def _gaussian_reclassify(rgb: np.ndarray, seed_mask: np.ndarray, max_samples: int = 20000) -> np.ndarray:
    """Re-decide every pixel by which of two fitted color distributions
    (foreground, background) it's actually closer to, seeded from
    `seed_mask` — a single-pass, dependency-free approximation of
    GrabCut's color-model step."""
    rgb64 = rgb.astype(np.float64)
    fg_pixels = rgb64[seed_mask]
    bg_pixels = rgb64[~seed_mask]
    if len(fg_pixels) < 10 or len(bg_pixels) < 10:
        return seed_mask  # not enough signal to fit two distributions

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
) -> np.ndarray:
    mask = _largest_filled_blob(_threshold_mask(rgb, border_width, threshold))
    if not refine:
        return mask
    return _largest_filled_blob(_gaussian_reclassify(rgb, mask))
