"""Extract a subject silhouette from a photo against a roughly plain
background.

Technique: sample the image border as the background color, threshold by
color distance from it, keep the largest connected blob, fill interior
holes. Classical, no learned model — works when the subject is shot
against a reasonably uniform wall/backdrop, which is the realistic case
for someone taking their own turnaround photos for this.
"""
import numpy as np
from scipy import ndimage


def extract_silhouette(rgb: np.ndarray, border_width: int = 12, threshold: float = 32.0) -> np.ndarray:
    h, w = rgb.shape[:2]
    border_pixels = np.concatenate([
        rgb[:border_width].reshape(-1, 3),
        rgb[-border_width:].reshape(-1, 3),
        rgb[:, :border_width].reshape(-1, 3),
        rgb[:, -border_width:].reshape(-1, 3),
    ]).astype(np.float64)
    background = np.median(border_pixels, axis=0)

    dist = np.linalg.norm(rgb.astype(np.float64) - background, axis=-1)
    mask = dist > threshold

    labeled, n = ndimage.label(mask)
    if n == 0:
        raise RuntimeError(
            "No subject found against the background — use a photo with a "
            "plain, evenly lit backdrop behind the person."
        )
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    largest = 1 + int(np.argmax(sizes))
    mask = labeled == largest

    mask = ndimage.binary_fill_holes(mask)
    return mask
