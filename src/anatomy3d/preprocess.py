"""Decode an input photo once into the normalized form the rest of the
pipeline needs: correctly oriented, RGB, and capped at a working
resolution.

Two real problems this fixes, not just a reformat:
- Phone photos carry an EXIF orientation tag; a naive decode can hand the
  pipeline a sideways or upside-down image, which throws off every "up is
  positive y" assumption downstream.
- Feeding a full 4000px phone photo straight into pose detection is pure
  waste — MediaPipe's accuracy doesn't improve past a modest resolution,
  so anything larger just slows detection down for nothing.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass
class PreparedImage:
    rgb: np.ndarray  # (H, W, 3) uint8
    width: int
    height: int


def load_image_rgb(
    path: str,
    max_dimension: int = 1280,
    sharpen: bool = True,
    normalize_contrast: bool = True,
) -> PreparedImage:
    try:
        im = Image.open(path)
    except Exception as exc:
        raise FileNotFoundError(f"Could not read image: {path}") from exc

    im = ImageOps.exif_transpose(im)  # correct phone-camera rotation
    im = im.convert("RGB")

    scale = max_dimension / max(im.width, im.height)
    if scale < 1.0:
        new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        # LANCZOS keeps edges crisper on downsampling than bilinear, which
        # matters here: the sharpening pass below only has real edges to
        # work with if the resize didn't mush them first.
        im = im.resize(new_size, Image.LANCZOS)

    if normalize_contrast:
        # Pulls washed-out or underexposed phone photos back to full
        # dynamic range per channel before detection runs on them.
        im = ImageOps.autocontrast(im, cutoff=1)

    if sharpen:
        # Unsharp mask: boosts edge contrast without the haloing a naive
        # sharpen kernel produces, so joint/silhouette edges stay clean for
        # both pose detection and the silhouette extraction used in
        # multi-photo mode.
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    return PreparedImage(rgb=np.asarray(im), width=im.width, height=im.height)
