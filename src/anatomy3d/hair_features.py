"""Real per-photo hair shape via MediaPipe's Hair Segmenter — same
Apache-2.0 MediaPipe model family as pose/face detection, not a new
dependency or license to clear. Gives a per-pixel hair mask from the
photo, used in procedural_body.py to shape a stylized hair volume from
the photo's own hairline instead of either ignoring hair entirely or
guessing a generic cap shape.

This is a segmentation model (a fixed foreground/background call per
pixel from the real photo), not a generative one — same category as
Face Landmarker, not the kind of model that reopens this project's
licensing concerns (see README's "Why from scratch" section).
"""
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .preprocess import load_image_rgb

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "hair_segmenter/float32/latest/hair_segmenter.tflite"
)
MODEL_CACHE_PATH = Path.home() / ".cache" / "anatomy3d-print" / "hair_segmenter.tflite"

# Confirmed directly against a real photo (MediaPipe's own portrait.jpg
# sample): category 1 is hair, category 0 is everything else — a binary
# mask, not an arbitrary label id guessed from documentation.
HAIR_CATEGORY = 1


def _ensure_model() -> str:
    if MODEL_CACHE_PATH.exists() and MODEL_CACHE_PATH.stat().st_size > 100_000:
        return str(MODEL_CACHE_PATH)
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(MODEL_CACHE_PATH) + ".part"
    urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, MODEL_CACHE_PATH)
    return str(MODEL_CACHE_PATH)


_segmenter = None  # lazy singleton: model loading is slow, worth reusing across calls


def _get_segmenter() -> mp_vision.ImageSegmenter:
    global _segmenter
    if _segmenter is None:
        base_options = mp_python.BaseOptions(model_asset_path=_ensure_model())
        options = mp_vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        _segmenter = mp_vision.ImageSegmenter.create_from_options(options)
    return _segmenter


def detect_hair_mask(image_path: str, max_dimension: "int | None" = None) -> Optional[np.ndarray]:
    """Returns a boolean (H, W) mask, True where the photo shows hair, in
    the same pixel coordinate system as the photo loaded with the same
    `max_dimension` (None = full resolution, same reasoning as
    face_features.py: hair is a small, fine-detailed region of a
    full-body frame and benefits from the extra pixels). Returns None
    (does not raise) if nothing is detected as hair — a bald head, a
    hat, or a failed read are all legitimate "no hair volume to add"
    outcomes for an optional refinement, not errors."""
    prepared = load_image_rgb(image_path, max_dimension=max_dimension)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=prepared.rgb)
    result = _get_segmenter().segment(mp_image)

    category_mask = result.category_mask.numpy_view().squeeze()
    hair_mask = category_mask == HAIR_CATEGORY
    if not hair_mask.any():
        return None
    return hair_mask
