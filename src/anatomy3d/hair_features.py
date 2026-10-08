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


# How many shoulder-widths the hair mask's own centroid may sit from the
# nose before being rejected as implausible. Not finely tuned (one real
# failure case to calibrate against so far — see below), but a real,
# checked bound rather than skipped entirely: confirmed directly that the
# segmenter can mislabel an entirely different body region as "hair" on a
# real photo (a dark tank top against bright skin/sky, on a photo this
# model otherwise handles fine for pose/face) — centroid 1.28 shoulder-
# widths from the nose, nowhere near the head. A real hairstyle's own
# mask, however unusual (long, windblown, a wide afro), should center
# much closer to the head than that; 0.75 gives real variation room
# while still catching a mask that's actually centered on the torso.
MAX_HAIR_CENTROID_SHOULDER_WIDTHS = 0.75


def detect_hair_mask(
    image_path: str, max_dimension: "int | None" = None, keypoints=None
) -> Optional[np.ndarray]:
    """Returns a boolean (H, W) mask, True where the photo shows hair, in
    the same pixel coordinate system as the photo loaded with the same
    `max_dimension` (None = full resolution, same reasoning as
    face_features.py: hair is a small, fine-detailed region of a
    full-body frame and benefits from the extra pixels). Returns None
    (does not raise) if nothing is detected as hair — a bald head, a
    hat, or a failed read are all legitimate "no hair volume to add"
    outcomes for an optional refinement, not errors.

    `keypoints`: optional DetectedKeypoints (landmarks.py) from the same
    photo. When given, sanity-checks the segmenter's output against where
    the head actually is — a real, confirmed failure mode, not a
    hypothetical: this exact model mislabeled a chunk of torso/clothing
    as hair on a real test photo, which would otherwise have been fed
    straight into the mesh as if it were real hair data. Without
    `keypoints`, this check is skipped (same None-on-nothing-detected
    behavior as before)."""
    prepared = load_image_rgb(image_path, max_dimension=max_dimension)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=prepared.rgb)
    result = _get_segmenter().segment(mp_image)

    category_mask = result.category_mask.numpy_view().squeeze()
    hair_mask = category_mask == HAIR_CATEGORY
    if not hair_mask.any():
        return None

    if keypoints is not None and "nose" in keypoints.joints and "left_shoulder" in keypoints.joints and "right_shoulder" in keypoints.joints:
        # keypoints may have been measured on a differently-sized load of
        # this same photo (detect_pose_landmarks caps at 1280px; this
        # function defaults to uncapped) — convert from keypoints' own
        # pixel space into this call's actual (prepared.width-sized) one,
        # not the other way around.
        scale = prepared.width / keypoints.image_width
        nose = np.array(keypoints.joints["nose"][:2]) * scale
        shoulder_w = np.linalg.norm(
            np.array(keypoints.joints["left_shoulder"][:2]) - np.array(keypoints.joints["right_shoulder"][:2])
        ) * scale
        ys, xs = np.where(hair_mask)
        centroid = np.array([xs.mean(), ys.mean()])
        dist_in_shoulder_widths = np.linalg.norm(centroid - nose) / shoulder_w if shoulder_w > 1e-6 else 0.0
        if dist_in_shoulder_widths > MAX_HAIR_CENTROID_SHOULDER_WIDTHS:
            print(
                f"Note: hair segmenter's result rejected — its centroid sits "
                f"{dist_in_shoulder_widths:.2f} shoulder-widths from the nose "
                f"(max {MAX_HAIR_CENTROID_SHOULDER_WIDTHS}), nowhere near the head; "
                "the head will stay bare instead of using a measurement that isn't hair."
            )
            return None

    return hair_mask
