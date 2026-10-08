"""Real per-photo face geometry (eyes, nose, chin) via MediaPipe Face
Landmarker — the same Apache-2.0 MediaPipe model family as the pose
landmarker already used in landmarks.py, not a new dependency or a new
license to clear. Face detection is a bonus refinement on top of the body
figure, not a requirement: if no face is detected (profile shot, face out
of frame, low-resolution crop), callers should fall back to the plain
body with no face detail rather than failing the whole pipeline.

The 468/478-point canonical face mesh topology this indexes into is
MediaPipe's own public index map, not something guessed here — every
index below was checked against a real detected face (MediaPipe's own
`portrait.jpg` sample image) before use: nose tip (1) sits between the
eyes and above the chin as expected, chin (152) is the lowest point on
the face, forehead (10) the highest, and the eye/jaw index pairs are
left-right symmetric around the face centerline, confirming these are
the landmarks they're named for rather than an off-by-some-index mixup.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .preprocess import load_image_rgb

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_CACHE_PATH = Path.home() / ".cache" / "anatomy3d-print" / "face_landmarker.task"

# Canonical face-mesh indices used here, verified against a real photo (see
# module docstring) rather than assumed from documentation alone.
NOSE_TIP = 1
NOSE_BRIDGE = 6
CHIN = 152
RIGHT_EYE = (33, 133, 159, 145)  # outer, inner, upper lid, lower lid
LEFT_EYE = (362, 263, 386, 374)  # inner, outer, upper lid, lower lid

_NEEDED_INDICES = {NOSE_TIP, NOSE_BRIDGE, CHIN, *RIGHT_EYE, *LEFT_EYE}


def _ensure_model() -> str:
    if MODEL_CACHE_PATH.exists() and MODEL_CACHE_PATH.stat().st_size > 1_000_000:
        return str(MODEL_CACHE_PATH)
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(MODEL_CACHE_PATH) + ".part"
    urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, MODEL_CACHE_PATH)
    return str(MODEL_CACHE_PATH)


_detector = None  # lazy singleton: model loading is slow, worth reusing across calls


def _get_detector() -> mp_vision.FaceLandmarker:
    global _detector
    if _detector is None:
        base_options = mp_python.BaseOptions(model_asset_path=_ensure_model())
        options = mp_vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        _detector = mp_vision.FaceLandmarker.create_from_options(options)
    return _detector


@dataclass
class FaceKeypoints:
    image_width: int
    image_height: int
    # name -> (x_px, y_px, z_raw) — z_raw is MediaPipe's own normalized face
    # depth (roughly the same scale as x, smaller/more negative = closer to
    # the camera); converting it to this project's local mm-ish units
    # happens in procedural_body.py, alongside the same conversion for pose
    # keypoints, to keep all SDF-coordinate-space logic in one place.
    points: dict


def detect_face_landmarks(image_path: str, max_dimension: "int | None" = None) -> Optional[FaceKeypoints]:
    """Returns None (does not raise) if no face is detected — face detail
    is an optional refinement on top of the body figure, so a photo where
    the face isn't usable should fall back gracefully, not fail the run.

    `max_dimension=None` (the default) uses the photo at full resolution,
    unlike pose detection's 1280px cap in landmarks.py. That cap is fine
    for pose — MediaPipe's landmark accuracy there doesn't improve past a
    modest resolution relative to the whole photo — but a face is often a
    small fraction of a full-body frame to begin with; downscaling it
    further before it ever reaches the face landmarker throws away
    exactly the pixels that determine whether a small face is usable at
    all (see build_body_mesh's adaptive-resolution skip path, which exists
    for faces that don't have enough pixels to work with)."""
    prepared = load_image_rgb(image_path, max_dimension=max_dimension)
    width, height = prepared.width, prepared.height

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=prepared.rgb)
    result = _get_detector().detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    points = {}
    for idx in _NEEDED_INDICES:
        lm = landmarks[idx]
        points[idx] = (lm.x * width, lm.y * height, lm.z)

    return FaceKeypoints(image_width=width, image_height=height, points=points)
