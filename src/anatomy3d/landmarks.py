"""2D body keypoint detection via MediaPipe Pose.

Uses the modern MediaPipe Tasks API (`mediapipe.tasks.python.vision`), not
the older `mediapipe.solutions.pose.Pose` API — confirmed by actually
running this against a real photo, not an assumption: the legacy
`solutions` API this module originally used does not exist in any
pip-installable mediapipe build for Python 3.13 (only `Image`,
`ImageFormat`, and `tasks` are exposed at the top level). Every prior test
in this project mocked `mediapipe.solutions` out entirely, so this gap
went unnoticed until the pipeline was run against real MediaPipe for the
first time.

Unlike the old API, the Tasks API doesn't bundle a model inside the pip
package — it needs a `.task` model file, fetched here from MediaPipe's own
model bucket (Apache-2.0, same as the rest of MediaPipe, no license gate)
and cached locally after the first run.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .preprocess import load_image_rgb
from .safety import mad_margin_above_minimum

# Hard floor: fitting falls apart below this. Also doubles as the MAD
# margin's reference point, so "just barely enough" (margin 0) and "not
# enough" (raise) share one number instead of two unrelated constants.
MIN_KEYPOINTS = 6

# MediaPipe BlazePose landmark indices for the joints we fit against —
# unchanged from the old API: both expose the same 33-point topology with
# the same indices, only how you get a detector and run it differs.
MEDIAPIPE_JOINT_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_CACHE_PATH = Path.home() / ".cache" / "anatomy3d-print" / "pose_landmarker_full.task"


def _ensure_model() -> str:
    if MODEL_CACHE_PATH.exists() and MODEL_CACHE_PATH.stat().st_size > 1_000_000:
        return str(MODEL_CACHE_PATH)
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(MODEL_CACHE_PATH) + ".part"
    urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, MODEL_CACHE_PATH)
    return str(MODEL_CACHE_PATH)


_detector = None  # lazy singleton: model loading is slow, worth reusing across calls


def _get_detector() -> mp_vision.PoseLandmarker:
    global _detector
    if _detector is None:
        base_options = mp_python.BaseOptions(model_asset_path=_ensure_model())
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        _detector = mp_vision.PoseLandmarker.create_from_options(options)
    return _detector


@dataclass
class DetectedKeypoints:
    image_width: int
    image_height: int
    # joint_name -> (x_px, y_px, visibility)
    joints: dict


# Two anatomical-plausibility checks, originally built for person_segmenter.py's
# separate pose-reach-limiting detector and ported here so the pose that
# actually drives the 3D mesh gets the same protection — the
# visibility/presence filter above already caught one real hallucination
# class (off-frame joints scored high on visibility but low on presence)
# but not this one: a real photo of a single hand against plain background
# (no body at all) got a 12-joint skeleton invented on it — nose, both
# shoulders, an elbow, wrist, hips, knees — all individually sitting on
# real "hand" pixels (nothing here checks mask corroboration, unlike the
# 2D reach-limiter, since there's no segmenter mask available at this
# point in the pipeline), but its knee landed ABOVE its hip — anatomically
# backwards.
#
# The fix: a real photographed person's joints run top-to-bottom in a
# fixed order — shoulders above hips above knees above ankles. Checked
# directly against 5 different real detected poses (standing, blurred,
# two-people, torso-cropped): zero exceptions, every single one.
_POSE_ORDER_CHAIN = ("shoulder", "hip", "knee", "ankle")

# A second, independent signal: lowering detection confidence to catch
# real but small/distant people also let a hallucinated skeleton through
# the ordering check above (an off-by-coincidence-vertical fake chain) —
# a fabricated shoulder pair only 113px apart in x but 706px apart in y,
# an almost-vertical "shoulder line" at 80.9 degrees from horizontal.
# Real shoulders run side-to-side, not stacked: checked directly against
# 6 real detected poses (including an awkward twisted lunge) and every
# one measured under 4 degrees. Hip angle was tried and rejected the same
# way: a real twisted lunge measured a real 60.5-degree hip angle (hips
# rotate independently of the camera; shoulders much less so), not a
# separable signal, unlike shoulder angle.
_POSE_MAX_SHOULDER_ANGLE_DEG = 30.0


def _pose_vertically_ordered(joints: dict) -> bool:
    ys = []
    for part in _POSE_ORDER_CHAIN:
        left, right = joints.get(f"left_{part}"), joints.get(f"right_{part}")
        pts = [p[1] for p in (left, right) if p is not None]
        ys.append(np.mean(pts) if pts else None)
    present = [y for y in ys if y is not None]
    # need at least 2 rungs of the chain present to check anything
    if len(present) < 2:
        return True
    prev = None
    for y in ys:
        if y is None:
            continue
        if prev is not None and y < prev:
            return False
        prev = y
    return True


def _pose_shoulders_level(joints: dict) -> bool:
    ls, rs = joints.get("left_shoulder"), joints.get("right_shoulder")
    if ls is None or rs is None:
        return True  # nothing to check without both shoulders
    dx, dy = rs[0] - ls[0], rs[1] - ls[1]
    angle = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9))
    return angle <= _POSE_MAX_SHOULDER_ANGLE_DEG


def detect_pose_landmarks(image_path: str, min_visibility: float = 0.5, min_presence: float = 0.5) -> DetectedKeypoints:
    prepared = load_image_rgb(image_path)
    width, height = prepared.width, prepared.height

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=prepared.rgb)
    result = _get_detector().detect(mp_image)

    if not result.pose_landmarks:
        raise RuntimeError(
            "No person detected in the image. Use a clear, front-facing full-body photo."
        )

    landmarks = result.pose_landmarks[0]
    joints = {}
    for name, idx in MEDIAPIPE_JOINT_INDEX.items():
        lm = landmarks[idx]
        # `visibility` (is this point occluded, given it's in frame) and
        # `presence` (is this point in frame at all) are two different
        # MediaPipe scores — confirmed as a real, separate gap, not a
        # guess: on a tightly-cropped portrait, the hip joints (actually
        # below the photo's bottom edge) scored visibility 0.85-0.90 (well
        # past the 0.5 cutoff) while their own presence scored only
        # 0.31-0.43 — visibility alone let two hallucinated, off-frame
        # joints straight through. Filtering on both catches that case;
        # visibility alone did not.
        if lm.visibility < min_visibility or lm.presence < min_presence:
            continue
        joints[name] = (lm.x * width, lm.y * height, lm.visibility)

    if len(joints) < MIN_KEYPOINTS:
        raise RuntimeError(
            f"Too few confident keypoints detected ({len(joints)}, need at least "
            f"{MIN_KEYPOINTS}) — use a photo where the full body (shoulders, hips, "
            "knees) is visible and unobstructed."
        )

    result = mad_margin_above_minimum(len(joints), MIN_KEYPOINTS, min_margin=0.3)
    if not result.ok:
        print(
            f"Warning: only {len(joints)} keypoints detected, {result.margin:.0%} "
            f"above the {MIN_KEYPOINTS}-keypoint minimum — pose estimation may be "
            "unreliable. A clearer full-body photo will give a better result."
        )

    if not _pose_vertically_ordered(joints):
        raise RuntimeError(
            "Detected pose fails an anatomical plausibility check (joints out of "
            "top-to-bottom order — e.g. a knee above a hip). This usually means "
            "the detector hallucinated a skeleton rather than finding a real one; "
            "use a clearer, unobstructed full-body photo."
        )
    if not _pose_shoulders_level(joints):
        raise RuntimeError(
            "Detected pose fails an anatomical plausibility check (shoulder line "
            f"more than {_POSE_MAX_SHOULDER_ANGLE_DEG:.0f} degrees from horizontal). "
            "This usually means the detector hallucinated a skeleton rather than "
            "finding a real one; use a clearer, unobstructed full-body photo."
        )

    return DetectedKeypoints(image_width=width, image_height=height, joints=joints)
