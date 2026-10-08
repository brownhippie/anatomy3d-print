"""2D body keypoint detection via MediaPipe Pose."""
from dataclasses import dataclass

import mediapipe as mp

from .preprocess import load_image_rgb
from .safety import mad_margin_above_minimum

# Hard floor: fitting falls apart below this. Also doubles as the MAD
# margin's reference point, so "just barely enough" (margin 0) and "not
# enough" (raise) share one number instead of two unrelated constants.
MIN_KEYPOINTS = 6

# MediaPipe BlazePose landmark indices for the joints we fit against.
MEDIAPIPE_JOINT_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class DetectedKeypoints:
    image_width: int
    image_height: int
    # joint_name -> (x_px, y_px, visibility)
    joints: dict


def detect_pose_landmarks(image_path: str, min_visibility: float = 0.5) -> DetectedKeypoints:
    prepared = load_image_rgb(image_path)
    width, height = prepared.width, prepared.height

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        result = pose.process(prepared.rgb)

    if not result.pose_landmarks:
        raise RuntimeError(
            "No person detected in the image. Use a clear, front-facing full-body photo."
        )

    landmarks = result.pose_landmarks.landmark
    joints = {}
    for name, idx in MEDIAPIPE_JOINT_INDEX.items():
        lm = landmarks[idx]
        if lm.visibility < min_visibility:
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

    return DetectedKeypoints(image_width=width, image_height=height, joints=joints)
