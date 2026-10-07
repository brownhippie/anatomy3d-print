"""2D body keypoint detection via MediaPipe Pose."""
from dataclasses import dataclass

import cv2
import mediapipe as mp

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
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        result = pose.process(rgb)

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

    if len(joints) < 6:
        raise RuntimeError(
            "Too few confident keypoints detected — use a photo where the full body "
            "(shoulders, hips, knees) is visible and unobstructed."
        )

    return DetectedKeypoints(image_width=width, image_height=height, joints=joints)
