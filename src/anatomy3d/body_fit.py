"""Optional, higher-fidelity body fitting via SMPL-X — NOT used by the
default pipeline.

SMPL-X's shape space (`betas`) is a PCA basis learned from thousands of real
body scans, so it already encodes plausible human proportions. Regularizing
`betas` toward zero during optimization keeps the fitted body inside that
learned distribution instead of letting an under-constrained single-photo
fit drift into an anatomically implausible shape.

This module is kept for research/non-commercial experimentation only: it
requires SMPL-X model weights, which are licensed for non-commercial
scientific research use (see https://smpl-x.is.tue.mpg.de/). Commercial use
requires a separate license from Meshcapade. The default pipeline
(`procedural_body.py`) does not use this module or depend on it.

Install extras with `pip install -r requirements-smplx.txt` before using
this module. This is a simplified, from-scratch reimplementation of the
general idea behind SMPLify-X-style fitting (2D-keypoint-driven
optimization of a SMPL family model) — not a port of that codebase.
"""
import numpy as np
import smplx
import torch

from .landmarks import DetectedKeypoints
from .mesh_types import BodyMesh as FittedBody
from .safety import mad_margin_above_minimum

MIN_FIT_KEYPOINTS = 6

# Standard SMPL/SMPL-X body joint order (first 22 joints of the model output).
SMPLX_JOINT_INDEX = {
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "nose": 15,  # approximated with the head joint
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


def _build_targets(keypoints: DetectedKeypoints, device: torch.device):
    names = [n for n in SMPLX_JOINT_INDEX if n in keypoints.joints]
    if len(names) < MIN_FIT_KEYPOINTS:
        raise RuntimeError(
            f"Not enough overlapping keypoints to fit a body model ({len(names)}, "
            f"need at least {MIN_FIT_KEYPOINTS})."
        )

    result = mad_margin_above_minimum(len(names), MIN_FIT_KEYPOINTS, min_margin=0.3)
    if not result.ok:
        print(
            f"Warning: only {len(names)} overlapping keypoints, {result.margin:.0%} "
            f"above the {MIN_FIT_KEYPOINTS}-keypoint minimum — the SMPL-X fit may "
            "be unreliable."
        )

    smplx_idx = torch.tensor([SMPLX_JOINT_INDEX[n] for n in names], device=device)

    h = keypoints.image_height
    w = keypoints.image_width
    targets = []
    weights = []
    for n in names:
        px, py, vis = keypoints.joints[n]
        tx = (px - w / 2.0) / h
        ty = (h / 2.0 - py) / h  # flip so "up" is positive, matching the body model
        targets.append((tx, ty))
        weights.append(vis)

    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return smplx_idx, targets, weights


def fit_body(
    keypoints: DetectedKeypoints,
    smplx_model_dir: str,
    gender: str = "neutral",
    num_betas: int = 10,
    coarse_iters: int = 300,
    refine_iters: int = 800,
    device: str = "cpu",
) -> FittedBody:
    device_t = torch.device(device)

    model = smplx.create(
        smplx_model_dir,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=num_betas,
    ).to(device_t)

    smplx_idx, targets, weights = _build_targets(keypoints, device_t)

    betas = torch.zeros(1, num_betas, device=device_t, requires_grad=True)
    body_pose = torch.zeros(1, 63, device=device_t, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=device_t, requires_grad=True)
    scale = torch.tensor([1.0], device=device_t, requires_grad=True)
    trans2d = torch.zeros(2, device=device_t, requires_grad=True)

    def project(joints):
        xy = joints[smplx_idx, :2]
        return scale * xy + trans2d

    def keypoint_loss(joints):
        proj = project(joints)
        err = ((proj - targets) ** 2).sum(dim=-1)
        return (weights * err).sum() / weights.sum()

    # Stage 1: camera + global orientation only, body held at the mean shape/pose.
    optim1 = torch.optim.Adam([global_orient, scale, trans2d], lr=0.05)
    for _ in range(coarse_iters):
        optim1.zero_grad()
        out = model(betas=betas, body_pose=body_pose, global_orient=global_orient)
        loss = keypoint_loss(out.joints[0])
        loss.backward()
        optim1.step()

    # Stage 2: refine shape and pose jointly, with priors keeping both plausible.
    optim2 = torch.optim.Adam(
        [betas, body_pose, global_orient, scale, trans2d], lr=0.02
    )
    for _ in range(refine_iters):
        optim2.zero_grad()
        out = model(betas=betas, body_pose=body_pose, global_orient=global_orient)
        loss = (
            keypoint_loss(out.joints[0])
            + 0.005 * (betas**2).mean()
            + 0.0005 * (body_pose**2).mean()
        )
        loss.backward()
        optim2.step()

    with torch.no_grad():
        out = model(betas=betas, body_pose=body_pose, global_orient=global_orient)
        vertices = out.vertices[0].cpu().numpy()
        faces = model.faces.astype(np.int64)

    return FittedBody(vertices=vertices, faces=faces)
