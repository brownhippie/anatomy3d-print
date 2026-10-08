import os
from typing import List, Optional, Union

from .face_features import detect_face_landmarks
from .landmarks import detect_pose_landmarks
from .mesh_export import export_obj, to_trimesh
from .preprocess import load_image_rgb
from .print_prep import export_stl
from .procedural_body import build_body_mesh
from .silhouette import extract_silhouette
from .visual_hull import SilhouetteView, carve_visual_hull


def _default_angles(n: int) -> List[float]:
    """Calibrated against an analytic ellipsoid with known ground-truth
    volume (depth axis deliberately the shape's short axis, so depth
    error shows up clearly): naive even-spacing from 0 degrees can land
    far from 90/270, the views that actually constrain depth. For 3
    views, even-spacing gives [0, 120, 240] (nearest to 90 is 30 degrees
    off) and measured +45% error on the depth extent; swapping to
    [0, 90, 270] (skip the back, keep both sides) measured -3% on the
    same shape. 4 already lands on all four cardinal angles via even
    spacing, so it's left alone; 5+ is also left on even spacing since
    the dense angular coverage from more views keeps any single gap
    small regardless of exact placement.
    """
    if n == 2:
        return [0.0, 90.0]
    if n == 3:
        return [0.0, 90.0, 270.0]
    return [i * 360.0 / n for i in range(n)]


def run_pipeline(
    images: Union[str, List[str]],
    out_stl_path: str,
    target_height_mm: float = 150.0,
    angles_deg: Optional[List[float]] = None,
    use_depth: bool = False,
) -> None:
    """One photo -> the capsule/SDF body builder (a geometric guess).
    Two or more photos (different angles around the subject) -> visual hull
    carving from their silhouettes, which uses the subject's *actual* shape
    instead of a guess. `angles_deg` defaults to evenly spaced rotation
    angles in the order the photos are given (front first), matching a
    turntable-style capture; pass explicit angles if you shot something
    else.

    `use_depth`: single-photo mode only. Sculpts the front surface using
    real per-pixel depth (Depth Anything V2 Small, see depth_source.py)
    instead of the flat symmetric-thickness guess. Opt-in and off by
    default, same as the SMPL-X path — it's a real, sizeable extra
    dependency (torch + transformers), not something that should change
    behavior just because it happens to be installed in a given
    environment. Requires `pip install -r requirements-depth.txt`.

    Single-photo mode also tries face detail (nose, chin, eye sockets,
    face_features.py) automatically — unlike `use_depth`, this needs no
    opt-in flag, since it's the same MediaPipe dependency already
    required for pose detection, not a new one. A missing/unusable face
    or a failed model fetch falls back to the generic head shape rather
    than failing the run.
    """
    os.makedirs(os.path.dirname(out_stl_path) or ".", exist_ok=True)

    paths = [images] if isinstance(images, str) else list(images)
    if not paths:
        raise ValueError("No images given.")

    if len(paths) == 1:
        keypoints = detect_pose_landmarks(paths[0])
        depth_rgb = None
        if use_depth:
            depth_rgb = load_image_rgb(paths[0], max_dimension=None).rgb
        try:
            face_keypoints = detect_face_landmarks(paths[0])
        except Exception as exc:  # noqa: BLE001 - a failed bonus feature shouldn't fail the run
            print(f"Note: face-detail detection failed ({exc}); using the generic head shape.")
            face_keypoints = None
        body = build_body_mesh(
            keypoints,
            target_height_mm=target_height_mm,
            depth_rgb=depth_rgb,
            face_keypoints=face_keypoints,
        )
    else:
        if angles_deg is None:
            angles_deg = _default_angles(len(paths))
        if len(angles_deg) != len(paths):
            raise ValueError("angles_deg must have one entry per image.")

        views = []
        for path, angle in zip(paths, angles_deg):
            # Full resolution here — unlike pose detection, boundary
            # precision in the silhouette directly limits carving accuracy.
            prepared = load_image_rgb(path, max_dimension=None)
            mask = extract_silhouette(prepared.rgb)
            views.append(SilhouetteView(mask=mask, angle_deg=angle))

        body = carve_visual_hull(views, target_height_mm=target_height_mm)

    obj_path = os.path.splitext(out_stl_path)[0] + ".obj"
    export_obj(body, obj_path)

    mesh = to_trimesh(body)
    export_stl(mesh, out_stl_path)

    print(f"Wrote {out_stl_path} (printing) and {obj_path} (render/animation).")
