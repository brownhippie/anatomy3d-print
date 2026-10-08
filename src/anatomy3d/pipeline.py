import os
from typing import List, Optional, Union

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
) -> None:
    """One photo -> the capsule/SDF body builder (a geometric guess).
    Two or more photos (different angles around the subject) -> visual hull
    carving from their silhouettes, which uses the subject's *actual* shape
    instead of a guess. `angles_deg` defaults to evenly spaced rotation
    angles in the order the photos are given (front first), matching a
    turntable-style capture; pass explicit angles if you shot something
    else.
    """
    os.makedirs(os.path.dirname(out_stl_path) or ".", exist_ok=True)

    paths = [images] if isinstance(images, str) else list(images)
    if not paths:
        raise ValueError("No images given.")

    if len(paths) == 1:
        keypoints = detect_pose_landmarks(paths[0])
        body = build_body_mesh(keypoints, target_height_mm=target_height_mm)
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
