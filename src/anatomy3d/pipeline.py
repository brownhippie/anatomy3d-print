import os

from .landmarks import detect_pose_landmarks
from .mesh_export import export_obj, to_trimesh
from .print_prep import export_stl
from .procedural_body import build_body_mesh


def run_pipeline(
    image_path: str,
    out_stl_path: str,
    target_height_mm: float = 150.0,
) -> None:
    os.makedirs(os.path.dirname(out_stl_path) or ".", exist_ok=True)

    keypoints = detect_pose_landmarks(image_path)
    body = build_body_mesh(keypoints, target_height_mm=target_height_mm)

    obj_path = os.path.splitext(out_stl_path)[0] + ".obj"
    export_obj(body, obj_path)

    mesh = to_trimesh(body)
    export_stl(mesh, out_stl_path)

    print(f"Wrote {out_stl_path} (printing) and {obj_path} (render/animation).")
