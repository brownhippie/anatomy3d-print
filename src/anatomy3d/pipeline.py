import os

from .body_fit import fit_body
from .landmarks import detect_pose_landmarks
from .mesh_export import export_obj, to_trimesh
from .print_prep import export_stl


def run_pipeline(
    image_path: str,
    smplx_model_dir: str,
    out_stl_path: str,
    gender: str = "neutral",
    device: str = "cpu",
) -> None:
    os.makedirs(os.path.dirname(out_stl_path) or ".", exist_ok=True)

    keypoints = detect_pose_landmarks(image_path)
    body = fit_body(keypoints, smplx_model_dir, gender=gender, device=device)

    obj_path = os.path.splitext(out_stl_path)[0] + ".obj"
    export_obj(body, obj_path)

    mesh = to_trimesh(body)
    export_stl(mesh, out_stl_path)

    print(f"Wrote {out_stl_path} (printing) and {obj_path} (render/animation).")
