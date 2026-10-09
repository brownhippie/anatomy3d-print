import os
from typing import List, Optional, Union

import numpy as np

from .face_features import detect_face_landmarks
from .hair_features import detect_hair_mask
from .landmarks import detect_pose_landmarks
from .mesh_export import export_glb, export_obj, to_trimesh
from .preprocess import load_image_rgb
from .print_prep import export_stl
from .procedural_body import build_body_mesh
from .silhouette import extract_silhouette
from .silhouette_relief import build_silhouette_relief_mesh
from .visual_hull import SilhouetteView, carve_visual_hull


# Anatomically-adjacent joint pairs — the straight-line bone segments
# silhouette.py's two-sided reclassify checks a color-ambiguous pixel's
# position against (see its `bones` param and POSITION_CORRIDOR_PX).
# Plain coordinate pairs, not landmark names, cross that module boundary,
# keeping silhouette.py itself free of any pose-detection-specific types.
_BONE_JOINT_PAIRS = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
]


def _bone_list(keypoints) -> List[tuple]:
    joints = keypoints.joints
    return [
        (np.array(joints[a][:2]), np.array(joints[b][:2]))
        for a, b in _BONE_JOINT_PAIRS
        if a in joints and b in joints
    ]


def _body_scale_px(keypoints) -> "float | None":
    """Shoulder width in pixels — the subject's own measured size in this
    photo, used to scale silhouette.py's position-based corridor/ceiling
    instead of the flat pixel constants they were calibrated with. See
    POSITION_CORRIDOR_MIN/MAX_RATIO's docstring: those constants were
    tuned on one 96.2px-shoulder-width photo and measurably broke on a
    second photo shot at 5.5x that scale. None when both shoulders aren't
    detected (silhouette.py falls back to the flat constants)."""
    joints = keypoints.joints
    if "left_shoulder" not in joints or "right_shoulder" not in joints:
        return None
    return float(np.linalg.norm(
        np.array(joints["left_shoulder"][:2]) - np.array(joints["right_shoulder"][:2])
    ))


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
    target_faces: Optional[int] = 20000,
    bake_color: bool = True,
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

    `bake_color`: single-photo mode only (multi-photo visual-hull mode
    carries no per-vertex color regardless — see carve_visual_hull). Set
    False to skip photo-color sampling entirely and get a plain,
    untextured mesh — useful when only the shape/geometry matters and
    color is a distraction (e.g. judging silhouette/fit accuracy without
    texture quality as a confound). The STL output already carries no
    color either way; this only affects the OBJ/GLB.

    Single-photo mode also tries face detail (nose, chin, eye sockets,
    face_features.py) automatically — unlike `use_depth`, this needs no
    opt-in flag, since it's the same MediaPipe dependency already
    required for pose detection, not a new one. A missing/unusable face
    or a failed model fetch falls back to the generic head shape rather
    than failing the run. It also runs face detection at the photo's full
    resolution (unlike pose detection's 1280px cap) — a face is often a
    small fraction of a full-body frame, so it benefits from every pixel
    available, see face_features.py.

    `target_faces`: the STL (not the OBJ, which stays full-detail for
    render/animation use) is simplified down to roughly this many faces
    after repair, see print_prep.py's simplify_for_output. The adaptive
    grid-resolution fixes in procedural_body.py can produce far more
    detail than an FDM print or a web preview needs to look identical;
    this trades that unneeded density back down for a smaller, faster
    file. Pass None to export at full detail.
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
        try:
            # Same default params detect_pose_landmarks uses internally
            # (load_image_rgb(path), no overrides) so this array is
            # pixel-identical to the one `keypoints` was measured against
            # — required for the silhouette mask and the joint positions
            # to share one coordinate system (see build_body_mesh).
            silhouette_rgb = load_image_rgb(paths[0]).rgb
            silhouette_mask = extract_silhouette(
                silhouette_rgb, bones=_bone_list(keypoints), body_scale_px=_body_scale_px(keypoints)
            )
        except Exception as exc:  # noqa: BLE001 - a failed bonus feature shouldn't fail the run
            print(f"Note: silhouette-based shape refinement failed ({exc}); using generic proportions.")
            silhouette_mask = None
        if silhouette_mask is not None:
            # Diagnostic only — not used to change the mesh yet. Measures
            # whether this photo's own silhouette supports a symmetry-based
            # shortcut at all (see anatomy3d.symmetry's docstring for what
            # the IoU score means and how it was validated) and surfaces
            # that finding instead of leaving it implicit.
            from .symmetry import describe_symmetry_finding, find_symmetry_axis

            symmetry_result = find_symmetry_axis(silhouette_mask)
            print(describe_symmetry_finding(symmetry_result, keypoints.image_width))
        try:
            # Detected at full resolution (same reasoning as face
            # detection — hair is a small, fine-detailed region), then
            # resized back down to match keypoints' own resolution so
            # pixel coordinates line up (see the same fix applied to face
            # landmarks in procedural_body.py's _face_to_local_xyz).
            hair_mask = detect_hair_mask(paths[0], keypoints=keypoints)
            if hair_mask is not None and hair_mask.shape != (keypoints.image_height, keypoints.image_width):
                from PIL import Image

                resized = Image.fromarray((hair_mask * 255).astype(np.uint8)).resize(
                    (keypoints.image_width, keypoints.image_height), Image.NEAREST
                )
                hair_mask = np.asarray(resized) > 127
        except Exception as exc:  # noqa: BLE001 - a failed bonus feature shouldn't fail the run
            print(f"Note: hair detection failed ({exc}); the head will stay bare.")
            hair_mask = None
        # Reuses silhouette_rgb when that step already succeeded (same
        # photo, same resolution as keypoints) instead of loading the
        # image a second time; falls back to a fresh load so texture
        # baking still works when silhouette extraction itself failed.
        # None (bake_color=False) reaches build_body_mesh unchanged —
        # it already treats a missing texture_rgb as "skip color baking"
        # (see procedural_body.py), not an error.
        texture_rgb = None
        if bake_color:
            texture_rgb = silhouette_rgb if silhouette_mask is not None else load_image_rgb(paths[0]).rgb
        body = build_body_mesh(
            keypoints,
            target_height_mm=target_height_mm,
            silhouette_mask=silhouette_mask,
            depth_rgb=depth_rgb,
            face_keypoints=face_keypoints,
            hair_mask=hair_mask,
            texture_rgb=texture_rgb,
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

    # glTF binary: a standard interchange format other tools/engines can
    # consume, carrying the same photo-baked per-vertex colors as the OBJ
    # (which the STL has no room for at all).
    glb_path = os.path.splitext(out_stl_path)[0] + ".glb"
    export_glb(body, glb_path)

    mesh = to_trimesh(body)
    export_stl(mesh, out_stl_path, target_faces=target_faces)

    print(f"Wrote {out_stl_path} (printing), {obj_path} (render/animation), and {glb_path} (glTF, for other tools).")


def run_silhouette_relief_pipeline(
    image_path: str,
    out_stl_path: str,
    target_height_mm: float = 150.0,
    target_faces: Optional[int] = 20000,
    use_depth: bool = False,
    depth_strength: float = 0.5,
) -> None:
    """A deliberately different, simpler path than run_pipeline's
    single-photo mode: the mesh's own front-on silhouette is built
    directly from the photo's 2D cutout (detect_person_alpha -- the same
    function the webapp's /jobs/{id}/cutout.png serves), not from
    capsule primitives fit to detected pose joints (see
    silhouette_relief.py for why that makes the two genuinely different
    shapes, not just different code paths to the same result).

    No pose detection, so no pose-plausibility requirement and no face/
    hair/color options -- the tradeoff for guaranteeing the mesh's
    silhouette matches the cutout exactly: no anatomical detail beyond
    what the 2D outline itself shows (fingers only as wide as the
    cutout's own hand shape, no fur/hair volume). Multi-photo visual-hull
    mode (run_pipeline) is the existing option when multiple viewing
    angles are available instead.

    `use_depth`: same optional extra and reasoning as run_pipeline's own
    use_depth (requires `pip install -r requirements-depth.txt`). Pure
    silhouette extrusion has no way to tell a region is angled toward
    the camera -- confirmed directly as a real gap, not a guess: a
    turned head came out just as symmetric as a straight-on torso. Real
    depth data biases the front surface only (see
    build_silhouette_relief_mesh's own docstring for the mechanism) --
    off by default since it's a sizeable extra dependency, same
    reasoning run_pipeline's use_depth already applies.

    Confirmed directly, honestly: real, but crude -- it reads as genuine
    surface variation where there was a flat symmetric dome before, not
    a clean recognizable face/snout shape. It's also most visible on the
    raw marching_cubes mesh; `target_faces` simplification for the final
    STL smooths a good deal of the fine bumps back out, same as it would
    for any other fine surface detail."""
    os.makedirs(os.path.dirname(out_stl_path) or ".", exist_ok=True)

    from .person_segmenter import detect_person_alpha

    rgb = load_image_rgb(image_path, max_dimension=None).rgb
    alpha = detect_person_alpha(rgb)
    if alpha is None:
        raise RuntimeError(
            "No person/animal detected in the image -- use a clear photo with "
            "the subject against a reasonably distinct background."
        )

    body = build_silhouette_relief_mesh(
        alpha,
        target_height_mm=target_height_mm,
        depth_rgb=rgb if use_depth else None,
        depth_strength=depth_strength,
    )

    obj_path = os.path.splitext(out_stl_path)[0] + ".obj"
    export_obj(body, obj_path)

    glb_path = os.path.splitext(out_stl_path)[0] + ".glb"
    export_glb(body, glb_path)

    mesh = to_trimesh(body)
    export_stl(mesh, out_stl_path, target_faces=target_faces)

    print(f"Wrote {out_stl_path} (printing), {obj_path} (render/animation), and {glb_path} (glTF, for other tools).")
