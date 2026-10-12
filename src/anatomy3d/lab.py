"""Combination Lab: every build option from the project's history -- current,
superseded and rejected alike -- as switches, enumerated exhaustively so a
photo can be run through all of them automatically and each result scored
the same way. Nothing here calls an external service; every model is a
free/open-source download used once, or this project's own code.

Two builder families, each the full Cartesian product of its own switches:
  capsule  procedural_body.build_body_mesh   (needs a usable human pose)
  outline  silhouette_relief.build_silhouette_relief_mesh
A combination's code is the family letter plus one digit per switch (the
chosen option's index), e.g. "O-0000000000"; option 0 is always the
current production setting.
"""
import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

FAMILIES = {
    "C": {
        "name": "Capsule body",
        "dims": [
            ("silhouette", "Body widths", [("on", "measured from photo outline"), ("off", "generic proportions")]),
            ("ring_fix", "Segment joins", [("on", "continuous (12 Oct fix)"), ("off", "piece by piece (old: rings)")]),
            ("limbs", "Limb depth", [("width", "width-aware"), ("flat", "flat 0.92 (old)")]),
            ("face", "Face detail", [("on", "nose/chin/eye shapes"), ("off", "none")]),
            ("hair", "Hair", [("on", "hair bumps"), ("off", "bare head")]),
            ("hands", "Hands", [("on", "palm + thumb"), ("off", "none")]),
            ("depth", "Photo depth", [("off", "off"), ("on", "on (rejected: beak face)")]),
            ("smooth", "Surface smoothing", [("on", "on"), ("off", "off (old)")]),
        ],
    },
    "O": {
        "name": "Outline (silhouette relief)",
        "dims": [
            ("matting", "Cutout", [("std", "standard"), ("birefnet", "+ fine-edge matting (BiRefNet 512)")]),
            ("profile", "Cross-section", [("poisson", "smooth 2D (12 Oct)"), ("row_ellipse", "per-row curve (11 Oct: ripples)"), ("tent", "straight ramp (9 Oct)")]),
            ("edge", "Outline edge", [("round", "rounded"), ("hard", "hard cut (old: seam)")]),
            ("depth_scale", "Thickness", [("0.75", "0.75"), ("0.5", "0.5"), ("1.0", "1.0"), ("1.5", "1.5")]),
            ("limbs", "Limb roundness", [("width", "width-aware 0.9"), ("flat", "same as torso")]),
            ("depth", "Photo depth", [("0.5", "on, 0.5"), ("off", "off"), ("1.0", "on, 1.0")]),
            ("face", "Face shaping", [("half", "half (0.45)"), ("off", "off"), ("full", "full (1.0)")]),
            ("rescue", "Layer rescue", [("auto", "animals only"), ("off", "off"), ("on", "forced (rejected on people)")]),
            ("legs", "Below the photo", [("auto", "complete legs if cut off"), ("bust", "end flat (bust)")]),
            ("smooth", "Surface smoothing", [("on", "on"), ("off", "off (old)")]),
        ],
    },
}

FINISHES = [("clay", "Sand clay"), ("ivory", "Ivory resin"), ("grey", "Studio grey"), ("photo", "Photo colour")]
SHADING = [("smooth", "Smooth"), ("flat", "Flat facets")]

# Historical versions from the project, expressed as overrides of option 0.
TEMPLATES = [
    ("8 Oct · first mannequin", "C", {"silhouette": 1, "ring_fix": 1, "limbs": 1, "face": 1, "hair": 1, "hands": 1, "smooth": 1}),
    ("8 Oct · measured body, face, hair", "C", {"ring_fix": 1, "limbs": 1, "smooth": 1}),
    ("10 Oct · capsule + depth", "C", {"ring_fix": 1, "limbs": 1, "depth": 1, "smooth": 1}),
    ("9 Oct · first outline build", "O", {"profile": 2, "edge": 1, "depth_scale": 1, "limbs": 1, "depth": 1, "face": 1, "smooth": 1}),
    ("10 Oct · outline + depth", "O", {"profile": 2, "edge": 1, "depth_scale": 1, "limbs": 1, "face": 1, "smooth": 1}),
    ("10 Oct · width-aware outline", "O", {"profile": 2, "edge": 1, "depth_scale": 1, "depth": 1, "face": 1, "smooth": 1}),
    ("11 Oct · curved, 0.75", "O", {"profile": 1, "edge": 1, "depth": 1, "face": 1, "smooth": 1}),
    ("12 Oct · smooth, no depth", "O", {"depth": 1, "face": 1}),
    ("12 Oct · smooth + face", "O", {"depth": 1, "face": 2}),
    ("Everything on, strongest", "O", {"matting": 1, "depth_scale": 2, "depth": 2, "face": 2, "rescue": 2}),
]


def family_size(fam: str) -> int:
    return int(np.prod([len(d[2]) for d in FAMILIES[fam]["dims"]]))


def total_builds() -> int:
    return sum(family_size(f) for f in FAMILIES)


def encode(fam: str, idx: "tuple[int, ...]") -> str:
    return fam + "-" + "".join(str(i) for i in idx)


def decode(code: str) -> "tuple[str, tuple[int, ...]]":
    fam, digits = code.split("-", 1)
    dims = FAMILIES[fam]["dims"]
    idx = tuple(int(c) for c in digits)
    if len(idx) != len(dims) or any(not 0 <= i < len(d[2]) for i, d in zip(idx, dims)):
        raise ValueError(f"Not a valid combination code: {code}")
    return fam, idx


def number(code: str) -> int:
    """1-based position in the full canonical list (capsule first)."""
    fam, idx = decode(code)
    offset = 0
    for f in FAMILIES:
        if f == fam:
            break
        offset += family_size(f)
    radices = [len(d[2]) for d in FAMILIES[fam]["dims"]]
    n = 0
    for i, r in zip(idx, radices):
        n = n * r + i
    return offset + n + 1


def options(code: str) -> dict:
    fam, idx = decode(code)
    return {d[0]: d[2][i][0] for d, i in zip(FAMILIES[fam]["dims"], idx)}


def describe(code: str) -> "list[tuple[str, str]]":
    fam, idx = decode(code)
    return [(d[1], d[2][i][1]) for d, i in zip(FAMILIES[fam]["dims"], idx)]


def template_name(code: str) -> Optional[str]:
    for name, fam, over in TEMPLATES:
        if _template_code(fam, over) == code:
            return name
    return None


def _template_code(fam: str, overrides: dict, base: "Optional[tuple[int, ...]]" = None) -> str:
    dims = FAMILIES[fam]["dims"]
    idx = list(base) if base else [0] * len(dims)
    for k, v in overrides.items():
        idx[[d[0] for d in dims].index(k)] = v
    return encode(fam, tuple(idx))


def run_order(families: "list[str]", bust: bool, seed: int = 0) -> "list[str]":
    """Every combination of the given families, most informative first:
    the current build, the historical versions, every single-switch change
    from the current build, every two-switch change, then the rest in a
    fixed shuffled order. `bust` sets the outline baseline to end flat."""
    order, seen = [], set()

    def add(code):
        if code not in seen:
            seen.add(code)
            order.append(code)

    bases = {}
    for fam in families:
        base = [0] * len(FAMILIES[fam]["dims"])
        if fam == "O" and bust:
            base[[d[0] for d in FAMILIES["O"]["dims"]].index("legs")] = 1
        bases[fam] = tuple(base)
        add(encode(fam, bases[fam]))
    for name, fam, over in TEMPLATES:
        if fam in families:
            add(_template_code(fam, over))
    for k in (1, 2):
        for fam in families:
            dims = FAMILIES[fam]["dims"]
            for which in itertools.combinations(range(len(dims)), k):
                for alts in itertools.product(*[[o for o in range(len(dims[w][2])) if o != bases[fam][w]] for w in which]):
                    idx = list(bases[fam])
                    for w, o in zip(which, alts):
                        idx[w] = o
                    add(encode(fam, tuple(idx)))
    rest = []
    for fam in families:
        for idx in itertools.product(*[range(len(d[2])) for d in FAMILIES[fam]["dims"]]):
            code = encode(fam, idx)
            if code not in seen:
                rest.append(code)
    random.Random(seed).shuffle(rest)
    order.extend(rest)
    return order


# ---------------------------------------------------------------- inputs


@dataclass
class LabInputs:
    rgb: np.ndarray
    alpha_std: np.ndarray
    alpha_birefnet: Optional[np.ndarray]
    animal_frac: float
    depth: np.ndarray
    face_mesh: Optional[np.ndarray]
    photo_mask: np.ndarray
    capsule: Optional[dict]
    capsule_unavailable: Optional[str]
    bust_hint: bool
    pose: Optional[dict] = None  # joint -> (x_px, y_px) on the full-size photo


def prepare_inputs(photo_path: str) -> LabInputs:
    """Everything that depends only on the photo, computed once per run."""
    from .depth_source import estimate_relative_depth
    from .face_features import find_face_mesh
    from .person_segmenter import detect_person_alpha_with_source
    from .preprocess import load_image_rgb

    rgb = load_image_rgb(photo_path, max_dimension=None).rgb
    detected = detect_person_alpha_with_source(rgb, refine=False)
    if detected is None:
        raise RuntimeError("No person or animal found in this photo.")
    alpha_std, animal_frac = detected
    try:
        refined = detect_person_alpha_with_source(rgb, refine=True)
        alpha_bir = refined[0] if refined else None
    except Exception as exc:  # noqa: BLE001 - one optional switch shouldn't sink the run
        print(f"Note: fine-edge matting unavailable ({exc}).")
        alpha_bir = None
    depth = estimate_relative_depth(rgb)
    try:
        face_mesh = find_face_mesh(rgb)
    except Exception:  # noqa: BLE001
        face_mesh = None
    mask = alpha_std > 0.5
    lab, n = ndimage.label(mask)
    if n > 1:
        mask = lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))))

    capsule, unavailable, bust = None, None, False
    try:
        capsule = _capsule_inputs(photo_path, rgb)
        from .procedural_body import CapsuleOptions, HipsNotDetectedError, build_body_mesh

        try:
            build_body_mesh(capsule["keypoints"], grid_resolution=60, options=CapsuleOptions(hands=False, smooth=False))
        except HipsNotDetectedError as exc:
            capsule, unavailable, bust = None, str(exc), True
    except Exception as exc:  # noqa: BLE001 - PoseUnusableError and friends
        capsule, unavailable = None, str(exc)
    try:
        from .landmarks import detect_pose_points

        pose = detect_pose_points(rgb)
    except Exception:  # noqa: BLE001
        pose = {}
    return LabInputs(rgb, alpha_std, alpha_bir, float(animal_frac), depth, face_mesh, mask, capsule, unavailable, bust, pose)


def _capsule_inputs(photo_path: str, rgb_full: np.ndarray) -> dict:
    from .face_features import detect_face_landmarks
    from .hair_features import detect_hair_mask
    from .landmarks import detect_pose_landmarks
    from .pipeline import _body_scale_px, _bone_list
    from .preprocess import load_image_rgb
    from .silhouette import extract_silhouette

    keypoints = detect_pose_landmarks(photo_path)
    try:
        face_keypoints = detect_face_landmarks(photo_path)
    except Exception:  # noqa: BLE001
        face_keypoints = None
    rgb_pose = load_image_rgb(photo_path).rgb
    try:
        sil = extract_silhouette(rgb_pose, bones=_bone_list(keypoints), body_scale_px=_body_scale_px(keypoints))
    except Exception:  # noqa: BLE001
        sil = None
    try:
        hair = detect_hair_mask(photo_path, keypoints=keypoints)
        if hair is not None and hair.shape != (keypoints.image_height, keypoints.image_width):
            hair = np.asarray(
                Image.fromarray((hair * 255).astype(np.uint8)).resize((keypoints.image_width, keypoints.image_height), Image.NEAREST)
            ) > 127
    except Exception:  # noqa: BLE001
        hair = None
    return {"keypoints": keypoints, "face": face_keypoints, "silhouette": sil, "hair": hair, "rgb_pose": rgb_pose, "rgb_full": rgb_full}


# ---------------------------------------------------------------- build


HEIGHT_MM = 150.0


def build(code: str, inputs: LabInputs):
    """The finished model (body + base disk) for one combination."""
    from .mesh_export import add_base_disk

    fam, _ = decode(code)
    o = options(code)
    if fam == "C":
        if inputs.capsule is None:
            raise RuntimeError(f"Capsule builder can't use this photo: {inputs.capsule_unavailable}")
        from .procedural_body import CapsuleOptions, build_body_mesh

        c = inputs.capsule
        body = build_body_mesh(
            c["keypoints"],
            target_height_mm=HEIGHT_MM,
            silhouette_mask=c["silhouette"] if o["silhouette"] == "on" else None,
            depth_rgb=c["rgb_full"] if o["depth"] == "on" else None,
            face_keypoints=c["face"] if o["face"] == "on" else None,
            hair_mask=c["hair"] if o["hair"] == "on" else None,
            texture_rgb=c["rgb_pose"],
            options=CapsuleOptions(
                ring_fix=o["ring_fix"] == "on",
                width_aware_limbs=o["limbs"] == "width",
                hands=o["hands"] == "on",
                smooth=o["smooth"] == "on",
            ),
        )
    else:
        from .silhouette_relief import build_silhouette_relief_mesh

        alpha = inputs.alpha_birefnet if (o["matting"] == "birefnet" and inputs.alpha_birefnet is not None) else inputs.alpha_std
        ds = float(o["depth_scale"])
        body = build_silhouette_relief_mesh(
            alpha,
            target_height_mm=HEIGHT_MM,
            depth_scale=ds,
            limb_depth_scale=0.9 if o["limbs"] == "width" else ds,
            depth_map=None if o["depth"] == "off" else inputs.depth,
            depth_strength=0.5 if o["depth"] == "off" else float(o["depth"]),
            animal_frac=inputs.animal_frac,
            bust_mode=o["legs"] == "bust",
            face_mesh=None if o["face"] == "off" else inputs.face_mesh,
            face_strength={"off": None, "half": 0.45, "full": 1.0}[o["face"]],
            profile=o["profile"],
            edge=o["edge"],
            layer_rescue=o["rescue"],
            smooth=o["smooth"] == "on",
            pose=inputs.pose,
        )
    return add_base_disk(body, HEIGHT_MM)


# ---------------------------------------------------------------- scoring

PREVIEW_FACES = 40_000
_RASTER = 420


def _front_heightmap(mesh: trimesh.Trimesh, floor: float):
    pts, _ = trimesh.sample.sample_surface(mesh, 1_200_000, seed=0)
    pts = pts[pts[:, 1] > floor]
    x0, x1, y0, y1 = pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()
    px = (y1 - y0) / _RASTER
    w = int(np.ceil((x1 - x0) / px)) + 1
    cols = ((pts[:, 0] - x0) / px).astype(int)
    rows = ((y1 - pts[:, 1]) / px).astype(int)
    z = np.full((_RASTER + 1, w), -np.inf)
    np.maximum.at(z, (rows, cols), pts[:, 2])
    sil = ndimage.binary_fill_holes(ndimage.binary_closing(np.isfinite(z), iterations=2))
    # Pixels inside the outline that no sample happened to land on take
    # their highest neighbour's height, so they don't shade as pits.
    for _ in range(3):
        missing = sil & ~np.isfinite(z)
        if not missing.any():
            break
        z = np.where(missing, ndimage.maximum_filter(z, size=3), z)
    # A pixel where no FRONT sample landed falls through to the back
    # surface and reads as a pit; a 3x3 grey closing removes those without
    # touching real features (all several pixels wide at this resolution).
    finite = np.isfinite(z)
    floor_val = z[finite].min() if finite.any() else 0.0
    closed = ndimage.grey_closing(np.where(finite, z, floor_val), size=(3, 3))
    zf = np.where(sil & finite, np.where(ndimage.binary_erosion(sil, iterations=2), closed, z), np.nan)
    return sil, zf, px


def _shade(sil, zf, px, size=None) -> Image.Image:
    """Clay-shaded image of a front height map (same look as the viewer)."""
    z = np.nan_to_num(zf, nan=np.nanmin(zf))
    z = ndimage.gaussian_filter(z, 0.8)
    gy, gx = np.gradient(z, px)
    n = np.dstack([-gx, gy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.45, 0.55, 0.7])
    light /= np.linalg.norm(light)
    lam = np.clip(n @ light, 0, 1)
    shade = 0.35 + 0.75 * lam
    base = np.array([240, 214, 189]) / 255.0
    img = np.where(sil[..., None], np.clip(base * shade[..., None], 0, 1), np.array([20, 24, 32]) / 255.0)
    im = Image.fromarray((img * 255).astype(np.uint8))
    if size:
        im.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGB", (size, size), (20, 24, 32))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        im = canvas
    return im


def _three_views(mesh: trimesh.Trimesh, floor: float, sil, zf, px, size: "tuple[int, int]" = (480, 320)) -> Image.Image:
    """Front, right side and rear next to each other at the SAME scale, so
    a model that's too thin front-to-back shows it at a glance."""
    v = mesh.vertices
    shots = [_shade(sil, zf, px)]
    for turned in (
        np.column_stack([v[:, 2], v[:, 1], -v[:, 0]]),  # right side to camera
        np.column_stack([-v[:, 0], v[:, 1], -v[:, 2]]),  # back to camera
    ):
        other = mesh.copy()
        other.vertices = turned
        o_sil, o_zf, _ = _front_heightmap(other, floor)
        shots.append(_shade(o_sil, o_zf, px))
    gap = max(8, shots[0].height // 30)
    strip = Image.new("RGB", (sum(im.width for im in shots) + gap * 2, max(im.height for im in shots)), (20, 24, 32))
    x = 0
    for im in shots:
        strip.paste(im, (x, 0))
        x += im.width + gap
    strip.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (20, 24, 32))
    canvas.paste(strip, ((size[0] - strip.width) // 2, (size[1] - strip.height) // 2))
    return canvas


def _procrustes_nme(src: np.ndarray, dst: np.ndarray) -> float:
    a, b = src - src.mean(0), dst - dst.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    aligned = (s.sum() / (a**2).sum()) * a @ (u @ vt)
    return 100.0 * float(np.linalg.norm(aligned - b, axis=1).mean() / np.linalg.norm(dst[33] - dst[263]))


def score(code: str, body, inputs: LabInputs) -> "tuple[dict, trimesh.Trimesh, Image.Image]":
    """Metrics, a light preview mesh (keeps vertex colours when small enough)
    and a thumbnail for one built combination."""
    from .face_features import detect_face_mesh
    from .mesh_export import _BASE_DISK_MAX_THICKNESS_MM, _BASE_DISK_MIN_THICKNESS_MM, _BASE_DISK_THICKNESS_FRAC
    from .print_prep import estimate_min_wall_thickness

    mesh = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
    if body.colors is not None:
        mesh.visual.vertex_colors = np.column_stack([body.colors, np.full(len(body.colors), 255, np.uint8)])
    mesh.merge_vertices()
    disk = float(np.clip(_BASE_DISK_THICKNESS_FRAC * HEIGHT_MM, _BASE_DISK_MIN_THICKNESS_MM, _BASE_DISK_MAX_THICKNESS_MM))
    floor = mesh.vertices[:, 1].min() + disk
    m = {"watertight": bool(mesh.is_watertight), "faces": int(len(mesh.faces))}

    preview = mesh if len(mesh.faces) <= PREVIEW_FACES else mesh.simplify_quadric_decimation(face_count=PREVIEW_FACES)
    m["min_wall_mm"] = round(float(estimate_min_wall_thickness(preview, sample_count=800)), 2)

    keep = preview.vertices[preview.faces][:, :, 1].min(axis=1) > floor + 0.5
    adj, ang = preview.face_adjacency, preview.face_adjacency_angles
    both = keep[adj[:, 0]] & keep[adj[:, 1]]
    m["spike_pct"] = round(100.0 * float((np.degrees(ang[both]) > 60).mean()), 2) if both.any() else 0.0

    sil, zf, px = _front_heightmap(preview, floor)
    m["shape_iou"] = round(_aligned_iou(sil, inputs.photo_mask), 1)
    thumb = _three_views(preview, floor, sil, zf, px)

    m["face_found"] = None
    got, head = None, None
    if inputs.face_mesh is not None:
        ys = np.nonzero(inputs.photo_mask)[0]
        my0, my1 = ys.min(), ys.max() + 1
        f_top = (inputs.face_mesh[:468, 1].min() - my0) / (my1 - my0)
        f_bot = (inputs.face_mesh[:468, 1].max() - my0) / (my1 - my0)
        pad = (f_bot - f_top) * 0.45
        r0, r1 = int(max(0, (f_top - pad) * _RASTER)), int(min(_RASTER, (f_bot + pad) * _RASTER))
        rows = np.nonzero(sil[r0:r1].any(axis=1))[0]
        if len(rows):
            band_sil, band_z = sil[r0:r1], zf[r0:r1]
            head = np.asarray(_shade(band_sil, band_z, px).resize((band_sil.shape[1] * 3, band_sil.shape[0] * 3), Image.LANCZOS))
            got = detect_face_mesh(head)
            m["face_found"] = got is not None
            if got is not None:
                m["face_err_pct"] = round(_procrustes_nme(got[:468, :2], inputs.face_mesh[:468, :2]), 1)

    s = m["shape_iou"] - 4.0 * m["spike_pct"] - (25.0 if not m["watertight"] else 0.0) - (20.0 if m["min_wall_mm"] < 0.8 else 0.0)
    if m.get("face_err_pct") is not None:
        s += max(0.0, 30.0 - m["face_err_pct"])
    m["score"] = round(s, 1)
    try:
        params = _parameters(m, inputs, preview, floor, sil, got, head)
        m["params"] = {k: v for k, v in params.items()}
        m["held_mean"] = round(float(np.mean([v[2] for v in params.values()])), 1)
    except Exception as exc:  # noqa: BLE001 - a measuring problem shouldn't lose the build
        m["params_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return m, preview, thumb


def _aligned_iou(sil: np.ndarray, mask: np.ndarray) -> float:
    canvas = _aligned_canvas(sil, mask)
    return 100.0 * float((canvas & mask).sum()) / float((canvas | mask).sum())


def _aligned_canvas(sil: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The model's front outline scaled to the photo subject's height and
    centred on it, in the photo's own pixel frame."""
    ys, xs = np.nonzero(mask)
    my0, my1, mx0, mx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sy, sx = np.nonzero(sil)
    sil = sil[sy.min():sy.max() + 1, sx.min():sx.max() + 1]
    th = my1 - my0
    tw = max(1, int(round(sil.shape[1] * th / sil.shape[0])))
    sil_r = np.asarray(Image.fromarray(sil.astype(np.uint8) * 255).resize((tw, th), Image.NEAREST)) > 127
    canvas = np.zeros_like(mask)
    left = (mx0 + mx1) // 2 - tw // 2
    a0, a1 = max(0, left), min(mask.shape[1], left + tw)
    canvas[my0:my1, a0:a1] = sil_r[:, a0 - left:a1 - left]
    return canvas

# ---------------------------------------------------------------- parameters
# Each combination is checked against these. "detected" ones are measured on
# the photo itself and on the model's front view (2D from the 3D model, same
# frame); "common" ones are what any printable human figure should hold.
# held = how well the model keeps it, 0-100 (rule per parameter below).
PARAMS = [
    ("outline", "Outline", "detected", "front view of the model over the photo's cutout", "overlap %"),
    ("aspect", "Height : width", "detected", "whole figure, model vs photo", "100 − % off"),
    ("w_head", "Head width", "detected", "outline width at the eyes, model vs photo", "100 − % off"),
    ("w_shoulder", "Shoulder width", "detected", "outline width at the shoulder joints", "100 − % off"),
    ("w_chest", "Chest width", "detected", "outline width a quarter of the way from shoulders to hips", "100 − % off"),
    ("w_waist", "Waist width", "detected", "outline width 70% of the way from shoulders to hips", "100 − % off"),
    ("face_shape", "Face shape (468 points)", "detected", "face points found on the model vs the photo's, error % of eye distance", "100 − 5 × error"),
    ("eye_span", "Eye spacing", "detected", "outer eye corners ÷ face width", "100 − % off"),
    ("nose_len", "Nose length", "detected", "bridge to nose base ÷ face height", "100 − % off"),
    ("mouth_w", "Mouth width", "detected", "mouth corners ÷ face width", "100 − % off"),
    ("jaw_w", "Jaw width", "detected", "jaw angles ÷ face width", "100 − % off"),
    ("face_hw", "Face height : width", "detected", "forehead–chin ÷ cheek–cheek", "100 − % off"),
    ("face_id", "Face embedding match", "detected", "SFace (Apache-2.0) cosine, model's face render vs the photo's face", "100 × cosine"),
    ("d_head", "Head depth", "common", "front-to-back ÷ side-to-side at the eyes; ANSUR II 1.29", "100 − % off"),
    ("d_chest", "Chest depth", "common", "front-to-back through the chest's centre line ÷ shoulder-joint distance; ANSUR II 0.61", "100 − % off"),
    ("d_waist", "Waist depth", "common", "front-to-back through the waist's centre line ÷ shoulder-joint distance; ANSUR II 0.57", "100 − % off"),
    ("sealed", "Sealed", "common", "mesh is watertight", "100 or 0"),
    ("wall", "Thinnest wall", "common", "2nd-percentile wall thickness; printable at 0.8 mm", "100 × wall ÷ 0.8, max 100"),
    ("smooth_ok", "No spikes", "common", "share of folds sharper than 60°", "100 − 20 × %"),
]
# Head: head length / head breadth. Chest and waist: depth through the
# centre line / biacromial (shoulder-joint) breadth -- measured against
# the shoulders because arms hanging against the body join the torso's
# cross-section and would inflate a width-based ratio's denominator.
_ANSUR_DEPTH = {"d_head": 1.2914, "d_chest": 0.611, "d_waist": 0.572}
_FACE_IDX = {"eye": (33, 263), "cheek": (234, 454), "brow_chin": (10, 152), "nose": (168, 2), "mouth": (61, 291), "jaw": (172, 397)}


def _held_ratio(model: float, target: float) -> int:
    return int(round(max(0.0, 100.0 * (1.0 - abs(model / target - 1.0))))) if target else 0


def _face_props(lm: np.ndarray) -> dict:
    d = lambda a, b: float(np.linalg.norm(lm[a, :2] - lm[b, :2]))  # noqa: E731
    fw, fh = d(*_FACE_IDX["cheek"]), d(*_FACE_IDX["brow_chin"])
    return {"eye_span": d(*_FACE_IDX["eye"]) / fw, "nose_len": d(*_FACE_IDX["nose"]) / fh, "mouth_w": d(*_FACE_IDX["mouth"]) / fw,
            "jaw_w": d(*_FACE_IDX["jaw"]) / fw, "face_hw": fh / fw}


def _levels(inputs: "LabInputs") -> dict:
    """Photo rows where the widths and depths are measured."""
    ys = np.nonzero(inputs.photo_mask)[0]
    my0, my1 = ys.min(), ys.max() + 1
    pose = inputs.pose or {}
    lv = {}
    if inputs.face_mesh is not None:
        lv["head"] = (float((inputs.face_mesh[33, 1] + inputs.face_mesh[263, 1]) / 2), float(inputs.face_mesh[:468, 0].mean()))
    sh = [pose[k] for k in ("left_shoulder", "right_shoulder") if k in pose]
    hp = [pose[k] for k in ("left_hip", "right_hip") if k in pose]
    if sh:
        sy, sx = float(np.mean([p[1] for p in sh])), float(np.mean([p[0] for p in sh]))
        lv["shoulder"] = (sy, sx)
        if hp:
            hy = float(np.mean([p[1] for p in hp]))
            lv["chest"] = (sy + 0.25 * (hy - sy), sx)
            lv["waist"] = (sy + 0.70 * (hy - sy), sx)
        else:
            lv["chest"] = (sy + 0.25 * (my1 - sy), sx)
    return {k: v for k, v in lv.items() if my0 < v[0] < my1 - 1}


def _run_width(row: np.ndarray, col: float) -> int:
    on = np.flatnonzero(row)
    if not len(on):
        return 0
    c = int(round(col))
    if not row[min(max(c, 0), len(row) - 1)]:
        c = int(on[np.argmin(np.abs(on - col))])
    l = c
    while l > 0 and row[l - 1]:
        l -= 1
    r = c
    while r < len(row) - 1 and row[r + 1]:
        r += 1
    return r - l + 1


def _centre_depth(mesh: trimesh.Trimesh, y: float, x_mid: float, half_band: float) -> Optional[float]:
    """Front-to-back extent of the cross-section at height y, within
    half_band of the body's centre line."""
    sec = mesh.section(plane_origin=[0.0, y, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if sec is None:
        return None
    pts = np.vstack(sec.discrete)
    near = np.abs(pts[:, 0] - x_mid) <= half_band
    return float(np.ptp(pts[near, 2])) if near.sum() >= 2 else None


def _depth_ratio(mesh: trimesh.Trimesh, y: float, x_mid: float) -> Optional[float]:
    sec = mesh.section(plane_origin=[0.0, y, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if sec is None:
        return None
    best = None
    for loop in sec.discrete:
        x0, x1 = loop[:, 0].min(), loop[:, 0].max()
        if x0 <= x_mid <= x1:
            area = (x1 - x0) * np.ptp(loop[:, 2])
            if best is None or area > best[0]:
                best = (area, x1 - x0, float(np.ptp(loop[:, 2])))
    return best[2] / best[1] if best and best[1] > 1e-6 else None


_sface = {}


def _face_embedding(rgb: np.ndarray):
    """SFace (OpenCV Zoo, Apache-2.0) with YuNet (MIT) to find the face.
    Downloaded once from the OpenCV Zoo mirror; None if unavailable."""
    import cv2

    if "rec" not in _sface:
        try:
            cache = Path.home() / ".cache" / "anatomy3d-print"
            cache.mkdir(parents=True, exist_ok=True)
            paths = {}
            for key, rel in (("rec", "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
                             ("det", "face_detection_yunet/face_detection_yunet_2023mar.onnx")):
                dst = cache / rel.split("/")[-1]
                if not dst.exists():
                    import os
                    import urllib.request

                    # per-process temp name: several Lab workers may fetch at once
                    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.part")
                    urllib.request.urlretrieve(f"https://huggingface.co/opencv/opencv_zoo/resolve/main/models/{rel}", tmp)
                    os.replace(tmp, dst)
                paths[key] = str(dst)
            _sface["rec"] = cv2.FaceRecognizerSF.create(paths["rec"], "")
            _sface["det"] = cv2.FaceDetectorYN.create(paths["det"], "", (320, 320), 0.6)
        except Exception as exc:  # noqa: BLE001 - an optional check
            print(f"Note: face embedding check unavailable ({exc}).")
            _sface["rec"] = None
    if _sface["rec"] is None:
        return None
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    _sface["det"].setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = _sface["det"].detect(bgr)
    if faces is None or not len(faces):
        return None
    f = faces[np.argmax(faces[:, 2] * faces[:, 3])]
    return _sface["rec"].feature(_sface["rec"].alignCrop(bgr, f))


def _photo_face_crop(rgb: np.ndarray, lm: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1 = lm[:468, 0].min(), lm[:468, 0].max(), lm[:468, 1].min(), lm[:468, 1].max()
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, 1.9 * max(x1 - x0, y1 - y0)
    h, w = rgb.shape[:2]
    crop = rgb[int(max(0, cy - side / 2)):int(min(h, cy + side / 2)), int(max(0, cx - side / 2)):int(min(w, cx + side / 2))]
    im = Image.fromarray(crop)
    zoom = 560.0 / max(im.size)
    return np.asarray(im.resize((max(1, int(im.width * zoom)), max(1, int(im.height * zoom))), Image.LANCZOS))


def _parameters(m: dict, inputs: "LabInputs", preview: trimesh.Trimesh, floor: float, sil: np.ndarray, render_face, head_img) -> dict:
    """{key: [model value, photo/target value, held 0-100]}; a key is left
    out when it can't apply to this photo (no face, no hips, ...)."""
    out = {}
    mask = inputs.photo_mask
    ys, xs = np.nonzero(mask)
    my0, my1, mx0, mx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    canvas = _aligned_canvas(sil, mask)
    out["outline"] = [m["shape_iou"], 100.0, int(round(m["shape_iou"]))]
    cy, cx = np.nonzero(canvas)
    if len(cx):
        ma, pa = (cy.max() - cy.min() + 1) / (cx.max() - cx.min() + 1), (my1 - my0) / (mx1 - mx0)
        out["aspect"] = [round(ma, 3), round(pa, 3), _held_ratio(ma, pa)]
    lv = _levels(inputs)
    for part in ("head", "shoulder", "chest", "waist"):
        if part in lv:
            row, col = lv[part]
            r = int(round(row))
            pw, mw = _run_width(mask[r], col), _run_width(canvas[r], col)
            if pw:
                out[f"w_{part}"] = [mw, pw, _held_ratio(mw, pw) if mw else 0]
    if inputs.face_mesh is not None:
        if render_face is None:
            for k in ("face_shape", "eye_span", "nose_len", "mouth_w", "jaw_w", "face_hw"):
                out[k] = [None, None, 0]
        else:
            nme = m.get("face_err_pct")
            out["face_shape"] = [nme, 0.0, int(round(max(0.0, 100.0 - 5.0 * nme)))]
            mp_, pp = _face_props(render_face), _face_props(inputs.face_mesh)
            for k in mp_:
                out[k] = [round(mp_[k], 3), round(pp[k], 3), _held_ratio(mp_[k], pp[k])]
        key = id(inputs)
        if key not in _sface:
            _sface[key] = _face_embedding(_photo_face_crop(inputs.rgb, inputs.face_mesh))
        pe = _sface.get(key)
        re_ = _face_embedding(head_img) if head_img is not None and pe is not None else None
        if pe is not None:
            import cv2

            cos = float(_sface["rec"].match(pe, re_, cv2.FaceRecognizerSF_FR_COSINE)) if re_ is not None else None
            out["face_id"] = [None if cos is None else round(cos, 3), 1.0, int(round(100 * max(0.0, cos))) if cos is not None else 0]
    if inputs.animal_frac <= 0.15:
        top = float(preview.vertices[:, 1].max())
        above = preview.vertices[preview.vertices[:, 1] > floor]
        mx_c, px_c = (above[:, 0].min() + above[:, 0].max()) / 2, (mx0 + mx1) / 2
        scale = (top - floor) / (my1 - my0)
        pose = inputs.pose or {}
        sw_mm = None
        if "left_shoulder" in pose and "right_shoulder" in pose:
            sw_mm = float(np.hypot(*(np.array(pose["left_shoulder"]) - np.array(pose["right_shoulder"])))) * scale
        for part, key in (("head", "d_head"), ("chest", "d_chest"), ("waist", "d_waist")):
            if part in lv:
                row, col = lv[part]
                y, x = top - (row - my0) * scale, mx_c + (col - px_c) * scale
                if part == "head":
                    ratio = _depth_ratio(preview, y, x)
                elif sw_mm:
                    ratio = _centre_depth(preview, y, x, 0.04 * sw_mm)
                    ratio = None if ratio is None else ratio / sw_mm
                else:
                    ratio = None
                if ratio is not None:
                    t = _ANSUR_DEPTH[key]
                    out[key] = [round(ratio, 2), t, _held_ratio(ratio, t)]
    out["sealed"] = [m["watertight"], True, 100 if m["watertight"] else 0]
    out["wall"] = [m["min_wall_mm"], 0.8, int(round(min(100.0, 100.0 * max(0.0, m["min_wall_mm"]) / 0.8)))]
    out["smooth_ok"] = [m["spike_pct"], 0.0, int(round(max(0.0, 100.0 - 20.0 * m["spike_pct"])))]
    return out


SCORE_FORMULA = (
    "score = shape match % − 4 × spikes % − 25 if not sealed − 20 if thinnest wall < 0.8 mm "
    "+ (30 − face error %) when a face is found"
)
