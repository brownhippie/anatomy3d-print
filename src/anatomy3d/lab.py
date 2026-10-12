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


def prepare_inputs(photo_path: str) -> LabInputs:
    """Everything that depends only on the photo, computed once per run."""
    from .depth_source import estimate_relative_depth
    from .face_features import detect_face_mesh
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
        face_mesh = detect_face_mesh(rgb)
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
    return LabInputs(rgb, alpha_std, alpha_bir, float(animal_frac), depth, face_mesh, mask, capsule, unavailable, bust)


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
    thumb = _shade(sil, zf, px, size=320)

    m["face_found"] = None
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
    return m, preview, thumb


def _aligned_iou(sil: np.ndarray, mask: np.ndarray) -> float:
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
    return 100.0 * float((canvas & mask).sum()) / float((canvas | mask).sum())

SCORE_FORMULA = (
    "score = shape match % − 4 × spikes % − 25 if not sealed − 20 if thinnest wall < 0.8 mm "
    "+ (30 − face error %) when a face is found"
)
