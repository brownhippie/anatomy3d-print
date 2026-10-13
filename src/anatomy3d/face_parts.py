"""Each part of the face, read from the photo, for the builder to shape it:
whether the mouth is open and what shows in it (teeth, tongue, the dark of
the mouth), whether each eye is open and where its iris is, the eyebrows'
hair, the nostrils, a beard, moustache or stubble, and scars.

- Expression: MediaPipe's face landmarker (Apache-2.0, the model
  face_features.py already uses) with its 52 blendshape scores -- jawOpen,
  eyeBlink, browInnerUp... -- on a crop round the face.
- Mouth: the inner lips' outline from the 478 landmarks; inside it, teeth
  are bright and colourless, the tongue red, the rest the mouth's dark.
- Eyes: lid gap over eye width per eye; irises from landmarks 468-477.
- Eyebrows: between the brows' upper and lower landmark lines, the pixels
  darker than the skin round them.
- Nostrils: dark blobs under the nose tip, between its wings.
- Facial hair: the head-parts map's hair on the lower face (a beard round
  the chin and jaw, a moustache over the lip); stubble: the jaw darker and
  rougher than the cheeks above it.
- Scars: thin lines (a Hessian ridge filter, light or dark) on the face's
  visible skin, away from the eyes, brows, lips, nostrils, the smile lines
  and the face's edge; only long, narrow, clear ones count.
All positions in the photo's own pixels.
"""
from typing import Optional

import numpy as np

INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
EYE = {"R": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
       "L": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]}
EYE_LIDS = {"R": (159, 145, 33, 133), "L": (386, 374, 263, 362)}  # upper, lower, outer, inner corner
IRIS = {"R": list(range(468, 473)), "L": list(range(473, 478))}
BROW_UP = {"R": [70, 63, 105, 66, 107], "L": [300, 293, 334, 296, 336]}
BROW_LO = {"R": [46, 53, 52, 65, 55], "L": [276, 283, 282, 295, 285]}
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

_MOUTH_OPEN = 0.08  # inner-lip gap over mouth width
_EYE_CLOSED = 0.12  # lid gap over eye width
_SCAR_MIN_LEN = 0.08  # x the face's size
_SCAR_MIN_CONTRAST = 12.0  # ridge response over the skin's median
_SCAR_MAX_W = 0.02  # x the face's size
_blend = None


def _poly(shape, pts) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("1", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in pts], fill=1)
    return np.asarray(img, dtype=bool)


def _blendshapes(rgb: np.ndarray, f: np.ndarray, size: float) -> dict:
    global _blend
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from PIL import Image

    from .face_features import _ensure_model
    from .head_parts import _square

    if _blend is None:
        opts = mp_vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
                                               num_faces=1, output_face_blendshapes=True)
        _blend = mp_vision.FaceLandmarker.create_from_options(opts)
    crop, _, _ = _square(rgb, float(f[:, 0].mean()), float(f[:, 1].mean()), int(2.0 * size))
    img = np.ascontiguousarray(np.asarray(Image.fromarray(crop).resize((512, 512), Image.BILINEAR)))
    res = _blend.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    if not res.face_blendshapes:
        return {}
    return {c.category_name: round(float(c.score), 3) for c in res.face_blendshapes[0]}


def _lab(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* of an RGB uint8 image (D65)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], dtype=np.float32)
    xyz /= np.array([0.9505, 1.0, 1.089], dtype=np.float32)
    fx = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * fx[..., 1] - 16, 500 * (fx[..., 0] - fx[..., 1]), 200 * (fx[..., 1] - fx[..., 2])], axis=-1)


def read_face(rgb: np.ndarray, face_mesh: np.ndarray, labels: Optional[np.ndarray] = None) -> Optional[dict]:
    from scipy import ndimage

    from .head_parts import FACE_SKIN, HAIR, MASK

    f = np.asarray(face_mesh, dtype=np.float64)
    if len(f) < 478:
        return None
    p = f[:, :2]
    size = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])))
    H, W = rgb.shape[:2]
    pad = int(0.3 * size)
    x0, y0 = max(int(p[:, 0].min()) - pad, 0), max(int(p[:, 1].min()) - pad, 0)
    x1, y1 = min(int(p[:, 0].max()) + pad, W), min(int(p[:, 1].max()) + pad, H)
    crop = rgb[y0:y1, x0:x1, :3]
    q = p - [x0, y0]
    shape = crop.shape[:2]
    lab = _lab(crop)
    L = lab[..., 0]
    lab_c = None if labels is None else labels[y0:y1, x0:x1]
    oval = _poly(shape, q[FACE_OVAL])
    skin = oval if lab_c is None else oval & (lab_c == FACE_SKIN)
    masked = lab_c is not None and (lab_c[oval] == MASK).mean() > 0.15
    skin_L = float(np.median(L[skin])) if skin.any() else float(np.median(L[oval]))
    out = {"box": (x0, y0, x1, y1), "size_px": size, "masked": bool(masked)}
    try:
        out["blend"] = _blendshapes(rgb, p, size)
    except Exception:  # noqa: BLE001 - the landmarks' own measures still hold
        out["blend"] = {}
    b = out["blend"]
    d = lambda a, c: float(np.linalg.norm(p[a] - p[c]))  # noqa: E731

    # mouth
    gap, width = d(13, 14), d(78, 308)
    is_open = (gap / max(width, 1e-6) > _MOUTH_OPEN or b.get("jawOpen", 0.0) > 0.35) and not masked
    mouth = {"open": bool(is_open), "open_ratio": round(gap / max(width, 1e-6), 3), "width_px": round(d(61, 291), 1),
             "upper_lip_px": round(d(0, 13), 1), "lower_lip_px": round(d(14, 17), 1)}
    if is_open:
        inside = _poly(shape, q[INNER_LIPS])
        if inside.sum() >= 12:
            a, bb = lab[..., 1], lab[..., 2]
            chroma = np.hypot(a, bb)
            teeth = inside & (L > max(0.85 * skin_L, 45.0)) & (chroma < 22)
            teeth = ndimage.binary_opening(teeth, iterations=1)
            # the tongue: a red mass low in the mouth, below the teeth (red
            # along the teeth's edges is gums and lips)
            ys_in = np.nonzero(inside)[0]
            low = np.arange(shape[0])[:, None] > (ys_in.min() + 0.55 * np.ptp(ys_in))
            if teeth.any():
                low &= np.arange(shape[0])[:, None] > np.median(np.nonzero(teeth)[0])
            tongue = ndimage.binary_opening(inside & ~teeth & (a > 18) & (L > 0.35 * skin_L) & low, iterations=2)
            if tongue.sum() < 0.08 * inside.sum():
                tongue = np.zeros_like(tongue)
            dark = inside & ~teeth & ~tongue
            mouth.update(teeth_share=round(float(teeth.sum() / inside.sum()), 3),
                         tongue_share=round(float(tongue.sum() / inside.sum()), 3),
                         dark_share=round(float(dark.sum() / inside.sum()), 3))
            mouth["inside"], mouth["teeth"], mouth["tongue"] = inside, teeth, tongue
    out["mouth"] = mouth

    # eyes
    eyes = {}
    for s in ("R", "L"):
        up, lo, oc, ic = EYE_LIDS[s]
        ratio = d(up, lo) / max(d(oc, ic), 1e-6)
        blink = b.get("eyeBlinkLeft" if s == "L" else "eyeBlinkRight", 0.0)
        iris = p[IRIS[s]]
        r_iris = float(np.mean(np.linalg.norm(iris[1:] - iris[0], axis=1)))
        eyes[s] = {"open": bool(ratio > _EYE_CLOSED and blink < 0.5), "open_ratio": round(ratio, 3),
                   "iris": (float(iris[0, 0]), float(iris[0, 1])), "iris_r_px": round(r_iris, 2),
                   "opening": _poly(shape, q[EYE[s]])}
    out["eyes"] = eyes

    # eyebrows: the dark hair between the brow lines
    brows = {}
    for s in ("R", "L"):
        band = _poly(shape, np.vstack([q[BROW_UP[s]], q[BROW_LO[s]][::-1]]))
        around = ndimage.binary_dilation(band, iterations=max(2, int(0.03 * size))) & ~band & skin
        if band.sum() < 6:
            continue
        ref = float(np.median(L[around])) if around.sum() > 10 else skin_L
        hair = band & (L < ref - 6.0)
        heights = [np.ptp(np.flatnonzero(hair[:, c])) + 1 for c in range(shape[1]) if hair[:, c].any()]
        brows[s] = {"mask": hair, "share": round(float(hair.sum() / band.sum()), 3),
                    "contrast": round(ref - float(np.median(L[hair])) if hair.any() else 0.0, 1),
                    "thickness_px": round(float(np.median(heights)) if heights else 0.0, 1)}
    out["brows"] = brows

    # nostrils: dark blobs under the tip, between the wings
    if not masked:
        zone = _poly(shape, q[[64, 98, 97, 2, 326, 327, 294, 278, 4, 48]])
        dark = zone & (L < 0.55 * skin_L)
        lab_n, n = ndimage.label(dark)
        blobs = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab_n == i)
            if len(ys) >= max(4, int(0.0004 * size * size)):
                blobs.append({"c": (float(xs.mean() + x0), float(ys.mean() + y0)), "r": float(np.sqrt(len(ys) / np.pi))})
        out["nostrils"] = sorted(blobs, key=lambda o: -o["r"])[:2]
    else:
        out["nostrils"] = []

    # facial hair
    fh = {"beard": None, "moustache": None, "stubble": 0.0}
    if lab_c is not None and not masked:
        nose_y, lip_y, chin_y = q[2, 1], q[0, 1], q[152, 1]
        yy = np.arange(shape[0])[:, None]
        lower = oval & (yy > lip_y)
        over_lip = oval & (yy > nose_y) & (yy < lip_y) & ~_poly(shape, q[OUTER_LIPS])
        hair = lab_c == HAIR
        if lower.any() and hair[lower].mean() > 0.12:
            fh["beard"] = hair & lower
        if over_lip.any() and hair[over_lip].mean() > 0.2:
            fh["moustache"] = hair & over_lip
        jaw = oval & (yy > lip_y) & (yy < chin_y) & ~_poly(shape, q[OUTER_LIPS])
        cheek = oval & (yy > q[1, 1] - 0.1 * size) & (yy < nose_y) & ~_poly(shape, q[[64, 4, 294, 2]])
        if jaw.sum() > 20 and cheek.sum() > 20:
            hp = L - ndimage.gaussian_filter(L, max(1.0, 0.01 * size))
            darker = float(np.median(L[cheek]) - np.median(L[jaw]))
            rough = float(hp[jaw].std() / (hp[cheek].std() + 1e-6))
            fh["stubble"] = round(float(np.clip((darker - 4.0) / 12.0, 0, 1) * np.clip(rough - 1.0, 0, 1)), 3)
    out["facial_hair"] = fh

    # scars
    out["scars"] = [] if masked else _scars(L, q, skin, size, shape, x0, y0)
    return out


def _scars(L: np.ndarray, q: np.ndarray, skin: np.ndarray, size: float, shape, x0: int, y0: int) -> list:
    from scipy import ndimage
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    from skimage.morphology import skeletonize

    sig = max(1.0, 0.006 * size)
    ev = hessian_matrix_eigvals(hessian_matrix(L, sigma=sig, order="rc", use_gaussian_derivatives=True))
    resp = np.maximum(np.abs(ev[0]), np.abs(ev[1])) * sig ** 2
    # away from the features, their creases and the face's edge
    keep = ndimage.binary_erosion(skin, iterations=int(3 * sig))
    for idx in (EYE["R"], EYE["L"], OUTER_LIPS, BROW_UP["R"] + BROW_LO["R"][::-1], BROW_UP["L"] + BROW_LO["L"][::-1],
                [64, 98, 97, 2, 326, 327, 294, 278, 4, 48]):
        keep &= ~ndimage.binary_dilation(_poly(shape, q[idx]), iterations=int(0.04 * size))
    # smile lines and the lines on from the mouth's corners to the jaw;
    # crow's feet; the forehead's lines: wrinkles, not scars
    for chain in ((64, 61, 150), (294, 291, 379), (48, 61, 172), (278, 291, 397), (33, 127), (263, 356)):
        line = np.zeros(shape, bool)
        for a, c in zip(chain[:-1], chain[1:]):
            for t in np.linspace(0, 1, 60):
                x, y = q[a] * (1 - t) + q[c] * t
                if 0 <= int(y) < shape[0] and 0 <= int(x) < shape[1]:
                    line[int(y), int(x)] = True
        keep &= ~ndimage.binary_dilation(line, iterations=int(0.07 * size))
    if keep.sum() < 50:
        return []
    med = float(np.median(resp[keep])) + 1e-6
    strong = keep & (resp > _SCAR_MIN_CONTRAST * med)
    lab_s, n = ndimage.label(strong)
    out = []
    for i in range(1, n + 1):
        comp = lab_s == i
        ys, xs = np.nonzero(comp)
        if len(ys) < 5:
            continue
        cov = np.cov(np.vstack([xs, ys]))
        w, v = np.linalg.eigh(cov)
        length = 4.0 * np.sqrt(max(w[1], 0))
        width = 4.0 * np.sqrt(max(w[0], 0))
        horizontal = abs(v[0, 1]) > 0.9  # the long axis lies across
        on_forehead = ys.mean() < q[[105, 334], 1].mean()
        if (length >= _SCAR_MIN_LEN * size and length >= 5.0 * max(width, 1.0) and width <= _SCAR_MAX_W * size
                and not (horizontal and on_forehead)):
            sk = skeletonize(comp)
            py, px = np.nonzero(sk)
            out.append({"points": np.column_stack([px + x0, py + y0]).astype(np.float32), "length_px": round(float(length), 1),
                        "width_px": round(float(width), 1)})
    return out
