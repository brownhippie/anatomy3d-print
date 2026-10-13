"""What each pixel round a person's head is: hair, face skin, other skin,
clothes, a cap or hat, a face mask, glasses, or another accessory (an
earring, a flower in the hair). The builder needs it to know the hair (its
outline, how thick it lies on the skull, whether it covers the ears, how far
down it hangs), to sculpt only the face the photo really shows (nothing under
a mask or behind sunglasses), and to give a cap its crown and peak.

Two open models, both run here on the CPU:
- MediaPipe's selfie multiclass segmenter (Apache-2.0, the one
  person_segmenter.py already runs on the whole photo), on a square crop
  round the face and down to the shoulders, where a head is big enough in its
  256 px input to tell an ear from the hair over it: hair / face skin / skin
  / clothes / accessory, pixel by pixel. It doesn't say which accessory --
  it split a cap into "hair" and "accessory" patches.
- CLIPSeg (CIDAS/clipseg-rd64-refined, Apache-2.0; the int8 ONNX export
  Xenova/clipseg-rd64-refined, 139 MB) names them: on the same crop, prompts
  for a face, hair, a cap or hat, a face mask, clothes and the background
  compete for each pixel; on a crop round the eyes, glasses and sunglasses
  are scored on their own (they lose to "a face" when competing).
"""
from pathlib import Path
from typing import Optional

import numpy as np

BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, ACCESSORY, HAT, MASK, GLASSES = range(9)
NAMES = ["background", "hair", "skin", "face", "clothes", "accessory", "hat", "mask", "glasses"]

_CROP = 4.0  # head crop side, x the face's size: an updo above, long hair below
_DROP = 0.35  # its centre below the face's centre, x the face's size
_SEG_INPUT = 512
_NAME_CROP = 2.6  # the crop accessories are named on, x the face's size, round the face

CLIPSEG_URL = "https://huggingface.co/Xenova/clipseg-rd64-refined/resolve/main/onnx/model_quantized.onnx"
CLIPSEG_CACHE = Path.home() / ".cache" / "anatomy3d-print" / "clipseg_rd64_refined_quantized.onnx"
_CLIP_INPUT = 352
_CLIP_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_CLIP_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# CLIP's own tokens for each prompt (its BPE tokenizer, computed once: no
# tokenizer needed at run time)
_HEAD_PROMPTS = {
    "face": [49406, 320, 1710, 49407],
    "hair": [49406, 2225, 49407],
    "hat": [49406, 320, 3938, 541, 3801, 49407],  # "a cap or hat"
    "mask": [49406, 320, 1710, 8306, 9462, 518, 4932, 537, 8231, 49407],  # "a face mask covering the mouth and nose"
    "clothes": [49406, 7080, 49407],
    "background": [49406, 518, 5994, 49407],
}
_EYE_PROMPTS = {
    "eyes": [49406, 3095, 537, 3575, 49407],  # "eyes and skin"
    "glasses": [49406, 5034, 6116, 49407],  # "eyeglasses"
    "sunglasses": [49406, 12906, 49407],
}
# how sure each has to be (measured on our test photos: glasses scored 0.70
# on a woman wearing them, at most 0.04 on five without; sunglasses up to
# 0.29 on shaded eyes under a cap)
_GLASSES_MIN = 0.5
_SUNGLASSES_MIN = 0.6
_GLASSES_AREA = 0.35  # the glasses: where they score above this
_HAT_AREA = 0.08  # a cap/hat counts from this share of the face's box
_MASK_AREA = 0.15  # a face mask counts from this share of the lower face

_clipseg_sess = None


def _clipseg_session():
    global _clipseg_sess
    if _clipseg_sess is None:
        import onnxruntime as ort

        from .person_segmenter import _ensure_model

        _clipseg_sess = ort.InferenceSession(_ensure_model(CLIPSEG_URL, CLIPSEG_CACHE), providers=["CPUExecutionProvider"])
    return _clipseg_sess


def _clipseg(img: np.ndarray, prompts: dict) -> dict:
    """Logits (352 x 352) per prompt for an RGB uint8 image."""
    from PIL import Image

    x = np.asarray(Image.fromarray(img).resize((_CLIP_INPUT, _CLIP_INPUT), Image.BICUBIC), dtype=np.float32) / 255.0
    x = ((x - _CLIP_MEAN) / _CLIP_STD).transpose(2, 0, 1)[None].astype(np.float32)
    names = list(prompts)
    n = max(len(t) for t in prompts.values())
    ids = np.full((len(names), n), 49407, dtype=np.int64)
    att = np.zeros_like(ids)
    for i, k in enumerate(names):
        ids[i, :len(prompts[k])] = prompts[k]
        att[i, :len(prompts[k])] = 1
    out = _clipseg_session().run(None, {"input_ids": ids, "pixel_values": x, "attention_mask": att})[0]
    return {k: out[i] for i, k in enumerate(names)}


def _square(rgb: np.ndarray, cx: float, cy: float, side: int):
    H, W = rgb.shape[:2]
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    crop = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1, :3]
    return crop, (x0, y0), (sx0, sy0, sx1, sy1)


def _back(a: np.ndarray, side: int, nearest: bool = False) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(a).resize((side, side), Image.NEAREST if nearest else Image.BILINEAR))


def detect_head_parts(rgb: np.ndarray, face_mesh: Optional[np.ndarray]) -> Optional[dict]:
    """{"labels": uint8 (H, W), one of the constants above per photo pixel
    (BACKGROUND outside the crop), "box": the crop (x0, y0, x1, y1),
    "found": which of hat / mask / glasses / sunglasses the face has,
    "hair": hair_pattern.read_hair's reading}. None without a face."""
    if face_mesh is None:
        return None
    import mediapipe as mp
    from scipy import ndimage

    from .person_segmenter import _get_segmenter

    H, W = rgb.shape[:2]
    f = np.asarray(face_mesh, dtype=np.float64)[:, :2]
    size = max(float(np.ptp(f[:, 0])), float(np.ptp(f[:, 1])))
    if size < 8:
        return None
    side = int(round(_CROP * size))
    crop, (x0, y0), (sx0, sy0, sx1, sy1) = _square(rgb, float(f[:, 0].mean()), float(f[:, 1].mean()) + _DROP * size, side)
    if sx1 <= sx0 or sy1 <= sy0:
        return None

    # hair / skin / clothes / accessory, pixel by pixel
    from PIL import Image

    img = np.ascontiguousarray(np.asarray(Image.fromarray(crop).resize((_SEG_INPUT, _SEG_INPUT), Image.BILINEAR)))
    result = _get_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    conf = [c.numpy_view().squeeze().astype(np.float32) for c in result.confidence_masks]
    part = np.stack([_back(c, side) for c in conf]).argmax(axis=0).astype(np.uint8)
    person = part != BACKGROUND

    # in crop pixels: the face's own landmarks and lines
    fx, fy = f[:, 0] - x0, f[:, 1] - y0
    eye_y = float((fy[33] + fy[263]) / 2.0)
    ruler = float(fy[2] - (fy[105] + fy[334]) / 2.0)  # brow line to nose base
    yy, xx = np.mgrid[0:side, 0:side]
    from PIL import ImageDraw

    oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
            149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    face_img = Image.new("1", (side, side), 0)
    ImageDraw.Draw(face_img).polygon([(float(fx[i]), float(fy[i])) for i in oval], fill=1)
    in_face = np.asarray(face_img, dtype=bool)
    face_box = float(np.ptp(fx)) * float(np.ptp(fy))

    labels = part.copy()
    found = {"hat": False, "mask": False, "glasses": False, "sunglasses": False}
    try:
        # named on a tighter crop round the head: a cap is too small to name
        # in the big one
        hside = max(int(round(_NAME_CROP * size)), 8)
        hx, hy = float(f[:, 0].mean()) - x0, float(f[:, 1].mean()) - y0
        hcrop, (hx0, hy0), _ = _square(crop, hx, hy, hside)
        logit = _clipseg(hcrop, _HEAD_PROMPTS)
        names = list(logit)
        hwin = np.stack([_back(logit[k].astype(np.float32), hside) for k in names]).argmax(axis=0)
        win = np.full((side, side), names.index("background"))
        ty0, tx0 = max(hy0, 0), max(hx0, 0)
        ty1, tx1 = min(hy0 + hside, side), min(hx0 + hside, side)
        win[ty0:ty1, tx0:tx1] = hwin[ty0 - hy0:ty1 - hy0, tx0 - hx0:tx1 - hx0]
        # a cap or hat: where it wins over face, hair and the rest, together
        # with the hair/accessory patches the segmenter split it into that
        # touch it above the eyes
        hat = (win == names.index("hat")) & person & (yy < eye_y - 0.1 * ruler)  # never below the eyes
        if hat.sum() >= _HAT_AREA * face_box:
            near = ((part == HAIR) | (part == ACCESSORY)) & (yy < eye_y - 0.2 * ruler)
            lab, _ = ndimage.label(near | hat)
            keep = np.unique(lab[hat])
            hat = np.isin(lab, keep[keep > 0])
            labels[hat] = HAT
            found["hat"] = True
        # a face mask: the lower face that isn't skin
        lower = in_face & (yy > eye_y + 0.25 * ruler)
        mask = lower & ((win == names.index("mask")) | (part == ACCESSORY) | (part == CLOTHES))
        if mask.sum() >= _MASK_AREA * max(lower.sum(), 1):
            mask = ndimage.binary_closing(mask, iterations=3) & lower
            labels[mask] = MASK
            found["mask"] = True
        # glasses: scored alone on a crop round the eyes
        ex, ey = float(np.mean(fx[[33, 133, 362, 263, 105, 334, 50, 280]])), float(np.mean(fy[[33, 133, 362, 263, 105, 334, 50, 280]]))
        eside = max(int(round(1.5 * np.ptp(fx[[33, 263, 234, 454]]))), 8)
        ecrop, (ex0, ey0), _ = _square(crop, ex, ey, eside)
        el = _clipseg(ecrop, _EYE_PROMPTS)
        sig = {k: 1.0 / (1.0 + np.exp(-v)) for k, v in el.items()}
        g, s = float(sig["glasses"].max()), float(sig["sunglasses"].max())
        if g >= _GLASSES_MIN or s >= _SUNGLASSES_MIN:
            found["glasses"] = g >= _GLASSES_MIN
            found["sunglasses"] = s >= _SUNGLASSES_MIN
            region = _back(np.maximum(sig["glasses"], sig["sunglasses"]).astype(np.float32), eside) > _GLASSES_AREA
            gy, gx = np.nonzero(region)
            gy, gx = gy + ey0, gx + ex0
            ok = (gy >= 0) & (gy < side) & (gx >= 0) & (gx < side)
            glasses = np.zeros((side, side), dtype=bool)
            glasses[gy[ok], gx[ok]] = True
            labels[glasses & person & (labels != HAT)] = GLASSES
    except Exception:  # noqa: BLE001 - without names, the segmenter's own parts still hold
        pass

    out = np.zeros((H, W), dtype=np.uint8)
    out[sy0:sy1, sx0:sx1] = labels[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    hair = None
    try:
        from .hair_pattern import read_hair

        hair = read_hair(rgb, out, (sx0, sy0, sx1, sy1), face_mesh, HAIR, HAT)
    except Exception:  # noqa: BLE001 - the hair's volume is still built without its pattern
        hair = None
    face = None
    try:
        from .face_parts import read_face

        face = read_face(rgb, face_mesh, out)
    except Exception:  # noqa: BLE001 - the face keeps its landmark shape without its parts
        face = None
    return {"labels": out, "box": (sx0, sy0, sx1, sy1), "found": found, "hair": hair, "face": face}
