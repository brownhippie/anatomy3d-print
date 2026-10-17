"""What a person in a photo wears, garment by garment, and which lies over
which -- for the builder to step one layer over the next (a jacket's edge
over the shirt, a tie on it, a belt round trousers, a scarf over all) and
for studies to check the model keeps those overlaps (study.layering).

CLIPSeg (see head_parts.py) names every pixel of the person: the garments
below, skin, hair and the background compete; the person is covered in
overlapping square crops (CLIPSeg's input is square), their scores
averaged. A garment counts when it covers enough of the person.
"""
from typing import Optional

import numpy as np

GARMENTS = {
    "shirt": [49406, 320, 2523, 49407],  # "a shirt"
    "tshirt": [49406, 320, 339, 268, 2523, 49407],  # "a t-shirt"
    "jacket": [49406, 320, 6164, 49407],  # "a jacket"
    "suit": [49406, 320, 3940, 6164, 49407],  # "a suit jacket"
    "coat": [49406, 320, 7356, 49407],  # "a coat"
    "sweater": [49406, 320, 11455, 49407],  # "a sweater"
    "hoodie": [49406, 320, 13444, 49407],  # "a hoodie"
    "dress": [49406, 320, 2595, 49407],  # "a dress"
    "skirt": [49406, 320, 12386, 49407],  # "a skirt"
    "trousers": [49406, 23172, 49407],  # "trousers"
    "jeans": [49406, 10157, 49407],  # "jeans"
    "shorts": [49406, 9680, 49407],  # "shorts"
    "tie": [49406, 320, 6066, 3422, 49407],  # "a necktie"
    "scarf": [49406, 320, 13365, 49407],  # "a scarf"
    "belt": [49406, 320, 7373, 49407],  # "a belt"
    "kimono": [49406, 320, 40024, 49407],  # "a kimono"
    "shoes": [49406, 4079, 49407],  # "shoes"
    "gloves": [49406, 12363, 49407],  # "gloves"
    "strap": [49406, 320, 3365, 13946, 49407],  # "a bag strap"
    "watch": [49406, 320, 19243, 1239, 49407],  # "a wristwatch"
    "necklace": [49406, 320, 7385, 49407],  # "a necklace"
    "buttons": [49406, 16188, 49407],  # "buttons"
    "pocket": [49406, 320, 8504, 49407],  # "a pocket"
    "badge": [49406, 320, 11301, 541, 6974, 49407],  # "a badge or medal"
    "lanyard": [49406, 320, 46808, 49407],  # "a lanyard"
    "zip": [49406, 320, 33670, 49407],  # "a zipper"
    "socks": [49406, 8774, 49407],  # "socks"
    "boots": [49406, 7319, 49407],  # "boots"
    "backpack": [49406, 320, 14894, 13946, 49407],  # "a backpack strap"
    "bracelet": [49406, 320, 8969, 49407],  # "a bracelet"
}
OTHERS = {
    "skin": [49406, 3575, 49407],
    "hair": [49406, 2225, 49407],
    "background": [49406, 518, 5994, 49407],
}
# parts of a garment read on their own after the garments (close up, or from
# how the garment is made): a shirt's collar, a jacket's lapels. Last in
# NAMES so the earlier labels keep their numbers.
PARTS = ("collar", "lapel")
NAMES = list(GARMENTS) + list(OTHERS) + list(PARTS)  # label i+1 = NAMES[i]; 0 = not the person
WORN = list(GARMENTS) + list(PARTS)  # what is worn (everything but skin, hair, the background)


def worn_ids() -> list:
    """The labels (1 + NAMES index) of everything worn."""
    return [NAMES.index(k) + 1 for k in WORN]
# how far out each lies (skin 0): what's outer stands over what's inner where they meet
LAYER = {"skin": 0, "shirt": 1, "tshirt": 1, "dress": 1, "kimono": 1, "trousers": 1, "jeans": 1, "shorts": 1, "skirt": 1.5,
         "sweater": 2, "hoodie": 2.5, "tie": 2.2, "jacket": 3, "suit": 3, "coat": 4, "belt": 3.2, "scarf": 5,
         "strap": 5.5, "watch": 2, "gloves": 1, "shoes": 1, "hair": 6, "necklace": 2.5, "lanyard": 4.5, "backpack": 5.5,
         "socks": 0.5, "boots": 1.2, "bracelet": 2, "collar": 1.5, "lapel": 3.5}
# small things that sit on whatever garment is round them: half a layer over it
ON_TOP = ("buttons", "pocket", "badge", "zip")
_TIE_CROP = (3.2, 4.5)  # the close looks for a tie and a collar: squares this many face widths across, from the chin down
_COLLAR_PROMPT = [49406, 320, 2523, 13497, 49407]  # "a shirt collar"
_COLLAR_BAND = 0.9  # ... a collar only this many face heights below the chin
_LAPEL_W = 0.55  # a lapel's width at its top, face widths (a jacket's lapels run along its opening, narrowing to the button)
_TIE_MARGIN = 0.3  # ... where "a necktie" outscores the other garments by this much (logits)
_MIN_SHARE = {"tie": 0.003, "belt": 0.003, "watch": 0.0005, "strap": 0.003, "gloves": 0.002, "shoes": 0.003, "necklace": 0.001,
              "buttons": 0.0005, "pocket": 0.002, "badge": 0.001, "lanyard": 0.002, "zip": 0.001, "socks": 0.002, "boots": 0.003,
              "backpack": 0.003, "bracelet": 0.0005, "collar": 0.001, "lapel": 0.003}
_MIN_SHARE_DEFAULT = 0.02


def read_clothes(rgb: np.ndarray, person: np.ndarray, face_mesh: Optional[np.ndarray] = None) -> Optional[dict]:
    """{"labels": uint8 (H, W) (0 off the person, else 1 + NAMES index),
    "found": {garment: share of the person}}."""
    from PIL import Image

    from .head_parts import _back, _clipseg, _square

    ys, xs = np.nonzero(person)
    if len(ys) < 500:
        return None
    H, W = person.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(x1 - x0, 64) * 1.05)
    prompts = {**GARMENTS, **OTHERS}
    acc = np.zeros((len(prompts), H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    n = max(1, int(np.ceil((y1 - y0) / (0.6 * side))))
    for cy in np.linspace(y0 + side / 2, max(y1 - side / 2, y0 + side / 2), n):
        crop, (cx0, cy0), (sx0, sy0, sx1, sy1) = _square(rgb, (x0 + x1) / 2.0, float(cy), side)
        logit = _clipseg(crop, prompts)
        for i, k in enumerate(prompts):
            acc[i, sy0:sy1, sx0:sx1] += _back(logit[k].astype(np.float32), side)[sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0]
        cnt[sy0:sy1, sx0:sx1] += 1
    # garments only where the body segmenter sees clothes (CLIPSeg alone
    # called faces and a face mask "a kimono"); skin and hair as it says
    seg = _segment(rgb)
    from .head_parts import ACCESSORY, BODY_SKIN, CLOTHES, FACE_SKIN, HAIR

    score = acc / np.maximum(cnt, 1)[None]
    g = len(GARMENTS)
    win = np.argmax(score[:g], axis=0)
    labels = np.zeros((H, W), np.uint8)
    worn = person & (cnt > 0) & np.isin(seg, (CLOTHES, ACCESSORY))
    if face_mesh is not None:
        # the head's own wear (cap, mask, glasses) is head_parts' to name
        worn &= np.arange(H)[:, None] > float(np.asarray(face_mesh)[152, 1])
    labels[worn] = win[worn] + 1
    collar = np.zeros((H, W), bool)
    if face_mesh is not None and "tie" in GARMENTS:
        # a tie and a shirt's collar, looked for close up: in crops as wide as
        # the person a tie is a narrow strip and the jacket round it always
        # outscored it -- round the neck and chest it can win where it is
        # (two sizes of crop: the tighter finds the knot, the wider the tie's
        # length down to the arms)
        fm = np.asarray(face_mesh)[:468, :2]
        fw = float(np.ptp(fm[:, 0]))
        fh = float(np.ptp(fm[:, 1]))
        chin = float(np.asarray(face_mesh)[152, 1])
        tie = np.zeros((H, W), bool)
        near = {k: GARMENTS[k] for k in ("tie", "shirt", "suit", "jacket", "coat", "sweater") if k in GARMENTS}
        near["collar"] = _COLLAR_PROMPT
        for mult in _TIE_CROP:
            tside = int(max(mult * fw, 64))
            crop, (cx0, cy0), (sx0, sy0, sx1, sy1) = _square(rgb, float(fm[:, 0].mean()), chin + tside / 2.0, tside)
            lg = _clipseg(crop, near)
            sc = {k: _back(lg[k].astype(np.float32), tside)[sy0 - cy0:sy1 - cy0, sx0 - cx0:sx1 - cx0] for k in near}
            others = np.max(np.stack([v for k, v in sc.items() if k not in ("tie", "collar")]), axis=0)
            tie[sy0:sy1, sx0:sx1] |= (sc["tie"] > others + _TIE_MARGIN) & (sc["tie"] > 0.0)
            collar[sy0:sy1, sx0:sx1] |= (sc["collar"] > np.maximum(others, sc["tie"]) + _TIE_MARGIN) & (sc["collar"] > 0.0)
        labels[tie & worn] = NAMES.index("tie") + 1
        rows = np.arange(H)[:, None]
        collar &= worn & ~tie & (rows > chin) & (rows < chin + _COLLAR_BAND * fh)
        labels[person & np.isin(seg, (BODY_SKIN, FACE_SKIN))] = NAMES.index("skin") + 1
    labels[person & (seg == HAIR)] = NAMES.index("hair") + 1
    labels[collar & (labels > 0) & (labels <= len(GARMENTS))] = NAMES.index("collar") + 1
    if face_mesh is not None:
        lap = lapels(labels, face_mesh)
        labels[lap] = NAMES.index("lapel") + 1
    total = float(person.sum())
    found = {}
    for k in WORN:
        i = NAMES.index(k)
        share = float((labels == i + 1).sum()) / total
        if k in ("shirt", "tshirt"):
            share += float((labels == NAMES.index("collar") + 1).sum()) / total
        if share >= _MIN_SHARE.get(k, _MIN_SHARE_DEFAULT):
            found[k] = round(share, 4)
        else:
            labels[labels == i + 1] = 0  # too little to trust
    _ = Image  # (PIL is CLIPSeg's resizer)
    out = {"labels": labels, "found": found}
    if face_mesh is not None:
        try:
            out["neckline"] = neckline(out, face_mesh)
        except Exception:  # noqa: BLE001 - unnamed then
            out["neckline"] = None
    return out


def lapels(labels: np.ndarray, face_mesh) -> np.ndarray:
    """A jacket's lapels, from how a jacket is made: they run along its
    opening (where the shirt, the tie or the collar shows between its two
    fronts), widest at the top and narrowing to the opening's foot, each
    _LAPEL_W face widths across at the top. The opening's own two edges in
    the photo place them; nothing when no jacket or no opening is read."""
    from scipy.ndimage import binary_closing, label as cc_label

    H, W = labels.shape
    jk = [NAMES.index(k) + 1 for k in ("suit", "jacket", "coat") if k in NAMES]
    op_ids = [NAMES.index(k) + 1 for k in ("shirt", "tshirt", "tie", "collar", "sweater", "dress") if k in NAMES]
    jacket = np.isin(labels, jk)
    if jacket.sum() < 200:
        return np.zeros((H, W), bool)
    fm = np.asarray(face_mesh)[:468, :2]
    fw = float(np.ptp(fm[:, 0]))
    cx = float(fm[:, 0].mean())
    chin = float(np.asarray(face_mesh)[152, 1])
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    opening = np.isin(labels, op_ids) & (rows > chin) & (np.abs(cols - cx) < 1.2 * fw)
    opening = binary_closing(opening, iterations=2)
    comps, n = cc_label(opening)
    if n == 0:
        return np.zeros((H, W), bool)
    # the opening: its largest piece (a fleck of shirt elsewhere isn't it)
    k = int(np.argmax(np.bincount(comps.ravel())[1:])) + 1
    op = comps == k
    rr = np.flatnonzero(op.any(axis=1))
    y_t, y_b = int(rr.min()), int(rr.max())
    if y_b - y_t < 0.5 * fw:
        return np.zeros((H, W), bool)
    out = np.zeros((H, W), bool)
    wmax = _LAPEL_W * fw
    for y in range(y_t, y_b + 1):
        xs = np.flatnonzero(op[y])
        if not len(xs):
            continue
        w = wmax * (1.0 - (y - y_t) / max(y_b - y_t, 1))
        if w < 1.0:
            continue
        xl, xr = int(xs.min()), int(xs.max())
        a, b = max(int(xl - w), 0), xl
        out[y, a:b] = jacket[y, a:b]
        a, b = xr + 1, min(int(xr + 1 + w), W)
        out[y, a:b] |= jacket[y, a:b]
    return out


def _segment(rgb: np.ndarray) -> np.ndarray:
    """MediaPipe's selfie multiclass parts (head_parts' labels 0-5) over the
    whole photo, at up to 1024 px."""
    import mediapipe as mp
    from PIL import Image

    from .head_parts import _back
    from .person_segmenter import _get_segmenter

    H, W = rgb.shape[:2]
    k = min(1.0, 1024.0 / max(H, W))
    img = np.ascontiguousarray(np.asarray(Image.fromarray(rgb[..., :3]).resize((max(1, int(W * k)), max(1, int(H * k))), Image.BILINEAR)))
    res = _get_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    conf = np.stack([np.asarray(Image.fromarray(c.numpy_view().squeeze().astype(np.float32)).resize((W, H), Image.BILINEAR))
                     for c in res.confidence_masks])
    _ = _back
    return conf.argmax(axis=0).astype(np.uint8)


def layer_map(reading: dict) -> np.ndarray:
    """Each pixel's layer (LAYER; the ON_TOP things half a layer over the
    garment round them), NaN where nothing was named."""
    from scipy.ndimage import distance_transform_edt

    lut = np.full(len(NAMES) + 1, np.nan, np.float32)
    for i, k in enumerate(NAMES):
        if k in LAYER:
            lut[i + 1] = LAYER[k]
    L = lut[reading["labels"]]
    top = np.isin(reading["labels"], [NAMES.index(k) + 1 for k in ON_TOP])
    if top.any():
        known = np.isfinite(L) & (reading["labels"] > 0)
        if known.any():
            iy, ix = distance_transform_edt(~known, return_distances=False, return_indices=True)
            L = np.where(top, L[iy, ix] + 0.5, L)
    return L


def neckline(reading: dict, face_mesh: np.ndarray) -> Optional[str]:
    """V, crew or collared: the shape of the skin showing below the chin."""
    labels = reading["labels"]
    fm = np.asarray(face_mesh)[:, :2]
    chin, face_h = float(fm[152, 1]), float(np.ptp(fm[:468, 1]))
    cx = float((fm[234, 0] + fm[454, 0]) / 2)
    skin = labels == NAMES.index("skin") + 1
    widths = []
    for k in np.linspace(0.15, 1.2, 12):
        r = int(chin + k * face_h)
        if r >= labels.shape[0]:
            break
        cols = np.flatnonzero(skin[r, max(int(cx - face_h), 0):int(cx + face_h)])
        widths.append(len(cols))
    widths = np.array(widths, float)
    if len(widths) < 6 or widths[:2].max() == 0:
        return None
    collar = any(k in reading["found"] for k in ("shirt", "tie")) and widths[2:].max() < 0.2 * face_h
    if collar:
        return "collared"
    deep = widths[3:8]
    if len(deep) and (deep > 0.1 * face_h).sum() >= 3 and np.all(np.diff(deep[deep > 0]) <= 0.05 * face_h):
        return "V"
    return "crew"
