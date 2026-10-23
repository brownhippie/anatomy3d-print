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
_COLLAR_REACH = 0.35  # ... and within this many face heights of the neck's skin
_COLLAR_MIN_W = 0.12  # ... and nothing of it narrower than this many face widths (a wing, not the shirt's strip by the tie)
_SHIRT_SMOOTH = 0.01  # the shirt in a jacket's opening: colours averaged over this many face widths (the weave's noise) ...
_SHIRT_NEARER = 0.5  # ... taken as the shirt where its colour is this much nearer the shirt's than the jacket's ...
_SHIRT_REACH = 1.2  # ... within this many face widths of the face's middle
_LAPEL_W = 0.55  # a lapel's width at its top, face widths (a jacket's lapels run along its opening, narrowing to the button) -- when the photo shows no edge
_LAPEL_SEEK = (0.12, 0.75)  # the lapel's outer edge looked for this many face widths out from the opening ...
_LAPEL_EDGE_PCT = 85.0  # ... where the jacket's shading changes more than this percentile of it does
_LAPEL_IN = 3.0  # ... one straight edge (notch to button): the points within this many px of it
_LAPEL_SHARE = 0.3  # ... taken when that holds for at least this share of the opening's rows
_LAPEL_RANGE = (0.25, 0.9)  # ... and the lapel it makes is this many face widths across at its widest (as lapels are)
_KNOT_LOOK = 0.6  # the tie's knot: its foot (where the tie is narrowest under the knot) within this many face widths of its top
_COLLAR_PAST_KNOT = 0.05  # a shirt's collar points end at the knot's foot, give or take this many face widths
_TIE_MARGIN = 0.3  # ... where "a necktie" outscores the other garments by this much (logits)
_TIE_SMOOTH = 0.02  # the tie's whole length: its colour, a pattern's dots averaged over this many face widths ...
_TIE_MATCH = 1.5  # ... within this many times its own colour spread (Lab, the 90th percentile over what was read) ...
_TIE_MATCH_MIN = 8.0  # ... (never tighter than this, Lab units)
_TIE_REACH = 0.6  # ... joined to it, this many face widths either side of its middle at most
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
        # the head's own wear (cap, mask, glasses) is head_parts' to name:
        # nothing on the face or above the eyes (cut at the chin's row, a
        # collar rising round the neck lost its top: a straight edge across
        # the throat that stood as a wall)
        from skimage.draw import polygon as _polygon

        from .face_parts import FACE_OVAL

        fmf = np.asarray(face_mesh)
        face = np.zeros((H, W), dtype=bool)
        rr, cc = _polygon(fmf[FACE_OVAL, 1], fmf[FACE_OVAL, 0], (H, W))
        face[rr, cc] = True
        eyes_y = float(fmf[[33, 263], 1].mean())
        worn &= ~face & (np.arange(H)[:, None] > eyes_y)
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
        labels[tie_whole(labels, rgb, face_mesh) & worn] = NAMES.index("tie") + 1
        rows = np.arange(H)[:, None]
        # a collar is made round the neck: the shirt where it meets the
        # neck's skin, down to the collar's band (what "a shirt collar" names
        # away from the neck is the shirt's front beside the tie)
        from scipy import ndimage
        from scipy.ndimage import distance_transform_edt as _edt

        neck = person & np.isin(seg, (BODY_SKIN, FACE_SKIN)) & (rows > chin) & ~face
        near_neck = _edt(~neck) <= _COLLAR_REACH * fh if neck.any() else np.zeros((H, W), dtype=bool)
        shirt = labels == NAMES.index("shirt") + 1 if "shirt" in NAMES else np.zeros((H, W), dtype=bool)
        collar = (collar | shirt) & near_neck
        collar &= worn & (labels != NAMES.index("tie") + 1) & (rows < chin + _COLLAR_BAND * fh)
        # (a collar's wings are broad: the shirt's narrow strips beside the
        # tie below the knot are its front -- taken as collar, they hung
        # from it as two legs down the tie)
        r_c = max(1, int(round(_COLLAR_MIN_W / 2.0 * fw)))
        disk_c = np.hypot(*np.mgrid[-r_c:r_c + 1, -r_c:r_c + 1]) <= r_c
        # (its own outline kept near its broad parts: opened alone, the wings
        # came out round)
        collar = collar & ndimage.binary_dilation(ndimage.binary_opening(collar, structure=disk_c), structure=disk_c)
        labels[person & np.isin(seg, (BODY_SKIN, FACE_SKIN))] = NAMES.index("skin") + 1
    labels[person & (seg == HAIR)] = NAMES.index("hair") + 1
    labels[collar & (labels > 0) & (labels <= len(GARMENTS))] = NAMES.index("collar") + 1
    if face_mesh is not None:
        labels[shirt_whole(labels, rgb, face_mesh)] = NAMES.index("shirt") + 1
        collar_at_knot(labels, face_mesh)
        lap = lapels(labels, face_mesh, rgb)
        labels[lap] = NAMES.index("lapel") + 1
    total = float(person.sum())
    found = {}
    for k in WORN:
        i = NAMES.index(k)
        share = float((labels == i + 1).sum()) / total
        enough = _MIN_SHARE.get(k, _MIN_SHARE_DEFAULT)
        if k in ("shirt", "tshirt"):
            c_share = float((labels == NAMES.index("collar") + 1).sum()) / total
            share += c_share
            if k == "shirt" and c_share >= _MIN_SHARE["collar"]:
                enough = 0.0  # (a collar is a shirt's: with its collar taken out, what shows of a shirt is little)
        if share >= enough and share > 0:
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


def tie_whole(labels: np.ndarray, rgb: np.ndarray, face_mesh) -> np.ndarray:
    """The whole tie, from how a tie is made: one strip of one cloth from the
    knot down. Close up, CLIPSeg finds it where it is wide (under the knot);
    the knot it calls the shirt's collar, and where the tie narrows between
    the jacket's fronts it calls it the jacket. Grown from what was read
    through what is joined to it and has its colour (a pattern averaged
    out); nothing when no tie was read."""
    from scipy.ndimage import binary_propagation, gaussian_filter
    from skimage.color import rgb2lab

    tie = labels == NAMES.index("tie") + 1
    if tie.sum() < 50:
        return tie
    fm = np.asarray(face_mesh)[:468, :2]
    fw = float(np.ptp(fm[:, 0]))
    chin = float(np.asarray(face_mesh)[152, 1])
    L = rgb2lab(np.asarray(rgb)[..., :3] / 255.0)
    L = np.stack([gaussian_filter(L[..., i], _TIE_SMOOTH * fw) for i in range(3)], axis=-1)
    d = np.linalg.norm(L - np.median(L[tie], axis=0), axis=-1)
    reach = max(_TIE_MATCH * float(np.percentile(d[tie], 90)), _TIE_MATCH_MIN)
    rows = np.arange(labels.shape[0])[:, None]
    cols = np.arange(labels.shape[1])[None, :]
    mid = float(np.median(np.nonzero(tie)[1]))
    like = (d < reach) & (labels > 0) & (rows > chin) & (np.abs(cols - mid) < _TIE_REACH * fw)
    return binary_propagation(tie & like, mask=like) | tie


def shirt_whole(labels: np.ndarray, rgb: np.ndarray, face_mesh) -> np.ndarray:
    """The shirt showing in a jacket's opening, by its colour: between the
    tie and the jacket's fronts the shirt shows as a narrow strip, and read
    patch by patch it is called the jacket (the lapels were then laid over
    it, and it stood as a wall beside the tie). The jacket's pixels joined to
    the shirt, its collar or the tie, below the chin, that are much nearer
    the shirt's colour than the jacket's. Nothing when no shirt or no jacket
    is read."""
    from scipy.ndimage import binary_propagation, gaussian_filter
    from skimage.color import rgb2lab

    none = np.zeros(labels.shape, bool)
    sh = np.isin(labels, [NAMES.index(k) + 1 for k in ("shirt", "collar") if k in NAMES])
    jacket = np.isin(labels, [NAMES.index(k) + 1 for k in ("suit", "jacket", "coat") if k in NAMES])
    if sh.sum() < 50 or jacket.sum() < 200:
        return none
    fm = np.asarray(face_mesh)[:468, :2]
    fw = float(np.ptp(fm[:, 0]))
    cx = float(fm[:, 0].mean())
    chin = float(np.asarray(face_mesh)[152, 1])
    L = rgb2lab(np.asarray(rgb)[..., :3] / 255.0)
    L = np.stack([gaussian_filter(L[..., i], _SHIRT_SMOOTH * fw) for i in range(3)], axis=-1)
    d_s = np.linalg.norm(L - np.median(L[sh], axis=0), axis=-1)
    d_j = np.linalg.norm(L - np.median(L[jacket], axis=0), axis=-1)
    rows = np.arange(labels.shape[0])[:, None]
    cols = np.arange(labels.shape[1])[None, :]
    like = jacket & (d_s < _SHIRT_NEARER * d_j) & (rows > chin) & (np.abs(cols - cx) < _SHIRT_REACH * fw)
    seed = sh | (labels == NAMES.index("tie") + 1)
    return binary_propagation(seed, mask=like | seed) & like


def knot_foot(labels: np.ndarray, face_mesh) -> Optional[int]:
    """The photo row where a tie's knot ends: where the tie, read row by
    row, is narrowest under the knot's widest, within _KNOT_LOOK face widths
    of its top. None when no tie is read."""
    from scipy.ndimage import uniform_filter1d

    if "tie" not in NAMES:
        return None
    t = labels == NAMES.index("tie") + 1
    w = t.sum(axis=1).astype(np.float64)
    rows = np.flatnonzero(w > 0)
    if len(rows) < 10:
        return None
    fw = float(np.ptp(np.asarray(face_mesh)[:468, 0]))
    top = int(rows.min())
    look = w[top:top + max(int(_KNOT_LOOK * fw), 10)]
    look = uniform_filter1d(look, 5)
    # (the knot's widest: the first peak from the top -- the blade below it
    # is wider than the knot)
    peaks = [i for i in range(3, len(look) - 3) if look[i] >= look[i - 3:i + 4].max() and look[i] > look[i + 3]]
    if not peaks:
        return None
    k = peaks[0]
    return top + k + int(np.argmin(look[k:]))


def collar_at_knot(labels: np.ndarray, face_mesh) -> None:
    """A shirt's collar as the photo shows it when a tie is worn: its two
    points lie over the shirt either side of the knot and end at the knot's
    foot (how a collar sits round a tie), so the shirt the photo shows down
    to there is collar, and below it the shirt's front -- the knot's foot
    read off the tie in the photo. In place; nothing without a tie."""
    foot = knot_foot(labels, face_mesh)
    if foot is None or "collar" not in NAMES or "shirt" not in NAMES:
        return
    fw = float(np.ptp(np.asarray(face_mesh)[:468, 0]))
    chin = float(np.asarray(face_mesh)[152, 1])
    end = int(round(foot + _COLLAR_PAST_KNOT * fw))
    rows = np.arange(labels.shape[0])[:, None]
    ci, si = NAMES.index("collar") + 1, NAMES.index("shirt") + 1
    shirtish = np.isin(labels, (ci, si))
    labels[shirtish & (rows > chin) & (rows <= end)] = ci
    labels[shirtish & (rows > end)] = si


def _lapel_edges(labels, rgb, op, jacket, fw):
    """The lapels' outer edges in the photo: on each side of the opening, the
    straight line (notch to button) through the most points where the
    jacket's shading changes sharply, _LAPEL_SEEK face widths out. {"L"|"R":
    (a, b)} with x = a*y + b, for the sides found."""
    from scipy import ndimage

    g = ndimage.gaussian_filter(np.asarray(rgb, dtype=np.float64)[..., :3].mean(axis=2), 1.5)
    mag = np.hypot(ndimage.sobel(g, 1), ndimage.sobel(g, 0))
    jin = ndimage.binary_erosion(jacket, iterations=6)  # (not the jacket's outline against what is behind it)
    if not jin.any():
        return {}, mag, 0.0
    thr = float(np.percentile(mag[jin], _LAPEL_EDGE_PCT))
    H, W = labels.shape
    pts = {"L": [], "R": []}
    rr = np.flatnonzero(op.any(axis=1))
    a0, b0 = int(_LAPEL_SEEK[0] * fw), int(_LAPEL_SEEK[1] * fw)
    for y in rr:
        xs = np.flatnonzero(op[y])
        for sd, x0, sgn in (("L", xs.min(), -1), ("R", xs.max(), 1)):
            xx = x0 + sgn * np.arange(a0, b0)
            xx = xx[(xx >= 0) & (xx < W)]
            if len(xx) < 5 or jin[y, xx].sum() < 5:
                continue
            m = np.where(jin[y, xx], mag[y, xx], 0.0)
            k = int(np.argmax(m))
            if m[k] > thr:
                pts[sd].append((float(xx[k]), float(y)))
    out = {}
    rng = np.random.default_rng(0)
    for sd, p in pts.items():
        if len(p) < max(10, _LAPEL_SHARE * len(rr)):
            continue
        p = np.asarray(p)
        best = None
        for _ in range(400):
            i, j = rng.choice(len(p), 2, replace=False)
            if abs(p[i, 1] - p[j, 1]) < 0.1 * fw:
                continue
            a = (p[j, 0] - p[i, 0]) / (p[j, 1] - p[i, 1])
            # (it closes on the opening toward the button)
            if (sd == "L" and a <= 0.15) or (sd == "R" and a >= -0.15):
                continue
            inl = np.abs(p[:, 0] - (a * p[:, 1] + p[i, 0] - a * p[i, 1])) < _LAPEL_IN
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < _LAPEL_SHARE * len(rr):
            continue
        a, b = np.polyfit(p[best, 1], p[best, 0], 1)
        out[sd] = (float(a), float(b))
    return out, mag, thr


def lapels(labels: np.ndarray, face_mesh, rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """A jacket's lapels: they run along its opening (where the shirt, the
    tie or the collar shows between its two fronts) out to their own edge.
    That edge as the photo shows it (_lapel_edges) when it reads as a
    lapel's width; else, from how a jacket is made, _LAPEL_W face widths
    across at the top narrowing to the opening's foot. The opening's own
    two edges in the photo place them; nothing when no jacket or no opening
    is read."""
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
    edges, mag, thr = _lapel_edges(labels, rgb, op, jacket, fw) if rgb is not None else ({}, None, 0.0)
    out = np.zeros((H, W), bool)
    wmax = _LAPEL_W * fw
    widest = {"L": 0.0, "R": 0.0}
    seen = {"L": np.zeros((H, W), bool), "R": np.zeros((H, W), bool)}
    for y in range(y_t, y_b + 1):
        xs = np.flatnonzero(op[y])
        if not len(xs):
            continue
        xl, xr = int(xs.min()), int(xs.max())
        if "L" in edges:
            a_, b_ = edges["L"]
            e = int(round(a_ * y + b_))
            if e < xl:
                seen["L"][y, e:xl] = jacket[y, e:xl]
                widest["L"] = max(widest["L"], xl - e)
        if "R" in edges:
            a_, b_ = edges["R"]
            e = int(round(a_ * y + b_))
            if e > xr + 1:
                seen["R"][y, xr + 1:e + 1] = jacket[y, xr + 1:e + 1]
                widest["R"] = max(widest["R"], e - xr)
        w = wmax * (1.0 - (y - y_t) / max(y_b - y_t, 1))
        if w < 1.0:
            continue
        for sd, a, b in (("L", max(int(xl - w), 0), xl), ("R", xr + 1, min(int(xr + 1 + w), W))):
            if sd not in edges:
                out[y, a:b] |= jacket[y, a:b]
    # below the opening's foot (the shirt and tie end there) the lapel goes
    # on down its own two edges -- the opening's, carried on straight, and its
    # outer one -- to where the photo shows what crosses it (a sleeve's edge:
    # the jacket's shading changing sharply across the lapel) or they meet
    fit_n = max(int(0.3 * fw), 5)
    tail = [y for y in range(max(y_b - fit_n, y_t), y_b + 1) if op[y].any()]
    for sd in edges:
        if len(tail) < 5:
            break
        xs_e = [float(np.flatnonzero(op[y]).min() if sd == "L" else np.flatnonzero(op[y]).max()) for y in tail]
        ai, bi = np.polyfit(tail, xs_e, 1)
        a_, b_ = edges[sd]
        for y in range(y_b + 1, min(y_b + int(_KNOT_LOOK * fw), H)):
            xi, xo = ai * y + bi, a_ * y + b_
            lo, hi = (int(round(xo)), int(round(xi))) if sd == "L" else (int(round(xi)) + 1, int(round(xo)) + 1)
            lo, hi = max(lo, 0), min(hi, W)
            if hi - lo < 2 or not jacket[y, lo:hi].all() or float(np.mean(mag[y, lo:hi] > thr)) > 0.5:
                break
            seen[sd][y, lo:hi] = True
    for sd in ("L", "R"):
        if sd in edges:
            if _LAPEL_RANGE[0] * fw <= widest[sd] <= _LAPEL_RANGE[1] * fw:
                out |= seen[sd]
            else:
                # (an edge too near or too far for a lapel: not one -- as made)
                for y in range(y_t, y_b + 1):
                    xs = np.flatnonzero(op[y])
                    w = wmax * (1.0 - (y - y_t) / max(y_b - y_t, 1))
                    if not len(xs) or w < 1.0:
                        continue
                    if sd == "L":
                        a, b = max(int(xs.min() - w), 0), int(xs.min())
                    else:
                        a, b = int(xs.max()) + 1, min(int(xs.max() + 1 + w), W)
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
