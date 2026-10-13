"""Studies: the server teaching the builder on its own, with no one in the
loop until there is something to approve.

A study holds a few photos and some areas of knobs.py (hair, ears, neck...).
It builds every photo with the production combination, scores each model
with the critics below, then moves one knob at a time (up or down a step,
in the knob's own range), keeps a move when the photos' mean score rises,
and halves the step when a whole round finds nothing (a pattern search).
Every move, kept or not, is logged with its score. Nothing it finds is used
until it is approved (knobs.save_tuned).

The critics, per photo, all automatic:
- the Lab's own score (lab.score): front outline vs the photo's cutout,
  spikes, sealed, thinnest wall, face points found on the model vs the
  photo's
- the Lab's "held" parameters: widths, face proportions, the face's
  embedding match, head/chest/waist depth vs ANSUR II
- bumps: how far the surface stands off a smoothed copy of itself on the
  sides, back and top, where the photo can't have put detail -- the fins,
  ridges, bars, cracks and stray lumps the builder has made before, as
  numbers
- hair flow: the model's sculpted strands, seen from the front over the
  photo's hair, against the strand directions read from the photo
"""
import json
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from . import knobs, lab

CODE = lab.CODE_PRODUCTION
START_STEP = 0.25  # of a knob's range
MIN_STEP = 1 / 32
MIN_GAIN = 0.15  # a move is kept when the mean score rises by more than this
GIVE_UP = 4.0  # a move is dropped early once its photos so far trail by this much each
THUMB = (720, 480)

# objective = score + HELD_W x held mean - BUMP_W x bumps
HELD_W = 0.3
BUMP_W = 3.0
FLOW_W = 0.1
SEAM_W = 0.1  # cracks where hair meets what's beside it, grooves between the legs under a skirt (%)
SYM_W = 0.1  # left/right balance off the photo's (% points)
STAND_W = 10.0  # the figure's centre outside its base: it would tip over
OVERHANG_W = 0.05  # % of the surface facing steeply down
EAR_W = 0.2  # the ears' reach beside the head off the photo's, % of the face's width
NAPE_W = 1.0  # the back of the neck's bumps, %
FIT_W = 0.5  # the posed body's joints (and bone lengths) on the photo's: its miss, % of the shoulders' width
LAYER_W = 0.05  # overlaps kept: what lies over what in the photo is in front on the model (0..100)
OPEN_W = 20.0  # mouth and eyes as open as in the photo (lid / lip gap over width)
BUMP_MM = 0.4  # bumps: % of the surface standing off its smoothed copy by more than this
OBJECTIVE = (f"study score = Lab score + {HELD_W} × held mean − {BUMP_W} × bumps "
             f"(% of the sides, back and top standing more than {BUMP_MM} mm off a smoothed copy of themselves) "
             f"+ {FLOW_W} × hair flow (the model's strands along the photo's, −100…100) "
             f"− {OPEN_W} × (how much more or less open the mouth and eyes are than in the photo) "
             f"+ {LAYER_W} × overlaps kept (% of the photo's sure over/under boundaries the model has the right way round) "
             f"− {FIT_W} × the posed body's joint miss (% of shoulder width) "
             f"− {EAR_W} × the ears' reach off the photo's (% of face width) − {NAPE_W} × the nape's bumps (%) "
             f"− {STAND_W} if it wouldn't stand on its base − {OVERHANG_W} × overhang (%) "
             f"− {SEAM_W} × (cracks at hair's seams + grooves under a skirt, %) − {SYM_W} × left/right balance off the photo's")


# ---------------------------------------------------------------- critics


def bumps(preview: trimesh.Trimesh, floor: float) -> dict:
    """Surface standing off a Taubin-smoothed copy of itself (mm, along the
    normal), where the photo can't have put detail: the sides, back and top
    (normals turned more than ~70 degrees from the camera), above the base."""
    smooth = preview.copy()
    trimesh.smoothing.filter_taubin(smooth, lamb=0.5, nu=0.53, iterations=12)
    v, n = preview.vertices, preview.vertex_normals
    off = np.abs(np.einsum("ij,ij->i", v - smooth.vertices, n))
    keep = (v[:, 1] > floor + 1.5) & (n[:, 2] < 0.35)
    if not keep.any():
        return {"bumps": 0.0, "bump_p99_mm": 0.0}
    # per vertex, weighted by the area round it
    area = np.zeros(len(v))
    np.add.at(area, preview.faces.ravel(), np.repeat(preview.area_faces / 3.0, 3))
    a, o = area[keep], off[keep]
    return {"bumps": round(100.0 * float(a[o > BUMP_MM].sum() / a.sum()), 2),
            "bump_p99_mm": round(float(np.percentile(o, 99)), 2)}


def hair_flow(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs") -> dict:
    """How well the model's sculpted strands run with the photo's: the
    front of the model over the photo's hair, as a height map in the
    photo's own frame (scaled to the subject, centred, as the Lab aligns
    it), fine relief only, its ridges' direction against the photo's
    strand direction; 100 = all along, 0 = no pattern or none matching,
    -100 = all across."""
    from scipy.ndimage import gaussian_filter, zoom

    from .hair_pattern import _structure
    from .head_parts import HAIR

    reading = (inputs.parts or {}).get("hair") or {}
    if reading.get("kind") is None or "theta" not in reading:
        return {}
    bx0, by0, bx1, by1 = reading["box"]
    hair = inputs.parts["labels"][by0:by1, bx0:bx1] == HAIR
    mask = inputs.photo_mask
    ys, xs = np.nonzero(mask)
    my0, my1, mx0, mx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    v = preview.vertices
    top = float(v[:, 1].max())
    above = v[v[:, 1] > floor]
    mm = (top - floor) / (my1 - my0)  # model mm per photo px
    mx_c, px_c = (above[:, 0].min() + above[:, 0].max()) / 2, (mx0 + mx1) / 2
    hy, hx = np.nonzero(hair)
    if len(hy) < 100:
        return {}
    k = max(1.0, max(np.ptp(hy), np.ptp(hx)) / 360.0)  # photo px per raster px
    ry0, ry1, rx0, rx1 = by0 + hy.min(), by0 + hy.max() + 1, bx0 + hx.min(), bx0 + hx.max() + 1
    cen = preview.triangles_center
    yl, yh = top - (ry1 - my0) * mm, top - (ry0 - my0) * mm
    xl, xh = mx_c + (rx0 - px_c) * mm, mx_c + (rx1 - px_c) * mm
    sel = (cen[:, 1] > yl) & (cen[:, 1] < yh) & (cen[:, 0] > xl) & (cen[:, 0] < xh) & (preview.face_normals[:, 2] > 0.2)
    if sel.sum() < 50:
        return {}
    sub = trimesh.Trimesh(v, preview.faces[sel], process=False)
    hgt, wid = int(np.ceil((ry1 - ry0) / k)), int(np.ceil((rx1 - rx0) / k))
    pts, _ = trimesh.sample.sample_surface(sub, int(min(4 * hgt * wid, 1_500_000)), seed=0)
    r = ((top - pts[:, 1]) / mm + my0 - ry0) / k
    c = ((pts[:, 0] - mx_c) / mm + px_c - rx0) / k
    ok = (r >= 0) & (r < hgt) & (c >= 0) & (c < wid)
    z = np.full((hgt, wid), -np.inf)
    np.maximum.at(z, (r[ok].astype(int), c[ok].astype(int)), pts[ok, 2])
    have = np.isfinite(z)
    z = np.where(have, z, np.nanmin(np.where(have, z, np.nan)))
    z_hp = z - gaussian_filter(z, 4.0)  # the pattern, not the head's shape
    hair_r = zoom(hair[hy.min():hy.max() + 1, hx.min():hx.max() + 1].astype(np.float32), (hgt / (ry1 - ry0), wid / (rx1 - rx0)), order=0)[:hgt, :wid] > 0.5
    hair_r = hair_r[:hgt, :wid] & have[: hair_r.shape[0], : hair_r.shape[1]]
    if hair_r.sum() < 50:
        return {}
    th_m, coh_m, _ = _structure(z_hp.astype(np.float32), hair_r, 1.0, 3.0)
    th_p = reading["theta"][hy.min():hy.max() + 1, hx.min():hx.max() + 1]
    co_p = reading["coherence"][hy.min():hy.max() + 1, hx.min():hx.max() + 1]
    f = (hgt / th_p.shape[0], wid / th_p.shape[1])
    c2 = zoom(np.cos(2 * th_p), f, order=1)[:hgt, :wid]
    s2 = zoom(np.sin(2 * th_p), f, order=1)[:hgt, :wid]
    co = zoom(co_p, f, order=1)[:hgt, :wid]
    hh = hair_r[: c2.shape[0], : c2.shape[1]]
    th_p2 = 0.5 * np.arctan2(s2, c2)
    agree = co[hh] * coh_m[: c2.shape[0], : c2.shape[1]][hh] * np.cos(2 * (th_m[: c2.shape[0], : c2.shape[1]][hh] - th_p2[hh]))
    return {"hair_flow": round(100.0 * float(agree.sum() / (co[hh].sum() + 1e-9)), 1)}


def front_in_photo(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs", box, max_px: int = 420):
    """The model's front as heights over a box of the photo, in the photo's
    own frame (the model scaled to the subject's height and centred on it,
    as the Lab aligns them): (z mm, have, k photo px per raster px)."""
    mask = inputs.photo_mask
    ys, xs = np.nonzero(mask)
    my0, my1, mx0, mx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    v = preview.vertices
    top = float(v[:, 1].max())
    above = v[v[:, 1] > floor]
    mm = (top - floor) / (my1 - my0)
    mx_c, px_c = (above[:, 0].min() + above[:, 0].max()) / 2, (mx0 + mx1) / 2
    bx0, by0, bx1, by1 = box
    k = max(1.0, max(by1 - by0, bx1 - bx0) / max_px)
    hgt, wid = int(np.ceil((by1 - by0) / k)), int(np.ceil((bx1 - bx0) / k))
    cen = preview.triangles_center
    yl, yh = top - (by1 - my0) * mm, top - (by0 - my0) * mm
    xl, xh = mx_c + (bx0 - px_c) * mm, mx_c + (bx1 - px_c) * mm
    sel = (cen[:, 1] > yl) & (cen[:, 1] < yh) & (cen[:, 0] > xl) & (cen[:, 0] < xh) & (preview.face_normals[:, 2] > 0.05)
    z = np.full((hgt, wid), -np.inf)
    if sel.sum() < 20:
        return z, np.isfinite(z), k
    sub = trimesh.Trimesh(v, preview.faces[sel], process=False)
    pts, _ = trimesh.sample.sample_surface(sub, int(min(4 * hgt * wid, 1_500_000)), seed=0)
    r = ((top - pts[:, 1]) / mm + my0 - by0) / k
    c = ((pts[:, 0] - mx_c) / mm + px_c - bx0) / k
    ok = (r >= 0) & (r < hgt) & (c >= 0) & (c < wid)
    np.maximum.at(z, (r[ok].astype(int), c[ok].astype(int)), pts[ok, 2])
    return z, np.isfinite(z), k


# which lies over which, for the overlaps check -- only pairs whose order a
# front photo leaves in no doubt (hair beside a face can be in front of the
# cheek or behind the head; a hand is skin lying over clothes: left out;
# a gloved hand lies over the clothes it rests on, a glove's cuff over the
# bare wrist)
_ALL_WORN = ("shirt", "tshirt", "sweater", "jacket", "suit", "coat", "dress", "hoodie", "kimono", "skirt", "trousers", "jeans", "shorts")
_OUTER = {
    "hat": ("hair", "face", "forehead"), "glasses": ("face", "forehead"), "mask": ("face",),
    "jacket": ("shirt", "tshirt", "sweater", "tie", "dress", "trousers", "jeans", "skirt", "shorts"),
    "suit": ("shirt", "tshirt", "sweater", "tie", "dress", "trousers", "jeans", "skirt", "shorts"),
    "coat": ("shirt", "tshirt", "sweater", "tie", "dress", "jacket", "suit", "hoodie"),
    "tie": ("shirt", "tshirt"), "sweater": ("shirt", "tshirt"), "hoodie": ("shirt", "tshirt"),
    "scarf": ("shirt", "tshirt", "sweater", "jacket", "suit", "coat", "dress", "hoodie", "kimono"),
    "belt": ("trousers", "jeans", "skirt", "dress", "shorts", "kimono"),
    "collar": ("neck",),
    "hair": ("forehead",), "earring": ("hair", "neck", "face"),
    "hand": _ALL_WORN, "gloves": _ALL_WORN + ("wrist",), "watch": ("wrist",), "bracelet": ("wrist",),
    "necklace": ("neck", "shirt", "tshirt", "dress", "sweater", "kimono"),
    "badge": _ALL_WORN, "lanyard": _ALL_WORN, "backpack": _ALL_WORN, "strap": _ALL_WORN, "buttons": _ALL_WORN, "pocket": _ALL_WORN,
    "zip": _ALL_WORN, "shoes": ("socks",), "trousers": ("shoes", "socks"), "jeans": ("shoes", "socks"), "skirt": ("socks",),
}
for _sleeved in ("shirt", "tshirt", "jacket", "suit", "coat", "sweater", "hoodie", "kimono"):
    _OUTER[_sleeved] = tuple(_OUTER.get(_sleeved, ())) + ("wrist",)
LAYER_MIN_MM = 0.1


def _codes(inputs: "lab.LabInputs", H: int, W: int):
    """One name per photo pixel for the overlaps check: the head parts' hat,
    hair, face, glasses and mask; the garments; 'neck' for skin in the band
    under the chin, 'collar' for any garment in that band."""
    from .clothes import GARMENTS, NAMES
    from .head_parts import FACE_SKIN, GLASSES, HAIR, HAT, MASK

    names = ["", "hat", "hair", "face", "glasses", "mask", "neck", "collar", "forehead", "earring", "hand", "wrist"] + list(GARMENTS)
    code = np.zeros((H, W), np.uint8)
    parts = inputs.parts or {}
    cl = parts.get("clothes")
    if cl:
        for i, k in enumerate(NAMES):
            if k in GARMENTS:
                code[cl["labels"] == i + 1] = names.index(k)
        if inputs.face_mesh is not None:
            fm = np.asarray(inputs.face_mesh)
            chin, face_h = float(fm[152, 1]), float(np.ptp(fm[:468, 1]))
            band = (np.arange(H)[:, None] > chin) & (np.arange(H)[:, None] < chin + 0.6 * face_h)
            skin = cl["labels"] == NAMES.index("skin") + 1
            code[band & skin] = names.index("neck")
            code[band & (code >= names.index("wrist") + 1)] = names.index("collar")
        # hands and wrists (skin by the photo's own hand points): a hand lies
        # over the clothes under it; a sleeve's cuff over the wrist
        pose = inputs.pose or {}
        if "left_shoulder" in pose and "right_shoulder" in pose:
            sw = float(np.linalg.norm(np.subtract(pose["left_shoulder"], pose["right_shoulder"])))
            yy_, xx_ = np.mgrid[0:H, 0:W]
            skin = cl["labels"] == NAMES.index("skin") + 1
            for sd in ("left", "right"):
                if f"{sd}_wrist" in pose:
                    wx, wy = pose[f"{sd}_wrist"]
                    tips = [pose[f"{sd}_{k}"] for k in ("index", "pinky") if f"{sd}_{k}" in pose]
                    if tips:
                        hx, hy = np.mean(tips, axis=0)
                        code[skin & ((xx_ - hx) ** 2 + (yy_ - hy) ** 2 < (0.12 * sw) ** 2)] = names.index("hand")
                    near_w = skin & ((xx_ - wx) ** 2 + (yy_ - wy) ** 2 < (0.07 * sw) ** 2) & (code != names.index("hand"))
                    code[near_w] = names.index("wrist")
    if parts.get("labels") is not None:
        hl = parts["labels"]
        for lab_, nm in ((HAT, "hat"), (HAIR, "hair"), (FACE_SKIN, "face"), (GLASSES, "glasses"), (MASK, "mask")):
            code[hl == lab_] = names.index(nm)
        if inputs.face_mesh is not None:
            from .head_parts import ACCESSORY

            fm = np.asarray(inputs.face_mesh)[:, :2]
            yy, xx = np.mgrid[0:H, 0:W]
            brow = float((fm[105, 1] + fm[334, 1]) / 2)
            ruler = float(fm[2, 1] - brow)
            # the forehead: face skin above the brows between the temples (hair
            # surely lies over it at the hairline)
            fh = (hl == FACE_SKIN) & (yy < brow) & (xx > min(fm[54, 0], fm[284, 0])) & (xx < max(fm[54, 0], fm[284, 0]))
            code[fh] = names.index("forehead")
            # earrings: an accessory hanging from below the ears, beside the face
            fw = float(abs(fm[454, 0] - fm[234, 0]))
            side = (xx < fm[234, 0] + 0.1 * fw) | (xx > fm[454, 0] - 0.1 * fw)
            ear_rows = (yy > float(fm[2, 1]) - 0.5 * ruler) & (yy < float(fm[152, 1]) + 1.5 * ruler)
            code[(hl == ACCESSORY) & side & ear_rows] = names.index("earring")
    return code, names


def layering(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs") -> dict:
    """Where the photo shows one thing surely over another (a hat over hair,
    glasses or a mask on the face, a jacket over the shirt, a tie on it, a
    scarf over clothes, a belt round trousers, a collar round the neck): %
    of those boundary points where the model has the outer one in front."""
    from scipy.ndimage import zoom

    H, W = inputs.photo_mask.shape
    code, names = _codes(inputs, H, W)
    pairs = np.zeros((len(names), len(names)), bool)  # pairs[a, b]: a lies over b
    for a, inner in _OUTER.items():
        for b in inner:
            if a in names and b in names:
                pairs[names.index(a), names.index(b)] = True
    ys, xs = np.nonzero(inputs.photo_mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    z, have, k = front_in_photo(preview, floor, inputs, box)
    o = code[box[1]:box[3], box[0]:box[2]]
    o = zoom(o, (z.shape[0] / o.shape[0], z.shape[1] / o.shape[1]), order=0)[: z.shape[0], : z.shape[1]]
    hits, total = 0, 0
    h_, w_ = min(o.shape[0], z.shape[0]), min(o.shape[1], z.shape[1])
    o, z = o[:h_, :w_], z[:h_, :w_]
    for dy, dx in ((0, 2), (2, 0), (2, 2), (2, -2)):
        ya, yb = slice(0, h_ - dy), slice(dy, h_)
        xa, xb = (slice(0, w_ - dx), slice(dx, w_)) if dx >= 0 else (slice(-dx, w_), slice(0, w_ + dx))
        A, B, ZA, ZB = o[ya, xa], o[yb, xb], z[ya, xa], z[yb, xb]
        ok = np.isfinite(ZA) & np.isfinite(ZB)
        a_over = pairs[A, B] & ok
        b_over = pairs[B, A] & ok
        hits += int((a_over & (ZA - ZB >= LAYER_MIN_MM)).sum() + (b_over & (ZB - ZA >= LAYER_MIN_MM)).sum())
        total += int(a_over.sum() + b_over.sum())
    if total < 20:
        return {}
    return {"layering": round(100.0 * hits / total, 1), "layering_n": total}


def ears_fit(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs") -> dict:
    """The ears from the front: row by row between the brows and the nose's
    base, how far the model's outline reaches each side against the
    photo's, % of the face's width (0 = the same). Sides where the photo's
    ear is under hair are left out."""
    if inputs.face_mesh is None or inputs.reference is None:
        return {}
    fm = np.asarray(inputs.face_mesh)[:, :2]
    ears = inputs.reference.get("ears") or {}
    r0, r1 = int((fm[105, 1] + fm[334, 1]) / 2), int(fm[2, 1])
    fw = float(abs(fm[454, 0] - fm[234, 0]))
    cx = float((fm[234, 0] + fm[454, 0]) / 2)
    box = (int(cx - 1.5 * fw), r0, int(cx + 1.5 * fw), r1 + 1)
    z, have, k = front_in_photo(preview, floor, inputs, box)
    mask = inputs.photo_mask
    errs = []
    for side, sgn in (("R", -1), ("L", 1)):
        if (ears.get(side) or {}).get("hidden"):
            continue
        for ry in range(z.shape[0]):
            py = int(box[1] + ry * k)
            if not 0 <= py < mask.shape[0]:
                continue
            row = np.flatnonzero(mask[py, max(box[0], 0):min(box[2], mask.shape[1])])
            mrow = np.flatnonzero(have[ry])
            if not len(row) or not len(mrow):
                continue
            p_edge = (row.max() if sgn > 0 else row.min()) + max(box[0], 0)
            m_edge = box[0] + (mrow.max() if sgn > 0 else mrow.min()) * k
            errs.append(abs(p_edge - m_edge) / fw)
    if len(errs) < 4:
        return {}
    return {"ear_err": round(100.0 * float(np.mean(errs)), 1)}


def nape_bumps(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs") -> dict:
    """The back of the neck between the hairline and the collar, behind and
    below the ears: % of it standing more than BUMP_MM off a smoothed copy
    of itself (it should be smooth, with only its broad curves)."""
    if inputs.face_mesh is None:
        return {}
    fm = np.asarray(inputs.face_mesh)[:, :2]
    ys, xs = np.nonzero(inputs.photo_mask)
    my0, my1, mx0, mx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    v, n = preview.vertices, preview.vertex_normals
    top = float(v[:, 1].max())
    above = v[v[:, 1] > floor]
    mm = (top - floor) / (my1 - my0)
    mx_c, px_c = (above[:, 0].min() + above[:, 0].max()) / 2, (mx0 + mx1) / 2
    face_h = float(np.ptp(fm[:468, 1]))
    y_hi = top - (float(fm[2, 1]) - my0) * mm  # the nose's base: about the nape's hairline
    y_lo = top - (float(fm[152, 1]) + 0.6 * face_h - my0) * mm  # into the collar
    half = 0.6 * float(abs(fm[454, 0] - fm[234, 0])) * mm
    cx = mx_c + ((fm[234, 0] + fm[454, 0]) / 2 - px_c) * mm
    zone = (v[:, 1] < y_hi) & (v[:, 1] > y_lo) & (np.abs(v[:, 0] - cx) < half) & (n[:, 2] < -0.2)
    if zone.sum() < 30:
        return {}
    smooth = preview.copy()
    trimesh.smoothing.filter_taubin(smooth, lamb=0.5, nu=0.53, iterations=12)
    off = np.abs(np.einsum("ij,ij->i", v - smooth.vertices, n))[zone]
    return {"nape_bumps": round(100.0 * float((off > BUMP_MM).mean()), 2)}


SEAM_MM = 3.0  # a depth jump this big across a seam reads as a crack


def seams_and_skirt(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs") -> dict:
    """hair_seams: where hair meets the face, ears, neck or clothes -- whose
    order a front photo can't settle -- % of those boundary points with a
    depth jump over SEAM_MM (a crack or a cliff). skirt_gap: under a dress
    or skirt, between the knees' columns, % of it sunk SEAM_MM behind the
    skirt's own median at that height (a groove between the legs)."""
    from scipy.ndimage import binary_dilation, zoom

    from .clothes import NAMES

    H, W = inputs.photo_mask.shape
    code, names = _codes(inputs, H, W)
    ys, xs = np.nonzero(inputs.photo_mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    z, have, k = front_in_photo(preview, floor, inputs, box)
    o = code[box[1]:box[3], box[0]:box[2]]
    o = zoom(o, (z.shape[0] / o.shape[0], z.shape[1] / o.shape[1]), order=0)[: z.shape[0], : z.shape[1]]
    h_, w_ = min(o.shape[0], z.shape[0]), min(o.shape[1], z.shape[1])
    o, z = o[:h_, :w_], z[:h_, :w_]
    out = {}
    hair = o == names.index("hair")
    others = np.isin(o, [names.index(n) for n in ("face", "neck", "collar", "earring")] + [i for i, n in enumerate(names) if n in _ALL_WORN])
    edge_h = hair & binary_dilation(others, iterations=2)
    edge_o = others & binary_dilation(hair, iterations=2)
    if edge_h.sum() > 20 and edge_o.sum() > 20:
        from scipy.ndimage import maximum_filter, minimum_filter

        zf = np.where(np.isfinite(z), z, np.nan)
        hi = maximum_filter(np.nan_to_num(zf, nan=-1e9), size=5)
        lo = minimum_filter(np.nan_to_num(zf, nan=1e9), size=5)
        jump = (hi - lo)[edge_h & np.isfinite(zf)]
        if len(jump):
            out["hair_seams"] = round(100.0 * float((jump > SEAM_MM).mean()), 1)
    cl = (inputs.parts or {}).get("clothes")
    pose = inputs.pose or {}
    if cl and "left_knee" in (inputs.reference or {}).get("joints_px", {}):
        jp = inputs.reference["joints_px"]
        skirt = np.isin(o, [names.index(n) for n in ("skirt", "dress") if n in names])
        x_lo, x_hi = sorted([(jp["left_knee"][0] - box[0]) / k, (jp["right_knee"][0] - box[0]) / k])
        y_hip = (min(jp["left_hip"][1], jp["right_hip"][1]) - box[1]) / k
        cols = np.arange(w_)
        zone = skirt & (cols[None, :] > x_lo) & (cols[None, :] < x_hi) & (np.arange(h_)[:, None] > y_hip) & np.isfinite(z)
        if zone.sum() > 30:
            sunk = 0
            for r in np.flatnonzero(zone.any(axis=1)):
                row = z[r][skirt[r] & np.isfinite(z[r])]
                if len(row) > 3:
                    sunk += int((z[r][zone[r]] < np.median(row) - SEAM_MM).sum())
            out["skirt_gap"] = round(100.0 * sunk / zone.sum(), 1)
    _ = (NAMES, pose)
    return out


def symmetry(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs") -> dict:
    """Left and right: the model's front outline against its mirror image
    (IoU), next to the photo's own against its mirror; the gap between the
    two, % points (0 = as balanced, or as lopsided, as the photo)."""
    from scipy.ndimage import zoom

    ys, xs = np.nonzero(inputs.photo_mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    z, have, k = front_in_photo(preview, floor, inputs, box)
    m = inputs.photo_mask[box[1]:box[3], box[0]:box[2]]
    m = zoom(m.astype(np.uint8), (have.shape[0] / m.shape[0], have.shape[1] / m.shape[1]), order=0)[: have.shape[0], : have.shape[1]] > 0

    def mirror_iou(a):
        cols = np.flatnonzero(a.any(axis=0))
        if not len(cols):
            return None
        c = int(round((cols.min() + cols.max()) / 2))
        half = min(c, a.shape[1] - 1 - c)
        left, right = a[:, c - half:c], a[:, c + 1:c + 1 + half][:, ::-1]
        return 100.0 * float((left & right).sum()) / max(float((left | right).sum()), 1.0)

    a, b = mirror_iou(have), mirror_iou(m)
    if a is None or b is None:
        return {}
    return {"symmetry_diff": round(abs(a - b), 1)}


def printability(preview: trimesh.Trimesh, floor: float, inputs: "lab.LabInputs") -> dict:
    """Stands: the figure's centre (by surface area) over its base disk's
    inner 80%. Overhang: % of the figure's surface facing steeply down
    (a printer has to hold it up)."""
    v = preview.vertices
    cen, area, fn = preview.triangles_center, preview.area_faces, preview.face_normals
    body = cen[:, 1] > floor + 2.0
    base = v[v[:, 1] < floor - 0.5]
    out = {}
    if len(base) > 10 and body.any():
        bc = base[:, [0, 2]].mean(0)
        rad = float(np.percentile(np.linalg.norm(base[:, [0, 2]] - bc, axis=1), 98))
        com = (cen[body][:, [0, 2]] * area[body, None]).sum(0) / area[body].sum()
        out["stands"] = 1.0 if np.linalg.norm(com - bc) <= 0.8 * rad else 0.0
    if body.any():
        down = body & (fn[:, 1] < -0.6)
        out["overhang_pct"] = round(100.0 * float(area[down].sum() / area[body].sum()), 2)
    return out


def objective(m: dict) -> float:
    return float(m.get("score", -100.0) + HELD_W * m.get("held_mean", 0.0) - BUMP_W * m.get("bumps", 0.0)
                 + FLOW_W * m.get("hair_flow", 0.0)
                 - OPEN_W * (m.get("mouth_open_diff", 0.0) + m.get("eyes_open_diff", 0.0))
                 + LAYER_W * m.get("layering", 0.0)
                 - FIT_W * 100.0 * (m.get("fit_kp") or 0.0)
                 - EAR_W * m.get("ear_err", 0.0) - NAPE_W * m.get("nape_bumps", 0.0)
                 - STAND_W * (1.0 - m.get("stands", 1.0)) - OVERHANG_W * m.get("overhang_pct", 0.0)
                 - SEAM_W * (m.get("hair_seams", 0.0) + m.get("skirt_gap", 0.0)) - SYM_W * m.get("symmetry_diff", 0.0))


def evaluate(inputs: "lab.LabInputs", values: dict):
    """Build and score one photo with `values` applied. Hold the build lock."""
    from .body_reference import fit_body_reference

    with knobs.applied(values):
        if inputs.pose and inputs.animal_frac <= 0.15:
            fitted = fit_body_reference(inputs.pose, inputs.face_mesh, inputs.photo_mask, gender=inputs.body, parts=inputs.parts)
            inputs.reference = fitted if fitted is not None and fitted["matched"] else None
        body = lab.build(CODE, inputs)
        m, preview, thumb = lab.score(CODE, body, inputs, thumb_size=THUMB)
    lab._sface.pop(id(inputs), None)  # keyed by object: never let a later photo reuse it
    if inputs.reference is not None:
        m["fit_kp"] = inputs.reference["report"].get("kp_median")  # the posed body's miss of the photo's joints
    m.update(bumps(preview, _floor(preview)))
    for critic in (hair_flow, layering, ears_fit, nape_bumps, printability, seams_and_skirt, symmetry):
        try:
            m.update(critic(preview, _floor(preview), inputs))
        except Exception:  # noqa: BLE001 - a photo it can't apply to just doesn't score it
            pass
    m.pop("params", None)
    m["objective"] = round(objective(m), 2)
    return m, thumb


def _floor(mesh: trimesh.Trimesh) -> float:
    from .mesh_export import _BASE_DISK_MAX_THICKNESS_MM, _BASE_DISK_MIN_THICKNESS_MM, _BASE_DISK_THICKNESS_FRAC

    disk = float(np.clip(_BASE_DISK_THICKNESS_FRAC * lab.HEIGHT_MM, _BASE_DISK_MIN_THICKNESS_MM, _BASE_DISK_MAX_THICKNESS_MM))
    return float(mesh.vertices[:, 1].min()) + disk


# ---------------------------------------------------------------- photos


def prepare(photo_path: str, body: str, cache: Path) -> "lab.LabInputs":
    """The photo's inputs, computed once and kept in `cache` (a pickle of our
    own arrays; recomputed if missing, e.g. after a redeploy)."""
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:  # noqa: BLE001 - a broken cache is just recomputed
            pass
    inputs = lab.prepare_inputs(photo_path, body=body, capsule_builder=False, matting=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    return inputs


# ---------------------------------------------------------------- search


_CODE_VERSION = None


def code_version() -> str:
    """A hash of the builder's own code: a study's scores only compare
    within one version of it."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        import hashlib

        h = hashlib.sha1()
        for f in sorted(Path(__file__).parent.glob("*.py")):
            h.update(f.read_bytes())
        _CODE_VERSION = h.hexdigest()[:12]
    return _CODE_VERSION


def rebase_if_code_changed(st: dict) -> bool:
    """New code since the study's scores: build its start and its best
    again before any more moves (else a move would be judged against a
    score the old code made), dropping any move part-built by the old code.
    True if it queued that."""
    if st.get("code") == code_version():
        return False
    c = st.get("cand")
    if c is not None:
        # a move part-built by the old code: its photos so far can't be put
        # beside the new code's (one photo scored by a new rule would swing
        # the mean) -- dropped, and tried again once the start is rebuilt
        if c.get("change") is not None and st.get("kind") != "neighbours":
            st.setdefault("queue", []).insert(0, [c["change"][0], c["change"][3]])
        st["cand"] = None
    if st.get("best_obj") is None:
        st["code"] = code_version()  # nothing scored yet: the first build starts again
        return True
    if st.get("kind") == "neighbours":
        # a neighbours run compares moves against one start: begun again from scratch by the autopilot
        st["status"], st["stale"] = "done", True
        return True
    st["code"] = code_version()
    st["rebase"] = ["base", "best"]
    return True


def new_state(sid: str, photos: list, areas: list, max_evals: int, seed: int = 0, kind: str = "tune") -> dict:
    """kind "tune": the pattern search. kind "neighbours": every number moved
    once (up a quarter of its range) from the values in use, every score
    kept, nothing applied -- to see which moves reach other areas' scores."""
    keys = [k.key for k in knobs.KNOBS if k.area in areas]
    base = knobs.current()
    if kind == "neighbours":
        max_evals = len(keys) + 1
    return {"kind": kind,
        "sid": sid, "created": time.time(), "status": "running", "areas": areas, "photos": photos,
        "keys": keys, "base": {k: base[k] for k in keys}, "x": {k: base[k] for k in keys},
        "step": START_STEP, "round": 0, "evals": 0, "max_evals": max_evals, "seed": seed,
        "queue": [], "improved": False, "best_obj": None, "base_obj": None, "best_per_photo": {}, "base_per_photo": {},
        "cand": None, "kept": [], "code": code_version(),
    }


def _move(k: "knobs.Knob", v: float, step: float, sign: int):
    u = (float(v) - k.lo) / (k.hi - k.lo)
    return knobs.clip(k, k.lo + min(max(u + sign * step, 0.0), 1.0) * (k.hi - k.lo))


def next_candidate(st: dict) -> Optional[dict]:
    """The next move to try, or None when the study is finished."""
    if st["best_obj"] is None:
        return {"values": dict(st["x"]), "change": None, "results": {}}
    if st.get("kind") == "neighbours":
        if st.get("queue") is None or (not st["queue"] and st["evals"] <= 1 and not st.get("queued")):
            st["queue"], st["queued"] = [[k, 1] for k in st["keys"]], True
        while st["queue"]:
            key, sign = st["queue"].pop(0)
            k = knobs.BY_KEY[key]
            to = _move(k, st["base"][key], NEIGHBOUR_STEP, sign)
            if to == st["base"][key]:
                to = _move(k, st["base"][key], NEIGHBOUR_STEP, -sign)
            if to != st["base"][key]:
                values = dict(st["base"])
                values[key] = to
                return {"values": values, "change": [key, st["base"][key], to, sign], "results": {}}
        return None
    if st.get("rebase"):
        role = st["rebase"][0]
        values = {**st["x"], **st["base"]} if role == "base" else dict(st["x"])
        return {"values": values, "change": None, "role": role, "results": {}}
    while True:
        if st["evals"] >= st["max_evals"]:
            return None
        if not st["queue"]:
            if not st["improved"] and st["round"] > 0:
                st["step"] /= 2.0
                if st["step"] < MIN_STEP:
                    return None
            st["round"] += 1
            st["improved"] = False
            rng = random.Random(st["seed"] * 1000 + st["round"])
            moves = [[k, s] for k in st["keys"] for s in (1, -1)]
            rng.shuffle(moves)
            st["queue"] = moves
        key, sign = st["queue"].pop(0)
        k = knobs.BY_KEY[key]
        to = _move(k, st["x"][key], st["step"], sign)
        if to == st["x"][key]:
            continue  # at the end of its range
        values = dict(st["x"])
        values[key] = to
        return {"values": values, "change": [key, st["x"][key], to, sign], "results": {}}


def finish_candidate(st: dict) -> dict:
    """Decides the finished (or dropped) candidate; returns its log line."""
    c = st["cand"]
    per = {pid: r.get("objective", -100.0) for pid, r in c["results"].items()}
    complete = len(per) == len(st["photos"])
    mean = float(np.mean(list(per.values()))) if per else -100.0
    line = {"i": st["evals"], "at": time.time(), "change": c["change"], "step": st["step"], "per_photo": per,
            "objective": round(mean, 2), "complete": complete,
            "metrics": {pid: {k: r.get(k) for k in ("score", "held_mean", "bumps", "shape_iou", "face_err_pct", "spike_pct", "hair_flow",
                                                    "face_err_eyes", "face_err_brows", "face_err_nose", "face_err_mouth", "face_err_jaw",
                                                    "mouth_open_diff", "eyes_open_diff", "layering", "fit_kp", "ear_err", "nape_bumps", "stands",
                                                    "overhang_pct", "hair_seams", "skirt_gap", "symmetry_diff", "error")}
                        for pid, r in c["results"].items()}}
    if c.get("role"):
        # the same values built again with new code
        role = c["role"]
        st[f"{role}_obj"] = round(mean, 2)
        st[f"{role}_per_photo"] = dict(per)
        st["rebase"] = [r for r in st.get("rebase", []) if r != role]
        line["kept"] = f"rebuilt {role} (new code)"
        st["cand"] = None
        return line
    if st["best_obj"] is None:
        st["best_obj"] = st["base_obj"] = round(mean, 2)
        st["best_per_photo"] = dict(per)
        st["base_per_photo"] = dict(per)
        line["kept"] = "baseline"
        line["all"] = {pid: {k_: v_ for k_, v_ in r.items() if isinstance(v_, (int, float)) and not isinstance(v_, bool)}
                       for pid, r in c["results"].items()}
    elif st.get("kind") == "neighbours":
        line["kept"] = False
        line["all"] = {pid: {k_: v_ for k_, v_ in r.items() if isinstance(v_, (int, float)) and not isinstance(v_, bool)}
                       for pid, r in c["results"].items()}
    elif complete and mean > st["best_obj"] + MIN_GAIN:
        key, _, to, sign = c["change"]
        line["kept"] = True
        line["gain"] = round(mean - st["best_obj"], 2)
        st["x"][key] = to
        st["best_obj"] = round(mean, 2)
        st["best_per_photo"] = dict(per)
        st["improved"] = True
        st["queue"].insert(0, [key, sign])  # keep going the same way
        st["kept"].append({"i": st["evals"], "key": key, "from": c["change"][1], "to": to, "gain": line["gain"],
                           "objective": st["best_obj"]})
    else:
        line["kept"] = False
    st["evals"] += 1
    st["cand"] = None
    return line


def trails(st: dict) -> bool:
    """The candidate is already clearly worse on the photos built so far."""
    c = st["cand"]
    if st.get("kind") == "neighbours":
        return False  # every score of every move is wanted
    if c["change"] is None or not c["results"]:
        return False
    lost = [st["best_per_photo"].get(pid, 0.0) - r.get("objective", -100.0) for pid, r in c["results"].items()]
    return float(np.mean(lost)) > GIVE_UP


NEIGHBOUR_STEP = 0.25
# each score's own areas: a move from any other area that shifts it is a
# neighbour's effect
METRIC_HOME = {
    "face_err_pct": ["face", "face parts"], "face_err_eyes": ["face parts"], "face_err_brows": ["face parts"],
    "face_err_nose": ["face parts", "face"], "face_err_mouth": ["face parts"], "face_err_jaw": ["face"],
    "mouth_open_diff": ["face parts"], "eyes_open_diff": ["face parts"], "hair_flow": ["hair pattern"],
    "layering": ["clothes", "headwear", "neck"], "ear_err": ["ears"], "nape_bumps": ["neck", "hair"],
    "bumps": ["hair", "clothes"], "shape_iou": ["clothes"], "fit_kp": ["bones"], "held_mean": ["clothes", "bones"],
    "spike_pct": ["clothes", "hair"], "min_wall_mm": ["clothes"], "overhang_pct": ["clothes"], "stands": ["bones"],
    "hair_seams": ["hair"], "skirt_gap": ["clothes"], "symmetry_diff": ["bones", "clothes"],
}
CROSS_Z = 2.0  # a move shifting a score by this many of its usual shifts counts


def neighbour_effects(lines: list) -> dict:
    """From a neighbours study's log: for every number moved, how much each
    score shifted (mean over the photos), in units of that score's usual
    shift across all moves; the moves that reach other areas' scores, and
    area by area how strongly one reaches the next."""
    base = next((l for l in lines if l.get("kept") == "baseline"), None)
    moves = [l for l in lines if l.get("change") and l.get("all")]
    if base is None or not moves:
        return {}
    metrics = sorted({m for l in moves for r in l["all"].values() for m in r} & set(METRIC_HOME))
    delta = {}
    for l in moves:
        key = l["change"][0]
        d = {}
        for m in metrics:
            vals = [r[m] - base["all"][pid][m] for pid, r in l["all"].items()
                    if m in r and pid in base["all"] and m in base["all"][pid]]
            if vals:
                d[m] = float(np.mean(vals))
        delta[key] = d
    scale = {}
    for m in metrics:
        a = np.abs([d[m] for d in delta.values() if m in d])
        scale[m] = float(np.median(a[a > 0])) if (a > 0).any() else 0.0
    cross, area_mat = [], {}
    for key, d in delta.items():
        a_k = knobs.BY_KEY[key].area if key in knobs.BY_KEY else "?"
        for m, v in d.items():
            if not scale.get(m):
                continue
            z = v / scale[m]
            for home in METRIC_HOME[m]:
                if home != a_k:
                    cell = area_mat.setdefault(a_k, {})
                    cell[home] = max(cell.get(home, 0.0), round(abs(z), 1))
            if a_k not in METRIC_HOME[m] and abs(z) >= CROSS_Z:
                cross.append({"knob": key, "area": a_k, "score": m, "score_areas": METRIC_HOME[m], "shift": round(v, 3),
                              "z": round(z, 1)})
    cross.sort(key=lambda c: -abs(c["z"]))
    return {"cross": cross, "areas": area_mat, "moves": len(moves)}


def area_groups(effects: dict, areas: list) -> list:
    """Areas that reach each other's scores strongly, joined: they are tuned
    together, so a gain in one isn't paid for by its neighbour."""
    parent = {a: a for a in areas}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for a, row in (effects.get("areas") or {}).items():
        for b, z in row.items():
            if a in parent and b in parent and z >= 2 * CROSS_Z:
                parent[find(a)] = find(b)
    groups = {}
    for a in areas:
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda g: (-len(g), g))


def summary_values(st: dict) -> list:
    out = []
    for key in st["keys"]:
        k = knobs.BY_KEY[key]
        out.append({"key": key, "name": k.name, "area": k.area, "what": k.what, "lo": k.lo, "hi": k.hi,
                    "base": st["base"][key], "best": st["x"][key], "code_default": knobs.DEFAULTS[key]})
    return out


def dumps(st: dict) -> str:
    return json.dumps(st, default=float)
