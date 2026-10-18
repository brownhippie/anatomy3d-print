"""A human body posed to the photo, as a 3D reference for the outline builder.

The outline says how wide every part is, never how deep, and a front photo
says nothing about where things sit front to back: the head forward of the
torso, the back of the skull and the nape, the curve of the back, forearms
folded on the belly. A real body shape knows all of that. This poses one --
the Anny body (NAVER, Apache-2.0) built on MakeHuman/MPFB2 meshes (CC0),
exported by tools/extract_body_reference.py -- so its joints land on the
photo's pose and face points under an orthographic camera, inside the
person's outline, with forearms that lie over the torso resting on its
front. silhouette_relief.build_silhouette_relief_mesh(reference=...) then
uses its front and back as the base surfaces.

numpy/scipy only. The fit is checked before it is used: fit_body_reference
returns None when the joints miss their photo points or the body leaves the
outline, and the builder falls back to its outline-based shapes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates, median_filter

_DATA = Path(__file__).parent / "data" / "body_reference.npz"

# posed bones; the rest keep the rest pose (A-pose arms, straight legs)
_TRUNK = ["root", "spine05", "spine04", "spine03", "spine02", "spine01", "neck01", "neck02", "neck03", "head"]
_ARM = ["clavicle", "shoulder01", "upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist"]
_LEG = ["upperleg01", "lowerleg01"]
# rotation prior per bone (larger = stiffer)
_PRIOR = {"root": 2.0, "spine": 6.0, "neck": 1.0, "head": 0.5, "clavicle": 6.0, "shoulder01": 6.0,
          "upperarm01": 0.05, "upperarm02": 0.5, "lowerarm01": 0.05, "lowerarm02": 0.5, "wrist": 0.5,
          "upperleg01": 0.2, "lowerleg01": 0.2}
# build (weight, muscle, height, proportions) moves at most this far from
# average, in Anny's own units, and costs this much per unit
_BUILD_RANGE = 0.5
_BUILD_PRIOR = 0.02
# Bones whose length the fit reads off the photo's points (both sides
# together): name -> (the rig's bones in it, the joint it ends at; None:
# the whole of it scaled about its first joint). A stretch of e lengthens
# the bone by that share; everything below it moves along with it.
_STRETCH = {
    "humerus": (["upperarm01", "upperarm02"], "lowerarm01"),  # shoulder to elbow
    "radius_ulna": (["lowerarm01", "lowerarm02"], "wrist"),  # elbow to wrist
    "femur": (["upperleg01", "upperleg02"], "lowerleg01"),  # hip to knee
    "tibia_fibula": (["lowerleg01", "lowerleg02"], "foot"),  # knee to ankle
    "clavicle": (["clavicle"], "shoulder01"),  # the shoulders' width
    "cervical": (["neck01", "neck02", "neck03"], "head"),  # the neck's length (C1-C7)
    "hand": (["wrist"], None),  # carpals, metacarpals, phalanges: the hand's size
    "foot": (["foot"], None),  # tarsals, metatarsals, toes: the foot's size (heel and big toe in the photo)
}
_STRETCH_RANGE = 0.15
_STRETCH_PRIOR = 0.05
_HEAD_RANGE = 0.25  # head size, +/- share of the body model's
_HEAD_PRIOR = 0.2
_KP_WEIGHT = {"shoulder": 1.0, "elbow": 1.0, "wrist": 0.3, "hip": 0.7, "knee": 1.0, "ankle": 1.0}
_FOOT_WEIGHT = 0.5
# the fit is used only when it matches the photo
_MAX_KP_ERR = 0.08  # median joint/face point miss, x shoulder distance
_MIN_INSIDE = 0.95  # share of the body's visible points inside the outline, or within
_INSIDE_TOL = 0.02  # this many metres of it (the builder fits widths to the outline afterwards)
_INSIDE_WEIGHT = 6.0
_LEG_STRAIGHT = 3.0  # cost of a thigh or shin turning toward/away from the camera
_ARM_PLANE = 0.5  # same for an upper arm or a forearm that isn't folded
# posture the photo can't show must still be a person's: front to back,
# ankles within this of their hips, the neck within this of the pelvis (m)
_MAX_LEG_LEAN = 0.20
_MAX_TRUNK_LEAN = 0.15
# Ears, read off the photo's outline beside the head. Their ruler is the
# face's own: brow line to nose base, about one ear's length on most people.
_BROW = {"R": 105, "L": 334}  # face mesh: top of each eyebrow (subject's right, left)
_FACE_EDGE = {"R": 234, "L": 454}  # face mesh: the face's side, where the ear starts
_EAR_DROP = 0.06  # the lobe's end: the outline stepping in by this much (x brow-to-nose-base)
_EAR_KINK = 0.05  # the ear's top: the outline turning out by this much (else the brow line)
_EAR_STANDS = (0.08, 0.6)  # how far out of the head's line an ear can stand
_EAR_LENGTH = (0.75, 1.3)  # how long an ear can be
_EAR_TURN = (-0.35, 0.8)  # radians an ear may swing in toward / out from the head
_EAR_SCALE = (0.8, 1.5)
_EAR_UNDER_HAIR = 0.6  # an ear's place beside the face this much hair (or hat): under it
# head_parts.py's labels (not imported: it pulls in the segmenters)
_HP_HAIR, _HP_SKIN, _HP_FACE, _HP_HAT = 1, 2, 3, 6
_EAR_BEND = 0.015  # the swing reaches all of it this far (metres) out from the ear's root
_MAX_FOREARM_LIFT = (0.10, 0.18)  # metres folded forearms/hands stand off the torso's front: median, most
_REST_MAX = 0.10  # fit: a folded forearm's surface stays within this of the torso's front


@lru_cache(maxsize=1)
def _data() -> dict:
    with np.load(_DATA) as d:
        return {k: d[k] for k in d.files}


def _body(gender: str) -> dict:
    d = _data()
    skin_w = d["skin_w"].astype(np.float64)
    part_w = np.zeros((len(skin_w), len(d["part_names"])))
    bone_part = d["bone_part"].astype(int)
    for j in range(skin_w.shape[1]):
        np.add.at(part_w, (np.arange(len(skin_w)), bone_part[d["skin_idx"][:, j]]), skin_w[:, j])
    part_w /= part_w.sum(1, keepdims=True) + 1e-12
    labels = [str(l) for l in d["bone_labels"]]
    parents = d["parents"].astype(int)
    head = labels.index("head")
    under = np.zeros(len(labels), dtype=bool)
    for j, p in enumerate(parents):
        under[j] = p >= 0 and (p == head or under[p])
    joints = d[f"{gender}_joints"].astype(np.float64)
    verts = d[f"{gender}_vertices"].astype(np.float64)
    sdv, sdj = _stretch_bases(labels, parents, joints, verts, d["skin_idx"].astype(np.int64), skin_w)
    return {"labels": labels, "parents": parents, "head_bone": head, "head_desc": np.flatnonzero(under),
            "stretch_dv": sdv, "stretch_dj": sdj,
            "head_w": part_w[:, [str(n) for n in d["part_names"]].index("head")],
            "verts": d[f"{gender}_vertices"].astype(np.float64), "joints": d[f"{gender}_joints"].astype(np.float64),
            "faces": d["faces"].astype(np.int64), "skin_idx": d["skin_idx"].astype(np.int64), "skin_w": skin_w,
            "kp_names": [str(k) for k in d["kp_names"]], "kp_idx": d["kp_idx"].astype(np.int64),
            "kp_w": d["kp_w"].astype(np.float64), "part_w": part_w, "part_names": [str(p) for p in d["part_names"]],
            "dverts": d[f"{gender}_dvertices"].astype(np.float64), "djoints": d[f"{gender}_djoints"].astype(np.float64),
            "ear_w": d["ear_w"].astype(np.float64)}


def _stretch_bases(labels, parents, joints, verts, skin_idx, skin_w):
    """Per _STRETCH entry: how every vertex and joint moves per unit of it,
    in the rest pose (vertices on the bone slide by how far along it they
    lie; everything below it by the whole lengthening)."""
    nb, nv = len(labels), len(verts)
    below = np.zeros((nb, nb), dtype=bool)  # below[a, b]: b hangs under a
    for b in range(nb):
        p = parents[b]
        while p >= 0:
            below[p, b] = True
            p = parents[p]
    W = np.zeros((nv, nb))
    np.add.at(W, (np.repeat(np.arange(nv), skin_idx.shape[1]), skin_idx.ravel()), skin_w.ravel())
    dv = np.zeros((len(_STRETCH), nv, 3))
    dj = np.zeros((len(_STRETCH), nb, 3))
    for i, (bones, tail) in enumerate(_STRETCH.values()):
        sides = ("L", "R") if f"{bones[0]}.L" in labels else ("",)
        for sd in sides:
            nm = (lambda b: f"{b}.{sd}") if sd else (lambda b: b)  # noqa: E731
            seg = [labels.index(nm(b)) for b in bones]
            h = joints[seg[0]]
            under = [b for b in np.flatnonzero(below[seg[0]]) if b not in seg]
            w_under = W[:, under].sum(1) if under else np.zeros(nv)
            if tail is None:
                w_all = W[:, seg].sum(1) + w_under
                dv[i] += w_all[:, None] * (verts - h)
                for b in seg[1:] + under:
                    dj[i, b] += joints[b] - h
                continue
            ax = joints[labels.index(nm(tail))] - h
            length = float(np.linalg.norm(ax))
            along = np.clip((verts - h) @ ax / length ** 2, 0.0, 1.0)
            dv[i] += (w_under + W[:, seg].sum(1) * along)[:, None] * ax
            for b in under:
                dj[i, b] += ax
            for b in seg[1:]:
                dj[i, b] += ax * float(np.clip((joints[b] - h) @ ax / length ** 2, 0.0, 1.0))
    return dv, dj


def _rest(body: dict, ph: np.ndarray, idx: Optional[np.ndarray] = None):
    """Rest-pose vertices (all, or `idx`) and joints for a build: Anny's
    weight/muscle/height/proportions changes, then the head scaled about its
    joint (heads vary against bodies more than one body model covers)."""
    nbuild = len(body["dverts"])
    dv = body["dverts"] if idx is None else body["dverts"][:, idx]
    v = (body["verts"] if idx is None else body["verts"][idx]) + np.einsum("p,pvi->vi", ph[:nbuild], dv)
    j = body["joints"] + np.einsum("p,pji->ji", ph[:nbuild], body["djoints"])
    k = ph[nbuild]
    c = j[body["head_bone"]]
    hw = body["head_w"] if idx is None else body["head_w"][idx]
    v = v + (k * hw)[:, None] * (v - c)
    j = j.copy()
    j[body["head_desc"]] = c + (1.0 + k) * (j[body["head_desc"]] - c)
    e = ph[nbuild + 1:nbuild + 1 + len(body["stretch_dv"])]
    if len(e) and np.any(e):
        sdv = body["stretch_dv"] if idx is None else body["stretch_dv"][:, idx]
        v = v + np.einsum("s,svi->vi", e, sdv)
        j = j + np.einsum("s,sji->ji", e, body["stretch_dj"])
    return v, j


def _rotmats(rv: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(rv, axis=1)[:, None, None]
    k = rv / np.maximum(th[:, :, 0], 1e-12)
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    return np.eye(3)[None] + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _bone_transforms(body: dict, bone_ids: np.ndarray, rv: np.ndarray):
    """World rotation and position of every joint. Each posed bone turns
    about the rest pose's own axes (x: the body's left, y: back, z: up), in
    its parent's turned frame."""
    parents, joints = body["parents"], body["joints"]
    local = np.tile(np.eye(3), (len(parents), 1, 1))
    local[bone_ids] = _rotmats(rv)
    Rw = np.empty_like(local)
    pw = np.empty_like(joints)
    for j, p in enumerate(parents):
        if p < 0:
            Rw[j], pw[j] = local[j], joints[j]
        else:
            Rw[j] = Rw[p] @ local[j]
            pw[j] = pw[p] + Rw[p] @ (joints[j] - joints[p])
    return Rw, pw


def _skin(body: dict, Rw, pw, idx: Optional[np.ndarray] = None) -> np.ndarray:
    v = body["verts"] if idx is None else body["verts"][idx]
    si = body["skin_idx"] if idx is None else body["skin_idx"][idx]
    sw = body["skin_w"] if idx is None else body["skin_w"][idx]
    rel = v[:, None, :] - body["joints"][si]
    moved = np.einsum("vkij,vkj->vki", Rw[si], rel) + pw[si]
    return (sw[:, :, None] * moved).sum(1)


def _photo_targets(pose: dict, face_mesh: Optional[np.ndarray]):
    names, xy, w = [], [], []
    for k, v in pose.items():
        kind = k.split("_", 1)[-1]
        if kind in _KP_WEIGHT:
            names.append(k)
            xy.append(v)
            w.append(_KP_WEIGHT[kind])
        elif kind in ("heel", "foot_index"):
            # the feet: the body knows them as heel and big toe
            names.append(k.replace("foot_index", "big_toe"))
            xy.append(v)
            w.append(_FOOT_WEIGHT)
    if face_mesh is not None:
        f = np.asarray(face_mesh)[:, :2]
        for n, p, wt in (("nose", f[1], 1.0), ("right_eye", (f[33] + f[133]) / 2, 1.0), ("left_eye", (f[362] + f[263]) / 2, 1.0),
                         ("right_ear", f[234], 0.6), ("left_ear", f[454], 0.6)):
            names.append(n)
            xy.append(p)
            w.append(wt)
    else:
        # no face landmarks (a profile, a small face): the pose detector's
        # own nose, eyes and ears say where the head points
        for n, wt in (("nose", 1.0), ("left_eye", 0.6), ("right_eye", 0.6), ("left_ear", 0.6), ("right_ear", 0.6)):
            if n in pose:
                names.append(n)
                xy.append(pose[n])
                w.append(wt)
    return names, np.asarray(xy, dtype=np.float64), np.asarray(w)


def measure_ears(photo_mask: np.ndarray, face_mesh: Optional[np.ndarray], px_per_m: float, parts: Optional[dict] = None):
    """Each ear as the photo's outline shows it: where it starts (the outline
    turning out of the hair or temple, or the brow line when it doesn't
    show), where the lobe ends (the outline stepping in to the jaw), and how
    far it stands out of the head's own line at its widest. Keyed by the
    subject's side ("L" is on the photo's right).

    With the head's parts (head_parts.py): an ear whose place beside the face
    is mostly hair or a hat is under it ({"hidden": True}: not built); a
    bump in the outline counts only if it's skin (a curl of hair or an
    earring isn't an ear); and an ear that shows but can't be measured runs,
    as most people's do, from the brow line to the nose base. With one ear
    measured, the other gets its size. Without parts, sides not measured are
    left out (the body model's own ear). Returns (ears, report)."""
    if face_mesh is None:
        return {}, {"source": "none"}
    f = np.asarray(face_mesh, dtype=np.float64)[:, :2]
    H, W = photo_mask.shape
    brow_mid = (f[_BROW["R"], 1] + f[_BROW["L"], 1]) / 2.0
    ruler = float(f[2, 1] - brow_mid)  # brow line to nose base (subnasale)
    if ruler < 6:
        return {}, {"source": "none"}
    k = max(2, int(round(0.05 * ruler)))
    xc = int(round((f[_FACE_EDGE["R"], 0] + f[_FACE_EDGE["L"], 0]) / 2.0))  # the head's middle
    labels = parts.get("labels") if parts else None
    seen, hidden, place = {}, set(), {}
    for side, out in (("R", -1), ("L", 1)):
        brow = float(f[_BROW[side], 1])
        base = float(f[2, 1]) + (brow - brow_mid)  # nose base, tilted with the brows
        x0 = float(f[_FACE_EDGE[side], 0])
        place[side] = (brow, base)
        if labels is not None:
            # what lies beside the face, brow line to nose base, out to the outline
            y_lo, y_hi = int(max(brow, 0)), int(min(base, H - 1))
            xs = np.arange(int(x0), int(x0 + out * 0.6 * ruler), out)
            xs = xs[(xs >= 0) & (xs < W)]
            if y_hi > y_lo and len(xs):
                zone = labels[y_lo:y_hi + 1][:, xs][photo_mask[y_lo:y_hi + 1][:, xs]]
                if len(zone) >= (0.1 * ruler) ** 2:
                    hair = np.isin(zone, (_HP_HAIR, _HP_HAT)).mean()
                    if hair >= _EAR_UNDER_HAIR:
                        hidden.add(side)
                        continue
        rows = np.arange(max(int(brow - 0.6 * ruler), k), min(int(base + 0.8 * ruler), H - k))
        if len(rows) < 4 * k or not 0 <= xc < W:
            continue
        # how far out of the face's side the outline reaches, row by row
        # (walking out from the head's middle; negative where it's inside)
        e = np.full(len(rows), np.nan)
        for i, y in enumerate(rows):
            line = photo_mask[y, xc:] if out > 0 else photo_mask[y, : xc + 1][::-1]
            if line[0]:
                gap = np.flatnonzero(~line)
                e[i] = (gap[0] if len(gap) else len(line)) - out * (x0 - xc)
        if np.isnan(e).any():
            continue
        e = median_filter(e, size=3, mode="nearest")
        ix = lambda y: int(np.clip(round(y - rows[0]), k, len(rows) - k - 1))  # noqa: E731
        # the lobe: the sharpest step in near the nose base
        lo, hi = ix(base - 0.35 * ruler), ix(base + 0.6 * ruler)
        drop = e[lo:hi] - e[lo + k:hi + k]
        if not len(drop) or drop.max() < _EAR_DROP * ruler:
            continue
        b = lo + int(np.argmax(drop))
        b += int(np.argmax(e[b:b + k] - e[b + 1:b + k + 1]))  # the row it steps in on
        # the top: the outline turning out near the brow line
        lo, hi = ix(brow - 0.5 * ruler), ix(brow + 0.35 * ruler)
        kink = e[lo + k:hi + k] - 2 * e[lo:hi] + e[lo - k:hi - k]
        t = lo + int(np.argmax(kink)) if len(kink) and kink.max() >= _EAR_KINK * ruler else ix(brow)
        if not _EAR_LENGTH[0] * ruler <= b - t <= _EAR_LENGTH[1] * ruler:
            continue
        # how far it stands out of the head's line: the outline against the
        # straight line from where it starts to the jaw below the lobe
        j = min(b + k, len(rows) - 1)
        line = e[t] + (e[j] - e[t]) * (np.arange(t, b + 1) - t) / max(j - t, 1)
        stands = e[t:b + 1] - line
        if not _EAR_STANDS[0] * ruler <= stands.max() <= _EAR_STANDS[1] * ruler:
            continue
        # where it reaches furthest out (below its top edge: a cap's brim or
        # hair there isn't the ear)
        o0 = t + int(round(0.15 * (b - t)))
        o = o0 + int(np.argmax(e[o0:b + 1]))
        if labels is not None:
            # the bump itself has to be skin
            bump = []
            for i in range(t, b + 1):
                y = int(rows[i])
                lo_x, hi_x = sorted((x0 + out * (line[i - t] + 1), x0 + out * e[i]))
                xs = np.arange(int(np.ceil(lo_x)), int(hi_x) + 1)
                xs = xs[(xs >= 0) & (xs < W)]
                bump.append(labels[y, xs])
            bump = np.concatenate(bump) if bump else np.zeros(0, dtype=np.uint8)
            if len(bump) and np.isin(bump, (_HP_FACE, _HP_SKIN)).mean() < 0.5:
                continue
        seen[side] = {"top": float(rows[t]), "bottom": float(rows[b]), "outer": [x0 + out * float(e[o]), float(rows[o])],
                      "stands_px": float(stands.max()), "top_from": "outline" if t != ix(brow) else "brow line"}
    ears = {sd: dict(m, source="measured") for sd, m in seen.items()}
    if len(seen) == 1:
        # one ear measured: the other is its size, level with its own brow
        (a, m), = seen.items()
        other = "L" if a == "R" else "R"
        if other not in hidden:
            dy = float(f[_BROW[other], 1] - f[_BROW[a], 1])
            ears[other] = {"top": m["top"] + dy, "bottom": m["bottom"] + dy, "outer": None, "source": "the other ear"}
    elif not seen and labels is not None:
        # not measured but showing: brow line to nose base
        for sd in "LR":
            if sd not in hidden:
                ears[sd] = {"top": place[sd][0], "bottom": place[sd][1], "outer": None, "source": "face"}
    for sd in hidden:
        ears[sd] = {"hidden": True, "source": "under hair"}
    report = {"source": {2: "both", 1: "left" if "L" in seen else "right", 0: "none"}[len(seen)],
              "sides": {sd: ears[sd]["source"] for sd in sorted(ears)}}
    if seen:
        length = float(np.mean([m["bottom"] - m["top"] for m in seen.values()]))
        report.update({"length_cm": round(100 * length / px_per_m, 1),
                       "stands_cm": {sd: round(100 * m["stands_px"] / px_per_m, 1) for sd, m in seen.items()},
                       "length_to_brow_nose": round(length / ruler, 2)})
        if len(seen) == 2:
            span = abs(seen["L"]["outer"][0] - seen["R"]["outer"][0])
            report["length_to_ear_span"] = round(length / span, 2)
    return ears, report


def place_ears(V: np.ndarray, ear_w: np.ndarray, head_axes: np.ndarray, ears: dict, faces: Optional[np.ndarray] = None,
               px_per_m: float = 1000.0, info: Optional[dict] = None) -> np.ndarray:
    """Each ear of the posed body (photo px) sized, moved up or down and
    swung out from the head about its front edge until it starts, ends and
    reaches as far out as the photo's (measure_ears). Turned in the head's
    own frame; the skin around the ear follows partway. The swing grows
    from nothing where the ear grows out of the head to all of it
    _EAR_BEND (metres) out along the ear (given the faces): the root stays
    on the skull, as a real ear's does."""
    V = np.asarray(V, dtype=np.float64)
    A = np.asarray(head_axes, dtype=np.float64)  # columns: the head's left, back, up (photo px frame)
    out = V.copy()
    for i, side in enumerate("LR"):
        t = ears.get(side)
        w = ear_w[:, i]
        core = w > 0.95  # the ear itself; below that, skin around its base
        if not t or t.get("hidden") or core.sum() < 10:
            continue
        u = 1.0 if side == "L" else -1.0  # outward, in the head's frame
        loc = V @ A
        lc = loc[core]
        hinge = np.array([u * (u * lc[:, 0]).min(), lc[:, 1].min(), (lc[:, 2].min() + lc[:, 2].max()) / 2.0])
        rel = loc - hinge
        bend = np.ones(len(V))
        if faces is not None:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import dijkstra

            F = np.asarray(faces, dtype=np.int64)
            e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
            e = e[core[e].all(axis=1)]
            g = coo_matrix((np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
            # the root: ear vertices with a neighbour outside the ear
            ee = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
            cross = core[ee[:, 0]] != core[ee[:, 1]]
            root = np.unique(ee[cross][core[ee[cross]]])
            if len(root):
                d = dijkstra(g, directed=False, indices=root, min_only=True)
                k = np.clip(np.where(np.isfinite(d), d, 0.0) / (_EAR_BEND * px_per_m), 0.0, 1.0)
                bend = np.where(core, k * k * (3.0 - 2.0 * k), 0.0)
        # outward on the photo: only measurable when the head faces it
        u_img = float(np.sign(u * A[0, 0]))
        fit_turn = t.get("outer") is not None and abs(A[0, 0]) > 0.5

        def moved(p):
            kk, dz, phi = np.exp(p[0]), p[1], p[2] * bend
            r = kk * rel
            a = r[:, 0] * np.cos(phi) + u * r[:, 1] * np.sin(phi)
            bk = -u * r[:, 0] * np.sin(phi) + r[:, 1] * np.cos(phi)
            new = (np.column_stack([a, bk, r[:, 2] + dz]) + hinge) @ A.T
            return V + w[:, None] * (new - V)

        def resid(p):
            q = moved(p)[core]
            r = [q[:, 1].min() - t["top"], q[:, 1].max() - t["bottom"]]
            if fit_turn:
                r.append(u_img * (q[:, 0] * u_img).max() - t["outer"][0])
            return np.array(r)

        height = float(np.ptp(lc[:, 2]))
        lo = np.array([np.log(_EAR_SCALE[0]), -0.5 * height, _EAR_TURN[0]])
        hi = np.array([np.log(_EAR_SCALE[1]), 0.5 * height, _EAR_TURN[1]])
        p = np.zeros(3)
        free = [0, 1, 2] if fit_turn else [0, 1]
        for _ in range(12):
            r = resid(p)
            J = np.zeros((len(r), 3))
            for j in free:
                e = np.zeros(3)
                e[j] = 1e-3 * (1.0 if j != 1 else height)
                J[:, j] = (resid(p + e) - r) / e[j]
            Jf = J[:, free]
            dp = -np.linalg.lstsq(Jf.T @ Jf + 1e-6 * np.eye(len(free)), Jf.T @ r, rcond=None)[0]
            p[free] = np.clip(p[free] + dp, lo[free], hi[free])
            if np.abs(dp).max() < 1e-4:
                break
        if info is not None:
            info[side] = {"scale": round(float(np.exp(p[0])), 3), "shift_px": round(float(p[1]), 1),
                          "turn_deg": round(float(np.degrees(p[2])), 1), "miss_px": np.round(resid(p), 1).tolist()}
        out = out + (moved(p) - V)
    return out


_NECK_WEIGHT = 3.0  # the bare neck's breadth, against the keypoints' (1 each)


def read_skin(parts: Optional[dict], face_mesh: Optional[np.ndarray], pose: Optional[dict], photo_mask: np.ndarray) -> dict:
    """The bare body the photo shows, measured before the body is posed, so
    the body's build comes from skin rather than the clothes over it: the
    neck's breadth (px) where its skin shows between the chin and the
    collar, row by row from the face's middle out, the median of rows that
    look like a neck (between 0.45 and 1.3 of the face's breadth). {} when
    there is no skin to measure."""
    out = {}
    cl = (parts or {}).get("clothes")
    if not cl or face_mesh is None:
        return out
    from .clothes import NAMES

    if "skin" not in NAMES:
        return out
    skin = (cl["labels"] == NAMES.index("skin") + 1) & photo_mask
    f = np.asarray(face_mesh)[:468, :2]
    fw, fh = float(np.ptp(f[:, 0])), float(np.ptp(f[:, 1]))
    chin = float(np.asarray(face_mesh)[152, 1])
    cx = int(round(float(f[:, 0].mean())))
    H, W = skin.shape
    widths = []
    for r in range(int(chin + 0.12 * fh), int(min(chin + 0.6 * fh, H - 1))):
        row = skin[r]
        if not (0 <= cx < W) or not row[cx]:
            continue
        a = cx
        while a > 0 and row[a - 1]:
            a -= 1
        b = cx
        while b < W - 1 and row[b + 1]:
            b += 1
        w = b - a + 1
        if 0.45 * fw <= w <= 1.3 * fw:
            widths.append(w)
    if len(widths) >= 4:
        out["neck_px"] = float(np.median(widths))
        out["neck_rows"] = len(widths)
        out["neck_over_face"] = round(out["neck_px"] / fw, 3)
    return out


def fit_body_reference(pose: Optional[dict], face_mesh: Optional[np.ndarray], photo_mask: np.ndarray,
                       gender: str = "male", parts: Optional[dict] = None) -> Optional[dict]:
    """The body posed onto the photo, in photo pixels: x right, y down, depth
    toward the back, with a report of how well it matches the photo
    ("matched" False: don't use it). None when there is not enough to pose
    it. `parts`: head_parts.detect_head_parts of the same photo (hair, skin,
    hat, mask, glasses round the head), carried on for the builder."""
    if not pose or "left_shoulder" not in pose or "right_shoulder" not in pose:
        return None
    body = _body(gender)
    labels = body["labels"]
    P = {k: np.asarray(v, dtype=np.float64) for k, v in pose.items()}
    sw_px = float(np.linalg.norm(P["left_shoulder"] - P["right_shoulder"]))
    if sw_px < 8:
        return None
    H, W = photo_mask.shape
    names, tgt, wts = _photo_targets(pose, face_mesh)
    kp_rows = [body["kp_names"].index(n) for n in names]
    kp_idx, kp_w = body["kp_idx"][kp_rows], body["kp_w"][kp_rows]

    legs = any(k.endswith(("knee", "ankle")) for k in pose)
    bones = list(_TRUNK) + [f"{b}.{s}" for s in "LR" for b in _ARM] + ([f"{b}.{s}" for s in "LR" for b in _LEG] if legs else [])
    bone_ids = np.array([labels.index(b) for b in bones])
    prior = np.array([next(v for k, v in _PRIOR.items() if b.startswith(k)) for b in bones])
    # the spine stays upright and facing the camera; the neck and head turn
    # and tilt as the face points say (heads turn all the time)
    pitch = np.array([i for i, b in enumerate(bones) if b == "root" or b.startswith("spine")])
    leg_i = np.array([i for i, b in enumerate(bones) if b.startswith(("upperleg", "lowerleg"))], dtype=int)

    # vertices the fit looks at: keypoint supports, hands, a spread of the rest
    rng = np.random.default_rng(0)
    nv = len(body["verts"])
    dom_part = body["part_w"].argmax(1)
    pn = body["part_names"]
    hand_v = {s: np.flatnonzero(dom_part == pn.index(f"hand.{s}")) for s in "LR"}
    # forearm and hand: what lies folded over the torso
    fore_v = {s: np.flatnonzero((dom_part == pn.index(f"forearm.{s}")) | (dom_part == pn.index(f"hand.{s}"))) for s in "LR"}
    torso_v = np.flatnonzero(dom_part == pn.index("torso"))
    labels_v = np.array(labels)[body["skin_idx"][np.arange(nv), body["skin_w"].argmax(1)]]
    # open hands and feet differ the most between photos and the body:
    # they don't count toward staying inside the outline
    loose = (dom_part == pn.index("hand.L")) | (dom_part == pn.index("hand.R")) | np.array([l.startswith(("foot", "toe")) for l in labels_v])
    spread = rng.choice(np.flatnonzero(~loose), 1500, replace=False)
    fore_s = {s: rng.choice(fore_v[s], min(200, len(fore_v[s])), replace=False) for s in "LR"}
    # torso vertices facing forward in the rest pose (normal toward -y)
    fn = np.cross(body["verts"][body["faces"][:, 1]] - body["verts"][body["faces"][:, 0]],
                  body["verts"][body["faces"][:, 2]] - body["verts"][body["faces"][:, 0]])
    vn = np.zeros((nv, 3))
    for j in range(3):
        np.add.at(vn, body["faces"][:, j], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    tf = torso_v[vn[torso_v, 1] < -0.2]
    tf_s = tf
    hand_tgt = {}
    for s, side in (("L", "left"), ("R", "right")):
        pts = [P[f"{side}_{k}"] for k in ("pinky", "index", "thumb") if f"{side}_{k}" in P]
        if pts:
            hand_tgt[s] = np.mean(pts, 0)
    leg_names = [f"{sd}_{j}" for sd in ("left", "right") for j in ("hip", "knee", "ankle")]
    leg_names += [f"{sd}_{j}" for sd in ("left", "right") for j in ("shoulder", "elbow", "wrist")]
    leg_rows = [body["kp_names"].index(n) for n in leg_names]
    lkp_idx, lkp_w = body["kp_idx"][leg_rows], body["kp_w"][leg_rows]
    # the bare neck, read from the photo's skin first: the body's build is
    # fitted to it (its girth there), not only to the joints
    skin_read = read_skin(parts, face_mesh, pose, photo_mask)
    neck_sides = None
    if "neck_px" in skin_read and "neck" in pn:
        nv_ = np.flatnonzero(dom_part == pn.index("neck"))
        if len(nv_) > 30:
            zz = body["verts"][nv_, 2]
            midz = (zz.min() + zz.max()) / 2.0
            ring = nv_[np.abs(zz - midz) < 0.15 * np.ptp(zz)]
            xs_ = body["verts"][ring, 0]
            lo_, hi_ = np.quantile(xs_, [0.1, 0.9])
            neck_sides = (ring[xs_ <= lo_], ring[xs_ >= hi_])
    neck_extra = [] if neck_sides is None else [neck_sides[0], neck_sides[1]]
    sub = np.unique(np.concatenate([kp_idx.ravel(), lkp_idx.ravel(), spread, tf_s, *fore_s.values(), *[hand_v[s] for s in hand_tgt], *neck_extra]))
    pos = {v: i for i, v in enumerate(sub)}

    def at(vs):
        return np.array([pos[v] for v in np.ravel(vs)]).reshape(np.shape(vs))

    kp_sub, spread_sub, tf_sub, lkp_sub = at(kp_idx), at(spread), at(tf_s), at(lkp_idx)
    neck_sub = None if neck_sides is None else (at(neck_sides[0]), at(neck_sides[1]))
    fore_sub = {s: at(fore_s[s]) for s in "LR"}
    hand_sub = {s: at(hand_v[s]) for s in hand_tgt}
    dist_out = distance_transform_edt(~photo_mask)

    # start: upper arms down from the A-pose; a forearm whose wrist lies
    # across the shoulders' span starts folded across the body
    rv0 = np.zeros((len(bones), 3))
    crossed = {}
    lo_x, hi_x = sorted((P["left_shoulder"][0], P["right_shoulder"][0]))
    for s, side, sign in (("L", "left", 1.0), ("R", "right", -1.0)):
        rv0[bones.index(f"upperarm01.{s}")] = [0.0, 0.70 * sign, 0.0]
        wr = P.get(f"{side}_wrist")
        crossed[s] = wr is not None and lo_x < wr[0] < hi_x
        if crossed[s]:
            rv0[bones.index(f"lowerarm01.{s}")] = [-0.5, 1.45 * sign, 0.0]
    Rw0, pw0 = _bone_transforms(body, bone_ids, np.zeros_like(rv0))
    k0 = (kp_w[..., None] * _skin(body, Rw0, pw0, sub)[kp_sub]).sum(1)
    il, ir = names.index("left_shoulder"), names.index("right_shoulder")
    s0 = sw_px / max(abs(k0[il, 0] - k0[ir, 0]), 1e-6)
    mid = (P["left_shoulder"] + P["right_shoulder"]) / 2
    cam0 = np.array([np.log(s0), mid[0], mid[1] + s0 * (k0[il, 2] + k0[ir, 2]) / 2])

    # Levenberg-Marquardt with exact derivatives. A small turn w (world axes)
    # of posed bone b about its joint c_b moves every vertex skinned to b or
    # its descendants by w x (x_vk - c_b) for each bone k it follows.
    nb = len(bones)
    desc = np.zeros((nb, len(labels)), dtype=bool)
    for i, b in enumerate(bone_ids):
        desc[i, b] = True
    for k, p in enumerate(body["parents"]):  # parents first: inherit
        if p >= 0:
            desc[:, k] |= desc[:, p]
    sub_si, sub_sw = body["skin_idx"][sub], body["skin_w"][sub]
    n_st = len(_STRETCH)
    n_ph = len(body["dverts"]) + 1 + n_st  # build, head size, then bone lengths
    ph_range = np.array([_BUILD_RANGE] * len(body["dverts"]) + [_HEAD_RANGE] + [_STRETCH_RANGE] * n_st)
    ph_prior = np.sqrt(np.array([_BUILD_PRIOR] * len(body["dverts"]) + [_HEAD_PRIOR] + [_STRETCH_PRIOR] * n_st))

    def rest_at(ph):
        return _rest(body, ph, sub)

    # the hierarchy by depth: (joints, their parents) per level
    depth = np.zeros(len(body["parents"]), dtype=int)
    for j_, p_ in enumerate(body["parents"]):
        depth[j_] = 0 if p_ < 0 else depth[p_] + 1
    levels = []
    for dpt in range(depth.max() + 1):
        js = np.flatnonzero(depth == dpt)
        levels.append((js, None if dpt == 0 else np.asarray(body["parents"])[js]))
    W_sub = np.zeros((len(sub), len(body["parents"])))  # the subset's skin weights, dense
    np.add.at(W_sub, (np.repeat(np.arange(len(sub)), sub_si.shape[1]), sub_si.ravel()), sub_sw.ravel())
    nbuild = len(body["dverts"])
    NW = 3 + 3 * nb + n_ph  # Jacobian columns: camera, each posed bone's turn, then build / head / bone lengths
    sub_dv, sub_hw, sub_sdv = body["dverts"][:, sub], body["head_w"][sub], body["stretch_dv"][:, sub]

    def rest_basis(verts_sub, joints):
        """How the subset's rest vertices and all rest joints move per unit of
        each build / head size / bone length value (all rest-pose shapes)."""
        Bv = np.zeros((n_ph, len(sub), 3))
        Bj = np.zeros((n_ph, len(joints), 3))
        Bv[:nbuild], Bj[:nbuild] = sub_dv, body["djoints"]
        c = joints[body["head_bone"]]
        Bv[nbuild] = sub_hw[:, None] * (verts_sub - c)
        Bj[nbuild, body["head_desc"]] = joints[body["head_desc"]] - c
        Bv[nbuild + 1:], Bj[nbuild + 1:] = sub_sdv, body["stretch_dj"]
        return Bv, Bj
    gy_d, gx_d = np.gradient(dist_out)

    def skew(a):
        z = np.zeros(a.shape[:-1])
        return np.stack([np.stack([z, -a[..., 2], a[..., 1]], -1), np.stack([a[..., 2], z, -a[..., 0]], -1),
                         np.stack([-a[..., 1], a[..., 0], z], -1)], -2)

    def log_rot(R):
        c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
        th = np.arccos(c)
        w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
        f = np.where(th > 1e-6, th / (2 * np.sin(np.maximum(th, 1e-6))), 0.5)
        return w * f[:, None]

    def world(local_sel, joints):
        parents = body["parents"]
        local = np.tile(np.eye(3), (len(parents), 1, 1))
        local[bone_ids] = local_sel
        Rw = np.empty_like(local)
        pw = np.empty_like(joints)
        for j, p in enumerate(parents):
            if p < 0:
                Rw[j], pw[j] = local[j], joints[j]
            else:
                Rw[j] = Rw[p] @ local[j]
                pw[j] = pw[p] + Rw[p] @ (joints[j] - joints[p])
        return Rw, pw

    def evaluate(cam, local_sel, ph, stage, jac=True):
        s = np.exp(cam[0])
        verts_sub, joints = rest_at(ph)
        Rw, pw = world(local_sel, joints)
        Rw_sub = Rw[sub_si]
        moved = np.einsum("vkij,vkj->vki", Rw_sub, verts_sub[:, None, :] - joints[sub_si]) + pw[sub_si]
        v = (sub_sw[:, :, None] * moved).sum(1)
        p2 = np.column_stack([cam[1] + s * v[:, 0], cam[2] - s * v[:, 2]])
        rv = log_rot(local_sel)
        par = np.array([body["parents"][b] for b in bone_ids])
        Rp = np.where((par >= 0)[:, None, None], Rw[np.maximum(par, 0)], np.eye(3)[None])
        res, rows = [], []
        if jac:
            # dv/dw for every subset vertex and posed bone: (V, nb, 3, 3)
            on = desc[:, sub_si]  # (nb, V, 9)
            wk = sub_sw[None] * on
            a = np.einsum("bvk,vki->vbi", wk, moved) - wk.sum(2).T[:, :, None] * pw[bone_ids][None]
            dv = -skew(a)  # w x a = -a x w
            dX, dZ, dY = dv[:, :, 0, :], dv[:, :, 2, :], dv[:, :, 1, :]
            # image-plane derivatives: (V, 2, 3 + 3 nb)
            Jp = np.zeros((len(v), 2, NW))
            Jp[:, 0, 0], Jp[:, 1, 0] = s * v[:, 0], -s * v[:, 2]
            Jp[:, 0, 1], Jp[:, 1, 2] = 1.0, 1.0
            Jp[:, 0, 3:3 + 3 * nb] = (s * dX).reshape(len(v), -1)
            Jp[:, 1, 3:3 + 3 * nb] = (-s * dZ).reshape(len(v), -1)
            # build, head size, bone lengths: rest shapes carried through the
            # pose (joints first, down the hierarchy, then the skinned vertices)
            Bv, Bj = rest_basis(verts_sub, joints)
            D = np.zeros_like(Bj)
            for js, ps in levels:  # a whole depth of the hierarchy at once
                if ps is None:
                    D[:, js] = Bj[:, js]
                else:
                    D[:, js] = D[:, ps] + np.matmul(Rw[ps][None], (Bj[:, js] - Bj[:, ps])[..., None])[..., 0]
            # sum_k w_k (R_k (Bv - Bj_k) + D_k) = (sum_k w_k R_k) Bv + sum_k w_k (D_k - R_k Bj_k),
            # both sums as products with the dense weight matrix
            M = (W_sub @ Rw.reshape(len(Rw), 9)).reshape(-1, 3, 3)
            E = D - np.einsum("jab,ijb->ija", Rw, Bj)
            first = M[None, :, :, 0] * Bv[..., 0:1] + M[None, :, :, 1] * Bv[..., 1:2] + M[None, :, :, 2] * Bv[..., 2:3]
            second = (W_sub @ E.transpose(1, 0, 2).reshape(len(Rw), -1)).reshape(len(sub), n_ph, 3).transpose(1, 0, 2)
            dvp = first + second
            Jp[:, 0, 3 + 3 * nb:] = s * dvp[:, :, 0].T
            Jp[:, 1, 3 + 3 * nb:] = -s * dvp[:, :, 2].T
            dYf = np.concatenate([dY.reshape(len(v), -1), dvp[:, :, 1].T], axis=1)

        def add(r, J=None):
            res.append(np.atleast_1d(r))
            if jac:
                rows.append(np.atleast_2d(J))

        kp2 = (kp_w[..., None] * p2[kp_sub]).sum(1)
        sq = np.sqrt(wts)[:, None]
        add(((kp2 - tgt) * sq / sw_px).ravel(),
            (np.einsum("kn,knij->kij", kp_w, Jp[kp_sub]) * sq[:, :, None] / sw_px).reshape(-1, NW) if jac else None)
        for sd, t in hand_tgt.items():
            add((p2[hand_sub[sd]].mean(0) - t) / sw_px, Jp[hand_sub[sd]].mean(0) / sw_px if jac else None)
        if neck_sub is not None:
            # the neck as broad as its bare skin in the photo
            da = p2[neck_sub[1], 0].mean() - p2[neck_sub[0], 0].mean()
            sg = 1.0 if da >= 0 else -1.0
            kn = np.sqrt(_NECK_WEIGHT) / sw_px
            add(kn * (sg * da - skin_read["neck_px"]),
                (kn * sg * (Jp[neck_sub[1], 0].mean(0) - Jp[neck_sub[0], 0].mean(0)))[None] if jac else None)
        pr = np.sqrt(0.02 * prior)
        add((pr[:, None] * rv).ravel(), None)
        if jac:
            Jr = np.zeros((nb, 3, NW))
            for i in range(nb):
                Jr[i, :, 3 + 3 * i:6 + 3 * i] = pr[i] * Rp[i].T
            rows[-1] = Jr.reshape(-1, NW)
        # a front photo can't show the trunk tilting forward/back or turning:
        # keep both near zero (x and z components of the trunk bones' turns)
        add(np.sqrt(0.5) * rv[pitch][:, [0, 2]].ravel(), None)
        if jac:
            Ju = np.zeros((2 * len(pitch), NW))
            for n, i in enumerate(pitch):
                Ju[2 * n, 3 + 3 * i:6 + 3 * i] = np.sqrt(0.5) * Rp[i].T[0]
                Ju[2 * n + 1, 3 + 3 * i:6 + 3 * i] = np.sqrt(0.5) * Rp[i].T[2]
            rows[-1] = Ju
        # legs stay in the photo's plane -- thigh and shin turned toward or
        # away from the camera only as far as the photo needs -- however they
        # bend within it (a lunge, a step to the side)
        lj = (lkp_w[..., None] * v[lkp_sub]).sum(1)  # hip, knee, ankle, shoulder, elbow, wrist per side
        dlj = np.einsum("kn,knc->kc", lkp_w, dYf[lkp_sub]) if jac else None
        segs = [(0, 1, _LEG_STRAIGHT), (1, 2, _LEG_STRAIGHT), (3, 4, _LEG_STRAIGHT), (4, 5, _LEG_STRAIGHT)] if len(leg_i) else []
        # arms too, more loosely; an arm folded across the body is exempt
        # (its elbow comes forward with it)
        segs += [(6, 7, _ARM_PLANE), (7, 8, _ARM_PLANE)] if not crossed.get("L") else []
        segs += [(9, 10, _ARM_PLANE), (10, 11, _ARM_PLANE)] if not crossed.get("R") else []
        for a, b, wgt_ in segs:
            length = max(float(np.linalg.norm(lj[b] - lj[a])), 1e-3)
            add(np.sqrt(wgt_) * (lj[b, 1] - lj[a, 1]) / length,
                np.concatenate([np.zeros(3), np.sqrt(wgt_) * (dlj[b] - dlj[a]).ravel() / length]) if jac else None)
        add(ph_prior * ph, np.hstack([np.zeros((n_ph, 3 + 3 * nb)), np.diag(ph_prior)]) if jac else None)
        if stage:
            # one residual per sampled vertex (zero below the photo's bottom
            # edge), so the count never changes between evaluations
            q = p2[spread_sub]
            vis = (q[:, 1] < H - 2).astype(float)
            qy, qx = np.clip(q[:, 1], 0, H - 1), np.clip(q[:, 0], 0, W - 1)
            d = map_coordinates(dist_out, [qy, qx], order=1)
            over_x = np.maximum(q[:, 0] - (W - 1), 0) + np.maximum(-q[:, 0], 0)
            k = np.sqrt(_INSIDE_WEIGHT / len(q)) / sw_px
            if jac:
                g = np.column_stack([map_coordinates(gx_d, [qy, qx], order=1) + np.sign(q[:, 0] - (W - 1)) * (over_x > 0),
                                     map_coordinates(gy_d, [qy, qx], order=1)]) * vis[:, None]
                Jin = k * np.einsum("vi,vij->vj", g, Jp[spread_sub])
            add(k * vis * (d + over_x), Jin if jac else None)
        tv = v[tf_sub]
        for sd in "LR":
            if not crossed.get(sd):
                continue
            av = v[fore_sub[sd]]
            dxz = ((av[:, None, [0, 2]] - tv[None, :, [0, 2]]) ** 2).sum(-1)
            wgt = np.exp(-dxz / 0.03 ** 2)
            ws = wgt.sum(1)
            ty = (wgt * tv[None, :, 1]).sum(1) / (ws + 1e-9)
            over = (ws > 0.5).astype(float)
            n = max(len(av), 1)
            # in front of the torso's front, resting on it: no surface point
            # further off it than a forearm's thickness and a little more
            f1 = np.maximum(av[:, 1] - (ty - 0.01), 0.0)
            f2 = np.maximum((ty - _REST_MAX) - av[:, 1], 0.0)
            Jy = np.zeros((len(av), NW))
            if jac:
                Jy[:, 3:] = dYf[fore_sub[sd]]
            add(np.sqrt(200.0 / n) * over * f1, (np.sqrt(200.0 / n) * (over * (f1 > 0))[:, None] * Jy) if jac else None)
            add(np.sqrt(200.0 / n) * over * f2, (-np.sqrt(200.0 / n) * (over * (f2 > 0))[:, None] * Jy) if jac else None)
            # and touching it: the nearest few points no more than 1.5 cm off
            # (always one residual, zero when too little lies over the torso)
            c, Jc = 0.0, np.zeros(NW)
            if over.sum() >= 8:
                gap = np.where(over > 0, ty - av[:, 1], np.inf)
                near = np.argsort(gap)[:8]
                c = max(float(gap[near].mean()) - 0.015, 0.0)
                Jc = -np.sqrt(50.0) * (c > 0) * Jy[near].mean(0)
            add(np.sqrt(50.0) * c, Jc if jac else None)
        r = np.concatenate(res)
        return (r, np.vstack(rows), Rp) if jac else (r, None, Rp)

    def step(cam, local_sel, ph, Rp, delta):
        cam = cam + delta[:3]
        w = delta[3:3 + 3 * nb].reshape(-1, 3)
        # world turn at the joint -> the bone's own frame
        local_sel = _rotmats(np.einsum("bji,bj->bi", Rp, w)) @ local_sel
        ph = np.clip(ph + delta[3 + 3 * nb:], -ph_range, ph_range)
        return cam, local_sel, ph

    def evaluate_full(cam, local_sel, ph, stage):
        """Residuals and their Jacobian (all columns analytic)."""
        return evaluate(cam, local_sel, ph, stage)

    cam, local_sel, ph = cam0, _rotmats(rv0), np.zeros(n_ph)
    for stage, iters in ((0, 40), (1, 60)):
        lam = 1e-3
        r, J, Rp = evaluate_full(cam, local_sel, ph, stage)
        cost = float(r @ r)
        for _ in range(iters):
            A = J.T @ J
            g = J.T @ r
            delta = -np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-9 * np.eye(len(A))), g)
            c2, l2, p2_ = step(cam, local_sel, ph, Rp, delta)
            r2, J2, Rp2 = evaluate_full(c2, l2, p2_, stage)
            cost2 = float(r2 @ r2)
            if cost2 < cost:
                done = cost - cost2 < 1e-7 * cost
                cam, local_sel, ph, r, J, Rp, cost = c2, l2, p2_, r2, J2, Rp2, cost2
                lam = max(lam / 3, 1e-7)
                if done:
                    break
            else:
                lam *= 4
                if lam > 1e6:
                    break
    final_cost = cost
    rv = log_rot(local_sel)
    x = np.concatenate([cam, rv.ravel()])

    def unpack(x):
        return x[:3], x[3:].reshape(-1, 3)

    cam, rv = unpack(x)
    s = float(np.exp(cam[0]))
    verts_b, joints_b = _rest(body, ph)
    body = dict(body, verts=verts_b, joints=joints_b)
    Rw, pw = _bone_transforms(body, bone_ids, rv)
    V = _skin(body, Rw, pw)
    px = np.column_stack([cam[1] + s * V[:, 0], cam[2] - s * V[:, 2], s * V[:, 1]])
    kp2 = (kp_w[..., None] * px[kp_idx, :2]).sum(1)
    err = np.linalg.norm(kp2 - tgt, axis=1)
    vis = (px[:, 1] < H - 2) & ~loose
    q = px[vis]
    inb = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0)
    inside = np.zeros(len(q), dtype=bool)
    inside[inb] = dist_out[q[inb, 1].astype(int), q[inb, 0].astype(int)] <= _INSIDE_TOL * s
    # folded forearms must rest on the body, not reach out in front of it
    Vw = V
    tv = Vw[tf]
    lifts = []
    for sd in "LR":
        if crossed.get(sd):
            av = Vw[fore_v[sd]]
            dxz = ((av[:, None, [0, 2]] - tv[None, :, [0, 2]]) ** 2).sum(-1)
            near = dxz.min(1) < 0.03 ** 2
            if near.any():
                lifts.append((tv[dxz.argmin(1), 1] - av[:, 1])[near])
    lifts = np.concatenate(lifts) if lifts else np.zeros(1)
    lift_med, lift_max = float(np.median(lifts)), float(lifts.max())
    # posture: joints front to back (y), metres
    J3 = pw
    lean_legs = max([abs(float(J3[labels.index(f"foot.{sd}"), 1] - J3[labels.index(f"upperleg01.{sd}"), 1])) for sd in "LR"]) if legs else 0.0
    lean_trunk = abs(float(J3[labels.index("neck01"), 1] - J3[labels.index("root"), 1]))
    # the folded-away wrists are guesses on the photo too; judge the rest
    judged = [e for n, e in zip(names, err) if not n.endswith("wrist")]
    kp_med = float(np.median(judged)) / sw_px if judged else 1.0
    inside_frac = float(inside.mean()) if len(q) else 0.0
    report = {"kp_err_px": {n: round(float(e), 1) for n, e in zip(names, err)}, "kp_median": round(kp_med, 3),
              "inside": round(inside_frac, 3), "forearm_lift_m": [round(lift_med, 3), round(lift_max, 3)],
              "lean_m": {"legs": round(lean_legs, 3), "trunk": round(lean_trunk, 3)}, "px_per_m": round(s, 1), "gender": gender, "cost": round(final_cost, 5),
              "build": {str(k): round(0.5 + float(v), 2) for k, v in zip(_data()["phenotypes"], ph)},
              "head_scale": round(1.0 + float(ph[len(_data()["phenotypes"])]), 3),
              "bones": {k: round(1.0 + float(v), 3) for k, v in zip(_STRETCH, ph[len(_data()["phenotypes"]) + 1:])},
              "matched": bool(kp_med <= _MAX_KP_ERR and inside_frac >= _MIN_INSIDE and lift_med <= _MAX_FOREARM_LIFT[0] and lift_max <= _MAX_FOREARM_LIFT[1]
                             and lean_legs <= _MAX_LEG_LEAN and lean_trunk <= _MAX_TRUNK_LEAN)}
    # every keypoint the body knows (photo px x, y, depth), visible or not:
    # the builder decides from them how much of the body the photo shows
    # the depth of each vertex's own bones: what the builder scales depth about
    core_depth = s * (body["skin_w"][:, :, None] * pw[body["skin_idx"]]).sum(1)[:, 1]
    allkp = (body["kp_w"][..., None] * px[body["kp_idx"]]).sum(1)
    joints_px = {n: [round(float(a), 1) for a in allkp[i]] for i, n in enumerate(body["kp_names"])}
    # the head's own axes in the photo's frame (x right, y down, depth back),
    # for turning the ears; the ears as the photo shows them
    head_axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]) @ Rw[body["head_bone"]]
    ears, report["ears"] = measure_ears(photo_mask, face_mesh, s, parts=parts)
    # the bare skin read first, and how broad the posed body's neck came out
    report["skin"] = skin_read
    if neck_sides is not None:
        report["skin"]["neck_body_px"] = round(abs(float(px[neck_sides[1], 0].mean() - px[neck_sides[0], 0].mean())), 1)
    if parts is not None:
        report["head_has"] = [k for k, v in parts.get("found", {}).items() if v]
    return {"matched": report["matched"], "report": report, "skin": skin_read, "vertices": px, "joints_px": joints_px, "core_depth": core_depth, "faces": body["faces"], "part_w": body["part_w"],
            "part_names": np.array(body["part_names"]), "px_per_m": s, "ear_w": body["ear_w"], "head_axes": head_axes, "ears": ears, "parts": parts}
