"""Body first: the photo read, a 3D body shaped and posed to it, clothes laid
on it, the photo's own detail on its surface, out as one closed solid.

The outline builder (silhouette_relief) samples the posed body into one
front and one back depth per point of the photo and builds from those, so
whatever overlaps in the photo (an arm across the chest, a hand on an arm,
an ear against the head) is one column. Here the posed body *is* the model:

1. the photo, read: pose and face points, outline, depth, clothes, hair
   (lab.prepare_inputs);
2. the 3D body: the Anny body (body_reference) shaped to the photo -- build,
   head size, bone lengths kept to human proportions -- and posed onto its
   points (fitted when the inputs are prepared);
3. cut where the photo is cut (a bust) -- the first thing done with the 3D
   body, so nothing after it spends time on legs, or trousers, that won't
   be there;
4. fitted out to the photo's outline along smoothed normals (the room
   clothes and hair take; the face held to its own points; what
   reaches below the cut trimmed off at the end), hands closed to mittens;
5. all four sides understood before anything is built on them
   (plan_four_sides): the front against the photo's outline and depth, each
   section's depth from the side against the human figure's (corrected
   toward it, within limits), the posture from the side, what the clothes
   and hair will cover round the back;
6. the clothes and hair on it: a part the photo shows mostly under clothes
   clothed all round (smooth as cloth lies, with a cloth's thickness), hair
   over the back of the head, each garment stepping over the one under it
   where they meet;
7. the photo's detail on the surface the camera sees: face shaping, the
   face's parts, clothes' steps and fine depth detail, pushed toward the
   camera on the vertices that are frontmost (an arm across the chest gets
   the sleeve's folds, the chest under it nothing);
8. out, scaled to the figure's height.

Hands come out as mittens (the body's are open with the fingers spread);
the hair is the body grown to the outline over the head (thick hair and
buns are the outline builder's still).
"""
from typing import Optional

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from .mesh_types import BodyMesh

_MITTEN_SMOOTH = 40  # hand smoothing passes: open fingers -> a mitten
_DETAIL_FACING = (0.25, 0.7)  # detail fades in between these shares of facing the camera
_SUBDIVIDE = 2
_GROW_NORMAL_SMOOTH = 30  # smoothing passes for the directions the body grows along
_CLOTHED = 0.5  # a part the photo shows at least this much under garments is clothed all round
_CLOTH_SMOOTH = 25  # smoothing passes for cloth over the unseen sides
_CLOTH_THICK = 0.004  # a garment's thickness over the body, metres
_HAIR_LAYER = 0.005  # short hair over the scalp, metres (more: the outline's growth)


def _largest_piece(V, F, per_vertex: dict):
    """The body without its separate eyes and teeth: the biggest connected
    piece, with each per-vertex array cut to it."""
    m = trimesh.Trimesh(V, F, process=False)
    labels = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(F))
    keep_f = labels == np.bincount(labels).argmax()
    used, inv = np.unique(F[keep_f], return_inverse=True)
    return V[used], inv.reshape(-1, 3), {k: (None if v is None else np.asarray(v)[used]) for k, v in per_vertex.items()}


def _cut_below(V, F, row: float, per_vertex: dict):
    """The body cut at a photo row (keeps what's above it: y runs down) and
    capped, still a closed solid (manifold3d); each per-vertex array carried
    through the cut (interpolated where the cut makes new points)."""
    from manifold3d import Manifold, Mesh

    names_, cols = [], [V]
    for k, a in per_vertex.items():
        if a is None:
            continue
        a = np.asarray(a, dtype=np.float64)
        a = a[:, None] if a.ndim == 1 else a
        names_.append((k, a.shape[1], per_vertex[k].ndim == 1))
        cols.append(a)
    props = np.hstack(cols).astype(np.float32)
    man = Manifold(mesh=Mesh(vert_properties=props, tri_verts=np.asarray(F, dtype=np.uint32)))
    if man.is_empty():
        raise RuntimeError("the posed body isn't a closed solid")
    man = man.trim_by_plane((0.0, -1.0, 0.0), -float(row))
    pieces = man.decompose()
    if len(pieces) > 1:
        man = max(pieces, key=lambda c: c.num_tri())
    m = man.to_mesh()
    P = np.asarray(m.vert_properties, dtype=np.float64)
    T = np.asarray(m.tri_verts, dtype=np.int64)
    # (points at one position with different values come back as separate
    # points -- a seam the later smoothing opened into cracks: one point
    # each, its values averaged)
    _, first, inv = np.unique(np.round(P[:, :3], 5), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    merged = np.zeros((len(first), P.shape[1]))
    np.add.at(merged, inv, P)
    merged /= np.bincount(inv, minlength=len(first))[:, None]
    P, T = merged, inv[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    out, i = {k: None for k in per_vertex}, 3
    for k, w, flat in names_:
        out[k] = P[:, i] if flat else P[:, i:i + w]
        i += w
    return P[:, :3], T, out


def _cover_unseen(V, F, part_w, names, inputs, ppm, still=None):
    from . import silhouette_relief as sr
    from .clothes import GARMENTS, NAMES

    parts = inputs.parts or {}
    clothes = parts.get("clothes")
    labels = parts.get("labels")
    shape = inputs.photo_mask.shape
    near, _ = _raster_front(V, F, shape)
    r = np.clip(np.rint(V[:, 1]).astype(int), 0, shape[0] - 1)
    c = np.clip(np.rint(V[:, 0]).astype(int), 0, shape[1] - 1)
    seen = np.isfinite(near[r, c]) & (V[:, 2] <= near[r, c] + 1.5)
    lap = sr._mesh_laplacian(F, len(V))
    dom = part_w.argmax(1)
    nrm = sr._vertex_normals(V, F)
    out = V.copy()
    if clothes:
        g_ids = [NAMES.index(g) + 1 for g in GARMENTS]
        worn = np.isin(clothes["labels"][r, c], g_ids)
        cover = np.zeros(len(V))
        for i, n in enumerate(names):
            if n in ("head", "hand.L", "hand.R"):
                continue
            on = (dom == i) & seen
            if on.sum() >= 20 and worn[on].mean() >= _CLOTHED:
                cover[dom == i] = 1.0
        if cover.any():
            cover = np.clip(sr._mesh_smooth(lap, cover, 6), 0.0, 1.0)
            Vs = np.column_stack([sr._mesh_smooth(lap, V[:, k], _CLOTH_SMOOTH) for k in range(3)])
            # smoothed where the camera doesn't see (the front keeps the
            # photo's own shape), then a cloth's thickness out along the normal
            hidden = cover * np.clip(sr._mesh_smooth(lap, (~seen).astype(np.float64), 4), 0.0, 1.0)
            out = out + hidden[:, None] * (Vs - V) + (cover * _CLOTH_THICK * ppm)[:, None] * nrm
            # the gap between cloth and body, from every side, filled: round
            # the torso the cloth spans its hollows; down the back it hangs
            # from the shoulder blades (seen from the front, the photo's own
            # shape stays)
            if "torso" in names:
                torso = (dom == names.index("torso")) & (cover > 0.5)
                fill = _cloth_fill(out, torso, (inputs.reference or {}).get("joints_px") or {})
                out = out + (cover * (1.0 - 0.7 * seen))[:, None] * (fill - out)
    if labels is not None and "head" in names:
        from .head_parts import HAIR, HAT

        head = part_w[:, names.index("head")]
        hairy = seen & np.isin(labels[r, c], (HAIR, HAT))
        if hairy.sum() >= 20:
            fm = inputs.face_mesh
            nape = float(np.asarray(fm)[2, 1]) if fm is not None else np.inf  # below the nose's base no hair at the back
            want = np.where(hairy, 1.0, 0.0)
            back = (~seen) & (head > 0.5) & (V[:, 1] < nape)
            want = np.where(back, 1.0, want)
            known = np.flatnonzero(seen | back | (head < 0.3))
            h = np.clip(sr._harmonic_fill(lap, known, want[known]), 0.0, 1.0) * np.clip(head * 2.0, 0.0, 1.0)
            # a hairline, not a fade: on or off, eased over a couple of vertices
            h = np.clip(sr._mesh_smooth(lap, (h > 0.5).astype(np.float64), 2), 0.0, 1.0)
            out = out + (h * _HAIR_LAYER * ppm)[:, None] * nrm
    if still is not None:
        out[still] = V[still]  # (the cut's cap keeps its place)
    return out


# human depth against width (ANSUR II, minimal clothing; the body here has the
# clothes' room on it): head length / breadth, chest and waist depth through
# the centre line / shoulder-joint breadth
_ANSUR = {"head": 1.2914, "chest": 0.611, "waist": 0.572}
_CLOTHED_DEPTH = 1.06  # the clothes' room on top of a bare body's depth
_PLAN_TOLERANCE = 0.08  # a section's depth within this share of the human figure's is left as it is
_PLAN_MAX_FIX = 0.15  # ... and moved at most this share toward it


def _views(V, F, res: float):
    """Depth maps of the body seen from the front, back, left and right
    (orthographic, photo px; nearer is larger), and each view's outline."""
    from .silhouette_relief import _raster_depth

    x, y, d = V[:, 0], V[:, 1], -V[:, 2]  # d: toward the camera
    out = {}
    for name, (u, w) in {"front": (x, d), "back": (-x, -d), "left": (d, -x), "right": (-d, x)}.items():
        u0, v0 = u.min(), y.min()
        gu, gv = (u - u0) / res, (y - v0) / res
        shape = (int(np.ceil(gv.max())) + 2, int(np.ceil(gu.max())) + 2)
        near, _ = _raster_depth(gu, gv, -w, F, shape)
        out[name] = {"height": np.where(np.isfinite(near), -near, np.nan), "outline": np.isfinite(near)}
    return out


def plan_four_sides(V, F, part_w, names, inputs, ref):
    """The posed body seen from all four sides before anything is built on
    it: the front against the photo (outline, which parts sit nearer), each
    section's depth from the side against the human figure's (corrected
    toward it, within limits), the posture from the side, and which parts
    the clothes and the hair will cover round the back. Returns the body
    (depths corrected) and the plan."""
    from scipy.stats import spearmanr

    mask = np.asarray(inputs.photo_mask, dtype=bool)
    shape = mask.shape
    J = {k: np.asarray(v, dtype=np.float64) for k, v in (ref.get("joints_px") or {}).items()}
    dom = part_w.argmax(1)
    plan = {"sections": {}, "fixed": {}}

    # front, against the photo
    near, _ = _raster_front(V, F, shape)
    cov = np.isfinite(near)
    plan["front_outline_iou"] = round(float((cov & mask).sum() / max((cov | mask).sum(), 1)), 3)
    if inputs.depth is not None and (cov & mask).sum() > 1000:
        d = np.asarray(inputs.depth, dtype=np.float64)
        rr, cc = np.mgrid[0:shape[0]:4, 0:shape[1]:4]
        sel = (cov & mask)[rr, cc]
        dd = map_coordinates(gaussian_filter(d, 0.01 * max(d.shape)), [rr[sel] * d.shape[0] / shape[0], cc[sel] * d.shape[1] / shape[1]], order=1)
        rho = spearmanr(dd, -near[rr[sel], cc[sel]]).correlation
        plan["front_depth_order"] = round(float(rho), 3) if np.isfinite(rho) else None

    # the sides: each section's depth against the human figure's
    if "left_shoulder" in J and "right_shoulder" in J:
        sw = float(np.linalg.norm(J["left_shoulder"][:2] - J["right_shoulder"][:2]))
        sh_y = (J["left_shoulder"][1] + J["right_shoulder"][1]) / 2.0
        cx = (J["left_shoulder"][0] + J["right_shoulder"][0]) / 2.0
        hip_y = (J["left_hip"][1] + J["right_hip"][1]) / 2.0 if "left_hip" in J and "right_hip" in J else sh_y + 1.6 * sw
        bottom = float(np.flatnonzero(mask.any(axis=1))[-1])
        sections = {"chest": sh_y + 0.25 * (hip_y - sh_y), "waist": sh_y + 0.7 * (hip_y - sh_y)}
        # through the torso's own surface (arms left out): front and back
        # where the centre line meets it
        torso_f = (dom[F] == names.index("torso")).all(axis=1) if "torso" in names else None
        if torso_f is not None and torso_f.any():
            tn, tf = _raster_front(V, F[torso_f], shape)
            for sec, row in sections.items():
                r_ = int(round(row))
                if r_ >= bottom - 2 or r_ < 0:
                    continue  # below where the photo is cut
                cols = np.arange(int(cx - 0.04 * sw), int(cx + 0.04 * sw) + 1)
                cols = cols[(cols >= 0) & (cols < shape[1])]
                ok = np.isfinite(tn[r_, cols]) & np.isfinite(tf[r_, cols])
                if ok.sum() < 2:
                    continue
                ratio = float(np.median((tf[r_, cols] - tn[r_, cols])[ok])) / sw
                plan["sections"][sec] = {"depth_over_shoulders": round(ratio, 3), "human": round(_ANSUR[sec] * _CLOTHED_DEPTH, 3)}
        # posture from the side: the head's centre ahead of the shoulders',
        # and the torso's lean (shoulders over hips), as shares of shoulder width
        head = dom == names.index("head") if "head" in names else np.zeros(len(V), bool)
        sh_z = (J["left_shoulder"][2] + J["right_shoulder"][2]) / 2.0 if len(J["left_shoulder"]) > 2 else None
        if head.any() and sh_z is not None:
            plan["head_forward"] = round(float((sh_z - V[head, 2].mean()) / sw), 3)
        if sh_z is not None and "left_hip" in J and len(J["left_hip"]) > 2:
            hip_z = (J["left_hip"][2] + J["right_hip"][2]) / 2.0
            plan["torso_lean"] = round(float((hip_z - sh_z) / sw), 3)
    if "head" in names and inputs.face_mesh is not None:
        fm = np.asarray(inputs.face_mesh)
        ear_w = ref.get("ear_w")
        no_ear = np.ones(len(V), bool) if ear_w is None else np.asarray(ear_w).max(axis=1) < 0.05
        head_f = ((dom[F] == names.index("head")) & no_ear[F]).all(axis=1)
        if head_f.sum() > 50:
            hn, hf = _raster_front(V, F[head_f], shape)
            r_ = int(round((fm[33, 1] + fm[263, 1]) / 2.0))  # the eyes' row (glabella level)
            if 0 <= r_ < shape[0]:
                row_ok = np.isfinite(hn[r_]) & np.isfinite(hf[r_])
                if row_ok.sum() > 4:
                    breadth = float(np.ptp(np.flatnonzero(row_ok)))
                    length = float(np.max((hf[r_] - hn[r_])[row_ok]))
                    plan["sections"]["head"] = {"length_over_breadth": round(length / max(breadth, 1e-6), 3), "human": _ANSUR["head"]}

    # correct each section's depth toward the human figure, about its own
    # core (front and back move apart or together; the outline, which the
    # photo confirms, stays)
    core = ref.get("core_depth")
    if core is not None:
        core = np.asarray(core, dtype=np.float64)
        if len(core) != len(V):
            core = None
    for sec, info in plan["sections"].items():
        have = info.get("depth_over_shoulders", info.get("length_over_breadth"))
        want = info["human"]
        off = have / want - 1.0
        if abs(off) <= _PLAN_TOLERANCE or core is None:
            continue
        if sec == "head" and ((inputs.parts or {}).get("hair") or {}).get("kind") in ("straight", "wavy", "curly", "corded"):
            continue  # (the head's length there is hair as much as skull: left to the hair)
        f = float(np.clip(want / have, 1.0 - _PLAN_MAX_FIX, 1.0 + _PLAN_MAX_FIX))
        if sec == "head":
            w = part_w[:, names.index("head")]
        else:
            row = sections[sec]
            w = part_w[:, names.index("torso")] * np.exp(-((V[:, 1] - row) / (0.25 * sw)) ** 2)
        V = V.copy()
        V[:, 2] = core + (1.0 + w * (f - 1.0)) * (V[:, 2] - core)
        plan["fixed"][sec] = round(f, 3)

    # each side's outline after the corrections (width and height, photo px),
    # kept with the plan
    views = _views(V, F, max(1.0, np.ptp(V[:, 1]) / 300.0))
    plan["views"] = {k: [int(v["outline"].any(axis=0).sum()), int(v["outline"].any(axis=1).sum())] for k, v in views.items()}
    plan["_views"] = views  # (the depth maps themselves, for a look: plan_sheet)

    # the back: what clothes and hair will cover there, from the front
    parts = inputs.parts or {}
    cl = parts.get("clothes")
    if cl:
        plan["clothed_all_round"] = sorted({k for k in cl.get("found", {}) if k in ("suit", "jacket", "coat", "shirt", "tshirt", "sweater",
                                                                               "hoodie", "dress", "kimono")})
    hair = parts.get("hair") or {}
    plan["hair_kind"] = hair.get("kind")
    return V, plan


_CLOTH_HANG = 0.12  # down the back cloth comes in at most this much per unit it falls (a jacket hangs nearly straight)


def _cloth_fill(V, on, joints: dict):
    """The cloth over the torso's vertices `on`, never inside the body: at
    each height it wraps the outside of the torso's cross-section (a tape
    round it -- the spine's groove, the small of the back, the hollows by the
    armpits bridged), and below the shoulder blades its back hangs nearly
    straight down rather than following the body in."""
    from scipy.spatial import ConvexHull, QhullError

    out = V.copy()
    idx = np.flatnonzero(on)
    if len(idx) < 50:
        return out
    if "left_shoulder" in joints and "right_shoulder" in joints:
        sw = float(np.linalg.norm(np.subtract(joints["left_shoulder"][:2], joints["right_shoulder"][:2])))
        sh_y = (joints["left_shoulder"][1] + joints["right_shoulder"][1]) / 2.0
    else:
        sw, sh_y = float(np.ptp(V[idx, 0])), float(V[idx, 1].min())
    h = max(0.03 * sw, 1.0)
    rows = np.floor((V[idx, 1] - V[idx, 1].min()) / h).astype(int)
    # round each section: out to its convex outline in (x, depth)
    for b in np.unique(rows):
        sel = idx[rows == b]
        if len(sel) < 8:
            continue
        P = out[sel][:, [0, 2]]
        try:
            hull = ConvexHull(P)
        except QhullError:
            continue
        c = P.mean(axis=0)
        H = P[hull.vertices]
        E0, E1 = H, np.roll(H, -1, axis=0)
        for k, v in zip(sel, P):
            d = v - c
            if np.hypot(*d) < 1e-6:
                continue
            # where the ray from the centre through the point leaves the hull
            e = E1 - E0
            den = d[0] * e[:, 1] - d[1] * e[:, 0]
            ok = np.abs(den) > 1e-12
            w = E0 - c
            t = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1.0), np.inf)
            u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1.0), -1.0)
            hit = ok & (t > 0) & (u >= -1e-9) & (u <= 1 + 1e-9)
            if hit.any():
                tt = float(t[hit].min())
                if tt > 1.0:
                    out[k, 0], out[k, 2] = c[0] + tt * d[0], c[1] + tt * d[1]
    # down the back: from the shoulder blades it hangs (depth runs toward the back)
    blade = sh_y + 0.25 * sw  # (about where the shoulder blades are fullest)
    xb = np.floor((out[idx, 0] - out[idx, 0].min()) / h).astype(int)
    yb = np.floor((out[idx, 1] - blade) / h).astype(int)
    back = out[idx, 2] >= np.median(out[idx, 2])  # the back half of each section
    for col in np.unique(xb):
        m_ = back & (xb == col)
        if m_.sum() < 3:
            continue
        ys = yb[m_]
        prof = {}
        for yy, zz in zip(ys, out[idx[m_], 2]):
            prof[yy] = max(prof.get(yy, -np.inf), zz)
        hang, best, last = {}, -np.inf, None
        for yy in sorted(prof):
            if yy < 0:
                hang[yy] = prof[yy]
                continue
            if last is not None:
                best -= _CLOTH_HANG * h * (yy - last)
            best = max(best, prof[yy])
            hang[yy] = best
            last = yy
        for k, yy in zip(idx[m_], ys):
            if yy >= 0:
                out[k, 2] = max(out[k, 2], hang[yy])
    return out


def _raster_front(V, F, shape):
    """Nearest depth (toward the back) at each photo pixel, +inf off the body."""
    from .silhouette_relief import _raster_depth

    near, far = _raster_depth(V[:, 0], V[:, 1], V[:, 2], F, shape)
    return near, far


def build_body_first(inputs, target_height_mm: float = 150.0, depth_strength: Optional[float] = 0.5,
                     face_strength: Optional[float] = None, clothes_steps: bool = True, bust: Optional[bool] = None) -> BodyMesh:
    from . import silhouette_relief as sr

    ref = inputs.reference
    if ref is None or not ref.get("matched", True):
        raise RuntimeError("No body could be posed on this photo (the body-first build needs pose points).")
    mask = np.asarray(inputs.photo_mask, dtype=bool)
    shape = mask.shape
    ppm = float(ref["px_per_m"])
    V = np.asarray(ref["vertices"], dtype=np.float64).copy()  # photo px: x right, y down, depth toward the back
    F = np.asarray(ref["faces"], dtype=np.int64)
    V, F, pv = _largest_piece(V, F, {"part_w": ref.get("part_w"), "core": ref.get("core_depth"), "ear_w": ref.get("ear_w")})
    part_w = pv["part_w"]
    names = [str(n) for n in ref["part_names"]]
    body_ref = dict(ref, part_w=part_w, core_depth=pv["core"], ear_w=pv["ear_w"])

    # the first thing done with the 3D body: cut where the photo is cut (a
    # bust), so everything after -- fitting it out to the outline, the
    # plan, clothes, hair, detail -- works only on the torso the figure will
    # be: no time on legs (or trousers) that won't be there
    rows = np.flatnonzero(mask.any(axis=1))
    cut_off = bust if bust is not None else bool(rows[-1] >= shape[0] - 3)
    cut_row = float(rows[-1]) if cut_off else None
    if cut_off:
        V, F, kept = _cut_below(V, F, cut_row, {"part_w": part_w, "core": body_ref.get("core_depth"), "ear_w": body_ref.get("ear_w")})
        part_w = kept["part_w"]
        part_w = part_w / np.maximum(part_w.sum(axis=1, keepdims=True), 1e-9)
        body_ref = dict(body_ref, part_w=part_w, core_depth=kept["core"], ear_w=kept["ear_w"])

    # 4. clothes' (and hair's) room: grown out to the photo's outline
    d_in, d_out = distance_transform_edt(mask), distance_transform_edt(~mask)
    # (grown along the normals of a smoothed copy: along its own, in tight
    # hollows -- lips, nostrils, eye corners, under the chin -- the grown
    # surface folded through itself)
    lap0 = sr._mesh_laplacian(F, len(V))
    Vsm = np.column_stack([sr._mesh_smooth(lap0, V[:, k], _GROW_NORMAL_SMOOTH) for k in range(3)])
    # the face doesn't grow: its shape is the photo's face points' (grown, its
    # lips, nostrils and lids were pushed into each other)
    face_keep = None
    if inputs.face_mesh is not None and "head" in names:
        from scipy.spatial import ConvexHull
        from skimage.measure import points_in_poly

        fm2 = np.asarray(inputs.face_mesh)[:468, :2]
        hull = fm2[ConvexHull(fm2).vertices]
        inside_face = points_in_poly(V[:, :2], hull)
        facing0 = -sr._vertex_normals(Vsm, F)[:, 2] > 0.2
        face_keep = inside_face & facing0 & (part_w[:, names.index("head")] > 0.5)
    # the cut's flat cap: its inside stays put (moved, its long thin
    # triangles folded into each other), its edge fits out with the sides
    # (held, it pulled the sides above it in; so it fits out with them and
    # its inside is put back after)
    cap_in = np.zeros(len(V), dtype=bool)
    if cut_row is not None:
        cap_in = (V[:, 1] >= cut_row - 0.5) & (sr._vertex_normals(V, F)[:, 1] > 0.9)
    V_cap = V[cap_in].copy()
    V, names, _ = sr._grow_to_outline(V, F, body_ref, part_w, mask, lambda P: (P[:, 0], P[:, 1]), shape, d_in, d_out, 1.0, mask,
                                      normals=sr._vertex_normals(Vsm, F), keep=face_keep)
    V[cap_in] = V_cap
    core = body_ref.get("core_depth")
    if core is not None:
        hw = part_w[:, names.index("head")] if "head" in names else 0.0
        m_ = hw * sr._REF_HEAD_MARGIN + (1.0 - hw) * sr._REF_DEPTH_MARGIN
        V[:, 2] = core + m_ * (V[:, 2] - core)
    if (ref.get("report") or {}).get("gender") == "male" and "torso" in names:
        V = sr._drape_chest(V, F, part_w[:, names.index("torso")], ppm, ref.get("joints_px") or {})
    if ref.get("ears") and body_ref.get("ear_w") is not None and ref.get("head_axes") is not None:
        from .body_reference import place_ears

        V = place_ears(V, np.asarray(body_ref["ear_w"], dtype=np.float64), ref["head_axes"], ref["ears"], faces=F, px_per_m=ppm)
    # mitten hands
    hand = np.zeros(len(V))
    for n in ("hand.L", "hand.R"):
        if n in names:
            hand = np.maximum(hand, part_w[:, names.index(n)])
    if hand.max() > 0.5:
        lap = sr._mesh_laplacian(F, len(V))
        Vs = np.column_stack([sr._mesh_smooth(lap, V[:, k], _MITTEN_SMOOTH) for k in range(3)])
        V = V + np.clip(hand * 1.5, 0.0, 1.0)[:, None] * (Vs - V)

    # the four sides, understood before anything is built on them
    V, plan = plan_four_sides(V, F, part_w, names, inputs, body_ref)
    plan["cut_at_row"] = cut_row

    # clothes and hair round the sides the photo doesn't show: a part the
    # photo shows mostly under clothes is clothed all round -- cloth lies
    # smooth over muscle and bone (no shoulder blades or spine through a
    # jacket) -- and a head with hair has it over the back of the skull too
    V = _cover_unseen(V, F, part_w, names, inputs, ppm, still=cap_in)

    # finer surface for the detail
    from trimesh.remesh import subdivide_loop

    V, F = subdivide_loop(V, F, iterations=_SUBDIVIDE)

    # 3. depth order against the photo's estimate (reported; the pose fit
    # already keeps forearms on the body's front)
    near, far = _raster_front(V, F, shape)
    cov = np.isfinite(near) & mask
    front = np.where(cov, -near, 0.0)  # toward the camera
    report = {"plan": plan}

    # 5. the photo's detail, toward the camera, on what the camera sees
    offset = np.zeros(shape)
    parts = inputs.parts or {}
    labels = parts.get("labels")
    fm = inputs.face_mesh
    if fm is not None:
        strength = face_strength if face_strength is not None else (sr._FACE_RELIEF_WITH_DEPTH if depth_strength else 1.0)
        if strength > 0:
            fr = sr._face_relief(fm, shape, 0, 0, 1.0, 0) * strength
            if labels is not None:
                from .head_parts import BODY_SKIN, FACE_SKIN

                fr = fr * gaussian_filter(np.isin(labels, (FACE_SKIN, BODY_SKIN)).astype(np.float32), 1.5)
            offset += fr
            face_read = parts.get("face")
            if face_read:
                try:
                    fp = sr._face_parts_relief(face_read, fm, shape, 0, 0, 1.0, 0, ppm)
                    if labels is not None:
                        from .head_parts import BODY_SKIN, FACE_SKIN, HAIR

                        fp = fp * gaussian_filter(np.isin(labels, (FACE_SKIN, BODY_SKIN, HAIR)).astype(np.float32), 1.0)
                    offset += fp
                except Exception as exc:  # noqa: BLE001
                    print(f"Note: face parts skipped ({exc}).")
    clothes = parts.get("clothes")
    if clothes_steps and clothes:
        try:
            offset += sr._clothes_relief(clothes, front, cov, 0, 0, 1.0, 0, ppm)
        except Exception as exc:  # noqa: BLE001
            print(f"Note: clothes steps skipped ({exc}).")
    if depth_strength and inputs.depth is not None:
        d = np.asarray(inputs.depth, dtype=np.float64)
        dh, dw = d.shape
        rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
        sampled = map_coordinates(d, [rr * (dh / shape[0]), cc * (dw / shape[1])], order=1, mode="nearest")
        vals = sampled[cov]
        if vals.size and vals.std() > 1e-9:
            dn = (sampled - vals.mean()) / vals.std()
            dn = dn - sr._masked_blur(dn, cov, max(0.05 * max(shape), 1.0))  # fine detail only: the body has the big shapes
            half = np.where(cov, (far - near) / 2.0, 0.0)
            rel = np.clip(half * dn * depth_strength, -sr._REF_RELIEF_MAX * ppm, sr._REF_RELIEF_MAX * ppm)
            offset += np.where(cov, rel, 0.0)
            # does the body sit as the photo's depth says? (nearer parts
            # should be nearer): rank agreement of the two, coarse
            if cov.sum() > 1000:
                from scipy.stats import spearmanr

                coarse = gaussian_filter(sampled, 0.02 * max(shape))
                sel = cov & (rr % 4 == 0) & (cc % 4 == 0)
                rho = spearmanr(coarse[sel], front[sel]).correlation  # (the estimate: larger is nearer)
                report["depth_order"] = round(float(rho), 3) if np.isfinite(rho) else None
    # (the estimate's sign: Depth Anything gives larger = nearer; checked by the order above)

    nrm = trimesh.Trimesh(V, F, process=False).vertex_normals
    facing = np.clip((-nrm[:, 2] - _DETAIL_FACING[0]) / (_DETAIL_FACING[1] - _DETAIL_FACING[0]), 0.0, 1.0)
    px, py = V[:, 0], V[:, 1]
    inb = (px >= 0) & (px < shape[1] - 1) & (py >= 0) & (py < shape[0] - 1)
    near_at = map_coordinates(np.where(np.isfinite(near), near, 1e9), [py, px], order=0, mode="nearest")
    visible = inb & (V[:, 2] <= near_at + 1.5)  # frontmost at its pixel (photo px)
    off_v = map_coordinates(offset, [py, px], order=1, mode="nearest")
    V[:, 2] = V[:, 2] - np.where(visible, off_v * facing, 0.0)

    # out, to the figure's frame (the bottom stays flat on the cut: clothes
    # and smoothing rounded it a little)
    if cut_row is not None:
        # (trimmed clean, still a closed solid: pinning what went below onto
        # the cut folded triangles over each other -- thin walls, cracks)
        V, F, _ = _cut_below(V, F, cut_row, {})
    mesh = trimesh.Trimesh(V, F, process=False)
    pieces = mesh.split(only_watertight=False)
    if len(pieces) > 1:
        mesh = max(pieces, key=lambda p: len(p.faces))
    P = np.asarray(mesh.vertices)
    xyz = np.column_stack([P[:, 0] - (P[:, 0].min() + P[:, 0].max()) / 2.0, -P[:, 1], -P[:, 2]])
    xyz[:, 1] -= xyz[:, 1].min()
    height = float(np.ptp(xyz[:, 1]))
    xyz = xyz * (target_height_mm / height if height > 1e-6 else 1.0)
    out = BodyMesh(vertices=xyz, faces=np.asarray(mesh.faces))
    out.report = report  # type: ignore[attr-defined]
    return out


def plan_sheet(plan: dict, size: int = 320):
    """The four sides of a plan side by side, shaded (a PIL image)."""
    from PIL import Image

    tiles = []
    for name in ("front", "left", "back", "right"):
        h = plan["_views"][name]["height"]
        fin = np.isfinite(h)
        z = np.where(fin, h, np.nanmin(h) if fin.any() else 0.0)
        gy, gx = np.gradient(z)
        n = np.dstack([-gx, -gy, np.ones_like(z)])
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        light = np.array([-0.4, -0.5, 0.75]) / np.linalg.norm([-0.4, -0.5, 0.75])
        v = np.where(fin, np.clip(n @ light, 0, 1) * 200 + 40, 22).astype(np.uint8)
        im = Image.fromarray(v)
        im.thumbnail((size, size))
        tile = Image.new("L", (size, size), 22)
        tile.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        tiles.append(tile)
    sheet = Image.new("L", (size * 4, size), 22)
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * size, 0))
    return sheet
