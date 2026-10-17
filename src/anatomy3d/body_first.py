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

_FIT_CHEST = (0.5, 4.0)  # a jacket's (or top's) room each side of the chest, cm (tailoring: ~1-3)
_FIT_WAIST = (0.0, 6.0)  # ... at the waist, cm
_FIT_COLLAR = (0.1, 1.5)  # a collar's room round the neck each side, cm
_FIT_SLEEVE = (0.2, 4.0)  # a sleeve's room round the arm each side, cm
_FIT_TIE = (4.0, 11.0)  # a tie's width, cm
_FIT_FRONT = 0.35  # the front stands off the layer under it by this share of the room at the sides (cloth lies on the chest)
_FIT_THICK = {"inner": 0.0015, "outer": 0.003, "tie": 0.004, "collar": 0.004, "lapel": 0.004}  # the cloth's own thickness, m (a tie's knot and lining more)
_FIT_SHOW = 1.6  # ... shown up this much on a figure (on a person cloth keeps close; on a model the gap is what shows it)
_HAIR_EDGE_SHARE = 0.35  # short hair at its hairline: this share of its layer ...
_HAIRLINE_TAPER = 0.01  # ... the full layer this far in (m)
_SB_JAW = 0.45  # the jaw's angle this many ear heights below the ear's bottom
_SB_ABOVE = 0.25  # side whiskers start at the temple's hair, this many ear heights above the ear's top
_SB_SIDEWAYS = 0.35  # ... only on the head's side (its surface facing sideways at least this much)
_SB_SMOOTH = 8  # its outline eased over this many mesh steps
_SB_TRIMMED = 0.6  # a trimmed whisker's layer, of a full one's
_HAIR_FLAT = ("sleek",)  # hair pulled flat to the head: no layer of its own over the unseen scalp (it raised folds there)
_HAIR_UNSEEN = 1.0  # the short hair's layer over the scalp the front doesn't see, of the seen layer's
_HAIR_SEAM_SMOOTH = 4  # the layer eased across where the front view stops seeing the scalp (mesh steps)
_HAIR_NORMAL_SMOOTH = 6  # ... and laid along the surface's direction smoothed this much (mesh steps)
_PIECE_MIN_H = 0.0008  # a garment piece: where it stands at least this over what is under it (m)
_PIECE_MIN_W = 0.006  # ... nothing of it narrower than this (m)
_PIECE_MIN_AREA = 0.0002  # ... at least this big (m^2: 2 cm^2)
_PIECE_DZ = 0.5  # a piece's depth layers (photo px)
_PIECE_MIN_FACING = 0.3  # ... and none where the surface turns this far side-on (the front view can't place it there)
_PIECE_EDGE_SMOOTH = 0.002  # a piece's outline smoothed over this much (m)
_PIECE_SINK = 0.002  # ... its back this far into the surface under it, so they join (m)
_POP_MAX = 0.014  # no piece stands more than this over what is beside it (m)
_POP_SPAN = 0.05  # a piece stands over the lowest layer within this far (a tie, collar or lapel as a whole)
_POP_BEVEL = 0.0015  # its edge softened over this
_POP_FINE = 1.6  # the clothes' folds, seams and creases deepened this much
_HOLLOW_SCALE = 0.01  # the body showing through the clothes is a hollow this broad or more (a fold is narrower)
_DETAIL_MAX = 0.008  # the photo's depth detail stands at most about this far (m), easing in
_DETAIL_FACING = (0.25, 0.7)  # detail fades in between these shares of facing the camera
_SUBDIVIDE = 2
_GROW_NORMAL_SMOOTH = 30  # smoothing passes for the directions the body grows along
_CLOTHED = 0.5  # a part the photo shows at least this much under garments is clothed all round
_CLOTH_SMOOTH = 25  # smoothing passes for cloth over the unseen sides
_CLOTH_THICK = 0.004  # a garment's thickness over the body, metres
_LONG_HAIR = ("straight", "wavy", "curly", "corded")  # hair kinds with a volume of their own (not a layer on the scalp)
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


_ARM_REACH_MAX = 0.3  # a forearm lengthened at most this share to reach the photo's wrist
_ARM_REST_GAP = 0.004  # the top forearm's underside clears the one under it by this, metres
_FOREARM_MAX = 0.92  # a forearm at most this long for its upper arm (radius / humerus: people ~0.75-0.85)
_ARM_TURN_MAX = 0.06  # a forearm turned at most this much (m) in depth to match the other's length
_ARM_LIFT_MAX = 0.07  # ... lifted at most this (about a forearm's thickness), metres


def top_arm(inputs) -> Optional[str]:
    """Which forearm lies on top when the arms are folded: the one whose wrist
    the photo shows (skin, a cuff, a watch round it); the other's wrist is
    under the top sleeve, all jacket. None when it can't tell."""
    from .clothes import NAMES

    P, cl = inputs.pose or {}, (inputs.parts or {}).get("clothes")
    if not cl or not all(k in P for k in ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder")):
        return None
    lab = cl["labels"]
    sw = float(np.linalg.norm(np.subtract(P["left_shoulder"][:2], P["right_shoulder"][:2])))
    shown = [NAMES.index(k) + 1 for k in ("skin", "watch", "bracelet", "shirt", "tshirt", "gloves") if k in NAMES]
    r = int(max(3, 0.06 * sw))
    share = {}
    for sd in ("left", "right"):
        x, y = (int(round(v)) for v in P[f"{sd}_wrist"][:2])
        win = lab[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        share[sd] = float(np.isin(win, shown).mean()) if win.size else 0.0
    if abs(share["left"] - share["right"]) < 0.25:
        return None
    return "left" if share["left"] > share["right"] else "right"


def _fold_arms(V, F, part_w, names, inputs, ref):
    """Folded arms as the photo holds them: each forearm (and its hand)
    reaching to where the photo's wrist is (the fit trusts wrists little and
    keeps bones near average length; folded forearms lie flat across the
    body at their full length), and the arm whose wrist shows lying over the
    other where they overlap."""
    P = inputs.pose or {}
    J = {k: np.asarray(v, dtype=np.float64) for k, v in (ref.get("joints_px") or {}).items()}
    info = {}
    if not all(k in P and k in J for k in ("left_elbow", "right_elbow", "left_wrist", "right_wrist")):
        return V, info
    V = V.copy()
    side = {"left": "L", "right": "R"}
    arm_w = {}
    moved = {}
    for sd, s_ in side.items():
        fw = part_w[:, names.index(f"forearm.{s_}")] if f"forearm.{s_}" in names else np.zeros(len(V))
        hw = part_w[:, names.index(f"hand.{s_}")] if f"hand.{s_}" in names else np.zeros(len(V))
        e, w = J[f"{sd}_elbow"][:2], J[f"{sd}_wrist"][:2]
        axis = w - e
        L = float(np.hypot(*axis))
        if L < 1e-6:
            continue
        # how far along elbow -> wrist each point is (0 at the elbow, 1 past the wrist)
        frac = np.clip(((V[:, :2] - e) @ axis) / (L * L), 0.0, 1.0)
        wt = np.clip(fw + hw, 0.0, 1.0) * np.where(hw > 0.5, 1.0, frac)
        arm_w[sd] = wt
        want = np.asarray(P[f"{sd}_wrist"][:2], dtype=np.float64)
        move = want - w
        reach = float(np.hypot(*(want - e))) / L - 1.0
        if reach > _ARM_REACH_MAX:
            move *= _ARM_REACH_MAX / reach
        V[:, :2] += wt[:, None] * move[None, :]
        info[f"{sd}_reach"] = round(reach, 3)
        moved[sd] = (wt, w + move, e)
    # the two forearms the same length (a body's two sides are the same
    # size): the photo shows a forearm shorter when it points toward or away
    # from the camera, so the longer is the truer -- the other turned that
    # much in depth, toward the camera
    if len(moved) == 2 and all(f"{sd}_elbow" in J and len(J[f"{sd}_elbow"]) > 2 for sd in moved):
        ln = {}
        for sd, (wt, w2, e) in moved.items():
            dz = float(J[f"{sd}_wrist"][2] - J[f"{sd}_elbow"][2])
            d2 = float(np.hypot(*(w2 - e)))
            ln[sd] = (d2, dz, float(np.hypot(d2, dz)))
        Lc = max(v[2] for v in ln.values())
        # ... but no longer than a forearm is for its upper arm (a misread
        # elbow had them longer than the upper arms)
        ups = [float(np.linalg.norm(np.subtract(J[f"{sd}_elbow"], J[f"{sd}_shoulder"]))) for sd in moved if f"{sd}_shoulder" in J]
        if ups:
            Lc = min(Lc, _FOREARM_MAX * min(ups))
        ppm = float(ref.get("px_per_m") or 0.0)
        info["forearm_px"] = {}
        for sd, (d2, dz, L3) in ln.items():
            if L3 > Lc + 1.0:
                # (shortened back toward the elbow: what its reach overshot)
                cut_ = (L3 - Lc) / max(L3, 1e-6)
                wt_, w2_, e_ = moved[sd]
                V[:, :2] -= (wt_ * cut_)[:, None] * (w2_ - e_)[None, :]
                info[f"{sd}_shortened"] = round(cut_, 3)
                d2, L3 = d2 * (1.0 - cut_), Lc
                dz = dz * (1.0 - cut_)
            if L3 < Lc - 1.0:
                want_dz = (np.sign(dz) if abs(dz) > 1e-6 else -1.0) * np.sqrt(max(Lc * Lc - d2 * d2, 0.0))
                delta = float(np.clip(want_dz - dz, -_ARM_TURN_MAX * ppm, _ARM_TURN_MAX * ppm))
                V[:, 2] += moved[sd][0] * delta
                info[f"{sd}_turned_px"] = round(delta, 1)
                L3 = float(np.hypot(d2, dz + delta))
            info["forearm_px"][sd] = round(L3, 1)
    # folded: both forearms lie across the body (more across than up and
    # down) and over each other (a hand raised to the collar isn't)
    sw = float(np.linalg.norm(np.subtract(P["left_shoulder"][:2], P["right_shoulder"][:2]))) if "left_shoulder" in P and "right_shoulder" in P else 0.0
    seg = {sd: (np.asarray(P[f"{sd}_elbow"][:2], dtype=np.float64), np.asarray(P[f"{sd}_wrist"][:2], dtype=np.float64)) for sd in side}
    across = all(abs(b[0] - a[0]) > 1.5 * abs(b[1] - a[1]) for a, b in seg.values())
    xs = {sd: sorted((a[0], b[0])) for sd, (a, b) in seg.items()}
    overlap = min(xs["left"][1], xs["right"][1]) - max(xs["left"][0], xs["right"][0])
    near_rows = abs(np.mean([seg["left"][0][1], seg["left"][1][1]]) - np.mean([seg["right"][0][1], seg["right"][1][1]])) < 0.3 * max(sw, 1.0)
    folded = across and near_rows and overlap > 0.2 * max(sw, 1.0)
    info["folded"] = bool(folded)
    top = top_arm(inputs) if folded else None
    info["top"] = top
    if top is not None and len(arm_w) == 2:
        under = "right" if top == "left" else "left"
        shape = inputs.photo_mask.shape
        sel = {sd: ((part_w[:, names.index(f"forearm.{side[sd]}")] + part_w[:, names.index(f"hand.{side[sd]}")])[F] > 0.5).all(axis=1)
               for sd in (top, under)}
        tn, tf = _raster_front(V, F[sel[top]], shape)
        un, _ = _raster_front(V, F[sel[under]], shape)
        both = np.isfinite(tf) & np.isfinite(un)
        if both.sum() > 20:
            # (depth runs toward the back) the top's back must sit in front of the other's front
            gap = _ARM_REST_GAP * float(ref["px_per_m"])
            need = min(float(np.percentile((tf - un)[both], 90)) + gap, _ARM_LIFT_MAX * float(ref["px_per_m"]))
            if need > 0:
                V[:, 2] -= arm_w[top] * need
            info["lifted_px"] = round(max(need, 0.0), 1)
    return V, info


def _cover_unseen(V, F, part_w, names, inputs, ppm, still=None):
    from . import silhouette_relief as sr
    from .clothes import GARMENTS, NAMES, WORN

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
        g_ids = [NAMES.index(g) + 1 for g in WORN]
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
            # (unseen because it faces away -- the back, the sides -- not because
            # an arm lies in front of it: smoothed there, the body under the
            # arm came forward into it)
            behind_arm = ~seen & (nrm[:, 2] < -0.2)
            hidden = cover * np.clip(sr._mesh_smooth(lap, (~seen & ~behind_arm).astype(np.float64), 4), 0.0, 1.0)
            # (in depth only: the cloth smooths what stands out of the body,
            # front to back -- smoothed sideways too, the torso drew away from
            # an arm beside it, a crack seen from the front)
            delta = hidden[:, None] * (Vs - V)
            delta[:, :2] = 0.0
            out = out + delta + (cover * _CLOTH_THICK * ppm)[:, None] * nrm
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
    # the outline the body was grown to already holds the clothes' room: no
    # vertex inside it is carried out past it (a cloth's thickness and the
    # gaps it spans, added there again, pushed the outline out)
    d_out = distance_transform_edt(~inputs.photo_mask)

    def past(P):
        return map_coordinates(d_out, [P[:, 1], P[:, 0]], order=1, mode="nearest")

    back_in = past(out) > past(V) + 0.5
    out[back_in, :2] = V[back_in, :2]
    return out


# human depth against width (ANSUR II, minimal clothing; the body here has the
# clothes' room on it): head length / breadth, chest and waist depth through
# the centre line / shoulder-joint breadth
_ANSUR = {"head": 1.2914, "chest": 0.611, "waist": 0.572}
_CLOTHED_DEPTH = 1.06  # the clothes' room on top of a bare body's depth
_PLAN_TOLERANCE = 0.08  # a section's depth within this share of the human figure's is left as it is
_PLAN_MAX_FIX = 0.25  # ... and moved at most this share toward it


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
        # measured the way the human figure's are: breadth the widest across
        # the head above the ears (the ears and the head's sides beside them
        # left out of it), length the deepest front to back from the brows up
        ear_w = ref.get("ear_w")
        ear = None if ear_w is None else np.asarray(ear_w).max(axis=1) > 0.5
        head_f = (dom[F] == names.index("head")).all(axis=1)
        r_ = int(round((fm[33, 1] + fm[263, 1]) / 2.0))  # the eyes' row (glabella level)
        if head_f.sum() > 50 and 0 <= r_ < shape[0]:
            hn, hf = _raster_front(V, F[head_f], shape)
            both = np.isfinite(hn) & np.isfinite(hf)
            rows = np.flatnonzero(both.any(axis=1))
            if len(rows) and rows[0] < r_:
                top = int(rows[0])
                ear_top = int(V[ear, 1].min()) - 2 if ear is not None and ear.any() else int(top + 0.6 * (r_ - top))
                ear_top = int(np.clip(ear_top, top + 2, r_))
                wid = [float(np.ptp(np.flatnonzero(both[r]))) for r in range(top, ear_top) if both[r].sum() > 4]
                thick = np.where(both[top:r_ + 1], (hf - hn)[top:r_ + 1], -np.inf).max()
                if wid and np.isfinite(thick):
                    breadth, length = max(wid), float(thick)
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
        if sec == "head" and ((inputs.parts or {}).get("hair") or {}).get("kind") in _LONG_HAIR:
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


_JACKET_DETAIL = 0.015  # under a jacket the photo's depth adds only detail finer than this share of the photo
_BUST_DRAPE = 0.2  # a bust under a jacket: the cloth spans it as one form (a ball this size, m, rolled over it)
_STIFF = ("suit", "jacket", "coat", "hoodie", "sweater", "kimono")  # garments a bust doesn't show through
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


class _Checks:
    """The reality check after every step of the build: what the step made,
    measured against the photo, the human body or a printable solid, with
    the range it should be in. A failed check doesn't stop the build; it
    says which step went wrong, in the report (and the Lab's results)."""

    def __init__(self):
        self.rows = []

    def add(self, step: str, check: str, value, lo=None, hi=None, unit: str = "") -> bool:
        ok = value is not None and np.isfinite(value) and (lo is None or value >= lo) and (hi is None or value <= hi)
        self.rows.append({"step": step, "check": check, "value": None if value is None else round(float(value), 3),
                          "range": [lo, hi], "unit": unit, "ok": bool(ok)})
        return bool(ok)

    def failed(self) -> list:
        return [f"{r['step']}: {r['check']} {r['value']}{r['unit']} (want {r['range'][0]}..{r['range'][1]})"
                for r in self.rows if not r["ok"]]


# what the checks hold the steps to
_LOCK_MOVE = 0.01  # a locked part moves at most this far (m) in a later step
_LOCK_IOU = 0.01  # the locked outline's overlap with the photo's drops at most this much
_LOCK_DEPTH_LESS = 0.03  # a locked section's depth: later steps take at most this share away (clothes only add)
_LOCK_DEPTH_MORE = 0.20  # ... and add at most this share (cloth spanning a bust or a chest)
_LOCK_HOLLOW_MM = 1.0  # ... and the hollow under the clothes deepens at most this much past its limit
_CHK_JOINT_MISS = 0.08  # the posed body's joints off the photo's, at most this share of the shoulders' width
_CHK_FACE_M = (0.10, 0.20)  # the face's breadth at the photo's scale, metres (people: ~0.13-0.15)
_CHK_FOREARM_LR = (0.97, 1.03)  # left forearm over right
_CHK_FOREARM_UPPER = (0.65, 0.95)  # forearm over upper arm (radius / humerus ~0.78)
_CHK_HAND_SW = (0.10, 0.40)  # a hand's breadth over the shoulders' width (a hand half hidden shows less)
_CHK_OUTLINE = 0.90  # the front's outline against the photo's (overlap)
_CHK_SECTION = (0.85, 1.15)  # a section's depth for its width against the human figure's, after the plan
_CHK_CLOTH_HOLLOW_MM = 6.0  # under clothes, no hollow deeper than this in the front (the body showing through)
_EYE_DOME = 0.5  # an open eye's ball domes up to this share of its set-back in the middle
_CHK_EYE_MM = 1.0  # an open eye sits at least this far behind the lids round it (mm, on the person)
_CHK_BROW_MM = 1.0  # the eyebrows stand at least this far off the skin round them (mm, on the person)
_CHK_PIECE_MM = (0.8, 16.0)  # a garment piece stands this much over what is under it (mm, on the person)
_CHK_HEAD_OUTLINE = 0.85  # the head and hair's outline against the photo's
_CHK_MOVE_MM = 24.0  # ... and no single point of the surface moves further than this in it
_CHK_WALL_MM = 0.8  # the thinnest wall a print holds (mm)
_CHK_SPIKES = 3.0  # outward folds sharper than 60 degrees, at most this share (%)
_LOCK_SPIKES = 0.5  # a step adds at most this share (% points) of such folds
_CHK_RELIEF_MM = 10.0  # the photo's depth detail moves the surface at most this much


_HAND_REACH = 0.55  # the photo's hand: its skin within this many shoulder widths of the wrist (a hand's length)
_HAND_MIN_AREA = 0.01  # ... at least this share of the shoulder width squared (less: hidden or gloved)
_HAND_SPAN = 0.03  # a hand over the body lies across bumps and gaps narrower than this (m)
_HAND_THICK_RATIO = 0.35  # a palm is this thick for its breadth (hand thickness / breadth at the knuckles)
_HAND_THICK_MIN = 0.015  # ... and never thinner than this (m)
_HAND_MIN_W = 0.008  # a hand's solid has nothing narrower than this (m)
_HAND_STUB = 0.15  # the body's own hand goes into the wrist, drawn to this share of its size


def _skin_hands(names, inputs):
    """The skin the photo shows comes first: each hand it shows is taken
    from the photo -- its skin past the wrist, split from the face and neck
    and the other hand where they touch (each skin pixel goes to whichever
    is nearer along the skin) -- and becomes a solid of its own once what it
    lies on is built (_hand_solids). Returns [(photo region, hand part)] and
    what was found per hand."""
    from skimage.graph import MCP_Geometric

    from .clothes import NAMES

    info, regions = {}, []
    P = inputs.pose or {}
    cl = (inputs.parts or {}).get("clothes")
    if not cl or "skin" not in NAMES or "left_shoulder" not in P or "right_shoulder" not in P:
        return regions, info
    shape = inputs.photo_mask.shape
    sw = float(np.linalg.norm(np.subtract(P["left_shoulder"][:2], P["right_shoulder"][:2])))
    skin = (cl["labels"] == NAMES.index("skin") + 1) & inputs.photo_mask
    if not skin.any():
        return regions, info
    ys, xs = np.nonzero(skin)

    def on_skin(pts):
        out = []
        for x, y in pts:
            d = np.hypot(xs - x, ys - y)
            i = int(np.argmin(d))
            if d[i] < 0.08 * sw:
                out.append((int(ys[i]), int(xs[i])))
        return out

    seeds = {}
    fm = inputs.face_mesh
    if fm is not None:
        fp = np.asarray(fm)[:468, :2]
        chin = fp[152] if len(fp) > 152 else fp[np.argmax(fp[:, 1])]
        seeds["face"] = on_skin([*fp[::10], chin + (0.0, 0.1 * sw), chin + (0.0, 0.2 * sw)])
    for side in ("left", "right"):
        k = [P[f"{side}_{t}"][:2] for t in ("index", "pinky", "thumb") if f"{side}_{t}" in P]
        if k and f"{side}_wrist" in P:
            tip = np.mean(np.asarray(k, dtype=np.float64), axis=0)
            wr = np.asarray(P[f"{side}_wrist"][:2], dtype=np.float64)
            seeds[side] = on_skin([*k, tip, (tip + wr) / 2.0])
    seeds = {k: v for k, v in seeds.items() if v}
    if not any(k in seeds for k in ("left", "right")):
        return regions, info
    cost = np.where(skin, 1.0, np.inf)
    dist = {}
    for k, v in seeds.items():
        dist[k], _ = MCP_Geometric(cost).find_costs(v)
    order = list(dist)
    owner = np.argmin(np.stack([dist[k] for k in order]), axis=0)
    for side, n in (("left", "hand.L"), ("right", "hand.R")):
        if side not in dist or n not in names:
            continue
        wr = np.asarray(P[f"{side}_wrist"][:2], dtype=np.float64)
        tip = np.mean([P[f"{side}_{t}"][:2] for t in ("index", "pinky", "thumb") if f"{side}_{t}" in P], axis=0)
        H = skin & (owner == order.index(side)) & np.isfinite(dist[side])
        hy, hx = np.nonzero(H)
        d = np.column_stack([hx, hy]).astype(np.float64) - wr
        hl2 = float(np.dot(tip - wr, tip - wr))
        ok = (d @ (tip - wr) >= -0.1 * hl2) & (np.hypot(d[:, 0], d[:, 1]) <= _HAND_REACH * sw)
        H = np.zeros(shape, dtype=bool)
        H[hy[ok], hx[ok]] = True
        if H.sum() < _HAND_MIN_AREA * sw * sw:
            info[side] = "too little of the hand shows"
            continue
        regions.append((H, n))
        on_face = 0.0
        if fm is not None:
            from skimage.measure import points_in_poly
            from scipy.spatial import ConvexHull

            fp2 = np.asarray(fm)[:468, :2]
            hy, hx = np.nonzero(H)
            on_face = float(points_in_poly(np.column_stack([hx, hy]), fp2[ConvexHull(fp2).vertices]).mean())
        info[side] = {"px": int(H.sum()), "breadth_sw": round(2.0 * float(distance_transform_edt(H).max()) / sw, 3),
                      "on_face": round(on_face, 3)}
    return regions, info


def _tuck_hands(V, part_w, names):
    """The body's own hands (open, fingers spread) drawn into the wrists,
    out of sight -- the photo's hands are built as solids of their own. The
    mesh, the vertices to hold from here on (grown out to the outline, their
    folds stood out) and each wrist's depth."""
    held = np.zeros(len(V), dtype=bool)
    wrist_z = {}
    for n in ("hand.L", "hand.R"):
        if n in names:
            w_ = part_w[:, names.index(n)]
            ring = (w_ > 0.4) & (w_ < 0.6)
            if ring.sum() >= 3:
                rc = V[ring].mean(axis=0)
                wrist_z[n] = float(rc[2])
                pull = np.clip((w_ - 0.5) / 0.2, 0.0, 1.0)
                V = V + pull[:, None] * ((rc + _HAND_STUB * (V - rc)) - V)
                held |= pull > 0.5
    return V, held, wrist_z


def _hand_solids(V, F, regions, ppm):
    """The hands the photo shows, each a closed solid: the photo's hand for its
    outline, its thickness from its own widths, lying on what is under it
    (sunk a little into it, so they join) or beside the body at its wrist's
    depth. [(vertices, faces)] in photo px."""
    from . import silhouette_relief as sr

    out = []
    if not regions:
        return out
    near, _ = _raster_front(V, F, regions[0][0].shape)
    have = np.isfinite(near)
    if not have.any():
        return out
    _, (iy, ix) = distance_transform_edt(~have, return_indices=True)
    rest = -near[iy, ix]  # (heights toward the camera)
    from scipy.ndimage import maximum_filter

    k = max(int(round(_HAND_SPAN * ppm)), 3)
    top = gaussian_filter(maximum_filter(rest, size=k), k / 2.0)  # across the top of what it lies on
    from scipy.ndimage import binary_fill_holes, binary_opening

    from skimage.morphology import medial_axis

    r = max(int(round(_HAND_MIN_W / 2.0 * ppm)), 1)
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    # (no slivers narrower than _HAND_MIN_W -- between fingers, round what
    # the hand holds -- and no holes: thin walls, tunnels)
    regions = [(binary_opening(binary_fill_holes(H), structure=disk), wz) for H, wz in regions]
    regions = [(H, wz) for H, wz in regions if H.sum() >= 20]
    # the two hands are the same size: the palm's breadth from the hand that
    # shows the most of it (a hand half hidden under an arm shows less)
    breadth = max((2.0 * float(distance_transform_edt(H).max()) for H, _ in regions), default=0.0)
    for H, wz in regions:
        # its volume from its own measurements: round across wherever it is
        # narrow (a finger is as thick as it is wide), and the palm as thick
        # as _HAND_THICK_RATIO of its breadth -- each point as thick as the
        # hand is wide there
        skel, D = medial_axis(H, return_distance=True)
        _, (sy, sx) = distance_transform_edt(~skel, return_indices=True)
        R = D[sy, sx]  # (the hand's half-width at the nearest point of its middle)
        cap = max(_HAND_THICK_RATIO * breadth, _HAND_THICK_MIN * ppm) / 2.0
        Rc = np.minimum(R, cap)
        half = gaussian_filter(np.where(H, np.sqrt(np.clip(D * (2.0 * Rc - D), 0.0, None)), 0.0), 1.0)
        half = np.minimum(half, Rc)
        # over the body: lying on it (its middle a half-width above what it
        # lies on, its back down to the body itself -- lying on the top of
        # the arms with gaps under it left tunnels through the joined
        # model); beside it, round about its own wrist's depth
        front = top + Rc + half
        back = rest - sr._HAND_SINK * ppm
        if wz is not None:
            over = gaussian_filter(have.astype(np.float64), 2.0)
            front = over * front + (1.0 - over) * (-wz + half)
            back = over * back + (1.0 - over) * (-wz - half)
        h0 = float(back[H].min()) - 3.0
        zs = np.arange(0.0, float(front[H].max()) - h0 + 3.0, 1.0)
        piece = sr._layer_solid(front - h0, back - h0, H, zs, 1.0, 1.0, True)
        if piece is not None:
            pv, pf = piece
            pv = np.column_stack([pv[:, 2], pv[:, 1], -(pv[:, 0] + h0)])
            if trimesh.Trimesh(pv, pf, process=False).volume < 0:
                pf = pf[:, ::-1]
            out.append((pv, pf))
    return out


class _Locks:
    """Each step's result, locked once it is made and checked: after every
    later step it is measured again, and a later step that would break it is
    applied only as far as everything locked before still holds (backed off
    toward where that step started) -- or not at all. So no step undoes what
    an earlier one got right."""

    _TRIES = (1.0, 0.75, 0.5, 0.25, 0.0)

    def __init__(self, chk: "_Checks"):
        self.chk, self.items = chk, []

    def lock(self, step: str, what: str, measure, ok, st: dict):
        m0 = measure(st)
        if m0 is not None:
            self.items.append((step, what, measure, ok, m0))

    def broken(self, st: dict) -> list:
        out = []
        for step, what, measure, ok, m0 in self.items:
            m = measure(st)
            if m is not None and not ok(m, m0):
                out.append(f"{what} (locked at {step}: {round(float(m0), 3)}, now {round(float(m), 3)})")
        return out

    def hold(self, step: str, before: dict, after: dict) -> dict:
        """`after` as far as the locks allow (same mesh as `before`)."""
        first = self.broken(after)
        if not first:
            self.chk.add(step, "everything locked before still holds", 1.0, lo=1.0)
            return after
        for t in self._TRIES[1:]:
            st = dict(after, V=before["V"] + t * (after["V"] - before["V"]))
            if not self.broken(st):
                break
        self.chk.add(step, "applied only this far, to keep: " + "; ".join(first), t, lo=1.0)
        return st


def _outline_iou(st: dict, mask) -> float:
    n, _ = _raster_front(st["V"], st["F"], mask.shape)
    c = np.isfinite(n)
    return float((c & mask).sum() / max((c | mask).sum(), 1))


def _part_centres(st: dict, names, parts) -> Optional[np.ndarray]:
    """Where each of `parts` is (the centre of its vertices, photo px)."""
    out = []
    for n in parts:
        if n in names:
            on = st["pw"][:, names.index(n)] > 0.5
            if on.sum() >= 10:
                out.append(st["V"][on].mean(axis=0))
    return np.asarray(out) if out else None


def _section_depth(st: dict, names, J: dict, mask, sec: str) -> Optional[float]:
    """A torso section's depth through its centre line over the shoulders'
    width (the four-side plan's own measure)."""
    if "torso" not in names or "left_shoulder" not in J or "right_shoulder" not in J:
        return None
    V, F = st["V"], st["F"]
    sw = float(np.linalg.norm(J["left_shoulder"][:2] - J["right_shoulder"][:2]))
    sh_y = (J["left_shoulder"][1] + J["right_shoulder"][1]) / 2.0
    cx = (J["left_shoulder"][0] + J["right_shoulder"][0]) / 2.0
    hip_y = (J["left_hip"][1] + J["right_hip"][1]) / 2.0 if "left_hip" in J and "right_hip" in J else sh_y + 1.6 * sw
    row = int(round(sh_y + {"chest": 0.25, "waist": 0.7}[sec] * (hip_y - sh_y)))
    if row < 0 or row >= float(np.flatnonzero(mask.any(axis=1))[-1]) - 2:
        return None
    tf_ = (st["pw"].argmax(1)[F] == names.index("torso")).all(axis=1)
    if not tf_.any():
        return None
    tn, tb = _raster_front(V, F[tf_], mask.shape)
    cols = np.arange(int(cx - 0.04 * sw), int(cx + 0.04 * sw) + 1)
    cols = cols[(cols >= 0) & (cols < mask.shape[1])]
    ok = np.isfinite(tn[row, cols]) & np.isfinite(tb[row, cols])
    return float(np.median((tb[row, cols] - tn[row, cols])[ok])) / sw if ok.sum() >= 2 else None


def _cloth_hollow_mm(st: dict, names, clothes, ppm: float) -> Optional[float]:
    """Under the clothes the photo shows, the front's deepest hollow (95th
    percentile, mm) a cloth wouldn't follow: the body showing through."""
    from scipy.ndimage import binary_erosion, grey_closing

    from .clothes import GARMENTS, WORN, NAMES as CNAMES

    if not clothes or "torso" not in names:
        return None
    V, F = st["V"], st["F"]
    Ft = F[(st["pw"][:, names.index("torso")][F] > 0.5).all(axis=1)]
    nt, _ = _raster_front(V, Ft, clothes["labels"].shape)
    on_t = np.isfinite(nt)
    worn = np.isin(clothes["labels"], [CNAMES.index(g) + 1 for g in WORN]) & on_t
    worn = binary_erosion(worn, iterations=int(max(0.02 * ppm, 1)))
    if worn.sum() <= 200:
        return None
    hgt = gaussian_filter(np.where(on_t, -nt, float(np.nanmin(np.where(on_t, -nt, np.nan)))), _HOLLOW_SCALE * ppm)  # (the body's hollows, not the folds')
    k = max(int(round(0.03 * ppm)), 3)
    return float(np.percentile((grey_closing(hgt, size=(k, k)) - hgt)[worn], 95)) / ppm * 1000.0


def _head_iou(st: dict, mask, fm) -> Optional[float]:
    if fm is None:
        return None
    chin = int(np.asarray(fm)[:468, 1].max())
    n, _ = _raster_front(st["V"], st["F"], mask.shape)
    mh, ph = np.isfinite(n[:chin]), mask[:chin]
    return float((mh & ph).sum() / max((mh | ph).sum(), 1))


def _spike_pct(st: dict) -> float:
    """The share of the surface's outward folds sharper than 60 degrees (the
    Lab's own spike measure)."""
    m = trimesh.Trimesh(st["V"], st["F"], process=False)
    sharp = (np.degrees(m.face_adjacency_angles) > 60) & m.face_adjacency_convex
    return 100.0 * float(sharp.mean()) if len(sharp) else 0.0


def _measure_fit(V_bare, V_dressed, F, pw, names, clothes, J: dict, ppm: float, fm) -> dict:
    """The tailoring: each garment measured against the body before it goes
    on -- the bare posed body against the outline the photo shows dressed:
    how much room (ease) the jacket or shirt has each side at the chest and
    waist, each sleeve round its arm, the collar round the neck; the tie's
    width and length. From those, each layer's height over the one under it
    (its gap, and its own thickness shown up for a figure). cm and mm on the
    person; heights in photo px."""
    from .clothes import NAMES

    shape = clothes["labels"].shape
    lab = clothes["labels"]
    found = clothes.get("found") or {}
    dom = pw.argmax(1)
    m = {}

    def width_at(Vx, part, row):
        f_ = (dom[F] == names.index(part)).all(axis=1) if part in names else None
        if f_ is None or not f_.any():
            return None
        n_, _ = _raster_front(Vx, F[f_], shape)
        r_ = int(round(row))
        if not 0 <= r_ < shape[0]:
            return None
        cols = np.flatnonzero(np.isfinite(n_[r_]))
        return float(np.ptp(cols)) if len(cols) > 2 else None

    def side_room(part, row):
        a, b = width_at(V_bare, part, row), width_at(V_dressed, part, row)
        return None if a is None or b is None else (b - a) / 2.0 / ppm * 100.0  # cm each side

    if "left_shoulder" in J and "right_shoulder" in J:
        sh_y = (J["left_shoulder"][1] + J["right_shoulder"][1]) / 2.0
        hip_y = (J["left_hip"][1] + J["right_hip"][1]) / 2.0 if "left_hip" in J and "right_hip" in J else sh_y + 1.6 * float(
            np.linalg.norm(J["left_shoulder"][:2] - J["right_shoulder"][:2]))
        m["chest_room_cm"] = side_room("torso", sh_y + 0.25 * (hip_y - sh_y))
        m["waist_room_cm"] = side_room("torso", sh_y + 0.7 * (hip_y - sh_y))
        if fm is not None:
            chin = float(np.asarray(fm)[152, 1])
            m["collar_room_cm"] = side_room("neck", (chin + sh_y) / 2.0)

    # each sleeve round its arm: (dressed area - bare area) / the arm's length, each side
    for part, (j0, j1) in {"upperarm": ("shoulder", "elbow"), "forearm": ("elbow", "wrist")}.items():
        rooms = []
        for sd, s_ in (("left", "L"), ("right", "R")):
            n = f"{part}.{s_}"
            if n not in names or f"{sd}_{j0}" not in J or f"{sd}_{j1}" not in J:
                continue
            L2 = float(np.linalg.norm(J[f"{sd}_{j1}"][:2] - J[f"{sd}_{j0}"][:2]))
            f_ = (dom[F] == names.index(n)).all(axis=1)
            if L2 < 0.05 * ppm or not f_.any():
                continue  # (pointing at the camera: no width to read)
            a = np.isfinite(_raster_front(V_bare, F[f_], shape)[0]).sum()
            b = np.isfinite(_raster_front(V_dressed, F[f_], shape)[0]).sum()
            rooms.append((b - a) / L2 / 2.0 / ppm * 100.0)
        if rooms:
            m[f"{part}_sleeve_room_cm"] = float(np.mean(rooms))
    if "tie" in found and "tie" in NAMES:
        t = lab == NAMES.index("tie") + 1
        ys = np.flatnonzero(t.any(axis=1))
        if len(ys):
            m["tie_width_cm"] = float(np.median(t.sum(axis=1)[ys])) / ppm * 100.0
            m["tie_length_cm"] = float(np.ptp(ys)) / ppm * 100.0
    m = {k: (round(v, 2) if v is not None else None) for k, v in m.items()}

    # each layer's height over the skin (px): the gap it stands off what is
    # under it (measured, kept within what tailoring allows) and its own
    # thickness, shown up for a figure
    def room(key, lo, hi, default):
        v = m.get(key)
        return float(np.clip(v if v is not None else default, lo, hi)) / 100.0  # metres

    inner_gap = room("collar_room_cm", *_FIT_COLLAR, 0.4) * _FIT_FRONT
    outer_gap = room("chest_room_cm", *_FIT_CHEST, 1.5) * _FIT_FRONT
    inner = ("shirt", "tshirt", "dress", "kimono")
    outer = ("suit", "jacket", "coat", "sweater", "hoodie")
    H = {"skin": 0.0}
    has_inner = any(k in found for k in inner)
    for g in inner:
        H[g] = (inner_gap + _FIT_THICK["inner"] * _FIT_SHOW) * ppm
    base = max((H[g] for g in inner if g in found), default=0.0) if has_inner else 0.0
    for g in outer:
        H[g] = base + (outer_gap + _FIT_THICK["outer"] * _FIT_SHOW) * ppm
    H["tie"] = base + _FIT_THICK["tie"] * _FIT_SHOW * ppm
    # a shirt's collar: its band stands round the neck over the shirt; a
    # lapel: the jacket's front folded back on itself, over the jacket
    H["collar"] = max((H[g] for g in inner if g in found), default=H["shirt"]) + _FIT_THICK["collar"] * _FIT_SHOW * ppm
    H["lapel"] = max((H[g] for g in outer if g in found), default=H["suit"]) + _FIT_THICK["lapel"] * _FIT_SHOW * ppm
    m["heights_mm"] = {k: round(v / ppm * 1000.0, 1) for k, v in H.items() if k in found or k == "skin"}
    m["_H"] = H
    return m


def _clothes_pop(clothes: dict, front, cov, ppm: float, H: dict, neck_rows=None, sharp: bool = False):
    """Offsets (photo px, toward the camera) for the clothes as tailored
    (_measure_fit): each piece -- a tie, a collar, a lapel, a jacket's front
    edge, a cuff -- stands over what is under it, as a whole (within
    _POP_SPAN), by that layer's measured height: its gap off the layer under
    it and its own thickness; its edge softened by a small bevel."""
    from scipy.ndimage import minimum_filter

    from .clothes import NAMES

    lab = clothes["labels"]
    lut = np.full(len(NAMES) + 1, np.nan)
    for k, h in H.items():
        if k in NAMES:
            lut[NAMES.index(k) + 1] = h
    Hp = lut[lab]
    if neck_rows is not None and "skin" in NAMES:
        # (skin is under the clothes only at the neck and in the V: a hand
        # lies on the jacket, not under it)
        rows_ = np.arange(lab.shape[0])[:, None]
        Hp = np.where((lab == NAMES.index("skin") + 1) & ~((rows_ >= neck_rows[0]) & (rows_ <= neck_rows[1])), np.nan, Hp)
    named = np.isfinite(Hp) & cov
    if not named.any():
        return (np.zeros(cov.shape), lab) if sharp else np.zeros(cov.shape)
    iy, ix = distance_transform_edt(~named, return_distances=False, return_indices=True)
    Hf = np.nan_to_num(np.where(named, Hp, Hp[iy, ix]), nan=0.0)
    span = max(int(round(_POP_SPAN * ppm)), 3)
    base = gaussian_filter(minimum_filter(np.where(cov, Hf, np.inf), size=span), span / 4.0)
    base = np.where(np.isfinite(base), base, Hf)
    lift = np.clip(Hf - base, 0.0, _POP_MAX * ppm)
    for k, t in (("belt", 0.004), ("watch", 0.006), ("bracelet", 0.003)):
        if k in NAMES:
            lift = lift + t * ppm * (lab == NAMES.index(k) + 1)
    if sharp:
        return lift, lab
    lift = gaussian_filter(lift, max(_POP_BEVEL * ppm, 0.5))
    gy, gx = np.gradient(front)
    facing = np.clip((1.0 / np.sqrt(1.0 + np.hypot(gy, gx) ** 2) - 0.35) / 0.35, 0.0, 1.0)  # (fading where the surface turns side-on)
    # on the clothes (and the skin at the neck) only: the face and the hands
    # are their own
    from .clothes import GARMENTS, NAMES, WORN

    worn = np.isin(lab, [NAMES.index(g) + 1 for g in WORN])
    if neck_rows is not None and "skin" in NAMES:
        rows_ = np.arange(lab.shape[0])[:, None]
        worn |= (lab == NAMES.index("skin") + 1) & (rows_ >= neck_rows[0]) & (rows_ <= neck_rows[1])
    on = np.clip(gaussian_filter(worn.astype(np.float64), 1.0) * 2.0, 0.0, 1.0)
    return np.where(cov, lift * facing * on, 0.0)


def _over_neighbour(height, outer, inner, reach_px: float, gap):
    """How far (px) each point of `outer` has to come forward to lie over the
    `inner` beside it: near where they meet, the outer one's face at least
    `gap` in front of the inner one's actual surface (a jacket over the tie
    and the shirt, a collar round the neck, hair over the forehead), eased
    out over `reach_px`. `height`: the surface's height toward the camera."""
    from scipy.ndimage import grey_dilation

    if not inner.any() or not outer.any():
        return np.zeros(height.shape)
    k = max(int(round(reach_px)), 3)
    ceiling = grey_dilation(np.where(inner, height, -1e9), size=(k, k))
    d = distance_transform_edt(~inner)
    w = np.clip(1.0 - d / max(reach_px, 1.0), 0.0, 1.0)
    need = np.where(outer & (ceiling > -1e8), ceiling + gap - height, 0.0)
    return gaussian_filter(np.clip(need, 0.0, None) * w * outer, 1.0)


def _garment_pieces(V, F, clothes: dict, lift, lab, ppm: float, neck_rows=None):
    """The clothes as pieces of their own: each garment region the photo shows
    standing over the layer under it (_clothes_pop's measured heights, with
    no bevel) -- a tie, a lapel and the jacket's front edge, the shirt's collar
    and V, a cuff -- becomes a closed solid on the finished surface, its face
    that surface (the folds on it) raised by the piece's height, its edges
    walls. [(vertices, faces)] in photo px, and what was made of each."""
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc_label

    from . import silhouette_relief as sr
    from .clothes import GARMENTS, NAMES, WORN

    shape = lab.shape
    near, _ = _raster_front(V, F, shape)
    have = np.isfinite(near)
    out, made = [], []
    if not have.any():
        return out, made
    _, (iy, ix) = distance_transform_edt(~have, return_indices=True)
    rest = -near[iy, ix]  # (heights toward the camera)
    gy, gx = np.gradient(np.where(have, rest, 0.0))
    facing = np.clip((1.0 / np.sqrt(1.0 + np.hypot(gy, gx) ** 2) - 0.35) / 0.35, 0.0, 1.0)
    r = max(int(round(_PIECE_MIN_W / 2.0 * ppm)), 1)
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    from .study import _OUTER

    names = [g for g in dict.fromkeys(list(WORN) + ["belt", "watch", "bracelet"]) if g in NAMES]
    rows_ = np.arange(shape[0])[:, None]
    neck_skin = (lab == NAMES.index("skin") + 1) & ((rows_ >= neck_rows[0]) & (rows_ <= neck_rows[1]) if neck_rows else False)
    for g in names:
        # (over what lies under it beside it, not only over its own surface)
        inner_names = [i for i in _OUTER.get(g, ()) + (("neck",) if g in ("shirt", "tshirt", "dress") else ()) if i in NAMES or i == "neck"]
        inner = np.isin(lab, [NAMES.index(i) + 1 for i in inner_names if i in NAMES]) | (neck_skin if "neck" in inner_names else False)
        own = lab == NAMES.index(g) + 1
        clear = _over_neighbour(np.where(have, rest, 0.0), own, inner & have, _POP_SPAN * ppm,
                                np.where(own, np.maximum(lift, _PIECE_MIN_H * ppm), 0.0))
        lift_g = np.minimum(np.maximum(lift, clear), _POP_MAX * ppm)  # (clearing a raised arm beside it, a piece climbed 3 cm)
        reg = own & have & (lift_g * facing > _PIECE_MIN_H * ppm) & (facing > _PIECE_MIN_FACING)
        if reg.sum() < 20:
            continue
        reg = binary_opening(binary_fill_holes(reg), structure=disk)
        # (its outline smoothed: the photo's label edge is a pixel staircase
        # that read as a row of steps along a lapel or a tie)
        reg = gaussian_filter(reg.astype(np.float64), _PIECE_EDGE_SMOOTH * ppm) > 0.5
        comps, n = cc_label(reg)
        for k in range(1, n + 1):
            P = comps == k
            if P.sum() < _PIECE_MIN_AREA * ppm * ppm:
                continue
            h = gaussian_filter(np.where(P, lift_g * facing, 0.0), 0.7)
            front = rest + h
            back = rest - _PIECE_SINK * ppm
            h0 = float(back[P].min()) - 3.0
            zs = np.arange(0.0, float(front[P].max()) - h0 + 3.0, _PIECE_DZ)  # (finer than a pixel: on a turning surface whole-pixel layers showed as terraces)
            piece = sr._layer_solid(front - h0, back - h0, P, zs, _PIECE_DZ, 1.0, True)
            if piece is None:
                continue
            pv, pf = piece
            pv = np.column_stack([pv[:, 2], pv[:, 1], -(pv[:, 0] + h0)])
            if trimesh.Trimesh(pv, pf, process=False).volume < 0:
                pf = pf[:, ::-1]
            out.append((pv, pf))
            made.append({"garment": g, "px": int(P.sum()), "height_mm": round(float(np.median(h[P])) / ppm * 1000.0, 1)})
    return out, made


def _sideburns_3d(V, F, part_w, names, ear_w, head_axes, read: dict, fm, ppm: float):
    """The side whiskers as read (face_parts._sideburns), on the side of the
    head in front of each ear: from the temple's hair down to the read
    length (0 at the ear's top, 1 at the jaw's angle), the read width (x the
    face's size), widening to the foot by the read flare (mutton chops) or
    narrowing (tapered), a short hair's layer thick (less where trimmed),
    laid along the smoothed surface. The same on both sides. Returns the
    surface and what was built."""
    from . import silhouette_relief as sr

    kind = read.get("type")
    if kind in (None, "none", "covered") or "head" not in names:
        return V, {"type": kind, "built": False}
    A = head_axes  # columns: the head's left, back, up (photo px)
    loc = V @ A
    head = (part_w[:, names.index("head")] > 0.5) & (ear_w.max(axis=1) < 0.5)  # (the skin round the ear's base is where they grow)
    if not head.any():
        return V, {"type": kind, "built": False}
    loc[:, 0] -= float(np.median(loc[head, 0]))  # (side to side from the head's middle)
    lap = sr._mesh_laplacian(F, len(V))
    n_s = sr._vertex_normals(V, F)
    n_s = np.column_stack([sr._mesh_smooth(lap, n_s[:, j], _HAIR_NORMAL_SMOOTH) for j in range(3)])
    n_s /= np.linalg.norm(n_s, axis=1, keepdims=True) + 1e-12
    nrm = n_s @ A  # (smoothed: the raw ones made its edge lumpy)
    fsize = max(float(np.ptp(np.asarray(fm)[:468, 0])), float(np.ptp(np.asarray(fm)[:468, 1])))
    h = np.zeros(len(V))
    built = {}
    for i, side in enumerate("LR"):
        core = ear_w[:, i] > 0.95
        if core.sum() < 10:
            continue
        u = 1.0 if side == "L" else -1.0
        e = loc[core]
        front, top, bottom = float(e[:, 1].min()), float(e[:, 2].max()), float(e[:, 2].min())
        ear_h = max(top - bottom, 1.0)
        jaw = bottom - _SB_JAW * ear_h
        z_start = top + _SB_ABOVE * ear_h
        z_end = top - float(read.get("length", 0.2)) * (top - jaw)
        w_top = float(read.get("width", 0.06)) * fsize
        flare = float(np.clip(read.get("flare", 1.0), 0.3, 2.5))
        # down the whisker: 0 at its top, 1 at its foot
        t = np.clip((z_start - loc[:, 2]) / max(z_start - z_end, 1.0), 0.0, 1.0)
        width = w_top * (1.0 + (flare - 1.0) * t) if read.get("end") != "tapered" else w_top * (1.0 - 0.8 * t)
        ahead = front - loc[:, 1]  # how far in front of the ear's front edge
        on = head & (u * nrm[:, 0] > _SB_SIDEWAYS) & (u * loc[:, 0] > 0) & (loc[:, 2] <= z_start) & (loc[:, 2] >= z_end - 0.1 * ear_h)
        across = np.clip(1.0 - (ahead - 0.5 * width) / np.maximum(0.25 * width, 1.0), 0.0, 1.0) * (ahead > -0.15 * ear_h)
        foot = np.clip((loc[:, 2] - (z_end - 0.1 * ear_h)) / (0.1 * ear_h), 0.0, 1.0)  # (a clean cut at the foot)
        hs = np.where(on, np.clip(across, 0.0, 1.0) * foot, 0.0)
        h = np.maximum(h, hs)
        built[side] = int((hs > 0.5).sum())
    if not built or h.max() <= 0:
        return V, {"type": kind, "built": False}
    h = np.clip(sr._mesh_smooth(lap, h, _SB_SMOOTH), 0.0, 1.0)
    thick = sr._HAIR_SHORT * ppm * (1.0 if read.get("fill") == "full" else _SB_TRIMMED)
    return V + (h * thick)[:, None] * n_s, {"type": kind, "built": True, "vertices": built,
                                             "thick_mm": round(thick / ppm * 1000.0, 2)}


def _set_eyes_back(V, F, face_read: dict, shape, ppm: float):
    from scipy.ndimage import binary_dilation

    from . import silhouette_relief as sr

    box = face_read.get("box")
    eyes = [e for e in (face_read.get("eyes") or {}).values() if e.get("open")]
    if box is None or not eyes:
        return V
    near, _ = _raster_front(V, F, shape)
    have = np.isfinite(near)
    surf = np.where(have, -near, np.nan)
    bx0, by0 = int(box[0]), int(box[1])
    off = np.zeros(shape)
    for e in eyes:
        mk = np.asarray(e["opening"], dtype=bool)
        op = np.zeros(shape, dtype=bool)
        h_, w_ = mk.shape
        ys_, xs_ = slice(max(by0, 0), min(by0 + h_, shape[0])), slice(max(bx0, 0), min(bx0 + w_, shape[1]))
        op[ys_, xs_] = mk[ys_.start - by0:ys_.stop - by0, xs_.start - bx0:xs_.stop - bx0]
        op &= have
        lids = binary_dilation(op, iterations=max(int(0.003 * ppm), 2)) & ~op & have
        if op.sum() < 6 or lids.sum() < 6:
            continue
        d = distance_transform_edt(op)
        dome = np.sqrt(np.clip(d / max(float(d.max()), 1e-6), 0.0, 1.0))
        target = float(np.mean(surf[lids])) - sr._EYE_INSET * ppm * (1.0 - _EYE_DOME * dome)
        off = np.where(op, np.clip(np.nan_to_num(surf) - target, 0.0, None), off)
    if not off.any():
        return V
    off = gaussian_filter(off, 0.7)
    px_, py_ = V[:, 0], V[:, 1]
    vis = V[:, 2] <= map_coordinates(np.where(have, near, 1e9), [py_, px_], order=0, mode="nearest") + 1.5
    V = V.copy()
    V[:, 2] += np.where(vis, map_coordinates(off, [py_, px_], order=1, mode="nearest"), 0.0)  # (depth runs toward the back)
    return V


def _eyes_set_back_mm(st: dict, face_read: dict, shape, ppm: float) -> Optional[float]:
    """How far (mm, on the person) the open eyes sit behind the lids round
    them on the model's front -- the smaller of the two eyes (closed eyes
    aren't counted)."""
    from scipy.ndimage import binary_dilation

    box = face_read.get("box")
    eyes = [e for e in (face_read.get("eyes") or {}).values() if e.get("open")]
    if box is None or not eyes:
        return None
    n, _ = _raster_front(st["V"], st["F"], shape)
    bx0, by0 = int(box[0]), int(box[1])
    out = []
    for e in eyes:
        mk = np.asarray(e["opening"], dtype=bool)
        if mk.sum() < 6:
            continue
        op = np.zeros(shape, dtype=bool)
        h_, w_ = mk.shape
        ys_, xs_ = slice(max(by0, 0), min(by0 + h_, shape[0])), slice(max(bx0, 0), min(bx0 + w_, shape[1]))
        op[ys_, xs_] = mk[ys_.start - by0:ys_.stop - by0, xs_.start - bx0:xs_.stop - bx0]
        lids = binary_dilation(op, iterations=max(int(0.003 * ppm), 2)) & ~op
        a, b = op & np.isfinite(n), lids & np.isfinite(n)
        if a.sum() >= 4 and b.sum() >= 4:
            out.append(float(np.mean(-n[b]) - np.mean(-n[a])) / ppm * 1000.0)
    return min(out) if out else None


def _brows_stand_mm(st: dict, face_read: dict, shape, ppm: float) -> Optional[float]:
    """How far (mm, on the person) the eyebrows stand off the skin just round
    them on the model's front -- the reading's brow hair against a ring of
    skin round it."""
    from scipy.ndimage import binary_dilation

    brows = face_read.get("brows") or {}
    box = face_read.get("box")
    if not brows or box is None:
        return None
    bm = np.zeros(shape, dtype=bool)
    bx0, by0 = int(box[0]), int(box[1])
    for b_ in brows.values():
        mk = np.asarray(b_["mask"], dtype=bool)
        h_, w_ = mk.shape
        ys_, xs_ = slice(max(by0, 0), min(by0 + h_, shape[0])), slice(max(bx0, 0), min(bx0 + w_, shape[1]))
        bm[ys_, xs_] |= mk[ys_.start - by0:ys_.stop - by0, xs_.start - bx0:xs_.stop - bx0]
    if bm.sum() < 10:
        return None
    ring = binary_dilation(bm, iterations=max(int(0.006 * ppm), 2)) & ~binary_dilation(bm, iterations=2)
    # (the forehead just above: below them the eye sockets sit lower anyway)
    ring &= np.arange(shape[0])[:, None] < float(np.mean(np.nonzero(bm)[0]))
    n, _ = _raster_front(st["V"], st["F"], shape)
    ok_b, ok_r = bm & np.isfinite(n), ring & np.isfinite(n)
    if ok_b.sum() < 10 or ok_r.sum() < 10:
        return None
    return float(np.mean(-n[ok_b]) - np.mean(-n[ok_r])) / ppm * 1000.0


def _subdivide(V, F, pw):
    """The finer surface for the detail, the part weights carried with it
    (Loop's new points are fixed blends of the old: the weights blend the
    same way)."""
    from trimesh.remesh import subdivide_loop

    V2, F2 = subdivide_loop(V, F, iterations=_SUBDIVIDE)
    cols = []
    for k in range(0, pw.shape[1], 3):
        blk = np.zeros((len(pw), 3))
        blk[:, :min(3, pw.shape[1] - k)] = pw[:, k:k + 3]
        cols.append(subdivide_loop(blk, F, iterations=_SUBDIVIDE)[0][:, :min(3, pw.shape[1] - k)])
    return V2, F2, np.clip(np.hstack(cols), 0.0, None)


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

    # reality check: the photo, read; the body, posed on it
    chk = _Checks()
    Pp = inputs.pose or {}
    fm0 = inputs.face_mesh
    chk.add("1 photo read", "pose points (shoulders, elbows, wrists, hips)",
            sum(f"{sd}_{j}" in Pp for sd in ("left", "right") for j in ("shoulder", "elbow", "wrist", "hip")), lo=4)
    chk.add("1 photo read", "face points found", 1.0 if fm0 is not None else 0.0, lo=1)
    chk.add("1 photo read", "outline's share of the photo", float(mask.mean()), 0.05, 0.95)
    Jr = ref.get("joints_px") or {}
    sw0 = float(np.linalg.norm(np.subtract(Pp["left_shoulder"][:2], Pp["right_shoulder"][:2]))) \
        if "left_shoulder" in Pp and "right_shoulder" in Pp else None
    both = [k for k in Pp if k in Jr]
    if sw0 and both:
        miss = np.mean([np.hypot(*np.subtract(Pp[k][:2], np.asarray(Jr[k])[:2])) for k in both]) / sw0
        chk.add("2 body posed", "joints off the photo's (x shoulder width)", miss, hi=_CHK_JOINT_MISS)
    if fm0 is not None:
        chk.add("2 body posed", "face breadth at the photo's scale", float(np.ptp(np.asarray(fm0)[:468, 0])) / ppm, *_CHK_FACE_M, unit=" m")

    locks = _Locks(chk)
    clothes = (inputs.parts or {}).get("clothes")
    Jn = {k: np.asarray(v, dtype=np.float64) for k, v in Jr.items()}

    # 3. cut where the photo is cut (a bust) -- the first thing done with the
    # 3D body, so everything after works only on the torso the figure will
    # be: no time on legs (or trousers) that won't be there
    rows = np.flatnonzero(mask.any(axis=1))
    cut_off = bust if bust is not None else bool(rows[-1] >= shape[0] - 3)
    cut_row = float(rows[-1]) if cut_off else None
    if cut_off:
        V, F, kept = _cut_below(V, F, cut_row, {"part_w": part_w, "core": body_ref.get("core_depth"), "ear_w": body_ref.get("ear_w")})
        part_w = kept["part_w"]
        part_w = part_w / np.maximum(part_w.sum(axis=1, keepdims=True), 1e-9)
        body_ref = dict(body_ref, part_w=part_w, core_depth=kept["core"], ear_w=kept["ear_w"])
        chk.add("3 cut", "closed solid after the cut", float(trimesh.Trimesh(V, F, process=False).is_watertight), lo=1)

    # 4. skin first: the hands the photo shows, read from its skin (the face
    # is held to its own points when the body grows, step 6)
    hand_regions, hands_read = _skin_hands(names, inputs)
    for sd in ("left", "right"):
        h_ = hands_read.get(sd)
        if isinstance(h_, dict):
            chk.add("4 skin first", f"{sd} hand breadth over shoulder width", h_["breadth_sw"], *_CHK_HAND_SW)
            chk.add("4 skin first", f"{sd} hand clear of the face", h_["on_face"], hi=0.02)

    # 5. the arms follow: each forearm reaching to where the photo's hand
    # begins (its wrist), the two the same length, folded arms one over the
    # other; the body's own hands drawn into the wrists
    V, arms = _fold_arms(V, F, part_w, names, inputs, ref)
    arms["hands"] = hands_read
    V, hand_held, wrist_z = _tuck_hands(V, part_w, names)
    hand_regions = [(H, wrist_z.get(n)) for H, n in hand_regions]
    fl = arms.get("forearm_px") or {}
    if len(fl) == 2:
        chk.add("5 arms", "left forearm over right", fl["left"] / max(fl["right"], 1e-6), *_CHK_FOREARM_LR)
    for sd, L in fl.items():
        if f"{sd}_shoulder" in Jn and f"{sd}_elbow" in Jn:
            up = float(np.linalg.norm(Jn[f"{sd}_elbow"] - Jn[f"{sd}_shoulder"]))
            chk.add("5 arms", f"{sd} forearm over upper arm", L / max(up, 1e-6), *_CHK_FOREARM_UPPER)
    st = {"V": V, "F": F, "pw": part_w}
    V_bare = V.copy()  # (the body before it is dressed: the tailoring measures against it)
    arm_parts = [n for n in ("forearm.L", "forearm.R", "hand.L", "hand.R") if n in names]
    locks.lock("5 arms", "forearms and hands where they were put", lambda s_: _part_centres(s_, names, arm_parts),
               lambda m, m0: m.shape == m0.shape and float(np.abs(m[:, :2] - m0[:, :2]).max()) <= _LOCK_MOVE * ppm, st)

    # 6. the outline: grown out to the photo's (clothes' and hair's room),
    # the face held to its own points, the ears on
    before = st
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
                                      normals=sr._vertex_normals(Vsm, F),
                                      keep=hand_held if face_keep is None else (face_keep | hand_held))
    V[cap_in] = V_cap
    core = body_ref.get("core_depth")
    if core is not None:
        hw = part_w[:, names.index("head")] if "head" in names else 0.0
        m_ = hw * sr._REF_HEAD_MARGIN + (1.0 - hw) * sr._REF_DEPTH_MARGIN
        V[:, 2] = core + m_ * (V[:, 2] - core)
    if ref.get("ears") and body_ref.get("ear_w") is not None and ref.get("head_axes") is not None:
        from .body_reference import place_ears

        V = place_ears(V, np.asarray(body_ref["ear_w"], dtype=np.float64), ref["head_axes"], ref["ears"], faces=F, px_per_m=ppm)
    st = locks.hold("6 outline", before, dict(st, V=V))
    V = st["V"]
    iou = _outline_iou(st, mask)
    chk.add("6 outline", "front outline against the photo's", iou, lo=_CHK_OUTLINE)
    locks.lock("6 outline", "front outline against the photo's", lambda s_: _outline_iou(s_, mask),
               lambda m, m0: m >= m0 - _LOCK_IOU, st)

    # 7. the four sides, understood before anything is built on them
    before = st
    V, plan = plan_four_sides(V, F, part_w, names, inputs, body_ref)
    plan["cut_at_row"] = cut_row
    plan["arms"] = arms
    st = locks.hold("7 four sides", before, dict(st, V=V))
    V = st["V"]
    for sec, info_ in plan["sections"].items():
        if sec == "head" and plan.get("hair_kind") in _LONG_HAIR:
            continue
        have = info_.get("depth_over_shoulders", info_.get("length_over_breadth"))
        chk.add("7 four sides", f"{sec} depth against the human figure's",
                have * plan["fixed"].get(sec, 1.0) / info_["human"], *_CHK_SECTION)
    for sec in ("chest", "waist"):
        locks.lock("7 four sides", f"{sec} depth", lambda s_, sec=sec: _section_depth(s_, names, Jn, mask, sec),
                   lambda m, m0: -_LOCK_DEPTH_LESS <= m / m0 - 1.0 <= _LOCK_DEPTH_MORE, st)

    # 8. the clothes, tailored: confirmed, then each measured against the
    # bare body before it goes on (its room at the chest, waist, sleeves,
    # collar; the tie), its height over the layer under it from those
    fit = None
    if clothes:
        fit = _measure_fit(V_bare, V, F, part_w, names, clothes, Jn, ppm, fm0)
        plan["fit"] = {k: v for k, v in fit.items() if not k.startswith("_")}
        chk.add("8 clothes and hair", "garments confirmed", float(len(clothes.get("found") or {})), lo=1)
        for key, rng, what in (("chest_room_cm", _FIT_CHEST, "room each side at the chest"), ("waist_room_cm", _FIT_WAIST, "room each side at the waist"),
                               ("collar_room_cm", _FIT_COLLAR, "collar's room round the neck"),
                               ("upperarm_sleeve_room_cm", _FIT_SLEEVE, "sleeve's room round the upper arm"),
                               ("forearm_sleeve_room_cm", _FIT_SLEEVE, "sleeve's room round the forearm"), ("tie_width_cm", _FIT_TIE, "tie's width")):
            if fit.get(key) is not None:
                chk.add("8 clothes and hair", what, fit[key], *rng, unit=" cm")

    # ... and put on, from every side: cloth over the chest
    # (a man's always; anyone's under a jacket, suit or coat -- a woman's
    # bust under one keeps its volume, but not its crease or cleavage), a
    # part the photo shows mostly under clothes clothed all round, cloth
    # spanning the gaps to the body, hair over the back of the skull
    before = st
    found = (clothes or {}).get("found") or {}
    stiff = any(k in found for k in _STIFF)
    draped = ((ref.get("report") or {}).get("gender") == "male" or stiff) and "torso" in names
    if draped:
        drape_r = _BUST_DRAPE if (ref.get("report") or {}).get("gender") != "male" else sr._REF_DRAPE
        V = sr._drape_chest(V, F, part_w[:, names.index("torso")], ppm, ref.get("joints_px") or {}, frontmost=True, drape=drape_r)
    V = _cover_unseen(V, F, part_w, names, inputs, ppm, still=cap_in)
    st = locks.hold("8 clothes and hair", before, dict(st, V=V))
    V = st["V"]
    hol = _cloth_hollow_mm(st, names, clothes, ppm)
    if hol is not None:
        chk.add("8 clothes and hair", "deepest hollow under the clothes (95%)", hol, hi=_CHK_CLOTH_HOLLOW_MM, unit=" mm")
        locks.lock("8 clothes and hair", "no body showing through the clothes", lambda s_: _cloth_hollow_mm(s_, names, clothes, ppm),
                   lambda m, m0: m <= max(m0, _CHK_CLOTH_HOLLOW_MM) + _LOCK_HOLLOW_MM, st)

    # 9. the detail, from the front, on what the camera sees (a finer
    # surface for it, the parts carried along)
    ew = body_ref.get("ear_w")
    ear_w9 = None
    if ew is not None and len(ew) == len(V):
        # (the ears carried along too: the side whiskers grow in front of them)
        V, F, pw_all = _subdivide(V, F, np.column_stack([part_w, np.asarray(ew, dtype=np.float64)]))
        part_w, ear_w9 = pw_all[:, :part_w.shape[1]], pw_all[:, part_w.shape[1]:]
    else:
        V, F, part_w = _subdivide(V, F, part_w)
    st = {"V": V, "F": F, "pw": part_w}
    before = st
    locks.lock("9 detail", "no new spikes", _spike_pct, lambda m, m0: m <= m0 + _LOCK_SPIKES, st)
    near, far = _raster_front(V, F, shape)
    cov = np.isfinite(near) & mask
    front = np.where(cov, -near, 0.0)  # toward the camera
    report = {"plan": plan}
    offset = np.zeros(shape)
    depth_detail = np.zeros(shape)
    pop_off, pop_lift, nr = None, None, None
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
    hair_read = parts.get("hair")
    if labels is not None and fm is not None and hair_read:
        # the hair's own pattern, read from the photo: strands, waves, curls,
        # ropes, ending at the hairline (as the outline builder has it)
        try:
            back_h = np.where(cov, -far, 0.0)
            hp, _ = sr._hair_pattern(front, back_h, front, back_h, cov, labels, hair_read, fm, 0, 0, 1.0, 0, ppm)
            offset += np.where(cov, hp, 0.0)
        except Exception as exc:  # noqa: BLE001 - the hair keeps its layer without the pattern
            print(f"Note: hair pattern skipped ({exc}).")
    if clothes_steps and clothes:
        try:
            if fit is not None:
                nr = None
                if fm is not None and sw0 and "left_shoulder" in Pp and "right_shoulder" in Pp:
                    nr = (float(np.asarray(fm)[152, 1]), (Pp["left_shoulder"][1] + Pp["right_shoulder"][1]) / 2.0 + 0.5 * sw0)
                # (the clothes as pieces of their own, step 10; their soft
                # steps kept here only in case the pieces can't be joined)
                pop_off = _clothes_pop(clothes, front, cov, ppm, fit["_H"], nr)
                pop_lift = _clothes_pop(clothes, front, cov, ppm, fit["_H"], nr, sharp=True)
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
            dn_fine = dn - sr._masked_blur(dn, cov, max(_JACKET_DETAIL * max(shape), 1.0))
            dn = dn - sr._masked_blur(dn, cov, max(0.05 * max(shape), 1.0))  # fine detail only: the body has the big shapes
            if clothes:
                from .clothes import GARMENTS, NAMES, WORN

                # on the clothes, their folds, seams and creases deepened so
                # they show on a figure; under a jacket finer still (its
                # lapels, seams, folds -- not the bust and the dip under it
                # the estimate sees through it)
                jk = gaussian_filter(np.isin(clothes["labels"], [NAMES.index(g) + 1 for g in _STIFF if g in NAMES]).astype(np.float64), 3.0) \
                    if stiff else np.zeros(shape)
                gm = gaussian_filter(np.isin(clothes["labels"], [NAMES.index(g) + 1 for g in WORN]).astype(np.float64), 3.0)
                dn = (1.0 + (_POP_FINE - 1.0) * gm) * (jk * dn_fine + (1.0 - jk) * dn)
            half = np.where(cov, (far - near) / 2.0, 0.0)
            # (as big as cloth's detail is -- folds, lapels, seams: scaled by
            # the body's thickness it reached 2-4 cm, cut flat at 1.5)
            M = _DETAIL_MAX * ppm
            rel = M * np.tanh(half * dn * depth_strength / M)
            offset += np.where(cov, rel, 0.0)
            depth_detail = np.where(cov, rel, 0.0)
            # does the body sit as the photo's depth says? (nearer parts
            # should be nearer): rank agreement of the two, coarse
            if cov.sum() > 1000:
                from scipy.stats import spearmanr

                coarse = gaussian_filter(sampled, 0.02 * max(shape))
                sel = cov & (rr % 4 == 0) & (cc % 4 == 0)
                rho = spearmanr(coarse[sel], front[sel]).correlation  # (the estimate: larger is nearer)
                report["depth_order"] = round(float(rho), 3) if np.isfinite(rho) else None
    # (the estimate's sign: Depth Anything gives larger = nearer; checked by the order above)
    from .head_parts import FACE_SKIN as _FS

    body_px = cov & ~(np.isin(labels, (_FS,)) if labels is not None else np.zeros_like(cov))  # (the face's relief is its own)
    # (the photo's own depth detail -- the clothes' tailored heights have
    # their own checks, step 8)
    chk.add("9 detail", "the photo's depth detail off the face, largest move (99%)",
            float(np.percentile(np.abs(depth_detail[body_px]), 99)) / ppm * 1000.0 if body_px.any() else 0.0, hi=_CHK_RELIEF_MM, unit=" mm")
    # what the clothes step locked (no body showing through them) kept in
    # place: where the detail would dig a hollow under a garment deeper than
    # a cloth spans, filled back to that -- there, not the detail cut back
    # everywhere
    if clothes and "torso" in names:
        from scipy.ndimage import binary_erosion, grey_closing

        from .clothes import GARMENTS, WORN, NAMES as CNAMES

        Ft = F[(part_w[:, names.index("torso")][F] > 0.5).all(axis=1)]
        nt, _ = _raster_front(V, Ft, shape)
        on_t = np.isfinite(nt) & cov & (np.abs(nt - np.where(cov, near, np.inf)) < 1.5)  # (the torso is what the camera sees there)
        worn = binary_erosion(np.isin(clothes["labels"], [CNAMES.index(g) + 1 for g in WORN]) & on_t,
                              iterations=int(max(0.02 * ppm, 1)))
        if worn.sum() > 200:
            h_new = gaussian_filter(np.where(on_t, -nt + offset, float(np.nanmin(np.where(on_t, -nt, np.nan)))), _HOLLOW_SCALE * ppm)  # (the folds kept)
            k = max(int(round(0.03 * ppm)), 3)
            # (to half the limit: the detail eases off where the surface turns
            # from the camera, and filled to the limit itself it fell short)
            dig = (grey_closing(h_new, size=(k, k)) - h_new) - 0.5 * _CHK_CLOTH_HOLLOW_MM / 1000.0 * ppm
            offset = offset + np.where(worn & (dig > 0), dig, 0.0)
    # (no push of the hair toward the camera where it meets the forehead:
    # moved toward the camera on a head turning away, it stood as a lip round
    # the hairline that read as a hairband; the hair's layer is laid along
    # the scalp below, thinning to its edge)
    nrm = trimesh.Trimesh(V, F, process=False).vertex_normals
    facing = np.clip((-nrm[:, 2] - _DETAIL_FACING[0]) / (_DETAIL_FACING[1] - _DETAIL_FACING[0]), 0.0, 1.0)
    px, py = V[:, 0], V[:, 1]
    inb = (px >= 0) & (px < shape[1] - 1) & (py >= 0) & (py < shape[0] - 1)
    near_at = map_coordinates(np.where(np.isfinite(near), near, 1e9), [py, px], order=0, mode="nearest")
    visible = inb & (V[:, 2] <= near_at + 1.5)  # frontmost at its pixel (photo px)
    off_v = map_coordinates(offset, [py, px], order=1, mode="nearest")
    V = V.copy()
    V[:, 2] = V[:, 2] - np.where(visible, off_v * facing, 0.0)
    if labels is not None and fm is not None:
        from .head_parts import HAIR, HAT

        # the hairline a step, on the photo's own pixels (on the coarse body
        # the hair's layer eased over a couple of vertices: no hairline, a
        # bare head) -- a layer over the scalp, out along it, not toward the
        # camera (faded as the surface turns, it was a millimetre)
        # (thinning toward the hairline as short hair does: a small step at
        # its edge, the full layer a centimetre in -- not a band)
        hmask = np.isin(labels, (HAIR, HAT)) & cov
        hairy = gaussian_filter(np.where(hmask, np.clip(_HAIR_EDGE_SHARE + (1.0 - _HAIR_EDGE_SHARE) * distance_transform_edt(hmask)
                                                         / (_HAIRLINE_TAPER * ppm), 0.0, 1.0), 0.0), 1.0)
        # (short hair only: longer hair's own volume is the outline's growth
        # -- pushed out along it too, its folds stood out as spikes)
        if hairy.max() > 0.5 and (hair_read or {}).get("kind") not in _LONG_HAIR:
            hv = np.where(visible, map_coordinates(hairy, [py, px], order=1, mode="nearest"), 0.0)
            # ... and over the scalp the front doesn't see (the crown, the
            # back of the head above the nape): left bare there, the layer
            # stopped where the front view does, a ridge over the top of the
            # head from ear to ear that read as a hairband
            if "head" in names and (hair_read or {}).get("kind") not in _HAIR_FLAT:
                head_w = part_w[:, names.index("head")]
                fm_ = np.asarray(fm)
                brow_y, nape_y = float(fm_[[105, 334], 1].mean()), float(fm_[2, 1])
                hd = head_w > 0.5
                if hd.any():
                    zc = float(np.median(V[hd, 2]))
                    zr = float(np.ptp(V[hd, 2]))
                    scalp = hd & ~visible & (V[:, 1] < nape_y) & ((V[:, 1] < brow_y) | (V[:, 2] > zc + 0.15 * zr))
                    # (thinning to the nape as to the front hairline)
                    nape_t = np.clip((nape_y - V[:, 1]) / (_HAIRLINE_TAPER * ppm * 2.0), 0.0, 1.0)
                    hv = np.where(scalp, _HAIR_UNSEEN * (_HAIR_EDGE_SHARE + (1.0 - _HAIR_EDGE_SHARE) * nape_t), hv)
            lap_f = sr._mesh_laplacian(F, len(V))
            hv = np.clip(sr._mesh_smooth(lap_f, hv, _HAIR_SEAM_SMOOTH), 0.0, 1.0)
            # (along the surface's direction smoothed: along each vertex's own,
            # noisy on the fine mesh, the step broke into shards)
            nrm_h = sr._vertex_normals(V, F)
            nrm_h = np.column_stack([sr._mesh_smooth(lap_f, nrm_h[:, j], _HAIR_NORMAL_SMOOTH) for j in range(3)])
            nrm_h /= np.linalg.norm(nrm_h, axis=1, keepdims=True) + 1e-12
            V = V + (hv * sr._HAIR_SHORT * ppm)[:, None] * nrm_h
    moved = np.linalg.norm(V - before["V"], axis=1)
    if labels is not None:
        lr = np.clip(np.rint(py).astype(int), 0, shape[0] - 1)
        lc = np.clip(np.rint(px).astype(int), 0, shape[1] - 1)
        moved = moved[~np.isin(labels[lr, lc], (_FS,))]  # (the face's relief is its own)
    chk.add("9 detail", "largest single move off the face (99.9%)", float(np.percentile(moved, 99.9)) / ppm * 1000.0 if len(moved) else 0.0,
            hi=_CHK_MOVE_MM, unit=" mm")
    st = locks.hold("9 detail", before, dict(st, V=V))
    V = st["V"]
    # the side whiskers as read: a measured place too, after the locks
    sb_read = ((parts.get("face") or {}).get("facial_hair") or {}).get("sideburns")
    if sb_read and ear_w9 is not None and ref.get("head_axes") is not None and fm is not None:
        try:
            V, sb_info = _sideburns_3d(V, F, part_w, names, ear_w9, np.asarray(ref["head_axes"], dtype=np.float64), sb_read, fm, ppm)
            report["sideburns"] = sb_info
            if sb_read.get("type") in ("short", "medium", "long", "joined"):
                chk.add("9 detail", f"side whiskers built as read ({sb_read.get('type')})", float(bool(sb_info.get("built"))), lo=1.0)
        except Exception as exc:  # noqa: BLE001
            print(f"Note: sideburns skipped ({exc}).")
    # the open eyes sit back behind their lids, measured on the surface as it
    # now is (after the locks: a measured place, not detail to ease off) and
    # against the lids themselves -- the body's own eyeballs stood in front
    # of them, so a set-back added on top left the eye level with its lids:
    # at the edge the eyeball's full set-back, doming up toward the middle
    V = _set_eyes_back(V, F, parts.get("face") or {}, shape, ppm)
    st = dict(st, V=V)
    hi_ = _head_iou(st, mask, fm)
    if hi_ is not None:
        chk.add("9 detail", "head and hair outline against the photo's", hi_, lo=_CHK_HEAD_OUTLINE)
    eb = _eyes_set_back_mm(st, parts.get("face") or {}, shape, ppm)
    if eb is not None:
        chk.add("9 detail", "open eyes sit back behind their lids", eb, lo=_CHK_EYE_MM, unit=" mm")
    bh = _brows_stand_mm(st, parts.get("face") or {}, shape, ppm)
    if bh is not None:
        chk.add("9 detail", "eyebrows stand off the skin round them", bh, lo=_CHK_BROW_MM, unit=" mm")

    # 10. the pieces of their own: the clothes the photo shows (each standing
    # over what is under it by its tailored height) and the hands (on what
    # they lie on) -- kept only if the whole stays one closed solid
    garments, made = ([], []) if pop_lift is None else _garment_pieces(V, F, clothes, pop_lift[0], pop_lift[1], ppm, nr)
    hands = _hand_solids(V, F, hand_regions, ppm) if hand_regions else []
    report["garment_pieces"] = made
    joined = None
    for parts_, what in ((garments + hands, "garments and hands"), (hands, "hands")):
        if not parts_:
            continue
        try:
            Vj, Fj = sr._join_solids(V, F, parts_)
            if trimesh.Trimesh(Vj, Fj, process=False).is_watertight:
                joined = what
                V, F = Vj, Fj
                break
        except Exception as exc:  # noqa: BLE001 - try with fewer pieces
            print(f"Note: {what} not joined ({exc}).")
    if garments:
        chk.add("10 pieces", "garment pieces made", float(len(garments)), lo=1)
        for m_ in made:
            chk.add("10 pieces", f"{m_['garment']} piece stands", m_["height_mm"], *_CHK_PIECE_MM, unit=" mm")
        chk.add("10 pieces", "garment pieces joined, one closed solid", float(joined == "garments and hands"), lo=1)
        if joined != "garments and hands" and pop_off is not None:
            # (the pieces couldn't be joined: the clothes' soft steps instead)
            px_, py_ = V[:, 0], V[:, 1]
            nz = map_coordinates(np.where(np.isfinite(near), near, 1e9), [py_, px_], order=0, mode="nearest")
            vis = V[:, 2] <= nz + 1.5
            V = V.copy()
            V[:, 2] -= np.where(vis, map_coordinates(pop_off, [py_, px_], order=1, mode="nearest"), 0.0)
    if hand_regions:
        chk.add("10 pieces", "hands joined, one closed solid", float(joined is not None), lo=1)

    # 11. out, to the figure's frame (trimmed clean at the cut: pinning what
    # went below onto it folded triangles over each other -- thin walls)
    if cut_row is not None:
        V, F, _ = _cut_below(V, F, cut_row, {})
    mesh = trimesh.Trimesh(V, F, process=False)
    pieces = mesh.split(only_watertight=False)
    if len(pieces) > 1:
        mesh = max(pieces, key=lambda p: len(p.faces))
    chk.add("11 out", "closed solid", float(mesh.is_watertight), lo=1)
    chk.add("11 out", "pieces dropped", float(len(pieces) - 1), hi=0)
    from .print_prep import estimate_min_wall_thickness

    mm_ = mesh.copy()
    mm_.apply_scale(target_height_mm / max(float(np.ptp(mm_.vertices[:, 1])), 1e-6))
    if len(mm_.faces) > 60000:
        mm_ = mm_.simplify_quadric_decimation(face_count=60000)
    chk.add("11 out", "thinnest wall", float(estimate_min_wall_thickness(mm_, sample_count=800)), lo=_CHK_WALL_MM, unit=" mm")
    chk.add("11 out", "spikes (outward folds over 60 degrees)", _spike_pct({"V": np.asarray(mesh.vertices), "F": np.asarray(mesh.faces)}),
            hi=_CHK_SPIKES, unit=" %")
    report["checks"] = chk.rows
    report["checks_failed"] = chk.failed()
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
