"""Measured first: the figure built whole in 3D, then checked against the photo once.

The order, each step checked in numbers against what the photo measured (no
render compared with the photo until the end):
  1. the body from the photo's measurements (the body model's build, head
     size and bone lengths as the fit solved them; the hands at the body's
     own proportions -- the fit stretches them to the pose's fingertip
     guesses, which on a hand half out of sight shrank it to its limit);
  2. (head, face, hair, teeth on it)
  3. (the clothes on it, sized by the tailoring's measurements)
  4. the pose read from the photo, applied to all of it at once by the
     skeleton, hands and fingers included;
  5. then the one render of the finished figure, checked against the photo:
     the major verification (lab.score is the same check, in the Lab).
"""
from typing import Optional

import numpy as np
import trimesh

from .mesh_types import BodyMesh

# the fingers at rest, as a hand hangs or lies: each joint bent this much toward the palm (rad)
_FINGER_REST = (0.25, 0.35, 0.25)  # knuckle, middle joint, last joint
_THUMB_REST = 0.15
_CHK_JOINT_MM = 25.0  # mm: a joint off the photo's (set by hand; the pose reader's own spread not yet measured)
_CHK_HAND_MM = 25.0  # mm: a hand point off the photo's
_CHK_SHOULDER = (0.95, 1.05)  # the body's shoulder points apart over the photo's
_CHK_JOINT_MISS = 0.08  # the body's shoulders, elbows, hips off the photo's, x shoulder width (the fit's own limit)
_CHK_HEAD = (0.9, 1.1)  # the head's breadth at the eyes over the face points' (cheek to cheek, x the head's own share)


class Checks:
    """Each step checked in numbers: what it made against what the photo
    measured, the human body, or a printable solid; with the range it
    should be in. A failed check doesn't stop the build; it is reported."""

    def __init__(self):
        self.rows = []

    def add(self, step: str, what: str, value, lo=None, hi=None, unit: str = ""):
        v = None if value is None else float(value)
        ok = v is not None and (lo is None or v >= lo) and (hi is None or v <= hi)
        self.rows.append({"step": step, "what": what, "value": None if v is None else round(v, 4), "lo": lo, "hi": hi,
                          "unit": unit, "ok": bool(ok)})

    def failed(self) -> list:
        return [f"{r['step']}: {r['what']} {r['value']}{r['unit']} (want {r['lo']}..{r['hi']})" for r in self.rows if not r["ok"]]


def solved_body(inputs):
    """The fit's solution (build, bones' turns, camera), solving it again when
    the inputs were read before it was kept."""
    ref = getattr(inputs, "reference", None)
    if ref is not None and ref.get("solved") is not None:
        return ref
    from .body_reference import fit_body_reference

    ref = fit_body_reference(inputs.pose, inputs.face_mesh, inputs.photo_mask, gender=getattr(inputs, "body", None) or "male",
                             parts=inputs.parts)
    if ref is None or not ref.get("matched"):
        raise RuntimeError("No body could be posed on this photo.")
    return ref


def _finger_rest(body: dict, side: str):
    """Bone ids and turns for a resting hand: each finger bent toward the palm
    joint by joint (about the axis across the hand), the thumb a little."""
    labels = body["labels"]
    J = body["joints"]
    j = lambda n: J[labels.index(f"{n}.{side}")]  # noqa: E731
    u = j("finger3-1") - j("wrist")
    u /= np.linalg.norm(u)
    across = j("finger2-1") - j("finger5-1")
    w = np.cross(u, across)
    w /= np.linalg.norm(w)  # out of one face of the hand
    # (the palm is the side the fingertips already lean to in the rest pose)
    tips = np.mean([j(f"finger{k}-3") for k in (2, 3, 4, 5)], axis=0) - np.mean([j(f"finger{k}-1") for k in (2, 3, 4, 5)], axis=0)
    palm = w if tips @ w > 0 else -w
    ids, rvs = [], []
    for k in (2, 3, 4, 5):
        for i, ang in zip((1, 2, 3), _FINGER_REST):
            a = j(f"finger{k}-{i}")
            nxt = j(f"finger{k}-{i + 1}") if i < 3 else a + (a - j(f"finger{k}-{i - 1}"))
            axis = np.cross(nxt - a, palm)
            axis /= np.linalg.norm(axis) + 1e-12
            ids.append(labels.index(f"finger{k}-{i}.{side}"))
            rvs.append(axis * ang)
    a, b = j("finger1-1"), j("finger1-2")
    axis = np.cross(b - a, palm)
    ids.append(labels.index(f"finger1-2.{side}"))
    rvs.append(axis / (np.linalg.norm(axis) + 1e-12) * _THUMB_REST)
    return ids, rvs


def _local_turns(body: dict, Rw_parent_free: np.ndarray, ids, world_rvs):
    """World-frame turns (in the rest pose) as each bone's own local turn: the
    rest pose has no turns, so they are the same -- kept as a function so a
    posed parent can be allowed for later."""
    return np.asarray(world_rvs, dtype=np.float64)


_FACE_FADE = 0.2  # the face's shape eased out to nothing over this share of the face's width past its outline (no edge: no mask)
_TEETH_BACK = 0.004  # m: the teeth's surface this far behind the lips' inner edge (a mouth open in a smile)
_SUBDIVIDE = 2  # the body's mesh split this many times (each a quarter of the triangles' size)
_FACE_SMOOTH = 3  # passes of smoothing over the face's movement (no groove the face points didn't ask for)


def _subdivide(V, F, attrs, times: int):
    """Each triangle split in four at its edges' middles, `times` over (the
    shape kept as it is: a finer mesh to carry the face's detail); each
    per-vertex attribute carried to the new points as their edge's mean.
    The original points keep their indices."""
    for _ in range(times):
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.ravel()
        n = len(V)
        V = np.vstack([V, V[uniq].mean(1)])
        attrs = np.vstack([attrs, attrs[uniq].mean(1)])
        nf = len(F)
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        ab, bc, ca = n + inv[:nf], n + inv[nf:2 * nf], n + inv[2 * nf:]
        F = np.vstack([np.column_stack([a, ab, ca]), np.column_stack([b, bc, ab]), np.column_stack([c, ca, bc]),
                       np.column_stack([ab, bc, ca])])
    return V, F, attrs


def _round_between(V, F, n0: int):
    """The new points (from n0 on) set on the smoothest surface through the
    original ones, which stay where the measurements put them: the skin
    curves between them instead of showing the coarse mesh's flat panels and
    their edges. (Each new point solves the squared neighbour difference = 0,
    the bending-least surface, with the originals held.)"""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    n = len(V)
    L = sp.identity(n, format="csr") - _smoother(F, n)
    B = (L @ L).tocsr()
    new, old = np.arange(n0, n), np.arange(n0)
    Bnn, Bno = B[new][:, new].tocsc(), B[new][:, old]
    out = V.copy()
    out[new] = spsolve(Bnn, -(Bno @ V[old]))
    return out


def _smoother(F, n: int):
    """The mesh's neighbour average as a sparse matrix (each point the mean of
    the points it shares an edge with)."""
    import scipy.sparse as sp

    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    return sp.diags(1.0 / np.maximum(d, 1.0)) @ A


# the face's measures, seen from the front: (name, face point, face point)
# on the face reader's numbering; the same points on the body by lm_vertex
FACE_MEASURES = (
    ("face width, cheek to cheek", 234, 454), ("eyes, outer corner to outer corner", 33, 263),
    ("right eye width", 33, 133), ("left eye width", 362, 263), ("nose width at the wings", 64, 294),
    ("mouth width, corner to corner", 61, 291), ("mouth opening, inner lips", 13, 14),
    ("upper lip to chin", 0, 152), ("nose tip to chin", 1, 152), ("brow line to chin", 9, 152),
    ("forehead top to chin", 10, 152), ("right eye opening", 159, 145), ("left eye opening", 386, 374),
    ("upper lip thickness", 0, 13), ("lower lip thickness", 14, 17), ("chin height, lip to chin", 17, 152),
    ("jaw width at the mouth", 172, 397), ("cheek width under the eyes", 116, 345), ("nose length, brow to tip", 168, 1),
    ("eyes, inner corner gap", 133, 362), ("right brow above the eye", 105, 159), ("left brow above the eye", 334, 386),
    ("brows, inner ends apart", 107, 336), ("brows, outer ends apart", 70, 300), ("nose bridge width", 122, 351),
    ("nose base width", 98, 327), ("nose tip to base", 1, 2), ("nose base to upper lip", 2, 0),
    ("cupid's bow width", 37, 267), ("right mouth corner to upper lip top", 61, 0), ("left mouth corner to upper lip top", 291, 0),
    ("right eye corner to mouth corner", 33, 61), ("left eye corner to mouth corner", 263, 291), ("chin width", 176, 400),
    ("face width at the nose base", 132, 361), ("face width at the ear lobes", 93, 323), ("jaw width at its corners", 58, 288),
    ("jaw width, low", 136, 365), ("jaw width near the chin", 150, 379), ("temples apart", 54, 284), ("forehead width", 67, 297),
)
_SIZE_MEASURES = (0, 1, 7, 9, 10)  # the head's size: widths and heights the face's expression barely moves
_CHK_FACE = (0.95, 1.05)  # model over photo, each measure


def face_measures(Vposed, fm, s: float, lookup=None) -> dict:
    """Each FACE_MEASURES distance, on the photo and on the model (both seen
    from the front, in mm at the body's own scale s px/m): {name: (photo mm,
    model mm)}. Vposed: the body's points (original numbering) posed as the
    photo, in metres."""
    lmv = _face_reference()["lm_vertex"] if lookup is None else lookup
    f = np.asarray(fm, dtype=np.float64)[:, :2]
    out = {}
    for name, a, b in FACE_MEASURES:
        ph = float(np.linalg.norm(f[a] - f[b])) / s * 1000.0
        mo = float(np.linalg.norm(Vposed[lmv[a]][[0, 2]] - Vposed[lmv[b]][[0, 2]])) * 1000.0
        out[name] = (ph, mo)
    return out


_FACE_MEASURE_W = 0.01  # m: each photo measure's row, per its tolerance, against one face point's (in m): a measure off by its tolerance weighs as a point 10 mm off
_FACE_DEPTH_TRUST = 0.3  # the face points' depth counts this much against their place seen from the front (one photo's estimate)
_FACE_REG = 1e-4  # each control's pull back to the average face (m^2 per unit^2): a control moves only when the photo's points ask for it
_FACE_LIMIT = 0.98  # a control this near its end: the photo would want more than the human range gives


_OPPOSITE = {"incr": "decr", "up": "down", "in": "out", "forward": "backward", "convex": "concave",
             "uncompress": "compress", "moreconvex": "moreconcave"}
_SIDEWAYS = ("jaw", "mouth")  # jawLeft/jawRight, mouthLeft/mouthRight: one move, either way
_UNSEEN = ("tongue", "eyeLook")  # the tongue and the eyeballs' turn: no face point sits on them


def _face_controls(fr):
    """The face's controls, each from 0 to 1, and which ones are opposites (incr
    and decr, up and down, jaw left and right, ...): a face takes one of a pair,
    never both. What the face points can't see is left out."""
    ds = dict(zip(map(str, fr["shape_names"]), fr["shape_d"]))
    ds.update(zip(map(str, fr["expr_names"]), fr["expr_d"]))
    names = [n for n in ds if not n.startswith(_UNSEEN)]
    at = {n: i for i, n in enumerate(names)}
    pairs = set()
    for n in names:
        parts = n.split("-")
        for a_, b_ in _OPPOSITE.items():
            if parts[-1] == a_ and "-".join(parts[:-1] + [b_]) in at:
                pairs.add((at[n], at["-".join(parts[:-1] + [b_])]))
        for st in _SIDEWAYS:
            if n == st + "Left" and st + "Right" in at:
                pairs.add((at[n], at[st + "Right"]))
    return np.stack([ds[n] for n in names]), names, sorted(pairs)


def _face_reference():
    from pathlib import Path

    with np.load(Path(__file__).parent / "data" / "face_reference.npz") as d:
        return {k: d[k] for k in d.files}


_SLIDE = np.array([58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288])  # the jaw and chin, below the ear lobes (beside the ears the edge is the ear)
_EDGE_GRAZE = 0.35  # a body point is on the edge seen by the camera when its surface turns this near side-on (|cos|)
_EDGE_RAY = 0.004  # m: how near the point's direction from the face's middle an edge point is taken


def _edge_points(V, F, cand, La, dview, ids):
    """For each photo outline point (ids, in the head's resting frame), the
    body point on the model's own edge seen along dview, in that point's
    direction from the face's middle: the outermost of the side-on points
    near that ray."""
    nrm = np.asarray(trimesh.Trimesh(V, F, process=False).vertex_normals)
    on = cand[np.abs(nrm[cand] @ dview) < _EDGE_GRAZE]
    c = La.mean(0)
    Q = V[on] - c
    Q = Q - np.outer(Q @ dview, dview)
    out = []
    for i in ids:
        u = La[i] - c
        u = u - (u @ dview) * dview
        u = u / max(np.linalg.norm(u), 1e-9)
        along = Q @ u
        off = np.linalg.norm(Q - np.outer(along, u), axis=1)
        sel = (off < _EDGE_RAY) & (along > 0)
        out.append(int(on[sel][np.argmax(along[sel])]) if sel.any() else -1)
    out = np.array(out)
    return np.where(out >= 0, out, _face_reference()["lm_vertex"][ids])


def _fit_face(body: dict, Rw, pw, fm, cam, chk: "Checks"):
    """The face from its own human controls: MakeHuman's face shape targets
    and the Face Units facial actions (CC0), each set between 0 and 1 -- the
    range they were made for, so the face stays a human face -- to bring the
    body's face points (face_reference's lookup) onto the photo's, carried
    into the head's resting frame. Seen from the front they count in full;
    their depth (one photo's estimate) counts _FACE_DEPTH_TRUST. Least
    squares with bounds, each control pulled back to the average face by
    _FACE_REG. The rest-pose vertices, changed; the controls set."""
    from scipy.optimize import lsq_linear

    fr = _face_reference()
    idx, lmv = fr["idx"], fr["lm_vertex"]
    D, names, pairs = _face_controls(fr)
    V = body["verts"].copy()
    h = body["head_bone"]
    s, cx, cy = cam
    fm = np.asarray(fm, dtype=np.float64)[:468]
    Xp = np.column_stack([(fm[:, 0] - cx) / s, fm[:, 2] / s, (cy - fm[:, 1]) / s])
    L = (Xp - pw[h]) @ Rw[h] + body["joints"][h]
    # each face point's body point, and its change under each control
    pos = np.full(len(V), -1)
    pos[idx] = np.arange(len(idx))

    def changes(vs):
        r_ = pos[vs]
        A_ = np.zeros((len(vs), 3, len(names)))
        A_[r_ >= 0] = np.transpose(D[:, r_[r_ >= 0], :], (1, 2, 0))
        return A_

    wts = np.array([1.0, _FACE_DEPTH_TRUST, 1.0])
    px_m = 1.0 / s  # one photo pixel, in m
    w = np.zeros(len(names))
    # the outline below the eyes: no fixed body point (it is wherever the
    # face's edge is from the camera); each turn, the body point on the
    # model's own edge in that point's direction (not the ears)
    head_v = np.flatnonzero((body["part_w"][:, body["part_names"].index("head")] > 0.5)
                            & (np.asarray(body["ear_w"]).reshape(len(V), -1).max(axis=1) < 0.3))
    Fh = body["faces"][np.isin(body["faces"], head_v).all(axis=1)]
    lmv0 = lmv.copy()
    for _ in range(4):
        lmv = lmv0.copy()
        dv = np.zeros_like(V)
        dv[idx] = np.tensordot(w, D, axes=(0, 0))
        Vc = V + dv
        if _ > 0:
            lmv[_SLIDE] = _edge_points(Vc, Fh, head_v, La, dview, _SLIDE)
        A = changes(lmv)
        P = Vc[lmv]
        # the photo's points onto the body's (a turn and a move, no size: the
        # head's size is the photo's, set before): the head's pose and the
        # depth's offset as one photo leaves them
        mx, my = L.mean(0), P.mean(0)
        U, S_, Wt = np.linalg.svd((L - mx).T @ (P - my))
        R = U @ Wt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Wt
        La = (L - mx) @ R + my
        dview = Rw[h][1] @ R  # the camera's line of sight, in the head's resting frame
        dview = dview / np.linalg.norm(dview)
        wpt = np.ones(468)
        if _ > 0:
            wpt[_SLIDE] = 0.0  # (the outline counts across its edge only, below)
        res = (La - V[lmv]) * wts * wpt[:, None]
        M = (A * wts[None, :, None] * wpt[:, None, None]).reshape(-1, len(names))
        if _ > 0:
            # the outline: each point's distance across the edge (outward from the face's middle, seen by the camera)
            c2 = La.mean(0)
            cr, ct = [], []
            for i in _SLIDE:
                u = La[i] - c2
                u = u - (u @ dview) * dview
                u = u / max(np.linalg.norm(u), 1e-9)
                wr = _FACE_MEASURE_W / px_m
                cr.append(wr * (u @ A[i]))
                ct.append(wr * (u @ (La[i] - V[lmv[i]])))
            M = np.vstack([M, np.array(cr)])
            res = np.concatenate([res.ravel(), np.array(ct)])
        else:
            res = res.ravel()
        # the photo's measures, each its own row (seen from the front): the
        # body's distance along the photo's, made the photo's
        mr, mt = [], []
        for _n, ia, ib in FACE_MEASURES:
            dP = (La[ib] - La[ia])[[0, 2]]
            u = (P[ib] - P[ia])[[0, 2]]
            u = u / max(np.linalg.norm(u), 1e-9)
            # each in units of its own tolerance (5%, or one photo pixel): a
            # millimetre on an eye's opening counts as much as on the face's width
            wr = _FACE_MEASURE_W / max(0.05 * float(np.linalg.norm(dP)), px_m)
            mr.append(wr * (u @ (A[ib] - A[ia])[[0, 2]]))
            mt.append(wr * (np.linalg.norm(dP) - u @ (V[lmv[ib]] - V[lmv[ia]])[[0, 2]]))
        M = np.vstack([M, np.array(mr)])
        res = np.concatenate([res, np.array(mt)])
        reg = np.sqrt(_FACE_REG) * np.eye(len(names))
        hi = np.ones(len(names))
        for _k in range(4):
            sol = lsq_linear(np.vstack([M, reg]), np.concatenate([res, np.zeros(len(names))]),
                             bounds=(np.zeros(len(names)), hi))
            w = sol.x
            both = [(i, j) for i, j in pairs if w[i] > 1e-3 and w[j] > 1e-3]
            if not both:
                break
            for i, j in both:  # the weaker of the two closed, and solved again
                hi[j if w[j] < w[i] else i] = 1e-9
    V[idx] = V[idx] + np.tensordot(w, D, axes=(0, 0))
    lmv[_SLIDE] = _edge_points(V, Fh, head_v, La, dview, _SLIDE)
    _DEBUG["face_lookup"] = lmv.copy()
    miss = np.linalg.norm((V[lmv] - La)[:, [0, 2]], axis=1)
    fw = float(np.ptp(La[:, 0]))
    chk.add("2 head", "face points off the photo's, seen from the front (median)", float(np.median(miss)) / max(fw, 1e-6), hi=0.03,
            unit=" of the face's width")
    chk.add("2 head", "face controls at the end of their human range", float((w > _FACE_LIMIT).sum()), unit=" controls")
    used = {n: round(float(x), 2) for n, x in zip(names, w) if x > 0.05}
    return V, used, dict(zip(names, w)), lmv


_DEBUG: dict = {}
_TEETH_NEAR = 8  # head points each tooth point follows
_TEETH_DEPTH = 0.006  # m: the teeth's front surface given this much body behind it (a front tooth is about 7 mm lip to tongue)


def _teeth(body: dict, verts_r, set_w: dict):
    """The base mesh's own upper and lower teeth (face_reference) on this
    body: placed by the head's own fit (the base mesh's neutral head onto this
    head: a turn, a size, a move per axis), moved by the same face controls
    the face took (only the jaw's move them; a smile moves the lips, not the
    teeth). The two bands' points (upper, lower): their curve round the
    mouth; their heights and depth are set from the lips (_teeth_to_photo)."""
    fr = _face_reference()
    from scipy.spatial import cKDTree

    # each tooth point moved as its nearest head points moved (base mesh ->
    # this body): nothing fitted, no size of its own
    X = fr["base_keep_v"].astype(np.float64)
    head = np.flatnonzero(body["part_w"][:, body["part_names"].index("head")] > 0.5)
    d, j = cKDTree(X[head]).query(fr["teeth_v"].astype(np.float64), k=_TEETH_NEAR)
    wn = 1.0 / np.maximum(d, 1e-5)
    wn /= wn.sum(1, keepdims=True)
    tv = fr["teeth_v"] + (wn[:, :, None] * (verts_r[head][j] - X[head][j])).sum(1)
    names = list(map(str, fr["shape_names"])) + list(map(str, fr["expr_names"]))
    dt = np.concatenate([fr["teeth_shape_d"], fr["teeth_expr_d"]])
    w = np.array([set_w.get(n, 0.0) for n in names])
    tv = tv + np.tensordot(w, dt, axes=(0, 0))
    up = fr["teeth_upper"]
    return tv[up], tv[~up]


_TEETH_BEHIND = 0.0015  # m: the upper teeth's front this far behind the lips' inner edge
_TEETH_UNDER_LIP = 0.004  # m: the upper row reaches this far up under the upper lip (hidden)
_TEETH_LOWER = 0.012  # m: the lower row's height at most (a lower front tooth's crown is about 9 mm)
_TEETH_BITE = (0.0015, 0.002)  # m: the lower front teeth behind and above the upper ones' edge (a closed bite)


def photo_teeth(rgb, fm) -> Optional[dict]:
    """The photo's teeth (face_parts.read_face: the white inside the inner
    lips), measured against the photo's own lips: how far down from the upper
    inner lip (13) toward the lower (14) the teeth's edge reaches and where
    they start, and the share of the mouth's inner corners (78, 308) they span
    across. None when no teeth show."""
    from .face_parts import read_face

    rd = read_face(np.asarray(rgb)[..., :3], fm)
    m = (rd or {}).get("mouth", {})
    t = m.get("teeth")
    if t is None or not np.any(t):
        return None
    x0, y0 = rd["box"][:2]
    ys, xs = np.nonzero(t)
    ys, xs = ys + y0, xs + x0
    f = np.asarray(fm, dtype=np.float64)[:, :2]
    y13, y14 = f[13, 1], f[14, 1]
    span = abs(f[308, 0] - f[78, 0])
    mid = np.abs(xs - (f[78, 0] + f[308, 0]) / 2.0) < 0.25 * span
    if mid.sum() < 4 or abs(y14 - y13) < 2:
        return None
    cols = np.unique(xs[mid])
    edge = float(np.median([ys[mid & (xs == c)].max() for c in cols]))
    top = float(np.median([ys[mid & (xs == c)].min() for c in cols]))
    seen = np.unique(xs)
    return {"edge": (edge - y13) / (y14 - y13), "top": (top - y13) / (y14 - y13),
            "width": float(seen.max() - seen.min() + 1) / max(span, 1e-6), "width_px": float(seen.max() - seen.min() + 1)}


def _teeth_to_photo(bands, Vr, seen: Optional[dict], chk: "Checks", s: float):
    """The teeth set from the lips (the body's own face points, resting
    head), each measure against the photo's own lips where it shows them:
    the upper row from _TEETH_UNDER_LIP up under the upper inner lip down to
    its edge (as far down between the lips as the photo's; half way without
    teeth in the photo), its front _TEETH_BEHIND behind that lip, as wide
    across the inner corners as the photo's (within 0.8..1.25 of the band's
    curve); the lower row _TEETH_BITE behind and over that edge, behind the
    lower lip, _TEETH_LOWER tall at most and never below the lower lip's
    bottom. Two closed arches: [(points, triangles), ...]."""
    lmv = _face_reference()["lm_vertex"]
    Bu, Bl = (b.copy() for b in bands)
    p13, p14, p17, c78, c308 = (Vr[lmv[i]] for i in (13, 14, 17, 78, 308))
    # front: behind the upper inner lip
    dy = p13[1] + _TEETH_BEHIND - Bu[:, 1].min()
    Bu[:, 1] += dy
    Bl[:, 1] += dy
    # across: the curve's width in front of the mouth's corners, made the photo's share of them
    cx = (c78[0] + c308[0]) / 2.0
    yc = (c78[1] + c308[1]) / 2.0
    near = Bu[:, 1] <= yc
    want = None if seen is None else seen["width"] * abs(c308[0] - c78[0])
    if want is not None and near.sum() > 4:
        k = float(np.clip(want / max(float(np.ptp(Bu[near, 0])), 1e-6), 0.8, 1.25))
        Bu[:, 0] = cx + k * (Bu[:, 0] - cx)
        Bl[:, 0] = cx + k * (Bl[:, 0] - cx)
        chk.add("2 head", "teeth across in front of the mouth's corners: model / photo",
                float(np.ptp(Bu[near, 0])) / max(want, 1e-6), 0.9, 1.1, unit=" model/photo")
    # down: the edge as far between the lips as the photo's
    share = 0.5 if seen is None else float(np.clip(seen["edge"], 0.0, 1.0))
    z_edge = p13[2] + share * (p14[2] - p13[2])
    upper = _tooth_arch(Bu, p13[2] + _TEETH_UNDER_LIP, z_edge)
    # the lower row: behind and over the upper edge, behind the lower lip
    dyl = max(Bu[:, 1].min() + _TEETH_BITE[0], p14[1] + _TEETH_BEHIND) - Bl[:, 1].min()
    Bl[:, 1] += dyl
    lt = z_edge + _TEETH_BITE[1]
    lb = max(lt - _TEETH_LOWER, p17[2] + _TEETH_BEHIND)
    lower = _tooth_arch(Bl, lt, min(lb, lt - 0.002))
    if seen is not None:
        Vu_ = upper[0]
        fr_ = Vu_[:, 1] < Vu_[:, 1].min() + 0.001  # the built row's front, middle
        got = (Vu_[fr_, 2].min() - p13[2]) / (p14[2] - p13[2])
        chk.add("2 head", "upper teeth edge, share of the way down between the lips: model / photo",
                float(got) / max(share, 1e-6), 0.95, 1.05)
    return [upper, lower]


_TEETH_STATIONS = 28  # along the arch
_TEETH_PROFILE = 20  # around the cross-section
_TEETH_ROUND = 6.0  # the cross-section's corners: a rounded box (superellipse power)


def _tooth_arch(P, z_top: float, z_bot: float):
    """One row of teeth as one clean closed arch: the band's own curve, seen
    from above, taken around the arch (angle about the point behind its
    middle); per station the band's front, and its top and bottom curve
    along the arch (a low-order fit: no zigzag) set to z_top and z_bot at the
    middle; a rounded-box cross-section
    _TEETH_DEPTH deep swept along it, the ends closed."""
    c = np.array([P[:, 0].mean(), P[:, 1].max()])
    th = np.arctan2(P[:, 0] - c[0], c[1] - P[:, 1])
    t = np.linspace(th.min(), th.max(), _TEETH_STATIONS)
    fit = lambda y: np.polyval(np.polyfit(th, y, 4), t)
    r = fit(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]))
    zt, zb = [], []
    edges = np.linspace(th.min(), th.max(), _TEETH_STATIONS + 1)
    for i in range(_TEETH_STATIONS):
        sel = (th >= edges[i]) & (th <= edges[i + 1])
        sel = sel if sel.any() else np.abs(th - t[i]) <= np.sort(np.abs(th - t[i]))[2]
        zt.append(P[sel, 2].max())
        zb.append(P[sel, 2].min())
    zt = np.polyval(np.polyfit(t, zt, 2), t)
    zb = np.polyval(np.polyfit(t, zb, 2), t)
    # the heights set (from the lips), the band's own curve along the arch kept
    mid = len(t) // 2
    zt, zb = z_top + (zt - zt[mid]), z_bot + (zb - zb[mid])
    front = np.column_stack([c[0] + r * np.sin(t), c[1] - r * np.cos(t)])
    out = np.column_stack([np.sin(t), -np.cos(t)])  # outward, seen from above
    a = np.linspace(0, 2 * np.pi, _TEETH_PROFILE, endpoint=False)
    e = 2.0 / _TEETH_ROUND
    cu, sv = np.cos(a), np.sin(a)
    u = 0.5 * (1.0 - np.sign(cu) * np.abs(cu) ** e)  # 0 at the front .. 1 at the back
    v = 0.5 * (1.0 + np.sign(sv) * np.abs(sv) ** e)  # 0 at the bottom .. 1 at the top
    V = []
    for i in range(_TEETH_STATIONS):
        xy = front[i][None, :] - (u * _TEETH_DEPTH)[:, None] * out[i][None, :]
        z = zb[i] + v * (zt[i] - zb[i])
        V.append(np.column_stack([xy, z]))
    V = np.vstack(V)
    n, m = _TEETH_STATIONS, _TEETH_PROFILE
    F = []
    for i in range(n - 1):
        for j in range(m):
            a0, a1 = i * m + j, i * m + (j + 1) % m
            b0, b1 = a0 + m, a1 + m
            F += [[a0, b0, a1], [a1, b0, b1]]
    c0, c1 = len(V), len(V) + 1
    V = np.vstack([V, V[:m].mean(0), V[-m:].mean(0)])
    for j in range(m):
        F += [[c0, j, (j + 1) % m], [c1, (n - 1) * m + (j + 1) % m, (n - 1) * m + j]]
    mesh = trimesh.Trimesh(V, np.array(F), process=False)
    trimesh.repair.fix_normals(mesh)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


_HEAD_TURN_MAX = 0.3  # rad: the head's and the neck top's turn from the body's fit, at most each way
_HEAD_TURN_COST = 0.05  # photo px per degree of turn: a turn is taken only for what it moves the face
_CHK_FACE_PLACE = 1.5  # photo px: the face points' median miss in place


def _head_to_photo(body: dict, bids, rv, fm, cam, chk: "Checks", ph=None, cerv: Optional[int] = None):
    """The head's own turn and one even bend shared by the three neck bones
    (a smooth arc, as a neck moves) -- and, given the build (ph) and the neck
    length's place in it (cerv), the neck's length within the body's bone
    range -- set so the posed face points land on the photo's (seen from the
    front, photo px; the outline left out: it is no fixed point), each degree
    costing _HEAD_TURN_COST px; turns within _HEAD_TURN_MAX of the body fit's.
    The new turns (all bones), and the build (ph, changed when given)."""
    from scipy.optimize import least_squares

    from . import body_reference as br
    from .face_parts import FACE_OVAL

    s, cx, cy = cam
    keep = np.setdiff1d(np.arange(468), FACE_OVAL)
    lmv = _face_reference()["lm_vertex"][keep]
    f = np.asarray(fm, dtype=np.float64)[keep, :2]
    labels = body["labels"]
    at = {labels[b]: i for i, b in enumerate(bids)}
    head = at["head"]
    neck = [at[n] for n in ("neck01", "neck02", "neck03") if n in at]
    rv = np.asarray(rv, dtype=np.float64).copy()
    use_len = ph is not None and cerv is not None
    nx = 7 if use_len else 6
    Vb = np.asarray(body["verts"], dtype=np.float64).copy()

    def turned(x):
        r = rv.copy()
        r[head] = rv[head] + x[:3]
        for i in neck:
            r[i] = rv[i] + x[3:6] / max(len(neck), 1)
        return r

    def built(x):
        if not use_len:
            return body
        p2 = np.asarray(ph, dtype=np.float64).copy()
        p2[cerv] += x[6]
        v, j = br._rest(body, p2, idx=lmv)
        Vb[lmv] = v
        return dict(body, verts=Vb, joints=j)

    def miss(x):
        b_ = built(x)
        V = br._skin(b_, *br._bone_transforms(b_, bids, turned(x)), idx=lmv)
        d = (np.column_stack([cx + s * V[:, 0], cy - s * V[:, 2]]) - f).ravel()
        return np.concatenate([d, _HEAD_TURN_COST * np.degrees(x[:6]) * np.sqrt(len(f))])

    n = len(f)
    lo = np.full(nx, -_HEAD_TURN_MAX)
    hi = np.full(nx, _HEAD_TURN_MAX)
    if use_len:
        lo[6] = -br._STRETCH_RANGE - float(ph[cerv])
        hi[6] = br._STRETCH_RANGE - float(ph[cerv])
    before = float(np.median(np.linalg.norm(miss(np.zeros(nx))[:2 * n].reshape(-1, 2), axis=1)))
    sol = least_squares(miss, np.zeros(nx), bounds=(lo, hi), loss="soft_l1", f_scale=3.0)
    _DEBUG["head_turn_deg" if not use_len else "head_place"] = np.round(np.r_[np.degrees(sol.x[:6]), sol.x[6:]], 3).tolist()
    after = float(np.median(np.linalg.norm(sol.fun[:2 * n].reshape(-1, 2), axis=1)))
    what = "head placed (turn, neck bend and length)" if use_len else "head turned"
    # (before the face is set, the average face's points can't sit on the photo's: recorded, not judged)
    chk.add("4 posed", f"{what}: face points (not the outline) off the photo's in place (median; before {before:.1f})", after,
            hi=None if use_len else _CHK_FACE_PLACE, unit=" photo px")
    chk.add("4 posed", f"{what}: largest turn or bend from before", float(np.degrees(np.abs(sol.x[:6]).max())),
            hi=np.degrees(_HEAD_TURN_MAX) - 0.1, unit=" degrees")
    ph2 = None
    if use_len:
        ph2 = np.asarray(ph, dtype=np.float64).copy()
        ph2[cerv] += float(sol.x[6])
        chk.add("1 body", "neck length against the body model's (bone range)", 1.0 + float(ph2[cerv]),
                1.0 - br._STRETCH_RANGE + 1e-3, 1.0 + br._STRETCH_RANGE - 1e-3)
    return turned(sol.x), ph2


_ARM_TURN_MAX = 0.9  # rad: a forearm's or wrist's turn from the body's fit, at most each way
_ARM_TURN_COST = 0.05  # photo px per degree of turn
_COLLIDE_GRID = 0.0015  # m: the torso's depth grid
_COLLIDE_GAP = 0.004  # m: an arm kept at least this far in front of the torso's surface (a sleeve and a shirt between them)
_COLLIDE_STEPS = (1.0, 5.0, 20.0, 60.0)  # the inside weight, per px of each point inside, raised in these steps
_ARM_REST_MAX = 0.025  # m: crossed forearms rest on what's behind them, the jacket and shirt between: at most this gap beyond _COLLIDE_GAP
_REST_W = 3.0  # the resting gap's weight, per photo px past it
_CURL_RANGE = (-1.0, 2.0)  # the fingers' curl: from straight (x0) to three times the resting bend (x3)
_CURL_COST = 2.0  # photo px per unit of curl change
_ELBOW_STARTS = (-0.5, 0.0, 0.5)  # rad: the upper arm's starting swings about the body's left-right axis
_HAND_STARTS = (-0.2, 0.0, 0.2)  # rad: the upper arm's starting swings, small, when a placed arm's hand is set again
_ARM_SAMPLE = 400  # each arm's points tested against the other arm's capsules
_SDF_GRID = 0.004  # m: the torso's distance field's cells
_ARM_DEPTH_W = 0.5  # a forearm's depth target (from the body: on the torso, or on the other forearm), per photo px
_ARM_HIDDEN_W = 0.2  # a pose point where the photo shows clothes or nothing (a guess at a hidden hand)
_ARM_ELBOW_W = 0.7  # the reader's elbow (matched to the sleeve's outside corner; the sleeve's thickness a guess)
_SLEEVE = 0.012  # m: a jacket sleeve and shirt over the elbow


def _solid_field(V, F, g: float, vg: float, pad: float = 0.03):
    """A closed-ish part of the body as a solid: inside is between its front
    and back surface along the line of sight (y), per (x, z) column on a g
    grid; then its distance field in 3D on a vg grid (m to its surface,
    minus inside). Returns sd(points) -> m, interpolated between cells."""
    from scipy.ndimage import distance_transform_edt, map_coordinates

    from .silhouette_relief import _raster_depth

    used = np.unique(F)
    lo = V[used].min(0) - pad
    hi = V[used].max(0) + pad
    shp = (int((hi[2] - lo[2]) / g) + 2, int((hi[0] - lo[0]) / g) + 2)
    front, back = _raster_depth((V[:, 0] - lo[0]) / g, (V[:, 2] - lo[2]) / g, V[:, 1], F, shp)
    n3 = np.ceil((hi - lo) / vg).astype(int) + 1
    xs = lo[0] + vg * np.arange(n3[0])
    ys = lo[1] + vg * np.arange(n3[1])
    zs = lo[2] + vg * np.arange(n3[2])
    ci = np.clip(((xs - lo[0]) / g).astype(int), 0, shp[1] - 1)
    ri = np.clip(((zs - lo[2]) / g).astype(int), 0, shp[0] - 1)
    fr = front[ri][:, ci].T
    bk = back[ri][:, ci].T
    occ = (ys[None, :, None] > fr[:, None, :]) & (ys[None, :, None] < bk[:, None, :]) & np.isfinite(fr)[:, None, :]
    sdf = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * vg

    def sd(Q):
        return map_coordinates(sdf, ((Q - lo) / vg).T, order=1, mode="nearest")

    sd.front_at = lambda X, Z: front[int(np.clip((Z - lo[2]) / g, 0, shp[0] - 1)), int(np.clip((X - lo[0]) / g, 0, shp[1] - 1))]
    sd.cols = (front, back, lo, g, shp)
    return sd


def _arms_to_photo(body: dict, bone_ids, rv, P: dict, cam, chk: "Checks", labels_px=None):
    """Each arm's turns (upper arm, forearm, wrist) and its hand's finger and
    thumb curl, set so the arm's points land on the photo's pose points (each
    degree costing _ARM_TURN_COST px, within _ARM_TURN_MAX of the body fit's)
    with no arm inside the body or the other arm. One arm at a time: the one
    whose wrist the photo hides lies behind where they cross and goes first,
    against the torso; the other against the torso and it; then the first
    hand again against the second arm (a hand grips the other arm). Each
    against solids made from the body's own mesh (_solid_field): first onto
    the photo's points (and a depth from the body: the arm behind lies on
    the torso, the one in front on it), then the inside weight raised step
    by step (_COLLIDE_STEPS) so the arm leaves the body by the least move.
    Each pose point counts by what the photo shows there: a hand point on
    visible skin and a wrist on skin or a watch in full; a point on clothes
    or nowhere (a guess at a hidden hand) _ARM_HIDDEN_W; the elbow (the
    sleeve's outside corner on a bent arm) _ARM_ELBOW_W. The new turns."""
    from scipy.optimize import least_squares

    from . import body_reference as br
    from .head_parts import ACCESSORY, BODY_SKIN

    s, cx, cy = cam
    labels = body["labels"]
    bone_ids = np.asarray(bone_ids)
    rv = np.asarray(rv, dtype=np.float64).copy()
    pos = {int(b): i for i, b in enumerate(bone_ids)}
    kpn = body["kp_names"]
    pn = body["part_names"]
    PWb = np.asarray(body["part_w"])
    Fb = np.asarray(body["faces"])
    hand_j = {"index": "finger2-2", "pinky": "finger5-2", "thumb": "finger1-3"}
    J = lambda n: labels.index(n)  # noqa: E731
    g, vg = _COLLIDE_GRID, _SDF_GRID

    def seen(n, ok):
        if labels_px is None:
            return True
        x, y = np.round(np.asarray(P[n][:2])).astype(int)
        H, W = labels_px.shape
        win = labels_px[max(y - 3, 0):min(y + 4, H), max(x - 3, 0):min(x + 4, W)]
        return bool(np.isin(win, ok).mean() > 0.5)

    Rw0, pw0 = br._bone_transforms(body, bone_ids, rv)
    V0 = br._skin(body, Rw0, pw0)
    torso = _solid_field(V0, Fb[(PWb[Fb, pn.index("torso")] > 0.5).all(axis=1)], g, vg)

    sides = []
    for side, sd in (("left", "L"), ("right", "R")):
        rows = {b: pos[J(f"{b}.{sd}")] for b in ("upperarm01", "lowerarm01", "wrist") if J(f"{b}.{sd}") in pos}
        if len(rows) < 3:
            continue
        tgt = []
        for n in (f"{side}_elbow", f"{side}_wrist", f"{side}_index", f"{side}_pinky", f"{side}_thumb"):
            if n not in P:
                continue
            if n.endswith("elbow"):
                wt = _ARM_ELBOW_W
            elif n.endswith("wrist"):
                wt = 1.0 if seen(n, (BODY_SKIN, ACCESSORY)) else _ARM_HIDDEN_W
            else:
                wt = 1.0 if seen(n, (BODY_SKIN,)) else _ARM_HIDDEN_W
            tgt.append((n, np.asarray(P[n][:2], dtype=np.float64), wt))
        ua = np.flatnonzero(PWb[:, pn.index(f"upperarm.{sd}")] > 0.5)
        fa = np.flatnonzero(PWb[:, pn.index(f"forearm.{sd}")] > 0.5)
        hd = np.flatnonzero(PWb[:, pn.index(f"hand.{sd}")] > 0.5)
        Jr = body["joints"]
        e_, w_ = Jr[J(f"lowerarm01.{sd}")], Jr[J(f"wrist.{sd}")]
        ax = w_ - e_
        tt = np.clip((body["verts"][fa] - e_) @ ax / (ax @ ax), 0, 1)
        dd = np.linalg.norm(body["verts"][fa] - (e_ + tt[:, None] * ax), axis=1)
        r_e = float(np.median(dd[tt < 0.25])) if (tt < 0.25).any() else float(np.median(dd))
        r_w = float(np.median(dd[tt > 0.75])) if (tt > 0.75).any() else float(np.median(dd))
        arm = np.concatenate([ua, fa, hd])
        sides.append(dict(side=side, sd=sd, rows=rows, tgt=tgt, ua=ua, fa=fa, hd=hd, arm=arm, r_e=r_e, r_w=r_w,
                          fing=[pos[J(f"finger{k}-{j}.{sd}")] for k in (2, 3, 4, 5) for j in (1, 2, 3) if J(f"finger{k}-{j}.{sd}") in pos],
                          thumb=[pos[J(f"finger1-2.{sd}")]] if J(f"finger1-2.{sd}") in pos else []))
    if not sides:
        return rv
    for a_ in sides:
        n = f"{a_['side']}_wrist"
        a_["front"] = n in P and seen(n, (BODY_SKIN, ACCESSORY))
    if len(sides) == 2 and sides[0]["front"] == sides[1]["front"]:
        sides[1]["front"] = not sides[0]["front"]
    sides.sort(key=lambda a_: a_["front"])  # the one behind first

    # skinning of the arms' points (and the wrist keypoints) as one product
    import scipy.sparse as sp

    nbn = len(labels)
    kv_all = np.unique(np.concatenate([body["kp_idx"][kpn.index(f"{a_['side']}_wrist")] for a_ in sides]))
    fast_v = np.unique(np.concatenate([a_["arm"] for a_ in sides] + [kv_all]))
    si, sw = body["skin_idx"][fast_v], body["skin_w"][fast_v]
    Wsk = sp.csr_matrix((sw.ravel(), (np.repeat(np.arange(len(fast_v)), si.shape[1]), si.ravel())), shape=(len(fast_v), nbn))
    vh = np.column_stack([body["verts"][fast_v], np.ones(len(fast_v))])
    at = np.full(len(body["verts"]), -1)
    at[fast_v] = np.arange(len(fast_v))

    def skin_fast(Rw, pw):
        M = np.concatenate([Rw, (pw - np.einsum("bij,bj->bi", Rw, body["joints"]))[:, :, None]], axis=2)
        T = (Wsk @ M.reshape(nbn, 12)).reshape(-1, 3, 4)
        return np.einsum("nij,nj->ni", T, vh)

    par = np.asarray(body["parents"])

    def subtree(roots):
        sub = set(roots)
        for j in range(len(par)):
            if par[j] in sub:
                sub.add(j)
        return sorted(sub)

    def where(n, Rw, pw, Qf, a_):
        kind = n.split("_", 1)[1]
        sd = a_["sd"]
        if kind == "elbow":
            e, sh_, w = pw[J(f"lowerarm01.{sd}")], pw[J(f"upperarm01.{sd}")], pw[J(f"wrist.{sd}")]
            u1, u2 = sh_ - e, w - e
            u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)
            o = -(u1 + u2)
            nn = np.linalg.norm(o)
            return e if nn < 0.3 else e + o / nn * (a_["r_e"] + _SLEEVE)
        if kind == "wrist":
            ki = kpn.index(n)
            return (body["kp_w"][ki][:, None] * Qf[at[body["kp_idx"][ki]]]).sum(0)
        return pw[J(f"{hand_j[kind]}.{sd}")]

    def fit(a_, rows, solids, rv_in, depth, rest, starts, swings=None, bound=None):
        """rest: the solids the forearm rests on (None: no resting); bound:
        each row's largest turn (rad; default _ARM_TURN_MAX)."""
        """One arm's listed bones' turns (+ curls) against the photo and the solids."""
        nt = 3 * len(rows)
        base = br._bone_transforms(body, bone_ids, rv_in)
        only = subtree([bone_ids[i] for i in rows])
        pts = np.concatenate([a_["fa"], a_["hd"]])[::2]  # (the upper arm meets the torso at the shoulder: not tested)
        cw = [0.0]

        def turned(x):
            r = rv_in.copy()
            for k, i in enumerate(rows):
                r[i] = rv_in[i] + x[3 * k:3 * k + 3]
            for i in a_["fing"]:
                r[i] = rv_in[i] * (1.0 + x[nt])
            for i in a_["thumb"]:
                r[i] = rv_in[i] * (1.0 + x[nt + 1])
            return r

        def miss(x):
            Rw, pw = br._bone_transforms(body, bone_ids, turned(x), base=base, only=only)
            Qf = skin_fast(Rw, pw)
            out = []
            for n, t, wt in a_["tgt"]:
                q = where(n, Rw, pw, Qf, a_)
                out += [wt * (cx + s * q[0] - t[0]), wt * (cy - s * q[2] - t[1])]
            e_, w_ = pw[J(f"lowerarm01.{a_['sd']}")], pw[J(f"wrist.{a_['sd']}")]
            for nm_, y in depth:
                q = w_ if nm_ == "wrist" else (0.5 * (e_ + w_) if nm_ == "mid" else pw[J(f"{hand_j[nm_]}.{a_['sd']}")])
                out.append(_ARM_DEPTH_W * (q[1] - y) * s)
            Q = Qf[at[pts]]
            ins = [np.maximum(_COLLIDE_GAP - f(Q), 0.0) * s * cw[0] for f in solids]
            extra = []
            if rest:
                # resting on what's behind it (the torso, or the forearm under it)
                Qr = Qf[at[a_["fa"][::2]]]
                d = np.min([f(Qr) for f in rest], axis=0)
                near_ = np.sort(d)[: max(len(d) // 10, 1)]
                extra.append(_REST_W * max(float(np.mean(near_)) - _COLLIDE_GAP - _ARM_REST_MAX, 0.0) * s)
            return np.concatenate([out, *ins, extra, _ARM_TURN_COST * np.degrees(x[:nt]) * np.sqrt(len(a_["tgt"])),
                                   _CURL_COST * x[nt:] * np.sqrt(len(a_["tgt"]))])

        bnd = np.repeat(bound if bound is not None else [_ARM_TURN_MAX] * len(rows), 3)
        lo = np.r_[-bnd, _CURL_RANGE[0], _CURL_RANGE[0]]
        hi_ = np.r_[bnd, _CURL_RANGE[1], _CURL_RANGE[1]]
        best = None
        swings = _ELBOW_STARTS if swings is None else swings
        for c0 in starts:
            for sw in swings:  # (the elbow back, level or forward: the photo can't tell)
                x0 = np.zeros(nt + 2)
                x0[nt:] = c0
                x0[0] = sw
                cw[0] = 0.0
                sol = least_squares(miss, x0, bounds=(lo, hi_))
                for cw[0] in _COLLIDE_STEPS:
                    sol = least_squares(miss, sol.x, bounds=(lo, hi_))
                _DEBUG.setdefault("arm_fit", []).append((a_["side"], len(rows), c0, sw, round(float(sol.cost), 1)))
                if best is None or sol.cost < best.cost:
                    best = sol
        sol = best
        return turned(sol.x), sol.x[:nt], sol.x[nt:]

    # each forearm's depth from the body: behind, on the torso; in front, on the arm behind
    def depth_for(a_, behind):
        tg = {n.split("_", 1)[1]: t for n, t, _w in a_["tgt"]}
        out = []
        if "wrist" not in tg:
            return out
        spots = [("wrist", tg["wrist"])] + ([("mid", (tg["wrist"] + tg["elbow"]) / 2.0)] if "elbow" in tg else [])
        for nm_, xy in spots:
            X, Z = (xy[0] - cx) / s, (cy - xy[1]) / s
            f = min(fn.front_at(X, Z) for fn in behind)
            if np.isfinite(f):
                r_ = a_["r_w"] if nm_ == "wrist" else 0.5 * (a_["r_e"] + a_["r_w"])
                out.append((nm_, f - _COLLIDE_GAP - r_))
        return out

    # where they cross the forearm in front lies on the one behind; each hand
    # the photo shows lies on what's under it (the other arm), a hidden one under it
    turns = {}
    a_ = sides[0]
    rows = [a_["rows"][b] for b in ("upperarm01", "lowerarm01", "wrist")]
    rv, tx, cx_ = fit(a_, rows, [torso], rv, depth_for(a_, [torso]), rest=[torso], starts=(0.3, 1.2))
    turns[a_["sd"]] = (tx, cx_)
    if len(sides) == 2:
        b_ = sides[1]
        Rw, pw = br._bone_transforms(body, bone_ids, rv)
        Va = br._skin(body, Rw, pw)
        back_fore = _solid_field(Va, Fb[np.isin(Fb, a_["fa"]).all(axis=1)], g, vg)
        rows = [b_["rows"][b] for b in ("upperarm01", "lowerarm01", "wrist")]
        rv, tx, cx_ = fit(b_, rows, [torso, back_fore], rv, depth_for(b_, [torso, back_fore]), rest=[torso, back_fore],
                          starts=(0.3, 1.2))
        turns[b_["sd"]] = (tx, cx_)
        # the hand behind, again: on top of the front arm where the photo shows it, out of it everywhere
        Rw, pw = br._bone_transforms(body, bone_ids, rv)
        Vb = br._skin(body, Rw, pw)
        front_arm = _solid_field(Vb, Fb[np.isin(Fb, b_["arm"]).all(axis=1)], g, vg)
        hd_depth = []
        r_h = 0.5 * a_["r_w"]
        for n, t, wt in a_["tgt"]:
            kind = n.split("_", 1)[1]
            if kind in ("index", "thumb") and wt >= 1.0:
                X, Z = (t[0] - cx) / s, (cy - t[1]) / s
                f = front_arm.front_at(X, Z)
                if np.isfinite(f):
                    hd_depth.append((kind, f - _COLLIDE_GAP - r_h))
        rv, tx2, cx2 = fit(a_, [a_["rows"][b] for b in ("upperarm01", "lowerarm01", "wrist")], [torso, front_arm], rv, hd_depth,
                           rest=[torso], starts=(0.0,), swings=_HAND_STARTS)
        turns[a_["sd"]] = (turns[a_["sd"]][0], turns[a_["sd"]][1] + cx2)

    # checked on the meshes themselves: each arm against the torso and the other arm, as solids
    Rw, pw = br._bone_transforms(body, bone_ids, rv)
    Vall = br._skin(body, Rw, pw)
    mmp = 1000.0 / s
    fields = {a_["sd"]: _solid_field(Vall, Fb[np.isin(Fb, a_["arm"]).all(axis=1)], g, vg) for a_ in sides}
    for a_ in sides:
        side, sd = a_["side"], a_["sd"]
        Qa = Vall[np.concatenate([a_["fa"], a_["hd"]])]
        thr = np.maximum(-torso(Qa), 0.0)
        for b_ in sides:
            if b_ is not a_:
                thr = np.maximum(thr, np.maximum(-fields[b_["sd"]](Qa), 0.0))
        tx, cx_ = turns[sd]
        chk.add("4 posed", f"{side} arm turns from the body's fit (largest)", float(np.degrees(np.abs(tx).max())),
                hi=np.degrees(_ARM_TURN_MAX) - 0.1, unit=" degrees")
        chk.add("4 posed", f"{side} fingers' curl, thumb's curl (x the resting hand's)", float(1.0 + cx_[0]),
                unit=f" ; thumb {1.0 + cx_[1]:.2f}")
        chk.add("4 posed", f"{side} forearm and hand through the skin of the body or the other arm, deeper than the field's "
                f"{vg * 1000:.0f} mm cell (share of their points)", float((thr > vg).mean()), hi=0.0)
        chk.add("4 posed", f"{side} forearm and hand: deepest point through the skin", float(thr.max()) * 1000.0, unit=" mm")
        under = [torso] + [_solid_field(Vall, Fb[np.isin(Fb, o_["fa"]).all(axis=1)], g, vg) for o_ in sides if o_ is not a_ and a_["front"]]
        d = np.min([f(Vall[a_["fa"]]) for f in under], axis=0)
        chk.add("4 posed", f"{side} forearm resting on {'the forearm under it or the torso' if a_['front'] else 'the torso'}"
                ": gap (nearest tenth of its points)", float(np.mean(np.sort(d)[: max(len(d) // 10, 1)])) * 1000.0,
                hi=(_COLLIDE_GAP + _ARM_REST_MAX) * 1000.0 + 2.0, unit=" mm")
        Qf = Vall[np.arange(len(Vall))]
        for n, t, wt in a_["tgt"]:
            ki = kpn.index(n) if n.endswith(("wrist",)) else None
            if ki is not None:
                q = (body["kp_w"][ki][:, None] * Vall[body["kp_idx"][ki]]).sum(0)
            else:
                q = where(n, Rw, pw, None, a_) if n.endswith("elbow") else pw[J(f"{hand_j[n.split('_', 1)[1]]}.{sd}")]
            dmm = float(np.hypot(cx + s * q[0] - t[0], cy - s * q[2] - t[1])) * mmp
            shown = wt >= 1.0
            chk.add("4 posed", f"{n.replace('_', ' ')} off the photo's ({'seen' if shown else 'hidden or sleeve: weighed less'})", dmm,
                    hi=(_CHK_HAND_MM if shown else None), unit=" mm")
    return rv


def build_measured(inputs, target_height_mm: float = 150.0, bust: Optional[bool] = None) -> BodyMesh:
    from . import body_reference as br

    chk = Checks()
    ref = solved_body(inputs)
    sol = ref["solved"]
    body = br._body(str(sol.get("gender") or "male"))
    labels = body["labels"]
    s, cx, cy = sol["cam"]
    ph = np.asarray(sol["ph"], dtype=np.float64).copy()
    nb = len(br._data()["phenotypes"])
    # 1. the body from the measurements; the hands at the body's own proportions
    hand_i = nb + 1 + list(br._STRETCH).index("hand")
    fitted_hand = 1.0 + float(ph[hand_i])
    ph[hand_i] = 0.0
    verts_r, joints_r = br._rest(body, ph)
    bids = np.asarray(sol["bone_ids"], dtype=int)
    rv0 = np.asarray(sol["rv"], dtype=np.float64)
    if inputs.face_mesh is not None:
        # the head's size from the photo's own face: the same face points on
        # both, their distances seen from the front (the fit sized it by the
        # ears, which the face's edge points are not)
        for _ in range(2):
            b0 = dict(body, verts=verts_r, joints=joints_r)
            Vp = br._skin(b0, *br._bone_transforms(b0, bids, rv0))
            fmz = face_measures(Vp, inputs.face_mesh, s)
            names_ = [FACE_MEASURES[i][0] for i in _SIZE_MEASURES]
            ratio = float(np.median([fmz[n][0] / max(fmz[n][1], 1e-6) for n in names_]))
            ph[nb] = float(np.clip((1.0 + ph[nb]) * ratio - 1.0, -br._HEAD_RANGE, br._HEAD_RANGE))
            verts_r, joints_r = br._rest(body, ph)
        chk.add("1 body", "head size against the body model's (human range +/-%d%%)" % round(br._HEAD_RANGE * 100),
                1.0 + float(ph[nb]), 1.0 - br._HEAD_RANGE + 1e-3, 1.0 + br._HEAD_RANGE - 1e-3)
        # the bigger (or smaller) head grows about its joint: its face moves;
        # the neck's length and the head's and neck's turns set again to
        # put it where the photo has it
        cerv = nb + 1 + list(br._STRETCH).index("cervical")
        rv0, ph = _head_to_photo(dict(body), bids, rv0, inputs.face_mesh, (s, cx, cy), chk, ph=ph, cerv=cerv)
        verts_r, joints_r = br._rest(body, ph)
    # 2. the head: the face from its own human controls, set to the photo's
    # face points (the pose read first, to carry them into the head's frame)
    face_used = {}
    if inputs.face_mesh is not None:
        b0 = dict(body, verts=verts_r, joints=joints_r)
        Rw0, pw0 = br._bone_transforms(b0, bids, rv0)
        verts_r0 = verts_r
        verts_r, face_used, face_w, face_lookup = _fit_face(b0, Rw0, pw0, inputs.face_mesh, (s, cx, cy), chk)
        teeth = _teeth_to_photo(_teeth(body, verts_r0, face_w), verts_r, photo_teeth(inputs.rgb, inputs.face_mesh), chk, s)
        b1 = dict(body, verts=verts_r, joints=joints_r)
        # the head turned (its own bone, and the neck's top) until the face
        # points sit where the photo's are -- their places, not only their
        # distances: the body's fit turned it by the ears and nose alone
        rv0, _ = _head_to_photo(b1, bids, rv0, inputs.face_mesh, (s, cx, cy), chk)
        Vp = br._skin(b1, *br._bone_transforms(b1, bids, rv0))
        px_mm = 1000.0 / s  # the photo's resolution here: one pixel, in mm
        for name, (pm, mm) in face_measures(Vp, inputs.face_mesh, s, face_lookup).items():
            tol = max((_CHK_FACE[1] - 1.0) * pm, px_mm)  # 5%, or one photo pixel where that is more
            chk.add("2 head", f"{name}: model {mm:.1f} mm, photo {pm:.1f} mm (+/-{tol:.1f})", mm / max(pm, 1e-6),
                    (pm - tol) / pm, (pm + tol) / pm, unit=" model/photo")
    else:
        teeth = _teeth_to_photo(_teeth(body, verts_r, {}), verts_r, None, chk, s)
    Vt = np.vstack([v for v, _ in teeth])
    Ft = np.vstack([f + sum(len(v) for v, _ in teeth[:i]) for i, (_, f) in enumerate(teeth)])
    teeth = (Vt, Ft)
    # a finer mesh (the shape as it is) to carry the face's and the hands'
    # detail: the skeleton's weights and the parts carried to the new points
    nbones = len(labels)
    Wd = np.zeros((len(verts_r), nbones))
    np.add.at(Wd, (np.repeat(np.arange(len(verts_r)), body["skin_idx"].shape[1]), body["skin_idx"].ravel()), body["skin_w"].ravel())
    ear0 = np.asarray(body["ear_w"], dtype=np.float64).reshape(len(verts_r), -1)
    Vs, Fs, At = _subdivide(verts_r, np.asarray(body["faces"], dtype=np.int64),
                            np.hstack([Wd, body["part_w"], ear0]), _SUBDIVIDE)
    Vs = _round_between(Vs, Fs, len(verts_r))
    np_ = body["part_w"].shape[1]
    Wd, PW, EW = At[:, :nbones], At[:, nbones:nbones + np_], At[:, nbones + np_:]
    top = np.argsort(-Wd, axis=1)[:, :9]
    sw = np.take_along_axis(Wd, top, axis=1)
    sw /= sw.sum(1, keepdims=True)
    # the teeth go with the head
    Vt, Ft = teeth
    nt, n_body = len(Vt), len(Vs)
    t_idx = np.zeros((nt, top.shape[1]), dtype=top.dtype)
    t_idx[:, 0] = body["head_bone"]
    t_w = np.zeros((nt, sw.shape[1]))
    t_w[:, 0] = 1.0
    t_pw = np.zeros((nt, PW.shape[1]))
    t_pw[:, body["part_names"].index("head")] = 1.0
    Vs, Fs = np.vstack([Vs, Vt]), np.vstack([Fs, Ft + n_body])
    top, sw = np.vstack([top, t_idx]), np.vstack([sw, t_w])
    PW, EW = np.vstack([PW, t_pw]), np.vstack([EW, np.zeros((nt, EW.shape[1]))])
    body = dict(body, verts=Vs, joints=joints_r, faces=Fs, skin_idx=top, skin_w=sw, part_w=PW, ear_w=EW)
    P = inputs.pose or {}
    kp_names = body["kp_names"]

    def kps(Vm):
        """The body's keypoints (its regressors, as the fit uses) in photo px."""
        q = (body["kp_w"][..., None] * Vm[body["kp_idx"]]).sum(1)
        return {n: np.array([cx + s * q[i, 0], cy - s * q[i, 2]]) for i, n in enumerate(kp_names)}

    sw_photo = None
    if "left_shoulder" in P and "right_shoulder" in P:
        sw_photo = float(np.linalg.norm(np.subtract(P["left_shoulder"][:2], P["right_shoulder"][:2])))
    chk.add("1 body", "hand at the body's own proportions (the fit had it at)", fitted_hand, unit=" of the body's")

    # 4. the pose, all of it at once: the fit's bones, and each hand at rest
    bone_ids = list(np.asarray(sol["bone_ids"], dtype=int))
    rv = list(rv0)
    for side in "LR":
        ids, rvs = _finger_rest(body, side)
        bone_ids += ids
        rv += list(rvs)
    # the forearms and wrists turned to put the wrists and hands where the photo has them
    rv = list(_arms_to_photo(body, bone_ids, rv, P, (s, cx, cy), chk, (inputs.parts or {}).get("labels")))
    Rw, pw = br._bone_transforms(body, np.asarray(bone_ids), np.asarray(rv))
    V = br._skin(body, Rw, pw)
    Jp = pw  # each bone's joint, posed (m)
    # the pose checked in numbers: the body's points on the photo's
    K = kps(V)
    if sw_photo:
        if "left_shoulder" in K and "right_shoulder" in K:
            chk.add("4 posed", "shoulders apart, body over photo", float(np.linalg.norm(K["left_shoulder"] - K["right_shoulder"])) / sw_photo, *_CHK_SHOULDER)
        # (the elbows: in the arms' own checks, against the sleeve's corner the photo shows)
        miss = [float(np.linalg.norm(K[n] - np.asarray(P[n][:2]))) / sw_photo for n in K
                if n in P and n.endswith(("shoulder", "hip")) and len(P[n]) >= 2]
        if miss:
            chk.add("4 posed", "shoulders, hips off the photo's (median, x shoulder width)", float(np.median(miss)), hi=_CHK_JOINT_MISS)
        # each joint on its own, in mm: where the photo has it
        mmp = 1000.0 / s
        for n in ("left_shoulder", "right_shoulder"):
            if n in K and n in P:
                chk.add("4 posed", f"{n.replace('_', ' ')} off the photo's", float(np.linalg.norm(K[n] - np.asarray(P[n][:2]))) * mmp,
                        hi=_CHK_JOINT_MM, unit=" mm")
    F = np.asarray(body["faces"], dtype=np.int64)
    # in the photo's frame: x right, y down, depth toward the back (px)
    px = np.column_stack([cx + s * V[:, 0], cy - s * V[:, 2], s * V[:, 1]])

    # the ears as the photo shows them: where each starts, where its lobe
    # ends, how far it stands out (measured on the photo's outline)
    ears = ref.get("ears") or {}
    if ears:
        head_axes = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]) @ Rw[body["head_bone"]]
        EWp = np.asarray(body["ear_w"], dtype=np.float64)
        # (to the photo's own outer edge: this head is the photo's width, so
        # no stand-out taken off the head's side)
        ears = {k: (dict(v, stands_px=None) if isinstance(v, dict) else v) for k, v in ears.items()}
        ear_info = {}
        # the photo's ear outline, row by row, where it is skin (not hair over it)
        mask = np.asarray(inputs.photo_mask, dtype=bool)
        labels_ = (inputs.parts or {}).get("labels")
        prof = {}
        for side in "LR":
            t = ears.get(side)
            if not t or t.get("hidden"):
                continue
            pr = []
            for r_ in range(int(t["top"]) + 1, int(t["bottom"])):
                xs_ = np.flatnonzero(mask[r_])
                if not len(xs_):
                    continue
                x_ = int(xs_.max() if side == "L" else xs_.min())
                if labels_ is not None and int(labels_[r_, x_ - 2 if side == "L" else x_ + 2]) in (br._HP_HAIR, br._HP_HAT):
                    continue
                pr.append((float(r_), float(x_)))
            prof[side] = pr
        px = br.place_ears(px, EWp, head_axes, ears, faces=F, px_per_m=s, info=ear_info, profile=prof)
        report_ears = ear_info
        # each ear's outline, row by row from its top to its lobe, against the photo's outline
        mask = np.asarray(inputs.photo_mask, dtype=bool)
        from .silhouette_relief import _raster_depth

        bf_ = F[F.max(1) < n_body]
        nb_, _ = _raster_depth(px[:, 0], px[:, 1], px[:, 2], bf_, mask.shape)
        msil = np.isfinite(nb_)
        for i, side in enumerate("LR"):
            t = ears.get(side)
            if not t or t.get("hidden"):
                continue
            rows_ = [int(r_) for r_, _ in prof.get(side, [])]
            diffs = []
            for r_ in rows_:
                pm_ = np.flatnonzero(mask[r_])
                mm_ = np.flatnonzero(msil[r_])
                if not len(pm_) or not len(mm_):
                    continue
                if side == "L":  # (on the photo's right)
                    diffs.append(float(mm_.max() - pm_.max()))
                else:
                    diffs.append(float(pm_.min() - mm_.min()))
            if diffs:
                d_ = np.array(diffs) * 1000.0 / s
                nm = "left" if side == "L" else "right"
                chk.add("2 head", f"{nm} ear outline, top to lobe where skin, past the photo's (median; + = out)", float(np.median(d_)),
                        -max(1.5 * 1000.0 / s, 2.0), max(1.5 * 1000.0 / s, 2.0), unit=" mm")
                chk.add("2 head", f"{nm} ear outline, top to lobe where skin: worst row", float(d_[np.argmax(np.abs(d_))]), unit=" mm")
                _DEBUG[f"ear_rows_{side}"] = np.round(d_, 1).tolist()
        mm = 1000.0 / s
        for i, side in enumerate("LR"):
            t = ears.get(side)
            core = EWp[:, i] > 0.95
            if not t or t.get("hidden") or core.sum() < 10:
                continue
            q = px[core]
            nm = "left" if side == "L" else "right"
            for what, mo, phv in (("top", q[:, 1].min(), t["top"]), ("lobe", q[:, 1].max(), t["bottom"])):
                chk.add("2 head", f"{nm} ear {what}: model off the photo's", abs(float(mo) - float(phv)) * mm, hi=max(2.0 * mm, 2.0),
                        unit=" mm")
    # the jaw and chin's outline against the photo's: each photo outline point
    # below the ear lobes, to the model's own edge seen from the front
    if inputs.face_mesh is not None:
        from scipy.ndimage import binary_erosion, distance_transform_edt

        from .silhouette_relief import _raster_depth

        JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
        fj = np.asarray(inputs.face_mesh, dtype=np.float64)[JAW, :2]
        k_ = 4.0
        x0_, y0_ = fj.min(0) - 30
        x1_, y1_ = fj.max(0) + 30
        shp = (int((y1_ - y0_) * k_), int((x1_ - x0_) * k_))
        nb_, _ = _raster_depth((px[:, 0] - x0_) * k_, (px[:, 1] - y0_) * k_, px[:, 2] * k_, F[F.max(1) < n_body], shp)
        sil = np.isfinite(nb_)
        # the head's own edge: the body below the jaw (the neck) is inside the outline's rows,
        # so the edge is taken on the head's skin only
        headf = F[(F.max(1) < n_body) & (np.asarray(body["part_w"])[F[:, 0].clip(0, n_body - 1), body["part_names"].index("head")] > 0.5)]
        nh_, _ = _raster_depth((px[:, 0] - x0_) * k_, (px[:, 1] - y0_) * k_, px[:, 2] * k_, headf, shp)
        hs = np.isfinite(nh_)
        edge = hs & ~binary_erosion(hs)
        dist = distance_transform_edt(~edge) / k_
        q = ((fj - [x0_, y0_]) * k_).astype(int)
        q[:, 0] = q[:, 0].clip(0, shp[1] - 1)
        q[:, 1] = q[:, 1].clip(0, shp[0] - 1)
        dj = dist[q[:, 1], q[:, 0]]
        _DEBUG["jaw"] = (hs, q, dj, (x0_, y0_, k_))
        chk.add("2 head", "jaw and chin outline off the photo's (median, front)", float(np.median(dj)) * 1000.0 / s,
                hi=max(1.5 * 1000.0 / s, 2.0), unit=" mm")
        chk.add("2 head", "jaw and chin outline off the photo's (worst point, front)", float(dj.max()) * 1000.0 / s, unit=" mm")
    # the teeth joined to the head: one solid
    from .silhouette_relief import _join_solids

    body_f, teeth_f = F[F.max(1) < n_body], F[F.min(1) >= n_body] - n_body
    if inputs.face_mesh is not None and len(teeth_f):
        # the teeth seen from the front only inside the photo's inner lips
        from matplotlib.path import Path as _Path

        from .face_parts import INNER_LIPS
        from .silhouette_relief import _raster_depth

        fl = np.asarray(inputs.face_mesh, dtype=np.float64)[INNER_LIPS, :2]
        k_ = 4.0
        x0_, y0_ = fl.min(0) - 20
        x1_, y1_ = fl.max(0) + 20
        shp = (int((y1_ - y0_) * k_), int((x1_ - x0_) * k_))
        gx, gy, gz = (px[:, 0] - x0_) * k_, (px[:, 1] - y0_) * k_, px[:, 2] * k_
        nb_, _ = _raster_depth(gx, gy, gz, body_f, shp)
        nt_, _ = _raster_depth(gx, gy, gz, teeth_f + n_body, shp)
        shown = np.isfinite(nt_) & (nt_ < nb_)
        yy, xx = np.mgrid[0:shp[0], 0:shp[1]]
        inside = _Path((fl - [x0_, y0_]) * k_).contains_points(np.column_stack([xx.ravel(), yy.ravel()]) + 0.5,
                                                              radius=1.0 * k_).reshape(shp)
        out_share = float((shown & ~inside).sum()) / max(float(shown.sum()), 1.0)
        _DEBUG["teeth_seen"] = (shown, inside, nb_, nt_)
        chk.add("2 head", "teeth seen from the front outside the photo's lips (share of the teeth seen)", out_share, hi=0.05)
        chk.add("2 head", "teeth seen, share of the photo's opening", float((shown & inside).sum()) / max(float(inside.sum()), 1.0))
    px, F = _join_solids(px[:n_body], body_f, [(px[n_body:], teeth_f)])
    # where the photo is cut (a bust), the figure is cut
    mask = np.asarray(inputs.photo_mask, dtype=bool)
    rows = np.flatnonzero(mask.any(axis=1))
    cut_off = bust if bust is not None else bool(rows[-1] >= mask.shape[0] - 3)
    if cut_off:
        from .body_first import _cut_below

        px, F, _ = _cut_below(px, F, float(rows[-1]), {})
    mesh = trimesh.Trimesh(px, F, process=False)
    chk.add("4 posed", "closed solid", float(mesh.is_watertight), lo=1)
    report = {"checks": chk.rows, "checks_failed": chk.failed(), "built": "measured first", "face_controls": face_used, "ears": locals().get("report_ears")}
    P3 = np.asarray(mesh.vertices)
    xyz = np.column_stack([P3[:, 0] - (P3[:, 0].min() + P3[:, 0].max()) / 2.0, -P3[:, 1], -P3[:, 2]])
    xyz[:, 1] -= xyz[:, 1].min()
    h = float(np.ptp(xyz[:, 1]))
    xyz *= target_height_mm / h if h > 1e-6 else 1.0
    out = BodyMesh(vertices=xyz, faces=np.asarray(mesh.faces))
    out.report = report  # type: ignore[attr-defined]
    out.photo_mesh = (P3, np.asarray(mesh.faces))  # type: ignore[attr-defined]  # in the photo's frame (px), for the check against it
    return out
