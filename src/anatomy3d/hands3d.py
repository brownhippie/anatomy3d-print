"""The hands as hands: the body's own hand (Anny's, at the size the body was
fitted to), placed on the skin the photo shows of it.

The skin is what the photo shows of a hand: where it is, how big, which way
it points. The rest -- its thickness, the knuckles, the fingers' own shape,
what is hidden under a sleeve or an arm -- is a human hand's. So: the body's
hand, its fingers drawn together (a hand at rest lies with them close; the
rest pose spreads them), turned, scaled within what the body's proportions
allow and moved to cover the skin the photo shows of it (what it covers
beyond that must be hidden: under the clothes, not out in the background),
then laid on what it rests on -- its palm on that surface, its back a hand's
thickness out -- with what the photo doesn't show of it behind whatever
covers it there.
"""
from typing import Optional

import numpy as np

_SPREADS = (0.25, 1.0, 1.8, 2.6)  # the fingers' spread tried, shares of the rest pose's (together; as at rest; apart; spread wide)
_PALM_MARGIN = 0.03  # the palm toward the camera only when it fits the skin better than the back by this much
_CURL_KEEP = 0.25  # ... and straightened, keeping this share of its curl (a hand lying on something lies along it)
_THUMBS = ((0.5, 0.5), (1.0, 0.6))  # the thumb tried in along the hand, or out: shares of its rest pose's spread, and of its turn under the palm
_SCALE_RANGE = (0.8, 1.25)  # the skin sets the hand's size within this of the body's own hand (a hand half hidden shows less)
_FIT_EXCESS = 0.5  # the fit: what of the hand lies past the skin (hidden) costs this much per skin's area ...
_PALM_PAST = 0.2  # ... a share of that for its palm (it goes on into the sleeve)
_KNUCKLES = 0.5  # ... the knuckles this share of the way from the wrist to the fingertips
_FIT_OUTSIDE = 1.5  # ... in the background (where nothing could hide it) this much
_FIT_WRIST = 0.3  # ... its wrist away from the body's wrist this much, per _FIT_WRIST_SPAN shoulder widths squared
_FIT_WRIST_SPAN = 0.25
_FOOT_MM = 1.0  # the hand's footprint, for the fit, at this resolution (mm)
_HIDDEN_BEHIND = 0.003  # what the photo doesn't show of a hand lies at least this far behind what covers it (m)
_SINK = 0.002  # the palm this far into what it rests on, so the two join (m)
_REST_SPAN = 0.03  # a hand lies across bumps and gaps narrower than this (m)
_PALM_EASE = 0.002  # the hand's palm side eased over this (m)
_MIN_THICK = 0.006  # ... nor thinner front to back (m)
_FILL_DOWN = 0.015  # the gap under it filled down to what it lies on when no deeper than this (m)
_MIN_W = 0.006  # nothing of the solid narrower than this (m)
_SUBDIVIDE = 1  # the hand's mesh made finer this many times (Loop) before it is laid down


def _ph(report: dict) -> Optional[np.ndarray]:
    """The body's build, as the fit had it (body_reference._rest's vector)."""
    from . import body_reference as br

    d = br._data()
    try:
        build = [float(report["build"][str(k)]) - 0.5 for k in d["phenotypes"]]
        head = float(report.get("head_scale", 1.0)) - 1.0
        bones = [float(report.get("bones", {}).get(k, 1.0)) - 1.0 for k in br._STRETCH]
    except (KeyError, TypeError, ValueError):
        return None
    return np.array(build + [head] + bones, dtype=np.float64)


_TEMPLATES: dict = {}


def template(report: dict, side: str, spread: float = _SPREADS[0], thumb=_THUMBS[0]):
    """The body's own hand `side` ("L" or "R"), closed at the wrist, its
    fingers' spread `spread` of the rest pose's, straightened: (local
    vertices (u: wrist to the middle knuckle, v: across, w: out of its
    back), faces), metres; None when it can't be had."""
    ph = _ph(report)
    if ph is None:
        return None
    key = (side, round(float(spread), 3), tuple(thumb), str(report.get("gender")), tuple(np.round(ph, 4)))
    if key not in _TEMPLATES:
        _TEMPLATES[key] = _template(report, side, spread, ph, thumb)
    return _TEMPLATES[key]


def _template(report, side, spread, ph, thumb):
    import trimesh

    from . import body_reference as br

    # (the hand at the body's own proportions, not the fit's stretch: that
    # was read off the pose's fingertip guesses; the skin sets it here)
    ph = ph.copy()
    ph[len(br._data()["phenotypes"]) + 1 + list(br._STRETCH).index("hand")] = 0.0
    body = br._body(str(report.get("gender") or "male"))
    verts, J = br._rest(body, ph)
    L = body["labels"]
    sw, si = body["skin_w"], body["skin_idx"]
    hb = [i for i, l in enumerate(L) if l.endswith(f".{side}") and l.startswith(("wrist", "finger", "metacarpal"))]
    if not hb:
        return None
    W = np.zeros((len(verts), len(L)))
    np.add.at(W, (np.repeat(np.arange(len(verts)), si.shape[1]), si.ravel()), sw.ravel())
    on = W[:, hb].sum(1) > 0.5
    F = body["faces"]
    Fh = F[on[F].all(axis=1)]
    j = lambda n: J[L.index(f"{n}.{side}")]  # noqa: E731
    O = j("wrist")
    u = j("finger3-1") - O
    u /= np.linalg.norm(u)
    across = j("finger2-1") - j("finger5-1")
    w = np.cross(u, across)
    w /= np.linalg.norm(w)
    v = np.cross(w, u)
    # (the fingertips curl a little toward the palm in the rest pose: the
    # back of the hand is the other way)
    tips = np.mean([j(f"finger{k}-3") for k in (2, 3, 4, 5)], axis=0) - np.mean([j(f"finger{k}-1") for k in (2, 3, 4, 5)], axis=0)
    tip_v = W[:, [L.index(f"finger{k}-3.{side}") for k in (2, 3, 4, 5)]].sum(1) > 0.5
    curl = float(((verts[tip_v] - O) @ w).mean() - (np.mean([j(f"finger{k}-1") for k in (2, 3, 4, 5)], axis=0) - O) @ w) if tip_v.any() else float(tips @ w)
    back = -w if curl > 0 else w

    def ang(d):
        return np.arctan2(d @ v, d @ u)

    Jm = {n: J[L.index(f"{n}.{side}")].copy() for n in [l.split(".")[0] for l in L if l.endswith(f".{side}")]}

    def turn_about(V, wts, p0, axis, angle, joints):
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        c, s_ = np.cos(angle), np.sin(angle)

        def rot(X):
            d = X - p0
            return p0 + d * c + np.cross(axis, d) * s_ + np.outer(d @ axis, axis) * (1.0 - c) if X.ndim == 2 else \
                p0 + d * c + np.cross(axis, d) * s_ + axis * (d @ axis) * (1.0 - c)
        V = V + wts[:, None] * (rot(V) - V)
        for n in joints:
            Jm[n] = rot(Jm[n])
        return V

    def bw(names):
        return W[:, [L.index(f"{n}.{side}") for n in names]].sum(1)

    def tip_of(k):
        sel = bw([f"finger{k}-3"]) > 0.5
        far = (verts[sel] - J[L.index(f"finger{k}-3.{side}")]) @ (J[L.index(f"finger{k}-3.{side}")] - J[L.index(f"finger{k}-2.{side}")])
        return verts[sel][int(np.argmax(far))].copy() if sel.any() else Jm[f"finger{k}-3"].copy()

    def pitch(d):
        return np.arcsin(np.clip((d @ w) / (np.linalg.norm(d) + 1e-12), -1.0, 1.0))

    V = verts.copy()
    # each finger straightened (its curl toward the palm, joint by joint, down
    # to _CURL_KEEP of it), then the thumb in along the hand
    for k in (2, 3, 4, 5):
        Jm[f"tip{k}"] = tip_of(k)
        chain = [f"finger{k}-1", f"finger{k}-2", f"finger{k}-3", f"tip{k}"]
        want = [_CURL_KEEP * pitch(Jm[chain[i + 1]] - Jm[chain[i]]) for i in range(3)]
        for i in range(3):
            d = Jm[chain[i + 1]] - Jm[chain[i]]
            axis = np.cross(w, d - w * (d @ w))
            V = turn_about(V, bw(chain[i:3]), Jm[chain[i]], axis, pitch(d) - want[i], chain[i + 1:])
    Jm["tip1"] = tip_of(1)
    th_names = ["finger1-1", "finger1-2", "finger1-3"]
    d = Jm["tip1"] - Jm["finger1-1"]
    V = turn_about(V, bw(th_names), Jm["finger1-1"], w, (1.0 - thumb[0]) * -np.arctan2(d @ v, d @ u), th_names[1:] + ["tip1"])
    d = Jm["tip1"] - Jm["finger1-1"]
    V = turn_about(V, bw(th_names), Jm["finger1-1"], np.cross(w, d - w * (d @ w)), (1.0 - thumb[1]) * pitch(d), th_names[1:] + ["tip1"])
    verts = V
    J = J.copy()
    for n, p_ in Jm.items():
        if not n.startswith("tip"):
            J[L.index(f"{n}.{side}")] = p_

    mid = ang(j("finger3-3") - j("finger3-1"))
    for k in (2, 4, 5):
        a = ang(j(f"finger{k}-3") - j(f"finger{k}-1"))
        turn = (mid + (a - mid) * spread) - a
        c, s_ = np.cos(turn), np.sin(turn)
        # (about the knuckle, in the hand's plane; each point by how much it
        # belongs to that finger, so the knuckle bends smoothly)
        fw = W[:, [L.index(f"finger{k}-{p}.{side}") for p in (1, 2, 3)]].sum(1)
        p0 = j(f"finger{k}-1")
        d = V - p0
        du, dv, dw = d @ u, d @ v, d @ w
        rot = p0 + np.outer(c * du - s_ * dv, u) + np.outer(s_ * du + c * dv, v) + np.outer(dw, w)
        V = np.where((fw > 0.0)[:, None], V + fw[:, None] * (rot - V), V)
    used = np.unique(Fh)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    m = trimesh.Trimesh(V[used], remap[Fh], process=True)
    m = _cap(m)
    if m is None or not m.is_watertight:
        return None
    if m.volume < 0:
        m.invert()
    for _ in range(_SUBDIVIDE):
        m = m.subdivide_loop(iterations=1)
    d = m.vertices - O
    loc = np.column_stack([d @ u, d @ np.cross(back, u), d @ back])  # (u, v, back: a right-handed frame)
    return loc, np.asarray(m.faces, dtype=np.int64)


def _cap(m):
    """The hand closed at the wrist: its opening fanned to its middle."""
    import trimesh

    F = np.asarray(m.faces)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    open_e = e[cnt[inv.ravel()] == 1]
    if not len(open_e):
        return m
    c = len(m.vertices)
    V = np.vstack([m.vertices, m.vertices[np.unique(open_e)].mean(0)])
    F2 = np.vstack([F, np.column_stack([open_e[:, 1], open_e[:, 0], np.full(len(open_e), c)])])
    return trimesh.Trimesh(V, F2, process=True)


def _footprint(loc, faces, mm: float):
    """The hand seen from its back: a mask in (u, v) at `mm` per pixel, and
    where its (0, 0) lies in it."""
    from skimage.draw import polygon

    uv = loc[:, :2] * 1000.0 / mm
    lo = uv.min(0) - 2.0
    q = uv - lo
    shape = (int(np.ceil(q[:, 1].max())) + 3, int(np.ceil(q[:, 0].max())) + 3)
    img = np.zeros(shape, dtype=bool)
    for f in faces:
        rr, cc = polygon(q[f, 1], q[f, 0], shape)
        img[rr, cc] = True
    return img, -lo


def fit(loc, faces, skin, person, wrist_px, ppm: float, sw_px: float, scale: Optional[float] = None):
    """Where the hand lies in the photo: its turn, size and wrist, so it
    covers the skin the photo shows of it and nothing of it lies out in the
    background. {"theta", "scale", "t", "cover", ...} or None."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
    from scipy.optimize import minimize

    img, org = _footprint(loc, faces, _FOOT_MM)
    soft = gaussian_filter(img.astype(np.float64), 1.5)
    area_mm2 = float(img.sum()) * _FOOT_MM ** 2
    ys, xs = np.nonzero(skin)
    if len(xs) < 50:
        return None
    n_skin = float(len(xs))
    P = np.column_stack([xs, ys]).astype(np.float64)
    # the hand's own points (its footprint), to see where they land
    fy, fx = np.nonzero(img[::2, ::2])
    Fuv = (np.column_stack([fx * 2, fy * 2]) - org) * _FOOT_MM / 1000.0  # metres
    bg_cost = gaussian_filter((~person).astype(np.float64), 1.0)
    skin_soft = gaussian_filter(skin.astype(np.float64), 1.0)
    # (past the skin the photo shows, the fingers cost more than the palm: a
    # palm goes on into the sleeve, but fingers out past the fingertips the
    # photo shows would be seen, unless something lies over them)
    knuckles = _KNUCKLES * float(loc[:, 0].max())
    Fw = np.where(Fuv[:, 0] > knuckles, 1.0, _PALM_PAST)
    cell_m2 = (2.0 * _FOOT_MM / 1000.0) ** 2
    wr = np.asarray(wrist_px, dtype=np.float64) if wrist_px is not None else None
    base_px = ppm  # px per metre at scale 1

    def place(th, sc, t):
        c, s_ = np.cos(th), np.sin(th)
        return c, s_, sc * base_px, t

    def cost(x):
        if scale is not None:
            x = [x[0], np.log(scale), x[1], x[2]]
        th, lsc, tx, ty = x
        sc = float(np.exp(lsc))
        c, s_, k, _ = place(th, sc, None)
        d = (P - (tx, ty)) / k
        # image -> hand: inverse of x = t + k (u e_u + v e_v), e_u = (c, s),
        # e_v = (s, -c) (its back toward the camera: a turn, not a mirror)
        u = d[:, 0] * c + d[:, 1] * s_
        v = d[:, 0] * s_ - d[:, 1] * c
        cu = u * 1000.0 / _FOOT_MM + org[0]
        cv = v * 1000.0 / _FOOT_MM + org[1]
        cov = float(map_coordinates(soft, [cv, cu], order=1, mode="constant").mean())
        hx = tx + k * (Fuv[:, 0] * c + Fuv[:, 1] * s_)
        hy = ty + k * (Fuv[:, 0] * s_ - Fuv[:, 1] * c)
        past = 1.0 - map_coordinates(skin_soft, [hy, hx], order=1, mode="constant")
        excess = float((Fw * past).sum()) * cell_m2 * k * k / n_skin
        out = float(map_coordinates(bg_cost, [hy, hx], order=1, mode="constant", cval=1.0).mean()) * area_mm2 * (k / 1000.0) ** 2 / n_skin
        pen = 0.0
        if wr is not None:
            pen = _FIT_WRIST * (np.hypot(tx - wr[0], ty - wr[1]) / (_FIT_WRIST_SPAN * sw_px)) ** 2
        lim = max(0.0, _SCALE_RANGE[0] - sc, sc - _SCALE_RANGE[1]) * 20.0
        return -cov + _FIT_EXCESS * excess + _FIT_OUTSIDE * out + pen + lim

    centre = P.mean(0)
    cuv = (np.column_stack(np.nonzero(img))[:, ::-1].mean(0) - org) * _FOOT_MM / 1000.0
    starts = []
    sc0 = 1.0 if scale is None else scale
    for th in np.radians(np.arange(0, 360, 15)):
        c, s_ = np.cos(th), np.sin(th)
        k = base_px * sc0
        t = centre - k * np.array([cuv[0] * c + cuv[1] * s_, cuv[0] * s_ - cuv[1] * c])
        x0 = [th, t[0], t[1]] if scale is not None else [th, 0.0, t[0], t[1]]
        starts.append((cost(x0), x0))
    starts.sort(key=lambda z: z[0])
    best = None
    for _, x0 in starts[:3]:
        r = minimize(cost, x0, method="Powell", options={"xtol": 1e-3, "ftol": 1e-5, "maxfev": 3000})
        if best is None or r.fun < best.fun:
            best = r
    bx = list(best.x) if scale is None else [best.x[0], np.log(scale), best.x[1], best.x[2]]
    th, lsc, tx, ty = bx
    sc = float(np.exp(lsc))
    c, s_ = np.cos(th), np.sin(th)
    k = sc * base_px
    d = (P - (tx, ty)) / k
    u, v = d[:, 0] * c + d[:, 1] * s_, d[:, 0] * s_ - d[:, 1] * c
    cov = float(map_coordinates(img.astype(np.float64), [v * 1000.0 / _FOOT_MM + org[1], u * 1000.0 / _FOOT_MM + org[0]], order=0, mode="constant").mean())
    return {"theta": float(th), "scale": round(sc, 3), "t": (float(tx), float(ty)), "cover": round(cov, 3), "cost": round(float(best.fun), 4)}


def _palm_side(loc, faces=None, top: bool = False, mm: float = 1.0):
    """Each point's palm side: the hand's own palm-side surface under it (its
    lowest w there; top: its back, the highest), from the hand's surface
    drawn from its back, eased over _PALM_EASE. (Taken point by point on a
    grid, most cells held only the back or only the palm: the hand came out
    a fifth of its thickness, crumpled.)"""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

    from .body_first import _raster_front

    q = loc[:, :2] * 1000.0 / mm
    lo = q.min(0) - 3.0
    # (drawn with depth w: the near surface is the lowest w at each point,
    # the far one the highest)
    Vr = np.column_stack([q[:, 0] - lo[0], q[:, 1] - lo[1], loc[:, 2]])
    shape = (int(np.ceil(Vr[:, 1].max())) + 4, int(np.ceil(Vr[:, 0].max())) + 4)
    near, far = _raster_front(Vr, faces, shape)
    surf = far if top else near
    have = np.isfinite(surf)
    if not have.any():
        return np.full(len(loc), loc[:, 2].max() if top else loc[:, 2].min())
    _, (iy, ix) = distance_transform_edt(~have, return_indices=True)
    g = gaussian_filter(surf[iy, ix], _PALM_EASE * 1000.0 / mm)
    return map_coordinates(g, [Vr[:, 1], Vr[:, 0]], order=1, mode="nearest")


def place(report: dict, side: str, skin, person, wrist_px, ppm: float, sw_px: float, scale: Optional[float] = None):
    """The hand on its skin: each spread of the fingers (_SPREADS), its back
    or its palm toward the camera, fitted (fit); the one that fits best (the
    back unless the palm fits better by _PALM_MARGIN). (local vertices,
    faces, placement) or None."""
    best = None

    def one(spread, thumb, palm):
        tpl = template(report, side, spread, thumb)
        if tpl is None:
            return None
        loc, faces = tpl
        # (the palm toward the camera: turned half round along the hand)
        l_ = loc * np.array([1.0, -1.0, -1.0]) if palm else loc
        p_ = fit(l_, faces, skin, person, wrist_px, ppm, sw_px, scale)
        if p_ is None:
            return None
        return (p_["cost"] + (_PALM_MARGIN if palm else 0.0), l_, faces, dict(p_, spread=spread, thumb=list(thumb), palm=palm))

    # (the fingers' spread and the thumb with its back toward the camera,
    # then that hand turned palm forward)
    for spread in _SPREADS:
        for thumb in _THUMBS:
            got = one(spread, thumb, False)
            if got is not None and (best is None or got[0] < best[0]):
                best = got
    if best is not None:
        got = one(best[3]["spread"], tuple(best[3]["thumb"]), True)
        if got is not None and got[0] < best[0]:
            best = got
    return None if best is None else best[1:]


def lay(loc, faces, placed: dict, skin, rest, have, ppm: float, wrist_depth: Optional[float] = None, print_px: float = 0.0):
    """The hand laid where `fit` put it: its palm on what it rests on (across
    gaps narrower than _REST_SPAN), its back its own thickness out; what the
    photo doesn't show of it behind whatever covers it there. (vertices,
    faces) in photo px (x right, y down, depth back), or None."""
    import trimesh
    from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

    th, sc = placed["theta"], placed["scale"]
    tx, ty = placed["t"]
    c, s_ = np.cos(th), np.sin(th)
    k = sc * ppm
    x = tx + k * (loc[:, 0] * c + loc[:, 1] * s_)
    y = ty + k * (loc[:, 0] * s_ - loc[:, 1] * c)
    span = max(int(round(_REST_SPAN * ppm)), 3)
    top = gaussian_filter(maximum_filter(rest, size=span), span / 2.0)
    under = map_coordinates(top, [y, x], order=1, mode="nearest")
    if wrist_depth is not None:
        # (where nothing of the body is behind it -- a hand hanging beside
        # it -- its middle at its own wrist's depth)
        over = map_coordinates(gaussian_filter(have.astype(np.float64), 2.0), [y, x], order=1, mode="constant")
        half_t = 0.5 * float(np.median(_palm_side(loc, faces, top=True) - _palm_side(loc, faces))) * k
        under = over * under + (1.0 - over) * (-wrist_depth - half_t)
    # (each part by its own palm side: the fingers slope from the knuckles,
    # and laid off the hand's lowest point they stood clear of the arm)
    low, high = _palm_side(loc, faces), _palm_side(loc, faces, top=True)
    h = (loc[:, 2] - low) * k
    z = under - _SINK * ppm + h  # toward the camera
    # what the photo doesn't show of it: behind what covers it there -- each
    # part of it moved back whole, its thickness kept (pressed flat point by
    # point it crumpled), eased across where the photo stops showing it
    H_, W_ = skin.shape
    xi, yi = np.clip(np.rint(x).astype(int), 0, W_ - 1), np.clip(np.rint(y).astype(int), 0, H_ - 1)
    front = gaussian_filter(np.where(have, rest, rest.min()), 1.5)
    need = np.zeros(skin.shape)
    cnt = np.zeros(skin.shape)
    back_by = np.clip(under + (high - low) * k - _SINK * ppm - (map_coordinates(front, [y, x], order=1, mode="nearest") - _HIDDEN_BEHIND * ppm), 0.0, None)
    np.maximum.at(need, (yi, xi), back_by)
    np.add.at(cnt, (yi, xi), 1.0)
    hidden = (cnt > 0) & ~skin
    need = np.where(hidden, need, 0.0)
    from scipy.ndimage import grey_dilation
    need = gaussian_filter(grey_dilation(need, size=(5, 5)), 2.0)
    z = z - map_coordinates(need, [y, x], order=1, mode="nearest")
    V = np.column_stack([x, y, -z])
    # (as a solid: its front and its back as the camera sees them, filled
    # between -- laid over a curved arm the hand's own surface folded through
    # itself in places, and it would not join)
    from scipy.ndimage import binary_fill_holes, binary_opening

    from . import silhouette_relief as sr
    from .body_first import _raster_front

    near, far = _raster_front(V, faces, skin.shape)
    P = np.isfinite(near)
    r = max(int(round(max(_MIN_W * ppm, print_px) / 2.0)), 1)
    P = binary_opening(binary_fill_holes(P), structure=np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r)
    if P.sum() < 20:
        return None
    from scipy.ndimage import distance_transform_edt

    got = np.isfinite(near) & np.isfinite(far)
    _, (iy, ix) = distance_transform_edt(~got, return_indices=True)  # (the holes filled in: their neighbours' depth)
    fr = np.where(P, -near[iy, ix], 0.0)
    bk = np.where(P, -far[iy, ix], 0.0)
    # (no edge thinner than a print holds -- front and back meet at its
    # outline -- and no thin gap between it and what it lies on: down to that
    # where it is close)
    bk = np.minimum(bk, fr - max(_MIN_THICK * ppm, print_px))
    bk = np.where(P & (bk - rest < _FILL_DOWN * ppm), np.minimum(bk, rest - _SINK * ppm), bk)
    h0 = float(bk[P].min()) - 3.0
    zs = np.arange(0.0, float(fr[P].max()) - h0 + 3.0, 1.0)
    piece = sr._layer_solid(fr - h0, bk - h0, P, zs, 1.0, 1.0, True)
    if piece is None:
        return None
    pv, pf = piece
    pv = np.column_stack([pv[:, 2], pv[:, 1], -(pv[:, 0] + h0)])
    if trimesh.Trimesh(pv, pf, process=False).volume < 0:
        pf = pf[:, ::-1]
    return pv, pf
