"""Garments as they are cut: flat pattern pieces with named edges, and the
seams that sew them together. One design per kind of garment, sized by
settings (a chart's length, chest, sleeve) and by the body it is cut for;
the cloth step (cloth.py) sews and drapes the pieces on the body.

`top()` is the design for knitted tops: a front and a back, two set-in
sleeves, and by its settings a hood (two side panels, a centre-back seam), a
front pocket, rib bands at the hem and cuffs. A tee, a crewneck sweatshirt
and a hoodie are this one design with different settings.

Pieces are in metres, flat: x across, y up. Each keeps its 2D shape: the
cloth is cut to it (its rest lengths) and it is where a print goes (its UVs).
"""
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay


def bezier(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def line(a, b, n=2):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


def arclen(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def resample(P, step):
    """a polyline at about `step` apart, its ends kept"""
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(int(np.ceil(d[-1] / step)), 1)
    s = np.linspace(0, d[-1], n + 1)
    return np.column_stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])])


@dataclass
class Piece:
    name: str
    edges: dict  # edge name -> its polyline (in order round the outline)
    order: list  # edge names round the outline, counter-clockwise
    pts: np.ndarray = None  # the cloth's points (2D)
    tris: np.ndarray = None
    edge_idx: dict = field(default_factory=dict)  # edge name -> its points' indices, in order

    def cut(self, step=0.014):
        """the piece as cloth: its outline at `step`, the inside filled the same, in triangles"""
        outline, owner = [], []
        for nm in self.order:
            P = resample(self.edges[nm], step)
            outline.append(P[:-1])
            owner += [nm] * (len(P) - 1)
        O = np.vstack(outline)
        path = Path(O)
        lo, hi = O.min(0), O.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step * 0.866))
        gx = gx + (np.arange(gx.shape[0]) % 2)[:, None] * step / 2
        G = np.column_stack([gx.ravel(), gy.ravel()])
        G = G[path.contains_points(G)]
        from scipy.spatial import cKDTree

        if len(G):
            dmin, _ = cKDTree(O).query(G)
            G = G[dmin > 0.6 * step]
        pts = np.vstack([O, G])
        tri = Delaunay(pts).simplices
        c = pts[tri].mean(1)
        keep = path.contains_points(c)
        a = np.linalg.norm(pts[tri[:, 1]] - pts[tri[:, 0]], axis=1)
        b = np.linalg.norm(pts[tri[:, 2]] - pts[tri[:, 1]], axis=1)
        cc = np.linalg.norm(pts[tri[:, 0]] - pts[tri[:, 2]], axis=1)
        keep &= np.maximum(np.maximum(a, b), cc) < 2.2 * step
        self.pts, self.tris = pts, tri[keep]
        # each edge's points, in order, its end shared with the next edge
        n = len(O)
        owner = np.array(owner)
        for k, nm in enumerate(self.order):
            idx = list(np.flatnonzero(owner == nm))
            idx.append((idx[-1] + 1) % n)
            self.edge_idx[nm] = np.array(idx)
        return self

    def length(self, nm):
        return arclen(self.edges[nm])


def top(chart: dict, body: dict, hood=False, pocket=False, rib=True, sleeve="long"):
    """The knitted top's pieces and seams.

    chart: the garment's own measurements (m): length (high point of the shoulder to the hem, rib
    included), chest_flat (across the chest laid flat), sleeve (shoulder seam to the cuff's edge).
    body: the body it is cut for (m): shoulder_half (centre to the shoulder point), neck_round,
    armhole_depth (high point of the shoulder to under the arm), biceps, wrist.
    Returns (pieces, seams): seams are (piece, edge, reversed?) pairs sewn together; a list of edges on
    one side is sewn end to end against the other side.
    """
    W = chart["chest_flat"]
    band = 0.065 if rib else 0.0
    H = chart["length"] - band
    ad = body["armhole_depth"]
    sx = body["shoulder_half"] + 0.02  # (a knitted top's shoulder seam sits a little past the shoulder point)
    nw = body["neck_round"] / (2 * np.pi) + 0.012  # half the neck opening across
    slope = 0.045
    pieces = {}

    def body_piece(name, drop):
        e = {}
        e["hem"] = line((-W / 2, 0), (W / 2, 0))
        e["side_p"] = line((W / 2, 0), (W / 2, H - ad))
        e["arm_p"] = bezier((W / 2, H - ad), (W / 2 - 0.035, H - ad + 0.07), (sx + 0.005, H - slope - 0.09), (sx, H - slope))
        e["shoulder_p"] = line((sx, H - slope), (nw, H))
        e["neck_p"] = bezier((nw, H), (nw, H - drop * 0.65), (nw * 0.55, H - drop), (0, H - drop))
        e["neck_m"] = bezier((0, H - drop), (-nw * 0.55, H - drop), (-nw, H - drop * 0.65), (-nw, H))
        e["shoulder_m"] = line((-nw, H), (-sx, H - slope))
        e["arm_m"] = bezier((-sx, H - slope), (-sx - 0.005, H - slope - 0.09), (-W / 2 + 0.035, H - ad + 0.07), (-W / 2, H - ad))
        e["side_m"] = line((-W / 2, H - ad), (-W / 2, 0))
        return Piece(name, e, ["hem", "side_p", "arm_p", "shoulder_p", "neck_p", "neck_m", "shoulder_m", "arm_m", "side_m"])

    pieces["front"] = body_piece("front", 0.085)
    pieces["back"] = body_piece("back", 0.022)

    # sleeves: the cap's edge as long as the armhole it is set into (a little more: eased in)
    arm_len = pieces["front"].length("arm_p") + pieces["back"].length("arm_p")
    Wb = body["biceps"] + 0.07
    Wc = (body["wrist"] + 0.10) if sleeve == "long" else Wb * 0.9
    Ls = (chart["sleeve"] - band) if sleeve == "long" else chart["sleeve"]

    def cap(ch, n=30):
        x = np.linspace(Wb / 2, 0, n)
        s = 1 - x / (Wb / 2)
        return np.column_stack([x, Ls - ch + ch * np.sin(np.pi / 2 * s) ** 1.4])

    lo_, hi_ = 0.02, 0.4
    for _ in range(40):  # the cap's height that makes its edge the armhole's length
        ch = (lo_ + hi_) / 2
        if 2 * arclen(cap(ch)) < arm_len * 1.03:
            lo_ = ch
        else:
            hi_ = ch
    for side in ("L", "R"):
        e = {}
        cp = cap(ch)
        e["cuff"] = line((-Wc / 2, 0), (Wc / 2, 0))
        e["under_p"] = line((Wc / 2, 0), (Wb / 2, Ls - ch))
        e["cap_p"] = cp
        e["cap_m"] = (cp * [-1, 1])[::-1]
        e["under_m"] = line((-Wb / 2, Ls - ch), (-Wc / 2, 0))
        pieces[f"sleeve_{side}"] = Piece(f"sleeve_{side}", e, ["cuff", "under_p", "cap_p", "cap_m", "under_m"])

    seams = [
        ([("front", "side_p", False)], [("back", "side_p", False)]),
        ([("front", "side_m", False)], [("back", "side_m", False)]),
        ([("front", "shoulder_p", False)], [("back", "shoulder_p", False)]),
        ([("front", "shoulder_m", False)], [("back", "shoulder_m", False)]),
        # left sleeve (+x): its x>0 half faces the front
        ([("sleeve_L", "cap_p", False)], [("front", "arm_p", False)]),
        ([("sleeve_L", "cap_m", True)], [("back", "arm_p", False)]),
        ([("sleeve_L", "under_p", False)], [("sleeve_L", "under_m", True)]),
        # right sleeve (-x): its x>0 half faces the back
        ([("sleeve_R", "cap_p", False)], [("back", "arm_m", True)]),
        ([("sleeve_R", "cap_m", True)], [("front", "arm_m", True)]),
        ([("sleeve_R", "under_p", False)], [("sleeve_R", "under_m", True)]),
    ]
    if rib:
        hem_round = 2 * W
        Lh = 0.86 * hem_round
        pieces["hem_band"] = Piece("hem_band", {"bottom": line((0, 0), (Lh, 0)), "end_p": line((Lh, 0), (Lh, band)),
                                                "top": line((Lh, band), (0, band)), "end_m": line((0, band), (0, 0))},
                                   ["bottom", "end_p", "top", "end_m"])
        seams.append(([("hem_band", "top", True)], [("front", "hem", False), ("back", "hem", True)]))
        seams.append(([("hem_band", "end_p", False)], [("hem_band", "end_m", True)]))
        for side in ("L", "R"):
            Lc = body["wrist"] + 0.03
            pieces[f"cuff_{side}"] = Piece(f"cuff_{side}", {"bottom": line((0, 0), (Lc, 0)), "end_p": line((Lc, 0), (Lc, band)),
                                                            "top": line((Lc, band), (0, band)), "end_m": line((0, band), (0, 0))},
                                           ["bottom", "end_p", "top", "end_m"])
            seams.append(([(f"cuff_{side}", "top", True)], [(f"sleeve_{side}", "cuff", False)]))
            seams.append(([(f"cuff_{side}", "end_p", False)], [(f"cuff_{side}", "end_m", True)]))
    if hood:
        Lb = pieces["front"].length("neck_p") + pieces["back"].length("neck_p")
        Hh = body.get("hood_height", 0.36)
        for side in ("p", "m"):
            e = {"bottom": line((0, 0), (Lb, 0)),
                 "back": np.vstack([bezier((Lb, 0), (Lb + 0.05, 0.10), (Lb + 0.02, 0.26), (Lb - 0.10, Hh - 0.01), 20),
                                    bezier((Lb - 0.10, Hh - 0.01), (Lb - 0.18, Hh + 0.01), (0.06, Hh + 0.01), (0, Hh - 0.03), 20)[1:]]),
                 "front": line((0, Hh - 0.03), (0, 0))}
            pieces[f"hood_{side}"] = Piece(f"hood_{side}", e, ["bottom", "back", "front"])
        # the hood's foot round the neckline: centre front, over the side of the neck, to the centre back
        seams.append(([("hood_p", "bottom", False)], [("front", "neck_p", True), ("back", "neck_p", False)]))
        seams.append(([("hood_m", "bottom", False)], [("front", "neck_m", False), ("back", "neck_m", True)]))
        seams.append(([("hood_p", "back", False)], [("hood_m", "back", False)]))
    if pocket:
        pw, ph, pt = 0.30, 0.19, 0.21
        pieces["pocket"] = Piece("pocket", {"bottom": line((-pw / 2, 0), (pw / 2, 0)), "side_p": line((pw / 2, 0), (pt / 2, ph)),
                                            "top": line((pt / 2, ph), (-pt / 2, ph)), "side_m": line((-pt / 2, ph), (-pw / 2, 0))},
                                 ["bottom", "side_p", "top", "side_m"])
    return pieces, seams
