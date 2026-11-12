"""The cloth step: a garment cut to its measurements and draped round a posed
body by simulation, so it hangs, folds and lies the way cloth does instead of
following the skin.

A garment is pieces of cloth (particles joined by stretch and bend links):
the body of the garment hangs from the shoulders, the sleeves from the
shoulder seams, ribbed bands (hem, cuffs) are made smaller than what they
go round so the cloth above them blouses, a hood lies down on the back, a
pocket is stitched on along its top and bottom. Gravity pulls, the body
pushes back (every particle kept a cloth's thickness off the skin), and the
links pull the cloth back to the lengths it was cut to: where a piece is
longer than the way it goes (a sleeve longer than the arm), it folds.

Position-based dynamics (Jacobi): no outside service, numpy only.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

G = np.array([0.0, 0.0, -9.81])  # m/s^2, z up
THICK = 0.004  # m: cloth kept this far off the skin
RING = 120  # particles round each ring


@dataclass
class Cloth:
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    edges: list = field(default_factory=list)  # (i, j, rest, stiffness)
    faces: list = field(default_factory=list)  # (a, b, c, piece)
    pins: dict = field(default_factory=dict)  # particle -> position it is held at

    def add(self, P):
        i0 = len(self.x)
        self.x = np.vstack([self.x, np.asarray(P, float).reshape(-1, 3)])
        return i0

    def link(self, i, j, rest=None, k=1.0):
        i, j = np.atleast_1d(i), np.atleast_1d(j)
        r = np.linalg.norm(self.x[i] - self.x[j], axis=1) if rest is None else np.broadcast_to(rest, i.shape)
        self.edges.append(np.column_stack([i, j, r, np.full(len(i), k)]))

    def tube(self, rings, piece, closed_end=False, rest_scale=None, bend=0.15):
        """rings (k, n, 3) as a tube of cloth; rest_scale (k, 2): each ring's (round, along) rest length
        over its cut shape. Returns the index of each particle (k, n)."""
        k, n, _ = rings.shape
        i0 = self.add(rings)
        idx = i0 + np.arange(k * n).reshape(k, n)
        rs = np.ones((k, 2)) if rest_scale is None else np.asarray(rest_scale, float)
        for r in range(k):
            a, b = idx[r], np.roll(idx[r], -1)
            self.link(a, b, np.linalg.norm(self.x[a] - self.x[b], axis=1) * rs[r, 0])
            self.link(idx[r], np.roll(idx[r], -2), np.linalg.norm(self.x[idx[r]] - self.x[np.roll(idx[r], -2)], axis=1) * rs[r, 0], bend)
            if r + 1 < k:
                a, b = idx[r], idx[r + 1]
                sc = 0.5 * (rs[r, 1] + rs[r + 1, 1])
                self.link(a, b, np.linalg.norm(self.x[a] - self.x[b], axis=1) * sc)
                c = np.roll(idx[r + 1], -1)
                self.link(a, c, np.linalg.norm(self.x[a] - self.x[c], axis=1) * sc, 0.5)
                if r + 2 < k:
                    d = idx[r + 2]
                    sc2 = (rs[r, 1] + rs[r + 1, 1] + rs[r + 2, 1]) / 3
                    self.link(a, d, np.linalg.norm(self.x[a] - self.x[d], axis=1) * sc2, bend)
                for j in range(n):
                    j2 = (j + 1) % n
                    self.faces += [(a[j], a[j2], b[j2], piece), (a[j], b[j2], b[j], piece)]
        if closed_end:
            tip = self.add(rings[-1].mean(0))
            for j in range(n):
                self.faces.append((idx[-1, j], idx[-1, (j + 1) % n], tip, piece))
            self.link(idx[-1], np.full(n, tip), np.linalg.norm(self.x[idx[-1]] - self.x[tip], axis=1))
        return idx

    def patch(self, grid, piece, bend=0.15):
        """an open sheet (k, n, 3)"""
        k, n, _ = grid.shape
        i0 = self.add(grid)
        idx = i0 + np.arange(k * n).reshape(k, n)
        for r in range(k):
            self.link(idx[r, :-1], idx[r, 1:])
            if n > 2:
                self.link(idx[r, :-2], idx[r, 2:], None, bend)
            if r + 1 < k:
                self.link(idx[r], idx[r + 1])
                self.link(idx[r, :-1], idx[r + 1, 1:], None, 0.5)
                for j in range(n - 1):
                    a, b, c, d = idx[r, j], idx[r, j + 1], idx[r + 1, j + 1], idx[r + 1, j]
                    self.faces += [(a, b, c, piece), (a, c, d, piece)]
            if r + 2 < k:
                self.link(idx[r], idx[r + 2], None, bend)
        return idx


def body_surface(V, F, n=90000, seed=0):
    """the body as points with outward normals (what the cloth collides with)"""
    import trimesh

    mesh = trimesh.Trimesh(V, F, process=False)
    P, fi = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    return np.vstack([P, V]), np.vstack([mesh.face_normals[fi], mesh.vertex_normals])


def _colours(ei, ej, n):
    """the links in groups that share no particle (each group solved at once, exactly)"""
    order = np.argsort(np.minimum(ei, ej))
    col = np.full(len(ei), -1)
    used = {}
    for e in order:
        a, b = ei[e], ej[e]
        ua, ub = used.setdefault(a, set()), used.setdefault(b, set())
        c = 0
        while c in ua or c in ub:
            c += 1
        col[e] = c
        ua.add(c)
        ub.add(c)
    return [np.flatnonzero(col == c) for c in range(col.max() + 1)]


def drape(cl: Cloth, surf, frames=120, substeps=10, iters=2, dt=1 / 60, thick=THICK, damping=0.995, friction=0.5,
          max_stretch=1.04, log=None, alt_surf=None, alt=None, push=None, push_frames=0):
    """let the cloth fall onto the body and settle: gravity in small steps, every link solved in turn
    (Gauss-Seidel by colour), the cloth kept off the skin, never stretched past `max_stretch`"""
    BP, BN = surf
    tree = cKDTree(BP)
    # some cloth may meet only part of the body (a hood falling back past the head): its own surface
    if alt_surf is not None and alt is not None:
        AP, AN = alt_surf
        atree = cKDTree(AP)
        alt = np.asarray(alt, bool)
    E = np.vstack(cl.edges)
    ei, ej = E[:, 0].astype(int), E[:, 1].astype(int)
    rest, k = E[:, 2], E[:, 3]
    groups = _colours(ei, ej, len(cl.x))
    x = cl.x.copy()
    v = np.zeros_like(x)
    pin = np.array(sorted(cl.pins), dtype=int)
    pin_at = np.array([cl.pins[i] for i in pin]) if len(pin) else np.zeros((0, 3))
    w = np.ones(len(x))
    w[pin] = 0.0
    h = dt / substeps

    def collide(p):
        _, kk = tree.query(p)
        q, nn = BP[kk], BN[kk]
        if alt_surf is not None and alt is not None and alt.any():
            _, ka = atree.query(p[alt])
            q, nn = q.copy(), nn.copy()
            q[alt], nn[alt] = AP[ka], AN[ka]
        s = ((p - q) * nn).sum(1)
        hit = s < thick
        p[hit] += (thick - s[hit])[:, None] * nn[hit]
        return hit

    def solve(p, kk_scale=1.0, limit=False):
        for g in groups:
            a, b = ei[g], ej[g]
            d = p[b] - p[a]
            L = np.linalg.norm(d, axis=1) + 1e-12
            if limit:
                over = L > rest[g] * max_stretch
                C_ = np.where(over, L - rest[g] * max_stretch, 0.0)
                kk = 1.0
            else:
                C_ = L - rest[g]
                kk = k[g] * kk_scale
            ws = w[a] + w[b] + 1e-12
            corr = (kk * C_ / (L * ws))[:, None] * d
            p[a] += w[a][:, None] * corr
            p[b] -= w[b][:, None] * corr

    for f in range(frames):
        for s in range(substeps):
            v += G * h * (w > 0)[:, None]
            if push is not None and f < push_frames:
                v += push * h * (w > 0)[:, None]
            v *= damping
            p = x + v * h
            for _ in range(iters):
                solve(p)
            solve(p, limit=True)
            hit = collide(p)
            if len(pin):
                p[pin] = pin_at          # (held cloth stays where it is held, whatever the body says)
                hit[pin] = False
            nv = (p - x) / h
            nv[hit] *= 1.0 - friction
            v, x = nv, p
        if log and f % 20 == 0:
            st = np.linalg.norm(x[ej] - x[ei], axis=1) / rest
            log(f"frame {f}: fastest {np.linalg.norm(v, axis=1).max():.3f} m/s, most stretched link {st.max():.2f}x")
    cl.x = x
    return cl


def hull2(P):
    if len(P) < 3:
        return None
    try:
        return P[ConvexHull(P).vertices]
    except Exception:
        return None


def ring_round(poly, c, n=RING):
    """a convex polygon sampled at n equal angles round c"""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    D = np.column_stack([np.cos(th), np.sin(th)])
    A, B = poly, np.roll(poly, -1, 0)
    out = np.zeros((n, 2))
    for kk, d in enumerate(D):
        best = 0.0
        for a, b in zip(A, B):
            e = b - a
            M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            t, u = np.linalg.solve(M, a - c)
            if t > best and -1e-9 <= u <= 1 + 1e-9:
                best = t
        if best <= 1e-6:   # (a ray that found no edge: the outline's farthest reach that way)
            best = max(float(((poly - c) @ d).max()), 1e-3)
        out[kk] = c + d * best
    return out


def perim(P):
    return float(np.linalg.norm(P - np.roll(P, -1, 0), axis=1).sum())


def grow(poly, d):
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    return hull2((poly[:, None, :] + np.column_stack([np.cos(th), np.sin(th)])[None] * d).reshape(-1, 2))


def section(V, F, origin, normal):
    import trimesh

    seg = trimesh.intersections.mesh_plane(trimesh.Trimesh(V, F, process=False), normal, origin)
    return seg.reshape(-1, 3) if len(seg) else np.zeros((0, 3))
