"""A mesh made finer where it needs to be, and nowhere else: every edge
longer than a given length split at its middle (in the faces marked, and in
their neighbours across a split edge, so no crack opens), each triangle
re-cut by how many of its edges were split, until no marked edge is longer.
Per-vertex values (part weights, ...) are carried to the new points as the
mean of the edge's two ends."""
import numpy as np


def refine(V, F, mark, max_len: float, values=(), rounds: int = 4):
    """(V, F, values) with every edge of a marked face (bool per face)
    longer than `max_len` split, repeatedly. `values`: per-vertex arrays."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    vals = [np.asarray(a, dtype=np.float64) for a in values]
    mark = np.asarray(mark, dtype=bool)
    for _ in range(rounds):
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])  # face j's edges at j, j+n, j+2n
        n = len(F)
        key = np.sort(e, axis=1)
        length = np.linalg.norm(V[key[:, 0]] - V[key[:, 1]], axis=1)
        want = (length > max_len) & np.tile(mark, 3)
        if not want.any():
            break
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        split_u = np.zeros(len(uniq), dtype=bool)
        split_u[inv[want]] = True
        mid = -np.ones(len(uniq), dtype=np.int64)
        su = np.flatnonzero(split_u)
        mid[su] = len(V) + np.arange(len(su))
        V = np.vstack([V, 0.5 * (V[uniq[su, 0]] + V[uniq[su, 1]])])
        vals = [np.concatenate([a, 0.5 * (a[uniq[su, 0]] + a[uniq[su, 1]])]) for a in vals]
        m = mid[inv].reshape(3, n).T  # per face: the midpoint of edge (0,1), (1,2), (2,0), or -1
        cnt = (m >= 0).sum(1)
        out, out_mark = [F[cnt == 0]], [mark[cnt == 0]]
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        # one edge split: rotate so it is edge (0,1)
        for r in range(3):
            sel = (cnt == 1) & (m[:, r] >= 0)
            if not sel.any():
                continue
            p, q, s = [F[sel, (r + i) % 3] for i in range(3)]
            mm = m[sel, r]
            out += [np.column_stack([p, mm, s]), np.column_stack([mm, q, s])]
            out_mark += [mark[sel]] * 2
        # two edges split: rotate so the unsplit one is edge (2,0)
        for r in range(3):
            sel = (cnt == 2) & (m[:, (r + 2) % 3] < 0)
            if not sel.any():
                continue
            p, q, s = [F[sel, (r + i) % 3] for i in range(3)]
            m01, m12 = m[sel, r], m[sel, (r + 1) % 3]
            out += [np.column_stack([p, m01, s]), np.column_stack([m01, m12, s]), np.column_stack([m01, q, m12])]
            out_mark += [mark[sel]] * 3
        sel = cnt == 3
        if sel.any():
            m01, m12, m20 = m[sel, 0], m[sel, 1], m[sel, 2]
            out += [np.column_stack([a[sel], m01, m20]), np.column_stack([m01, b[sel], m12]),
                    np.column_stack([m20, m12, c[sel]]), np.column_stack([m01, m12, m20])]
            out_mark += [mark[sel]] * 4
        F = np.vstack(out)
        mark = np.concatenate(out_mark)
    return V, F, vals
