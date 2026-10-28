"""Exports the face's own human controls to src/anatomy3d/data/face_reference.npz:
MakeHuman's face shape targets (nose, mouth, chin, cheeks, eyes, eyebrows,
forehead, head shape) and the Face Units facial actions (jaw open, smile,
blink, ...), both CC0 1.0, as they ship in the Anny package's data
(data/mpfb2/targets, data/faceunits01/targets/faceunits). Each is a change of
the body's points, at most what the target was made to do (weight 1): set
between 0 and 1, a face stays a human face.

The targets index MakeHuman's base mesh; the body here (body_reference.npz)
is Anny's "anny" layout of it: the base mesh's body, eye and tongue faces,
less the 32 points Anny's edit takes out, in base-mesh order. Offsets are in
decimetres with y up; here metres, x the body's left, y back, z up (Anny's own
world transform: 0.1 and a quarter turn about x).

Reads data files only (no package code is run): point it at the unpacked
Anny wheel's `anny` folder.

usage: python tools/extract_face_reference.py /path/to/anny
       python tools/extract_face_reference.py /path/to/anny landmarks [check.png]
       python tools/extract_face_reference.py /path/to/anny teeth
"""
import gzip
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "anatomy3d" / "data" / "face_reference.npz"
GROUPS = ["nose", "mouth", "chin", "cheek", "eyes", "eyebrows", "forehead", "head"]
SKIP = ("head-age", "head-trans", "head-angle", "head-fat")  # (moves the whole head, or ages it: the fit's and the build's)
DROPPED = list(range(1778, 1794)) + list(range(8450, 8466))  # (Anny's mesh edit)


def read_obj(root: Path):
    V, groups, g = [], {}, None
    for line in open(root / "data/mpfb2/3dobjs/base.obj"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            g = line.split(maxsplit=1)[1].strip()
            groups.setdefault(g, [])
        elif line.startswith("f ") and g is not None:
            groups[g].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(V), groups


def base_mesh(root: Path):
    V, groups, g = [], {}, None
    for line in open(root / "data/mpfb2/3dobjs/base.obj"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            g = line.split(maxsplit=1)[1].strip()
            groups.setdefault(g, [])
        elif line.startswith("f ") and g is not None:
            groups[g].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    used = set()
    for k in ("body", "helper-l-eye", "helper-r-eye", "helper-tongue"):
        for f in groups[k]:
            used.update(f)
    keep = np.array(sorted(used - set(DROPPED)))
    return len(V), keep


def read_target(path: Path, n: int) -> np.ndarray:
    d = np.zeros((n, 3))
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and not line.startswith("#"):
                d[int(p[0])] = [float(x) for x in p[1:]]
    # decimetres, y up -> metres, x left, y back, z up
    return 0.1 * np.column_stack([d[:, 0], -d[:, 2], d[:, 1]])


def main():
    root = Path(sys.argv[1])
    n_base, keep = base_mesh(root)
    ours = np.load(OUT.with_name("body_reference.npz"))["male_vertices"]
    assert len(keep) == len(ours), (len(keep), len(ours))
    shapes, expr = {}, {}
    for g in GROUPS:
        for p in sorted((root / "data/mpfb2/targets" / g).glob("*.target.gz")):
            name = p.name[:-len(".target.gz")]
            if not name.startswith(SKIP):
                shapes[name] = read_target(p, n_base)[keep]
    for p in sorted((root / "data/faceunits01/targets/faceunits").glob("*.target")):
        expr[p.stem] = read_target(p, n_base)[keep]
    allD = np.stack(list(shapes.values()) + list(expr.values()))
    idx = np.flatnonzero(np.abs(allD).max(axis=(0, 2)) > 0)
    data = {
        "idx": idx.astype(np.int32),
        "shape_names": np.array(list(shapes)),
        "shape_d": np.stack([shapes[k][idx] for k in shapes]).astype(np.float32),
        "expr_names": np.array(list(expr)),
        "expr_d": np.stack([expr[k][idx] for k in expr]).astype(np.float32),
    }
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB): {len(shapes)} shape targets, {len(expr)} facial actions, "
          f"{len(idx)} points")


if __name__ == "__main__" and len(sys.argv) == 2:
    main()


OPEN_FOR_LOOKUP = {"jawOpen": 0.45, "eyeWideLeft": 0.5, "eyeWideRight": 0.5}


def landmarks_on_body():
    """Where each of the face reader's 468 points falls on this body's own
    face: the face reader run once on a plain shaded picture of it (front,
    orthographic) with the jaw open and the eyes wide (OPEN_FOR_LOOKUP: with
    the mouth shut the upper and lower inner-lip points fall on the same seam,
    and the lids on the same line), each point taken back onto the face's
    surface and its nearest body point kept. A lookup made once, generic to
    every photo (the body's points keep their numbers whatever the face does)."""
    sys.path.insert(0, str(OUT.parents[1]))
    import trimesh

    from anatomy3d import body_reference as br
    from anatomy3d import lab
    from anatomy3d.face_features import detect_face_mesh
    from anatomy3d.silhouette_relief import _raster_depth

    body = br._body("male")
    V, F = body["verts"].copy(), body["faces"]
    fr = np.load(OUT)
    en = [str(n) for n in fr["expr_names"]]
    for n, w in OPEN_FOR_LOOKUP.items():
        V[fr["idx"]] += w * fr["expr_d"][en.index(n)]
    head = body["part_w"][:, body["part_names"].index("head")] > 0.5
    Fh = F[head[F].all(1)]
    ppm = 5000.0  # px per metre of the picture
    x0, z1 = V[head, 0].min() - 0.03, V[head, 2].max() + 0.03
    W, H = int((np.ptp(V[head, 0]) + 0.06) * ppm), int((np.ptp(V[head, 2]) + 0.06) * ppm)
    px, py = (V[:, 0] - x0) * ppm, (z1 - V[:, 2]) * ppm
    near, _ = _raster_depth(px, py, V[:, 1] * ppm, Fh, (H, W))  # (nearest to the camera at -y: smallest depth)
    sil = np.isfinite(near)
    zf = np.where(sil, -near, np.nan)
    img = np.asarray(lab._shade(sil, zf, 1.0))
    got = detect_face_mesh(img)
    if got is None:
        raise RuntimeError("no face found on the body's own face")
    lm = np.asarray(got, dtype=np.float64)[:468]
    r = np.clip(lm[:, 1].round().astype(int), 0, H - 1)
    c = np.clip(lm[:, 0].round().astype(int), 0, W - 1)
    # a point over no surface (the open mouth's dark): the nearest surface's depth
    from scipy.ndimage import distance_transform_edt

    _, (ir, ic) = distance_transform_edt(~np.isfinite(near), return_indices=True)
    depth = near[ir[r, c], ic[r, c]] / ppm
    P = np.column_stack([lm[:, 0] / ppm + x0, depth, z1 - lm[:, 1] / ppm])
    hv = np.flatnonzero(head)
    from scipy.spatial import cKDTree

    d, j = cKDTree(V[hv]).query(P)
    lv = hv[j]
    # the inner lips: through the open mouth the point can fall on what lies
    # behind them (tongue, mouth floor); the lip is the front-most surface there
    INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    fx, fz = V[hv, 0], V[hv, 2]
    for i in INNER_LIPS:
        near_ = np.hypot(fx - P[i, 0], fz - P[i, 2]) < 0.002
        if near_.any():
            c = hv[near_]
            k = c[np.argmin(V[c, 1])]
            lv[i] = k
            d[i] = float(np.linalg.norm(V[k] - P[i]))
    return lv, d, img, lm


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "landmarks":
    lv, dist, img, lm = landmarks_on_body()
    d = dict(np.load(OUT))
    d["lm_vertex"] = lv.astype(np.int32)
    d["lm_dist_m"] = dist.astype(np.float32)
    np.savez_compressed(OUT, **d)
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).convert("RGB")
    dr = ImageDraw.Draw(im)
    for x, y in lm[:, :2]:
        dr.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 0, 0))
    im.save(Path(sys.argv[3]) if len(sys.argv) > 3 else Path("face_landmarks_check.png"))
    print("landmarks on the body's face: nearest point off by (mm) median", round(float(np.median(dist)) * 1000, 2), "max", round(float(dist.max()) * 1000, 2))


def teeth(root: Path):
    """The base mesh's own upper and lower teeth (open bands that follow the
    mouth), with their change under each face control, and the base mesh's
    neutral body points to place them on this body (by the head's own fit)."""
    V, groups = read_obj(root)
    n_base, keep = base_mesh(root)
    to_m = lambda X: 0.1 * np.column_stack([X[:, 0], -X[:, 2], X[:, 1]])
    d = dict(np.load(OUT))
    tv, tf, up = [], [], []
    for k, is_up in (("helper-upper-teeth", 1), ("helper-lower-teeth", 0)):
        q = groups[k]
        u = np.unique(np.concatenate([np.array(x) for x in q]))
        at = {int(v): i + len(tv) for i, v in enumerate(u)}
        for x in q:
            for i in range(1, len(x) - 1):
                tf.append([at[x[0]], at[x[i]], at[x[i + 1]]])
        tv += list(u)
        up += [is_up] * len(u)
    tv = np.array(tv)
    where = {p.name[:-len(".target.gz")]: p for g in GROUPS for p in (root / "data/mpfb2/targets" / g).glob("*.target.gz")}
    shapes = [read_target(where[str(n)], n_base)[tv] for n in d["shape_names"]]
    expr = [read_target(root / "data/faceunits01/targets/faceunits" / f"{n}.target", n_base)[tv] for n in d["expr_names"]]
    d.update(teeth_v=to_m(V[tv]).astype(np.float32), teeth_f=np.array(tf, dtype=np.int32),
             teeth_upper=np.array(up, dtype=bool), teeth_shape_d=np.stack(shapes).astype(np.float32),
             teeth_expr_d=np.stack(expr).astype(np.float32), base_keep_v=to_m(V[keep]).astype(np.float32))
    np.savez_compressed(OUT, **d)
    print("teeth:", len(tv), "points,", len(tf), "triangles; deltas moving them:",
          int((np.abs(np.stack(shapes)).max(axis=(1, 2)) > 0).sum()), "shape,",
          int((np.abs(np.stack(expr)).max(axis=(1, 2)) > 0).sum()), "facial actions")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "teeth":
    teeth(Path(sys.argv[1]))
