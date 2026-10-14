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
   points; already fitted when the inputs are prepared;
3. its depth checked against the photo's: each part's front against the
   depth estimate (report["depth_order"]);
4. the clothes on it: the body grown along its normals out to the photo's
   outline (the room clothes and hair take), each garment stepping over the
   one under it where they meet;
5. the photo's detail on the surface the camera sees: face shaping, the
   face's parts, clothes' steps and fine depth detail, pushed toward the
   camera on the vertices that are frontmost (an arm across the chest gets
   the sleeve's folds, the chest under it nothing);
6. out: cut where the photo is cut (a bust), scaled to the figure's height.

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


def _cover_unseen(V, F, part_w, names, inputs, ppm):
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
    V, names, _ = sr._grow_to_outline(V, F, body_ref, part_w, mask, lambda P: (P[:, 0], P[:, 1]), shape, d_in, d_out, 1.0, mask,
                                      normals=sr._vertex_normals(Vsm, F), keep=face_keep)
    core = pv["core"]
    if core is not None:
        hw = part_w[:, names.index("head")] if "head" in names else 0.0
        m_ = hw * sr._REF_HEAD_MARGIN + (1.0 - hw) * sr._REF_DEPTH_MARGIN
        V[:, 2] = core + m_ * (V[:, 2] - core)
    if (ref.get("report") or {}).get("gender") == "male" and "torso" in names:
        V = sr._drape_chest(V, F, part_w[:, names.index("torso")], ppm, ref.get("joints_px") or {})
    if ref.get("ears") and pv["ear_w"] is not None and ref.get("head_axes") is not None:
        from .body_reference import place_ears

        V = place_ears(V, np.asarray(pv["ear_w"], dtype=np.float64), ref["head_axes"], ref["ears"], faces=F, px_per_m=ppm)
    # mitten hands
    hand = np.zeros(len(V))
    for n in ("hand.L", "hand.R"):
        if n in names:
            hand = np.maximum(hand, part_w[:, names.index(n)])
    if hand.max() > 0.5:
        lap = sr._mesh_laplacian(F, len(V))
        Vs = np.column_stack([sr._mesh_smooth(lap, V[:, k], _MITTEN_SMOOTH) for k in range(3)])
        V = V + np.clip(hand * 1.5, 0.0, 1.0)[:, None] * (Vs - V)

    # clothes and hair round the sides the photo doesn't show: a part the
    # photo shows mostly under clothes is clothed all round -- cloth lies
    # smooth over muscle and bone (no shoulder blades or spine through a
    # jacket) -- and a head with hair has it over the back of the skull too
    V = _cover_unseen(V, F, part_w, names, inputs, ppm)

    # finer surface for the detail
    from trimesh.remesh import subdivide_loop

    V, F = subdivide_loop(V, F, iterations=_SUBDIVIDE)

    # 3. depth order against the photo's estimate (reported; the pose fit
    # already keeps forearms on the body's front)
    near, far = _raster_front(V, F, shape)
    cov = np.isfinite(near) & mask
    front = np.where(cov, -near, 0.0)  # toward the camera
    report = {}

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

    # 6. out: cut where the photo is cut, then to the figure's frame
    mesh = trimesh.Trimesh(V, F, process=False)
    rows = np.flatnonzero(mask.any(axis=1))
    cut_off = bust if bust is not None else bool(rows[-1] >= shape[0] - 3)
    if cut_off:
        # (manifold3d: cut and capped, still a closed solid; keeps y above the photo's bottom)
        from manifold3d import Manifold, Mesh

        man = Manifold(mesh=Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                                 tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
        if man.is_empty():
            raise RuntimeError("the posed body isn't a closed solid")
        man = man.trim_by_plane((0.0, -1.0, 0.0), -float(rows[-1]))
        mm_ = man.to_mesh()
        mesh = trimesh.Trimesh(np.asarray(mm_.vert_properties[:, :3], dtype=np.float64), np.asarray(mm_.tri_verts, dtype=np.int64),
                               process=False)
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
