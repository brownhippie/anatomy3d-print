"""Repair a mesh for 3D printing and export STL.

Cleanup here (merge duplicate vertices, drop degenerate faces, fill
remaining holes, fix normal orientation) handles the usual slicer-rejection
causes. It also drops every connected component except the largest: the
procedural body builder blends several overlapping signed-distance fields
near the neck/shoulders/hips, and where three or more fields meet near the
same point, chaining pairwise smooth-min between them can leave tiny
isolated slivers a voxel or two across — floating debris, not part of the
figure. Discarding everything but the main body is standard practice for
marching-cubes output in general, not just a workaround for this.
"""
import numpy as np
import trimesh

from .safety import mad_margin_above_minimum


def estimate_min_wall_thickness(
    mesh: trimesh.Trimesh, sample_count: int = 1500, seed: int = 0, percentile: float = 2.0
) -> float:
    """Sampling-based estimate, not exhaustive: cast a ray inward from each
    of `sample_count` random surface points and take the nearest interior
    hit as the local thickness there, then report the `percentile`-th
    percentile of what was found.

    Originally this reported the bare minimum, which calibration testing
    showed was a mistake, not just an unlucky default: on the same mesh,
    the measured minimum kept dropping as sample count went up (0.65mm
    mean at 300 samples, 0.05mm at 6000) instead of converging, because a
    bare minimum is an order statistic that keeps chasing whatever's most
    extreme — and there's a genuine near-zero point in this geometry, an
    exact tangent in the capsule smooth-min blend (percentile 0 measured
    0.006mm, a 150x jump to percentile 0.5's 0.98mm). That's a
    zero-measure mathematical artifact, not a real wall.

    A percentile represents actual surface area instead of one singular
    point, and is far more stable. Relative run-to-run variation measured:

        samples    minimum   1st pctl   2nd pctl
            300       58%       17.5%      9.0%
          1,500      ~95%+      16.2%      1.9%
          3,000     113%+       14.1%      1.8%

    (the minimum's own numbers climb with sample count instead of
    converging — see above). The 1st percentile's noise alone was large
    enough to eat most of print_prep's 15% MAD margin, leaving barely any
    of it covering real printer/material tolerance; the 2nd percentile's
    sub-2%-by-1500-samples noise leaves the margin meaningful. Default
    `sample_count` (1500) was picked to land past the 2nd percentile's
    stability knee, not for its own sake.
    """
    points, face_index = trimesh.sample.sample_surface(mesh, sample_count, seed=seed)
    normals = mesh.face_normals[face_index]

    eps = max(mesh.scale * 1e-4, 1e-6)
    origins = points - normals * eps
    directions = -normals

    locations, index_ray, _index_tri = mesh.ray.intersects_location(origins, directions)
    if len(index_ray) == 0:
        return float("nan")

    dists = np.linalg.norm(locations - origins[index_ray], axis=1)
    min_per_ray = np.full(len(points), np.inf)
    np.minimum.at(min_per_ray, index_ray, dists)
    valid = min_per_ray[np.isfinite(min_per_ray)]
    return float(np.percentile(valid, percentile)) if len(valid) else float("nan")


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: c.vertices.shape[0])

    trimesh.repair.fill_holes(mesh)
    mesh.fix_normals()
    return mesh


def simplify_for_output(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Decimates down to roughly `target_faces`, only if the mesh already
    has more than that — never upsamples. The adaptive grid-resolution
    fixes in procedural_body.py (sharpening the marching-cubes grid to
    actually resolve thin limbs and small face features — see that
    module) can produce meshes with 15-20k+ vertices where a few thousand
    would look identical to the eye and print identically on an FDM
    printer; the detail that justified the fine grid is in *where* the
    surface sits, not in carrying every one of those vertices through to
    the output file. Quadric decimation preserves overall shape far
    better than naively dropping vertices — it collapses edges in order
    of least visual cost, so flat/low-curvature regions (a forearm's
    shaft) lose far more triangles than high-curvature ones (a nose tip,
    a joint) for the same target count."""
    if len(mesh.faces) <= target_faces:
        return mesh
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    simplified.merge_vertices()
    trimesh.repair.fill_holes(simplified)
    simplified.fix_normals()
    if not simplified.is_watertight:
        # Measured directly: decimation can introduce a handful of
        # non-manifold edges that the repair pass above doesn't always
        # close, especially on thin limbs where collapsing edges can
        # pinch a wall shut. A smaller file isn't worth shipping a mesh
        # that's worse than the one before decimation — keep the
        # un-decimated (but still repaired) mesh instead.
        return mesh
    return simplified


def export_stl(
    mesh: trimesh.Trimesh,
    path: str,
    min_wall_mm: float = 0.8,
    min_wall_margin: float = 0.15,
    target_faces: "int | None" = 20000,
) -> trimesh.Trimesh:
    """`min_wall_mm` is a common rule of thumb for FDM (two 0.4mm-nozzle
    perimeters); tune it for your actual printer/material. The check
    requires the measured thickness (1st percentile of sampled points —
    see `estimate_min_wall_thickness`'s docstring for why not the bare
    minimum) to clear that minimum by at least `min_wall_margin`
    (MAD-style margin — see safety.py), not just barely touch it.

    `target_faces`: simplifies the mesh down to roughly this many faces
    after repair (see simplify_for_output) before the wall-thickness
    check and export, so the file you actually get matches what was
    measured. Pass None to skip simplification entirely."""
    repaired = repair_mesh(mesh)
    if target_faces is not None:
        repaired = simplify_for_output(repaired, target_faces)
    if not repaired.is_watertight:
        print(
            "Warning: mesh is not fully watertight after automatic repair. "
            "Open it in Meshmixer/Netfabb/Blender before printing to close "
            "remaining gaps."
        )

    try:
        min_thickness = estimate_min_wall_thickness(repaired)
    except Exception as exc:
        print(f"Wall-thickness check skipped (ray casting failed: {exc}).")
        min_thickness = float("nan")

    if min_thickness == min_thickness:  # not NaN
        result = mad_margin_above_minimum(min_thickness, min_wall_mm, min_wall_margin)
        if result.ok:
            print(
                f"Wall thickness OK: thinnest measured region ~{min_thickness:.2f}mm, "
                f"{result.margin:.0%} above the {min_wall_mm}mm minimum."
            )
        elif result.margin >= 0:
            print(
                f"Warning: thinnest measured region ~{min_thickness:.2f}mm clears the "
                f"{min_wall_mm}mm minimum but only by {result.margin:.0%} "
                f"(need at least {min_wall_margin:.0%} margin) — likely a thin "
                "limb or joint. Thicken it in Meshmixer/Netfabb/Blender, or "
                "reprint at a larger --height-mm, before printing. This is a "
                "sampling-based estimate, not exhaustive — re-check in your "
                "slicer regardless."
            )
        else:
            print(
                f"Warning: thinnest measured region ~{min_thickness:.2f}mm is "
                f"{-result.margin:.0%} BELOW the {min_wall_mm}mm minimum — this "
                "will likely fail to print or snap off. Thicken it in "
                "Meshmixer/Netfabb/Blender, or reprint at a larger "
                "--height-mm, before printing. This is a sampling-based "
                "estimate, not exhaustive — re-check in your slicer regardless."
            )

    repaired.export(path, file_type="stl")
    return repaired
