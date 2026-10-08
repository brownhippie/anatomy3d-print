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


def estimate_min_wall_thickness(mesh: trimesh.Trimesh, sample_count: int = 1500, seed: int = 0) -> float:
    """Sampling-based estimate, not exhaustive: cast a ray inward from each
    of `sample_count` random surface points and take the nearest interior
    hit as the local thickness there, then report the smallest found. With
    a few thousand samples this reliably catches a broadly thin region
    (a whole limb, say); it can still miss one single pin-thin spot the
    samples happened not to land near."""
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
    return float(valid.min()) if len(valid) else float("nan")


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


def export_stl(
    mesh: trimesh.Trimesh,
    path: str,
    min_wall_mm: float = 0.8,
    min_wall_margin: float = 0.15,
) -> trimesh.Trimesh:
    """`min_wall_mm` is a common rule of thumb for FDM (two 0.4mm-nozzle
    perimeters); tune it for your actual printer/material. The check
    requires the thinnest sampled point to clear that minimum by at least
    `min_wall_margin` (MAD-style margin — see safety.py), not just
    barely touch it."""
    repaired = repair_mesh(mesh)
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
                f"Wall thickness OK: thinnest sampled point ~{min_thickness:.2f}mm, "
                f"{result.margin:.0%} above the {min_wall_mm}mm minimum."
            )
        elif result.margin >= 0:
            print(
                f"Warning: thinnest sampled point ~{min_thickness:.2f}mm clears the "
                f"{min_wall_mm}mm minimum but only by {result.margin:.0%} "
                f"(need at least {min_wall_margin:.0%} margin) — likely a thin "
                "limb or joint. Thicken it in Meshmixer/Netfabb/Blender, or "
                "reprint at a larger --height-mm, before printing. This is a "
                "sampling-based estimate, not exhaustive — re-check in your "
                "slicer regardless."
            )
        else:
            print(
                f"Warning: thinnest sampled point ~{min_thickness:.2f}mm is "
                f"{-result.margin:.0%} BELOW the {min_wall_mm}mm minimum — this "
                "will likely fail to print or snap off. Thicken it in "
                "Meshmixer/Netfabb/Blender, or reprint at a larger "
                "--height-mm, before printing. This is a sampling-based "
                "estimate, not exhaustive — re-check in your slicer regardless."
            )

    repaired.export(path, file_type="stl")
    return repaired
