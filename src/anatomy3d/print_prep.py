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
import trimesh


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


def export_stl(mesh: trimesh.Trimesh, path: str) -> trimesh.Trimesh:
    repaired = repair_mesh(mesh)
    if not repaired.is_watertight:
        print(
            "Warning: mesh is not fully watertight after automatic repair. "
            "Open it in Meshmixer/Netfabb/Blender before printing to close "
            "remaining gaps and thicken any walls thinner than your "
            "printer's nozzle width."
        )
    repaired.export(path, file_type="stl")
    return repaired
