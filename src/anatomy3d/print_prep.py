"""Repair a mesh for 3D printing and export STL.

SMPL-X's template topology is closed by construction, so most fits are
already watertight. This pass still runs standard cleanup (merge duplicate
vertices, drop degenerate faces, fill any remaining holes, fix normal
orientation) because reprojection-driven optimization can occasionally
produce near-degenerate triangles that a slicer will reject.
"""
import trimesh


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
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
