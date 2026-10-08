"""Convert fitted body vertices/faces into a trimesh.Trimesh and export OBJ/GLB."""
import trimesh

from .mesh_types import BodyMesh


def to_trimesh(body: BodyMesh) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
    if body.colors is not None:
        mesh.visual.vertex_colors = body.colors
    return mesh


def export_obj(body: BodyMesh, path: str) -> None:
    to_trimesh(body).export(path, file_type="obj")


def export_glb(body: BodyMesh, path: str) -> None:
    """glTF binary — a standard, widely-supported interchange format (game
    engines, DCC tools, most 3D viewers) for anything downstream that
    wants this mesh with its real per-vertex photo colors, which an STL
    carries no color data for at all and plain OBJ support for varies by
    tool. Confirmed by round-tripping an OBJ with baked colors through
    this exact call and reloading it — the colors survive."""
    to_trimesh(body).export(path, file_type="glb")
