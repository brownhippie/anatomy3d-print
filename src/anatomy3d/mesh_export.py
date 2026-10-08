"""Convert fitted body vertices/faces into a trimesh.Trimesh and export OBJ."""
import trimesh

from .mesh_types import BodyMesh


def to_trimesh(body: BodyMesh) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)
    if body.colors is not None:
        mesh.visual.vertex_colors = body.colors
    return mesh


def export_obj(body: BodyMesh, path: str) -> None:
    to_trimesh(body).export(path, file_type="obj")
