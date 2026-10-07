"""Convert fitted body vertices/faces into a trimesh.Trimesh and export OBJ."""
import trimesh

from .body_fit import FittedBody


def to_trimesh(body: FittedBody) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=body.vertices, faces=body.faces, process=False)


def export_obj(body: FittedBody, path: str) -> None:
    to_trimesh(body).export(path, file_type="obj")
