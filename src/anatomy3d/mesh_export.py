"""Convert fitted body vertices/faces into a trimesh.Trimesh and export OBJ/GLB."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .mesh_types import BodyMesh

# A flat cylindrical base/pedestal under the figure, the way a miniature
# sits on its own printed base -- requested directly for every pipeline's
# output, preview included, not just the printable STL.
#
# Sized relative to the figure itself rather than fixed millimeters, since
# `target_height_mm` varies a lot across pipeline calls: _BASE_DISK_MARGIN
# makes the disk a bit wider than the figure's own footprint (so it reads
# as a base the figure stands ON, not a slab cut to its exact outline --
# "a bit bigger than flat" was read as both "has real thickness" and "a
# bit larger than the figure's own footprint"), and _BASE_DISK_MIN_RADIUS_FRAC
# floors that for a figure whose footprint is degenerate or tiny (e.g. a
# single thin limb reaching the ground).
_BASE_DISK_MARGIN = 1.2
_BASE_DISK_MIN_RADIUS_FRAC = 0.35
_BASE_DISK_THICKNESS_FRAC = 0.05
_BASE_DISK_MIN_THICKNESS_MM = 2.0
_BASE_DISK_MAX_THICKNESS_MM = 10.0
_BASE_DISK_SECTIONS = 64
_BASE_DISK_COLOR = np.array([150, 150, 155], dtype=np.uint8)  # neutral stone-grey plinth


def add_base_disk(body: BodyMesh, target_height_mm: float) -> BodyMesh:
    """Returns a new BodyMesh with a flat cylindrical disk unioned beneath
    `body`, centered under its own horizontal footprint. Y is this
    project's height axis throughout (every pipeline measures height off
    it) so the disk is built around the X/Z plane and placed just under
    `body`'s own minimum Y.

    A REAL boolean union (needs the manifold3d package -- see
    requirements.txt), not just overlapping geometry concatenated
    together: confirmed directly that the latter doesn't work with this
    project's own repair_mesh (print_prep.py), which deliberately keeps
    only the largest connected component to drop marching-cubes slivers
    (see that function's docstring) -- a disk merely overlapping the
    figure is its own, smaller component, so repair_mesh silently
    discarded it every time before this used a real union.

    The boolean op re-triangulates, so it can't carry body.colors through
    directly; colors are reattached afterward by nearest-vertex lookup
    against the two ORIGINAL meshes (figure keeps its own baked photo
    colors, the disk gets the flat plinth color), which is approximate
    right at the weld seam but correct everywhere else. Falls back to
    returning `body` unchanged (no disk, but still a valid run) if the
    union itself fails -- same "a failed bonus feature shouldn't fail the
    run" principle this project already applies to face/hair/depth."""
    verts = body.vertices
    if len(verts) == 0:
        return body

    xz = verts[:, [0, 2]]
    center_xz = (xz.min(axis=0) + xz.max(axis=0)) / 2.0
    footprint_radius = float(np.linalg.norm(xz - center_xz, axis=1).max())
    radius = max(footprint_radius * _BASE_DISK_MARGIN, _BASE_DISK_MIN_RADIUS_FRAC * target_height_mm)
    thickness = float(
        np.clip(_BASE_DISK_THICKNESS_FRAC * target_height_mm, _BASE_DISK_MIN_THICKNESS_MM, _BASE_DISK_MAX_THICKNESS_MM)
    )
    # A little overlap into the figure's own lowest geometry (rather than
    # just touching it) gives the boolean union an unambiguous, non-zero
    # volume to merge at the seam.
    overlap = thickness * 0.2

    base_y = float(verts[:, 1].min())
    disk = trimesh.creation.cylinder(radius=radius, height=thickness, sections=_BASE_DISK_SECTIONS)
    # trimesh builds the cylinder centered on the origin along its own Z
    # axis -- rotate it onto Y (this project's up axis) before placing it.
    disk.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    disk.apply_translation([center_xz[0], base_y - thickness / 2.0 + overlap, center_xz[1]])

    figure_mesh = trimesh.Trimesh(vertices=verts, faces=body.faces, process=False)
    try:
        combined = trimesh.boolean.union([figure_mesh, disk])
    except Exception as exc:  # noqa: BLE001 - a failed bonus feature shouldn't fail the run
        print(f"Note: could not add the base disk ({exc}); the figure will print/render without one.")
        return body

    colors = None
    if body.colors is not None:
        source_points = np.vstack([verts, disk.vertices])
        source_colors = np.vstack([body.colors, np.tile(_BASE_DISK_COLOR, (len(disk.vertices), 1))])
        nearest = cKDTree(source_points).query(combined.vertices)[1]
        colors = source_colors[nearest]
    return BodyMesh(vertices=np.asarray(combined.vertices), faces=np.asarray(combined.faces), colors=colors)


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
