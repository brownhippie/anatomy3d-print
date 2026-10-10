"""Reconstructs a full textured 3D mesh from a single photo via Microsoft's
TRELLIS.2 (https://hf.co/spaces/microsoft/TRELLIS.2, MIT-licensed model and
code), instead of this project's own silhouette+depth-heuristic pipeline.

Confirmed directly against this project's own real test photos, not assumed:
- Human portrait: a single clean, connected, watertight-ish mesh with
  genuinely recognizable face/hair/clothing detail and correct color,
  clearly better than anything procedural_body.py or silhouette_relief.py
  produce on their own.
- Animal (a labrador, side profile, turned head -- the same photo that
  motivated silhouette_relief.py's own depth-rescue work): recognizable
  color and head/ear/leg structure, but the mesh came back FRAGMENTED into
  disconnected floating pieces across every camera angle checked -- not a
  single printable solid. One real test, not an exhaustive benchmark, but a
  confirmed failure, not a guess -- see run_trellis_pipeline's own docstring
  for how this is handled rather than silently shipped.

Runs on Hugging Face's own ZeroGPU infrastructure, not locally -- this
project's environment has no GPU, so this is also the only way to get this
quality of result at all right now. Needs network access; anonymous calls
are heavily rate-limited (confirmed: exhausted within a couple of calls), so
an HF_TOKEN environment variable (a free Hugging Face account's own
read-only access token, https://huggingface.co/settings/tokens) is required
for real use, not just nice-to-have.

A real caveat for commercial use, worth being honest about rather than
glossing over: this project previously removed SMPL-X specifically to stay
commercially clean without a separate license negotiation (see git history --
"Replace SMPL-X with a from-scratch, commercially-unencumbered body
builder"). TRELLIS.2's own license (MIT) doesn't have that problem. What
DOES still need solving before relying on this in an actual commercial
product: this module calls Microsoft's own free, public community Space --
infrastructure this project doesn't own or control (uptime, ToS, quota, and
continued existence are all Microsoft's call, not this project's). Fine for
development/testing and as an optional mode; a real commercial deployment
would need to either self-host TRELLIS.2 on this project's own GPU
infrastructure (the model/weights are MIT, so that's legally fine, just not
yet built), or use a paid API with an actual commercial SLA instead of a
free community demo.
"""
import os
from typing import Optional

import numpy as np
import trimesh

from .mesh_types import BodyMesh

_TRELLIS_SPACE = "microsoft/TRELLIS.2"

# The Space's own UI defaults, confirmed working directly against real test
# photos -- not re-tuned, just carried over as-is.
_SS_GUIDANCE_STRENGTH = 7.5
_SS_GUIDANCE_RESCALE = 0.7
_SS_SAMPLING_STEPS = 12
_SS_RESCALE_T = 5.0
_SHAPE_SLAT_GUIDANCE_STRENGTH = 7.5
_SHAPE_SLAT_GUIDANCE_RESCALE = 0.5
_SHAPE_SLAT_SAMPLING_STEPS = 12
_SHAPE_SLAT_RESCALE_T = 3.0
_TEX_SLAT_GUIDANCE_STRENGTH = 1.0
_TEX_SLAT_GUIDANCE_RESCALE = 0.0
_TEX_SLAT_SAMPLING_STEPS = 12
_TEX_SLAT_RESCALE_T = 3.0


def trellis_available() -> bool:
    try:
        import gradio_client  # noqa: F401
    except ImportError:
        return False
    return True


def generate_mesh_via_trellis(
    image_path: str,
    resolution: str = "1024",
    seed: int = 0,
    decimation_target: int = 300000,
    texture_size: int = 2048,
) -> BodyMesh:
    """Calls the TRELLIS.2 Space's own session-based API (start_session ->
    preprocess_image -> image_to_3d -> extract_glb, the same sequence its own
    Gradio UI performs) and returns the result as this project's BodyMesh,
    with per-vertex color baked from the GLB's texture (see mesh_types.py).

    Raises RuntimeError with the Space's own message on failure -- most
    commonly a ZeroGPU quota error: a free account gets a limited daily
    quota (confirmed directly: two real generations in one session was
    enough to exhaust it, surfacing "You have exceeded your free ZeroGPU
    quota... Subscribe to Hugging Face PRO..."), and anonymous calls with no
    HF_TOKEN at all are rate-limited even more tightly. The Space's own
    message already tells the caller exactly which case it hit.
    """
    from gradio_client import Client, handle_file

    token = os.environ.get("HF_TOKEN")
    client = Client(_TRELLIS_SPACE, token=token)

    client.predict(api_name="/start_session")
    preprocessed = client.predict(input=handle_file(image_path), api_name="/preprocess_image")
    preprocessed_path = preprocessed if isinstance(preprocessed, str) else preprocessed["path"]

    client.predict(
        image=handle_file(preprocessed_path),
        seed=seed,
        resolution=resolution,
        ss_guidance_strength=_SS_GUIDANCE_STRENGTH,
        ss_guidance_rescale=_SS_GUIDANCE_RESCALE,
        ss_sampling_steps=_SS_SAMPLING_STEPS,
        ss_rescale_t=_SS_RESCALE_T,
        shape_slat_guidance_strength=_SHAPE_SLAT_GUIDANCE_STRENGTH,
        shape_slat_guidance_rescale=_SHAPE_SLAT_GUIDANCE_RESCALE,
        shape_slat_sampling_steps=_SHAPE_SLAT_SAMPLING_STEPS,
        shape_slat_rescale_t=_SHAPE_SLAT_RESCALE_T,
        tex_slat_guidance_strength=_TEX_SLAT_GUIDANCE_STRENGTH,
        tex_slat_guidance_rescale=_TEX_SLAT_GUIDANCE_RESCALE,
        tex_slat_sampling_steps=_TEX_SLAT_SAMPLING_STEPS,
        tex_slat_rescale_t=_TEX_SLAT_RESCALE_T,
        api_name="/image_to_3d",
    )

    glb_path, _ = client.predict(
        decimation_target=decimation_target, texture_size=texture_size, api_name="/extract_glb"
    )

    scene = trimesh.load(glb_path)
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) if isinstance(scene, trimesh.Scene) else scene

    colors: Optional[np.ndarray] = None
    try:
        mesh.visual = mesh.visual.to_color()
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
    except Exception:
        colors = None  # texture extraction is a bonus, not required -- same pattern pipeline.py uses elsewhere

    return BodyMesh(vertices=np.asarray(mesh.vertices, dtype=np.float64), faces=np.asarray(mesh.faces), colors=colors)
