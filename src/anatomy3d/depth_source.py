"""Optional: real per-pixel depth from a single photo, via Depth Anything
V2 Small (24.8M params, Apache-2.0 — verified directly against the
official repo, not assumed from a secondary source; Base/Large/Giant are
CC-BY-NC-4.0 and don't qualify, so only Small is used here). Requires
`pip install -r requirements-depth.txt` (torch + transformers); the
default pipeline works fine without it, just with the flat symmetric
depth guess `procedural_body.py` already uses.
"""
import numpy as np

_pipe = None  # lazy singleton: loading the model is slow, worth reusing


def depth_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def estimate_relative_depth(rgb: np.ndarray) -> np.ndarray:
    """Returns an (H, W) float array, raw from the model with no sign
    correction applied. The caller (procedural_body.py's depth sculpting)
    currently ASSUMES higher value = closer to camera; that assumption is
    not yet checked against a real photo — see the note where it's used."""
    global _pipe
    if _pipe is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Depth estimation needs torch + transformers: "
                "pip install -r requirements-depth.txt"
            ) from exc
        _pipe = pipeline(
            task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
        )

    from PIL import Image

    result = _pipe(Image.fromarray(rgb))
    return np.asarray(result["predicted_depth"].squeeze(), dtype=np.float64)
