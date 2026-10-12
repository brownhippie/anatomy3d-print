"""Real per-pixel relative depth from a single photo, via Depth Anything V2
Small (24.8M params, Apache-2.0; Base/Large/Giant are CC-BY-NC-4.0 and
don't qualify). Runs the onnx-community ONNX export through onnxruntime,
which is already a base dependency, so it works on the production server
with no torch/transformers install.
"""
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
DEPTH_MODEL_CACHE_PATH = Path.home() / ".cache" / "anatomy3d-print" / "depth_anything_v2_small.onnx"
# The model's own preprocessing (preprocessor_config.json in that repo):
# shorter side 518, both sides a multiple of 14 (its ViT patch size),
# ImageNet mean/std.
_SHORT_SIDE = 518
_PATCH = 14
_MAX_LONG_SIDE = 518 * 2  # bounds memory on very tall/wide photos
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        from .person_segmenter import _ensure_model

        options = ort.SessionOptions()
        # Same memory settings as the segmenter's ONNX model: allocate only
        # what each run needs instead of a pre-grown arena that never shrinks.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        _session = ort.InferenceSession(
            _ensure_model(DEPTH_MODEL_URL, DEPTH_MODEL_CACHE_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return _session


def _model_size(h: int, w: int) -> "tuple[int, int]":
    scale = _SHORT_SIDE / min(h, w)
    if max(h, w) * scale > _MAX_LONG_SIDE:
        scale = _MAX_LONG_SIDE / max(h, w)
    snap = lambda v: max(_PATCH, int(round(v * scale / _PATCH)) * _PATCH)  # noqa: E731
    return snap(h), snap(w)


def estimate_relative_depth(rgb: np.ndarray) -> np.ndarray:
    """(H, W) float64 relative inverse depth at `rgb`'s own resolution:
    higher = closer to the camera. Relative only -- no units, no fixed
    zero -- so callers normalise it against the subject themselves."""
    h, w = rgb.shape[:2]
    mh, mw = _model_size(h, w)
    arr = np.asarray(Image.fromarray(rgb).resize((mw, mh), Image.BICUBIC), dtype=np.float32) / 255.0
    arr = ((arr - _MEAN) / _STD).transpose(2, 0, 1)[None]
    session = _get_session()
    out = session.run(None, {session.get_inputs()[0].name: arr})[0]
    depth = np.asarray(out, dtype=np.float32).reshape(out.shape[-2], out.shape[-1])
    return np.asarray(Image.fromarray(depth).resize((w, h), Image.BICUBIC), dtype=np.float64)
