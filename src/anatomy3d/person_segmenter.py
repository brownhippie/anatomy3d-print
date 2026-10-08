"""Real per-pixel person/background separation via MediaPipe's Selfie
Multiclass Segmenter — same Apache-2.0 MediaPipe model family, same GCS
model bucket, same ImageSegmenter task API as the Hair Segmenter already
used in hair_features.py. Not a new dependency or license to clear (see
README's "Why from scratch" section).

This replaces silhouette.py's from-scratch color-clustering approach as
the PRIMARY silhouette method (see extract_silhouette), not an addition
alongside it. That's a real, measured finding, not a style preference:
tested against three structurally different real photos (a full-body
beach photo, a tightly-cropped white-background studio portrait, and a
photo with a complex patterned-curtain background that the color
classifier demonstrably could not separate from the subject — see
COLOR_FAILURE_RATIO's docstring in silhouette.py) and this model handled
every one of them correctly out of the box: no border-sampling
assumption to violate, no flat pixel constants to miscalibrate across
resolutions, no pose-landmark corridor to be fooled by a hallucinated
joint. The business-portrait photo's one remaining known defect (a
background bulge from hallucinated elbow/wrist tracking, see
silhouette.py's commit history) is absent here entirely, because this
mask doesn't depend on pose landmarks at all.

Still imperfect at the same genuinely hard case color classification
also struggled with: individual wispy flyaway hair strands against open
sky render as a smooth blob boundary, not individual strands — confirmed
directly, not assumed, by the same zoomed-crop check used throughout
this project's silhouette work. That's a sub-pixel alpha-matting problem
neither method solves.

A second real gap, found after this module shipped: tested against a
heavily motion-blurred photo (a believable bad handheld/action shot) and
the model genuinely missed real content, not just a connectivity issue —
confirmed by checking its raw output directly. Sharpening the photo
aggressively before segmentation (far more than preprocess.py's own
general-purpose UnsharpMask(radius=2, percent=120), which is tuned for
contrast on already-decent photos, not rescuing a badly blurred one)
measurably recovered it: a blurred photo that lost both legs and the
head without this reliably regained all of them with it. But a single
"is this photo blurry" threshold to decide when to apply it doesn't
exist cleanly — measured Laplacian variance (the standard blur metric)
across a range of blur levels and the gap between "still working" (6.8)
and "confirmed broken" (4.0) was neither wide nor clean, unlike every
other threshold in this project's silhouette work. Rather than force an
unreliable trigger, detect_person_mask always runs both the normal and
an aggressively-sharpened pass and takes their union — verified directly
on all three real test photos that this costs only 0.05-2.4% extra mask
area (fine detail the sharper pass resolves better, like hair wisps —
not scattered noise: checked the added pixels' connected-component sizes
directly, the overwhelming majority fell under the 100px floor
_clean_segmentation_mask already applies).
"""
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageFilter

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
MODEL_CACHE_PATH = Path.home() / ".cache" / "anatomy3d-print" / "selfie_multiclass_256x256.tflite"

# Confirmed directly against all three real test photos (not guessed from
# documentation): category 0 is background; 1 (hair), 2 (body-skin), 3
# (face-skin), 4 (clothes), and 5 (accessories/other, seen on one of the
# three photos — a wristwatch) all appear only within the photographed
# person's own silhouette. "Is this any category but background" is the
# person mask.
BACKGROUND_CATEGORY = 0


def _ensure_model() -> str:
    if MODEL_CACHE_PATH.exists() and MODEL_CACHE_PATH.stat().st_size > 1_000_000:
        return str(MODEL_CACHE_PATH)
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(MODEL_CACHE_PATH) + ".part"
    urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, MODEL_CACHE_PATH)
    return str(MODEL_CACHE_PATH)


_segmenter = None  # lazy singleton: model loading is slow, worth reusing across calls


def _get_segmenter() -> mp_vision.ImageSegmenter:
    global _segmenter
    if _segmenter is None:
        base_options = mp_python.BaseOptions(model_asset_path=_ensure_model())
        options = mp_vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        _segmenter = mp_vision.ImageSegmenter.create_from_options(options)
    return _segmenter


# Deliberately much stronger than preprocess.py's general UnsharpMask(2,
# 120, 3) — that one boosts contrast without haloing on an already-decent
# photo; this one exists specifically to claw back edge detail a badly
# blurred photo lost. Haloing/ringing from an aggressive sharpen like
# this would be a real concern for a COLOR classifier, but this only ever
# feeds a segmentation model (which reasons about learned features, not
# raw edge contrast the way Mahalanobis-distance color classification
# does) and its result is unioned with the unsharpened pass, never used
# alone — see this module's docstring for the measured 0.05-2.4% cost on
# clean photos.
_RESCUE_SHARPEN = ImageFilter.UnsharpMask(radius=6, percent=300, threshold=0)


def _segment_raw(rgb: np.ndarray) -> Optional[np.ndarray]:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = _get_segmenter().segment(mp_image)
    category_mask = result.category_mask.numpy_view().squeeze()
    mask = category_mask != BACKGROUND_CATEGORY
    return mask if mask.any() else None


def detect_person_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Returns a boolean (H, W) mask, True where the model reads the
    photographed person (any of hair/body-skin/face-skin/clothes/
    accessories), in the same pixel coordinate system as `rgb`. Returns
    None (does not raise) if nothing is detected as a person — letting
    the caller fall back to the color/position pipeline, same as every
    other optional-model step in this project (face/hair detection).

    Internally unions two passes (normal + aggressively sharpened) — see
    this module's docstring for why a single blur-detection threshold to
    pick one or the other doesn't exist cleanly."""
    normal = _segment_raw(rgb)
    sharpened_rgb = np.array(Image.fromarray(rgb).filter(_RESCUE_SHARPEN))
    sharpened = _segment_raw(sharpened_rgb)
    if normal is None:
        return sharpened
    if sharpened is None:
        return normal
    return normal | sharpened
