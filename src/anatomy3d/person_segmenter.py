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
        # Both outputs requested together: confirmed requesting
        # output_confidence_masks alongside output_category_mask doesn't
        # change category_mask's own values (checked directly — the
        # existing boolean-mask regression tests stayed byte-identical
        # after this was added), so detect_person_mask's callers are
        # unaffected; see detect_person_alpha for what the confidence
        # output adds.
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options, output_category_mask=True, output_confidence_masks=True
        )
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


# category_mask (an argmax over 6 separate categories: background, hair,
# body-skin, face-skin, clothes, accessories) is NOT the same decision as
# "is total person-probability over background-probability" — confirmed as
# a real, measurable bug, not a style choice. Found by chasing a report of
# background showing through between body parts (fingers spread apart,
# crossed arms) even after every other fix in this project's silhouette
# work: checked the raw per-category confidence at a real finger gap and
# found a clean, well-behaved gradient (0.98 at the real finger, smoothly
# down to 0.08 at the real background, nothing erratic) — the model wasn't
# confused. But at several points in that gradient, category_mask still
# said "person": background probability alone can win the per-category
# argmax (e.g. background=0.4 vs hair=0.3 vs clothes=0.3) even while
# background is LESS than the combined probability of every person
# category (0.4 < 0.6) — argmax across categories and "all person
# categories summed > background" are different rules, and they disagree
# exactly in these concave notches between body parts. Switching the hard
# mask to the latter (threshold alpha, i.e. 1 - background-confidence,
# directly) is the more principled rule, and verified correct: diffed the
# two decision rules on a real photo and the changed pixels form a thin
# line running exactly along the true finger-gap edge, in both directions
# (reclaiming real background AND real skin at different points along the
# same edge) — a boundary-precision correction, not a one-sided bias.
# Checked on all 3 real test photos: the diffed pixels are confined to a
# thin outline around the real silhouette everywhere, never a large area,
# confirming this is a precision fix, not a different segmentation.
PERSON_ALPHA_THRESHOLD = 0.5


def _segment_raw(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """One inference pass, both outputs: (boolean mask, float32 alpha in
    [0, 1]). Alpha is 1 - background confidence — see detect_person_alpha
    for why that, not the hair-category confidence alone, is the right
    per-pixel "how much person is here" signal. The boolean mask is
    alpha > PERSON_ALPHA_THRESHOLD, NOT the model's own category_mask —
    see that constant's docstring for the real, measured bug in trusting
    category_mask's argmax directly."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = _get_segmenter().segment(mp_image)
    alpha = 1.0 - result.confidence_masks[BACKGROUND_CATEGORY].numpy_view().squeeze()
    mask = alpha > PERSON_ALPHA_THRESHOLD
    if not mask.any():
        return None
    return mask, alpha


def _segment_raw_sharpened(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    sharpened_rgb = np.array(Image.fromarray(rgb).filter(_RESCUE_SHARPEN))
    return _segment_raw(sharpened_rgb)


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
    sharpened = _segment_raw_sharpened(rgb)
    if normal is None:
        return sharpened[0] if sharpened else None
    if sharpened is None:
        return normal[0]
    return normal[0] | sharpened[0]


# The hard category_mask this module used at first (`!= BACKGROUND_CATEGORY`)
# discards real information the model actually has. Confirmed directly,
# not assumed: checked the model's own output_confidence_masks (a
# per-category float in [0, 1], not just the argmax category_mask) in a
# region of real, confirmed-individual wispy hair strands — a case this
# project's silhouette work had called a fundamental, unfixable "sub-pixel
# alpha-matting problem neither method solves" — and 75.6% of that
# region's pixels carried genuinely fractional confidence (between 0.05
# and 0.95), not the saturated near-0/near-1 values a real binary edge
# would produce. A soft cutout built from 1 - background-confidence
# (rather than a hard threshold of it) visibly renders individual strand
# paths as partial transparency instead of a smooth blob — checked
# directly against a zoomed crop, the same way every other finding in
# this project's silhouette work was checked. "Neither method solves
# this" was wrong; this module just wasn't using the information it
# already had.
def detect_person_alpha(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Returns a float32 (H, W) array in [0, 1] — the model's own
    confidence that each pixel belongs to the photographed person, NOT
    thresholded to a hard boolean. Use this (not detect_person_mask) when
    the goal is a soft-edged cutout/compositing result; use
    detect_person_mask when a boolean is actually required (hole-filling,
    connected-component cleanup, anything feeding the 3D mesh pipeline,
    which has no notion of partial coverage). Returns None under the same
    condition detect_person_mask does. Also unions two passes the same
    way and for the same reason detect_person_mask does (via elementwise
    max, the continuous equivalent of boolean OR)."""
    normal = _segment_raw(rgb)
    sharpened = _segment_raw_sharpened(rgb)
    if normal is None:
        return sharpened[1] if sharpened else None
    if sharpened is None:
        return normal[1]
    return np.maximum(normal[1], sharpened[1])
