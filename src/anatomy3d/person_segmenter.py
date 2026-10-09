"""Real per-pixel subject/background separation via MediaPipe's Selfie
Multiclass Segmenter for people and its DeepLabV3 Image Segmenter for
animals — same Apache-2.0 MediaPipe model family, same GCS model bucket,
same ImageSegmenter task API as the Hair Segmenter already used in
hair_features.py. Not a new dependency or license to clear (see README's
"Why from scratch" section).

Animals are a real, intended subject for this pipeline, not an edge case
to reject. Found directly: a MediaPipe test photo of children on a pony
showed the pony's head and body only partially included (ghosted, not
solid) — the selfie model has no animal categories at all, so it was
guessing. DeepLabV3 (PASCAL VOC's 21 classes: background, person, and 20
object categories including bird/cat/cow/dog/horse/sheep) gives a real,
trained signal for those categories instead. Both models run through the
same pipeline below (sharpened-pass rescue, hysteresis, crop-refine) and
their results are unioned, so an animal subject gets the identical
treatment a human subject does, not a lesser fallback.

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
from scipy import ndimage

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

ANIMAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "deeplab_v3/float32/latest/deeplab_v3.tflite"
)
ANIMAL_MODEL_CACHE_PATH = Path.home() / ".cache" / "anatomy3d-print" / "deeplab_v3.tflite"

# DeepLabV3's 21 PASCAL VOC categories, confirmed directly (not guessed
# from documentation): segmenting a real photo of a horse produced
# category 13 — this is the standard PASCAL VOC 2012 class order (0
# background, 1 aeroplane, 2 bicycle, 3 bird, 4 boat, 5 bottle, 6 bus,
# 7 car, 8 cat, 9 chair, 10 cow, 11 diningtable, 12 dog, 13 horse,
# 14 motorbike, 15 person, 16 pottedplant, 17 sheep, 18 sofa, 19 train,
# 20 tvmonitor). Only the actual animal categories go in the animal
# channel — the rest (vehicles, furniture, plants, "person" — already
# handled by the selfie model above) are not subjects this pipeline
# treats as foreground.
ANIMAL_CATEGORIES = (3, 8, 10, 12, 13, 17)  # bird, cat, cow, dog, horse, sheep


def _ensure_model(url: str, cache_path: Path) -> str:
    if cache_path.exists() and cache_path.stat().st_size > 1_000_000:
        return str(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(cache_path) + ".part"
    urlretrieve(url, tmp_path)
    os.replace(tmp_path, cache_path)
    return str(cache_path)


_segmenter = None  # lazy singleton: model loading is slow, worth reusing across calls
_animal_segmenter = None


def _get_segmenter() -> mp_vision.ImageSegmenter:
    global _segmenter
    if _segmenter is None:
        base_options = mp_python.BaseOptions(model_asset_path=_ensure_model(MODEL_URL, MODEL_CACHE_PATH))
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


def _get_animal_segmenter() -> mp_vision.ImageSegmenter:
    global _animal_segmenter
    if _animal_segmenter is None:
        base_options = mp_python.BaseOptions(
            model_asset_path=_ensure_model(ANIMAL_MODEL_URL, ANIMAL_MODEL_CACHE_PATH)
        )
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options, output_category_mask=True, output_confidence_masks=True
        )
        _animal_segmenter = mp_vision.ImageSegmenter.create_from_options(options)
    return _animal_segmenter


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

# A limb the camera frame cuts off (an arm reaching past the photo's own
# edge, not a real wrist/fingertip boundary) genuinely confuses the
# model — confirmed directly: a forearm reaching toward a photo's right
# edge measured a smooth, real gradient from 0.8 down to 0.22-0.3 right
# at the edge, well above the flat ~0.08-0.12 noise floor measured on
# plain backgrounds elsewhere in this module, but still under
# PERSON_ALPHA_THRESHOLD — so the hard cutoff was dropping real arm
# pixels, visibly "missing some of the hand" in a rendered cutout. This
# is hysteresis thresholding (the same technique Canny edge detection
# uses for a confident core with a real, elevated-but-fading boundary):
# a pixel counts as person if it crosses PERSON_ALPHA_THRESHOLD itself,
# or is connected through a chain of above-this-floor pixels to one that
# does. Set comfortably above the measured noise floor so flat
# background never bridges into a real mask.
PERSON_ALPHA_LOW_THRESHOLD = 0.2

# Reused as-is for the animal channel below: both models' confidence
# masks are a softmax over their own categories (sum to 1 per pixel), so
# the same probability-space thresholds transfer without recalibration.


def _hysteresis_mask(alpha: np.ndarray) -> np.ndarray:
    strong = alpha > PERSON_ALPHA_THRESHOLD
    weak = alpha > PERSON_ALPHA_LOW_THRESHOLD
    labeled, n = ndimage.label(weak)
    if n == 0:
        return strong
    strong_labels = np.unique(labeled[strong])
    strong_labels = strong_labels[strong_labels != 0]
    return np.isin(labeled, strong_labels)


def _segment_raw_person(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """One inference pass, both outputs: (boolean mask, float32 alpha in
    [0, 1]). Alpha is 1 - background confidence — see detect_person_alpha
    for why that, not the hair-category confidence alone, is the right
    per-pixel "how much person is here" signal. The boolean mask is a
    hysteresis threshold of alpha (see PERSON_ALPHA_LOW_THRESHOLD), NOT
    the model's own category_mask — see that constant's docstring for
    the real, measured bug in trusting category_mask's argmax directly."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = _get_segmenter().segment(mp_image)
    alpha = 1.0 - result.confidence_masks[BACKGROUND_CATEGORY].numpy_view().squeeze()
    mask = _hysteresis_mask(alpha)
    if not mask.any():
        return None
    return mask, alpha


def _segment_raw_animal(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """Same shape of result as _segment_raw_person, via DeepLabV3's
    animal categories instead of the selfie model's person categories.
    Alpha is the summed confidence across ANIMAL_CATEGORIES (the animal
    equivalent of "1 - background": both models' confidence masks are a
    softmax over their own categories, so summing the subject categories
    and subtracting background from 1 are the same quantity)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = _get_animal_segmenter().segment(mp_image)
    confidences = [result.confidence_masks[i].numpy_view().squeeze() for i in ANIMAL_CATEGORIES]
    alpha = np.clip(np.sum(confidences, axis=0), 0.0, 1.0)
    mask = _hysteresis_mask(alpha)
    if not mask.any():
        return None
    return mask, alpha


def _segment_coarse(rgb: np.ndarray, raw_fn) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """The normal+sharpened union, factored out of detect_person_mask so
    _refine_crop (and detect_person_alpha) can reuse it on an image crop
    too, not just the full photo. `raw_fn` is _segment_raw_person or
    _segment_raw_animal — the same pipeline runs both channels, see this
    module's docstring."""
    normal = raw_fn(rgb)
    sharpened_rgb = np.array(Image.fromarray(rgb).filter(_RESCUE_SHARPEN))
    sharpened = raw_fn(sharpened_rgb)
    if normal is None:
        return sharpened
    if sharpened is None:
        return normal
    normal_mask, normal_alpha = normal
    sharp_mask, sharp_alpha = sharpened

    # Background the clean (unsharpened) pass finds fully enclosed by the
    # detected person — no path out to the photo's edge — is a real
    # anatomical gap (between fingers, an "OK" gesture's circle, the open
    # floor/wall between two separated legs), not photo noise. Confirmed
    # directly on a real standing-person photo: the clean pass correctly
    # read the gap between the legs as background (alpha 0.13), but the
    # aggressive sharpen pass confidently relabeled the whole enclosed
    # pocket as person (alpha 0.90-0.97) — ringing from radius=6,
    # percent=300 sharpening right at the gap's edges, not real rescued
    # detail. So the sharpened pass is only trusted to ADD new person
    # pixels in background that's still open to the rest of the frame;
    # an enclosed pocket stays whatever the clean pass called it,
    # regardless of what the sharpened pass says.
    labeled_bg, _ = ndimage.label(~normal_mask)
    border_labels = set(np.unique(labeled_bg[0, :])) | set(np.unique(labeled_bg[-1, :]))
    border_labels |= set(np.unique(labeled_bg[:, 0])) | set(np.unique(labeled_bg[:, -1]))
    border_labels.discard(0)
    enclosed_bg = (labeled_bg != 0) & ~np.isin(labeled_bg, list(border_labels))

    # The sharpened pass only gets to raise alpha where IT crosses its own
    # hard-mask threshold AND that pixel isn't in a clean-pass-enclosed
    # background pocket — not everywhere via a raw elementwise max.
    # Confirmed directly: the aggressive sharpen also amplifies
    # texture/shadow noise in a flat (non-enclosed) background wall into
    # weak partial "person" confidence (0.08 -> up to 0.29) — never
    # enough to cross the hard mask's 0.5 threshold, but enough to show
    # up as visible background smudging in the soft-alpha cutout, since
    # that output isn't thresholded at all.
    rescue_allowed = sharp_mask & ~enclosed_bg
    alpha = np.where(rescue_allowed, np.maximum(normal_alpha, sharp_alpha), normal_alpha)
    return normal_mask | rescue_allowed, alpha


# The model's own fixed internal processing resolution (the "256x256" in
# this file's model name) is spent on the whole input frame, not the
# person in it — confirmed directly: feeding a 2000x1500 image, the
# returned confidence mask comes back upsampled to that full 2000x1500
# size, but the actual inference ran at the model's small fixed internal
# resolution before that upsampling. When the person (and especially
# something fine like fingers) only fills a fraction of the photo, most
# of that fixed budget is spent resolving empty background, not finger
# boundaries. Re-running the same segmenter on a tight crop around the
# already-detected person spends that same fixed budget on the subject
# almost exclusively, which is strictly more resolution on the part that
# needs it. Skipped when the subject already fills most of the frame
# (_CROP_REFINE_MAX_FRAC) since a crop then buys nothing.
_CROP_REFINE_MAX_FRAC = 0.6
_CROP_REFINE_MARGIN_FRAC = 0.2


def _refine_crop(
    rgb: np.ndarray, mask: np.ndarray, alpha: np.ndarray, raw_fn
) -> "tuple[np.ndarray, np.ndarray]":
    h, w = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return mask, alpha
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1
    if bbox_h >= _CROP_REFINE_MAX_FRAC * h and bbox_w >= _CROP_REFINE_MAX_FRAC * w:
        return mask, alpha
    margin_y, margin_x = int(bbox_h * _CROP_REFINE_MARGIN_FRAC), int(bbox_w * _CROP_REFINE_MARGIN_FRAC)
    cy0, cy1 = max(0, y0 - margin_y), min(h, y1 + margin_y + 1)
    cx0, cx1 = max(0, x0 - margin_x), min(w, x1 + margin_x + 1)
    refined = _segment_coarse(rgb[cy0:cy1, cx0:cx1], raw_fn)
    if refined is None:
        return mask, alpha
    refined_mask, refined_alpha = refined
    # Same rule as the sharpened-pass merge above, for the same reason:
    # the crop pass only gets to raise alpha where IT crosses its own
    # hard-mask threshold, not everywhere via a raw elementwise max.
    # Confirmed directly on a photo with a small, distant subject: a
    # blanket max let a visible rectangular "halo" tint the whole crop
    # region, because re-cropping tight around a small subject changes
    # the model's field of view enough to generally nudge up background
    # confidence across the crop, not just at the subject's own edges —
    # the same kind of noise the sharpened pass produces, just from a
    # context shift instead of sharpening artifacts.
    out_alpha = alpha.copy()
    out_alpha[cy0:cy1, cx0:cx1] = np.where(
        refined_mask, np.maximum(alpha[cy0:cy1, cx0:cx1], refined_alpha), alpha[cy0:cy1, cx0:cx1]
    )
    return _hysteresis_mask(out_alpha), out_alpha


# A dense multi-subject photo confuses the model in a way cropping
# around the WHOLE group can't fix, and _refine_crop's max-only merge
# wouldn't fix it either even if it tried — confirmed directly on a
# real photo of four pets (two cats, two dogs) side by side: a patch of
# plain pink studio backdrop between a poodle and a corgi scored 0.68
# "dog" confidence (above the hard-mask threshold) from the full-frame
# pass alone, not from any pipeline logic here. Re-running the model on
# a tight crop around just the corgi (excluding the other pets) read
# that same patch at 0.08 — correctly background. The model isn't
# missing resolution there, it's reading the OTHER nearby animals as
# context for "probably more animal nearby." So unlike _refine_crop
# (which only ever raises alpha, since it exists to rescue missed
# detail), this one needs to lower it too — it replaces alpha outright
# within each subject's own tight crop, trusting that an
# uncluttered, single-subject view is more reliable there than the
# crowded full-frame one. Guarded two ways: only runs when there are 2+
# separately-detected subjects to begin with (an ordinary single person
# or animal never touches this path), and never overwrites pixels
# already claimed by a DIFFERENT detected subject, so one subject's
# crop can't degrade a neighboring subject's own correct reading.
#
# Detecting those 2+ subjects can't just look for separate components in
# the regular mask — confirmed directly: on the four-pets photo, the
# false-positive confidence bridging two of them was itself enough to
# connect all four pets into a single blob (1 component, not 4), so a
# check for 2+ ordinary mask components never even saw the problem.
# Seeding from a stricter confidence floor instead avoids this: real fur
# at the four pets' own cores read 0.85+ throughout, well above the
# bridging patch's 0.68 peak, so a core this strict separates them
# correctly regardless of how confident the bridge in between is.
_MULTI_SUBJECT_CORE_THRESHOLD = 0.85
_MULTI_SUBJECT_MIN_COMPONENT_PX = 50

# A subject's own crop must cover its WHOLE visible extent, not just its
# strict-confidence core — confirmed directly: a cat sitting next to a
# much larger, more confidently-read corgi had its own core reduced to a
# 28px-tall sliver (just its whiskers), far smaller than its actual
# face. That tiny core's own crop+margin never reached the cat's own
# ears/forehead, while the corgi's much bigger crop (sized off its own,
# much larger core) did reach there instead, correctly read "not
# corgi" from its own perspective, and wrongly dragged the cat's own
# face down with it. _MULTI_SUBJECT_EXTENT_THRESHOLD finds each
# subject's true territory with a Voronoi-style partition: any pixel
# at least this confident is assigned to whichever significant core is
# nearest, not left to whichever subject's crop happens to reach it
# first. Set comfortably above the ~0.1-0.2 background noise floor
# measured elsewhere in this module, well below both the cat's own
# ~0.4 mean facial confidence and the bridging patch's 0.68 that this
# whole mechanism exists to catch.
_MULTI_SUBJECT_EXTENT_THRESHOLD = 0.3

# How far a subject's territory is allowed to extend past its own core,
# as a fraction of the photo's shorter side — see the pad comment at its
# use site for the real failure this bounds. Measured directly against
# two real, conflicting cases rather than guessed: too small (0.03-0.05)
# left a penguin's own head unreached, reproducing the original
# between-heads bridging; too large (0.15, the value this replaced)
# let the territory extension itself reach far enough back toward a
# neighbor to partly reopen the corgi/poodle bridging fix. 0.07 is the
# measured value that cleanly resolves both at once.
_MULTI_SUBJECT_TERRITORY_PAD_FRAC = 0.07

# Territory expansion is for subjects whose own core is too small to
# represent their real extent (the cat's own core was its 29x74px
# whiskers, not its whole face) — a subject whose core is already a
# reasonable size doesn't need it and does measurably worse with it.
# Confirmed directly: the corgi's own core (382x399, plenty to crop
# tight around on its own) still had its territory-expanded crop reach
# wide enough to pull the poodle back into view, reproducing the
# original bridging confusion the whole per-subject mechanism exists to
# remove. A core at least this large in both dimensions uses its own
# bbox directly instead, the same as the first, successful version of
# this fix did before territory expansion was added for the cat.
_MULTI_SUBJECT_SELF_SUFFICIENT_CORE_PX = 150

# Re-counting subjects after each round (instead of once) matters
# because fixing one round's confusion can reveal or resolve another —
# a subject's own core can grow once a neighboring false claim on it is
# removed, which can in turn change the count or shape of what's left
# to resolve. Each round is its own fresh "how many subjects are here,
# and what does each one's own crop say" pass; it stops once nothing
# changes by more than the tolerance (the result has stabilized) or
# after a bounded number of rounds, whichever comes first — never
# unbounded, since each round is a full extra set of model calls.
_MULTI_SUBJECT_MAX_ROUNDS = 3
_MULTI_SUBJECT_CONVERGENCE_TOL = 0.01


def _refine_per_subject(rgb: np.ndarray, mask: np.ndarray, alpha: np.ndarray, raw_fn) -> "tuple[np.ndarray, np.ndarray]":
    h, w = mask.shape
    current = alpha

    for _round in range(_MULTI_SUBJECT_MAX_ROUNDS):
        core = current > _MULTI_SUBJECT_CORE_THRESHOLD
        labeled, n = ndimage.label(core)
        if n == 0:
            break
        sizes = ndimage.sum(core, labeled, index=range(1, n + 1))
        significant = [i + 1 for i, s in enumerate(sizes) if s >= _MULTI_SUBJECT_MIN_COMPONENT_PX]
        if len(significant) < 2:
            break  # one subject (or none) left to resolve -- this pass is done

        # Every significant core is a Voronoi seed; every pixel at least
        # _MULTI_SUBJECT_EXTENT_THRESHOLD confident is assigned to
        # whichever seed is nearest — see that constant's docstring for
        # why this, not each core's own (possibly tiny) bounding box,
        # sizes each subject's crop.
        significant_seeds = np.where(np.isin(labeled, significant), labeled, 0)
        nearest_idx = ndimage.distance_transform_edt(
            significant_seeds == 0, return_distances=False, return_indices=True
        )
        territory_of = significant_seeds[tuple(nearest_idx)]
        extent = current > _MULTI_SUBJECT_EXTENT_THRESHOLD

        # Each subject gets its own tight crop+margin, sized from its
        # own territory, and its own independent reading there, same as
        # a single-subject refine. best_claim collects, per pixel, the
        # MOST SKEPTICAL reading ANY subject's own crop gave it within
        # ITS OWN territory — so where two subjects' crops overlap (both
        # reach the same pixel), that pixel is checked against both
        # before anything is decided. Found directly on a puppy/kitten
        # photo: a patch of dirt was golden-tan, close enough to the
        # puppy's own fur color that the puppy's own isolated crop read
        # it as confidently puppy (0.9998) — but the kitten's isolated
        # crop (different fur color, no such confusion) read the same
        # patch as confidently background (0.19). Taking the subjects'
        # MAX would hand the pixel to whichever one hallucinates harder;
        # taking the MIN means any one subject's skepticism can veto a
        # neighbor's overconfidence, consistent with the "only ever
        # lower, never raise" rule the next step already applies.
        # Pixels no subject's territory reaches at all stay unclaimed
        # (sentinel +inf) and are left untouched.
        best_claim = np.full((h, w), np.inf, dtype=np.float32)
        any_crop_ran = False
        for component_id in significant:
            # The territory itself has to stay bounded near this
            # subject's OWN core, not just wherever the Voronoi
            # partition happens to reach — confirmed directly: the
            # corgi's own territory, sized this way with no bound,
            # ballooned past its own 399px-wide core out to 813px wide
            # (nearly the whole photo), because "closest significant
            # seed" still reaches a long way across any broad,
            # ambiguous, weakly-elevated area. That undid the whole
            # point of cropping tight around just the corgi: its own
            # crop was wide enough to see the other pets again and
            # reproduced the original confusion. Capping how far
            # territory can extend past the core keeps a small core
            # (the cat's own whisker-sized core) free to grow enough to
            # reach the rest of its face, while a core that's already
            # sizeable (the corgi's) stays close to its own real extent.
            core_component = labeled == component_id
            core_ys, core_xs = np.where(core_component)
            core_y0, core_y1 = core_ys.min(), core_ys.max()
            core_x0, core_x1 = core_xs.min(), core_xs.max()
            core_h, core_w = core_y1 - core_y0 + 1, core_x1 - core_x0 + 1

            self_sufficient = (
                core_h >= _MULTI_SUBJECT_SELF_SUFFICIENT_CORE_PX and core_w >= _MULTI_SUBJECT_SELF_SUFFICIENT_CORE_PX
            )
            # The bounded territory (same pad used for a small core) also
            # supplements a self-sufficient one — confirmed directly on a
            # penguin photo: a penguin's body core was plenty big on its
            # own, but its head and neck, being a separate and slightly
            # less confident region, fell outside that core entirely, so
            # a crop sized from the body alone never reached the gap
            # between two birds' heads. Bounding this the same way
            # already-validated for small cores keeps it from
            # reintroducing the earlier reach-into-a-neighbor problem.
            pad = int(_MULTI_SUBJECT_TERRITORY_PAD_FRAC * min(h, w))
            bound_y0, bound_y1 = max(0, core_y0 - pad), min(h, core_y1 + pad + 1)
            bound_x0, bound_x1 = max(0, core_x0 - pad), min(w, core_x1 + pad + 1)
            territory_full = extent & (territory_of == component_id)
            territory = np.zeros_like(territory_full)
            territory[bound_y0:bound_y1, bound_x0:bound_x1] = territory_full[bound_y0:bound_y1, bound_x0:bound_x1]
            terr_ys, terr_xs = np.where(territory)

            if self_sufficient:
                # A core this size can be cropped tight around its own
                # bbox directly — no need to grow it to find its own
                # main body. Ownership of the ambiguous pixels around it
                # (ambiguous enough to never have reached the strict
                # core threshold themselves, the exact pixels this whole
                # mechanism exists to correct) is everything in the crop
                # that isn't ANOTHER subject's own confirmed core — the
                # same rule the first, working version of this fix used
                # throughout. The bounded territory only extends the
                # crop bbox itself, for cases like a head sitting just
                # outside the body's own core.
                y0, y1, x0, x1 = core_y0, core_y1, core_x0, core_x1
                if terr_ys.size:
                    y0, y1 = min(y0, terr_ys.min()), max(y1, terr_ys.max())
                    x0, x1 = min(x0, terr_xs.min()), max(x1, terr_xs.max())
            else:
                if terr_ys.size == 0:
                    continue
                y0, y1, x0, x1 = terr_ys.min(), terr_ys.max(), terr_xs.min(), terr_xs.max()
            bbox_h, bbox_w = y1 - y0 + 1, x1 - x0 + 1
            if bbox_h >= _CROP_REFINE_MAX_FRAC * h and bbox_w >= _CROP_REFINE_MAX_FRAC * w:
                continue
            margin_y, margin_x = int(bbox_h * _CROP_REFINE_MARGIN_FRAC), int(bbox_w * _CROP_REFINE_MARGIN_FRAC)
            cy0, cy1 = max(0, y0 - margin_y), min(h, y1 + margin_y + 1)
            cx0, cx1 = max(0, x0 - margin_x), min(w, x1 + margin_x + 1)
            refined = _segment_coarse(rgb[cy0:cy1, cx0:cx1], raw_fn)
            if refined is None:
                continue
            _, refined_alpha = refined
            any_crop_ran = True
            if self_sufficient:
                crop_labels = labeled[cy0:cy1, cx0:cx1]
                own_territory_in_crop = (crop_labels == 0) | (crop_labels == component_id)
            else:
                own_territory_in_crop = territory[cy0:cy1, cx0:cx1]
            region = best_claim[cy0:cy1, cx0:cx1]
            best_claim[cy0:cy1, cx0:cx1] = np.where(
                own_territory_in_crop, np.minimum(region, refined_alpha), region
            )

        if not any_crop_ran:
            break

        # Only ever LOWER alpha here, never raise it, exactly where some
        # subject's own crop actually examined the pixel. Found
        # directly: an unconditional replace fixed the four-pets
        # bridging case but broke a single real person's photo — a
        # dark-sleeved arm's own core fragmented away from the rest of
        # the body (a local confidence dip, not a second subject), and
        # cropping tight around just that fragment hit the exact same
        # context-shift problem _refine_crop's own gating already
        # exists to prevent: the zoomed-in view nudged up confidence
        # over a real patch of wall. The minimum() here can't
        # reintroduce that failure mode (it only ever removes
        # confidence, never adds it) while still correcting the
        # original bug, since the false-positive bridge between two
        # real subjects was a case of too-HIGH confidence to begin
        # with. A contested pixel with no confident claim from any
        # subject (best_claim stays low) is left low too — omitted
        # rather than guessed into either one's silhouette.
        claimed = np.isfinite(best_claim)
        next_alpha = np.where(claimed, np.minimum(current, best_claim), current)

        changed = np.abs(next_alpha - current).max()
        current = next_alpha
        if changed < _MULTI_SUBJECT_CONVERGENCE_TOL:
            break

    return _hysteresis_mask(current), current


def _segment_full(rgb: np.ndarray, raw_fn) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    coarse = _segment_coarse(rgb, raw_fn)
    if coarse is None:
        return None
    mask, alpha = _refine_crop(rgb, *coarse, raw_fn)
    return _refine_per_subject(rgb, mask, alpha, raw_fn)


def _segment_full_person(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    return _segment_full(rgb, _segment_raw_person)


def _segment_full_animal(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    return _segment_full(rgb, _segment_raw_animal)


def _segment_full_animal_robust(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """DeepLabV3 isn't rotation-robust the way the selfie model turned
    out to be — confirmed directly, not assumed: on a real MediaPipe
    test photo rotated 90 degrees, the animal channel found nothing at
    all (completely empty), while the same rotation applied to a human
    photo left the selfie model's result just as solid as the upright
    version (mask area barely moved, 8.0% -> 8.6%). An animal
    photographed sideways shouldn't get a worse result than a person
    photographed sideways. So when the upright pass finds nothing, retry
    at 90/180/270 degrees before giving up — cheap in the common case
    (no animal in the photo, the upright pass already returns nothing
    quickly) and brings a sideways animal photo back to the same solid
    result an upright one gets."""
    upright = _segment_full_animal(rgb)
    if upright is not None:
        return upright
    for k in (1, 2, 3):
        rotated = _segment_full_animal(np.ascontiguousarray(np.rot90(rgb, k)))
        if rotated is not None:
            mask, alpha = rotated
            return np.ascontiguousarray(np.rot90(mask, -k)), np.ascontiguousarray(np.rot90(alpha, -k))
    return None


# A dense cluster of small enclosed background pockets is texture-
# confusion noise, not a deliberate gap — confirmed directly on a real
# photo: a horse's woven saddle blanket produced 14 separate tiny
# enclosed holes packed into one ~50x80px patch (most within 2-15px of
# their nearest neighbor), and the model gave every one of them zero
# elevated confidence (alpha 0.04-0.24, right at the background noise
# floor measured elsewhere in this module) — not a borderline call, the
# model simply has no category for saddle tack. A plain size cutoff
# can't tell this apart from a real gap: these holes (up to 130px) are
# individually *larger* than a real OK-sign-gesture gap's own smaller
# half (119px) on another test photo. But every real gap found in this
# project's testing is a single compact void, and even where
# anti-aliasing visibly splits one into two pieces (both hands'
# OK-sign circles did), those pieces stayed 35-51px apart — well
# outside the dilation radius used here. So cluster on proximity
# instead of area: only 3 or more small enclosed pockets within
# _NOISE_CLUSTER_DILATION_PX of each other count as noise: an isolated
# pocket or a close pair (exactly what a real gap split in two looks
# like) is left alone.
_NOISE_CLUSTER_DILATION_PX = 10
_NOISE_CLUSTER_MIN_COMPONENTS = 3


def _fill_noise_clusters(mask: np.ndarray, alpha: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    bg = ~mask
    labeled, n = ndimage.label(bg)
    if n == 0:
        return mask, alpha
    border_labels = set(np.unique(labeled[0, :])) | set(np.unique(labeled[-1, :]))
    border_labels |= set(np.unique(labeled[:, 0])) | set(np.unique(labeled[:, -1]))
    border_labels.discard(0)
    enclosed = (labeled != 0) & ~np.isin(labeled, list(border_labels))
    if not enclosed.any():
        return mask, alpha

    dilated = ndimage.binary_dilation(enclosed, iterations=_NOISE_CLUSTER_DILATION_PX)
    dilated_labels, _ = ndimage.label(dilated)
    group_members: "dict[int, set]" = {}
    for group_id, component_id in zip(dilated_labels[enclosed], labeled[enclosed]):
        group_members.setdefault(int(group_id), set()).add(int(component_id))

    noisy_ids = [
        cid for members in group_members.values() if len(members) >= _NOISE_CLUSTER_MIN_COMPONENTS for cid in members
    ]
    if not noisy_ids:
        return mask, alpha
    fill = np.isin(labeled, noisy_ids)
    return mask | fill, np.where(fill, 1.0, alpha).astype(alpha.dtype)


def _combine_channels(rgb: np.ndarray) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    person = _segment_full_person(rgb)
    animal = _segment_full_animal_robust(rgb)
    if person is None and animal is None:
        return None
    if person is None:
        mask, alpha = animal
    elif animal is None:
        mask, alpha = person
    else:
        mask, alpha = person[0] | animal[0], np.maximum(person[1], animal[1])
    return _fill_noise_clusters(mask, alpha)


def detect_person_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Returns a boolean (H, W) mask, True where the model reads the
    photographed subject — a person (any of hair/body-skin/face-skin/
    clothes/accessories) or an animal (bird/cat/cow/dog/horse/sheep,
    see ANIMAL_CATEGORIES) — in the same pixel coordinate system as
    `rgb`. Returns None (does not raise) if nothing is detected, letting
    the caller fall back to the color/position pipeline, same as every
    other optional-model step in this project (face/hair detection).

    Internally unions the person and animal channels, each of which
    itself unions two full-frame passes (normal + aggressively
    sharpened) and then a crop-refine pass — see this module's docstring
    for the sharpening and _refine_crop's docstring for the crop step.
    A final pass fills small clustered background pockets neither
    channel recognized — see _fill_noise_clusters."""
    combined = _combine_channels(rgb)
    return combined[0] if combined else None


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
    confidence that each pixel belongs to the photographed subject
    (person or animal), NOT thresholded to a hard boolean. Use this (not
    detect_person_mask) when the goal is a soft-edged cutout/compositing
    result; use detect_person_mask when a boolean is actually required
    (hole-filling, connected-component cleanup, anything feeding the 3D
    mesh pipeline, which has no notion of partial coverage). Returns
    None under the same condition detect_person_mask does. Each channel
    goes through the same full-frame-union-plus-crop-refine pipeline —
    see _segment_full — and the two channels' alphas are combined with
    elementwise max, the continuous equivalent of detect_person_mask's
    boolean OR. Also goes through the same clustered-background-pocket
    fill detect_person_mask does — see _fill_noise_clusters."""
    combined = _combine_channels(rgb)
    return combined[1] if combined else None
