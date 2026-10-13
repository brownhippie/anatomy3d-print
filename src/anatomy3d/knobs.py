"""The builder's tunable numbers, for the server to study on its own
(webapp/study_api.py): each one's module, name, the range it may move in and
what it shapes. A study moves them, the critics in study.py score the
results, and the values you approve are written to tuned.json, which every
build then uses.

A value is applied only while the code still has the default it was tuned
from: change a constant in the code and its old tuned value is dropped, not
applied on top of the new one.
"""
import contextlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Knob:
    module: str  # under anatomy3d
    name: str
    lo: float
    hi: float
    area: str
    what: str
    integer: bool = False

    @property
    def key(self) -> str:
        return f"{self.module}.{self.name}"


_SR, _BR = "silhouette_relief", "body_reference"
KNOBS = [
    # hair
    Knob(_SR, "_HAIR_SHELL_MAX", 0.03, 0.07, "hair", "thickest hair lying round the skull (m)"),
    Knob(_SR, "_HAIR_FEATHER", 0.003, 0.012, "hair", "hair over skin feathers out over this (m)"),
    Knob(_SR, "_HAIR_SMOOTH", 0.002, 0.008, "hair", "hair and hat surfaces smoothed across their edges (m)"),
    Knob(_SR, "_HAIR_THIN", 0.006, 0.02, "hair", "hair thinner than this is the head's own outline (m)"),
    Knob(_SR, "_HAIR_FACE_MEET", 0.006, 0.02, "hair", "hair this near the face comes down to the cheek (m)"),
    Knob(_SR, "_HAIR_NECK_BEHIND", 0.003, 0.015, "hair", "hair beside the neck lies this far behind its front (m)"),
    Knob(_SR, "_HAIR_NAPE_EDGE", 0.002, 0.008, "hair", "hairline at the nape ends over this (m)"),
    Knob(_SR, "_HAIR_FRINGE", 0.006, 0.02, "hair", "a fringe over the forehead, at most (m)"),
    Knob(_SR, "_HAIR_LAYER", 0.004, 0.015, "hair", "long hair over shoulders or chest (m)"),
    Knob(_SR, "_HAIR_HANG", 0.006, 0.02, "hair", "long hair beside the neck, half-thickness at least (m)"),
    Knob(_SR, "_REF_HAIR", 0.006, 0.02, "hair", "most the head grows front to back for hair (m)"),
    Knob(_SR, "_HAIR_BACK_FROM_SIDES", 0.5, 1.3, "hair", "round the back: this share of the hair's reach past the skull at the sides"),
    # the hair's pattern
    Knob(_SR, "_HAIR_STRAND_W", 0.003, 0.01, "hair pattern", "a sculpted strand (lock) is this wide (m)"),
    Knob(_SR, "_HAIR_STRAND_LEN", 0.02, 0.1, "hair pattern", "a strand runs this long before the next (m)"),
    Knob(_SR, "_HAIR_STRAND_DEPTH", 0.0005, 0.003, "hair pattern", "grooves between strands, this deep either way (m)"),
    Knob(_SR, "_HAIR_WAVE_DEPTH", 0.001, 0.006, "hair pattern", "a wave's crest stands this proud (m)"),
    Knob(_SR, "_HAIR_WAVE_LEN", 0.06, 0.2, "hair pattern", "a wave's length when the photo didn't measure one (m)"),
    Knob(_SR, "_HAIR_CURL", 0.006, 0.025, "hair pattern", "a curl is this across (m)"),
    Knob(_SR, "_HAIR_CURL_DEPTH", 0.001, 0.005, "hair pattern", "a curl stands this proud (m)"),
    Knob(_SR, "_HAIR_COIL", 0.003, 0.012, "hair pattern", "a tight coil is this across (m)"),
    Knob(_SR, "_HAIR_COIL_DEPTH", 0.0005, 0.003, "hair pattern", "a coil stands this proud (m)"),
    Knob(_SR, "_HAIRLINE_EDGE", 0.002, 0.01, "hair pattern", "the pattern ends over this at the hairline (m)"),
    # what's worn on the head
    Knob(_SR, "_HAT_PEAK", 0.2, 0.45, "headwear", "a cap's peak reaches this far forward (x face width)"),
    Knob(_SR, "_MASK_SPAN", 0.02, 0.06, "headwear", "a face mask spans hollows narrower than this (m)"),
    Knob(_SR, "_GLASSES_SPAN", 0.01, 0.03, "headwear", "glasses' lenses span the eye sockets like this (m)"),
    Knob(_SR, "_GLASSES_FRAME", 0.001, 0.004, "headwear", "glasses' frame stands this proud (m)"),
    # neck and what covers it
    Knob(_SR, "_NECK_COLLAR", 0.005, 0.015, "neck", "a collar stands this far off the neck (m)"),
    Knob(_SR, "_NECK_LACE", 0.0015, 0.005, "neck", "a necklace stands this proud (m)"),
    Knob(_SR, "_NECK_WEAR_MAX", 0.015, 0.035, "neck", "a scarf or roll neck stands at most this far off (m)"),
    Knob(_SR, "_NECK_COLLAR_BELOW", 0.01, 0.03, "neck", "a collar reaches this far below the neck's base (m)"),
    Knob(_SR, "_NECK_COLLAR_RISE", 0.005, 0.02, "neck", "round the back a collar rides this much higher (m)"),
    # ears
    Knob(_SR, "_REF_EAR_HAIR", 0.008, 0.025, "ears", "an ear whose base grew out more than this is under hair (m)"),
    Knob(_SR, "_REF_EAR_SINK", 0.002, 0.008, "ears", "the ear's skirt sinks into the head by this (m)"),
    Knob(_SR, "_REF_EAR_THICK", 0.003, 0.006, "ears", "an ear's thickness (m)"),
    Knob(_BR, "_EAR_DROP", 0.03, 0.1, "ears", "the lobe's end: outline stepping in by this (x brow-to-nose)"),
    Knob(_BR, "_EAR_KINK", 0.03, 0.08, "ears", "the ear's top: outline turning out by this (x brow-to-nose)"),
    Knob(_BR, "_EAR_BEND", 0.008, 0.025, "ears", "an ear's swing reaches all of it this far from its root (m)"),
    Knob(_BR, "_EAR_UNDER_HAIR", 0.4, 0.8, "ears", "an ear's place this much hair (or hat): under it"),
    # body and clothes
    Knob(_SR, "_REF_DRAPE", 0.04, 0.12, "clothes", "cloth over a man's chest: the ball rolled over it (m)"),
    Knob(_SR, "_REF_HANG", 0.08, 0.25, "clothes", "cloth falls back at most this much per unit down"),
    Knob(_SR, "_REF_BACK_SMOOTH", 0.01, 0.03, "clothes", "clothing over the back: smoothing radius (m)"),
    Knob(_SR, "_REF_SIDE_SMOOTH", 15, 45, "clothes", "extra smoothing passes on sideways surfaces", integer=True),
    Knob(_SR, "_REF_RELIEF_MAX", 0.008, 0.025, "clothes", "photo-depth detail on top of the body (m)"),
    Knob(_SR, "_REF_EDGE_SLACK", 0.008, 0.025, "clothes", "the body's edge may pass the photo outline by this (m)"),
    Knob(_SR, "_REF_DEPTH_MARGIN", 1.0, 1.2, "clothes", "body depth over the bare reference body"),
    Knob(_SR, "_REF_HEAD_MARGIN", 1.0, 1.12, "clothes", "head depth over the bare reference head"),
    Knob(_SR, "_REF_HAND_SHRINK", 0.2, 0.5, "clothes", "hands on the body shrunk to this share about the wrist"),
    # face
    Knob(_SR, "_FACE_RELIEF_FADE_FRAC", 0.1, 0.3, "face", "face detail fades out over this share of the face"),
    Knob(_SR, "_FACE_RELIEF_SMOOTH_FRAC", 0.01, 0.03, "face", "face detail smoothing"),
    Knob(_SR, "_FACE_RELIEF_WITH_DEPTH", 0.3, 0.6, "face", "face detail strength when photo depth is on"),
]
BY_KEY = {k.key: k for k in KNOBS}
AREAS = list(dict.fromkeys(k.area for k in KNOBS))


def _module(k: Knob):
    return importlib.import_module(f"anatomy3d.{k.module}")


def _read(k: Knob):
    return getattr(_module(k), k.name)


# what the code itself says, before anything tuned is applied
DEFAULTS = {k.key: _read(k) for k in KNOBS}


def current() -> dict:
    return {k.key: _read(k) for k in KNOBS}


def clip(k: Knob, v: float):
    v = min(max(float(v), k.lo), k.hi)
    return int(round(v)) if k.integer else v


def _set(values: dict) -> None:
    for key, v in values.items():
        k = BY_KEY.get(key)
        if k is not None:
            setattr(_module(k), k.name, clip(k, v))


@contextlib.contextmanager
def applied(values: dict):
    """Builds inside use `values`; the previous ones come back after. Not
    thread-safe on its own: call it holding the lock every build takes."""
    before = current()
    _set(values)
    try:
        yield
    finally:
        _set(before)


# ---------------------------------------------------------------- tuned.json

def tuned_path() -> Path:
    import os
    import tempfile

    env = os.environ.get("ANATOMY3D_TUNED")
    if env:
        return Path(env)
    if Path("/data").is_dir():
        return Path("/data/tuned.json")
    return Path(tempfile.gettempdir()) / "anatomy3d-tuned.json"


def read_tuned() -> dict:
    try:
        return json.loads(tuned_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_tuned() -> dict:
    """Puts approved values in place (process-wide). Returns what was
    applied and what was dropped because the code's default moved since."""
    saved = read_tuned().get("values", {})
    use, stale = {}, []
    for key, entry in saved.items():
        k = BY_KEY.get(key)
        if k is None or entry.get("code_default") != DEFAULTS[key]:
            stale.append(key)
            continue
        use[key] = entry["value"]
    _set({**DEFAULTS, **use})
    return {"applied": use, "stale": stale}


def save_tuned(values: dict, source: str) -> dict:
    changed = {key: v for key, v in values.items() if key in BY_KEY and clip(BY_KEY[key], v) != DEFAULTS[key]}
    data = {
        "at": time.time(),
        "source": source,
        "values": {key: {"value": clip(BY_KEY[key], v), "code_default": DEFAULTS[key]} for key, v in changed.items()},
    }
    p = tuned_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)
    return load_tuned()


def clear_tuned() -> None:
    try:
        tuned_path().unlink()
    except FileNotFoundError:
        pass
    _set(DEFAULTS)


def describe(values: Optional[dict] = None) -> list:
    values = current() if values is None else values
    return [
        {"key": k.key, "name": k.name, "area": k.area, "what": k.what, "lo": k.lo, "hi": k.hi,
         "value": values.get(k.key, DEFAULTS[k.key]), "default": DEFAULTS[k.key]}
        for k in KNOBS
    ]
