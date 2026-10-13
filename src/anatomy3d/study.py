"""Studies: the server teaching the builder on its own, with no one in the
loop until there is something to approve.

A study holds a few photos and some areas of knobs.py (hair, ears, neck...).
It builds every photo with the production combination, scores each model
with the critics below, then moves one knob at a time (up or down a step,
in the knob's own range), keeps a move when the photos' mean score rises,
and halves the step when a whole round finds nothing (a pattern search).
Every move, kept or not, is logged with its score. Nothing it finds is used
until it is approved (knobs.save_tuned).

The critics, per photo, all automatic:
- the Lab's own score (lab.score): front outline vs the photo's cutout,
  spikes, sealed, thinnest wall, face points found on the model vs the
  photo's
- the Lab's "held" parameters: widths, face proportions, the face's
  embedding match, head/chest/waist depth vs ANSUR II
- bumps: how far the surface stands off a smoothed copy of itself on the
  sides, back and top, where the photo can't have put detail -- the fins,
  ridges, bars, cracks and stray lumps the builder has made before, as
  numbers
"""
import json
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from . import knobs, lab

CODE = "O-" + "0" * len(lab.FAMILIES["O"]["dims"])  # the production combination
START_STEP = 0.25  # of a knob's range
MIN_STEP = 1 / 32
MIN_GAIN = 0.15  # a move is kept when the mean score rises by more than this
GIVE_UP = 4.0  # a move is dropped early once its photos so far trail by this much each
THUMB = (720, 480)

# objective = score + HELD_W x held mean - BUMP_W x bumps
HELD_W = 0.3
BUMP_W = 3.0
BUMP_MM = 0.4  # bumps: % of the surface standing off its smoothed copy by more than this
OBJECTIVE = (f"study score = Lab score + {HELD_W} × held mean − {BUMP_W} × bumps "
             f"(% of the sides, back and top standing more than {BUMP_MM} mm off a smoothed copy of themselves)")


# ---------------------------------------------------------------- critics


def bumps(preview: trimesh.Trimesh, floor: float) -> dict:
    """Surface standing off a Taubin-smoothed copy of itself (mm, along the
    normal), where the photo can't have put detail: the sides, back and top
    (normals turned more than ~70 degrees from the camera), above the base."""
    smooth = preview.copy()
    trimesh.smoothing.filter_taubin(smooth, lamb=0.5, nu=0.53, iterations=12)
    v, n = preview.vertices, preview.vertex_normals
    off = np.abs(np.einsum("ij,ij->i", v - smooth.vertices, n))
    keep = (v[:, 1] > floor + 1.5) & (n[:, 2] < 0.35)
    if not keep.any():
        return {"bumps": 0.0, "bump_p99_mm": 0.0}
    # per vertex, weighted by the area round it
    area = np.zeros(len(v))
    np.add.at(area, preview.faces.ravel(), np.repeat(preview.area_faces / 3.0, 3))
    a, o = area[keep], off[keep]
    return {"bumps": round(100.0 * float(a[o > BUMP_MM].sum() / a.sum()), 2),
            "bump_p99_mm": round(float(np.percentile(o, 99)), 2)}


def objective(m: dict) -> float:
    return float(m.get("score", -100.0) + HELD_W * m.get("held_mean", 0.0) - BUMP_W * m.get("bumps", 0.0))


def evaluate(inputs: "lab.LabInputs", values: dict):
    """Build and score one photo with `values` applied. Hold the build lock."""
    from .body_reference import fit_body_reference

    with knobs.applied(values):
        if inputs.pose and inputs.animal_frac <= 0.15:
            fitted = fit_body_reference(inputs.pose, inputs.face_mesh, inputs.photo_mask, gender=inputs.body, parts=inputs.parts)
            inputs.reference = fitted if fitted is not None and fitted["matched"] else None
        body = lab.build(CODE, inputs)
        m, preview, thumb = lab.score(CODE, body, inputs, thumb_size=THUMB)
    lab._sface.pop(id(inputs), None)  # keyed by object: never let a later photo reuse it
    m.update(bumps(preview, _floor(preview)))
    m.pop("params", None)
    m["objective"] = round(objective(m), 2)
    return m, thumb


def _floor(mesh: trimesh.Trimesh) -> float:
    from .mesh_export import _BASE_DISK_MAX_THICKNESS_MM, _BASE_DISK_MIN_THICKNESS_MM, _BASE_DISK_THICKNESS_FRAC

    disk = float(np.clip(_BASE_DISK_THICKNESS_FRAC * lab.HEIGHT_MM, _BASE_DISK_MIN_THICKNESS_MM, _BASE_DISK_MAX_THICKNESS_MM))
    return float(mesh.vertices[:, 1].min()) + disk


# ---------------------------------------------------------------- photos


def prepare(photo_path: str, body: str, cache: Path) -> "lab.LabInputs":
    """The photo's inputs, computed once and kept in `cache` (a pickle of our
    own arrays; recomputed if missing, e.g. after a redeploy)."""
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:  # noqa: BLE001 - a broken cache is just recomputed
            pass
    inputs = lab.prepare_inputs(photo_path, body=body, capsule_builder=False, matting=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache)
    return inputs


# ---------------------------------------------------------------- search


def new_state(sid: str, photos: list, areas: list, max_evals: int, seed: int = 0) -> dict:
    keys = [k.key for k in knobs.KNOBS if k.area in areas]
    base = knobs.current()
    return {
        "sid": sid, "created": time.time(), "status": "running", "areas": areas, "photos": photos,
        "keys": keys, "base": {k: base[k] for k in keys}, "x": {k: base[k] for k in keys},
        "step": START_STEP, "round": 0, "evals": 0, "max_evals": max_evals, "seed": seed,
        "queue": [], "improved": False, "best_obj": None, "base_obj": None, "best_per_photo": {}, "base_per_photo": {},
        "cand": None, "kept": [],
    }


def _move(k: "knobs.Knob", v: float, step: float, sign: int):
    u = (float(v) - k.lo) / (k.hi - k.lo)
    return knobs.clip(k, k.lo + min(max(u + sign * step, 0.0), 1.0) * (k.hi - k.lo))


def next_candidate(st: dict) -> Optional[dict]:
    """The next move to try, or None when the study is finished."""
    if st["best_obj"] is None:
        return {"values": dict(st["x"]), "change": None, "results": {}}
    while True:
        if st["evals"] >= st["max_evals"]:
            return None
        if not st["queue"]:
            if not st["improved"] and st["round"] > 0:
                st["step"] /= 2.0
                if st["step"] < MIN_STEP:
                    return None
            st["round"] += 1
            st["improved"] = False
            rng = random.Random(st["seed"] * 1000 + st["round"])
            moves = [[k, s] for k in st["keys"] for s in (1, -1)]
            rng.shuffle(moves)
            st["queue"] = moves
        key, sign = st["queue"].pop(0)
        k = knobs.BY_KEY[key]
        to = _move(k, st["x"][key], st["step"], sign)
        if to == st["x"][key]:
            continue  # at the end of its range
        values = dict(st["x"])
        values[key] = to
        return {"values": values, "change": [key, st["x"][key], to, sign], "results": {}}


def finish_candidate(st: dict) -> dict:
    """Decides the finished (or dropped) candidate; returns its log line."""
    c = st["cand"]
    per = {pid: r.get("objective", -100.0) for pid, r in c["results"].items()}
    complete = len(per) == len(st["photos"])
    mean = float(np.mean(list(per.values()))) if per else -100.0
    line = {"i": st["evals"], "at": time.time(), "change": c["change"], "step": st["step"], "per_photo": per,
            "objective": round(mean, 2), "complete": complete,
            "metrics": {pid: {k: r.get(k) for k in ("score", "held_mean", "bumps", "shape_iou", "face_err_pct", "spike_pct", "error")}
                        for pid, r in c["results"].items()}}
    if st["best_obj"] is None:
        st["best_obj"] = st["base_obj"] = round(mean, 2)
        st["best_per_photo"] = dict(per)
        st["base_per_photo"] = dict(per)
        line["kept"] = "baseline"
    elif complete and mean > st["best_obj"] + MIN_GAIN:
        key, _, to, sign = c["change"]
        line["kept"] = True
        line["gain"] = round(mean - st["best_obj"], 2)
        st["x"][key] = to
        st["best_obj"] = round(mean, 2)
        st["best_per_photo"] = dict(per)
        st["improved"] = True
        st["queue"].insert(0, [key, sign])  # keep going the same way
        st["kept"].append({"i": st["evals"], "key": key, "from": c["change"][1], "to": to, "gain": line["gain"],
                           "objective": st["best_obj"]})
    else:
        line["kept"] = False
    st["evals"] += 1
    st["cand"] = None
    return line


def trails(st: dict) -> bool:
    """The candidate is already clearly worse on the photos built so far."""
    c = st["cand"]
    if c["change"] is None or not c["results"]:
        return False
    lost = [st["best_per_photo"].get(pid, 0.0) - r.get("objective", -100.0) for pid, r in c["results"].items()]
    return float(np.mean(lost)) > GIVE_UP


def summary_values(st: dict) -> list:
    out = []
    for key in st["keys"]:
        k = knobs.BY_KEY[key]
        out.append({"key": key, "name": k.name, "area": k.area, "what": k.what, "lo": k.lo, "hi": k.hi,
                    "base": st["base"][key], "best": st["x"][key], "code_default": knobs.DEFAULTS[key]})
    return out


def dumps(st: dict) -> str:
    return json.dumps(st, default=float)
