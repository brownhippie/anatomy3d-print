"""The ledger: every render read with the plugin (mad_plugin.py; docs/PLUGIN.md).

One row per change. A change is one render: a Generate in the app, or one
build of the Lab. It is counted by the server's own render counter, never by
the clock. A row holds, for every quantity the Lab measures:
- what the photo gave (input side);
- what the model made (machine side);
- R, the reading the plugin takes;
- the version of the check that measured it.
Rows live on the volume (ledger.jsonl) and survive redeploys.

The plugin only reads. This module feeds it and acts on what it read in two
places, both one-way:
- The push guard. Each commit builds the held job (the same photo and the
  same settings every time) and the plugin reads that window. Against the
  held reference (the first window, never re-learned) it gives the drift.
  Against the commit before (the plugin's local level, with the held width)
  it gives the gate: a commit that moves a must-hold quantity out on more
  than GUARD_SHARE of the held job, or leaves a held build unsealed or below
  the printer's 0.8 mm wall, fails, and nothing it would pick is used.
- The production pick, the setting the app builds. It moves only to a
  setting that passed the gate and is at least as close to the photo on
  every must-hold quantity and closer on one (no averaging), or one you
  rated higher than the current pick. A setting you rated 2 stars or less is
  never picked.

A check whose code changes gets a new version, and that version starts its
own first spins. Nothing else ever re-learns the reference.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- where

def root() -> Path:
    env = os.environ.get("ANATOMY3D_LEDGER_DIR")
    if env:
        r = Path(env)
    elif Path("/data").is_dir() and os.access("/data", os.W_OK):
        r = Path("/data/ledger")
    else:
        r = Path(tempfile.gettempdir()) / "anatomy3d-ledger"
    r.mkdir(parents=True, exist_ok=True)
    return r


def commit() -> str:
    """The commit this process runs (Railway sets RAILWAY_GIT_COMMIT_SHA)."""
    sha = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")
    if sha:
        return sha[:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(Path(__file__).resolve().parent),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def machine() -> str:
    """What builds a render: the commit, plus the tuned values in effect
    (tuned.json, which the Study's autopilot can change by itself). Either
    changing is a new machine: a new window, and the held job built again."""
    c = commit()
    try:
        from . import knobs

        # (as load_tuned puts them in place: a value whose code default moved since is dropped)
        t = {k: e.get("value") for k, e in (knobs.read_tuned().get("values") or {}).items()
             if k in knobs.BY_KEY and e.get("code_default") == knobs.DEFAULTS[k]}
    except Exception:  # noqa: BLE001
        t = {}
    if not t:
        return c
    h = hashlib.sha1(json.dumps(sorted(t.items())).encode()).hexdigest()[:6]
    return f"{c}+t{h}"


def photo_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


_lock = threading.Lock()

# ---------------------------------------------------------------- the quantities

# key in lab.PARAMS -> (kind, how R is taken)
#   kind "ratio": the model's value over the photo's (or the human reference's); 1 = as the photo; off either way is out
#   kind "high": higher is better; kind "low": lower is better
#   R "model": the model's own value; "over": model / photo
QUANTITIES = {
    "outline": ("high", "model"),
    "aspect": ("ratio", "over"),
    "w_head": ("ratio", "over"),
    "w_shoulder": ("ratio", "over"),
    "w_chest": ("ratio", "over"),
    "w_waist": ("ratio", "over"),
    "face_shape": ("low", "model"),
    "eye_span": ("ratio", "over"),
    "nose_len": ("ratio", "over"),
    "mouth_w": ("ratio", "over"),
    "jaw_w": ("ratio", "over"),
    "face_hw": ("ratio", "over"),
    "face_id": ("high", "model"),
    "face_depth": ("ratio", "over"),
    "hair": ("high", "model"),
    "d_head": ("ratio", "over"),
    "d_chest": ("ratio", "over"),
    "d_waist": ("ratio", "over"),
    "sealed": ("high", "model"),
    "wall": ("high", "model"),
    "smooth_ok": ("low", "model"),
    # from the Lab's metrics, not its parameters
    "garment_edges": ("high", "model"),
    "clothes": ("high", "model"),
}
# Each reading in its own medium: the unit it is in, and the resolution it is
# measured to in that unit (how finely the measure itself can tell two values
# apart: its raster, its rounding, or the printer's layer). A change smaller
# than the resolution is no change: a null is a bound at the channel's
# resolution (docs/PLUGIN.md, section 5).
UNIT = {
    "outline": "% of the photo's cutout area covered by the model's front view (420-row raster)",
    "aspect": "model height:width over the photo's (ratio of two length ratios)",
    "w_head": "model width over the photo's, at the eyes' row (photo px over photo px)",
    "w_shoulder": "model width over the photo's, at the shoulder joints' row (photo px over photo px)",
    "w_chest": "model width over the photo's, a quarter of the way from shoulders to hips (photo px over photo px)",
    "w_waist": "model width over the photo's, 70% of the way from shoulders to hips (photo px over photo px)",
    "face_shape": "% of the eye distance: mean miss of the 468 face points found on the model's front render, after aligning",
    "eye_span": "model's eye-corner span ÷ face width, over the photo's",
    "nose_len": "model's bridge-to-base ÷ face height, over the photo's",
    "mouth_w": "model's mouth width ÷ face width, over the photo's",
    "jaw_w": "model's jaw-angle width ÷ face width, over the photo's",
    "face_hw": "model's forehead-chin ÷ cheek-cheek, over the photo's",
    "face_id": "cosine between face embeddings (SFace) of the model's render and the photo, −1 to 1",
    "face_depth": "model's face depth (mean mm the brow, eyes, cheeks, lips, chin sit behind the nose tip) over the photo's, both in mm on the figure",
    "hair": "share of the front surface area (mm²) over the photo's hair sculpted as hair (folds over 30°), 0 to 1",
    "d_head": "head length ÷ breadth, over the human reference's (ANSUR II men: 1.29)",
    "d_chest": "chest depth ÷ shoulder-joint distance, over the human reference's (ANSUR II men: 0.61)",
    "d_waist": "waist depth ÷ shoulder-joint distance, over the human reference's (ANSUR II men: 0.57)",
    "sealed": "1 when the mesh is closed (watertight), 0 when not",
    "wall": "mm, thinnest wall (2nd percentile) on the print as made",
    "smooth_ok": "% of the 40,000-face preview's fold edges that are outward and sharper than 60° (spikes)",
    "garment_edges": "% of the photo's garment-over-garment edges standing as a step on the model",
    "clothes": "% of the photo's sure over/under boundaries the model has the right way round",
}
RESOLUTION = {
    "outline": 0.25,       # one raster row of 420 is 0.24% of the height
    "aspect": 0.005,       # a pixel on a ~400 px span
    "w_head": 0.005, "w_shoulder": 0.005, "w_chest": 0.005, "w_waist": 0.005,  # a pixel on 200-600 px widths
    "face_shape": 0.1,     # rounded to 0.1 %
    "eye_span": 0.005, "nose_len": 0.005, "mouth_w": 0.005, "jaw_w": 0.005, "face_hw": 0.005,  # a pixel on the face's ~250 px
    "face_id": 0.001,      # rounded to 0.001
    "face_depth": 0.01,    # 0.01 mm on ~5 mm of depth at the raster's 0.37 mm cells: rounded to 2 places
    "hair": 0.005,         # rounded to 3 places
    "d_head": 0.01, "d_chest": 0.01, "d_waist": 0.01,  # rounded to 2 places
    "sealed": 0.5,         # 0 or 1
    "wall": 0.05,          # the printer's layer: 0.05 mm
    "smooth_ok": 0.01,     # rounded to 0.01 %
    "garment_edges": 0.1, "clothes": 0.1,  # rounded to 0.1 %
}
# the check's version: raised when the check's own code changes (a new version starts its own first spins)
VERSION = {"smooth_ok": 2, "face_depth": 2, "hair": 2}  # smooth_ok 2: hair and ears left out (99762eb); face_depth 2: in mm both sides; hair 2: by area
WALL_MIN_MM = 0.8  # the printer's: thinnest wall a print holds
GUARD_SHARE = 10.0  # net % of a family's held settings moved further from the photo (worse minus better) against the commit before: the push fails
HELD_N = int(os.environ.get("ANATOMY3D_HELD_N", "160"))  # the held job: the first this many settings of the held photo's run order (the B family whole is among them)


def readings(metrics: dict) -> dict:
    """{quantity: R} from lab.score's metrics."""
    out = {}
    p = metrics.get("params") or {}
    for k, (kind, how) in QUANTITIES.items():
        if k == "garment_edges":
            v = metrics.get("garment_edges")
        elif k == "clothes":
            v = metrics.get("clothes_pct")
        elif k in p and p[k][0] is not None:
            m, t = p[k][0], p[k][1]
            if isinstance(m, bool):
                m = 1.0 if m else 0.0
            if how == "over":
                v = float(m) / float(t) if t not in (None, 0, False) else None
            else:
                v = float(m)
        else:
            v = None
        if v is not None:
            out[k] = round(float(v), 4)
    return out


def photo_side(metrics: dict) -> dict:
    """What the photo gave, per quantity (input side)."""
    p = metrics.get("params") or {}
    return {k: (p[k][1] if not isinstance(p[k][1], bool) else int(p[k][1])) for k in p if k in QUANTITIES}


def make_row(source: str, counter: int, code: str, photo: str, metrics: dict, **extra) -> dict:
    rf = metrics.get("checks_failed")
    row = {
        "source": source, "counter": int(counter), "at": time.time(), "commit": machine(), "code": code, "photo": photo,
        "R": readings(metrics), "photo_side": photo_side(metrics),
        "v": {k: VERSION.get(k, 1) for k in QUANTITIES},
        # the noise channel: where the excess shows at the machine's end
        "noise": {"spikes": metrics.get("spike_pct"), "checks_failed": len(rf) if isinstance(rf, list) else rf,
                  "face_found": metrics.get("face_found"), "seconds": metrics.get("seconds")},
    }
    row.update(extra)
    return row


def append(row: dict) -> None:
    with _lock:
        with open(root() / "ledger.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


_cache = {"key": None, "rows": []}


def rows(source: Optional[str] = None) -> "list[dict]":
    """Every row (or one source's), read once per change of the file."""
    p = root() / "ledger.jsonl"
    try:
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
    except OSError:
        return []
    if _cache["key"] != key:
        out = []
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        _cache.update(key=key, rows=out)
    return [r for r in _cache["rows"] if source is None or r.get("source") == source]


def next_counter() -> int:
    """The render counter: one more each render, kept on the volume (the
    system's own rhythm, shared by the server and its build processes)."""
    import fcntl

    p = root() / "counter.txt"
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            n = int(f.read().strip() or 0)
        except ValueError:
            n = 0
        if n == 0:
            rs = rows()
            n = max((r.get("counter", 0) for r in rs), default=0)
        n += 1
        f.seek(0)
        f.truncate()
        f.write(str(n))
        fcntl.flock(f, fcntl.LOCK_UN)
    return n


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _write_json(p: Path, data) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


# ---------------------------------------------------------------- the held job, read with the plugin

def held() -> dict:
    """{"photo": sha, "codes": [...]}: the held photo and its held settings."""
    return _read_json(root() / "held.json", {})


def set_held(photo: str, codes: "list[str]") -> None:
    _write_json(root() / "held.json", {"photo": photo, "codes": list(codes)})


def held_windows(q: str, family: str) -> "list[tuple[str, dict]]":
    """[(label, {code: R})] for quantity q at its current version: the held
    job's Lab builds of one builder family on the held photo, one window per
    commit (or per run, for builds from before commits were recorded), in
    the order they first came. Windows that built under 60% of the most
    any window built are left out; of the rest only the settings every
    window has are kept, so every window is the same job."""
    h = held()
    if not h:
        return []
    codes = {c for c in (h.get("codes") or []) if c.startswith(family + "-")}
    ver = VERSION.get(q, 1)
    by, first = {}, {}
    for r in rows("lab"):
        if r.get("photo") != h["photo"] or r.get("code") not in codes or (r.get("v") or {}).get(q) != ver:
            continue
        v = (r.get("R") or {}).get(q)
        if v is None:
            continue
        c = r.get("commit") or "unknown"
        by.setdefault(c, {})[r["code"]] = v
        first.setdefault(c, r.get("counter", 0))
    if not by:
        return []
    most = max(len(w) for w in by.values())
    wins = [(c, w) for c, w in by.items() if len(w) >= 0.6 * most]
    common = set.intersection(*[set(w) for _, w in wins])
    wins = [(c, {k: w[k] for k in common}) for c, w in wins]
    return sorted(wins, key=lambda kv: first[kv[0]])


def read_held(q: str, family: str, min_codes: int = 6):
    """The plugin's reading of quantity q over one family's held job:
    ({"windows", "settings", "first_spins"}, reference) or (None, why). One
    window per commit; first spins = the first commit's window (warm-up
    none: a build is counted, not heated); local level = the commit before
    (with the first spins' width held)."""
    import numpy as np
    import pandas as pd

    from . import mad_plugin as mp

    wins = held_windows(q, family)
    if not wins:
        return None, "no held builds yet"
    common = set(wins[0][1])
    if len(common) < min_codes:
        return None, "the held job isn't built yet"
    codes = sorted(common)
    n = len(codes)
    t = pd.DataFrame([{"label": c, q: w[k]} for c, w in wins for k in codes])
    t["run"] = (t["label"] != t["label"].shift()).cumsum()
    # the steady flag needs the spread inside a render; none is logged yet, so
    # every render counts as steady (the change-to-change step is not an
    # inside spread: consecutive rows here are different settings)
    spread = "_spread"
    t[spread] = 0.0
    old = dict(mp.P)
    mp.P.update(N_FIRST=n, WARMUP=0, WINDOW=n, W_LOCAL=n, MIN_LOCAL=min(10, n))
    try:
        state, why = mp.take_first_spins(t, [q], spread)
        if state is None:
            return None, why
        tr = mp.read_changes(t, [q], spread, state)
        out = []
        for lab_ in tr["label"].unique():
            out += mp.windows(tr[tr["label"] == lab_], [q], lab_)
    finally:
        mp.P.clear()
        mp.P.update(old)
    ref = state["refs"][q]
    for w in out:
        for k, v in list(w.items()):
            if isinstance(v, float) and not np.isfinite(v):
                w[k] = None
    return {"windows": out, "settings": n, "first_spins": wins[0][0]}, {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in ref.items()}


def _bad_share(w: dict, q: str, local: bool, level: Optional[float]) -> float:
    """The share of the window out on the side that is further from the
    photo: below for higher-is-better, above for lower-is-better, and for a
    ratio the side away from 1 (out toward 1 is closer, not worse)."""
    kind = QUANTITIES[q][0]
    s = "_local" if local else ""
    past, short = w.get(f"{q}_past{s}_pct") or 0.0, w.get(f"{q}_short{s}_pct") or 0.0
    if kind == "high":
        return short
    if kind == "low":
        return past
    if level is None:
        return max(past, short)
    return past if level >= 1.0 else short


GUARD_MIN_SETTINGS = 6  # a quantity read on fewer held settings than this doesn't decide
KICK_TOL = 0.005  # (only for a quantity with no resolution of its own: under 0.5% of itself is no move)


def held_by_commit() -> "list[tuple[str, dict]]":
    """[(label, {code: row})]: the held job's Lab builds on the held photo,
    one window per commit (or run), in the order they first came."""
    h = held()
    if not h:
        return []
    codes = set(h.get("codes") or [])
    by, first = {}, {}
    for r in rows("lab"):
        if r.get("photo") == h["photo"] and r.get("code") in codes and r.get("R"):
            c = r.get("commit") or "unknown"
            by.setdefault(c, {})[r["code"]] = r
            first.setdefault(c, r.get("counter", 0))
    return sorted(by.items(), key=lambda kv: first[kv[0]])


def _worse(q: str, now: float, was: float) -> bool:
    """Further from the photo by more than the measure's own resolution."""
    d = _closeness(q, now) - _closeness(q, was)
    return d > RESOLUTION.get(q, KICK_TOL * max(abs(was), 1e-3))


def guard(for_commit: str) -> dict:
    """The push guard for one commit. Each held setting's kick (this
    commit's build of it against the commit before's) per quantity: the
    commit fails when, in a builder family, more than GUARD_SHARE % of the
    held settings moved further from the photo on any quantity, or a held
    build is newly unsealed or below the printer's wall. The plugin's
    reading against the first spins (drift, and the share out past the
    first spins' floor or ceiling) is kept beside it for each quantity."""
    import warnings

    wins = held_by_commit()
    out = {"commit": for_commit, "at": time.time(), "quantities": {}, "passed": True, "why": [], "against": None}
    labels = [c for c, _ in wins]
    if for_commit not in labels:
        out["why"].append("this commit hasn't built the held job")
        out["passed"] = False
        return out
    i = labels.index(for_commit)
    cur = wins[i][1]
    fams = sorted({c.split("-")[0] for c in cur})
    against = {}
    for fam in fams:
        mine = {c for c in cur if c.startswith(fam + "-")}
        for c, w in reversed(wins[:i]):
            theirs = {k for k in w if k.startswith(fam + "-")}
            if mine and len(theirs & mine) >= 0.6 * len(mine):
                against[fam] = (c, w)
                break
    out["against"] = {f: a[0] for f, a in against.items()}
    for fam in fams:
        prev = against.get(fam)
        for q in QUANTITIES:
            entry = {}
            if prev:
                n = worse = better = 0
                for code, r in cur.items():
                    if not code.startswith(fam + "-") or code not in prev[1]:
                        continue
                    a, b = r["R"].get(q), prev[1][code]["R"].get(q)
                    if a is None or b is None or (r.get("v") or {}).get(q) != (prev[1][code].get("v") or {}).get(q):
                        continue
                    n += 1
                    worse += _worse(q, a, b)
                    better += _worse(q, b, a)
                if n >= GUARD_MIN_SETTINGS:
                    net = 100.0 * (worse - better) / n
                    entry.update(settings=n, worse=worse, better=better, net_worse_pct=round(net, 1))
                    if net > GUARD_SHARE:
                        out["passed"] = False
                        out["why"].append(f"{fam} {q}: {worse} of {n} held settings further from the photo than on {prev[0]}, {better} closer")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got, ref = read_held(q, fam)
            if got is not None:
                ws = [w for w in got["windows"] if w["label"] == for_commit]
                if ws:
                    entry.update(drift_pct=ws[-1].get(f"{q}_drift_pct"), past_pct=ws[-1].get(f"{q}_past_pct"),
                                 short_pct=ws[-1].get(f"{q}_short_pct"), level=ref.get("L"), floor=ref.get("floor"),
                                 ceiling=ref.get("ceil"), first_spins=got["first_spins"])
            if entry:
                out["quantities"][f"{fam}:{q}"] = entry
    # the printer's limits
    for code, r in cur.items():
        prev = against.get(code.split("-")[0])
        if prev:
            was = (prev[1].get(code) or {}).get("R") or {}
            R = r["R"]
            if R.get("sealed") == 0.0 and was.get("sealed") == 1.0:
                out["passed"] = False
                out["why"].append(f"{code}: no longer sealed")
            if R.get("wall") is not None and R["wall"] < WALL_MIN_MM <= (was.get("wall") or 0):
                out["passed"] = False
                out["why"].append(f"{code}: thinnest wall {R['wall']} mm, below {WALL_MIN_MM} mm")
    _write_json(root() / f"guard_{for_commit.replace(':', '_')}.json", out)
    return out


# ---------------------------------------------------------------- your ratings

def rate(source: str, ref: str, code: str, stars: int, note: str = "") -> None:
    append({"source": "rating", "of": source, "ref": ref, "code": code, "stars": int(stars), "note": note[:500],
            "at": time.time(), "commit": machine(), "counter": next_counter()})


PRODUCTION = "production"  # the app's own build, whatever photo: its renders' ratings are rated under this name


def production_like(code: str) -> bool:
    """True for the app's own production build: an "app:..." render of its
    own pipeline with plain tweaks, or a Lab code with option 0 everywhere
    (but the outline's "below the photo", which follows the photo)."""
    if code == PRODUCTION or (code.startswith("app:") and not code.endswith(":tweaked")):
        return True
    from . import lab

    try:
        fam, idx = lab.decode(code)
    except Exception:  # noqa: BLE001
        return False
    legs = [d[0] for d in lab.FAMILIES[fam]["dims"]].index("legs") if fam == "O" else -1
    return fam in ("O", "C") and all(i == 0 for k, i in enumerate(idx) if k != legs)


def _family_key(code: str) -> str:
    """Ratings of one setting on any photo count together: the outline's
    "below the photo" follows the photo, so it's left out of the key."""
    if production_like(code):
        return PRODUCTION
    if code.startswith("O-"):
        from . import lab

        try:
            _, idx = lab.decode(code)
            legs = [d[0] for d in lab.FAMILIES["O"]["dims"]].index("legs")
            return "O-" + "".join("x" if k == legs else str(i) for k, i in enumerate(idx))
        except Exception:  # noqa: BLE001
            return code
    return code


def ratings_by_code() -> "dict[str, float]":
    """Mean stars per setting (keyed by _family_key), from every rating:
    app renders and Lab builds."""
    acc = {}
    for r in rows("rating"):
        if r.get("code"):
            acc.setdefault(_family_key(r["code"]), []).append(r["stars"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def challenger(current: str) -> Optional[str]:
    """The setting to build beside the current one when you ask to compare:
    of the held photo's latest builds that pass the printer's limits and
    that you haven't rated yet (or rated over 2), the one closer to the
    photo than the current one on the most quantities, net of those where
    it's further."""
    latest = {}
    for c, w in reversed(held_by_commit() or []):
        if current in w:
            latest = {k: r["R"] for k, r in w.items()}
            break
    if current not in latest:
        return None
    stars = ratings_by_code()
    cur = latest[current]
    best, best_n = None, None
    for code, R in sorted(latest.items()):
        k = _family_key(code)
        if production_like(code) or not _gates(R) or k in stars:
            continue
        shared = [q for q in QUANTITIES if q in R and q in cur]
        n = sum(_worse(q, cur[q], R[q]) for q in shared) - sum(_worse(q, R[q], cur[q]) for q in shared)
        if best_n is None or n > best_n:
            best, best_n = code, n
    return best


# ---------------------------------------------------------------- the production pick

def _closeness(q: str, v: float) -> float:
    """How far from the photo (smaller is closer), per the quantity's kind."""
    kind = QUANTITIES[q][0]
    return abs(v - 1.0) if kind == "ratio" else (-v if kind == "high" else v)


def _gates(R: dict) -> bool:
    return R.get("sealed", 1.0) == 1.0 and (R.get("wall") is None or R["wall"] >= WALL_MIN_MM)


def dominates(a: dict, b: dict) -> bool:
    """a is at least as close to the photo as b on every quantity both have, and closer on one."""
    shared = [q for q in QUANTITIES if q in a and q in b]
    if not shared:
        return False
    return not any(_worse(q, a[q], b[q]) for q in shared) and any(_worse(q, b[q], a[q]) for q in shared)


def pick() -> dict:
    """The setting the app builds: {"code": ..., "commit": ..., "why": ...};
    {} means the app's own production build."""
    return _read_json(root() / "pick.json", {})


def choose(for_commit: str, current: str) -> dict:
    """The production pick after a commit's held job is read. Changes only when
    the commit passed the guard."""
    g = _read_json(root() / f"guard_{for_commit.replace(':', '_')}.json", None)
    now = pick()
    cur = now.get("code") or current
    if not g or not g.get("passed"):
        return now
    h = held()
    latest = {}
    for r in rows("lab"):
        if r.get("commit") == for_commit and r.get("photo") == h.get("photo"):
            latest[r["code"]] = r.get("R") or {}
    if cur not in latest:
        return now
    stars = ratings_by_code()
    ok = {c: R for c, R in latest.items() if _gates(R) and stars.get(_family_key(c), 3) > 2}
    best, why = cur, "no setting is at least as close on every quantity and closer on one"
    # your ratings first: a setting you rated higher than the current one
    cur_stars = stars.get(_family_key(cur), 0)
    rated = sorted((c for c in ok if _family_key(c) in stars and stars[_family_key(c)] > cur_stars and not production_like(c)),
                   key=lambda c: -stars[_family_key(c)])
    if rated:
        best, why = rated[0], f"you rated it {stars[_family_key(rated[0])]:.1f} stars, higher than the current {cur_stars:.1f}"
    else:
        better = [c for c, R in ok.items() if c != cur and dominates(R, latest[cur])]
        if better:
            # of those, the one that's itself dominated by none of the others
            top = [c for c in better if not any(dominates(ok[o], ok[c]) for o in better if o != c)]
            best = sorted(top)[0]
            why = f"at least as close to the photo as {cur} on every quantity, closer on one"
    out = {"code": best, "commit": for_commit, "why": why, "at": time.time(), "was": cur}
    _write_json(root() / "pick.json", out)
    with open(root() / "picks.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


# ---------------------------------------------------------------- the Lab's earlier builds

def backfill(runs) -> int:
    """Rows for builds made before the ledger existed: `runs` is
    [(meta, [result rows], photo sha)] oldest first. Their commit wasn't
    recorded, so each run is its own label ("run:<id>"). The checks
    changed in place before versions existed: a run that has the hair check
    already counts spikes without the hair (smooth_ok version 2)."""
    n = 0
    for meta, results, sha in runs:
        label = meta.get("commit") or f"run:{meta.get('run_id')}"
        for r in results:
            if r.get("status") != "ok":
                continue
            p = r.get("params") or {}
            v = {k: VERSION.get(k, 1) for k in QUANTITIES}
            if "hair" not in p:
                v["smooth_ok"] = 1
            row = make_row("lab", next_counter(), r["code"], sha, r, run_id=meta.get("run_id"), backfilled=True)
            row["commit"], row["v"], row["at"] = label, v, r.get("at") or meta.get("created")
            append(row)
            n += 1
    return n


def held_production() -> str:
    """The app's own build as a code on the held photo (its first outline setting)."""
    from . import lab

    return next((c for c in (held().get("codes") or []) if c.startswith("O-")), lab.CODE_PRODUCTION)


def choose_after_rating() -> dict:
    """A new rating can move the pick at once (it's you, not the machine,
    that changed): chosen again on the newest window that passed its guard."""
    guards = sorted(root().glob("guard_*.json"), key=lambda q: q.stat().st_mtime)
    for gp in reversed(guards):
        g = _read_json(gp, {})
        if g.get("passed"):
            return choose(g["commit"], held_production())
    return pick()
