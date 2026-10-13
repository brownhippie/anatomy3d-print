"""Studies: the server teaching the builder by itself (see anatomy3d/study.py).

The Lab's worker (lab_api.py) takes turns: one Lab combination, one study
build, so both move while neither holds up a person's own Generate (each
build takes the same lock). Under STUDY_ROOT/<sid>/:
  state.json        the search: photos, knobs, current best, the move in progress
  log.jsonl         every move tried and its score
  p/<pid>.<ext>     the photos
  t/base|best|cand/<pid>.png   three-view renders (in the container's temp
                    space instead when the volume is full)
The photos' prepared inputs are cached outside the volume (recomputed after a
redeploy). Nothing a study finds is used until it is applied
(POST /api/study/{sid}/apply -> knobs.save_tuned).
"""
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from anatomy3d import knobs, study


def _root() -> Path:
    env = os.environ.get("ANATOMY3D_STUDY_DIR")
    if env:
        root = Path(env)
    elif Path("/data").is_dir() and os.access("/data", os.W_OK):
        root = Path("/data/study")
    else:
        root = Path(tempfile.gettempdir()) / "anatomy3d-study"
    root.mkdir(parents=True, exist_ok=True)
    return root


STUDY_ROOT = _root()
CACHE_ROOT = Path(tempfile.gettempdir()) / "anatomy3d-study-cache"
SID_RE = re.compile(r"^[0-9a-f]{10}$")
PID_RE = re.compile(r"^p[0-9]{1,2}$")
MIN_FREE = 30e6  # bytes left on the volume below which renders aren't written

router = APIRouter(prefix="/api/study")
_lock = threading.Lock()
_tuned_status = {"applied": {}, "stale": []}


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=float))
    tmp.replace(path)


def _dir(sid: str) -> Path:
    if not SID_RE.match(sid) or not (STUDY_ROOT / sid / "state.json").exists():
        raise HTTPException(404, "No such study.")
    return STUDY_ROOT / sid


def _free() -> float:
    try:
        return shutil.disk_usage(STUDY_ROOT).free
    except OSError:
        return 0.0


def _renders(sid: str) -> Path:
    """Where a study's renders go: on the volume while it has room, else in
    the container's own temp space (lost on a redeploy)."""
    vol = STUDY_ROOT / sid / "t"
    if vol.is_dir() or _free() > MIN_FREE:
        vol.mkdir(parents=True, exist_ok=True)
        return vol
    return CACHE_ROOT / sid / "t"


def _repair(sid: str, st: dict, generation_lock: threading.Lock) -> bool:
    """Renders lost with the container's temp space (a redeploy while the
    volume was full) are built again, one per step. True if it built one."""
    t = _renders(sid)
    for which, values in (("base", {**st["x"], **st["base"]}), ("best", st["x"])):
        for p in st["photos"]:
            if (t / which / f"{p['id']}.png").exists():
                continue
            d = STUDY_ROOT / sid
            try:
                with generation_lock:
                    inputs = study.prepare(str(d / "p" / p["file"]), p["body"], CACHE_ROOT / sid / f"{p['id']}.pkl")
                    _, thumb = study.evaluate(inputs, values)
                (t / which).mkdir(parents=True, exist_ok=True)
                thumb.save(t / which / f"{p['id']}.png", optimize=True)
            except Exception as exc:  # noqa: BLE001 - leave it missing; the study goes on
                print(f"Study {sid}: couldn't render {which} {p['id']}: {exc}")
                return False
            return True
    return False


def clear_idle_caches() -> None:
    """Renders kept in temp space move to the volume once it has room;
    inputs cached for studies that aren't running go."""
    if not CACHE_ROOT.is_dir():
        return
    for d in CACHE_ROOT.iterdir():
        st = _read(STUDY_ROOT / d.name / "state.json", None)
        if (d / "t").is_dir() and st is not None and _free() > MIN_FREE * 4:
            shutil.copytree(d / "t", STUDY_ROOT / d.name / "t", dirs_exist_ok=True)
            shutil.rmtree(d / "t", ignore_errors=True)
        if st is None or (st.get("status") != "running" and not (d / "t").is_dir()):
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- autopilot
# The server teaches itself: whenever no study runs it judges the last
# finished one, applies it if it passes (below), and starts the next on the
# same photos with the next area of numbers, from the values now in use --
# so every study has to beat what the one before left, and the bar rises
# by itself. A finished study is applied only when its best beats its start
# by AUTO_GAIN on the mean and no photo got worse than AUTO_WORST.

AUTO_EVALS = 120
AUTO_GAIN = 0.5
AUTO_WORST = 1.0
AUTO_AREAS = ["hair", "hair pattern", "headwear", "neck", "ears", "clothes", "face"]


def _autopilot_path() -> Path:
    return STUDY_ROOT / "autopilot.json"


def autopilot_state() -> dict:
    return {"on": True, "next": 0, **_read(_autopilot_path(), {})}


def _judge(st: dict) -> "tuple[bool, str]":
    if not st.get("kept") or st.get("best_obj") is None or st.get("base_obj") is None:
        return False, "nothing kept"
    gain = st["best_obj"] - st["base_obj"]
    if gain < AUTO_GAIN:
        return False, f"gain {gain:.2f} under {AUTO_GAIN}"
    worst = min(st["best_per_photo"].get(k, -1e9) - v for k, v in st["base_per_photo"].items())
    if worst < -AUTO_WORST:
        return False, f"a photo got worse by {-worst:.2f}"
    return True, f"gain {gain:.2f}, no photo worse than {-min(worst, 0):.2f}"


def autopilot_step(generation_lock: threading.Lock) -> bool:
    """Called by the worker when no study runs. True if it did something."""
    ap = autopilot_state()
    if not ap["on"]:
        return False
    studies = []
    for d in STUDY_ROOT.iterdir():
        st = _read(d / "state.json", None) if d.is_dir() else None
        if st:
            studies.append((st["created"], d, st))
    if not studies:
        return False  # needs a first study (its photos) from someone
    studies.sort(key=lambda t: t[0])
    _, d, st = studies[-1]
    if st["status"] == "paused":
        return False  # paused by hand: leave it
    if st["status"] == "done" and not st.get("judged"):
        ok, why = _judge(st)
        if ok:
            values = {**knobs.current(), **st["x"]}
            with generation_lock:
                _tuned_status.update(knobs.save_tuned(values, source=f"study {st['sid']} (applied by itself)"))
            st["applied"] = time.time()
        st["judged"] = {"applied": ok, "why": why, "at": time.time()}
        with _lock:
            _write(d / "state.json", st)
        return True
    if st["status"] != "done":
        return False
    # the next study: same photos, next area, from the values in use now
    area = AUTO_AREAS[ap["next"] % len(AUTO_AREAS)]
    sid = uuid.uuid4().hex[:10]
    nd = STUDY_ROOT / sid
    (nd / "p").mkdir(parents=True)
    for p in st["photos"]:
        shutil.copy2(d / "p" / p["file"], nd / "p" / p["file"])
    new = study.new_state(sid, st["photos"], [area], AUTO_EVALS, seed=int(time.time()) % 1000)
    new["started"] = time.time()
    new["auto"] = True
    with _lock:
        _write(nd / "state.json", new)
        _write(_autopilot_path(), {**ap, "next": ap["next"] + 1})
    return True


@router.post("/autopilot")
def set_autopilot(on: bool = True):
    ap = autopilot_state()
    ap["on"] = bool(on)
    with _lock:
        _write(_autopilot_path(), ap)
    return ap


def load_tuned_at_startup() -> None:
    _tuned_status.update(knobs.load_tuned())


# ---------------------------------------------------------------- worker


def active() -> Optional[str]:
    best, best_t = None, -1.0
    for d in STUDY_ROOT.iterdir():
        st = _read(d / "state.json", None) if d.is_dir() else None
        if st and st.get("status") == "running" and st.get("started", st["created"]) > best_t:
            best, best_t = d.name, st.get("started", st["created"])
    return best


def work_one(sid: str, generation_lock: threading.Lock) -> None:
    """One step: prepare a photo, or build one photo for the move in progress."""
    d = STUDY_ROOT / sid
    with _lock:
        st = _read(d / "state.json", None)
    if not st or st.get("status") != "running":
        return
    if st["cand"] is None and study.rebase_if_code_changed(st):
        with _lock:
            _write(d / "state.json", st)
        return
    if st["cand"] is None and st["best_obj"] is not None and not st.get("rebase") and _repair(sid, st, generation_lock):
        return
    if st["cand"] is None:
        st["cand"] = study.next_candidate(st)
        if st["cand"] is None:
            st["status"] = "done"
            st["finished"] = time.time()
            with _lock:
                _write(d / "state.json", st)
            return
    todo = [p for p in st["photos"] if p["id"] not in st["cand"]["results"]]
    p = todo[0]
    try:
        with generation_lock:
            inputs = study.prepare(str(d / "p" / p["file"]), p["body"], CACHE_ROOT / sid / f"{p['id']}.pkl")
            m, thumb = study.evaluate(inputs, st["cand"]["values"])
        del inputs
        which = "base" if st["best_obj"] is None else "cand"
        (_renders(sid) / which).mkdir(parents=True, exist_ok=True)
        thumb.save(_renders(sid) / which / f"{p['id']}.png", optimize=True)
    except Exception as exc:  # noqa: BLE001 - a failed build scores as one
        m = {"objective": -100.0, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    st["cand"]["results"][p["id"]] = m
    line = None
    if len(st["cand"]["results"]) == len(st["photos"]) or study.trails(st):
        first = st["best_obj"] is None
        line = study.finish_candidate(st)
        t = _renders(sid)
        role = str(line.get("kept", ""))
        if first and (t / "base").is_dir():
            shutil.copytree(t / "base", t / "best", dirs_exist_ok=True)
        elif (line.get("kept") is True or role.startswith("rebuilt")) and (t / "cand").is_dir():
            dest = "base" if role == "rebuilt base (new code)" else "best"
            shutil.copytree(t / "cand", t / dest, dirs_exist_ok=True)
        shutil.rmtree(t / "cand", ignore_errors=True)
    with _lock:
        now = _read(d / "state.json", {})
        if now.get("status") != "running":  # paused meanwhile: keep the pause, and the work done
            st["status"] = now.get("status", st["status"])
        st["applied"] = now.get("applied", st.get("applied"))
        _write(d / "state.json", st)
        if line is not None:
            with open(d / "log.jsonl", "a") as f:
                f.write(json.dumps(line, default=float) + "\n")


# ---------------------------------------------------------------- API


MIN_ABOVE_START = 0.5  # by default a study shows only what beats its start by this


def _min_score(st: dict) -> Optional[float]:
    if st.get("min_score") is not None:
        return float(st["min_score"])
    return None if st.get("base_obj") is None else round(float(st["base_obj"]) + MIN_ABOVE_START, 2)


def _summary(st: dict) -> dict:
    return {
        "sid": st["sid"], "status": st["status"], "created": st["created"], "areas": st["areas"],
        "photos": [{"id": p["id"], "name": p["name"], "body": p["body"]} for p in st["photos"]],
        "evals": st["evals"], "max_evals": st["max_evals"], "round": st["round"], "step": st["step"],
        "base_obj": st["base_obj"], "best_obj": st["best_obj"], "kept": len(st["kept"]),
        "building": None if st["cand"] is None else {"change": st["cand"]["change"], "done": len(st["cand"]["results"])},
        "applied": st.get("applied"), "judged": st.get("judged"), "auto": st.get("auto", False),
    }


@router.get("")
def list_studies():
    out = []
    for d in sorted(STUDY_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        st = _read(d / "state.json", None) if d.is_dir() else None
        if st:
            out.append(_summary(st))
    tuned = knobs.read_tuned()
    return {"studies": out, "areas": knobs.AREAS, "objective": study.OBJECTIVE, "autopilot": autopilot_state(),
            "auto_rule": f"applies a finished study by itself when its mean beats its start by {AUTO_GAIN} and no photo got worse than {AUTO_WORST}",
            "tuned": {"source": tuned.get("source"), "at": tuned.get("at"), "values": tuned.get("values", {}),
                      "stale": _tuned_status["stale"]},
            "free_mb": round(_free() / 1e6)}


@router.get("/checklist")
def checklist():
    """Everything being taught (anatomy3d/curriculum.py), with what the
    studies have done for each item's numbers."""
    from anatomy3d import curriculum

    kept, tried = {}, {}
    for d in STUDY_ROOT.iterdir():
        st = _read(d / "state.json", None) if d.is_dir() else None
        if not st:
            continue
        for key in st.get("keys", []):
            tried[key] = tried.get(key, 0) + 1
        for k in st.get("kept", []):
            kept.setdefault(k["key"], []).append(round(float(k["gain"]), 2))
    out, groups = [], {}
    for it in curriculum.items():
        g = groups.setdefault(it.group, {"group": it.group, "total": 0, "done": 0, "studied": 0})
        g["total"] += 1
        g["done"] += it.done()
        studied = any(k in tried for k in it.knobs)
        g["studied"] += studied
        out.append({"id": it.id, "group": it.group, "name": it.name, "read": it.read, "sculpt": it.sculpt,
                    "knobs": len(it.knobs), "critic": it.critic, "note": it.note, "rig": it.rig, "done": it.done(),
                    "studied": studied, "kept_gains": [g_ for k in it.knobs for g_ in kept.get(k, [])]})
    return {"groups": list(groups.values()), "items": out,
            "total": len(out), "done": sum(i["done"] for i in out)}


@router.post("")
async def create_study(photos: List[UploadFile] = File(...), bodies: str = Form(""), areas: str = Form(""),
                       max_evals: int = Form(300)):
    if not 1 <= len(photos) <= 8:
        raise HTTPException(400, "A study takes 1 to 8 photos.")
    area_list = [a for a in areas.split(",") if a in knobs.AREAS] or list(knobs.AREAS)
    body_list = [b if b in ("male", "female") else "male" for b in bodies.split(",")] if bodies else []
    sid = uuid.uuid4().hex[:10]
    d = STUDY_ROOT / sid
    (d / "p").mkdir(parents=True)
    plist = []
    for i, up in enumerate(photos):
        suffix = Path(up.filename or "photo.jpg").suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            suffix = ".jpg"
        (d / "p" / f"p{i}{suffix}").write_bytes(await up.read())
        plist.append({"id": f"p{i}", "name": (up.filename or f"photo {i + 1}")[:80], "file": f"p{i}{suffix}",
                      "body": body_list[i] if i < len(body_list) else "male"})
    st = study.new_state(sid, plist, area_list, max(10, min(int(max_evals), 2000)), seed=int(time.time()) % 1000)
    st["started"] = time.time()
    with _lock:
        for other in STUDY_ROOT.iterdir():
            o = _read(other / "state.json", None) if other.is_dir() else None
            if o and o.get("status") == "running":
                o["status"] = "paused"
                _write(other / "state.json", o)
        _write(d / "state.json", st)
    return _summary(st)


@router.get("/{sid}")
def get_study(sid: str, log: int = 60):
    d = _dir(sid)
    st = _read(d / "state.json", {})
    lines = []
    p = d / "log.jsonl"
    if p.exists():
        lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    t = _renders(sid)
    floor = _min_score(st)
    shown = [x for x in lines if x.get("kept") == "baseline" or str(x.get("kept", "")).startswith("rebuilt")
             or floor is None or x.get("objective", -1e9) >= floor]
    kept_shown = [k for k in st["kept"] if floor is None or k["objective"] >= floor]
    return {
        **_summary(st),
        "min_score": floor,
        "hidden": {"moves": len(lines) - len(shown), "kept": len(st["kept"]) - len(kept_shown)},
        "apply_ok": bool(st["kept"]) and st["best_obj"] is not None and (floor is None or st["best_obj"] >= floor),
        "objective_formula": study.OBJECTIVE,
        "values": study.summary_values(st),
        "kept_moves": kept_shown,
        "base_per_photo": st["base_per_photo"], "best_per_photo": st["best_per_photo"],
        "renders": {p["id"]: {w: f"/api/study/{sid}/render/{w}/{p['id']}.png" for w in ("base", "best")
                              if (t / w / f"{p['id']}.png").exists()} for p in st["photos"]},
        "log": shown[-max(0, min(log, 500)):],
        "log_count": len(shown),
    }


@router.get("/{sid}/render/{which}/{pid}.png")
def render(sid: str, which: str, pid: str):
    _dir(sid)
    p = _renders(sid) / which / f"{pid}.png"
    if which not in ("base", "best") or not PID_RE.match(pid) or not p.exists():
        raise HTTPException(404, "No render.")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/{sid}/photo/{pid}")
def photo(sid: str, pid: str):
    d = _dir(sid)
    st = _read(d / "state.json", {})
    for p in st.get("photos", []):
        if p["id"] == pid:
            return FileResponse(d / "p" / p["file"])
    raise HTTPException(404, "No such photo.")


_generation_lock: Optional[threading.Lock] = None


def set_generation_lock(lock: threading.Lock) -> None:
    global _generation_lock
    _generation_lock = lock


@router.post("/tuned/clear")
def clear_tuned():
    with _generation_lock:
        knobs.clear_tuned()
    _tuned_status.update({"applied": {}, "stale": []})
    return {"ok": True}


@router.post("/{sid}/threshold")
def threshold(sid: str, value: Optional[float] = None):
    """Only results scoring at least `value` show (and can be applied);
    no value: back to the default (the start + MIN_ABOVE_START)."""
    d = _dir(sid)
    with _lock:
        st = _read(d / "state.json", {})
        st["min_score"] = value
        _write(d / "state.json", st)
    return {"min_score": _min_score(st)}


@router.post("/{sid}/{action}")
def control(sid: str, action: str):
    d = _dir(sid)
    if action not in ("pause", "resume", "apply"):
        raise HTTPException(404, "Unknown action.")
    with _lock:
        st = _read(d / "state.json", {})
        if action == "apply":
            floor = _min_score(st)
            if st.get("best_obj") is None or (floor is not None and st["best_obj"] < floor):
                raise HTTPException(409, f"The best result ({st.get('best_obj')}) is below the minimum score ({floor}).")
            values = {**knobs.current(), **st["x"]}
            with _generation_lock:  # no build mid-way while the numbers change
                result = knobs.save_tuned(values, source=f"study {sid}")
            _tuned_status.update(result)
            st["applied"] = time.time()
        elif st.get("status") != "done":
            if action == "resume":
                for other in STUDY_ROOT.iterdir():
                    o = _read(other / "state.json", None) if other.is_dir() else None
                    if o and o.get("status") == "running" and other.name != sid:
                        o["status"] = "paused"
                        _write(other / "state.json", o)
                st["started"] = time.time()
            st["status"] = "running" if action == "resume" else "paused"
        _write(d / "state.json", st)
    return _summary(st)

