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
        if first and (t / "base").is_dir():
            shutil.copytree(t / "base", t / "best", dirs_exist_ok=True)
        elif line.get("kept") is True and (t / "cand").is_dir():
            shutil.copytree(t / "cand", t / "best", dirs_exist_ok=True)
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


def _summary(st: dict) -> dict:
    return {
        "sid": st["sid"], "status": st["status"], "created": st["created"], "areas": st["areas"],
        "photos": [{"id": p["id"], "name": p["name"], "body": p["body"]} for p in st["photos"]],
        "evals": st["evals"], "max_evals": st["max_evals"], "round": st["round"], "step": st["step"],
        "base_obj": st["base_obj"], "best_obj": st["best_obj"], "kept": len(st["kept"]),
        "building": None if st["cand"] is None else {"change": st["cand"]["change"], "done": len(st["cand"]["results"])},
        "applied": st.get("applied"),
    }


@router.get("")
def list_studies():
    out = []
    for d in sorted(STUDY_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        st = _read(d / "state.json", None) if d.is_dir() else None
        if st:
            out.append(_summary(st))
    tuned = knobs.read_tuned()
    return {"studies": out, "areas": knobs.AREAS, "objective": study.OBJECTIVE,
            "tuned": {"source": tuned.get("source"), "at": tuned.get("at"), "values": tuned.get("values", {}),
                      "stale": _tuned_status["stale"]},
            "free_mb": round(_free() / 1e6)}


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
    return {
        **_summary(st),
        "objective_formula": study.OBJECTIVE,
        "values": study.summary_values(st),
        "kept_moves": st["kept"],
        "base_per_photo": st["base_per_photo"], "best_per_photo": st["best_per_photo"],
        "renders": {p["id"]: {w: f"/api/study/{sid}/render/{w}/{p['id']}.png" for w in ("base", "best")
                              if (t / w / f"{p['id']}.png").exists()} for p in st["photos"]},
        "log": lines[-max(0, min(log, 500)):],
        "log_count": len(lines),
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


@router.post("/{sid}/{action}")
def control(sid: str, action: str):
    d = _dir(sid)
    if action not in ("pause", "resume", "apply"):
        raise HTTPException(404, "Unknown action.")
    with _lock:
        st = _read(d / "state.json", {})
        if action == "apply":
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

