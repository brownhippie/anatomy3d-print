"""Combination Lab: automated runs of every build combination for one photo.

Upload a photo -> a run is created with the full ordered list of
combinations (anatomy3d.lab.run_order). One background worker builds and
scores them one at a time, taking the same lock as normal generation per
combination so a person's own Generate is never stuck behind a long run.
Everything a run produces lives under LAB_ROOT/<run_id>/ so it survives
restarts and redeploys when LAB_ROOT is on a persistent volume:
  meta.json      photo name, families, status, created
  order.json     every combination code, in run order
  results.jsonl  one line per finished combination (metrics or error)
  ratings.json   your stars/notes per code
  m/<code>.obj.gz, t/<code>.png   preview model and thumbnail
"""
import contextlib
import gzip
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from trimesh.exchange.obj import export_obj

from anatomy3d import lab

import study_api


def _lab_root() -> Path:
    env = os.environ.get("ANATOMY3D_LAB_DIR")
    if env:
        root = Path(env)
    elif Path("/data").is_dir() and os.access("/data", os.W_OK):
        root = Path("/data/lab")
    else:
        root = Path(tempfile.gettempdir()) / "anatomy3d-lab"
    root.mkdir(parents=True, exist_ok=True)
    return root


LAB_ROOT = _lab_root()
# Stop storing preview models (metrics and thumbnails still kept) once the
# lab's own folder passes this, so a long run can't fill the volume.
LAB_MAX_BYTES = int(float(os.environ.get("ANATOMY3D_LAB_MAX_GB", "9")) * 1e9)
# Below this much free space on the volume a run keeps only its metrics (no
# thumbnails or models), so a full volume never stops it or anything else.
LAB_MIN_FREE = 150e6
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")
CODE_RE = re.compile(r"^[A-Z]-[0-9]{1,16}$")

router = APIRouter(prefix="/api/lab")
_state_lock = threading.Lock()
_generation_lock: Optional[threading.Lock] = None
_inputs_cache: dict = {}  # run_id -> LabInputs (only the active run's)
_lab_bytes = {"total": None}


# ---------------------------------------------------------------- storage


def _run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(404, "No such run.")
    d = LAB_ROOT / run_id
    if not (d / "meta.json").exists():
        raise HTTPException(404, "No such run.")
    return d


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _results(d: Path) -> "dict[str, dict]":
    out = {}
    p = d / "results.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["code"]] = r
    return out


def _disk_free() -> float:
    try:
        return shutil.disk_usage(LAB_ROOT).free
    except OSError:
        return 0.0


def _folder_bytes() -> int:
    if _lab_bytes["total"] is None:
        _lab_bytes["total"] = sum(f.stat().st_size for f in LAB_ROOT.rglob("*") if f.is_file())
    return _lab_bytes["total"]


# ---------------------------------------------------------------- worker


def _active_run() -> Optional[str]:
    best, best_t = None, -1.0
    for d in LAB_ROOT.iterdir():
        meta = _read_json(d / "meta.json", None) if d.is_dir() else None
        if meta and meta.get("status") == "running" and meta.get("started", 0) > best_t:
            best, best_t = d.name, meta["started"]
    return best


def _inputs(run_id: str, d: Path):
    if run_id not in _inputs_cache:
        _inputs_cache.clear()
        meta = _read_json(d / "meta.json", {})
        _inputs_cache[run_id] = lab.prepare_inputs(str(d / meta["photo_file"]))
    return _inputs_cache[run_id]


def _work_one(run_id: str) -> bool:
    d = LAB_ROOT / run_id
    meta = _read_json(d / "meta.json", {})
    order = _read_json(d / "order.json", [])
    done = _results(d)
    pending = [c for c in order if c not in done and c.split("-")[0] in meta.get("families", [])]
    if not pending:
        with _state_lock:
            meta = _read_json(d / "meta.json", {})
            meta["status"] = "done"
            meta["finished"] = time.time()
            _write_json(d / "meta.json", meta)
        return False
    code = pending[0]
    t0 = time.time()
    row = {"code": code, "n": lab.number(code), "at": t0}
    with _generation_lock:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if "families_checked" not in meta:
                    inputs = _inputs(run_id, d)
                    with _state_lock:
                        meta = _read_json(d / "meta.json", {})
                        fams = (["O"] if inputs.capsule is None else ["C", "O"]) + (["B"] if inputs.reference is not None else [])
                        meta.update(families=fams, families_checked=True, capsule_unavailable=inputs.capsule_unavailable)
                        if inputs.bust_hint and not meta.get("reordered_for_bust"):
                            _write_json(d / "order.json", lab.run_order(fams, bust=True))
                            meta["reordered_for_bust"] = True
                        _write_json(d / "meta.json", meta)
                    return True
                inputs = _inputs(run_id, d)
                body = lab.build(code, inputs)
                metrics, preview, thumb = lab.score(code, body, inputs)
            row.update(status="ok", **metrics)
            rep = getattr(body, "report", None) or {}
            if "checks" in rep:
                # each step's reality check: how many ran, which failed
                row["checks_n"] = len(rep["checks"])
                row["checks_failed"] = rep["checks_failed"]
            added = 0
            if _disk_free() > LAB_MIN_FREE:
                (d / "t").mkdir(exist_ok=True)
                thumb.save(d / "t" / f"{code}.png", optimize=True)
                added = (d / "t" / f"{code}.png").stat().st_size
            if _folder_bytes() < LAB_MAX_BYTES and _disk_free() > LAB_MIN_FREE:
                (d / "m").mkdir(exist_ok=True)
                data = gzip.compress(export_obj(preview, include_normals=False, include_texture=False, digits=3).encode(), 6)
                (d / "m" / f"{code}.obj.gz").write_bytes(data)
                added += len(data)
                row["model"] = True
            else:
                row["model"] = False
            _lab_bytes["total"] = _folder_bytes() + added
        except Exception as exc:  # noqa: BLE001 - one bad combination mustn't stop the run
            row.update(status="fail", error=f"{type(exc).__name__}: {str(exc)[:200]}")
    row["seconds"] = round(time.time() - t0, 1)
    with _state_lock:
        with open(d / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    return True


def _release_memory() -> None:
    """Give freed memory back to the system after each job: a build peaks near
    2 GB in large numpy arrays, and without this the process kept what it had
    freed until the service ran out (8 GB) and was restarted mid-run."""
    import ctypes
    import gc

    # the models only reading a photo needs (cut-out, depth, the parts'
    # reader) and their working memory: 1.2 GB held between builds that never
    # use them -- loaded again when the next photo is read
    try:
        from anatomy3d import depth_source, head_parts, person_segmenter

        depth_source._session = None
        person_segmenter._birefnet_session = None
        head_parts._clipseg_sess = None
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _worker_loop() -> None:
    turn = 0
    while True:
        run_id = None
        try:
            turn += 1
            housekeeping()
            sid = study_api.active()
            if sid is None and study_api.autopilot_step(_generation_lock):
                _release_memory()
                continue
            run_id = _active_run()
            if sid and (run_id is None or turn % 3):  # a running study gets two turns to a run's one
                run_id = None
                study_api.work_one(sid, _generation_lock)
                _release_memory()
                time.sleep(0.3)
                continue
            if run_id is None:
                time.sleep(3)
                continue
            _work_one(run_id)
            _release_memory()
            time.sleep(0.3)  # let a person's own Generate take the lock between combinations
        except Exception as exc:  # noqa: BLE001 - keep the worker alive whatever happens
            print(f"Lab worker error on {run_id or 'a study'}: {exc}")
            if run_id:
                with _state_lock:
                    d = LAB_ROOT / run_id
                    meta = _read_json(d / "meta.json", {})
                    meta.update(status="error", error=str(exc)[:300])
                    _write_json(d / "meta.json", meta)
            time.sleep(5)


def start_worker(generation_lock: threading.Lock) -> None:
    global _generation_lock
    _generation_lock = generation_lock
    threading.Thread(target=_worker_loop, name="lab-worker", daemon=True).start()


# ---------------------------------------------------------------- API


@router.get("/space")
def space():
    fams = []
    for key, fam in lab.FAMILIES.items():
        fams.append({
            "key": key,
            "name": fam["name"],
            "count": lab.family_size(key),
            "dims": [{"key": k, "label": label, "options": [o[1] for o in opts]} for k, label, opts in fam["dims"]],
        })
    params = [{"key": k, "label": label, "kind": kind, "how": how, "held_rule": rule} for k, label, kind, how, rule in lab.PARAMS]
    views = len(lab.FINISHES) * len(lab.SHADING)
    return {
        "params": params,
        "families": fams,
        "total_builds": lab.total_builds(),
        "finishes": [f[1] for f in lab.FINISHES],
        "shading": [s[1] for s in lab.SHADING],
        "total_views": lab.total_builds() * views,
        "templates": [{"name": n, "code": lab._template_code(f, o)} for n, f, o in lab.TEMPLATES],
        "score_formula": lab.SCORE_FORMULA,
    }


@router.post("/runs")
async def create_run(image: UploadFile = File(...)):
    run_id = uuid.uuid4().hex[:12]
    d = LAB_ROOT / run_id
    d.mkdir(parents=True)
    suffix = Path(image.filename or "photo.jpg").suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        suffix = ".jpg"
    (d / f"photo{suffix}").write_bytes(await image.read())
    now = time.time()
    with _state_lock:
        for other in LAB_ROOT.iterdir():
            m = _read_json(other / "meta.json", None) if other.is_dir() else None
            if m and m.get("status") == "running":
                m["status"] = "paused"
                _write_json(other / "meta.json", m)
        _write_json(d / "order.json", lab.run_order(["C", "O", "B"], bust=False))
        _write_json(d / "meta.json", {
            "run_id": run_id, "photo_name": image.filename or "photo", "photo_file": f"photo{suffix}",
            "created": now, "started": now, "status": "running", "families": ["C", "O", "B"],
        })
    return {"run_id": run_id}


def _summary(d: Path) -> dict:
    meta = _read_json(d / "meta.json", {})
    order = _read_json(d / "order.json", [])
    results = _results(d)
    planned = [c for c in order if c.split("-")[0] in meta.get("families", ["C", "O"])]
    ok = [r for r in results.values() if r["status"] == "ok"]
    secs = [r["seconds"] for r in list(results.values())[-40:]]
    remaining = len(planned) - len(results)
    avg = (sum(secs) / len(secs)) if secs else None
    return {
        **{k: meta.get(k) for k in ("run_id", "photo_name", "status", "created", "families", "capsule_unavailable", "error")},
        "total": len(planned),
        "done": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "eta_hours": round(remaining * (avg + 0.3) / 3600, 1) if avg else None,
        "photo_url": f"/api/lab/runs/{meta.get('run_id')}/photo",
        "rated": len(_read_json(d / "ratings.json", {})),
    }


@router.get("/runs")
def list_runs():
    runs = []
    for d in sorted(LAB_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "meta.json").exists():
            runs.append(_summary(d))
    return {"runs": runs, "storage_gb": round(_folder_bytes() / 1e9, 2), "storage_cap_gb": round(LAB_MAX_BYTES / 1e9, 1)}


def _item(run_id: str, r: dict, ratings: dict) -> dict:
    code = r["code"]
    fam, idx = lab.decode(code)
    changed = [
        f"{label}: {opts[i][1]}" for (key, label, opts), i in zip(lab.FAMILIES[fam]["dims"], idx) if i != 0
    ]
    item = {
        "code": code, "n": r["n"], "family": lab.FAMILIES[fam]["name"], "template": lab.template_name(code),
        "changed": changed, "options": [f"{a}: {b}" for a, b in lab.describe(code)], "status": r["status"],
        "rating": ratings.get(code),
    }
    if r["status"] == "ok":
        item.update({k: r.get(k) for k in ("score", "shape_iou", "spike_pct", "watertight", "min_wall_mm", "face_found", "face_err_pct", "seconds", "params", "held_mean", "photo_detail", "checks_n", "checks_failed", "clothes_pct", "garment_edges", "garment_step_mm", "verified")})
        item["thumb_url"] = f"/api/lab/runs/{run_id}/thumb/{code}.png"
        item["model_url"] = f"/api/lab/runs/{run_id}/model/{code}.obj" if r.get("model") else None
    else:
        item["error"] = r.get("error")
    return item


@router.get("/runs/{run_id}")
def get_run(run_id: str, sort: str = "score", family: str = "all", offset: int = 0, limit: int = 40, only: str = "all",
            min_score: Optional[float] = None):
    """`min_score`: only combinations scoring at least this show; by
    default what the production combination scored on this photo (nothing
    below it is worth a look). Pass a very low number to see everything."""
    d = _run_dir(run_id)
    ratings = _read_json(d / "ratings.json", {})
    all_rows = list(_results(d).values())
    prod = next((r for r in all_rows if r["code"] == lab.CODE_PRODUCTION and r["status"] == "ok"), None)
    floor = min_score if min_score is not None else (prod.get("score") if prod else None)
    if only == "failed":
        floor = None  # the failures themselves, unscored
    rows = [r for r in all_rows if floor is None or (r["status"] == "ok" and (r.get("score") or -1e9) >= floor)
            or r["code"] in ratings]
    if family in ("C", "O", "B"):
        rows = [r for r in rows if r["code"].startswith(family + "-")]
    if only == "rated":
        rows = [r for r in rows if r["code"] in ratings]
    elif only == "failed":
        rows = [r for r in rows if r["status"] != "ok"]
    elif only == "ok":
        rows = [r for r in rows if r["status"] == "ok"]
    elif only == "verified":
        # sealed, printable, and the photo's clothes recognised on it (lab.score)
        rows = [r for r in rows if r["status"] == "ok" and r.get("verified")]
    if sort == "score":
        rows.sort(key=lambda r: (r["status"] != "ok", -(r.get("score") or -1e9)))
    elif sort == "held":
        rows.sort(key=lambda r: (r["status"] != "ok", -(r.get("held_mean") or -1e9)))
    elif sort == "rating":
        rows.sort(key=lambda r: (-(ratings.get(r["code"], {}).get("stars") or 0), -(r.get("score") or -1e9)))
    elif sort == "newest":
        rows.sort(key=lambda r: -r["at"])
    elif sort == "number":
        rows.sort(key=lambda r: r["n"])
    offset, limit = max(0, offset), max(1, min(limit, 200))
    return {
        "run": _summary(d),
        "count": len(rows),
        "min_score": floor,
        "production_score": prod.get("score") if prod else None,
        "hidden": len(all_rows) - len([r for r in all_rows if floor is None or (r["status"] == "ok" and (r.get("score") or -1e9) >= floor)
                                       or r["code"] in ratings]),
        "items": [_item(run_id, r, ratings) for r in rows[offset:offset + limit]],
    }


class Rating(BaseModel):
    code: str
    stars: int
    note: str = ""


@router.post("/runs/{run_id}/rate")
def rate(run_id: str, rating: Rating):
    d = _run_dir(run_id)
    if not CODE_RE.match(rating.code):
        raise HTTPException(400, "Bad combination code.")
    lab.decode(rating.code)
    with _state_lock:
        ratings = _read_json(d / "ratings.json", {})
        if rating.stars <= 0 and not rating.note.strip():
            ratings.pop(rating.code, None)
        else:
            ratings[rating.code] = {"stars": max(0, min(5, rating.stars)), "note": rating.note[:500], "at": time.time()}
        _write_json(d / "ratings.json", ratings)
    return {"ok": True, "rated": len(ratings)}


@router.post("/runs/{run_id}/{action}")
def control_run(run_id: str, action: str):
    if action == "drop-models":
        return drop_models(run_id)
    if action not in ("pause", "resume"):
        raise HTTPException(404, "Unknown action.")
    d = _run_dir(run_id)
    with _state_lock:
        if action == "resume":
            for other in LAB_ROOT.iterdir():
                m = _read_json(other / "meta.json", None) if other.is_dir() else None
                if m and m.get("status") == "running" and other.name != run_id:
                    m["status"] = "paused"
                    _write_json(other / "meta.json", m)
        meta = _read_json(d / "meta.json", {})
        if meta.get("status") != "done":
            meta["status"] = "running" if action == "resume" else "paused"
            meta["started"] = time.time() if action == "resume" else meta.get("started")
            _write_json(d / "meta.json", meta)
    return _summary(d)


@router.get("/runs/{run_id}/export")
def export(run_id: str):
    d = _run_dir(run_id)
    ratings = _read_json(d / "ratings.json", {})
    rows = sorted(_results(d).values(), key=lambda r: r["n"])
    return JSONResponse({
        "run": _summary(d),
        "score_formula": lab.SCORE_FORMULA,
        "ratings": [
            {"code": c, "n": lab.number(c), "options": lab.describe(c), **v} for c, v in sorted(ratings.items(), key=lambda kv: -kv[1]["stars"])
        ],
        "results": [{**r, "options": lab.describe(r["code"])} for r in rows],
    })


@router.get("/runs/{run_id}/photo")
def photo(run_id: str):
    d = _run_dir(run_id)
    meta = _read_json(d / "meta.json", {})
    return FileResponse(d / meta["photo_file"])


@router.get("/runs/{run_id}/thumb/{code}.png")
def thumb(run_id: str, code: str):
    d = _run_dir(run_id)
    if not CODE_RE.match(code) or not (d / "t" / f"{code}.png").exists():
        raise HTTPException(404, "No thumbnail.")
    return FileResponse(d / "t" / f"{code}.png", media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@router.get("/runs/{run_id}/model/{code}.obj")
def model(run_id: str, code: str):
    d = _run_dir(run_id)
    p = d / "m" / f"{code}.obj.gz"
    if not CODE_RE.match(code) or not p.exists():
        raise HTTPException(404, "No stored model for this combination.")
    return FileResponse(p, media_type="text/plain", headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=86400"})


@router.get("/runs/{run_id}/stl/{code}")
async def stl(run_id: str, code: str):
    """Full-quality printable STL, rebuilt on demand (previews are light)."""
    d = _run_dir(run_id)
    if not CODE_RE.match(code):
        raise HTTPException(400, "Bad combination code.")
    lab.decode(code)
    out = d / "stl"
    out.mkdir(exist_ok=True)
    path = out / f"{code}.stl"

    def make():
        from anatomy3d.mesh_export import to_trimesh
        from anatomy3d.print_prep import export_stl

        with _generation_lock, contextlib.redirect_stdout(io.StringIO()):
            body = lab.build(code, _inputs(run_id, d))
            export_stl(to_trimesh(body), str(path), target_faces=20000)

    if not path.exists():
        await run_in_threadpool(make)
    return FileResponse(path, media_type="model/stl", filename=f"figure-{code}.stl")


# ---------------------------------------------------------------- storage


def _bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


@router.get("/storage")
def storage():
    """What each run keeps on the volume: 3D previews (m/), thumbnails (t/)."""
    runs = []
    for d in sorted(LAB_ROOT.iterdir(), key=lambda p: p.stat().st_mtime):
        meta = _read_json(d / "meta.json", None) if d.is_dir() else None
        if meta:
            runs.append({"run_id": d.name, "photo_name": meta.get("photo_name"), "status": meta.get("status"),
                         "created": meta.get("created"), "models_mb": round(_bytes(d / "m") / 1e6, 1),
                         "thumbs_mb": round(_bytes(d / "t") / 1e6, 1), "rated": len(_read_json(d / "ratings.json", {}))})
    return {"runs": runs, "free_mb": round(_disk_free() / 1e6), "cap_gb": round(LAB_MAX_BYTES / 1e9, 1)}


def drop_models(run_id: str):
    """Removes a run's stored 3D previews (its scores, thumbnails and your
    ratings stay; Download STL still rebuilds any combination)."""
    d = _run_dir(run_id)
    meta = _read_json(d / "meta.json", {})
    if meta.get("status") == "running":
        raise HTTPException(409, "Pause the run first.")
    freed = _bytes(d / "m")
    shutil.rmtree(d / "m", ignore_errors=True)
    with _state_lock:
        rows = list(_results(d).values())
        for r in rows:
            r["model"] = False
        tmp = d / "results.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(d / "results.jsonl")
    _lab_bytes["total"] = None
    return {"freed_mb": round(freed / 1e6, 1), "free_mb": round(_disk_free() / 1e6)}


# ---------------------------------------------------------------- housekeeping
# The server clears out what nobody needs, on its own, so the volume never
# fills: rebuilt STLs after a day, study input caches once a study stops,
# and -- while the volume has under KEEP_FREE left or the Lab's folder is
# over its cap -- the 3D previews of runs that aren't running, oldest first,
# then their thumbnails (never a thumbnail you rated). Scores, ratings,
# photos, studies' results and tuned.json are never touched. Everything
# removed is logged to housekeeping.jsonl next to the Lab's folder.

KEEP_FREE = float(os.environ.get("ANATOMY3D_KEEP_FREE_GB", "1")) * 1e9
STL_TTL = 24 * 3600
HOUSEKEEPING_EVERY = 600
_housekeeping = {"last": 0.0}


def _housekeeping_log() -> Path:
    return LAB_ROOT.parent / "housekeeping.jsonl"


def _note(what: str, freed: int) -> None:
    if freed <= 0:
        return
    try:
        with open(_housekeeping_log(), "a") as f:
            f.write(json.dumps({"at": time.time(), "what": what, "freed_mb": round(freed / 1e6, 1)}) + "\n")
    except OSError:
        pass


def housekeeping(force: bool = False) -> None:
    now = time.time()
    if not force and now - _housekeeping["last"] < HOUSEKEEPING_EVERY:
        return
    _housekeeping["last"] = now
    runs = []
    for d in LAB_ROOT.iterdir():
        meta = _read_json(d / "meta.json", None) if d.is_dir() else None
        if meta:
            runs.append((meta.get("created", 0), d, meta))
    runs.sort(key=lambda r: r[0])
    for _, d, _ in runs:
        old = [p for p in (d / "stl").glob("*.stl") if now - p.stat().st_mtime > STL_TTL] if (d / "stl").is_dir() else []
        freed = sum(p.stat().st_size for p in old)
        for p in old:
            p.unlink(missing_ok=True)
        _note(f"rebuilt STLs of run {d.name}", freed)
    study_api.clear_idle_caches()

    def short() -> bool:
        return _disk_free() < KEEP_FREE or _folder_bytes() > LAB_MAX_BYTES

    idle = [(d, meta) for _, d, meta in runs if meta.get("status") != "running"]
    for d, _ in idle:
        if not short():
            break
        if (d / "m").is_dir():
            freed = _bytes(d / "m")
            drop_models(d.name)
            _note(f"3D previews of run {d.name}", freed)
    for d, _ in idle:
        if not short():
            break
        if (d / "t").is_dir():
            rated = set(_read_json(d / "ratings.json", {}))
            freed = 0
            for p in (d / "t").glob("*.png"):
                if p.stem not in rated:
                    freed += p.stat().st_size
                    p.unlink(missing_ok=True)
            _lab_bytes["total"] = None
            _note(f"unrated thumbnails of run {d.name}", freed)


@router.get("/housekeeping")
def housekeeping_log(limit: int = 30):
    p = _housekeeping_log()
    lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return {"removed": lines[-max(1, min(limit, 200)):][::-1], "keep_free_gb": round(KEEP_FREE / 1e9, 1),
            "free_mb": round(_disk_free() / 1e6)}
