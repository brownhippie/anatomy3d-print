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
                        fams = ["O"] if inputs.capsule is None else ["C", "O"]
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
            (d / "t").mkdir(exist_ok=True)
            thumb.save(d / "t" / f"{code}.png", optimize=True)
            added = (d / "t" / f"{code}.png").stat().st_size
            if _folder_bytes() < LAB_MAX_BYTES:
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


def _worker_loop() -> None:
    while True:
        run_id = None
        try:
            run_id = _active_run()
            if run_id is None:
                time.sleep(3)
                continue
            _work_one(run_id)
            time.sleep(0.3)  # let a person's own Generate take the lock between combinations
        except Exception as exc:  # noqa: BLE001 - keep the worker alive whatever happens
            print(f"Lab worker error on {run_id}: {exc}")
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
    views = len(lab.FINISHES) * len(lab.SHADING)
    return {
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
        _write_json(d / "order.json", lab.run_order(["C", "O"], bust=False))
        _write_json(d / "meta.json", {
            "run_id": run_id, "photo_name": image.filename or "photo", "photo_file": f"photo{suffix}",
            "created": now, "started": now, "status": "running", "families": ["C", "O"],
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
        item.update({k: r.get(k) for k in ("score", "shape_iou", "spike_pct", "watertight", "min_wall_mm", "face_found", "face_err_pct", "seconds")})
        item["thumb_url"] = f"/api/lab/runs/{run_id}/thumb/{code}.png"
        item["model_url"] = f"/api/lab/runs/{run_id}/model/{code}.obj" if r.get("model") else None
    else:
        item["error"] = r.get("error")
    return item


@router.get("/runs/{run_id}")
def get_run(run_id: str, sort: str = "score", family: str = "all", offset: int = 0, limit: int = 40, only: str = "all"):
    d = _run_dir(run_id)
    ratings = _read_json(d / "ratings.json", {})
    rows = list(_results(d).values())
    if family in ("C", "O"):
        rows = [r for r in rows if r["code"].startswith(family + "-")]
    if only == "rated":
        rows = [r for r in rows if r["code"] in ratings]
    elif only == "failed":
        rows = [r for r in rows if r["status"] != "ok"]
    elif only == "ok":
        rows = [r for r in rows if r["status"] == "ok"]
    if sort == "score":
        rows.sort(key=lambda r: (r["status"] != "ok", -(r.get("score") or -1e9)))
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
