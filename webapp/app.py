"""Web front-end for the photo -> printable-mesh pipeline.

Generates the model server-side, then serves it to an in-browser three.js
viewer (static/app.js) so you can look at the actual result — rotated,
sitting on a ground plane like a print bed — before downloading anything.
Each generation is a "job": its OBJ/STL live in their own temp directory,
served back by id, swept away after JOB_TTL_SECONDS so disk usage doesn't
grow unbounded on a long-running deployment.
"""
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from anatomy3d.pipeline import run_pipeline, run_silhouette_relief_pipeline, run_trellis_pipeline
from anatomy3d.person_segmenter import detect_person_alpha
from anatomy3d.preprocess import load_image_rgb
from anatomy3d.landmarks import PoseUnusableError
from anatomy3d.procedural_body import HipsNotDetectedError
from anatomy3d.trellis_source import get_quota_status, token_configured, token_preview

app = FastAPI(title="anatomy3d-print")

WEBAPP_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")

JOBS_DIR = Path(tempfile.gettempdir()) / "anatomy3d-jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOB_TTL_SECONDS = 2 * 60 * 60  # 2 hours — generous for someone to come back and re-download
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _detect_commit() -> str:
    """Computed once at process start, not per-request -- a direct,
    unambiguous way to confirm a given browser/app session is actually
    talking to a given deploy (compare against the commit a change was
    pushed as), instead of guessing from whether the UI looks different.
    RAILWAY_GIT_COMMIT_SHA is set automatically on Railway; falls back to
    asking git directly for local/non-Railway runs, and to "unknown"
    only if neither is available (e.g. a source-less container)."""
    sha = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")
    if sha:
        return sha[:7]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(WEBAPP_DIR), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


_COMMIT = _detect_commit()

# Process-lifetime counters, not persisted — restarts reset them, same as
# this process's own JOBS_DIR. Good enough for "is this getting used /
# falling back a lot" visibility; not a billing-grade audit log.
_trellis_stats = {"attempts": 0, "successes": 0, "fallbacks": 0}

# In-memory job status, polled by the client instead of held open over one
# long request -- see /api/generate's own comment for why: a phone app
# backgrounded (or killed and reopened) mid-generation was losing the
# result even though the actual work, running in its own thread here,
# never stopped. Each entry: {"status": "running"} while in flight, then
# either {"status": "done", **result} or {"status": "error", "detail": ...}
# once _run_job finishes, plus "created_at" so _sweep_old_jobs can expire
# it the same way job_dir itself already gets swept.
_job_state: "dict[str, dict]" = {}
_job_state_lock = threading.Lock()

# Confirmed directly, not assumed, as the cause of the whole app going
# unresponsive during a generate call -- not just that one request, but
# totally unrelated ones too: Railway's own http logs showed plain GET
# /health requests stuck for 15-20s whenever a generate call was in
# flight. Cause: /api/generate and /fit were `async def` endpoints that
# called the heavy, synchronous, CPU-bound run_pipeline/run_trellis_pipeline
# directly -- that blocks FastAPI's single asyncio event loop for the
# entire duration (tens of seconds to several minutes), during which NO
# request of any kind can be served, generate or otherwise.
# run_in_threadpool moves that blocking call to a worker thread so the
# event loop stays free. A lock still serializes the actual pipeline work
# itself (this is a small, CPU-constrained container -- two generate
# calls fighting over the same cores wouldn't finish any faster run truly
# in parallel) AND because contextlib.redirect_stdout reassigns the
# process-wide sys.stdout, which two concurrent calls would corrupt for
# each other if allowed to run at once.
_generation_lock = threading.Lock()

# Combination Lab (see lab_api.py). Imported by path so it resolves the same
# whether the server is started as `webapp.app:app` or `app:app --app-dir webapp`.
sys.path.insert(0, str(WEBAPP_DIR))
from lab_api import router as lab_router, start_worker as start_lab_worker, production_like as lab_production_like  # noqa: E402
from anatomy3d import ledger  # noqa: E402
import study_api  # noqa: E402

app.include_router(lab_router)
app.include_router(study_api.router)


@app.on_event("startup")
def _start_lab_worker() -> None:
    # approved study values (tuned.json) in place before any build
    study_api.load_tuned_at_startup()
    study_api.set_generation_lock(_generation_lock)
    start_lab_worker(_generation_lock)


def _run_pipeline_capturing(fn, *args, **kwargs) -> "list[str]":
    """Runs `fn` (a run_pipeline/run_trellis_pipeline call) with its
    print() progress messages captured instead of going to the real
    stdout, serialized against any other in-flight generation via
    _generation_lock -- see that lock's own docstring for why. Meant to
    be called via run_in_threadpool, never directly on the event loop."""
    with _generation_lock:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            fn(*args, **kwargs)
        return [line for line in captured.getvalue().splitlines() if line.strip()]


def _sweep_old_jobs() -> None:
    now = time.time()
    for entry in JOBS_DIR.iterdir():
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > JOB_TTL_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except FileNotFoundError:
            pass  # another request already cleaned it up — fine
    with _job_state_lock:
        expired = [jid for jid, s in _job_state.items() if now - s.get("created_at", now) > JOB_TTL_SECONDS]
        for jid in expired:
            _job_state.pop(jid, None)


# The full-detail OBJ (kept for download) can run ~1M triangles / 40MB on
# Silhouette relief -- far too much for a phone to fetch just to preview.
PREVIEW_MAX_FACES = 100_000


def _write_preview(obj_path: Path, out_path: Path) -> bool:
    """A lighter copy of the OBJ for the in-app viewer only, when the full
    one is over PREVIEW_MAX_FACES. Shape only: the viewer applies its own
    material and computes its own smooth normals. Returns False (viewer
    falls back to the full OBJ) when no preview is needed or it fails."""
    try:
        full = trimesh.load(str(obj_path), process=False, force="mesh")
        if len(full.faces) <= PREVIEW_MAX_FACES:
            return False
        preview = full.simplify_quadric_decimation(face_count=PREVIEW_MAX_FACES)
        out_path.write_text(
            export_obj(preview, include_normals=False, include_color=False, include_texture=False, digits=4)
        )
        return True
    except Exception:
        return False


def _write_cutout(image_path: str, out_path: Path) -> bool:
    """Best-effort 2D cutout of the front photo as a real transparent PNG
    (soft per-pixel alpha, not a hard boolean) — reuses the same
    detect_person_alpha this job's own silhouette step already calls, at
    full resolution for the same reason extract_silhouette is (boundary
    precision). A missing/failed detection is not fatal to the job, same
    "a failed bonus feature shouldn't fail the run" pattern pipeline.py
    uses for face/hair/silhouette. Returns whether a file was written."""
    try:
        rgb = load_image_rgb(image_path, max_dimension=None).rgb
        alpha = detect_person_alpha(rgb)
        if alpha is None:
            return False
        rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        return True
    except Exception:
        return False


def _job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(404, "Not found.")
    path = JOBS_DIR / job_id
    if not path.is_dir():
        raise HTTPException(404, "Job not found or expired.")
    return path


@app.get("/", response_class=HTMLResponse)
def index():
    # style.css/app.js are referenced by plain /static/... URLs with no
    # cache-busting of their own, so a browser (or an Android WebView,
    # which tends to cache harder) can keep serving an old cached copy of
    # them even once a fresh index.html has landed -- a real, confirmed
    # cause of a deploy visibly not taking effect on a phone (the page
    # loads, the build tag even updates since that's a live fetch, but
    # the old CSS/JS is still what's actually running). Tagging both URLs
    # with the current commit makes them a new URL on every deploy, so
    # the stale cached copy is simply never asked for again.
    html = (WEBAPP_DIR / "templates" / "index.html").read_text()
    html = html.replace('href="/static/style.css"', f'href="/static/style.css?v={_COMMIT}"')
    html = html.replace('src="/static/app.js"', f'src="/static/app.js?v={_COMMIT}"')
    return html


@app.get("/study", response_class=HTMLResponse)
def study_page():
    return (WEBAPP_DIR / "templates" / "study.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/version")
def version():
    return {"commit": _COMMIT}


@app.get("/api/trellis-status")
def trellis_status():
    """Lets the UI show whether TRELLIS.2 mode is actually usable right
    now, and how it's been doing in this server process — not a secret
    endpoint, just visibility, same spirit as /health. token_preview never
    exposes the real token (see trellis_source.py)."""
    return {
        "token_configured": token_configured(),
        "token_preview": token_preview(),
        "quota": get_quota_status(),
        "stats": dict(_trellis_stats),
    }


# The three quick adjustments the Result tab offers (see app.js). Face
# detail and photo depth shape outline-based builds; thickness stretches
# any model front-to-back. None = photo depth off.
_FACE_DETAIL = {"off": 0.0, "normal": None, "strong": 1.0}
_PHOTO_DEPTH = {"off": None, "normal": 0.5, "strong": 1.0}


@app.post("/api/generate")
async def api_generate(
    images: List[UploadFile] = File(...),
    height_mm: float = Form(150.0),
    use_depth: bool = Form(False),
    method: str = Form("capsule"),
    description: str = Form(""),
    bust_mode: bool = Form(False),
    crop_frac: Optional[float] = Form(None),
    thickness: float = Form(1.0),
    face_detail: str = Form("normal"),
    photo_depth: str = Form("normal"),
    body: str = Form("male"),
):
    if crop_frac is not None and not 0 < crop_frac <= 1:
        raise HTTPException(400, f"crop_frac must be in (0, 1], got {crop_frac}")
    if not 0.5 <= thickness <= 2.0:
        raise HTTPException(400, f"thickness must be between 0.5 and 2.0, got {thickness}")
    if face_detail not in _FACE_DETAIL or photo_depth not in _PHOTO_DEPTH:
        raise HTTPException(400, "face_detail and photo_depth must each be 'off', 'normal' or 'strong'.")
    if body not in ("male", "female"):
        raise HTTPException(400, "body must be 'male' or 'female'.")
    tweaks = {"thickness": thickness, "face_detail": face_detail, "photo_depth": photo_depth, "body": body}
    if method not in ("capsule", "trellis", "silhouette_relief"):
        raise HTTPException(400, f"Unknown method '{method}' -- use 'capsule', 'trellis', or 'silhouette_relief'.")
    if method in ("trellis", "silhouette_relief") and len(images) != 1:
        raise HTTPException(400, f"The {method} method takes exactly one photo -- it has no multi-view mode.")

    # Not yet consumed by either reconstruction method -- confirmed
    # directly, not assumed: TRELLIS.2's own Space API has no text
    # parameter on its image_to_3d endpoint, and neither of this project's
    # own pipelines has a text-conditioning hook. Stored with the job
    # purely as user-facing metadata/reference for now (see app.js's own
    # honest hint text next to the field), and a natural place to plug in
    # real text-conditioning later if this project adds a model that
    # supports it. Capped so a pasted essay doesn't bloat job storage.
    description = description.strip()[:2000]

    _sweep_old_jobs()

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    if description:
        (job_dir / "description.txt").write_text(description)

    input_dir = job_dir / "input"
    input_dir.mkdir()
    image_paths = []
    for i, image in enumerate(images):
        path = input_dir / f"{i}_{image.filename or 'upload.jpg'}"
        with open(path, "wb") as f:
            f.write(await image.read())
        image_paths.append(str(path))

    out_stl = job_dir / "model.stl"

    with _job_state_lock:
        _job_state[job_id] = {"status": "running", "created_at": time.time()}

    # Handed off to its own real OS thread, not awaited here -- the point
    # is that this request can return immediately, before generation even
    # starts, so the client isn't holding one long-lived connection open
    # for the whole job. A phone backgrounding the app (or the OS killing
    # it outright) no longer loses the result: the thread keeps running on
    # the server regardless of whether anyone is still connected, same as
    # it always did -- the only thing that changes is the client now polls
    # /api/jobs/{id}/status for it instead of blocking on one response,
    # so reopening the app (with the job_id it saved before backgrounding)
    # picks the finished result straight back up. daemon=True so a server
    # restart doesn't hang waiting on an orphaned job.
    threading.Thread(
        target=_run_job,
        args=(job_id, method, image_paths, height_mm, use_depth, bust_mode, crop_frac, description, job_dir, out_stl, tweaks),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "running"}


_WALL_RE = re.compile(r"thinnest measured region ~([0-9.]+)mm")


def _thinnest_wall(messages: "list[str]") -> Optional[float]:
    """The thinnest wall the print file's own export measured (print_prep.export_stl
    says it in its messages), mm at the model's own size; None if it wasn't measured.
    Read, not measured again: measuring it here cost a render about a minute."""
    for line in reversed(messages or []):
        m = _WALL_RE.search(line)
        if m:
            return round(float(m.group(1)), 2)
    return None


def _prints(mesh: trimesh.Trimesh, wall: Optional[float]) -> bool:
    return bool(mesh.is_watertight) and (wall is None or wall >= ledger.WALL_MIN_MM)


def _pick_for(method: str, image_paths: "list[str]") -> Optional[str]:
    """The production pick, when it is something other than the app's own
    build and this request can use it (one photo, the built-in methods)."""
    if method not in ("capsule", "silhouette_relief") or len(image_paths) != 1:
        return None
    code = (ledger.pick() or {}).get("code")
    if not code or lab_production_like(code):
        return None
    return code


def _build_with_pick(code: str, photo: str, height_mm: float, job_dir: Path, report_progress) -> "tuple[list[str], str, str]":
    """lab.build(code) in a process of its own (a build can take 6 GB), under
    the generation lock, written as the app's own files."""
    import json as _json

    report_progress(f"Building with the picked setting {code}", 0.2)
    (job_dir / "pick_request.json").write_text(_json.dumps({"code": code, "photo": photo, "height_mm": height_mm}))
    with _generation_lock:
        rc = subprocess.run([sys.executable, str(WEBAPP_DIR / "lab_job.py"), "build", str(job_dir)], timeout=40 * 60).returncode
    done = _json.loads((job_dir / "pick_done.json").read_text()) if (job_dir / "pick_done.json").exists() else {}
    if rc != 0 or "code" not in done:
        raise RuntimeError(done.get("error") or f"its process ended ({rc})")
    return [f"Built with setting {done['code']}."] + list(done.get("messages") or []), "picked setting", done["code"]


def _code_for(used_method: str, tweaks: dict) -> str:
    """The app's own build as a label for the ledger and your ratings."""
    plain = tweaks.get("thickness", 1.0) == 1.0 and tweaks.get("face_detail") == "normal" and tweaks.get("photo_depth") == "normal"
    return f"app:{used_method}" + ("" if plain else ":tweaked")


def _queue_measure(job_id, photo, job_dir, code, method, tweaks, height_mm, watertight, wall, pick, render_seconds) -> None:
    import json as _json

    p = ledger.root() / "pending" / job_id
    p.mkdir(parents=True, exist_ok=True)
    shutil.copy(photo, p / ("photo" + Path(photo).suffix.lower()))
    model = job_dir / "model.obj"
    shutil.copy(model if model.exists() else job_dir / "model.stl", p / ("model.obj" if model.exists() else "model.stl"))
    (p / "meta.json").write_text(_json.dumps({
        "counter": ledger.next_counter(), "at": time.time(), "photo": "photo" + Path(photo).suffix.lower(),
        "model": "model.obj" if model.exists() else "model.stl", "code": code, "method": method, "tweaks": tweaks,
        "height_mm": height_mm, "at_size": {"watertight": bool(watertight), "wall_mm": wall}, "pick": pick,
        "render_seconds": render_seconds,
    }))


def _run_job(
    job_id: str,
    method: str,
    image_paths: "list[str]",
    height_mm: float,
    use_depth: bool,
    bust_mode: bool,
    crop_frac: "Optional[float]",
    description: str,
    job_dir: Path,
    out_stl: Path,
    tweaks: Optional[dict] = None,
    force_code: Optional[str] = None,
    compare_of: Optional[str] = None,
) -> None:
    """Runs entirely on its own background thread (see the Thread(...)
    call in api_generate above) -- never called directly from the event
    loop, and its caller does not wait on it. Dispatches method/fallback,
    builds the final result the client's status poll will see, and writes
    it into _job_state; never raises back into that thread's caller since
    there is no caller left listening by the time this runs."""
    tweaks = tweaks or {"thickness": 1.0, "face_detail": "normal", "photo_depth": "normal", "body": "male"}
    thickness = tweaks["thickness"]
    face_strength = _FACE_DETAIL[tweaks["face_detail"]]
    depth_strength = _PHOTO_DEPTH[tweaks["photo_depth"]]

    def report_progress(stage: str, frac: float) -> None:
        # Overwrites the "running" entry _job_state already has for this
        # job with its stage/progress added -- the status poll (and
        # resumePendingJob after an app restart) reads this same key, so
        # this is the one place a percentage/countdown in the UI comes
        # from. Harmless if a job finishes between this read and write:
        # the "done"/"error" write that follows replaces it entirely.
        with _job_state_lock:
            _job_state[job_id] = {
                "status": "running",
                "stage": stage,
                "progress": frac,
                "created_at": time.time(),
            }

    def run_capsule() -> "tuple[list[str], Optional[str]]":
        # A waist-up photo has no hips for the capsule builder to anchor a
        # torso to; rebuild it as a bust instead of failing the job.
        try:
            # Depth stays off for capsules: measured on a real full-body photo,
            # it pulled the face into a beak and broke the hand into spikes.
            # It's used by the outline-based builds below instead.
            return _run_pipeline_capturing(
                run_pipeline,
                image_paths,
                str(out_stl),
                target_height_mm=height_mm,
                use_depth=False,
                progress_cb=report_progress,
                thickness=thickness,
            ), None
        except (HipsNotDetectedError, PoseUnusableError) as exc:
            # A waist-up photo (no hips) or one with no usable human pose at
            # all -- an animal, most often -- can't be built from capsules,
            # but its outline still can. Real depth is always on here: it's
            # what gives an outline-based model its arms, clothing and face.
            if len(image_paths) != 1:
                raise
            is_bust = isinstance(exc, HipsNotDetectedError)
            label = "Rebuilding as a bust" if is_bust else "Building from the outline"
            messages = _run_pipeline_capturing(
                run_silhouette_relief_pipeline,
                image_paths[0],
                str(out_stl),
                target_height_mm=height_mm,
                use_depth=depth_strength is not None,
                depth_strength=depth_strength or 0.5,
                thickness=thickness,
                face_strength=face_strength,
                body=tweaks.get("body", "male"),
                bust_mode=is_bust,
                crop_frac=crop_frac,
                # Continue the count from where the capsule attempt stopped
                # instead of restarting it.
                progress_cb=lambda stage, frac: report_progress(f"{label}: {stage}", 0.5 + 0.5 * frac),
            )
            if is_bust:
                note = (
                    "No hips in this photo, so it was built as a bust: Silhouette relief "
                    "in bust mode (no legs added, flat base where the photo ends)."
                )
                return [note] + messages, "silhouette_relief (bust)"
            note = f"The body builder couldn't use this photo ({exc}), so it was built from the photo's outline instead."
            return [note] + messages, "silhouette_relief (outline)"

    def run_default() -> "tuple[list[str], str]":
        if method == "trellis":
            _trellis_stats["attempts"] += 1
            try:
                messages = _run_pipeline_capturing(
                    run_trellis_pipeline,
                    image_paths[0],
                    str(out_stl),
                    target_height_mm=height_mm,
                    progress_cb=report_progress,
                    thickness=thickness,
                )
                _trellis_stats["successes"] += 1
                used_method = method
            except Exception as trellis_exc:
                # Fall back to this project's own always-available pipeline rather
                # than just failing the job -- TRELLIS.2 depends on an external
                # service with its own quota/uptime this project doesn't control
                # (see trellis_source.py), so a visitor still gets a model instead
                # of an error when it's unavailable. Weaker result, but a real one.
                _trellis_stats["fallbacks"] += 1
                fallback_note = f"TRELLIS.2 failed, falling back to the built-in method: {trellis_exc}"
                messages, bust_method = run_capsule()
                used_method = bust_method or "capsule (fallback)"
                messages = [fallback_note] + messages
        elif method == "silhouette_relief":
            messages = _run_pipeline_capturing(
                run_silhouette_relief_pipeline,
                image_paths[0],
                str(out_stl),
                target_height_mm=height_mm,
                use_depth=use_depth and depth_strength is not None,
                depth_strength=depth_strength or 0.5,
                thickness=thickness,
                face_strength=face_strength,
                body=tweaks.get("body", "male"),
                bust_mode=bust_mode,
                crop_frac=crop_frac,
                progress_cb=report_progress,
            )
            used_method = method
        else:
            messages, bust_method = run_capsule()
            used_method = bust_method or method
        return messages, used_method

    t_render = time.time()
    used_code = None
    try:
        pk = force_code or _pick_for(method, image_paths)
        if pk:
            try:
                messages, used_method, used_code = _build_with_pick(pk, image_paths[0], height_mm, job_dir, report_progress)
            except Exception as pick_exc:  # noqa: BLE001 - the app's own build instead
                messages, used_method = run_default()
                messages = [f"The picked setting {pk} couldn't build this photo ({pick_exc}); built with the app's own setting."] + messages
        else:
            messages, used_method = run_default()
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        with _job_state_lock:
            _job_state[job_id] = {"status": "error", "detail": str(exc), "created_at": time.time()}
        return

    mesh = trimesh.load(str(out_stl), process=True)
    # the printer's limits on this model at its own size, before it's handed over
    wall = _thinnest_wall(messages)
    if used_code and not _prints(mesh, wall):
        what = "an unsealed model" if not mesh.is_watertight else f"a {wall} mm wall"
        try:
            again, used_method = run_default()
            messages = [f"The picked setting {used_code} gave {what}; built again with the app's own setting."] + again
            used_code = None
            mesh = trimesh.load(str(out_stl), process=True)
            wall = _thinnest_wall(messages)
        except Exception:  # noqa: BLE001 - the picked one is kept then
            pass
    height_actual = float(mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())

    # Front photo only — a cutout is a single-view concept, same as the
    # "front first" convention run_pipeline's own docstring documents for
    # multi-photo visual-hull mode.
    has_cutout = _write_cutout(image_paths[0], job_dir / "cutout.png")

    report_progress("Preparing 3D preview", 0.99)
    has_preview = _write_preview(job_dir / "model.obj", job_dir / "preview.obj")

    result = {
        "status": "done",
        "created_at": time.time(),
        "job_id": job_id,
        "obj_url": f"/jobs/{job_id}/model.obj",
        "preview_url": f"/jobs/{job_id}/preview.obj" if has_preview else f"/jobs/{job_id}/model.obj",
        "stl_url": f"/jobs/{job_id}/model.stl",
        "glb_url": f"/jobs/{job_id}/model.glb",
        "messages": messages,
        "method_used": used_method,
        "description": description or None,
        "tweaks": tweaks,
        "stats": {
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "height_mm": height_actual,
            "watertight": bool(mesh.is_watertight),
        },
    }
    if has_cutout:
        result["cutout_url"] = f"/jobs/{job_id}/cutout.png"
    code = used_code or _code_for(used_method, tweaks)
    result["code"] = code
    result["compare_of"] = compare_of
    result["compare_url"] = None if compare_of else f"/api/jobs/{job_id}/compare"
    result["request"] = {"method": method, "height_mm": height_mm, "use_depth": use_depth, "bust_mode": bust_mode, "crop_frac": crop_frac}
    result["stats"]["thinnest_wall_mm"] = wall
    result["rate_url"] = f"/api/jobs/{job_id}/rate"
    with _job_state_lock:
        _job_state[job_id] = result
    # the reading, after the model is handed over (the Lab's worker measures it between its own builds)
    try:
        _queue_measure(job_id, image_paths[0], job_dir, code, used_method, tweaks, height_mm, mesh.is_watertight, wall,
                       used_code, round(time.time() - t_render, 1))
    except Exception as exc:  # noqa: BLE001 - the reading never costs the render
        print(f"Ledger: render {job_id} not queued: {exc}")


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str):
    """Polled by the client instead of holding one request open for the
    whole generation (see api_generate) -- "running" while _run_job is
    still working, then whatever it last wrote: "done" with the full
    result the old synchronous response used to return directly, or
    "error" with the failure detail. 404 for a job_id this process never
    saw (wrong id) or has since swept (see _sweep_old_jobs) -- the client
    treats either the same way, same as a job_dir 404 already does."""
    with _job_state_lock:
        state = _job_state.get(job_id)
    if state is None:
        raise HTTPException(404, "Job not found or expired.")
    return state


@app.post("/api/jobs/{job_id}/rate")
def rate_job(job_id: str, stars: int = Form(...), note: str = Form("")):
    """Your verdict on a render: 1-5 stars and a note, into the ledger. The
    production pick moves to a setting you rated higher, and never to one
    you rated 2 or less."""
    if not JOB_ID_RE.match(job_id) or not 1 <= stars <= 5:
        raise HTTPException(400, "stars must be 1 to 5.")
    with _job_state_lock:
        state = _job_state.get(job_id)
    code = (state or {}).get("code")
    if not code:
        raise HTTPException(404, "Job not found or expired.")
    ledger.rate("app", job_id, code, stars, note)
    try:
        now = ledger.choose_after_rating()
    except Exception:  # noqa: BLE001
        now = ledger.pick()
    return {"ok": True, "code": code, "stars": stars, "pick": now.get("code")}


@app.post("/api/jobs/{job_id}/compare")
def compare_job(job_id: str):
    """The same photo built with the strongest other setting (ledger.challenger),
    to rate beside the first: the app learns the setting you rate higher."""
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(404, "Job not found or expired.")
    with _job_state_lock:
        state = dict(_job_state.get(job_id) or {})
    src = JOBS_DIR / job_id / "input"
    if state.get("status") != "done" or not src.is_dir():
        raise HTTPException(404, "Job not found or expired.")
    current = (ledger.pick() or {}).get("code") or ledger.held_production()
    code = ledger.challenger(current)
    if not code:
        raise HTTPException(409, "No other setting to compare yet: the held job hasn't been read on this server.")
    req = state.get("request") or {}
    new_id = uuid.uuid4().hex
    jd = JOBS_DIR / new_id
    (jd / "input").mkdir(parents=True)
    paths = []
    for f in sorted(src.iterdir()):
        shutil.copy(f, jd / "input" / f.name)
        paths.append(str(jd / "input" / f.name))
    with _job_state_lock:
        _job_state[new_id] = {"status": "running", "created_at": time.time()}
    threading.Thread(
        target=_run_job,
        args=(new_id, req.get("method", "capsule"), paths[:1], float(req.get("height_mm", 150.0)), bool(req.get("use_depth", False)),
              bool(req.get("bust_mode", False)), req.get("crop_frac"), "", jd, jd / "model.stl", state.get("tweaks")),
        kwargs={"force_code": code, "compare_of": job_id},
        daemon=True,
    ).start()
    return {"job_id": new_id, "code": code}


@app.get("/jobs/{job_id}/model.obj")
def get_obj(job_id: str):
    path = _job_dir(job_id) / "model.obj"
    if not path.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(path, media_type="text/plain", filename="figure.obj")


@app.get("/jobs/{job_id}/preview.obj")
def get_preview(job_id: str):
    path = _job_dir(job_id) / "preview.obj"
    if not path.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(path, media_type="text/plain")


@app.get("/jobs/{job_id}/model.stl")
def get_stl(job_id: str):
    path = _job_dir(job_id) / "model.stl"
    if not path.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(path, media_type="model/stl", filename="figure.stl")


@app.get("/jobs/{job_id}/model.glb")
def get_glb(job_id: str):
    path = _job_dir(job_id) / "model.glb"
    if not path.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(path, media_type="model/gltf-binary", filename="figure.glb")


@app.get("/jobs/{job_id}/cutout.png")
def get_cutout(job_id: str):
    path = _job_dir(job_id) / "cutout.png"
    if not path.exists():
        raise HTTPException(404, "No cutout for this job (detection may have failed on the front photo).")
    return FileResponse(path, media_type="image/png", filename="cutout.png")


@app.post("/fit")
async def fit(
    images: List[UploadFile] = File(...),
    height_mm: float = Form(150.0),
    use_depth: bool = Form(False),
):
    """Direct-download endpoint for scripting/curl use — the main UI at
    `/` uses /api/generate instead, to preview before downloading."""
    with tempfile.TemporaryDirectory() as tmp:
        image_paths = []
        for i, image in enumerate(images):
            path = os.path.join(tmp, f"{i}_{image.filename or 'upload.jpg'}")
            with open(path, "wb") as f:
                f.write(await image.read())
            image_paths.append(path)

        out_stl = os.path.join(tmp, "figure.stl")
        try:
            await run_in_threadpool(
                _run_pipeline_capturing, run_pipeline, image_paths, out_stl, target_height_mm=height_mm, use_depth=use_depth
            )
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

        persisted = os.path.join(tempfile.gettempdir(), f"anatomy3d-{os.getpid()}.stl")
        os.replace(out_stl, persisted)
        return FileResponse(persisted, filename="figure.stl", media_type="model/stl")
