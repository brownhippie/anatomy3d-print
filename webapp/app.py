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
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from anatomy3d.pipeline import run_pipeline, run_trellis_pipeline
from anatomy3d.person_segmenter import detect_person_alpha
from anatomy3d.preprocess import load_image_rgb
from anatomy3d.trellis_source import get_quota_status, token_configured, token_preview

app = FastAPI(title="anatomy3d-print")

WEBAPP_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")

JOBS_DIR = Path(tempfile.gettempdir()) / "anatomy3d-jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOB_TTL_SECONDS = 2 * 60 * 60  # 2 hours — generous for someone to come back and re-download
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Process-lifetime counters, not persisted — restarts reset them, same as
# this process's own JOBS_DIR. Good enough for "is this getting used /
# falling back a lot" visibility; not a billing-grade audit log.
_trellis_stats = {"attempts": 0, "successes": 0, "fallbacks": 0}


def _sweep_old_jobs() -> None:
    now = time.time()
    for entry in JOBS_DIR.iterdir():
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > JOB_TTL_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except FileNotFoundError:
            pass  # another request already cleaned it up — fine


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
    return (WEBAPP_DIR / "templates" / "index.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.post("/api/generate")
async def api_generate(
    images: List[UploadFile] = File(...),
    height_mm: float = Form(150.0),
    use_depth: bool = Form(False),
    method: str = Form("capsule"),
    description: str = Form(""),
):
    if method not in ("capsule", "trellis"):
        raise HTTPException(400, f"Unknown method '{method}' -- use 'capsule' or 'trellis'.")
    if method == "trellis" and len(images) != 1:
        raise HTTPException(400, "The trellis method takes exactly one photo -- it has no multi-view mode.")

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
    out_obj = job_dir / "model.obj"
    out_glb = job_dir / "model.glb"

    used_method = method
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            if method == "trellis":
                _trellis_stats["attempts"] += 1
                try:
                    run_trellis_pipeline(image_paths[0], str(out_stl), target_height_mm=height_mm)
                    _trellis_stats["successes"] += 1
                except Exception as trellis_exc:
                    # Fall back to this project's own always-available pipeline rather
                    # than just failing the request -- TRELLIS.2 depends on an external
                    # service with its own quota/uptime this project doesn't control
                    # (see trellis_source.py), so a visitor still gets a model instead
                    # of an error when it's unavailable. Weaker result, but a real one.
                    _trellis_stats["fallbacks"] += 1
                    print(f"TRELLIS.2 failed, falling back to the built-in method: {trellis_exc}")
                    run_pipeline(image_paths, str(out_stl), target_height_mm=height_mm, use_depth=use_depth)
                    used_method = "capsule (fallback)"
            else:
                run_pipeline(image_paths, str(out_stl), target_height_mm=height_mm, use_depth=use_depth)
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        messages = [line for line in captured.getvalue().splitlines() if line.strip()]
        raise HTTPException(400, str(exc)) from exc

    messages = [line for line in captured.getvalue().splitlines() if line.strip()]

    mesh = trimesh.load(str(out_stl), process=True)
    height_actual = float(mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())

    # Front photo only — a cutout is a single-view concept, same as the
    # "front first" convention run_pipeline's own docstring documents for
    # multi-photo visual-hull mode.
    has_cutout = _write_cutout(image_paths[0], job_dir / "cutout.png")

    result = {
        "job_id": job_id,
        "obj_url": f"/jobs/{job_id}/model.obj",
        "stl_url": f"/jobs/{job_id}/model.stl",
        "glb_url": f"/jobs/{job_id}/model.glb",
        "messages": messages,
        "method_used": used_method,
        "description": description or None,
        "stats": {
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "height_mm": height_actual,
            "watertight": bool(mesh.is_watertight),
        },
    }
    if has_cutout:
        result["cutout_url"] = f"/jobs/{job_id}/cutout.png"
    return result


@app.get("/jobs/{job_id}/model.obj")
def get_obj(job_id: str):
    path = _job_dir(job_id) / "model.obj"
    if not path.exists():
        raise HTTPException(404, "Not found.")
    return FileResponse(path, media_type="text/plain", filename="figure.obj")


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
            run_pipeline(image_paths, out_stl, target_height_mm=height_mm, use_depth=use_depth)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

        persisted = os.path.join(tempfile.gettempdir(), f"anatomy3d-{os.getpid()}.stl")
        os.replace(out_stl, persisted)
        return FileResponse(persisted, filename="figure.stl", media_type="model/stl")
