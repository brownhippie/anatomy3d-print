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

import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from anatomy3d.pipeline import run_pipeline

app = FastAPI(title="anatomy3d-print")

WEBAPP_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")

JOBS_DIR = Path(tempfile.gettempdir()) / "anatomy3d-jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOB_TTL_SECONDS = 2 * 60 * 60  # 2 hours — generous for someone to come back and re-download
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _sweep_old_jobs() -> None:
    now = time.time()
    for entry in JOBS_DIR.iterdir():
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > JOB_TTL_SECONDS:
                shutil.rmtree(entry, ignore_errors=True)
        except FileNotFoundError:
            pass  # another request already cleaned it up — fine


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


@app.post("/api/generate")
async def api_generate(
    images: List[UploadFile] = File(...),
    height_mm: float = Form(150.0),
    use_depth: bool = Form(False),
):
    _sweep_old_jobs()

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

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

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            run_pipeline(image_paths, str(out_stl), target_height_mm=height_mm, use_depth=use_depth)
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        messages = [line for line in captured.getvalue().splitlines() if line.strip()]
        raise HTTPException(400, str(exc)) from exc

    messages = [line for line in captured.getvalue().splitlines() if line.strip()]

    mesh = trimesh.load(str(out_stl), process=True)
    height_actual = float(mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())

    return {
        "job_id": job_id,
        "obj_url": f"/jobs/{job_id}/model.obj",
        "stl_url": f"/jobs/{job_id}/model.stl",
        "glb_url": f"/jobs/{job_id}/model.glb",
        "messages": messages,
        "stats": {
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "height_mm": height_actual,
            "watertight": bool(mesh.is_watertight),
        },
    }


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
