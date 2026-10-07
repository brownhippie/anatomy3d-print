"""Minimal web front-end for the photo -> printable-mesh pipeline."""
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from anatomy3d.pipeline import run_pipeline

SMPLX_MODEL_DIR = os.environ.get("SMPLX_MODEL_DIR", "models/smplx")

app = FastAPI(title="anatomy3d-print")

INDEX_HTML = """
<!doctype html>
<html>
<head><title>anatomy3d-print</title></head>
<body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
  <h1>anatomy3d-print</h1>
  <p>Upload a clear, front-facing, full-body photo. You'll get back a
  3D-printable STL (plus an OBJ for other uses).</p>
  <form action="/fit" method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Generate model</button>
  </form>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/fit")
async def fit(image: UploadFile = File(...)):
    if not os.path.exists(SMPLX_MODEL_DIR):
        raise HTTPException(
            500,
            f"SMPL-X model files not found at {SMPLX_MODEL_DIR}. "
            "See README for how to provide them.",
        )

    with tempfile.TemporaryDirectory() as tmp:
        image_path = os.path.join(tmp, image.filename or "upload.jpg")
        with open(image_path, "wb") as f:
            f.write(await image.read())

        out_stl = os.path.join(tmp, "figure.stl")
        try:
            run_pipeline(image_path, SMPLX_MODEL_DIR, out_stl)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

        # FileResponse streams before the TemporaryDirectory is cleaned up on
        # most ASGI servers, but copy to a stable path to be safe across workers.
        persisted = os.path.join(tempfile.gettempdir(), f"anatomy3d-{os.getpid()}.stl")
        os.replace(out_stl, persisted)
        return FileResponse(persisted, filename="figure.stl", media_type="model/stl")
