"""Minimal web front-end for the photo -> printable-mesh pipeline."""
import os
import tempfile
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from anatomy3d.pipeline import run_pipeline

app = FastAPI(title="anatomy3d-print")

INDEX_HTML = """
<!doctype html>
<html>
<head><title>anatomy3d-print</title></head>
<body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
  <h1>anatomy3d-print</h1>
  <p>Upload one clear, front-facing, full-body photo for a quick geometric
  guess, or several photos shot at different angles around the subject
  (front first) for a visual-hull reconstruction of the actual shape.
  You'll get back a 3D-printable STL (plus an OBJ for other uses).</p>
  <form action="/fit" method="post" enctype="multipart/form-data">
    <input type="file" name="images" accept="image/*" required multiple><br><br>
    <label>Height (mm): <input type="number" name="height_mm" value="150" min="20" max="1000"></label><br><br>
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
async def fit(images: List[UploadFile] = File(...), height_mm: float = Form(150.0)):
    with tempfile.TemporaryDirectory() as tmp:
        image_paths = []
        for i, image in enumerate(images):
            path = os.path.join(tmp, f"{i}_{image.filename or 'upload.jpg'}")
            with open(path, "wb") as f:
                f.write(await image.read())
            image_paths.append(path)

        out_stl = os.path.join(tmp, "figure.stl")
        try:
            run_pipeline(image_paths, out_stl, target_height_mm=height_mm)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

        # FileResponse streams before the TemporaryDirectory is cleaned up on
        # most ASGI servers, but copy to a stable path to be safe across workers.
        persisted = os.path.join(tempfile.gettempdir(), f"anatomy3d-{os.getpid()}.stl")
        os.replace(out_stl, persisted)
        return FileResponse(persisted, filename="figure.stl", media_type="model/stl")
