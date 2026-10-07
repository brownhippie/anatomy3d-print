# anatomy3d-print

Fit a photo of a person to an anatomically-correct parametric body model
(SMPL-X), then export a watertight, 3D-printable mesh.

## How it works

1. **`landmarks.py`** — runs MediaPipe Pose on the input photo to get 2D
   body keypoints (the "pattern recognition" step).
2. **`body_fit.py`** — optimizes SMPL-X shape (`betas`), pose, global
   orientation, and a weak-perspective camera so the model's projected
   joints line up with the detected 2D keypoints. SMPL-X's shape space is
   trained on thousands of real body scans, so regularizing `betas` toward
   zero during fitting keeps the result inside that learned, anatomically
   plausible distribution rather than letting it collapse into an
   arbitrary blob.
3. **`print_prep.py`** — repairs the resulting mesh (merges vertices, fills
   holes, fixes normals) and exports STL/OBJ.
4. **`pipeline.py`** / **`cli.py`** — wires the three steps together.

This is original glue code built around two existing, proven components
(MediaPipe for 2D keypoints, SMPL-X for the body prior) — not a wrapper
around a packaged image-to-3D product.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

### SMPL-X model files (required, not included)

SMPL-X model weights are distributed under their own license and can't be
redistributed in this repo. Register and download them yourself:

1. Go to https://smpl-x.is.tue.mpg.de, register, accept the license.
2. Download `SMPLX_NEUTRAL.npz` (or MALE/FEMALE variants).
3. Place it under `models/smplx/SMPLX_NEUTRAL.npz` in this repo (the
   `models/` directory is gitignored).

## Usage

```bash
python -m anatomy3d.cli \
  --image path/to/photo.jpg \
  --smplx-model-dir models/smplx \
  --gender neutral \
  --out output/figure.stl
```

Outputs an STL ready for a slicer, plus an OBJ of the same mesh for
rendering/animation use elsewhere.

## Desktop app (.exe)

`desktop_app.py` is a Tkinter GUI over the same pipeline — pick a photo,
pick your SMPL-X model folder, pick an output path, click Generate.

Run it directly:

```bash
python desktop_app.py
```

To get a standalone Windows `.exe`, push a tag (`git tag v0.1 && git push
origin v0.1`) or trigger the "Build Windows .exe" workflow manually from
the Actions tab — it runs PyInstaller on a real `windows-latest` runner
(a Windows binary has to be built on Windows; this repo can't cross-compile
it) and uploads `anatomy3d-print.exe` as a build artifact, attaching it to
the release if triggered by a tag.

The exe will be large (torch + mediapipe bundled) and slow to start the
first time — that's expected for a PyInstaller `--onefile` build with
these dependencies.

## Web app / Railway deployment

`webapp/app.py` is a small FastAPI front end: upload a photo, get back an
STL. Run it locally with:

```bash
uvicorn app:app --app-dir webapp --reload
```

To deploy on Railway, the included `Dockerfile` and `railway.toml` are
ready to go once this repo is connected to a Railway service. The only
thing you must set yourself is where the container gets its SMPL-X
weights from — they're license-gated and never committed here:

1. Host your own downloaded `SMPLX_NEUTRAL.npz` somewhere only you
   control (a private S3/GCS URL, a private Hugging Face repo file URL
   with a token, etc.) — never a public URL, since redistributing it
   violates the SMPL-X license.
2. Set the Railway service variable `SMPLX_MODEL_URL` to that private
   URL.
3. `entrypoint.sh` downloads it into the container on startup before
   launching the server.

## Limitations (read before printing)

- **Single photo, frontal pose assumed.** Anything not visible in the
  photo (the back, occluded limbs) is filled in by the body prior, not
  reconstructed from real data. Likeness accuracy drops fast outside the
  photographed angle. Front + side + back photos and multi-view fitting
  is the natural next step if this matters to you.
- **Weak-perspective camera.** The fitting ignores true depth/perspective
  distortion. Fine for a roughly frontal, centered photo; not accurate for
  close-up or extreme-angle shots.
- **No clothing or texture.** SMPL-X models the body shape, not clothing
  — output is a bare body mesh. Texturing/clothing reconstruction is a
  separate, harder problem not covered here.
- **Check the STL in your slicer before printing.** The repair step fixes
  common issues (non-manifold edges, small holes) but thin parts (fingers,
  ankles) may still need manual thickening in Meshmixer/Netfabb/Blender
  depending on your printer's nozzle size and material.

## Project layout

```
src/anatomy3d/
  landmarks.py     2D pose keypoint detection
  body_fit.py       SMPL-X optimization against keypoints
  mesh_export.py    OBJ export
  print_prep.py     mesh repair + STL export
  pipeline.py        orchestration
  cli.py             command-line entry point
webapp/app.py        FastAPI web front end
desktop_app.py       Tkinter desktop GUI (-> .exe via PyInstaller)
Dockerfile           Railway/container build for the web app
entrypoint.sh        Fetches SMPL-X weights at container start, then serves
railway.toml         Railway build/deploy config
.github/workflows/   CI: builds the Windows .exe on tag push
models/              SMPL-X weights go here (not tracked)
```
