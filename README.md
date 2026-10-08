# anatomy3d-print

Fit a photo of a person to a stylized 3D body figure, built entirely from
scratch (no licensed model, no gated training data), and export a
watertight, 3D-printable mesh.

## Why "from scratch"

An earlier version of this project used SMPL-X, a body model trained on
real 3D body scans. It looks more realistic, but its free license is
**non-commercial research use only** — commercial use requires a separate
paid license from Meshcapade. Since the goal here is a project you can
eventually commercialize without anyone else's permission, the default
pipeline no longer depends on it. The SMPL-X path still exists as an
optional, clearly-marked research mode (see below) for anyone who wants
higher fidelity under that license.

## Two reconstruction modes

**One photo** → `procedural_body.py`: a geometric guess. Places a skeleton
from MediaPipe keypoints, wraps each bone in a tapered capsule sized from
generic figure-proportion ratios (figure-drawing references, not a
licensed dataset) scaled by that bone's own measured length, blends
overlapping capsules with a smooth-minimum of their signed distance fields,
extracts the result with marching cubes. Fast, but it's a stylized guess —
it doesn't know the actual shape of the person in the photo, only their
joint proportions.

**Two or more photos, shot from different angles around the subject**
(front first, then rotating — a phone-selfie "turnaround" or someone else
walking around you works) → `visual_hull.py`: classical visual hull / space
carving (Laurentini 1994; Kutulakos & Seitz 2000). `silhouette.py` extracts
each photo's subject outline in two stages: threshold against the sampled
background color (carried over from unrelated research of mine on a
different project, where it was specified for the same background-removal
problem), then a single-pass statistical reclassification — fit
foreground/background color distributions from that threshold result and
re-decide every pixel by which it actually fits, the same core idea as
GrabCut's color model without pulling in OpenCV for it. (A third stage,
snapping the boundary to image edges with an active contour, was tried and
reverted — on a figure with a narrow neck and separated legs, it pulled the
contour straight across both and deleted the head and most of the legs.
Documented in `silhouette.py` in case it's worth revisiting with a
shape-aware safety check instead of the area-ratio one that let it through.)
Each silhouette rules out everything outside it; carving a voxel grid down
to what survives every view recovers the subject's *actual* cross-section,
not a guess. Two photos (front + side) already constrain both width and
depth; more photos narrow it further. This is strictly more information
than one photo can ever give — the real tradeoff for better accuracy is
taking more photos, not a bigger model.

Both paths go through `print_prep.py` (repairs the mesh: merges vertices,
fills holes, fixes normals, drops stray disconnected debris down to the
main body) and export STL/OBJ via `pipeline.py` / `cli.py`.

Every piece here is either measured directly from your photos or a
well-known, unencumbered formula/algorithm — no third-party model weights,
no training-data license, nothing to clear before you can use this
commercially.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

No model downloads, no license gate, no registration needed.

## Usage

One photo (capsule guess):

```bash
python -m anatomy3d.cli \
  --image path/to/photo.jpg \
  --out output/figure.stl \
  --height-mm 150
```

Multiple photos (visual hull — front first, then rotating around the subject):

```bash
python -m anatomy3d.cli \
  --image front.jpg side.jpg back.jpg \
  --out output/figure.stl \
  --height-mm 150
```

`--angles` lets you override the assumed rotation angles (degrees) if you
didn't shoot an even turntable; it defaults to front+side (0°, 90°) for two
photos, evenly spaced for three or more.

Outputs an STL ready for a slicer, plus an OBJ of the same mesh for
rendering/animation use elsewhere. `--height-mm` sets the printed figure's
height (a photo gives no reliable real-world scale, so the whole figure is
normalized to this target height rather than guessed from the photo).

Every photo (either mode) goes through `preprocess.py` first: EXIF
rotation correction, downscaling to a sane working resolution, contrast
normalization, and unsharp-mask sharpening — a soft/dim phone photo feeds
MediaPipe and the silhouette extractor much worse edges than a crisp one,
and this costs nothing to fix before detection runs.

## Printability check (MAD margin)

Every export runs a wall-thickness check before writing the STL:
`print_prep.estimate_min_wall_thickness` samples points across the mesh
surface and casts a ray inward from each to measure local thickness there,
then reports the thinnest point found. That gets compared against a
minimum printable thickness (0.8mm by default — two perimeters at a
typical 0.4mm nozzle; tune it for your printer/material) using a MAD-style
margin: `(actual - minimum) / minimum`, requiring at least 15% of headroom
past the minimum, not just barely clearing it. This convention is carried
over from unrelated research of mine on a different project (a physical
safety margin for a laser display), adapted here — there, higher was
dangerous (a limit not to exceed); here, lower is dangerous (a floor not
to go under), so the margin direction is mirrored (see `safety.py`).

This is a sampling-based estimate, not an exhaustive check — with a few
thousand samples it reliably catches a broadly thin region (a whole limb),
but could miss one single pin-thin spot the samples didn't land near.
Still check the result in your slicer.

## Desktop app (.exe)

`desktop_app.py` is a Tkinter GUI over the same pipeline — pick a photo,
set the target height, pick an output path, click Generate.

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

## Web app / Railway deployment

`webapp/app.py` is a small FastAPI front end: upload a photo, get back an
STL. Run it locally with:

```bash
uvicorn app:app --app-dir webapp --reload
```

The included `Dockerfile` and `railway.toml` deploy as-is on Railway —
no license-gated files to host or inject, since the default pipeline
doesn't need any.

## Limitations (read before printing)

**Single-photo (capsule) mode:**
- **Stylized, not realistic.** Tapered limbs, a simple head, no face, no
  hands/fingers, no clothing. It will not look like a scanned likeness of
  the person. That's the direct trade-off for not depending on a licensed
  body model.
- **No real depth.** Built with a fixed, stylized front-to-back thickness,
  not measured depth. Proportions (limb lengths, torso width) come from the
  photo; roundness doesn't.

**Multi-photo (visual hull) mode:**
- **Can't recover concavities.** This is a fundamental property of visual
  hull, not a bug: the armpit gap between an arm and the torso, for
  example, won't be recovered unless some view's silhouette actually shows
  daylight through that gap. Arms held close to the body tend to fuse into
  the torso in the result.
- **Assumes consistent framing.** All photos need the subject at roughly
  the same distance/zoom and centered the same way — the carving shares one
  pixel-to-world scale across every view (see `visual_hull.py`'s docstring).
  A photo shot noticeably closer or further than the others will distort
  the result.
- **No face/hands/clothing texture**, same as single-photo mode — this
  recovers shape, not surface detail.

**Both modes:**
- **Check the STL in your slicer before printing.** The repair step fixes
  common issues (non-manifold edges, small holes) but very thin joints may
  still need manual cleanup in Meshmixer/Netfabb/Blender depending on your
  printer's nozzle size and material.

## Optional: higher-fidelity SMPL-X mode (non-commercial only)

`body_fit.py` still contains a from-scratch SMPLify-X-style fitter that
optimizes an actual SMPL-X body to the photo's keypoints, which looks
noticeably more realistic. It's not used by the default pipeline or wired
into the CLI/web/desktop apps. To experiment with it:

1. `pip install -r requirements-smplx.txt`
2. Register at https://smpl-x.is.tue.mpg.de, accept the license (read it —
   it's non-commercial research use only), and download the "SMPL-X v1.1"
   package. You need `models/smplx/SMPLX_NEUTRAL.npz` specifically (not the
   plain "SMPL" or "SMPL+H" packages).
3. Call `anatomy3d.body_fit.fit_body(...)` directly (see its docstring) —
   it's deliberately not connected to `pipeline.py`, so using it is a
   conscious opt-in, not something that happens by accident in a
   commercial build.

## Project layout

```
src/anatomy3d/
  preprocess.py       EXIF fix, resize, contrast + sharpen
  landmarks.py         2D pose keypoint detection
  procedural_body.py    single-photo: capsule/SDF body builder
  silhouette.py          multi-photo: background removal -> mask
  visual_hull.py           multi-photo: voxel carving -> mesh
  body_fit.py                optional: SMPL-X fitting (non-commercial license)
  mesh_types.py        shared BodyMesh type
  mesh_export.py        OBJ export
  safety.py              generic MAD-style margin checks
  print_prep.py            mesh repair + wall-thickness check (safety.py) + STL export
  pipeline.py                orchestration (picks capsule vs. hull by photo count)
  cli.py                    command-line entry point
webapp/app.py        FastAPI web front end
desktop_app.py       Tkinter desktop GUI (-> .exe via PyInstaller)
Dockerfile           Railway/container build for the web app
railway.toml         Railway build/deploy config
.github/workflows/   CI: builds the Windows .exe on tag push
requirements-smplx.txt  optional extras for the SMPL-X research mode
```
