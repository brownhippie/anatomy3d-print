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

No license gate, no registration needed — but pose detection does fetch a
~9MB model file automatically on first run (`landmarks.py`, cached at
`~/.cache/anatomy3d-print/`). This is MediaPipe's own official model,
Apache-2.0 licensed like the rest of MediaPipe, from Google's model
bucket, not something you need to register for or accept a license for —
unlike SMPL-X, downloading it is just normal setup, not a legal step.

MediaPipe's native library also needs a few system packages even for
CPU-only use — `libgl1`, `libglib2.0-0`, `libgles2`, `libegl1` (already in
the Dockerfile; install them yourself if running outside a container,
e.g. `apt install libgl1 libglib2.0-0 libgles2 libegl1` on
Debian/Ubuntu). Found by actually running real pose detection end-to-end
in this project's dev environment, not assumed from documentation — every
earlier test in this project had mocked MediaPipe out entirely, which is
exactly why this was still broken: `landmarks.py` originally called
`mediapipe.solutions.pose.Pose(...)`, an older API that plain doesn't
exist in any current pip-installable MediaPipe build (confirmed on
0.10.30 through 1.1.0) — only `mediapipe.tasks.python.vision`, the
current API, which is what this module now uses.

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
photos, front+both-sides (0°, 90°, 270°, skipping the back) for three, and
evenly spaced for four or more. These aren't arbitrary — measured against
an analytic shape with known ground-truth volume, naive even-spacing for 3
views ([0°, 120°, 240°]) landed 30° short of the depth-constraining 90°/270°
angles and measured +45% error on that axis; swapping to [0°, 90°, 270°]
measured -3% on the same shape. 4+ views land on or near the cardinal
angles on their own.

Outputs an STL ready for a slicer, plus an OBJ of the same mesh for
rendering/animation use elsewhere. `--height-mm` sets the printed figure's
height (a photo gives no reliable real-world scale, so the whole figure is
normalized to this target height rather than guessed from the photo).

Every photo (either mode) goes through `preprocess.py` first: EXIF
rotation correction, downscaling to a sane working resolution, contrast
normalization, and unsharp-mask sharpening — a soft/dim phone photo feeds
MediaPipe and the silhouette extractor much worse edges than a crisp one,
and this costs nothing to fix before detection runs.

## MAD margins throughout

`safety.py` holds a small MAD-style margin utility — `(actual - minimum) /
minimum`, requiring real headroom past a limit rather than a bare pass/fail
at the edge — carried over from unrelated research of mine on a different
project (a physical safety margin for a laser display). There, higher was
dangerous (a ceiling not to exceed); most limits in this pipeline are
floors (a minimum not to go under), so `mad_margin_above_minimum` mirrors
the formula direction; `mad_margin_below_maximum` keeps the original one.

Every place this pipeline used to do a bare `if count < minimum: raise`
now reports the real margin instead, warning when a hard requirement is
only barely met even though it technically passed. None of the reference
values below were left as guesses — each was checked against real output
(synthetic test meshes with known/controllable ground truth) and adjusted
where the data disagreed with the original pick.

- **Wall thickness** (`print_prep.py`) — `estimate_min_wall_thickness`
  samples points across the mesh surface, casts a ray inward from each to
  measure local thickness, checked against a minimum printable thickness
  (0.8mm default — two perimeters at a typical 0.4mm nozzle). This one
  needed a real fix, not just a margin tweak: it originally reported the
  bare minimum across samples, which calibration testing showed doesn't
  converge — on a fixed mesh, the measured value kept dropping as sample
  count rose (0.65mm mean at 300 samples, 0.05mm at 6000) because a bare
  minimum is an order statistic that keeps chasing whatever's most
  extreme, and this geometry has a genuine near-zero point: an exact
  mathematical tangent in the capsule smooth-min blend (percentile-0
  measured 0.006mm, a 150x jump to percentile-0.5's 0.98mm) — a
  zero-measure artifact, not a real wall. It now reports the 2nd
  percentile instead, which represents actual surface area rather than
  one singular point and measured far more stable (~9% relative
  run-to-run variation at 300 samples, ~4% by 750, versus the old
  minimum's ~58%+ and climbing). Concretely: meshes that used to report
  "77% BELOW the minimum, will likely fail to print" with the old metric
  now correctly measure 1.8-3mm and pass — the old check was a false
  alarm, not a conservative one. Still sampling-based, not exhaustive;
  still check the result in your slicer.
- **Keypoint count** (`landmarks.py`, `body_fit.py`) — how many of the
  needed body landmarks MediaPipe actually detected with confidence.
- **Capsule count** (`procedural_body.py`) — how many of the 10 possible
  body-part capsules had enough keypoints to place. Calibration surfaced
  two real structural bugs here, not just a number to retune: (1) the
  mesh's height-normalization scale was computed *before* discarding
  debris fragments (see the next section), so a debris sliver sitting
  beyond the real body's top/bottom silently shrank the final printed
  height by ~12% in one measured case; (2) missing hips, or legs that
  don't reach the ankles, meant the mesh's lowest point was the pelvis or
  a knee instead of a foot — scaling that truncated span to fill the full
  target height inflated the figure by 2-4x in measured cases, and raw
  capsule count didn't catch it (5+ capsules could still be present).
  Both found their own hard gates (hips and full leg chains now required,
  with a clear error instead of a silently wrong mesh) rather than a
  margin tweak. With those gates in place, torso+both legs always forms,
  so the true floor is 5 capsules, not the original guess of 3.
- **View count** (`visual_hull.py`) — reasoned differently from the
  others: 2 views (front+side) is an intentionally supported mode, not a
  degraded one, so the margin is measured against a 4-view full-turntable
  *recommendation*, not the hard 2-view mathematical floor — otherwise
  every 2-view run would trip a "barely passing" warning for using the
  pipeline as designed. Calibrating this against an analytic ellipsoid
  with known ground-truth volume also surfaced a default-angle bug: naive
  even-spacing for 3 views lands 30 degrees short of the depth-constraining
  90/270 angles and measured +45% error on that axis; `pipeline.py` now
  uses [0, 90, 270] for 3 views specifically, which measured -3% on the
  same shape (see `_default_angles`'s docstring).
- **Color-fit sample count** (`silhouette.py`) — how many foreground/
  background pixels the statistical reclassifier had to fit its color
  model from. Testing against synthetic overlapping-color classes with
  known ground truth found classification *accuracy* essentially flat
  across the whole range tested (color overlap sets a hard ceiling more
  data doesn't lift), but run-to-run *variance* dropped sharply with more
  samples (std 0.028 at 16 samples, 0.002 by ~2000) — the original 200
  sat in the still-noisy part of that curve; recalibrated to 800, past
  the knee where returns flatten out.

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
