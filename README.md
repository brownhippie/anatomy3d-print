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

**One photo** → `procedural_body.py`: places a skeleton from MediaPipe
keypoints, wraps each bone in a tapered capsule, blends overlapping
capsules with a smooth-minimum of their signed distance fields, extracts
the result with marching cubes. Capsule radii start from generic
figure-proportion ratios (figure-drawing references, not a licensed
dataset) scaled by that bone's own measured length, but are then refined
against the photo's own silhouette where one is available — see "Real
body shape, not generic capsules" below — so the output isn't just joint
proportions wrapped in a generic mannequin shape, though it's still a
single 2D photo's worth of information, not a scan.

Within that single-photo mode, the body is modeled **section by section**
rather than as one tapered torso capsule: head, chest, waist, hips, and
each limb each get their own capsule, and each cross-section gets its own
front-to-back/side-to-side depth ratio instead of assuming a round
cross-section everywhere (people aren't round — a chest is noticeably
wider than it is deep, hips less so, a head is close to even). These
ratios (`DEPTH_RATIO_*` in `procedural_body.py`) are generic, widely-cited
figure-drawing/character-modeling proportions, the same kind of
unencumbered reference the radius ratios already used — not a licensed
anthropometric dataset, and still a stylized mannequin, not a medical
cast. The waist sits narrower than both chest and hips, and each torso
capsule's depth ratio is itself interpolated end-to-end (chest ratio →
waist ratio → hip ratio) so the flattening changes smoothly along the
torso instead of jumping where the chest and hip capsules meet. This adds
real shape information beyond joint positions alone without pulling in
any model weights or training data.

Single-photo mode also adds real **face detail** (nose, chin, eye
sockets) when a face is detected — see "Face detail" below. It still does
not attempt fingers, clothing folds, or surface texture (skin, fabric,
hair strands) — those need a learned *generative* model (one trained to
hallucinate plausible fine detail, not just locate real landmarks), which
reopens the exact licensing problem this project exists to avoid, so
they're deliberately out of scope.

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

## Face detail

Single-photo mode automatically runs `face_features.py` on the same
photo, using MediaPipe's **Face Landmarker** — the same Apache-2.0
MediaPipe model family as the pose landmarker already required, not a new
dependency or license to clear. When a face is detected, it overlays a
nose, chin, and recessed eye sockets built from the photo's own measured
landmark positions on top of the generic head capsule, instead of a
single round head shape. It's automatic, not an opt-in flag — unlike
`--use-depth` (which pulls in torch), this needs nothing beyond what
single-photo mode already requires, and a missing/unusable face just
falls back to the plain head.

Face detection runs at the photo's full resolution rather than inheriting
pose detection's 1280px cap — a face is often small in a full-body frame
and benefits from the extra pixels — which means face landmarks and the
body skeleton usually come from two differently-scaled copies of the same
photo. Missed in initial testing (every synthetic photo used happened to
already be under 1280px, where this made no difference) and only caught
by testing against an artificially large (3000px) real photo: without
correcting for it, face geometry would land at the wrong size and
position relative to the body on any real phone photo above that
resolution. Fixed by rescaling face-local coordinates into the body
skeleton's own coordinate space before combining them — both describe the
same photo, just sampled at different pixel densities.

What's real here versus what's still generic: the nose/chin/eye *shape*
(how big, how rounded) is still a generic proportion like the rest of the
body, scaled off the photo's own measured eye-to-eye distance (the
standard figure-drawing "one eye-width" unit) — but the *positions* are
the photo's own, including a real measured nose protrusion depth, not a
guess. Confirmed against a real photo (MediaPipe's own `portrait.jpg`
sample) before use, not assumed from documentation: the nose tip lands
between the eyes and above the chin as it should, the chin is the lowest
point on the face, and the eye/jaw landmark pairs come out left-right
symmetric around the face centerline.

This needed its own real fix, not just wiring up a new model: a nose or
chin sized right for an actual face is tiny next to a whole body's
bounding box, and at this project's normal marching-cubes resolution the
voxel size is bigger than the feature itself — the added geometry is
mathematically there but gets rounded away before it reaches the output
mesh. Measured directly on a synthetic full-body test: at the default
resolution, the output mesh came out byte-for-byte identical with and
without the face additions. A first guess at how fine a grid this needs
(several voxels across the smallest feature's radius) was also checked
against real numbers and was wrong — a voxel modestly *larger* than the
radius still rounded the nose away completely, while one roughly
*matching* the radius measured a bump within ~10% of the exact analytic
prediction. So the grid is sharpened automatically, only as far as needed
and only near the face's own scale (not globally, which would be far more
expensive for no benefit elsewhere in the mesh), capped to bound runtime
cost (~12s for a full-body figure at the cap, versus ~2s without face
detail). If even that capped resolution still wouldn't resolve the
feature — a face that's a very small fraction of the frame, e.g. a
distant full-body shot — it's skipped rather than silently shipped as
geometry too small for any viewer or slicer to ever see; a closer or
half-body photo lets it actually show up.

**Not attempted, and why:** eyeballs/eyebrows/lips as distinct shapes,
individual facial proportions beyond the stylized sizing above, and any
skin-surface detail (pores, wrinkles) — Face Landmarker gives real
*positions* for a fixed set of named points, not a full learned face
*shape* model, so going further than this would mean either hand-adding
many more generic primitives per point (quickly a lot of code for
diminishing realism) or bringing in a learned 3D face-shape/generative
model, which reopens the licensing problem described above.

## Real body shape, not generic capsules

Everything above still describes a figure built from *generic* proportion
formulas — real joint positions, but a formula-driven shape in between
them. Single-photo mode now also measures the photo's own silhouette
(`silhouette.py`, reused from multi-photo mode — works on one photo too)
at several points along each torso and limb segment, instead of a single
straight taper between two joints: an actually narrower waist, a visibly
wider chest, a flexed arm, come from the photo's own outline rather than
a generic human-average shape. Each segment becomes a short chain of
mini-capsules (default 6 per segment), each one's radius read from the
silhouette at that point and clamped to a sane range around the generic
estimate (see below for why that clamp matters), rather than one capsule
linearly interpolated end to end.

This needed a real structural fix to actually work, not just wiring up
the measurement: chaining many short capsules through the same
smooth-minimum blending the rest of this project uses adds a small
"fillet" bulge at every joint between them (correct behavior for blending
two genuinely separate parts, but these chain pieces already meet
exactly at a shared point with matching radius — no gap to smooth over).
Across 6 pieces per segment that compounded into visibly fatter,
blobbier limbs than any individual measurement called for — caught by
rendering a real photo's output and comparing it to the source, not
assumed correct from the math. Fixed by hard-unioning a chain's own
pieces (they already meet seamlessly) and reserving smooth-min for where
a chain joins a genuinely different part (the torso, another limb).

The safety clamp earned its keep against a real failure, not a
hypothetical one: `silhouette.py`'s background-removal assumes a roughly
uniform background color, and a real test photo with a complex one (sand
and sky, not a studio backdrop) got the person correctly separated from
the sky but merged with the sand near the legs. A naive clamp still let
that corruption through at 1.8x the generic estimate on *every single
sample* down the whole thigh — not an occasional outlier a generous bound
could absorb, but a search that never found a real edge at all. Fixed by
telling those two cases apart explicitly: if the silhouette search
exhausts itself without the mask ever ending, that's not a measurement,
and the point falls back to the plain generic radius instead of a
clamped-but-still-wrong one.

## Hair

Single-photo mode also runs MediaPipe's **Hair Segmenter** (same
Apache-2.0 model family, no new license) automatically, giving a real
per-pixel hair mask from the photo. A stylized hair volume — not
individual strands — is built by sampling that mask radially around the
head: in each of 24 directions around the head center, if the photo's
hair extends meaningfully past the head's own surface there, a small
sphere is added reaching out to roughly that point; directions with no
hair (or where the segmenter found nothing worth trusting) get nothing.
A short, close-cropped cut produces a thin halo close to the head; a
fuller or side-swept style produces a bigger, more asymmetric one —
verified directly: rendering just a head against a real detected
hairstyle produced a visibly asymmetric shape matching that photo's own
side part, not a generic symmetric cap.

Two failure modes surfaced from running this against real, imperfect
photos, not caught by the geometry alone:
- A direction pointing down toward the neck can find "hair" that's
  really the mask misreading dark clothing or a shadow, and even when
  real, hair sampled in that direction geometrically plunges into the
  already-dense neck/torso capsules and changes nothing — confirmed
  directly (the body's own field was far deeper there than the hair
  bump's), a wasted, correct-but-invisible sample. Downward-pointing
  directions are now skipped — anatomically hair doesn't grow pointing
  down through the neck anyway, so nothing legitimate is lost.
- Several directions on a real adversarial photo (a busy beach
  background, a dark wetsuit) had the mask stay "true" in that direction
  all the way out to the search limit without a real edge ever
  appearing — four separate directions landing on the *exact same*
  radius, which is what a large contiguous misdetection looks like, not
  four coincidentally identical real measurements. Same fix as the
  silhouette case above: a search that never finds its own edge is
  discarded rather than trusted as a giant hairstyle.

Net result: on a clean, well-lit photo with a clear hairstyle, this adds
a real, photo-matched hair volume. On a photo where the hair signal is
genuinely too unreliable (the adversarial case above), it now correctly
adds nothing rather than guessing wrong — the same standard the rest of
this project holds its MAD-margin checks to.

**Still not attempted, and why:** individual strands, flyaways, and any
other fine hair texture — a stylized volume is the ceiling for a
segmentation-mask-driven approach; strand-level detail would need a
generative hair model, which reopens the licensing problem described
above. Muscle definition beyond what the silhouette's own outline shows
(visible striation, not just overall limb width) has the same ceiling.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

No license gate, no registration needed — but pose detection (and, for
single photos, face detection) fetch small model files automatically on
first run: ~9MB for pose (`landmarks.py`) and ~4MB for the face landmarker
(`face_features.py`), both cached at `~/.cache/anatomy3d-print/`. These
are MediaPipe's own official models, Apache-2.0 licensed like the rest of
MediaPipe, from Google's model bucket, not something you need to register
for or accept a license for — unlike SMPL-X, downloading them is just
normal setup, not a legal step.

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
- **Capsule count** (`procedural_body.py`) — how many of the 11 possible
  body-part capsules had enough keypoints to place (the torso counts as
  two — chest and hips either side of the waist — since the section-by-
  section split, not one). Calibration surfaced two real structural bugs
  here, not just a number to retune: (1) the mesh's height-normalization
  scale was computed *before* discarding debris fragments (see the next
  section), so a debris sliver sitting beyond the real body's top/bottom
  silently shrank the final printed height by ~12% in one measured case;
  (2) missing hips, or legs that don't reach the ankles, meant the mesh's
  lowest point was the pelvis or a knee instead of a foot — scaling that
  truncated span to fill the full target height inflated the figure by
  2-4x in measured cases, and raw capsule count didn't catch it (5+
  capsules could still be present). Both found their own hard gates (hips
  and full leg chains now required, with a clear error instead of a
  silently wrong mesh) rather than a margin tweak. With those gates in
  place, chest+hips+both legs always forms, so the true floor is 6
  capsules — confirmed directly (dropping every optional joint except the
  gated ones reaches exactly 6, never fewer), not a guess.
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

`webapp/app.py` is a FastAPI front end: upload photos, and it generates
the model server-side, then shows the *actual result* in an interactive
3D preview (three.js, the model sitting on a ground plane with orbit
controls) before you download anything — not just a bare file handoff.
The pipeline's own messages (wall-thickness checks, face-detail warnings)
show up as readable cards in the page instead of only ever reaching a
server log. Run it locally with:

```bash
uvicorn app:app --app-dir webapp --reload
```

Then open `http://localhost:8000`. `webapp/static/vendor/three/` vendors
three.js directly (MIT licensed, fetched via npm and version-pinned)
rather than loading it from a CDN at runtime, so the deployment has no
external JS dependency to go down.

`/api/generate` (used by the preview UI) returns JSON with a preview URL
and mesh stats; `/fit` is still there as a direct-download endpoint for
scripting/curl use, same as before.

The included `Dockerfile` and `railway.toml` deploy as-is on Railway —
no license-gated files to host or inject, since the default pipeline
doesn't need any.

## Limitations (read before printing)

**Single-photo (capsule) mode:**
- **Stylized, not realistic.** Tapered limbs, a nose/chin/eye-socket
  overlay when a face is detected (see "Face detail" above) but no
  eyebrows/lips/individual likeness, no hands/fingers, no clothing. It
  will not look like a scanned likeness of the person. That's the direct
  trade-off for not depending on a licensed body model.
- **Face detail needs the face to be a decent fraction of the frame.** A
  distant full-body photo often won't have enough resolution budget to
  render it (the pipeline detects this and skips gracefully rather than
  silently doing nothing) — a closer or half-body photo works better.
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

## Optional: TRELLIS.2 mode (external model, MIT-licensed)

`trellis_source.py` / `pipeline.py`'s `run_trellis_pipeline` hand the photo
to Microsoft's TRELLIS.2 instead of building the mesh with this project's
own code — confirmed directly on real test photos to give noticeably better
results on human subjects (one clean, connected, correctly-colored mesh) than
the from-scratch pipeline. It's exposed in the webapp as the "TRELLIS.2"
reconstruction method, single photo only.

Unlike SMPL-X above, TRELLIS.2's own license (MIT) is commercially fine —
but this integration calls Microsoft's free, public Hugging Face Space, not
something this project hosts or controls, so it isn't a real commercial
deployment path as-is (see `trellis_source.py`'s own docstring for what that
would actually take: self-hosting the MIT-licensed weights, or a paid API
with a real SLA). Treat it as a development/testing mode and an optional
higher-quality choice for end users, not a production guarantee.

Confirmed directly, not assumed: the SAME real test on an animal photo came
back recognizable but fragmented into disconnected pieces, not a printable
solid — `run_trellis_pipeline` refuses animal subjects for this reason
(using the existing `animal_frac` signal) rather than silently shipping a
known-broken result.

1. `pip install -r requirements-trellis.txt`
2. Set an `HF_TOKEN` environment variable to a free Hugging Face account's
   own read-only access token (https://huggingface.co/settings/tokens) —
   anonymous calls are heavily rate-limited (confirmed: exhausted within a
   couple of calls).

## Project layout

```
src/anatomy3d/
  preprocess.py       EXIF fix, resize, contrast + sharpen
  landmarks.py         2D pose keypoint detection
  face_features.py      single-photo: real face landmarks (nose/chin/eyes)
  hair_features.py       single-photo: real hair mask -> stylized hair volume
  procedural_body.py    single-photo: capsule/SDF body builder
  silhouette.py          multi-photo: background removal -> mask
  visual_hull.py           multi-photo: voxel carving -> mesh
  body_fit.py                optional: SMPL-X fitting (non-commercial license)
  trellis_source.py            optional: TRELLIS.2 external reconstruction (MIT license)
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
requirements-trellis.txt  optional extras for the TRELLIS.2 external mode
```
