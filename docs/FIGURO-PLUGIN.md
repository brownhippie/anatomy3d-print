# Figuro through the plugin (the plan page)

The plugin is `src/anatomy3d/mad_plugin.py`, vendored unchanged; its description is `docs/PLUGIN.md`. It only reads. `src/anatomy3d/ledger.py` feeds it and acts on its readings in two places.

## The border

| Plugin | In Figuro |
|---|---|
| The change | One render: an app Generate, or one Lab build. Counted by the server's render counter (`counter.txt` on the volume), never by the clock. |
| Label (phase) | The machine that built it: the commit, plus the tuned values in effect (`ledger.machine()`). Either changing is a new window. |
| Input side, R | What the photo gave for each quantity (`photo_side` in each row). |
| Machine side, R | What the model made, the same quantity measured on it (`R`). Where a quantity has a photo side, R is the model's value over the photo's: 1 means as the photo. |
| Held thing | Input side: the photo. Machine side: the settings and the machine. The held job holds both: the held photo, and its first 160 settings. |
| R_t (level, floor, ceiling) | From the first spins: the first window of the held job, held and never re-learned. A check whose code changes gets a new version, and that version starts its own first spins. |
| Steady flag | Needs the spread inside a render. None is logged yet, so every render counts as steady. |
| Kick | A held setting's reading on this machine minus the same setting's on the machine before. |
| Noise channel | Spikes, failed reality checks, face not found on the model, refused or failed builds. |
| Mirror | Needs both sides moving. In the Lab the photo is held, so the input side never moves. App renders of different photos give it, once there are enough. Untested until then. |
| Reset | Only a check's own version (above) or a new held photo. |
| Your verdict | A rating (1–5 stars, a note) on an app render or a Lab build: `ledger.rate`. |

## Every quantity in its own medium

The resolution is how finely the measure itself can tell two values apart: its raster, its rounding, or the printer's layer. A move smaller than that is no move. As `docs/PLUGIN.md` says, a null is a bound at the channel's resolution.

| Quantity | Kind | Unit | Resolution | Version |
|---|---|---|---|---|
| `outline` | high | % of the photo's cutout area covered by the model's front view (420-row raster) | 0.25 | 1 |
| `aspect` | ratio | model height:width over the photo's (ratio of two length ratios) | 0.005 | 1 |
| `w_head` | ratio | model width over the photo's, at the eyes' row (photo px over photo px) | 0.005 | 1 |
| `w_shoulder` | ratio | model width over the photo's, at the shoulder joints' row (photo px over photo px) | 0.005 | 1 |
| `w_chest` | ratio | model width over the photo's, a quarter of the way from shoulders to hips (photo px over photo px) | 0.005 | 1 |
| `w_waist` | ratio | model width over the photo's, 70% of the way from shoulders to hips (photo px over photo px) | 0.005 | 1 |
| `face_shape` | low | % of the eye distance: mean miss of the 468 face points found on the model's front render, after aligning | 0.1 | 1 |
| `eye_span` | ratio | model's eye-corner span ÷ face width, over the photo's | 0.005 | 1 |
| `nose_len` | ratio | model's bridge-to-base ÷ face height, over the photo's | 0.005 | 1 |
| `mouth_w` | ratio | model's mouth width ÷ face width, over the photo's | 0.005 | 1 |
| `jaw_w` | ratio | model's jaw-angle width ÷ face width, over the photo's | 0.005 | 1 |
| `face_hw` | ratio | model's forehead-chin ÷ cheek-cheek, over the photo's | 0.005 | 1 |
| `face_id` | high | cosine between face embeddings (SFace) of the model's render and the photo, −1 to 1 | 0.001 | 1 |
| `face_depth` | ratio | model's face depth (mean mm the brow, eyes, cheeks, lips, chin sit behind the nose tip) over the photo's, both in mm on the figure | 0.01 | 2 |
| `hair` | high | share of the front surface area (mm²) over the photo's hair sculpted as hair (folds over 30°), 0 to 1 | 0.005 | 2 |
| `d_head` | ratio | head length ÷ breadth, over the human reference's (ANSUR II men: 1.29) | 0.01 | 1 |
| `d_chest` | ratio | chest depth ÷ shoulder-joint distance, over the human reference's (ANSUR II men: 0.61) | 0.01 | 1 |
| `d_waist` | ratio | waist depth ÷ shoulder-joint distance, over the human reference's (ANSUR II men: 0.57) | 0.01 | 1 |
| `sealed` | high | 1 when the mesh is closed (watertight), 0 when not | 0.5 | 1 |
| `wall` | high | mm, thinnest wall (2nd percentile) on the print as made | 0.05 | 1 |
| `smooth_ok` | low | % of the 40,000-face preview's fold edges that are outward and sharper than 60° (spikes) | 0.01 | 2 |
| `garment_edges` | high | % of the photo's garment-over-garment edges standing as a step on the model | 0.1 | 1 |
| `clothes` | high | % of the photo's sure over/under boundaries the model has the right way round | 0.1 | 1 |

Kinds:
- **ratio:** 1 is as the photo (or the human reference); off either way is further.
- **high:** higher is closer.
- **low:** lower is closer.

## What acts on the readings

- **The held job, on every new machine.** At startup, and whenever the tuned values change, the Lab builds the held job first. Then the run that was going continues.
- **The push guard** (`ledger.guard`). For each builder family, it compares each held setting with the same setting on the machine before. The machine fails when the net share of settings further from the photo exceeds 10.0% on any quantity read on at least 6 settings. It also fails when a held build is newly unsealed, or newly below the printer's 0.8 mm wall. The plugin's reading against the first spins (drift, past and short shares) is kept beside it.
- **The production pick** (`ledger.choose`). It changes only after a passed guard, or after your rating. It moves to:
  - a setting you rated higher than the current one; or
  - a setting at least as close to the photo on every quantity and closer on one (no averaging).
  It never moves to a setting you rated 2 or less.
- **Gate before handing over.** If a render built with the pick is unsealed or below 0.8 mm, it is built again with the app's own setting.
- **Compare.** Builds the same photo with the setting closest to the photo on the most quantities, net, among those not rated yet. You rate both.
- **The Study's autopilot** applies tuned values only when no study photo is further from itself on any quantity.
- **No-cost rule.** An app render is measured by the Lab's worker after the model is handed over. The wall is read from the print file's own export, not measured again.
- **`tools/push_guard.py`** runs the same guard on a working tree before a push.
