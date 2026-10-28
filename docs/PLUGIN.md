# The general plugin: the frame at any border between two systems

Written 2026-10-26 at the owner's request: one description of every parameter of 𝔐 = 𝔐_sys(R_t − R), so that the same reading can be put at any border (a PC heating, the Bhippie camera and its Pi, a battery, anything), checked for what it holds there, and left running. Nothing here is new to the framework; it is the owner's words from the corpus and from October, with the places to fill in. Code: `mad_plugin.py` (reads only; acts on nothing).

Updated 2026-10-28 at the owner's request: "it should also mention each number and figures point to its own system, domain or medium". Every number and figure below says what it points to: §1b names the places a number can point to, the arrows (→) in §3 tag each figure on the three borders, §3b tags the plugin's own numbers, and the plugin writes the same tag beside every number it reads (`--sys`, §4). No reading changed: the same file and settings give the same numbers as on 26 Oct, on the made-up test file and on 6,000 camera frames from 28 Oct.

## 0. The line, and the three things to find
𝔐 = 𝔐_sys(R_t − R). For any quantity: **R_t**, the amount the change needs (the level the system's own first spins show; or a limit it must not pass, from its own settings or a datasheet, as a ceiling); **R**, the value this change took; **𝔐_sys**, only if the size matters and not just the sign; the plugin uses 1/R_t so 𝔐 is a fraction of the need. The gate is the sign of (R_t − R). For a floor, the minimum the change needs, the sign flips. Something goes from positive to negative because it has no other way; it is read by the path and the two points at both ends; nothing else is there.

R and R_t always point to the same system, in the same medium and unit: a need is only the need of the system it was read from.

## 1. The border: which side is which
Every border has an **input side** (what is given: the supply, the push per unit) and a **machine side** (what is taken or done: the draw, the work). One thing is held on each side: on the motor, the supply holds the voltage and the motor holds its speed. The held thing on one side is what moves on the other. The floor of the machine's draw is the ceiling of the input's push per unit when the top is held, so the two sides are one gate read from two ends; the plugin checks that (the mirror) when both sides are given. What sits between the two readings is **the extra**: in health, the input's own motion; after wear, more than that. Nothing goes back from the machine to the input; the extra is diluted at the machine's end, and where it goes (heat, vibration, light, errors, dropped frames) is a channel you log beside, if you have one.

## 1b. Every number points to its own system, domain or medium
A number is read for one system only, and it says which. Every number and figure in the plugin points to one of these:

| Points to | What it is | Its numbers |
|---|---|---|
| Side A's system | the quantity read as R on side A (`--a`) | its level, floor, ceiling and widths; its margins, gates, nudge and kick per change; its past and short shares, drift, depth and loop per window |
| Side B's system | the quantity read as R on side B (`--b`) | the same set, for B |
| The border between A and B | what sits between the two readings | the extra, its spread, the mirror slope |
| The rhythm's system | the system whose own cycle is one change | every count of changes: first spins (500), warm-up (300), local level (50), window (200), loop length, "from change N" |
| The dilution channel | the machine's end, in the medium where the excess shows: heat (thermal), vibration (mechanical), light (optical), errors and dropped frames (compute) | the noise ratio and the channel's first-spins level |
| The held thing | what one side keeps constant; the label names its setting | the label; one held state has one set of first spins |
| The steadiness source | the within-change spread (of side A unless a spread column is given) | the steady band (95th percentile), the steady flag, the unsteady share |
| The instrument | the logger: its sample clock and the cut settings | sample rate, the rhythm in Hz, cut band, crossing level, sample positions; never a reading of the system |

Each named column gets three things: **whose system** it is, its **domain or medium** (electrical, mechanical, thermal, optical, compute), and its **unit**. The unit belongs to the medium, its zero included. 𝔐 is a fraction of the need, so its size is the system's own only when the scale starts at the medium's own zero: kelvin for heat. °C is fine only where a difference is taken, as in the warm-up rule. This point was raised in the 28 Oct sense check (`PLAN_bhippiee_plugin_everywhere.md` §4b, whose table waits on the owner).

A number from another system is not R of this border, even when it sits in the same file. On record from the camera run (28 Oct, `PLAN_bhippie_camera.md`, "Errors on record"):
- frame-time jitter on the sensor's own frame clock points to the camera, not to the Pi's machine end, where its excess shows as heat and dropped frames;
- the watched object's vibration frequency points to the object: another border, with its own change;
- one reference held across every frame-rate setting read the setting, not the system: each setting is its own held state.

## 2. The change: the system's own rhythm, never seconds
One change = one cycle of the system's own rhythm: a shaft turn, a camera frame, a work cycle of the processor, a charge pulse, a breath. It comes from the system, not from a clock. The plugin takes either a file that is already one row per change, or a raw periodic signal and cuts the changes from it (rising crossings with hysteresis, as the piezo and the commutation pulses were cut). Nothing is read per second; if a rate per second is wanted later, it is 1/f of the rhythm, outside the reading.

The change points to the rhythm's system, and every count of changes is counted in it: shaft turns on the motor, job cycles on the PC, camera frames on the camera + Pi. A sample rate in Hz points to the logger's own clock (the instrument) and is used only to find the rhythm.

## 3. The parameters, with the motor beside two other borders
Each figure carries an arrow: → whose system, domain or medium, unit.

| Parameter | In the frame (owner's words) | On the DC motor (done) | PC heating (to fill in) | Bhippie camera + Pi (to fill in) |
|---|---|---|---|---|
| The change | one cycle of the system's own rhythm | one shaft turn → the shaft, mechanical; cut from six commutation pulses → the commutator, seen in the current, electrical; or from one piezo rotation → the motor body, mechanical | one work cycle: one frame rendered, one request served, one fixed job repeated → the job, compute; from the job's own counter, not the clock | one frame from the camera → the camera's own frame clock (the sensor), which sets the Pi's work per frame; the frame counter is the rhythm |
| Input side, R | what is given per change | current taken per turn → supply into the motor, electrical, A; push per ampere V÷I → the supply side, electrical, V/A; power V×I → electrical, W | power drawn per cycle → the PSU or the board's own power counter, electrical, W; or voltage × current of the rail per cycle → the rail, electrical, W | the Pi rail's current per frame → the Pi's supply, electrical, A; or SoC power per frame if the board reports it → electrical, W |
| Machine side, R | what is taken or done per change | speed per turn → the shaft, mechanical, counted as the turn's length on the logger's clock; torque per turn where measured → the shaft, mechanical, N·m | work done per cycle (instructions, frames, bytes) → the processor, compute, counts; time the cycle took in its own counts → the processor's own counter, compute | the frame's work: bytes written → the Pi, compute, bytes; processing time per frame in the pipeline's own counts → the Pi's processor, compute, ns of its own counter. Exposure → the camera, optical: a setting of the held scene, not the Pi's work |
| Held thing, each side | what that side keeps constant | supply holds V → the supply, electrical, V; motor holds speed → the shaft, mechanical | the PSU holds the rail voltage → electrical, V; the job holds its size → compute | the rail holds its voltage → the Pi's supply, electrical, V; the scene/settings hold the frame's size → the camera and the room, optical (light, exposure, gain); each setting is its own held state |
| The need, R_t (level) | what a healthy change takes; from the system's own first spins, held | mean of the first 500 steady turns after warm-up → the quantity's own system and unit; 500 and the warm-up → shaft turns | mean over the first 500 steady cycles of the fixed job after the box has warmed → the quantity's own; 500 → job cycles | mean over the first 500 steady frames at fixed scene and settings after warm-up → the quantity's own; 500 → camera frames |
| Floor and ceiling | the minimum the change needs; the most it can take | 1st and 99th percentile of the first spins → each quantity's own first spins, its own unit | same; a datasheet limit may be a second ceiling beside it → the part it is written for (max rail current → the rail, electrical, A; max temperature → the chip, thermal, K) | same; the board's throttling temperature is a ceiling from its own settings → the SoC, thermal, K |
| Steady change | a change whose inside spread is within the first spins' band | spread of the six pulse lengths → the commutator, seen in the current, electrical; a fraction of the turn | spread of the cycle's own sub-steps if logged → the job, compute; else the cycle length's spread → the processor's own counter | spread of the frame's sub-timings if logged → the Pi's pipeline, compute; else frame length spread → the camera's frame clock, not the Pi |
| Residual | what is still there from the last change | the carried state of the rotor → the rotor, mechanical, read through the current, electrical | heat → the chip, thermal; cache → the processor, compute; both carried into the next cycle | buffers → the Pi, compute; heat → the SoC, thermal; both carried into the next frame |
| Kick | what came in this change (the slope changing) | current this turn minus last turn → side A, electrical, A | power this cycle minus last → side A, electrical, W | rail current this frame minus last → side A, electrical, A |
| Total, nudge | residual + kick = R; nudge = R − level, its sign the event | per turn → the quantity's own, % of its own level | per cycle → the quantity's own | per frame → the quantity's own |
| 𝔐, the margin | 𝔐_sys(R_t − R); the thing that drifts; drift = its median over a window; loop = changes between crossings of the level | per turn → the quantity's own system, a fraction of its need, no unit | per cycle → the same | per frame → the same |
| Return | the stabilisation after a sudden push, the coming back to the level | the loop length → shaft turns; unchanged by wear; what does not come back is the drift | the cycles it takes to come back after a burst → job cycles | the frames it takes to come back after a burst → camera frames |
| Noise / dilution channel | where the excess shows at the machine's end | piezo or vibration RMS per turn → the motor body, mechanical, the sensor's counts; the bounce at the hand-over (energy back ÷ energy in) → the border, electrical, no unit | temperature per cycle if logged per cycle → the chip, thermal, K; fan speed → the cooler, mechanical, its own count; error counts → compute; throttling events → the board's power and heat management, counts | SoC temperature per frame if logged → the Pi's chip, thermal, K; dropped frames → the Pi's end, compute, counts; throttle flags → the Pi's board, power and heat; frame-time jitter on the sensor clock → the camera, so not the Pi's channel (withdrawn as the Pi's, 28 Oct) |
| The extra | what sits between the two sides' readings | the voltage's own motion in health → the supply, electrical, % of its level | the rail's own motion → the PSU, electrical | the rail's own motion → the Pi's supply, electrical |
| Reset | only when a part is physically replaced or repaired; then new first spins | → the part replaced | → the part replaced | → the part replaced |

## 3b. The plugin's own numbers, and what each points to
Fixed settings (`P` in `mad_plugin.py`):

| Number | Value | Points to |
|---|---|---|
| First spins | 500 changes | the rhythm's system: counted in its own changes |
| Warm-up | 300 changes; or, with a temperature column, where it settles within 1.0 of its level | the changes → the rhythm's system; the 1.0 → the temperature channel's unit (a difference, so °C or K alike) |
| Steady band | 95th percentile of the within-change spread | the steadiness source (side A unless `--spread` is given) |
| Ceiling, floor | 99th, 1st percentile | each quantity's own first spins, its own system and unit |
| Local level | mean of the previous 50 changes (at least 10), with the held width | the quantity's own system; the 50 and 10 → changes of the rhythm |
| Window | 200 changes | the rhythm's system |
| Rhythm search (raw signal only) | 0.5 Hz up to 0.45 × the sample rate | the instrument: the logger's own clock |
| Cut (raw signal only) | band 0.6–1.6 × the rhythm; crossings at 0.3 SD; changes kept within 30 % of the median length, at least 4 samples | the instrument, applied to the cut signal's own medium |

What it writes:

| File | Number | Points to |
|---|---|---|
| `_reference.json` | `rhythm_hz` | the rhythm's system, counted on the logger's clock (instrument) |
| | `first_spins_from_change`, `loop_len`, `n_steady` | changes of the rhythm |
| | `noise0` | the dilution channel |
| | `L`, `ceil`, `floor`, `width_up`, `width_down` of a quantity | that quantity's system and unit |
| | the same under `kick_<quantity>` | that quantity: its change from the last change |
| | `band` | the steadiness source |
| | `points_to` | the names given at first spins, held with them |
| `_changes.csv` | `_M_ceil`, `_M_floor`, `_M_L`, their `_local` forms, `_gate`, `_gate_local`, `_nudge_local_pct`, `_kick`, `_kick_M_ceil` | the quantity in the column's name |
| | `steady` | the steadiness source |
| | `noise_ratio` | the dilution channel |
| | `extra_pct` | the border between A and B |
| | `start`, `length` (raw signal only) | the instrument's samples |
| `_windows.csv` | `first_change`, `changes`, `_loop_len` | changes of the rhythm |
| | `_past_pct`, `_short_pct`, their `_local` forms, `_drift_pct`, `_depth_pct`, `_kick_past_pct` | the quantity in the column's name |
| | `unsteady_pct` | the steadiness source |
| | `noise_ratio`, `temp` | the dilution channel; the temperature's own medium |
| | `mirror_slope`, `extra_sd_pct` | the border between A and B |

## 4. What the plugin reads, per change and per window
Per change (file `<name>_changes.csv`): for every quantity, 𝔐 about the ceiling, the floor and the level (held, from the first spins) and about the local level (previous 50 changes with the held width); the gate sign (in / past / short) held and local; the kick and its margin against the kick ceiling; the steady flag; the noise ratio; the mirror of the two sides per change when both are given (nudge of A and nudge of B, their sum = the extra).
Per window of W changes (file `<name>_windows.csv`): shares past the ceiling and below the floor (held and local); the drift of 𝔐 (median about the level); loop length and depth; share of kicks past their ceiling; unsteady share; noise ratio; temperature if given; the mirror's slope and the extra's spread over the window.

What each number points to, written by the plugin:
- Name every column it reads with `--sys COLUMN="whose system, domain or medium, unit"`, and the change itself with `--sys change="..."`. On the motor, for example: `--sys change="one shaft turn, cut from the piezo on the motor body (mechanical)" --sys current="supply into the motor (input side), electrical, A" --sys vib="the motor body (machine's end), mechanical, piezo counts"`.
- `<name>_points_to.json`: every number in the three files above, with what it points to and what it is.
- Every printed line carries the tag beside its numbers in braces, e.g. `current {supply into the motor (input side), electrical, A} past 1.5 %`.
- `<name>_reference.json` keeps the names given at first spins (`points_to`), held with them.
- A column left unnamed is read the same; its numbers are marked `{not named}`, and the plugin says once which columns are not named.

## 5. Checks that come with it, in every use
- A plan page before the first real file: which quantity is R on each side, what the change is, where the first spins start, and for each line its alternative (heat, supply sag, the job changing, the scene changing).
- Points-to: before the first real file, every column the plugin reads is named to its own system, domain or medium, and unit (§1b). A number that points to another system is not R of this border; it can be logged beside, under its own name. If a later pass names a column differently from the held first spins, the plugin says so: a held reference applies to its own system only.
- The mirror: if both sides are given, their nudges must come out opposite with slope near one and the extra small and memoryless in health. If they do not, the two sides are not one border, or the held thing is not held.
- The no-cost rule: the reading must cost the system nothing (on the motor: shunt drop under 1 % → of the supply's voltage, electrical; speed unchanged → the shaft). On a PC or a Pi, the logging itself is a load; log at the rhythm's rate and no faster, and compare the job's own counts with and without the logger.
- One-way drift: the held reference is never re-learned. A reading coming back means a repair or a replacement happened; then new first spins.
- Untested is not wrong; a null is a bound at the channel's resolution, not an absence. Say the bound, and the system and medium the channel's resolution belongs to.

## 6. Limits
The plugin reads; it never acts. It needs a rhythm it can find and a per-change value on at least one side. Heat is readable only if a temperature is logged per change; one reading per minute is a per-window number, not a per-change one, and it climbs with running time, so it cannot be separated from running time without matching. Where the dilution channel is not a sensor you own, the claim about dilution stays a claim. The plugin cannot tell by itself whether a name is right: `--sys` records what the person running it says each column points to, and the sense of that is checked in the plan page.
