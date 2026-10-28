# The general plugin: the frame at any border between two systems

Written 2026-10-26 at the owner's request: one description of every parameter of 𝔐 = 𝔐_sys(R_t − R), so that the same reading can be put at any border (a PC heating, the Bhippie camera and its Pi, a battery, anything), checked for what it holds there, and left running. Nothing here is new to the framework; it is the owner's words from the corpus and from October, with the places to fill in. Code: `mad_plugin.py` (reads only; acts on nothing).

## 0. The line, and the three things to find
𝔐 = 𝔐_sys(R_t − R). For any quantity: **R_t**, the amount the change needs (the level the system's own first spins show; or a limit it must not pass, from its own settings or a datasheet, as a ceiling); **R**, the value this change took; **𝔐_sys**, only if the size matters and not just the sign; the plugin uses 1/R_t so 𝔐 is a fraction of the need. The gate is the sign of (R_t − R). For a floor, the minimum the change needs, the sign flips. Something goes from positive to negative because it has no other way; it is read by the path and the two points at both ends; nothing else is there.

## 1. The border: which side is which
Every border has an **input side** (what is given: the supply, the push per unit) and a **machine side** (what is taken or done: the draw, the work). One thing is held on each side: on the motor, the supply holds the voltage and the motor holds its speed. The held thing on one side is what moves on the other. The floor of the machine's draw is the ceiling of the input's push per unit when the top is held, so the two sides are one gate read from two ends; the plugin checks that (the mirror) when both sides are given. What sits between the two readings is **the extra**: in health, the input's own motion; after wear, more than that. Nothing goes back from the machine to the input; the extra is diluted at the machine's end, and where it goes (heat, vibration, light, errors, dropped frames) is a channel you log beside, if you have one.

## 2. The change: the system's own rhythm, never seconds
One change = one cycle of the system's own rhythm: a shaft turn, a camera frame, a work cycle of the processor, a charge pulse, a breath. It comes from the system, not from a clock. The plugin takes either a file that is already one row per change, or a raw periodic signal and cuts the changes from it (rising crossings with hysteresis, as the piezo and the commutation pulses were cut). Nothing is read per second; if a rate per second is wanted later, it is 1/f of the rhythm, outside the reading.

## 3. The parameters, with the motor beside two other borders

| Parameter | In the frame (owner's words) | On the DC motor (done) | PC heating (to fill in) | Bhippie camera + Pi (to fill in) |
|---|---|---|---|---|
| The change | one cycle of the system's own rhythm | one shaft turn (six commutation pulses; or one piezo rotation) | one work cycle: one frame rendered, one request served, one fixed job repeated; from the job's own counter, not the clock | one frame from the camera; the frame counter is the rhythm |
| Input side, R | what is given per change | current taken per turn (A); push per ampere V÷I; power V×I | power drawn per cycle (W from the PSU or the board's own power counter); or voltage × current of the rail per cycle | the Pi rail's current per frame (A), or SoC power per frame if the board reports it |
| Machine side, R | what is taken or done per change | speed per turn; torque per turn where measured | work done per cycle (instructions, frames, bytes); time the cycle took in its own counts | the frame's work: bytes written, exposure, processing time per frame in the pipeline's own counts |
| Held thing, each side | what that side keeps constant | supply holds V; motor holds speed | the PSU holds the rail voltage; the job holds its size | the rail holds its voltage; the scene/settings hold the frame's size |
| The need, R_t (level) | what a healthy change takes; from the system's own first spins, held | mean of the first 500 steady turns after warm-up | mean over the first 500 steady cycles of the fixed job after the box has warmed | mean over the first 500 steady frames at fixed scene and settings after warm-up |
| Floor and ceiling | the minimum the change needs; the most it can take | 1st and 99th percentile of the first spins | same; a datasheet limit (max rail current, max temperature) may be a second ceiling beside it | same; the board's throttling temperature is a ceiling from its own settings |
| Steady change | a change whose inside spread is within the first spins' band | spread of the six pulse lengths | spread of the cycle's own sub-steps if logged; else the cycle length's spread | spread of the frame's sub-timings if logged; else frame length spread |
| Residual | what is still there from the last change | the carried state of the rotor | heat and cache carried into the next cycle | buffers and heat carried into the next frame |
| Kick | what came in this change (the slope changing) | current this turn minus last turn | power this cycle minus last | rail current this frame minus last |
| Total, nudge | residual + kick = R; nudge = R − level, its sign the event | per turn | per cycle | per frame |
| 𝔐, the margin | 𝔐_sys(R_t − R); the thing that drifts; drift = its median over a window; loop = changes between crossings of the level | per turn | per cycle | per frame |
| Return | the stabilisation after a sudden push, the coming back to the level | the loop length, unchanged by wear; what does not come back is the drift | the cycles it takes to come back after a burst | the frames it takes to come back after a burst |
| Noise / dilution channel | where the excess shows at the machine's end | piezo or vibration RMS per turn; the bounce at the hand-over (energy back ÷ energy in) | temperature per cycle if logged per cycle; fan speed; error counts; throttling events | SoC temperature per frame if logged; dropped frames; throttle flags; frame-time jitter |
| The extra | what sits between the two sides' readings | the voltage's own motion in health | the rail's own motion | the rail's own motion |
| Reset | only when a part is physically replaced or repaired; then new first spins | | | |

## 4. What the plugin reads, per change and per window
Per change (file `<name>_changes.csv`): for every quantity, 𝔐 about the ceiling, the floor and the level (held, from the first spins) and about the local level (previous 50 changes with the held width); the gate sign (in / past / short) held and local; the kick and its margin against the kick ceiling; the steady flag; the noise ratio; the mirror of the two sides per change when both are given (nudge of A and nudge of B, their sum = the extra).
Per window of W changes (file `<name>_windows.csv`): shares past the ceiling and below the floor (held and local); the drift of 𝔐 (median about the level); loop length and depth; share of kicks past their ceiling; unsteady share; noise ratio; temperature if given; the mirror's slope and the extra's spread over the window.

## 5. Checks that come with it, in every use
- A plan page before the first real file: which quantity is R on each side, what the change is, where the first spins start, and for each line its alternative (heat, supply sag, the job changing, the scene changing).
- The mirror: if both sides are given, their nudges must come out opposite with slope near one and the extra small and memoryless in health. If they do not, the two sides are not one border, or the held thing is not held.
- The no-cost rule: the reading must cost the system nothing (on the motor: shunt drop under 1 %, speed unchanged). On a PC or a Pi, the logging itself is a load; log at the rhythm's rate and no faster, and compare the job's own counts with and without the logger.
- One-way drift: the held reference is never re-learned. A reading coming back means a repair or a replacement happened; then new first spins.
- Untested is not wrong; a null is a bound at the channel's resolution, not an absence. Say the bound.

## 6. Limits
The plugin reads; it never acts. It needs a rhythm it can find and a per-change value on at least one side. Heat is readable only if a temperature is logged per change; one reading per minute is a per-window number, not a per-change one, and it climbs with running time, so it cannot be separated from running time without matching. Where the dilution channel is not a sensor you own, the claim about dilution stays a claim.
