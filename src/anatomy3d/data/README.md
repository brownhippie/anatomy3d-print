# Data files

`torso_profile.npz` — human torso cross-section shape (front and back
surfaces across the torso's width, at heights from the shoulder joints to
the hip joints), measured from the MakeHuman base mesh. Used by
`silhouette_relief.py` so outline-based models get a human back (flatter
upper back, spine groove, seat) instead of a mirror of their front.

Source: `anny/data/mpfb2/3dobjs/base.obj` in the `anny` 0.6.1 wheel on PyPI
(NAVER LABS Europe), which bundles the MakeHuman/MPFB2 base mesh. That data
is released under CC0 1.0 (its LICENSE.md in the same wheel); MakeHuman's
own FAQ confirms the base mesh and targets have been CC0 since 2020.
The profile is of an average young adult man: the base mesh plus the
MakeHuman macro targets `{african,asian,caucasian}-male-young` at 1/3 each
(same wheel, same CC0 licence; put them in a `targets/` folder next to
base.obj). Regenerate with
`python tools/extract_torso_profile.py base.obj src/anatomy3d/data/torso_profile.npz`.

`body_reference.npz` — an adult man and an adult woman of average build in
the rest A-pose: 13,718-vertex body mesh, 104-bone skeleton, skinning
weights, COCO keypoint weights, and linear changes for weight, muscle,
height and proportions. The man carries MakeHuman's own flatter-chest
targets (`torso-muscle-pectoral-decr`, `breast-point-decr`); each ear's
vertices are weighted 0..1 from MakeHuman's `l/r-ear-trans-up` targets so the
builder can size and turn the ears. `body_reference.py` poses it onto a
photo's pose and face points so outline-based models get a human body's depth
(head over the torso, back of the skull and nape, the back's curve, forearms
resting on the belly, ears standing off the head) as their base shape.

Source: the Anny body model, `anny` 0.6.1 on PyPI (NAVER LABS Europe).
Anny's code and its COCO keypoint weights are Apache-2.0
(Copyright (C) 2025 NAVER Corp.); the body mesh and rig it is built on are
MakeHuman/MPFB2 assets released under CC0 1.0 (`anny/data/mpfb2/LICENSE.md`
in the wheel), the chest and ear targets included. Only the default "anny" topology is used -- not Anny's
optional SMPL-X topology, which is non-commercial. Regenerate with
`pip install anny` and `python tools/extract_body_reference.py`.
