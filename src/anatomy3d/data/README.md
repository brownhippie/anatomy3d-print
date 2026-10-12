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
