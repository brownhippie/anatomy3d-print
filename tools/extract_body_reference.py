"""Exports the reference body used by src/anatomy3d/body_reference.py to
src/anatomy3d/data/body_reference.npz: an adult male and an adult female
(average build) from the Anny body model, in its rest A-pose, with the
skeleton, skinning weights and COCO keypoint weights needed to pose and fit
it without torch.

Needs `pip install anny` (NAVER, Apache-2.0; brings torch) -- only to run
this tool, not the app. The body mesh and rig are MakeHuman/MPFB2 assets
(CC0 1.0); the keypoint weights are part of Anny (Apache-2.0).

usage: python tools/extract_body_reference.py
"""
from pathlib import Path

import numpy as np
import torch

import anny

OUT = Path(__file__).resolve().parents[1] / "src" / "anatomy3d" / "data" / "body_reference.npz"
AGE = 0.7  # between Anny's "young" (0.667) and "old" (1.0) anchors: an adult
PHENOTYPES = ["weight", "muscle", "height", "proportions"]
PARTS = ["head", "neck", "torso", "upperarm.L", "upperarm.R", "forearm.L", "forearm.R", "hand.L", "hand.R", "legs"]


def part_of_bone(label):
    if label == "head" or label.startswith("eye"):
        return 0
    if label.startswith("neck"):
        return 1
    if label.startswith("upperarm"):
        return 3 if label.endswith(".L") else 4
    if label.startswith("lowerarm"):
        return 5 if label.endswith(".L") else 6
    if label.startswith(("wrist", "finger", "metacarpal")):
        return 7 if label.endswith(".L") else 8
    if label.startswith(("upperleg", "lowerleg", "foot", "toe")):
        return 9
    return 2


def main():
    m = anny.Anny(skinning_method="lbs")
    labels = list(m.bone_labels)
    parents = np.array([-1 if p is None else int(p) for p in m.bone_parents], dtype=np.int16)
    assert all(p < j for j, p in enumerate(parents)), "bones must be listed parents first"
    kr = anny.KeypointsRegressor.coco(m)
    weights = kr.regression_weights.cpu().numpy()
    keep = max(int((weights > 1e-6).sum(1).max()), 1)
    kp_idx = np.argsort(-weights, axis=1)[:, :keep]
    kp_w = np.take_along_axis(weights, kp_idx, axis=1)
    kp_w /= kp_w.sum(1, keepdims=True)
    data = {
        "bone_labels": np.array(labels),
        "parents": parents,
        "bone_part": np.array([part_of_bone(l) for l in labels], dtype=np.int8),
        "part_names": np.array(PARTS),
        "faces": m.get_triangular_faces().cpu().numpy().astype(np.uint16),
        "skin_idx": m.vertex_bone_indices.cpu().numpy().astype(np.uint8),
        "skin_w": m.vertex_bone_weights.cpu().numpy().astype(np.float32),
        "kp_names": np.array(kr.labels),
        "kp_idx": kp_idx.astype(np.uint16),
        "kp_w": kp_w.astype(np.float32),
    }
    def rest(gender, **kw):
        with torch.no_grad():
            out = m(phenotype_kwargs={"gender": gender, "age": AGE, **kw})
        # rest_vertices: the same frame as the bones (the posed output is root-relative)
        return out["rest_vertices"][0].cpu().numpy(), out["rest_bone_poses"][0, :, :3, 3].cpu().numpy()

    data["phenotypes"] = np.array(PHENOTYPES)
    for name, gender in (("male", 0.0), ("female", 1.0)):
        v, j = rest(gender)
        data[f"{name}_vertices"] = v.astype(np.float32)
        data[f"{name}_joints"] = j.astype(np.float32)
        # build (weight, muscle, height, proportions) as linear changes per
        # unit around the average (0.5), measured over 0.3..0.7
        dv, dj = [], []
        for ph in PHENOTYPES:
            v_hi, j_hi = rest(gender, **{ph: 0.7})
            v_lo, j_lo = rest(gender, **{ph: 0.3})
            dv.append((v_hi - v_lo) / 0.4)
            dj.append((j_hi - j_lo) / 0.4)
        data[f"{name}_dvertices"] = np.stack(dv).astype(np.float16)
        data[f"{name}_djoints"] = np.stack(dj).astype(np.float32)
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), {len(labels)} bones, "
          f"{len(data['male_vertices'])} vertices, keypoints {list(kr.labels)}")


if __name__ == "__main__":
    main()
