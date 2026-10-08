from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BodyMesh:
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)
    colors: Optional[np.ndarray] = None  # (V, 3) uint8, per-vertex — see procedural_body._bake_photo_colors
