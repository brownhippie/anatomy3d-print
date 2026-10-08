from dataclasses import dataclass

import numpy as np


@dataclass
class BodyMesh:
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)
