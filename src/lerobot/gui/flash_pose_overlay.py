"""Fixed display axes shared by the experiment's image and 3D views."""

import numpy as np

from lerobot.policies.flash.surface_observation import initial_landmarks


def reference_axes(reference: np.ndarray) -> np.ndarray:
    """Origin and three endpoints in the initial surface's coordinates.

    This is a display gizmo, not a new definition of the estimator's pose.
    Never recompute it from the growing surface or the current visible points.
    """
    anchors, basis = initial_landmarks(reference)
    length = np.linalg.norm(np.ptp(reference, axis=0)) / 3
    return np.concatenate((anchors[:1], anchors[0] + basis.T * length))
