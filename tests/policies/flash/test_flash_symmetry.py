"""Declared physical symmetry must remove unobservable tracker spin."""

from pathlib import Path

import numpy as np

from lerobot.policies.flash.symmetry import AxialSymmetry


def _turn_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def test_annotated_dowel_axis_removes_real_static_fit_jitter():
    fixture = Path(__file__).parent / "fixtures" / "dowel_axial_jitter_20261020.npz"
    with np.load(fixture) as recorded:
        points = recorded["points"]
        direct = recorded["direct_poses"]
        baseline = recorded["demo_reference"]
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(points.T))
    symmetry = AxialSymmetry("dowel", vectors[:, -1], center, baseline)
    canonical = [symmetry.canonicalize(pose) for pose in direct]

    raw_turns = [_turn_deg(a, b) for a, b in zip(direct, direct[1:], strict=False)]
    corrected_turns = [_turn_deg(a, b) for a, b in zip(canonical, canonical[1:], strict=False)]
    assert max(raw_turns) > 15  # the live 15-degree replanning envelope was exceeded
    assert max(corrected_turns) < 7
    for raw, fixed in zip(direct, canonical, strict=True):
        np.testing.assert_allclose(raw[:3, :3] @ center + raw[:3, 3], fixed[:3, :3] @ center + fixed[:3, 3])
