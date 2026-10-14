"""Held-view correction without registration back to the designation view."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.gui.api.manipulation import acquisition, held_views, holds, recording
from tests.gui.test_stream_objects import _end_place_state, _run_place_act


def pose(angle, position):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", angle, degrees=True).as_matrix()
    out[:3, 3] = position
    return out


def test_reference_origins_do_not_change_the_attachment_correction():
    """Camera origins differ; every observed point must land on its demo counterpart."""
    wanted = pose([9, -6, 3], [0.007, -0.002, 0.003])
    observations = []
    for i in range(6):
        p = np.array([0.01 * i, -0.02, 0.07])
        observations.append(
            {
                "frame": i % 2,
                "desired": wanted[:3, :3] @ p + wanted[:3, 3],
                "observed": p,
                "rotation": wanted[:3, :3],
            }
        )
    result = held_views.consensus(observations, np.eye(4))
    assert result["orientation"] == "measured"
    np.testing.assert_allclose(result["fix"], wanted, atol=1e-12)


def test_ambiguous_rotation_keeps_prediction_but_moves_the_matched_patch():
    predicted = pose([0, 0, 11], [0.03, 0.02, 0.01])
    observed = np.array([0.06, -0.01, 0.09])
    desired = np.array([0.05, 0.02, 0.08])
    observations = [
        {
            "frame": i % 2,
            "desired": desired,
            "observed": observed,
            "rotation": Rotation.from_euler("z", 8 if i % 2 else -8, degrees=True).as_matrix(),
        }
        for i in range(6)
    ]
    result = held_views.consensus(observations, predicted)
    assert result["orientation"] == "predicted" and result["reference_rotation_disagreement_deg"] > 3
    np.testing.assert_allclose(result["fix"][:3, :3], predicted[:3, :3])
    np.testing.assert_allclose((result["fix"] @ np.r_[observed, 1])[:3], desired)
    assert held_views.consensus(observations, None) is None, "no attachment prior may be invented"


def test_contradictory_positions_cannot_be_averaged_into_a_verified_hold():
    observations = [
        {
            "frame": i % 2,
            "desired": np.array([i * 0.02, 0, 0]),
            "observed": np.zeros(3),
            "rotation": np.eye(3),
        }
        for i in range(6)
    ]
    assert held_views.consensus(observations, np.eye(4)) is None


@pytest.mark.parametrize("checked", [True, False])
def test_current_measurement_uses_held_reference_coordinates_and_geometry(monkeypatch, checked):
    bc = pose([11, 30, -8], [0.12, -0.08, 0.4])
    live_tip = pose([6, 18, -25], [0.1, -0.2, 0.06])
    tips = np.stack([pose([8, 21, -20], [0.03, -0.2, 0.1]), pose([10, 20, -23], [0.04, -0.2, 0.09])])
    wanted = pose([2, -3, 1], [0.007, -0.003, 0.002])
    mask = np.ones((24, 24), bool)
    intr = {"fx": 120, "fy": 120, "cx": 12, "cy": 12}
    rgb, depth = np.zeros((24, 24, 3), np.uint8), np.full((24, 24), 0.4)
    demo = SimpleNamespace(tips=tips, recording="saved")
    monkeypatch.setattr(
        held_views,
        "references",
        lambda *a: [
            {"frame": f, "sample": i, "mask": mask, "refresh_mask": True} for i, f in enumerate([17, 31])
        ],
    )
    monkeypatch.setattr(holds, "_recording_intr", lambda *a: intr)
    monkeypatch.setattr(recording, "_stream_frame", lambda *a: (rgb, depth))

    async def view(*a):
        return rgb, depth, intr, live_tip, mask

    calls = []

    async def locate(obj, rgb, depth, intr, click, stopped, **kwargs):
        ref = kwargs["reference"]
        calls.append((obj, ref["frame"]))
        delta = np.linalg.inv(bc) @ live_tip @ np.linalg.inv(wanted) @ np.linalg.inv(tips[ref["sample"]]) @ bc
        return {
            "ok": True,
            "delta": delta,
            "inliers": 100,
            "card_points": 120,
            "surface_constraint": {"pose": delta.tolist() if checked else None},
            "reference_mask": mask,
            "reason": "",
        }

    monkeypatch.setattr(holds, "_held_view", view)
    monkeypatch.setattr(acquisition, "_locate", locate)
    result, problem, attempts = asyncio.run(
        held_views.measure(demo, "arbitrary-tool", bc, np.eye(4), lambda: False)
    )
    assert {c[1] for c in calls} == {17, 31}
    assert attempts
    if checked:
        assert not problem and result["n"] == 5
        np.testing.assert_allclose(result["fix"], wanted, atol=1e-12)
    else:
        assert result is None, "a strong appearance match cannot bypass rejected geometry"


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("missing", ["carry", "both"])
def test_runner_measures_after_lift_even_without_designation_based_demo_hold(tmp_path, monkeypatch, missing):
    from lerobot.gui.api import manipulation

    calls = []
    correction = np.eye(4)
    correction[0, 3] = -0.012  # synthetic lift changed the held position by another 6 mm

    async def measure(demo, obj, bc, predicted, stopped):
        calls.append((obj, predicted.copy()))
        return (
            {
                "fix": correction,
                "orientation": "predicted",
                "n": 3,
                "views": 3,
                "spread_mm": 0.0,
                "spread_deg": 0.0,
                "reason": "rotation unobserved",
            },
            "",
            [{"reference_frame": 70}],
        )

    monkeypatch.setattr(held_views, "measure", measure)
    try:
        _run_place_act(
            tmp_path,
            monkeypatch,
            weak_carry=missing == "carry",
            blind=missing == "both",
            slip=True,
            allow_grasp_prediction=True,
        )
        act = manipulation.runtime._state.act
        assert calls and act.ok, act.reason
        assert act.place["hold_evidence"] == "partly measured"
        np.testing.assert_allclose(act.place["fix"], correction)
        assert act.place["held_view_attempts"][0]["reference_frame"] == 70
        assert not np.allclose(calls[0][1], correction), "prediction is a seed, not the returned measurement"
    finally:
        _end_place_state()
