"""Delayed group catch-up must not repeat movement already accepted visually."""

import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from lerobot.gui.api.manipulation import groups, runtime, tracking
from lerobot.showservo import motion as core, pose_verification

FIXTURES = Path(__file__).parents[1] / "fixtures/showservo/transport_20261016"


@pytest.mark.parametrize("run", ["20261016_110227", "20261016_110310"])
def test_recorded_group_motion_does_not_repeat_an_overridden_visual_update(monkeypatch, run):
    data = json.loads((FIXTURES / f"{run}.json").read_text())
    check = data["check"]
    # Rename the recorded object: behavior must not depend on the scene label.
    check["object"] = "unseen-object-name"
    name = check["object"]
    feed = runtime._GroupsFeed()
    feed.frames[name] = deque((t, np.asarray(p)) for t, p in data["group_frames"])
    monkeypatch.setattr(runtime._state, "groups_feed", feed)
    monkeypatch.setattr(runtime._state, "groups_with_acts", True)
    stamp = check["t_frame"]
    visual = np.asarray(check["constraint"]["pose"])
    naive_motion = core.carried_motion(list(feed.frames[name]), stamp)
    groups._accept_transport(check, stamp)
    actual = groups._carried(name, stamp) @ visual
    if data["expect_transport_blocked"]:
        np.testing.assert_allclose(actual, visual, atol=1e-12)
        with np.load(FIXTURES / f"{run}.npz") as z:
            centre = z["reference"].mean(axis=0)
        added = (naive_motion @ visual @ np.r_[centre, 1]) - (visual @ np.r_[centre, 1])
        assert np.linalg.norm(added[:3]) > 0.015  # recorded catch-up would add >15 mm
    else:
        # The second run's accepted anchor agrees with groups. Subsequent real
        # group movement must remain usable, not be disabled by a blanket rule.
        np.testing.assert_allclose(actual, naive_motion @ visual, atol=1e-12)
        assert not np.allclose(actual, visual)


@pytest.mark.parametrize("run", ["20261016_110227", "20261016_110310"])
def test_recorded_rgbd_surface_decision_survives_plane_batching(run):
    data = json.loads((FIXTURES / f"{run}.json").read_text())
    with np.load(FIXTURES / f"{run}.npz") as z:
        diagnostic = pose_verification.surface(
            z["raw_pose"],
            z["reference"],
            z["depth"],
            z["mask"],
            data["intr"],
            noise_m=core.POINT_NOISE_M,
            tol_m=core.ACT_REACH_TOL_M,
            tol_deg=core.ACT_REACH_TOL_DEG,
        )
        decision = tracking._surface_decision(
            z["raw_pose"],
            z["geometry"],
            z["depth"],
            z["mask"],
            data["intr"],
            diagnostic,
        )
        assert decision["state"] == data["expected_surface_state"]
        np.testing.assert_allclose(decision["pose"], z["expected_pose"], atol=1e-12, rtol=0)
