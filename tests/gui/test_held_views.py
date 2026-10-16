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
    tips = np.stack([pose([8, 21, -20], [0.03, -0.2, 0.1]), pose([30, 40, -23], [0.04, -0.2, 0.09])])
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


def test_partial_rotation_does_not_extrapolate_position_to_hidden_reference_points(monkeypatch):
    mask = np.ones((24, 24), bool)
    rgb, depth = np.zeros((24, 24, 3), np.uint8), np.full((24, 24), 0.4)
    intr = {"fx": 120, "fy": 120, "cx": 12, "cy": 12}
    demo = SimpleNamespace(tips=np.stack([np.eye(4)] * 2), recording="saved")
    anchor = np.array([0.035, 0.01, 0.4])
    monkeypatch.setattr(
        held_views,
        "references",
        lambda *a: [{"frame": f, "sample": i, "mask": mask} for i, f in enumerate([17, 31])],
    )
    monkeypatch.setattr(holds, "_recording_intr", lambda *a: intr)
    monkeypatch.setattr(recording, "_stream_frame", lambda *a: (rgb, depth))

    async def view(*a):
        return rgb, depth, intr, np.eye(4), mask

    async def locate(*args, reference, **kwargs):
        delta = pose([0, 10 if reference["frame"] == 17 else 30, 0], [0, 0, 0])
        delta[:3, 3] = anchor - delta[:3, :3] @ anchor
        return {
            "ok": True,
            "delta": delta,
            "inliers": 100,
            "card_points": 120,
            "surface_constraint": {"pose": delta.tolist(), "reference_point": anchor.tolist()},
        }

    monkeypatch.setattr(holds, "_held_view", view)
    monkeypatch.setattr(acquisition, "_locate", locate)
    result, problem, attempts = asyncio.run(
        held_views.measure(demo, "arbitrary-held-object", np.eye(4), np.eye(4), lambda: False)
    )
    assert not problem and result["orientation"] == "predicted"
    np.testing.assert_allclose(result["fix"], np.eye(4), atol=1e-12)
    assert all(a["accepted"] for a in attempts)
    np.testing.assert_allclose(attempts[0]["observation"]["desired"], anchor)


@pytest.mark.usefixtures("act_simulation")
@pytest.mark.parametrize("missing", ["carry", "both"])
def test_runner_measures_after_lift_even_without_designation_based_demo_hold(tmp_path, monkeypatch, missing):
    from lerobot.gui.api import manipulation

    calls = []
    correction = np.eye(4)
    correction[0, 3] = -0.012  # synthetic lift changed the held position by another 6 mm

    async def watch(demo, obj, bc, prediction, stopped, publish, log, **kwargs):
        import time

        from lerobot.gui.api import jog

        calls.append((obj, prediction().copy()))
        while not stopped():
            if jog.current_tip_and_anchor()[0][2, 3] > 0.04:
                publish(
                    {
                        "fix": correction,
                        "orientation": "predicted",
                        "n": 3,
                        "views": 3,
                        "spread_mm": 0.0,
                        "spread_deg": 0.0,
                        "reason": "rotation unobserved",
                        "accepted_at": time.time(),
                        "observed_at": time.time(),
                    }
                )
                log("", [{"reference_frame": 70}])
                return
            await asyncio.sleep(0.001)

    async def stationary(*args, **kwargs):
        pytest.fail("prediction-enabled carry must not wait for a stationary held check")

    monkeypatch.setattr(held_views, "watch", watch)
    monkeypatch.setattr(held_views, "measure", stationary)
    try:
        demo, sim, _views, measured = _run_place_act(
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
        assert act.place["held_observer_started_before_carry"]
        from tests.gui.test_stream_objects import PLACE_AT, _TipKinematics

        wanted = measured["box"] @ demo.tips[PLACE_AT["place_end"]] @ correction
        actual = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        np.testing.assert_allclose(actual[:3, 3], wanted[:3, 3], atol=0.0005)
        assert not np.allclose(calls[0][1], correction), "prediction is a seed, not the returned measurement"
    finally:
        _end_place_state()


@pytest.mark.usefixtures("act_simulation")
def test_runner_applies_measurement_that_arrives_during_contact_and_rejects_after_release(
    tmp_path, monkeypatch
):
    import time

    from lerobot.gui.api import jog, manipulation
    from tests.gui.test_stream_objects import PLACE_AT, _TipKinematics

    correction = np.eye(4)
    correction[0, 3] = -0.012
    published = []

    async def watch(demo, obj, bc, prediction, stopped, publish, log, **kwargs):
        lifted = False
        while not stopped():
            height = jog.current_tip_and_anchor()[0][2, 3]
            lifted |= height > 0.04
            if lifted and height < 0.035:
                result = {
                    "fix": correction,
                    "orientation": "measured",
                    "reason": "",
                    "accepted_at": time.time(),
                    "observed_at": time.time(),
                }
                publish(result)
                published.append(result)
                while not stopped():
                    await asyncio.sleep(0.001)
                publish({**result, "fix": np.eye(4)})
                return
            await asyncio.sleep(0.001)

    monkeypatch.setattr(held_views, "watch", watch)
    try:
        demo, sim, _views, measured = _run_place_act(tmp_path, monkeypatch, allow_grasp_prediction=True)
        act = manipulation.runtime._state.act
        assert act.ok, act.reason
        assert published, "the estimate must arrive after the corrected hover, during descent"
        np.testing.assert_allclose(act.place["fix"], correction)
        wanted = measured["box"] @ demo.tips[PLACE_AT["place_end"]] @ correction
        actual = _TipKinematics().forward_kinematics(sim["streamed"][-1])
        np.testing.assert_allclose(actual[:3, 3], wanted[:3, 3], atol=0.0005)
        assert act.place["hold_used"] == "held-view registration"
    finally:
        _end_place_state()


def test_references_follow_the_selected_attachment_and_interpolate_image_time(monkeypatch):
    from lerobot.gui.api.manipulation import planning
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    tips = np.stack([pose([0, 0, i * 10], [i * 0.01, 0, 0]) for i in range(10)])
    q = np.zeros((10, len(MOTOR_NAMES)))
    q[:, MOTOR_NAMES.index("gripper")] = [0, 0, 0, 80, 80, 80, 79, 60, 50, 50]
    demo = SimpleNamespace(
        recording="saved",
        objects={"tool": {"masks": np.ones((10, 3, 3))}},
        t0=100.0,
        t=np.arange(10),
        tips=tips,
        q_cmd=q,
    )
    monkeypatch.setattr(
        planning, "_holding", lambda d: {"frame": "tool", "grip": 3, "release": 8, "closing": 1}
    )
    monkeypatch.setattr(recording, "_stream_times", lambda p: np.arange(10) + 100.5)
    refs = held_views.references(demo, "tool")
    assert [r["frame"] for r in refs] == [3, 4], "exclude pre-grasp and the beginning of opening"
    np.testing.assert_allclose(refs[0]["tip"], pose([0, 0, 35], [0.035, 0, 0]), atol=1e-12)
    assert held_views.references(demo, "another tool") == []


@pytest.fixture
def held_scene(monkeypatch):
    from lerobot.showservo import surface_constraint

    mask = np.ones((24, 24), bool)
    intr = {"fx": 120, "fy": 120, "cx": 12, "cy": 12}
    rgb, depth = np.zeros((24, 24, 3), np.uint8), np.full((24, 24), 0.4)
    tips = np.stack([pose([0, 0, 0], [0, 0, 0.1]), pose([0, 40, 0], [0, 0, 0.1])])
    demo = SimpleNamespace(tips=tips, recording="saved")
    bank = [
        {
            "frame": i,
            "sample": i,
            "tip": tip,
            "mask": mask,
            "depth": depth,
            "points": surface_constraint.depth_points(depth, mask, intr),
        }
        for i, tip in enumerate(tips)
    ]
    monkeypatch.setattr(holds, "_recording_intr", lambda d: intr)
    monkeypatch.setattr(held_views, "references", lambda *a: bank)
    monkeypatch.setattr(recording, "_stream_frame", lambda *a: (rgb, depth))
    state = SimpleNamespace(tip=tips[0], depth=depth, stamp=1.0, ok=True, calls=[])

    async def view(*args):
        state.stamp += 0.01
        return rgb, state.depth, intr, state.tip, mask

    async def locate(*args, reference, **kwargs):
        state.calls.append(reference["frame"])
        delta = state.tip @ np.linalg.inv(reference["tip"])
        return {
            "object": "tool",
            "ok": state.ok,
            "inliers": 100 if state.ok else 0,
            "card_points": 120,
            "delta": delta,
            "surface_constraint": {"pose": delta.tolist()},
        }

    monkeypatch.setattr(holds, "_held_view", view)
    monkeypatch.setattr(acquisition, "_locate", locate)
    return demo, bank, state, intr


def test_reference_bank_covers_view_changes_and_ranking_tracks_current_view(held_scene, monkeypatch):
    demo, bank, state, intr = held_scene
    duplicates = [dict(bank[0], frame=i) for i in range(30)] + [dict(bank[1], frame=40)]
    monkeypatch.setattr(held_views, "references", lambda *a: duplicates)
    prepared = held_views.prepare_references(demo, "tool", np.eye(4))
    assert len(prepared) == 2, "many identical frames cannot crowd out a different viewing angle"
    for ref in prepared:
        ranked = held_views.rank_references(
            prepared, demo, np.eye(4), ref["tip"], np.eye(4), state.depth, intr
        )
        assert ranked[0][0]["frame"] == ref["frame"]
        assert ranked[0][1]["predicted_depth_support"] == pytest.approx(1.0)


@pytest.mark.parametrize("change", ["arm", "attachment", "surface"])
def test_failed_identical_views_are_skipped_until_new_information_arrives(held_scene, change):
    demo, bank, state, _ = held_scene
    state.ok = False
    failed = {}
    seed = np.eye(4)

    def measure():
        return asyncio.run(
            held_views.measure(
                demo,
                "tool",
                np.eye(4),
                seed,
                lambda: False,
                reference_bank=bank,
                failed_views=failed,
                observation_clock=lambda: state.stamp,
            )
        )

    assert measure()[0] is None
    assert len(state.calls) == 2
    assert measure()[0] is None
    assert len(state.calls) == 2, "the same failed stationary view should not trigger repeated inference"
    if change == "arm":
        state.tip = pose([0, 12, 0], [0, 0, 0])
    elif change == "attachment":
        seed[0, 3] = 0.012
    else:
        state.depth = state.depth + 0.012
    measure()
    assert len(state.calls) > 2, "changed geometry must permit a new attempt"


def test_rolling_evidence_updates_on_next_measurement_and_cannot_republish_stale_pose(held_scene):
    demo, bank, state, _ = held_scene
    observations, published = [], []

    def measure():
        return asyncio.run(
            held_views.measure(
                demo,
                "tool",
                np.eye(4),
                np.eye(4),
                lambda: False,
                reference_bank=bank,
                observation_window=observations,
                observation_clock=lambda: state.stamp,
                publish=published.append,
            )
        )

    result, _, _ = measure()
    assert result is not None and len(state.calls) == 5
    previous = result["observed_stamp"]
    result, _, _ = measure()
    assert result is not None and len(state.calls) == 6, "one new view updates the existing consensus"
    assert result["observed_stamp"] > previous
    state.ok = False
    result, _, _ = measure()
    assert result is None and len(published) == 2, "old accepted evidence is not a fresh observation"
    state.stamp += 10
    measure()
    assert not observations, "old attachment evidence expires"


def test_a_new_consensus_outlier_cannot_refresh_old_accepted_evidence(held_scene, monkeypatch):
    demo, bank, state, _ = held_scene
    observations, published = [], []

    def measure():
        return asyncio.run(
            held_views.measure(
                demo,
                "tool",
                np.eye(4),
                np.eye(4),
                lambda: False,
                reference_bank=bank,
                observation_window=observations,
                observation_clock=lambda: state.stamp,
                publish=published.append,
            )
        )

    first, _, _ = measure()
    locate = acquisition._locate
    calls = 0

    async def outlier(*args, **kwargs):
        nonlocal calls
        calls += 1
        found = await locate(*args, **kwargs)
        found["delta"][0, 3] += 0.05
        state.ok = False  # the following images provide no accepted evidence
        return found

    monkeypatch.setattr(acquisition, "_locate", outlier)
    result, _, _ = measure()
    assert result is None and len(published) == 1
    old_consensus = held_views.consensus(observations, np.eye(4))
    assert old_consensus["observed_stamp"] == first["observed_stamp"]
    assert old_consensus["observed_stamp"] < observations[-1]["stamp"]
