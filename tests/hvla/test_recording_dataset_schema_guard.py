"""A resumed recording dataset must match the robot in front of you.

Measured incident (2026-08-23): eval/eval_ball_0823_3 was created at 16:27 by a
robot whose stereo top camera publishes top_l/top_r. At 17:21 the same repo_id
was resumed by an older robot profile publishing a single unsplit `top`. The
schema is frozen at creation, so the first frame failed inside validate_frame
with "Missing features: {top_l, top_r} / Extra features: {top}" -- after the
robot had connected and the checkpoint had loaded, and with nothing in the
message naming the real cause (a resume across two robot configurations).

The guard turns that into a startup refusal that names both column sets.
"""

from __future__ import annotations

import pytest

from lerobot.policies.hvla.s1_process import _assert_schema_matches_robot

STEREO = {
    "observation.images.left_wrist": {},
    "observation.images.right_wrist": {},
    "observation.images.top_l": {},
    "observation.images.top_r": {},
    "observation.state": {},
    "action": {},
}
UNSPLIT = {
    "observation.images.left_wrist": {},
    "observation.images.right_wrist": {},
    "observation.images.top": {},
    "observation.state": {},
    "action": {},
}


def test_the_incident_is_refused_at_startup():
    with pytest.raises(ValueError) as e:
        _assert_schema_matches_robot("eval/eval_ball_0823_3", STEREO, UNSPLIT)
    msg = str(e.value)
    # The message has to name what differs; "Missing features" alone is what
    # made the original failure unreadable.
    assert "observation.images.top_l" in msg and "observation.images.top" in msg
    assert "new dataset name" in msg, "must say how to proceed"


def test_the_same_robot_resumes_silently():
    _assert_schema_matches_robot("eval/x", STEREO, dict(STEREO))


def test_a_resolution_change_is_not_a_schema_change():
    """Shapes are re-encodable; the column SET is what cannot be appended to."""
    wider = {k: ({"shape": (720, 2560, 3)} if "images" in k else {}) for k in STEREO}
    _assert_schema_matches_robot("eval/x", STEREO, wider)


def test_a_missing_camera_is_caught_in_both_directions():
    fewer = {k: v for k, v in STEREO.items() if k != "observation.images.left_wrist"}
    with pytest.raises(ValueError):
        _assert_schema_matches_robot("eval/x", STEREO, fewer)  # robot lost a camera
    with pytest.raises(ValueError):
        _assert_schema_matches_robot("eval/x", fewer, STEREO)  # robot gained one


def test_the_resume_path_actually_applies_the_guard(tmp_path, monkeypatch):
    """Pin the CALL SITE, not just the rule.

    The rule tests above still pass with the check deleted from
    _create_or_resume_dataset -- which is exactly how the original defect
    shipped: a correct idea nothing invoked. This drives the real resume path.
    """
    import lerobot.datasets.lerobot_dataset as ds_mod
    import lerobot.utils.constants as const_mod
    from lerobot.policies.hvla.s1_process import _create_or_resume_dataset

    repo_id = "eval/resumed_with_another_robot"
    (tmp_path / repo_id).mkdir(parents=True)
    monkeypatch.setattr(const_mod, "HF_LEROBOT_HOME", tmp_path)

    class _Meta:
        features = STEREO

    class _Existing:
        meta = _Meta()

    monkeypatch.setattr(ds_mod.LeRobotDataset, "resume", classmethod(lambda cls, *a, **k: _Existing()))

    with pytest.raises(ValueError, match="different set of columns"):
        _create_or_resume_dataset(repo_id=repo_id, fps=30, features=UNSPLIT, robot_type="bi_openarm_follower")

    # ...and the matching robot still resumes.
    got = _create_or_resume_dataset(repo_id=repo_id, fps=30, features=dict(STEREO), robot_type="x")
    assert got is not None
