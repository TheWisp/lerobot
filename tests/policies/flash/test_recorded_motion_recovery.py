"""Real sparse RGB-D inputs, followed through matching, pose and map consumers.

Known production failures stay strict xfails. The opt-in benchmark policy is
also checked here so the proposed recovery must satisfy the SAME assertions.
No saved pose, expected surface fit, or future image enters the estimator.
"""

import json
import os
from copy import deepcopy

import cv2
import numpy as np
import pytest

pytest.importorskip("small_gicp", reason="Recorded pipeline uses the small_gicp registration backend")

from benchmarks.flash_motion_recovery import FIXTURE, load_fixture, replay
from lerobot.policies.flash.surface_observation import move


@pytest.fixture(scope="module")
def recorded():
    return load_fixture()


@pytest.fixture(scope="module", autouse=True)
def opencv_threads():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(previous)


@pytest.fixture(scope="module")
def runs(recorded):
    cache = {}

    def run(case, policy, missing=False):
        key = case, policy, missing
        if key not in cache:
            cache[key] = replay(case, policy=policy, missing=missing, fixture=recorded)
        return cache[key]

    return run


def assert_visible_recovery(rows, recorded, first_observed_index):
    """Score published reference centre against independent RGB review regions.

    These deliberately broad regions check visible location, not exact 6D pose
    or symmetry-sensitive orientation. They exclude the stale cube position.
    """
    labels = json.loads((FIXTURE / "review_regions.json").read_text())
    for row in rows[first_observed_index:]:
        n = row["frame"]
        assert row["status"] == "observed_candidate", (n, row["status"], row["details"].get("reason"))
        assert row["pose"] is not None
        center = move(np.array(row["reference"]), np.array(row["pose"])).mean(0)
        assert np.isfinite(center).all() and center[2] > 0
        frame, masks = recorded[1][n]
        camera = frame.intrinsics
        projected = (camera @ center)[:2] / center[2]
        left, top, right, bottom = labels[str(n)]
        assert left <= projected[0] <= right and top <= projected[1] <= bottom, (n, projected)
        depth = frame.metric_depth()
        measured = depth[masks["object"] & np.isfinite(depth) & (depth > 0)]
        assert len(measured) > 0
        assert abs(center[2] - np.median(measured)) < 0.02, (n, center[2], np.median(measured))
        assert row["accepted_frame"] == n, "A carried prediction is not a recovered observation"
        assert row["map_version"] > rows[0]["map_version"]


@pytest.mark.parametrize(
    "case,policy",
    [
        ("cube", "baseline"),
        ("pointer", "baseline"),
        pytest.param(
            "ring",
            "baseline",
            marks=pytest.mark.xfail(strict=True, reason="Kalman fusion rejects recovered ring geometry"),
        ),
        ("cube", "recover_reset"),
        ("pointer", "recover_reset"),
        ("ring", "recover_reset"),
    ],
)
def test_sparse_recording_recovers_and_tracks_following_frames(case, policy, runs, recorded):
    rows = runs(case, policy)
    assert rows[0]["status"] == "initialized"
    assert_visible_recovery(rows, recorded, 1)


@pytest.mark.parametrize("case", ["cube", "pointer"])
@pytest.mark.parametrize("policy", ["baseline", "recover_reset"])
def test_reappearing_object_recovers_without_surviving_image_identities(case, policy, runs, recorded):
    rows = runs(case, policy, True)
    hidden = rows[1]
    assert hidden["status"] != "observed_candidate"
    assert hidden["accepted_frame"] == rows[0]["frame"]
    assert hidden["map_version"] == rows[0]["map_version"]
    # A successful return cannot be explained by secretly preserving 2D tracks.
    assert rows[2]["details"]["image_pairs"] == 0
    assert_visible_recovery(rows, recorded, 2)


@pytest.mark.parametrize("case", ["cube", "pointer", "ring"])
def test_missing_measurements_do_not_refresh_the_map_or_claim_observation(case, runs):
    rows = runs(case, "recover_reset", True)
    initial, hidden = rows[:2]
    assert hidden["status"] in {"predicted", "lost"}
    assert hidden["accepted_frame"] == initial["accepted_frame"]
    assert hidden["map_version"] == initial["map_version"]
    np.testing.assert_array_equal(hidden["surface"], initial["surface"])
    # Complement: recovery must actually move the object and advance the map.
    recovered = rows[2]
    assert recovered["status"] == "observed_candidate"
    assert recovered["accepted_frame"] == recovered["frame"]
    before = move(np.array(initial["reference"]), np.array(initial["pose"])).mean(0)
    after = move(np.array(initial["reference"]), np.array(recovered["pose"])).mean(0)
    assert np.linalg.norm(after - before) > 0.01


def test_fixture_contains_only_eleven_lossless_observations(recorded):
    import hashlib

    meta, frames = recorded
    assert len(frames) == 11
    assert (FIXTURE / "observations.npz").stat().st_size < 7 * 1024 * 1024
    for n, (frame, masks) in frames.items():
        info = meta["frames"][str(n)]
        assert hashlib.sha256(frame.rgb.tobytes()).hexdigest() == info["rgb_sha256"]
        assert hashlib.sha256(frame.depth.tobytes()).hexdigest() == info["depth_sha256"]
        assert masks["object"].shape == frame.depth.shape == frame.rgb.shape[:2]


@pytest.mark.parametrize("policy", ["baseline", "recover_reset", "rematch_reset"])
def test_recovery_does_not_accept_the_green_ring_as_the_cube(policy, recorded):
    meta, frames = recorded
    meta = deepcopy(meta)
    # Actual cube initialization followed by actual SAM ring observations.
    # This is a deliberate wrong-object control, not a chronological recording.
    meta["cases"]["wrong_object"] = [2735, 4759, 4765]
    rows = replay("wrong_object", policy=policy, fixture=(meta, frames))
    for row in rows[1:]:
        assert row["status"] == "lost"
        assert row["pose"] is None
        assert row["accepted_frame"] == rows[0]["frame"]
        assert row["map_version"] == rows[0]["map_version"]


@pytest.fixture(scope="module")
def ufm_model():
    if os.environ.get("FLASH_RUN_UFM_RECOVERY") != "1":
        pytest.skip("Opt in to cached real UFM inference with FLASH_RUN_UFM_RECOVERY=1")
    pytest.importorskip("uniflowmatch")
    from lerobot.policies.flash.surface_tracking import load_ufm

    return load_ufm(revision="980825b59331413bb51e5773f9272bfce525bce2")


@pytest.mark.parametrize("case", ["cube", "pointer", "ring"])
def test_real_ufm_rematches_the_retained_image_after_total_loss(case, recorded, ufm_model):
    rows = replay(
        case, policy="rematch_reset", fixture=recorded, missing=True, model=ufm_model, backend="ufm"
    )
    assert rows[1]["accepted_frame"] == rows[0]["frame"]
    assert rows[1]["map_version"] == rows[0]["map_version"]
    assert_visible_recovery(rows, recorded, 2)
    assert rows[2]["details"]["image_pairs"] >= 6


def test_real_ufm_reacquisition_rejects_the_wrong_object(recorded, ufm_model):
    meta, frames = recorded
    meta = deepcopy(meta)
    meta["cases"]["wrong_object"] = [2735, 4759, 4765]
    rows = replay(
        "wrong_object", policy="rematch_reset", fixture=(meta, frames), model=ufm_model, backend="ufm"
    )
    assert all(row["status"] == "lost" and row["pose"] is None for row in rows[1:])
    assert all(row["accepted_frame"] == rows[0]["frame"] for row in rows[1:])
