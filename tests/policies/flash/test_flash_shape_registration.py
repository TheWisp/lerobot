"""A real visible RGB-D surface can recover startup without false certainty."""

from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.matched_rgbd import MatchedRGBDPoseBackend, ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.registration import DegenerateRegistrationError
from lerobot.policies.flash.shape_registration import register_visible_shape
from lerobot.policies.flash.trajectory import Evidence

FIXTURE = Path(__file__).parent / "fixtures/dowel_live_visible_shape_20261020.npz"
DRIFT_FIXTURE = Path(__file__).parent / "fixtures/dowel_static_fit_drift_20261020.npz"


class EmptyMatches:
    def __init__(self, mask):
        self.last_masks = {"object": mask}

    def matches(self, _frame):
        return ()


def _fixture():
    with np.load(FIXTURE) as saved:
        return {key: saved[key] for key in saved.files}


def test_real_visible_surface_recovers_pose_but_fragment_and_other_object_abstain():
    saved = _fixture()
    fit = register_visible_shape(
        saved["model"],
        saved["depth"],
        saved["full_mask"],
        saved["intrinsics"],
        0.001,
        tolerance_m=0.005,
        min_points=12,
    )
    assert max(fit.model_p90_m, fit.live_p90_m) < 0.005
    assert fit.agreeing_seeds >= 1
    for mask in (saved["fragment_mask"], saved["other_object_mask"]):
        with pytest.raises(DegenerateRegistrationError):
            register_visible_shape(
                saved["model"],
                saved["depth"],
                mask,
                saved["intrinsics"],
                0.001,
                tolerance_m=0.005,
                min_points=12,
            )


def test_pose_backend_uses_measured_shape_only_when_enabled_and_appearance_fails():
    saved = _fixture()
    frame = RGBDFrame(
        0,
        np.zeros((*saved["depth"].shape, 3), dtype=np.uint8),
        saved["depth"],
        saved["intrinsics"],
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "cal",
    )
    model = ObjectPointModel("object", "v1", saved["model"])
    reference = ObjectReference(
        "object", "v1", (ReferenceSnippet("demo", 0, 0, "object", np.eye(4), "mask", "cal"),)
    )
    for enabled, expected in [(False, Evidence.LOST), (True, Evidence.OBSERVED)]:
        backend = MatchedRGBDPoseBackend(
            (model,),
            EmptyMatches(saved["full_mask"]),
            threshold_m=0.005,
            min_inliers=12,
            trials=20,
            visible_shape_fallback=enabled,
            clock=lambda: 1.1,
        )
        backend.initialize((reference,))
        (estimate,) = backend.update(frame)
        assert estimate.evidence == expected
        assert (estimate.pose is not None) == enabled
        assert estimate.direct_translation_observed == enabled
        if enabled:
            assert estimate.estimator == "visible_shape_rgbd"
            assert estimate.last_direct_observation_s == frame.source_time_s


def test_real_static_dowel_rejects_low_residual_but_wrong_object_frame():
    with np.load(DRIFT_FIXTURE) as saved:
        data = {key: saved[key] for key in saved.files}
    first_shape = register_visible_shape(
        data["model"],
        data["depth_0"],
        data["mask_0"],
        data["intrinsics"],
        0.001,
        tolerance_m=0.005,
        min_points=12,
    )
    second_shape = register_visible_shape(
        data["model"],
        data["depth_13"],
        data["mask_13"],
        data["intrinsics"],
        0.001,
        tolerance_m=0.005,
        min_points=12,
    )
    assert max(first_shape.model_p90_m, first_shape.live_p90_m) < 0.005
    assert max(second_shape.model_p90_m, second_shape.live_p90_m) < 0.005
    assert (
        np.linalg.norm(first_shape.camera_from_object[:3, 3] - second_shape.camera_from_object[:3, 3]) > 0.02
    )  # The real mask barely moved, but ICP slid the object frame.

    provider = EmptyMatches(data["mask_0"])
    backend = MatchedRGBDPoseBackend(
        (ObjectPointModel("object", "v1", data["model"]),),
        provider,
        threshold_m=0.005,
        min_inliers=12,
        trials=20,
        visible_shape_fallback=True,
        clock=lambda: 2.0,
    )
    backend.initialize(
        (
            ObjectReference(
                "object", "v1", (ReferenceSnippet("demo", 0, 0, "object", np.eye(4), "mask", "cal"),)
            ),
        )
    )
    first = RGBDFrame(
        0,
        np.zeros((*data["depth_0"].shape, 3), dtype=np.uint8),
        data["depth_0"],
        data["intrinsics"],
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "cal",
    )
    second = RGBDFrame(13, first.rgb, data["depth_13"], data["intrinsics"], 0.001, np.eye(4), 1.5, 1.5, "cal")
    assert backend.update(first)[0].evidence == Evidence.OBSERVED
    provider.last_masks["object"] = data["mask_13"]
    assert backend.update(second)[0].evidence == Evidence.LOST
    assert "object frame" in backend.last_failure["object"]
