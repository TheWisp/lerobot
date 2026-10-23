"""Multiview appearance matches feed the common metric pose estimator."""

import pickle

import numpy as np

from lerobot.policies.flash.appearance import (
    AppearanceBackendFactory,
    AppearanceModel,
    ReferenceAppearanceFrame,
    Sam3MaskSource,
    build_appearance_model,
    build_appearance_pose_backend,
)
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.trajectory import Evidence

POINTS = np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [-0.02, 0.02, 0], [0.02, 0.02, 0], [0, 0, 0]])


class Extractor:
    def __init__(self, answers):
        self.answers = iter(answers)

    def extract(self, rgb, mask):
        return next(self.answers)


class Masks:
    def __init__(self, mask):
        self.current = {"object": mask}

    def masks(self, frame):
        return self.current


def frame(frame_id, pixels, source_time):
    depth = np.zeros((24, 24), dtype=np.uint16)
    depth[np.rint(pixels[:, 1]).astype(int), np.rint(pixels[:, 0]).astype(int)] = 1000
    k = np.array([[100.0, 0, 10], [0, 100.0, 10], [0, 0, 1.0]])
    return RGBDFrame(
        frame_id,
        np.zeros((24, 24, 3), dtype=np.uint8),
        depth,
        k,
        0.001,
        np.eye(4),
        source_time,
        source_time,
        "camera-v1",
    )


def test_demo_and_live_backend_construction_register_the_same_view(monkeypatch):
    demo_xy = POINTS[:, :2] * 100 + 10
    live_xy = POINTS[:, :2] * 100 + [12, 10]
    demo0, demo1 = frame(0, demo_xy, 0.0), frame(1, demo_xy, 1.0)
    live = frame(2, live_xy, 5.0)
    object_from_camera = np.eye(4)
    object_from_camera[2, 3] = -1
    s0 = ReferenceSnippet("dataset", 0, 0, "object", object_from_camera, "mask0", "camera-v1")
    s1 = ReferenceSnippet("dataset", 0, 1, "object", object_from_camera, "mask1", "camera-v1")
    reference = ObjectReference("object", "object-v1", (s0, s1))
    features = np.eye(len(POINTS), dtype=np.float32)
    extractor = Extractor(((demo_xy[:3], features[:3]), (demo_xy[3:], features[3:]), (live_xy, features)))
    mask = np.ones((24, 24), dtype=bool)
    appearance = build_appearance_model(
        reference,
        (
            ReferenceAppearanceFrame(s0, demo0, mask),
            ReferenceAppearanceFrame(s1, demo1, mask),
        ),
        extractor,
    )
    np.testing.assert_allclose(appearance.points.points_object_m, POINTS)
    masks = Masks(mask)
    backend = build_appearance_pose_backend(
        (appearance,),
        extractor,
        masks,
        min_similarity=0.9,
        threshold_m=0.001,
        min_inliers=3,
        trials=20,
        visible_shape_fallback=False,
        clock=lambda: 5.1,
    )
    backend.initialize((reference,))
    (estimate,) = backend.update(live)
    assert estimate.evidence == Evidence.OBSERVED
    np.testing.assert_allclose(estimate.pose[:3, 3], [0.02, 0, 1], atol=1e-8)

    import lerobot.policies.flash.appearance as appearance_module

    monkeypatch.setattr(appearance_module, "Sam3MaskSource", lambda *args, **kwargs: Masks(mask))
    monkeypatch.setattr(
        "lerobot.policies.flash.dino_features.DinoPatchExtractor",
        lambda **kwargs: Extractor(((live_xy, features),)),
    )
    live_backend = AppearanceBackendFactory(
        (appearance,), {"object": "wooden tool"}, "overhead", 0.9, 0.001, 3, 20
    )()
    live_backend.initialize((reference,))
    (live_estimate,) = live_backend.update(live)
    assert live_estimate.evidence == Evidence.OBSERVED
    np.testing.assert_allclose(live_estimate.pose, estimate.pose, atol=1e-8)

    masks.current = {}
    (lost,) = backend.update(live)
    assert lost.evidence == Evidence.LOST


def test_sam3_mask_source_shares_one_adapter_and_resets_on_discontinuity():
    class Adapter:
        def __init__(self, **kwargs):
            self.calls = []

        def set_camera(self, camera_id):
            self.calls.append(("camera", camera_id))

        def set_control(self, settings):
            self.calls.append(("prompt", settings["prompt"]))

        def reset(self):
            self.calls.append(("reset",))

        def segment(self, rgb):
            self.calls.append(("segment",))
            return {"tool": np.ones(rgb.shape[:2], dtype=bool), "ring": np.zeros(rgb.shape[:2], dtype=bool)}

    made = []

    def factory(**kwargs):
        adapter = Adapter(**kwargs)
        made.append(adapter)
        return adapter

    source = Sam3MaskSource({"held": "tool", "target": "ring"}, camera_id="overhead", adapter_factory=factory)
    xy = np.array([[10.0, 10.0]])
    assert set(source.masks(frame(1, xy, 1.0))) == {"held", "target"}
    source.masks(frame(2, xy, 2.0))
    source.masks(frame(1, xy, 3.0))
    assert len(made) == 1
    assert made[0].calls.count(("segment",)) == 3
    assert made[0].calls.count(("reset",)) == 1


def test_appearance_factory_can_cross_spawn_boundary_without_loaded_models():
    model = AppearanceModel(ObjectPointModel("object", "v1", POINTS), np.eye(len(POINTS)))
    factory = AppearanceBackendFactory((model,), {"object": "wooden tool"}, "overhead", 0.75, 0.005, 3, 100)
    restored = pickle.loads(pickle.dumps(factory))
    assert restored.prompts_by_role == {"object": "wooden tool"}
    np.testing.assert_allclose(restored.models[0].points.points_object_m, POINTS)
