"""Robot episode decoding and editor tracking must agree with live tracking."""

import gzip
import json
import struct

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.flash_live_scene import object_evidence
from lerobot.policies.flash.demo_document import DemoKeyframe, DemoObject, load_document
from lerobot.policies.flash.demo_tracking import DemoFrames, track_demo
from lerobot.policies.flash.live_surface_session import LIVE_METHOD, LiveSurfaceSession
from lerobot.policies.flash.surface_session_config import defaults
from tests.policies.flash.test_shared_image_tracking import frame, masks


@pytest.fixture
def rgbd_episode(tmp_path):
    first = frame(0)
    height, width = first.depth.shape
    k = first.intrinsics
    calibration = {
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": k[0, 0],
            "fy": k[1, 1],
            "cx": k[0, 2],
            "cy": k[1, 2],
        }
    }
    dataset = LeRobotDataset.create(
        repo_id="test/demo-rgbd",
        root=tmp_path / "rgbd",
        fps=10,
        use_videos=True,
        features={
            "observation.images.camera": {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
                "info": {"camera": calibration},
            },
            "observation.images.camera_depth": {
                "dtype": "image",
                "shape": (height, width, 1),
                "names": ["height", "width", "channels"],
                "info": {
                    "is_depth_map": True,
                    "camera": {**calibration, "aligned_to": "observation.images.camera"},
                },
            },
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        },
    )
    for index in range(4):
        current = frame(index)
        dataset.add_frame(
            {
                "task": "test",
                "observation.images.camera": current.rgb,
                "observation.images.camera_depth": current.depth[..., None],
                "observation.state": np.array([index], dtype=np.float32),
                "action": np.array([index], dtype=np.float32),
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset("test/demo-rgbd", root=dataset.root)


class Masks:
    prompts = {"object_1": "left", "object_2": "right"}

    def masks(self, frame):
        values = masks()
        return {"object_1": values["left"], "object_2": values["right"]}


def test_editor_uses_actual_decoded_depth_and_live_session(rgbd_episode, tmp_path):
    pytest.importorskip("small_gicp")
    document, _ = load_document(rgbd_episode, 0)
    document.objects = [DemoObject(id="object_1", name="left"), DemoObject(id="object_2", name="right")]
    document.reference_frame = 1
    document.keyframes = [DemoKeyframe(frame=3, reference="object_2")]
    decoded = list(DemoFrames(rgbd_episode, 0, document))
    assert [value.frame_id for value in decoded] == [1, 2, 3]
    np.testing.assert_allclose(decoded[0].metric_depth(), frame(1).metric_depth(), atol=1e-6)
    np.testing.assert_allclose([value.source_time_s for value in decoded], [0.1, 0.2, 0.3])
    output = tmp_path / "tracked"
    track_demo(rgbd_episode, 0, document, output, mask_source=Masks())
    live = LiveSurfaceSession(defaults(), Masks())
    try:
        for current in decoded:
            row, _, _ = live.step(current)
            saved = json.loads((output / "frames" / f"{current.frame_id:06d}.json").read_text())
            packet = gzip.decompress((output / "scenes" / f"{current.frame_id:06d}.bin.gz").read_bytes())
            header_size = struct.unpack("<I", packet[:4])[0]
            header = json.loads(packet[4 : 4 + header_size])
            depth = np.frombuffer(packet, dtype="<f4", count=current.depth.size, offset=4 + header_size)
            np.testing.assert_array_equal(depth.reshape(current.depth.shape), current.metric_depth())
            assert header["frame"] == current.frame_id
            assert header["objects"] == saved["objects"]
            assert header["camera"] == saved["camera"]
            assert header["scene_group_tracks"] == saved["scene_group_tracks"]
            assert [
                {key: value for key, value in group.items() if key not in {"links", "predicted_links"}}
                for group in header["scene_group_tracks"]
            ] == row["scene_group_tracks"]
            assert (
                len(packet) == 4 + header_size + header["depth_bytes"]
            )  # RGB uses the existing episode video.
            for role, value in row["objects"].items():
                expected = object_evidence(value, LIVE_METHOD)
                assert saved["objects"][role]["status"] == expected["status"]
                assert expected["pose"] is not None  # Not a vacuous 'both lost' comparison.
                np.testing.assert_allclose(saved["objects"][role]["pose"], expected["pose"], atol=1e-9)
                np.testing.assert_allclose(
                    saved["objects"][role]["surface"], live.engine.candidates[LIVE_METHOD][role].store.points
                )
        assert json.loads((output / "frames" / "000003.json").read_text())["scene_group_tracks"]
        assert json.loads((output / "status.json").read_text())["state"] == "complete"
        with np.load(output / "surfaces.npz") as surfaces:
            assert set(surfaces.files) == {"object_1", "object_2"}
            assert len(surfaces["object_1"]) > 20
    finally:
        live.close()


def test_missing_alignment_is_not_guessed(rgbd_episode):
    document, _ = load_document(rgbd_episode, 0)
    del rgbd_episode.meta.features["observation.images.camera_depth"]["info"]["camera"]["aligned_to"]
    with pytest.raises(ValueError, match="color-aligned"):
        DemoFrames(rgbd_episode, 0, document)


def test_regenerated_depth_keeps_metric_units(rgbd_episode):
    from lerobot.policies.flash.depth_enhancement import DepthOutput, tracking_frame_with_depth_source

    document, _ = load_document(rgbd_episode, 0)
    recorded = next(iter(DemoFrames(rgbd_episode, 0, document)))

    class Model:
        def predict(self, inputs):
            return DepthOutput(
                np.full(inputs.rgb.shape[:2], 0.42, np.float32), "test", inputs.sensor_depth_m > 0
            )

    regenerated = tracking_frame_with_depth_source(recorded, Model())
    np.testing.assert_allclose(regenerated.metric_depth(), 0.42, atol=1e-7)


def test_stereo_model_requires_recorded_pair(rgbd_episode):
    document, _ = load_document(rgbd_episode, 0)
    with pytest.raises(ValueError, match="no stereo IR"):
        DemoFrames(rgbd_episode, 0, document).require_depth_source("foundation_stereo")


def test_saved_scene_preserves_model_depth_instead_of_sensor_depth(rgbd_episode, tmp_path, monkeypatch):
    from lerobot.policies.flash import depth_enhancement

    document, _ = load_document(rgbd_episode, 0)
    document.objects = [DemoObject(id="object_1", name="left"), DemoObject(id="object_2", name="right")]
    expected = np.full(frame(0).depth.shape, 0.42123456, dtype=np.float32)
    expected[:4, :4] = 0  # Missing depth must not become filled geometry.

    class Model:
        def predict(self, inputs):
            return depth_enhancement.DepthOutput(expected.copy(), "test", inputs.sensor_depth_m > 0)

    monkeypatch.setattr(depth_enhancement, "make_depth_source", lambda source: Model())
    output = tmp_path / "model-tracked"
    track_demo(rgbd_episode, 0, document, output, mask_source=Masks(), depth_source="test")
    packet = gzip.decompress((output / "scenes" / "000000.bin.gz").read_bytes())
    header_size = struct.unpack("<I", packet[:4])[0]
    header = json.loads(packet[4 : 4 + header_size])
    restored = np.frombuffer(packet, dtype="<f4", count=expected.size, offset=4 + header_size)
    np.testing.assert_array_equal(restored.reshape(expected.shape), expected)
    assert header["depth_source"] == "test"
