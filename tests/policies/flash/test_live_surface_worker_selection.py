"""Exercise picker commands through the running worker's actual frame loop."""

from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.policies.flash import live_surface_worker as worker
from lerobot.policies.flash.perception import RGBDFrame


@pytest.mark.parametrize("clicked_frame", [2, 3, 0], ids=["delayed-display", "current-display", "expired"])
def test_reselect_after_tracking_uses_processed_frame_and_bad_click_does_not_pause(
    tmp_path, monkeypatch, clicked_frame
):
    from lerobot.cameras import realsense
    from lerobot.gui import flash_live_scene
    from lerobot.overlays import adapters

    # Fake only hardware/model producers. Keep the worker loop, command routing,
    # frame cache, click validation, selection reset and packet encoding real.
    frames = {
        i: RGBDFrame(
            i,
            np.full((8, 10, 3), i, np.uint8),
            np.full((8, 10), 0.4 + i / 1000, np.float32),
            np.array([[10, 0, 5], [0, 10, 4], [0, 0, 1]], float),
            1.0,
            np.eye(4),
            float(i),
            float(i) + 0.1,
            "test",
        )
        for i in range(1, 5)
    }
    current = [1]
    commands = Queue()
    stop = Event()
    states, sessions, processed = [], [], []
    waits = [0]

    class BoundedStop:
        # A broken paused loop should fail an assertion rather than hang pytest.
        def is_set(self):
            return stop.is_set()

        def set(self):
            stop.set()

        def wait(self, timeout):
            waits[0] += 1
            if waits[0] > 10:
                stop.set()
            return stop.is_set()

    event_calls = [0]

    def event():
        event_calls[0] += 1
        return BoundedStop() if event_calls[0] == 1 else Event()

    intr = SimpleNamespace(fx=10, fy=10, ppx=5, ppy=4)
    profile = SimpleNamespace(
        get_intrinsics=lambda: intr,
        width=lambda: 10,
        height=lambda: 8,
        fps=lambda: 30,
        format=lambda: "test",
    )
    profile.as_video_stream_profile = lambda: profile
    camera = SimpleNamespace(
        is_connected=True,
        connect=lambda: None,
        disconnect=lambda: None,
        depth_processing={},
        rs_profile=SimpleNamespace(
            get_stream=lambda *_: profile,
            get_device=lambda: SimpleNamespace(
                first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001)
            ),
        ),
    )

    class DepthStream:
        error = None

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def snapshot(self):
            return frames[current[0]], float(current[0])

        def join(self, **kwargs):
            pass

    class Session:
        def __init__(self, settings, source, *args):
            self.mask_source = source
            self.closed = False
            sessions.append(self)

        def measure_masks(self, frame):
            assert not self.closed
            return frame.frame_id, {}

        def step(self, frame, **kwargs):
            assert not self.closed
            processed.append(frame)
            return {"frame": frame.frame_id, "objects": {}, "mask_ms": 0.0, "processing_ms": 0.1}, {}, {}

        def close(self):
            self.closed = True

    write_snapshot = worker.write_snapshot

    def snapshot(path, value):
        write_snapshot(path, value)
        if path.name != "state.json":
            return
        states.append(value)
        if value["status"] == "preview_model":
            commands.put({"id": "first", "action": "select", "point": [4, 4], "frame_id": 1})
        elif value["status"] == "tracking":
            if len(processed) == 3:
                commands.put({"id": "second", "action": "select", "point": [6, 5], "frame_id": clicked_frame})
            if len(processed) >= 4:
                stop.set()
            else:
                current[0] += 1

    monkeypatch.setattr(worker, "Event", event)
    monkeypatch.setattr(worker, "Queue", lambda **_: commands)
    monkeypatch.setattr(
        worker, "Thread", lambda **_: SimpleNamespace(start=lambda: None, join=lambda **_: None)
    )
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(worker, "_freeze_startup_heap", lambda: None)
    monkeypatch.setattr(worker, "make_depth_source", lambda _: object())
    monkeypatch.setattr(worker, "needs_stereo_pair", lambda _: False)
    monkeypatch.setattr(worker, "LiveDepthStream", DepthStream)
    monkeypatch.setattr(worker, "LiveSurfaceSession", Session)
    monkeypatch.setattr(worker, "write_snapshot", snapshot)
    monkeypatch.setattr(realsense, "RealSenseCamera", lambda _: camera)
    monkeypatch.setattr(
        adapters,
        "Sam3TrackByDetectionAdapter",
        lambda **_: SimpleNamespace(set_control=lambda _: None, reset=lambda: None),
    )
    monkeypatch.setattr(
        flash_live_scene,
        "live_scene_packet",
        lambda frame, live, row, *args, **kwargs: flash_live_scene.encode_rgbd_scene(frame, row),
    )
    worker.run(
        tmp_path,
        {
            "serial": "test",
            "width": 10,
            "height": 8,
            "fps": 30,
            "preview_fps": 15,
            "depth_source": "foundation_stereo",
            "pipeline_settings": {
                "config": {"compiled_surface_fit": False, "reuse_surface_search": False},
                "shared_matching": {"backend": "lk"},
            },
        },
    )
    errors = [s for s in states if s["status"] == "error"]
    assert len(processed) == 4, "A picker command must not freeze the existing processed stream"
    if clicked_frame:
        assert not errors
        assert len(sessions) == 2
        # The replacement gets precisely the shown RGB and MODEL depth, even
        # when the producer is newer. Never substitute the raw sensor depth.
        assert processed[-1] is frames[clicked_frame]
        assert sessions[-1].mask_source.seed_points_by_role == {"object": (6, 5)}
    else:
        assert len(errors) == 1 and "no longer available" in errors[0]["message"]
        assert len(sessions) == 1
        assert processed[-1] is frames[4]
    assert states[-1]["status"] == "tracking"
