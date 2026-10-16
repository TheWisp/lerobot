"""Reference caching saves feature work without caching a live pose decision."""

from types import SimpleNamespace

import cv2
import numpy as np

from tests.gui.test_stream_objects import worker  # noqa: F401


def test_reference_cache_invalidates_mask_intrinsics_and_files(tmp_path, monkeypatch, worker):  # noqa: F811
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    rgb = tmp_path / "rgb/000001.jpg"
    depth = tmp_path / "depth/000001.png"
    cv2.imwrite(str(rgb), np.zeros((4, 4, 3), np.uint8))
    cv2.imwrite(str(depth), np.ones((4, 4), np.uint16))
    tier = SimpleNamespace()
    intr = worker.CameraIntrinsics(600, 600, 2, 2)
    ref = {"recording": str(tmp_path), "frame": 1, "mask": np.ones((4, 4), bool)}
    cards = []

    def card(*args):
        result = object()
        cards.append(result)
        return result

    monkeypatch.setattr(worker, "Card", card)
    first = worker._reference_card(ref, tier, intr)
    assert worker._reference_card(ref, tier, intr) is first
    assert len(cards) == 1
    ref["mask"][0, 0] = False
    assert worker._reference_card(ref, tier, intr) is not first
    changed = worker._reference_card(ref, tier, worker.CameraIntrinsics(601, 600, 2, 2))
    assert len(cards) == 3
    cv2.imwrite(str(depth), np.full((4, 4), 2, np.uint16))
    assert worker._reference_card(ref, tier, intr) is not changed
    assert len(cards) == 4
    monkeypatch.setenv("LEROBOT_REFERENCE_CACHE", "0")
    worker._reference_card(ref, tier, intr)
    worker._reference_card(ref, tier, intr)
    assert len(cards) == 6
