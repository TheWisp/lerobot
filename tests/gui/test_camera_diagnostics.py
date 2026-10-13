"""Camera diagnostics use the existing owner and leave the event loop free."""

import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from lerobot.gui.api import showservo as api


def test_capture_saves_paired_arrays_and_sensor_metadata_off_event_loop(monkeypatch, tmp_path):
    main_thread = threading.get_ident()

    def capture(**kwargs):
        assert threading.get_ident() != main_thread
        assert kwargs["frames"] == 1
        return [
            {
                "rgb": np.full((2, 2, 3), 42, np.uint8),
                "aligned_raw_depth": np.full((2, 2), 413),
                "filtered_depth": np.full((2, 2), 405),
                "color_metadata": {"timestamp_ms": 100},
                "depth_metadata": {"timestamp_ms": 99},
            }
        ]

    cam = SimpleNamespace(
        capture_diagnostics=capture,
        color_intrinsics=lambda: {"fx": 100},
        rotation=None,
        enable_decimation=False,
        enable_spatial=True,
        enable_temporal=True,
        enable_hole_filling=False,
    )
    monkeypatch.setattr(api, "_session", api._Session(tmp_path, cam))
    result = asyncio.run(api.camera_diagnostics(api.CameraDiagnosticsBody(frames=1)))
    from pathlib import Path

    root = Path(result["path"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["filters"]["enable_temporal"]
    assert manifest["frames"][0]["depth_metadata"]["timestamp_ms"] == 99
    with np.load(root / "000000.npz") as data:
        assert (data["aligned_raw_depth"] == 413).all()
        assert (data["filtered_depth"] == 405).all()


def test_no_camera_does_not_open_one(monkeypatch):
    monkeypatch.setattr(api, "_session", None)
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.camera_diagnostics(api.CameraDiagnosticsBody()))
    assert error.value.status_code == 409
