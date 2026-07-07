"""Unit tests for the policy-internal overlays (cross-attention + input-gradient saliency).

Covers the pure pieces, no GPU / no model weights needed:
  - SharedAuxBuffer round-trip (the cross-process seam)
  - PolicyAttention / PolicySaliency adapters (transparent without a publisher; heatmap with one)
  - the S1 capture linchpins: recomputed softmax == SDPA's weights, and per-camera slicing
  - compute_input_saliency's no-image-features guard (the full grad pass is GPU/weights-bound,
    verified live, not in CI).
"""

import time

import numpy as np

from lerobot.overlays.adapters import ADAPTERS, PolicySaliencyAdapter
from lerobot.overlays.aux_ipc import SharedAuxBuffer
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy


def test_aux_buffer_roundtrip():
    w = SharedAuxBuffer(cameras={"front": (16, 16), "wrist": (16, 16)}, model="policy_saliency", create=True)
    try:
        assert w.saliency_seq("front") == 0
        g = np.random.rand(16, 16).astype(np.float32)
        w.write_saliency("front", g)
        assert w.read_pass_ms() is None  # never published
        w.write_pass_ms(46.5)
        r = SharedAuxBuffer(create=False)
        try:
            assert set(r.cameras) == {"front", "wrist"}
            assert r.saliency_seq("front") == 1
            assert r.read_saliency("wrist") is None  # never written
            grid, _ts = r.read_saliency("front")
            assert grid.shape == (16, 16)
            assert np.allclose(grid, g, atol=1e-6)
            ms, ts = r.read_pass_ms()  # the policy-side cost, readable cross-process with freshness
            assert abs(ms - 46.5) < 1e-3
            assert time.time() - ts < 5.0
        finally:
            r.cleanup()
    finally:
        w.cleanup()


def test_aux_buffer_bad_write_is_noop():
    w = SharedAuxBuffer(cameras={"front": (16, 16)}, create=True)
    try:
        w.write_saliency("front", np.zeros((8, 8), np.float32))  # wrong shape -> ignored
        w.write_saliency("nope", np.zeros((16, 16), np.float32))  # unknown camera -> ignored
        assert w.saliency_seq("front") == 0
    finally:
        w.cleanup()


def test_adapter_transparent_without_publisher():
    a = PolicySaliencyAdapter(device="cpu")
    a.set_camera("front")
    out = a.infer(np.zeros((48, 64, 3), np.uint8))
    assert out.shape == (48, 64, 4)
    assert out.sum() == 0  # no aux published (e.g. the data tab) -> fully transparent


def test_adapter_heatmap_with_publisher():
    w = SharedAuxBuffer(cameras={"front": (16, 16)}, model="policy_saliency", create=True)
    a = PolicySaliencyAdapter(device="cpu")
    try:
        grid = np.full((16, 16), 0.01, np.float32)
        grid[4, 6] = 1.0  # single attention peak
        w.write_saliency("front", grid)
        a.set_camera("front")
        out = a.infer(np.zeros((48, 64, 3), np.uint8))
        assert out.shape == (48, 64, 4)
        assert out[..., 3].max() > 100  # the peak is rendered opaque
        # alpha peak lands near the upscaled grid peak: col ~6/16*64=24, row ~4/16*48=12
        yx = np.unravel_index(int(np.argmax(out[..., 3])), out[..., 3].shape)
        assert abs(yx[0] - 12) < 10 and abs(yx[1] - 24) < 12
    finally:
        a.reset()
        w.cleanup()


def test_adapter_flat_grid_transparent():
    w = SharedAuxBuffer(cameras={"front": (16, 16)}, create=True)
    a = PolicySaliencyAdapter(device="cpu")
    try:
        w.write_saliency("front", np.zeros((16, 16), np.float32))  # all-zero -> peak 0
        a.set_camera("front")
        out = a.infer(np.zeros((48, 64, 3), np.uint8))
        assert out.sum() == 0
    finally:
        a.reset()
        w.cleanup()


def test_saliency_adapter_registered_and_coexists():
    # Both policy-internal overlays stay registered (attention preserved for A/B; saliency is default).
    assert "policy_saliency" in ADAPTERS
    assert ADAPTERS["policy_saliency"] is PolicySaliencyAdapter
    # Runtime-switchable render styles; default shows the cool (blue) end, inferno preserved as an option.
    assert PolicySaliencyAdapter.DEFAULT_STYLE == "blue_yellow"
    assert PolicySaliencyAdapter.DEFAULT_STYLE in PolicySaliencyAdapter.STYLES
    assert "inferno" in PolicySaliencyAdapter.STYLES


def test_saliency_style_switch_changes_render():
    # set_control({'style': ...}) switches the look at runtime; gated 'spotlight' leaves the cool
    # background clear, 'heatmap' tints it — so heatmap has strictly more opaque pixels.
    w = SharedAuxBuffer(cameras={"front": (64, 64)}, model="policy_saliency", create=True)
    a = PolicySaliencyAdapter(device="cpu")
    try:
        grid = np.full((64, 64), 0.05, np.float32)
        grid[20:40, 20:40] = 1.0
        w.write_saliency("front", grid)
        a.set_camera("front")
        a.set_control({"style": "spotlight"})  # gated: cool background fully transparent
        spot = a.infer(np.zeros((96, 128, 3), np.uint8))
        a.set_control({"style": "heatmap"})  # full: cool background tinted too
        heat = a.infer(np.zeros((96, 128, 3), np.uint8))
        assert (heat[..., 3] > 0).sum() > (spot[..., 3] > 0).sum()
        a.set_control({"style": "bogus"})  # unknown -> unchanged (idempotent)
        assert a._style == "heatmap"
    finally:
        a.reset()
        w.cleanup()


def test_saliency_smooth_control():
    # The smoothing slider rides set_control({'smooth': sigma}); clamps to >=0, ignores garbage.
    a = PolicySaliencyAdapter(device="cpu")
    assert a._smooth == PolicySaliencyAdapter.SMOOTH_SIGMA  # instance default from the class knob
    a.set_control({"smooth": 0.0})
    assert a._smooth == 0.0
    a.set_control({"smooth": 2.5})
    assert a._smooth == 2.5
    a.set_control({"smooth": -1})  # negative -> clamped to 0
    assert a._smooth == 0.0
    a.set_control({"smooth": "bad"})  # non-numeric -> ignored (unchanged)
    assert a._smooth == 0.0


def test_overlay_control_reader():
    # The policy reads the GUI-selected saliency method from the worker's control block; None when
    # no overlay worker is up (which also serves as a demand signal).
    from lerobot.overlays.overlay_ipc import OverlayControlReader, SharedOverlayBuffer

    assert OverlayControlReader().config() is None  # no worker -> None
    w = SharedOverlayBuffer(cameras={"front": (4, 4)}, model="policy_saliency", create=True)
    try:
        w.write_control({"config": {"method": "rollout", "style": "cividis"}})
        assert OverlayControlReader().config() == {"method": "rollout", "style": "cividis"}
    finally:
        w.cleanup()


def test_compute_attention_rollout_empty_without_image_features():
    # The cheap guard; the full rollout needs DINOv2 weights + GPU (verified offline, not in CI).
    stub = FlowMatchingS1Policy.__new__(FlowMatchingS1Policy)
    stub.config = type("C", (), {"image_features": []})()
    assert stub.compute_attention_rollout({}) == {}


def test_saliency_adapter_heatmap_64():
    # The saliency grid rides the same aux seam at a higher resolution (64x64, not 16x16).
    w = SharedAuxBuffer(cameras={"front": (64, 64)}, model="policy_saliency", create=True)
    a = PolicySaliencyAdapter(device="cpu")
    try:
        grid = np.full((64, 64), 0.001, np.float32)
        grid[20:24, 30:34] = 1.0  # a localized saliency blob
        w.write_saliency("front", grid)
        a.set_camera("front")
        out = a.infer(np.zeros((96, 128, 3), np.uint8))
        assert out.shape == (96, 128, 4)
        assert out[..., 3].max() > 100  # the blob renders opaque
    finally:
        a.reset()
        w.cleanup()


def test_saliency_adapter_transparent_without_publisher():
    a = PolicySaliencyAdapter(device="cpu")
    a.set_camera("front")
    out = a.infer(np.zeros((48, 64, 3), np.uint8))
    assert out.shape == (48, 64, 4) and out.sum() == 0  # no live policy (e.g. data tab) -> nothing drawn


def test_compute_input_saliency_empty_without_image_features():
    # The cheap guard (returns {} before any forward), testable without a backbone. The full grad
    # pass needs DINOv2 weights + GPU and is verified live, not in CI.
    stub = FlowMatchingS1Policy.__new__(FlowMatchingS1Policy)
    stub.config = type("C", (), {"image_features": []})()
    assert stub.compute_input_saliency({}) == {}


def test_publish_aux_demand_gated():
    """Regression (the always-on cost): `SaliencyPublisher.publish` must NOT compute saliency when no
    overlay worker is up (the OverlayControlReader.config() demand signal is None), and MUST compute
    when one is."""
    from unittest.mock import MagicMock

    from lerobot.overlays.saliency_publisher import SaliencyPublisher

    policy = MagicMock(spec=["compute_input_saliency"])  # no .model -> no capture_attention touch
    policy.compute_input_saliency.return_value = {}
    # empty image_keys -> compute returns {} -> _publish_saliency early-returns cleanly
    pub = SaliencyPublisher(policy, [], mode="saliency", every=1)
    pub._ctrl = MagicMock()

    # No overlay worker up -> demand off -> saliency must NOT be computed.
    pub._ctrl.config.return_value = None
    pub.publish({})
    pub.publish({})
    policy.compute_input_saliency.assert_not_called()

    # Overlay up -> demand on -> computed.
    pub._ctrl.config.return_value = {"method": "gradient"}
    pub.publish({})
    policy.compute_input_saliency.assert_called_once()


def test_overlay_control_reader_is_reliable_demand_signal():
    """config() is the demand signal: None with no worker, the written config while one is up, and None
    again after the control segment is unlinked (clean stop) — never a stale latch."""
    import os

    import numpy as np

    from lerobot.overlays.overlay_ipc import (
        _CONTROL_BYTES,
        _CONTROL_SHM,
        _PREFIX,
        OverlayControlReader,
        _write_json,
    )
    from lerobot.policies.hvla.ipc import SharedBlock

    if os.path.exists(_CONTROL_SHM):
        os.remove(_CONTROL_SHM)  # a stale worker's segment would defeat the "no worker" assert
    r = OverlayControlReader()
    assert r.config() is None  # no worker
    blk = SharedBlock(name=_PREFIX + "control", shape=(_CONTROL_BYTES,), dtype=np.uint8, create=True)
    try:
        _write_json(blk, {"config": {"method": "rollout"}})
        assert r.config() == {"method": "rollout"}  # worker up -> demand on
    finally:
        blk.unlink()  # clean stop unlinks the segment
    assert r.config() is None  # reliably off after stop, not a stale "rollout"
