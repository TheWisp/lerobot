"""Unit tests for the policy-internal overlays (cross-attention + input-gradient saliency).

Covers the pure pieces, no GPU / no model weights needed:
  - SharedAuxBuffer round-trip (the cross-process seam)
  - PolicyAttention / PolicySaliency adapters (transparent without a publisher; heatmap with one)
  - the S1 capture linchpins: recomputed softmax == SDPA's weights, and per-camera slicing
  - compute_input_saliency's no-image-features guard (the full grad pass is GPU/weights-bound,
    verified live, not in CI).
"""

import numpy as np
import torch

from lerobot.overlays.adapters import ADAPTERS, PolicyAttentionAdapter, PolicySaliencyAdapter
from lerobot.overlays.aux_ipc import SharedAuxBuffer
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Model, FlowMatchingS1Policy


def test_aux_buffer_roundtrip():
    w = SharedAuxBuffer(cameras={"front": (16, 16), "wrist": (16, 16)}, model="policy_attention", create=True)
    try:
        assert w.saliency_seq("front") == 0
        g = np.random.rand(16, 16).astype(np.float32)
        w.write_saliency("front", g)
        r = SharedAuxBuffer(create=False)
        try:
            assert set(r.cameras) == {"front", "wrist"}
            assert r.saliency_seq("front") == 1
            assert r.read_saliency("wrist") is None  # never written
            grid, _ts = r.read_saliency("front")
            assert grid.shape == (16, 16)
            assert np.allclose(grid, g, atol=1e-6)
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
    a = PolicyAttentionAdapter(device="cpu")
    a.set_camera("front")
    out = a.infer(np.zeros((48, 64, 3), np.uint8))
    assert out.shape == (48, 64, 4)
    assert out.sum() == 0  # no aux published (e.g. the data tab) -> fully transparent


def test_adapter_heatmap_with_publisher():
    w = SharedAuxBuffer(cameras={"front": (16, 16)}, model="policy_attention", create=True)
    a = PolicyAttentionAdapter(device="cpu")
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
    a = PolicyAttentionAdapter(device="cpu")
    try:
        w.write_saliency("front", np.zeros((16, 16), np.float32))  # all-zero -> peak 0
        a.set_camera("front")
        out = a.infer(np.zeros((48, 64, 3), np.uint8))
        assert out.sum() == 0
    finally:
        a.reset()
        w.cleanup()


def test_softmax_recompute_matches_sdpa():
    # The capture recomputes softmax(q·kᵀ/√d); it must equal the weights SDPA actually used,
    # so the heatmap is the policy's real attention, not an approximation of it.
    torch.manual_seed(0)
    b, h, t, n, d = 1, 4, 3, 7, 8
    q, k, v = torch.randn(b, h, t, d), torch.randn(b, h, n, d), torch.randn(b, h, n, d)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    aw = torch.softmax((q @ k.transpose(-2, -1)) * (d**-0.5), dim=-1)
    assert torch.allclose(aw @ v, sdpa, atol=1e-5)


def test_attention_grids_slicing():
    # Cameras occupy the FIRST n_cams*patches context positions; extras (state, s2) follow.
    stub = FlowMatchingS1Model.__new__(FlowMatchingS1Model)
    stub._ctx_layout = {"n_cams": 2, "patches_per_cam": 4}  # 2x2 grid per camera
    cam0, cam1, extras = [0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [99.0, 98.0]
    stub._last_attn_ctx = torch.tensor([cam0 + cam1 + extras])
    grids = FlowMatchingS1Model._attention_grids(stub)
    assert grids is not None and len(grids) == 2
    assert np.array_equal(grids[0], np.array([[0, 1], [2, 3]], np.float32))
    assert np.array_equal(grids[1], np.array([[10, 11], [12, 13]], np.float32))


def test_attention_grids_guards():
    stub = FlowMatchingS1Model.__new__(FlowMatchingS1Model)
    stub._last_attn_ctx, stub._ctx_layout = None, None
    assert FlowMatchingS1Model._attention_grids(stub) is None  # nothing captured
    stub._last_attn_ctx = torch.zeros(1, 7)
    stub._ctx_layout = {"n_cams": 1, "patches_per_cam": 5}  # 5 is not a square -> bail, don't guess
    assert FlowMatchingS1Model._attention_grids(stub) is None


# ---- input-gradient saliency overlay (the gradient method that replaced attention as the default) ----


def test_saliency_adapter_registered_and_coexists():
    # Both policy-internal overlays stay registered (attention preserved for A/B; saliency is default).
    assert {"policy_attention", "policy_saliency"} <= set(ADAPTERS)
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
