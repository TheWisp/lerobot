"""The pipeline's picture stages: resize to the profile, blend the overlay.

Written on tensors without a device in them, so the same code runs on CPU
tensors in CI and on the GPU on the host; the tests run on both when both
exist and pin that they agree.
"""

import numpy as np
import pytest
import torch

from lerobot.gui.live_video.stages import blend_overlay, resize_to_width

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _frame(h: int, w: int, device: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=g).to(device)


def _overlay(h: int, w: int, device: str, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (h, w, 4), dtype=torch.uint8, generator=g).to(device)


@pytest.mark.parametrize("device", DEVICES)
class TestResize:
    def test_keeps_the_aspect_ratio_and_lands_on_even_sizes(self, device):
        out = resize_to_width(_frame(720, 1280, device), 320)
        assert tuple(out.shape) == (180, 320, 3)
        assert out.dtype == torch.uint8
        assert out.device.type == device
        out = resize_to_width(_frame(600, 960, device), 320)
        assert tuple(out.shape) == (200, 320, 3)

    def test_an_odd_height_rounds_to_even(self, device):
        out = resize_to_width(_frame(601, 960, device), 320)
        assert out.shape[0] % 2 == 0
        assert out.shape[1] == 320

    def test_never_upscales(self, device):
        f = _frame(120, 200, device)
        out = resize_to_width(f, 320)
        assert out.shape == f.shape
        assert torch.equal(out, f)

    def test_averages_the_source(self, device):
        # A checkerboard of 0 and 255 downscaled by two is mid-grey everywhere.
        f = torch.zeros(400, 640, 3, dtype=torch.uint8)
        f[0::2, 1::2] = 255
        f[1::2, 0::2] = 255
        out = resize_to_width(f.to(device), 320)
        assert int(out.min()) >= 126
        assert int(out.max()) <= 129

    def test_does_not_modify_its_input(self, device):
        f = _frame(600, 960, device)
        before = f.clone()
        resize_to_width(f, 320)
        assert torch.equal(f, before)


@pytest.mark.parametrize("device", DEVICES)
class TestBlend:
    def test_a_transparent_overlay_changes_nothing(self, device):
        f = _frame(200, 320, device)
        ov = _overlay(200, 320, device)
        ov[..., 3] = 0
        assert torch.equal(blend_overlay(f, ov), f)

    def test_an_opaque_overlay_replaces_the_frame(self, device):
        f = _frame(200, 320, device)
        ov = _overlay(200, 320, device)
        ov[..., 3] = 255
        assert torch.equal(blend_overlay(f, ov), ov[..., :3])

    def test_matches_the_float_reference_within_one_level(self, device):
        f = _frame(200, 320, device)
        ov = _overlay(200, 320, device)
        out = blend_overlay(f, ov).cpu().numpy().astype(np.int32)
        fn = f.cpu().numpy().astype(np.float64)
        on = ov.cpu().numpy().astype(np.float64)
        a = on[..., 3:4] / 255.0
        ref = np.rint(fn * (1.0 - a) + on[..., :3] * a).astype(np.int32)
        assert np.abs(out - ref).max() <= 1

    def test_an_overlay_of_another_size_is_resized_to_the_frame(self, device):
        # The adapter draws at the source size; the blend happens at the
        # profile's. A box over the left half at 720p must land on the left
        # half at 320 wide.
        f = torch.full((180, 320, 3), 40, dtype=torch.uint8, device=device)
        ov = torch.zeros(720, 1280, 4, dtype=torch.uint8, device=device)
        ov[:, :640, :3] = 200
        ov[:, :640, 3] = 255
        out = blend_overlay(f, ov)
        assert torch.equal(out[:, :150], torch.full((180, 150, 3), 200, dtype=torch.uint8, device=device))
        assert torch.equal(out[:, 170:], torch.full((180, 150, 3), 40, dtype=torch.uint8, device=device))

    def test_does_not_modify_its_inputs(self, device):
        f = _frame(200, 320, device)
        ov = _overlay(200, 320, device)
        fb, ob = f.clone(), ov.clone()
        blend_overlay(f, ov)
        assert torch.equal(f, fb)
        assert torch.equal(ov, ob)

    def test_a_partial_alpha_lands_between_the_two(self, device):
        f = torch.full((8, 8, 3), 0, dtype=torch.uint8, device=device)
        ov = torch.full((8, 8, 4), 200, dtype=torch.uint8, device=device)
        ov[..., 3] = 128
        out = blend_overlay(f, ov)
        assert 99 <= int(out.min()) <= int(out.max()) <= 101


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs both devices")
def test_cpu_and_cuda_agree_within_one_level():
    f = _frame(600, 960, "cpu")
    ov = _overlay(600, 960, "cpu")
    cpu = blend_overlay(resize_to_width(f, 320), ov)
    gpu = blend_overlay(resize_to_width(f.cuda(), 320), ov.cuda()).cpu()
    assert cpu.shape == gpu.shape
    assert (cpu.int() - gpu.int()).abs().max() <= 1
