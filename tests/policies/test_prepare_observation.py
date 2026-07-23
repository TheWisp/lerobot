"""Equivalence tests for the CUDA fast path in prepare_observation_for_inference.

The fast path (uint8 images + CUDA device) uploads raw uint8 and converts on
GPU; the default path converts to float32 and permutes on CPU. Both must
produce bit-identical tensors. See src/lerobot/policies/utils.py.
"""

import numpy as np
import pytest
import torch

from lerobot.policies.utils import prepare_observation_for_inference


def _reference_prepare(observation: dict, device: torch.device) -> dict:
    """The pre-optimization implementation, kept verbatim as the oracle."""
    observation = dict(observation)
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            if observation[name].dtype == torch.uint8:
                observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)
    observation["task"] = ""
    observation["robot_type"] = ""
    return observation


def _make_obs(shapes=((720, 2560, 3), (600, 960, 3)), seed=0):
    rng = np.random.default_rng(seed)
    obs = {
        "observation.images.top": rng.integers(0, 256, shapes[0], dtype=np.uint8),
        "observation.images.cam": rng.integers(0, 256, shapes[1], dtype=np.uint8),
        "observation.state": rng.random(16).astype(np.float32),
    }
    return obs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the fast path")
@pytest.mark.parametrize("shapes", [((720, 2560, 3), (600, 960, 3)), ((64, 64, 1), (48, 96, 3))])
def test_cuda_fast_path_bit_identical(shapes):
    """Fast path output matches the CPU reference within 1 ulp (~6e-8).

    CUDA float32 division can round differently from CPU division by up to
    1 ulp per pixel; everything else is exact. atol=1e-7 covers it.
    """
    obs = _make_obs(shapes)
    device = torch.device("cuda")
    ref = _reference_prepare(obs, device)
    new = prepare_observation_for_inference(dict(obs), device)
    for key in ref:
        if isinstance(ref[key], torch.Tensor):
            assert torch.allclose(ref[key], new[key], atol=1e-7, rtol=0), key
            assert ref[key].is_contiguous() == new[key].is_contiguous(), key
            assert ref[key].shape == new[key].shape, key
            assert ref[key].dtype == new[key].dtype, key


def test_cpu_device_uses_default_path():
    """On CPU the fast path must not engage — output equals the reference."""
    obs = _make_obs()
    device = torch.device("cpu")
    ref = _reference_prepare(obs, device)
    new = prepare_observation_for_inference(dict(obs), device)
    for key in ref:
        if isinstance(ref[key], torch.Tensor):
            assert torch.equal(ref[key], new[key]), key


def test_non_uint8_images_unchanged():
    """Float images never take the fast path (they are not uint8)."""
    rng = np.random.default_rng(1)
    obs = {"observation.images.cam": rng.random((48, 64, 3)).astype(np.float32)}
    device = torch.device("cpu")
    ref = _reference_prepare(obs, device)
    new = prepare_observation_for_inference(dict(obs), device)
    assert torch.equal(ref["observation.images.cam"], new["observation.images.cam"])
