"""State dropout on the real SmolVLA prefix path, without downloading model weights."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")),
]


class TinyVLM(nn.Module):
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        return tokens[..., None].expand(-1, -1, 8).float()


def prefix_model(device):
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(state_dropout_p=0.0)
    model.vlm_with_expert = TinyVLM()
    model.state_proj = nn.Linear(24, 8)
    # Nonzero bias verifies masking happens AFTER projection.
    nn.init.constant_(model.state_proj.bias, 2.0)
    model.add_image_special_tokens = False
    model.prefix_length = -1
    return model.to(device)


def inputs(device, n=4096):
    return (
        [torch.randn(n, 2, 8, device=device) for _ in range(2)],
        [torch.ones(n, device=device, dtype=torch.bool) for _ in range(2)],
        torch.ones(n, 3, device=device, dtype=torch.long),
        torch.ones(n, 3, device=device, dtype=torch.bool),
        torch.randn(n, 24, device=device, requires_grad=True),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_prefix_masks_only_projected_state_and_preserves_inputs_and_gradients(device):
    model = prefix_model(device).train()
    args = inputs(device)
    before = args[-1].detach().clone()
    expected, pad, att = model.embed_prefix(*args)
    model.config.state_dropout_p = 0.2
    actual, actual_pad, actual_att = model.embed_prefix(*args)
    dropped = actual[:, -1].eq(0).all(-1)
    assert 0.17 < dropped.float().mean().item() < 0.23
    assert torch.equal(actual[:, :-1], expected[:, :-1])  # images + language untouched
    assert torch.equal(actual[~dropped], expected[~dropped])  # no 1/(1-p) rescaling
    assert torch.equal(actual_pad, pad) and torch.equal(actual_att, att)
    assert torch.equal(args[-1], before)
    actual.sum().backward()
    assert args[-1].grad[dropped].eq(0).all()
    assert args[-1].grad[~dropped].abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_and_eval_preserve_output_and_random_stream(device):
    model = prefix_model(device)
    args = inputs(device, 4)
    expected = model.embed_prefix(*args)
    for training, p in [(True, 0.0), (False, 0.2), (False, 1.0)]:
        model.train(training)
        model.config.state_dropout_p = p
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        actual = model.embed_prefix(*args)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


@pytest.mark.parametrize("device", DEVICES)
def test_full_dropout_removes_projection_bias_without_adding_weights(device):
    model = prefix_model(device).train()
    keys = set(model.state_dict())
    args = inputs(device, 4)
    model.config.state_dropout_p = 1.0
    result = model.embed_prefix(*args)[0]
    assert result[:, -1].eq(0).all()
    assert result[:, :-1].abs().sum() > 0
    assert keys == set(model.state_dict())


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf"), True, "0.2"])
def test_reject_invalid_probability(bad):
    with pytest.raises(ValueError, match="state_dropout_p"):
        SmolVLAConfig(state_dropout_p=bad)


def test_probability_roundtrip_and_old_config_defaults_off(tmp_path):
    cfg = SmolVLAConfig(state_dropout_p=0.2)
    cfg.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).state_dropout_p == 0.2
    path = tmp_path / "config.json"
    legacy = json.loads(path.read_text())
    legacy.pop("state_dropout_p")
    path.write_text(json.dumps(legacy))
    assert PreTrainedConfig.from_pretrained(tmp_path).state_dropout_p == 0.0
