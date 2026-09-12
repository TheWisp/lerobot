"""Optional HVLA state dropout preserves the existing path when disabled."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy
from lerobot.policies.hvla.s1.flow_matching.state_dropout import StateTokenDropout
from lerobot.policies.hvla.s1.flow_matching.train import (
    build_arg_parser,
    checkpoint_config_dict,
    validate_resume_training_contract,
)

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")),
]


def config(**overrides):
    return FlowMatchingS1Config(
        **(
            {
                "chunk_size": 4,
                "action_dim": 2,
                "state_dim": 2,
                "action_feature_names": ["a.pos", "b.pos"],
                "state_feature_names": ["a.pos", "b.pos"],
                "robot_state_feature": True,
                "hidden_dim": 16,
                "num_heads": 2,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "dim_feedforward": 32,
                "s2_latent_dim": 8,
                "s2_proj_hidden": 16,
                "use_dino_backbone": False,
                "image_features": {},
                "num_inference_steps": 2,
                "rtc_max_delay": 0,
            }
            | overrides
        )
    )


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_and_eval_are_identity_without_rng_consumption(device):
    token = torch.ones(32, 1, 16, device=device)
    for p, training in [(0.0, True), (0.2, False), (1.0, False)]:
        module = StateTokenDropout(p).train(training)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        assert module(token) is token
        assert module.rng_state() is None
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        if gpu_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p", [0.2, 1.0])
def test_drop_is_per_sample_whole_token_unscaled_and_not_inplace(device, p):
    torch.manual_seed(1337)
    token = torch.linspace(1, 2, 4096 * 16, device=device).reshape(4096, 1, 16).requires_grad_()
    before = token.detach().clone()
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    result = StateTokenDropout(p)(token)
    dropped = (result == 0).all(dim=(1, 2))
    assert 0.17 < float(dropped.float().mean()) < 0.23 if p == 0.2 else bool(dropped.all())
    assert torch.equal(result[~dropped], before[~dropped])
    assert torch.equal(token, before)
    result.sum().backward()
    assert torch.equal(token.grad[dropped], torch.zeros_like(token.grad[dropped]))
    assert torch.equal(token.grad[~dropped], torch.ones_like(token.grad[~dropped]))
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if gpu_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)


@pytest.mark.parametrize("device", DEVICES)
def test_rng_survives_training_state_serialization_without_model_weight_keys(device):
    token = torch.ones(256, 1, 16, device=device)
    module = StateTokenDropout(0.2)
    module(token)
    checkpoint = io.BytesIO()
    torch.save({"state_dropout_rng_state": module.rng_state()}, checkpoint)
    expected = module(token)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True, map_location=device)
    restored = StateTokenDropout(0.2)
    restored.restore_rng_state(saved["state_dropout_rng_state"], token.device)
    assert torch.equal(restored(token), expected)
    assert restored.state_dict() == {}
    assert list(restored.parameters()) == []
    StateTokenDropout().restore_rng_state(None, token.device)  # legacy checkpoint


@pytest.mark.parametrize("device", DEVICES)
def test_disabled_policy_matches_legacy_identity_path_in_training_and_inference(device):
    torch.manual_seed(1337)
    policy = FlowMatchingS1Policy(config()).to(device)
    legacy = FlowMatchingS1Policy(config()).to(device)
    legacy.model.state_dropout = nn.Identity()
    legacy.load_state_dict(policy.state_dict(), strict=True)
    assert policy.state_dict().keys() == legacy.state_dict().keys()
    batch = {
        "observation.state": torch.randn(8, 2, device=device),
        "action": torch.randn(8, 4, 2, device=device),
    }
    for training in [True, False]:
        policy.train(training)
        legacy.train(training)
        torch.manual_seed(72)
        expected, _ = legacy(batch)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(72)
        actual, _ = policy(batch)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_rng)
    restored_cfg = FlowMatchingS1Config.from_checkpoint_dict(
        checkpoint_config_dict(config(state_dropout_p=1.0))
    )
    restored = FlowMatchingS1Policy(restored_cfg).to(device).eval()
    restored.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        torch.manual_seed(72)
        expected = legacy.predict_action_chunk(batch)
        torch.manual_seed(72)
        actual = restored.predict_action_chunk(batch)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_only_state_token_changes_with_three_images_and_aux_off(device):
    from lerobot.policies.hvla.s1.protocol import S2_LATENT_KEY

    class PatchBackbone(nn.Module):
        def forward_features(self, images):
            return {"x_norm_patchtokens": images.mean(dim=(-1, -2)).unsqueeze(1)}

    policy = FlowMatchingS1Policy(config()).to(device).train()
    model = policy.model
    model.backbone = PatchBackbone()
    model.image_proj = nn.Linear(3, 16).to(device)
    model._backbone_grad_ckpt = False
    model.obs_encoder = nn.Identity()  # inspect the actual tokens delivered to the encoder
    keys = ["observation.images.top_l", "observation.images.right_wrist", "observation.images.ball_view"]
    model.config.image_features = dict.fromkeys(keys, (3, 8, 8))
    batch = {k: torch.rand(128, 3, 8, 8, device=device) for k in keys}
    batch.update(
        {
            "observation.state": torch.randn(128, 2, device=device),
            "action": torch.randn(128, 4, 2, device=device),
            S2_LATENT_KEY: torch.randn(128, 8, device=device),
        }
    )
    snapshot = {k: v.clone() for k, v in batch.items()}
    expected = model.encode_observations(batch)
    for p in [0.2, 1.0]:
        model.state_dropout = StateTokenDropout(p)
        actual = model.encode_observations(batch)
        assert actual.shape == (128, 5, 16)
        assert torch.equal(actual[:, :3], expected[:, :3])  # both cameras and masked image
        assert torch.equal(actual[:, 4:], expected[:, 4:])  # S2, if supplied, unchanged
        dropped = (actual[:, 3] == 0).all(dim=1)
        assert dropped.any()
        assert dropped.all() if p == 1 else (~dropped).any()
        assert torch.equal(actual[~dropped, 3], expected[~dropped, 3])
        for key in snapshot:
            assert torch.equal(batch[key], snapshot[key])
    assert model.ball_aux_head is None
    assert model.ball_proj is None


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True, "0.2"])
def test_invalid_probability_refused_by_config_and_module(bad):
    with pytest.raises(ValueError, match="state_dropout_p"):
        config(state_dropout_p=bad).validate_feature_contract()
    with pytest.raises(ValueError, match="state_dropout_p"):
        StateTokenDropout(bad)


def test_dropout_requires_state_and_old_configs_default_to_off():
    with pytest.raises(ValueError, match="requires observation.state"):
        config(
            state_dropout_p=0.2, robot_state_feature=False, state_dim=0, state_feature_names=[]
        ).validate_feature_contract()
    cfg = config(state_dropout_p=0.2)
    saved = checkpoint_config_dict(cfg)
    assert FlowMatchingS1Config.from_checkpoint_dict(saved).state_dropout_p == 0.2
    validate_resume_training_contract(saved, cfg)
    with pytest.raises(ValueError, match="state_dropout_p"):
        validate_resume_training_contract(saved, config())
    saved.pop("state_dropout_p")
    assert FlowMatchingS1Config.from_checkpoint_dict(saved).state_dropout_p == 0
    validate_resume_training_contract(saved, config())


def test_cli_checkbox_shortcut_and_explicit_probability_are_unambiguous():
    parser = build_arg_parser()
    base = ["--dataset-repo-id", "test/local", "--output-dir", "/unused-test-output"]
    assert parser.parse_args(base).state_dropout_p == 0
    assert parser.parse_args(base + ["--state-dropout"]).state_dropout_p == 0.2
    assert parser.parse_args(base + ["--state-dropout-p", "0.35"]).state_dropout_p == 0.35
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--state-dropout", "--state-dropout-p", "0.35"])
