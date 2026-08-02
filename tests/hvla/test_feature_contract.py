"""Robot-agnostic feature contract tests for HVLA Flow S1."""

import numpy as np
import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Model
from lerobot.policies.hvla.s1.flow_matching.train import configure_from_dataset_features
from lerobot.policies.hvla.s1_process import _resolve_policy_feature_order, obs_to_s1_batch


def test_training_contract_comes_from_dataset_metadata():
    features = {
        "action": {
            "dtype": "float32",
            "shape": [3],
            "names": ["base.turn", "arm.lift", "tool.close"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [5],
            "names": ["base.turn", "base.speed", "arm.lift", "arm.load", "tool.close"],
        },
        "observation.images.overhead_custom": {
            "dtype": "video",
            "shape": [721, 1283, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.tool_custom": {
            "dtype": "image",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    config = FlowMatchingS1Config()

    configure_from_dataset_features(config, features, resize_to=(192, 256))

    assert config.action_dim == 3
    assert config.action_feature_names == ["base.turn", "arm.lift", "tool.close"]
    assert config.state_dim == 5
    assert config.state_feature_names == [
        "base.turn",
        "base.speed",
        "arm.lift",
        "arm.load",
        "tool.close",
    ]
    assert list(config.image_features) == [
        "observation.images.overhead_custom",
        "observation.images.tool_custom",
    ]
    assert config.image_resize_shape == (192, 256)


def test_training_contract_rejects_dataset_without_images():
    config = FlowMatchingS1Config()
    features = {
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
    }

    with pytest.raises(ValueError, match="at least one visual feature"):
        configure_from_dataset_features(config, features, resize_to=(224, 224))


def test_training_contract_records_absent_robot_state():
    config = FlowMatchingS1Config()
    features = {
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "observation.images.custom": {
            "dtype": "video",
            "shape": [120, 160, 3],
            "names": ["height", "width", "channels"],
        },
    }

    configure_from_dataset_features(config, features, resize_to=(112, 112))

    assert config.robot_state_feature is False
    assert config.state_dim == 0
    assert config.state_feature_names == []


def test_named_checkpoint_retains_training_order():
    resolved = _resolve_policy_feature_order(
        checkpoint_names=["tool.close", "base.turn"],
        runtime_names=["base.turn", "arm.lift", "tool.close"],
        expected_dim=2,
        feature_kind="action",
    )
    assert resolved == ["tool.close", "base.turn"]


def test_flow_checkpoint_refuses_unnamed_runtime_layout_even_when_dimensions_match():
    with pytest.raises(ValueError, match="runtime order is not a safe substitute"):
        _resolve_policy_feature_order(
            checkpoint_names=[],
            runtime_names=["a", "b"],
            expected_dim=2,
            feature_kind="state",
        )


def test_policy_without_name_metadata_may_explicitly_use_matching_runtime_order():
    resolved = _resolve_policy_feature_order(
        checkpoint_names=[],
        runtime_names=["a", "b"],
        expected_dim=2,
        feature_kind="state",
        allow_unnamed_runtime_order=True,
    )
    assert resolved == ["a", "b"]


def test_training_contract_rejects_unnamed_actions():
    config = FlowMatchingS1Config()
    features = {
        "action": {"dtype": "float32", "shape": [2], "names": None},
        "observation.images.custom": {
            "dtype": "video",
            "shape": [120, 160, 3],
            "names": ["height", "width", "channels"],
        },
    }

    with pytest.raises(ValueError, match="one ordered name per value"):
        configure_from_dataset_features(config, features, resize_to=(224, 224))


def test_model_contract_has_no_robot_dimension_defaults():
    config = FlowMatchingS1Config()

    assert config.action_dim is None
    assert config.robot_state_feature is None
    assert config.state_dim is None
    with pytest.raises(ValueError, match="action_dim must be resolved"):
        config.validate_feature_contract()


def test_stateless_model_does_not_create_an_untrained_state_projection():
    config = FlowMatchingS1Config(
        action_dim=2,
        action_feature_names=["a", "b"],
        robot_state_feature=False,
        state_dim=0,
        use_dino_backbone=False,
        hidden_dim=16,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        s2_latent_dim=8,
        s2_proj_hidden=4,
    )

    model = FlowMatchingS1Model(config)

    assert model.state_proj is None


def test_runtime_state_can_differ_from_action_layout():
    obs = {
        "motor.pos": 2.0,
        "motor.vel": 3.0,
        "motor.torque": 4.0,
        "custom_view": np.zeros((8, 12, 3), dtype=np.uint8),
    }

    batch = obs_to_s1_batch(
        obs,
        s1_image_keys=["observation.images.custom_view"],
        shared_cache=None,
        s2_latent_key="observation.s2_latent",  # gitleaks:allow
        device=torch.device("cpu"),
        joint_names=["motor.pos"],
        state_feature_names=["motor.pos", "motor.vel", "motor.torque"],
    )

    assert batch["observation.state"].shape == (1, 3)
    assert batch["observation.state"].tolist() == [[2.0, 3.0, 4.0]]


def test_runtime_omits_state_for_stateless_checkpoint():
    batch = obs_to_s1_batch(
        {"custom_view": np.zeros((8, 12, 3), dtype=np.uint8)},
        s1_image_keys=["observation.images.custom_view"],
        shared_cache=None,
        s2_latent_key="observation.s2_latent",  # gitleaks:allow
        device=torch.device("cpu"),
        joint_names=["motor.pos"],
        state_feature_names=[],
    )

    assert "observation.state" not in batch
