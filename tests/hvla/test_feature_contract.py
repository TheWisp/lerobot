"""Dataset-derived tensor layout tests for HVLA Flow S1."""

import numpy as np
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Model
from lerobot.policies.hvla.s1.flow_matching.train import configure_from_dataset_features
from lerobot.policies.hvla.s1_process import _resolve_policy_feature_order, obs_to_s1_batch


def test_training_layout_comes_from_dataset_metadata():
    config = FlowMatchingS1Config()
    features = {
        "action": {"dtype": "float32", "shape": [3], "names": ["a", "b", "c"]},
        "observation.state": {
            "dtype": "float32",
            "shape": [5],
            "names": ["a.pos", "a.vel", "b.pos", "b.vel", "c.pos"],
        },
        "observation.images.custom": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }

    configure_from_dataset_features(config, features, resize_to=(192, 256))

    assert config.action_dim == 3
    assert config.robot_state_feature is True
    assert config.state_dim == 5
    assert list(config.image_features) == ["observation.images.custom"]
    assert config.image_resize_shape == (192, 256)


def test_training_layout_records_absent_state():
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


def test_stateless_model_omits_state_projection():
    config = FlowMatchingS1Config(
        action_dim=2,
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

    assert FlowMatchingS1Model(config).state_proj is None


def test_runtime_state_dimension_can_differ_from_action_dimension():
    runtime_state_names = ["motor.pos", "motor.vel", "motor.torque"]
    state_names = _resolve_policy_feature_order(
        checkpoint_names=[],
        runtime_names=runtime_state_names,
        expected_dim=3,
        feature_kind="state",
        allow_unnamed_runtime_order=True,
    )
    observation = {
        "motor.pos": 2.0,
        "motor.vel": 3.0,
        "motor.torque": 4.0,
        "custom": np.zeros((8, 12, 3), dtype=np.uint8),
    }

    batch = obs_to_s1_batch(
        observation,
        s1_image_keys=["observation.images.custom"],
        shared_cache=None,
        s2_latent_key="observation.s2_latent",  # gitleaks:allow
        device=torch.device("cpu"),
        joint_names=["motor.pos"],
        state_feature_names=state_names,
    )

    assert batch["observation.state"].shape == (1, 3)
    assert batch["observation.state"].tolist() == [[2.0, 3.0, 4.0]]
