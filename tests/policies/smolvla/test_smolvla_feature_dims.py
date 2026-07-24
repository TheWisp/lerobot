import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.utils.constants import ACTION, OBS_STATE


def test_feature_validation_grows_padding_widths_from_dataset():
    config = SmolVLAConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(48,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(40,)),
        },
    )

    config.validate_features()

    assert config.max_state_dim == 48
    assert config.max_action_dim == 40


def test_feature_validation_keeps_legacy_width_as_floor():
    config = SmolVLAConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )

    config.validate_features()

    assert config.max_state_dim == 32
    assert config.max_action_dim == 32


def test_pad_vector_rejects_truncation():
    with pytest.raises(ValueError, match="smaller dimension"):
        pad_vector(torch.zeros(2, 48), 32)
