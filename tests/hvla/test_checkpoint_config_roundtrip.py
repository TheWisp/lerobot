"""Every field that changes inference behaviour must survive a checkpoint.

``checkpoint_config_dict`` is an explicit allowlist, so adding a config field
without adding it there produces a checkpoint that loads fine and silently runs
with the default. That is worse than a crash: rtc_soft_len was trained at 2 and
reloaded at 0, so the weights were trained against a blended prior and then
sampled with a hard pin, and the only symptom was a quietly worse result.

A round trip over non-default values is the check that catches it — a key list
would just be restating the allowlist.
"""

import pytest

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.train import checkpoint_config_dict


def _valid_config(**overrides):
    """A config that satisfies the feature contract, so it can round-trip."""
    base = dict(
        action_dim=4,
        action_feature_names=["j0.pos", "j1.pos", "j2.pos", "j3.pos"],
        robot_state_feature=True,
        state_dim=4,
        state_feature_names=["j0.pos", "j1.pos", "j2.pos", "j3.pos"],
        image_features={"observation.images.top": 224},
    )
    base.update(overrides)
    return FlowMatchingS1Config(**base)


# Fields whose value changes what the sampler computes. A checkpoint that loses
# any of these produces different actions from the ones it was trained for.
INFERENCE_CRITICAL = {
    "chunk_size": 33,
    "hidden_dim": 128,
    "num_heads": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 5,
    "dim_feedforward": 256,
    "num_inference_steps": 7,
    "rtc_max_delay": 4,
    "rtc_drop_prob": 0.35,
    "rtc_soft_len": 3,
    "rtc_soft_hmax": 6,
    "state_position_std_floor": 0.25,
    "use_relative_actions": True,
}


@pytest.mark.parametrize("field,value", sorted(INFERENCE_CRITICAL.items()))
def test_inference_critical_field_is_persisted(field, value):
    cfg = _valid_config()
    setattr(cfg, field, value)
    dumped = checkpoint_config_dict(cfg)
    assert field in dumped, (
        f"{field} is not persisted by checkpoint_config_dict, so a checkpoint "
        f"trained with it silently reloads at the dataclass default"
    )
    assert dumped[field] == value


def test_soft_rtc_defaults_round_trip_as_hard_rtc():
    """An untouched config must still describe Hard RTC after a round trip."""
    cfg = _valid_config()
    dumped = checkpoint_config_dict(cfg)
    assert dumped["rtc_soft_len"] == 0
    restored = FlowMatchingS1Config.from_checkpoint_dict(dumped)
    assert restored.rtc_soft_len == 0
    assert restored.rtc_soft_hmax == cfg.rtc_soft_hmax


def test_soft_rtc_value_round_trips():
    """The regression that shipped: soft_len=2 must not come back as 0."""
    cfg = _valid_config(rtc_soft_len=2, rtc_soft_hmax=6)
    restored = FlowMatchingS1Config.from_checkpoint_dict(checkpoint_config_dict(cfg))
    assert (restored.rtc_soft_len, restored.rtc_soft_hmax) == (2, 6)


def test_a_checkpoint_without_soft_fields_loads_as_hard_rtc():
    """Checkpoints predating Soft RTC must keep their original behaviour."""
    dumped = checkpoint_config_dict(_valid_config())
    dumped.pop("rtc_soft_len")
    dumped.pop("rtc_soft_hmax")
    restored = FlowMatchingS1Config.from_checkpoint_dict(dumped)
    assert restored.rtc_soft_len == 0
