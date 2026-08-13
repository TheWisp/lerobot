"""predict_action_chunk must honour an explicit committed-prefix length.

This is the caller-level test. tests/hvla/test_soft_window_is_live.py exercises
sample_actions, which was never wrong — it takes prefix_len as an argument. The
defect lived one layer up, where predict_action_chunk derived prefix_len from
the prefix tensor's row count, so a caller supplying e(d) rows for the soft
window had all of them hard-pinned.

Mutation check: restoring `prefix_len = action_prefix.shape[1]` in
predict_action_chunk must make test_extra_rows_are_blended_not_pinned fail.
"""

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy
from lerobot.policies.hvla.s1.protocol import ACTION_PREFIX_KEY

CHUNK = 12
D = 2
SOFT = 2


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _policy(device, soft_len):
    cfg = FlowMatchingS1Config(
        action_dim=8,
        action_feature_names=[f"j{i}.pos" for i in range(8)],
        robot_state_feature=True,
        state_dim=8,
        state_feature_names=[f"j{i}.pos" for i in range(8)],
        image_features={"observation.images.top": 224},
        chunk_size=CHUNK,
        hidden_dim=64,
        num_heads=4,
        num_decoder_layers=2,
        num_encoder_layers=1,
        dim_feedforward=128,
        num_inference_steps=4,
        rtc_soft_len=soft_len,
        rtc_soft_hmax=8,
    )
    return FlowMatchingS1Policy(cfg).to(device).eval()


def _batch(device, prefix):
    return {
        "observation.images.top": torch.randn(1, 3, 224, 224, device=device),
        "observation.state": torch.zeros(1, 8, device=device),
        ACTION_PREFIX_KEY: prefix,
    }


def test_extra_rows_are_blended_not_pinned(device):
    """The regression: e(d) rows in, only d of them committed."""
    policy = _policy(device, SOFT)
    prefix = torch.randn(1, D + SOFT, 8, device=device)
    torch.manual_seed(5)
    out = policy.predict_action_chunk(_batch(device, prefix), prefix_len=D)

    assert torch.allclose(out[:, :D], prefix[:, :D], atol=1e-5), "committed rows must be exact"
    for j in range(D, D + SOFT):
        assert not torch.allclose(out[:, j], prefix[:, j], atol=1e-4), (
            f"chunk[{j}] equals the prior — the soft window was hard-pinned, which makes "
            "every downstream seam metric compare the previous chunk with itself"
        )


def test_omitting_prefix_len_keeps_the_old_shape_derived_behaviour(device):
    """Existing callers that pass only a prefix are unaffected."""
    policy = _policy(device, 0)
    prefix = torch.randn(1, 3, 8, device=device)
    torch.manual_seed(5)
    out = policy.predict_action_chunk(_batch(device, prefix))
    assert torch.allclose(out[:, :3], prefix[:, :3], atol=1e-5)


def test_prefix_len_larger_than_supplied_rows_is_rejected(device):
    policy = _policy(device, SOFT)
    prefix = torch.randn(1, 2, 8, device=device)
    with pytest.raises(AssertionError):
        policy.predict_action_chunk(_batch(device, prefix), prefix_len=5)
