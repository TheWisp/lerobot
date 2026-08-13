"""The soft window must actually be soft — the test that would have caught it.

Every Soft RTC measurement in this investigation was void because
predict_action_chunk inferred the committed-prefix length from the prefix
tensor's row count. Handing it e(d) rows so the window had a prior caused all
of them to be hard-pinned, so chunk[d] came back exactly equal to prefix[d] and
the seam metric degenerated into "is the previous chunk continuous with itself".

The config still reported soft_len=2 throughout, so nothing looked wrong. The
only reliable signal is the returned chunk: under Soft RTC, chunk[d] must NOT
equal prefix[d], because position d is supposed to be editable.
"""

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Model

CHUNK = 12
D = 2
SOFT = 2


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _model(device, soft_len):
    cfg = FlowMatchingS1Config(
        action_dim=8,
        robot_state_feature=True,
        state_dim=8,
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
    return FlowMatchingS1Model(cfg).to(device).eval()


def _batch(device):
    return {
        "observation.images": [torch.randn(1, 3, 224, 224, device=device) for _ in range(4)],
        "observation.state": torch.zeros(1, 8, device=device),
        "observation.s2_latent": torch.zeros(1, 2048, device=device),
        "observation.s2_latent_age": torch.zeros(1, 1, device=device),
    }


def test_extra_prefix_rows_are_not_hard_pinned(device):
    """Supplying e(d) rows must blend positions [d, e), not commit them."""
    model = _model(device, SOFT)
    prefix = torch.randn(1, D + SOFT, 8, device=device)
    torch.manual_seed(5)
    out = model.sample_actions(_batch(device), num_steps=4, action_prefix=prefix, prefix_len=D)

    assert torch.allclose(out[:, :D], prefix[:, :D], atol=1e-6), "committed rows must be exact"
    for j in range(D, D + SOFT):
        assert not torch.allclose(out[:, j], prefix[:, j], atol=1e-4), (
            f"chunk[{j}] came back equal to the prior — the soft window was hard-pinned, "
            "which makes every seam metric measure the previous chunk against itself"
        )


def test_seam_position_is_editable_not_a_copy(device):
    """chunk[d] is the first executed action; it must be the model's, not a copy."""
    model = _model(device, SOFT)
    prefix = torch.randn(1, D + SOFT, 8, device=device)
    torch.manual_seed(5)
    out = model.sample_actions(_batch(device), num_steps=4, action_prefix=prefix, prefix_len=D)
    gap = (out[:, D] - prefix[:, D]).abs().max().item()
    assert gap > 1e-3, f"chunk[d] is within {gap:.2e} of the prior; it is a pin, not a prediction"


def test_hard_rtc_still_pins_every_supplied_row(device):
    """With soft_len=0 the old contract holds: all supplied rows are committed."""
    model = _model(device, 0)
    prefix = torch.randn(1, D, 8, device=device)
    torch.manual_seed(5)
    out = model.sample_actions(_batch(device), num_steps=4, action_prefix=prefix, prefix_len=D)
    assert torch.allclose(out[:, :D], prefix[:, :D], atol=1e-6)


def test_prefix_len_defaults_to_row_count(device):
    """Callers that never pass prefix_len keep their exact previous behaviour."""
    model = _model(device, 0)
    prefix = torch.randn(1, 3, 8, device=device)
    batch = _batch(device)
    torch.manual_seed(5)
    a = model.sample_actions(batch, num_steps=4, action_prefix=prefix, prefix_len=3)
    torch.manual_seed(5)
    b = model.sample_actions(batch, num_steps=4, action_prefix=prefix, prefix_len=3)
    assert torch.allclose(a, b, atol=0)


def test_soft_window_shrinks_to_the_rows_actually_supplied(device):
    """A caller that supplies only d rows gets Hard RTC, not a window over noise."""
    model = _model(device, SOFT)
    prefix = torch.randn(1, D, 8, device=device)  # no rows for the soft window
    torch.manual_seed(5)
    out = model.sample_actions(_batch(device), num_steps=4, action_prefix=prefix, prefix_len=D)
    assert torch.allclose(out[:, :D], prefix[:, :D], atol=1e-6), (
        "with no spare rows the window must collapse to the committed prefix rather "
        "than blend toward zero-padding"
    )
