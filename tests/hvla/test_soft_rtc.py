"""Soft RTC (arXiv:2605.25537) — weights, training tensors, and sampler.

Soft RTC generalises the training-time RTC we already had: the binary mask
1[j < d] becomes continuous weights w_j. The generalisation is only safe if the
degenerate setting is *exactly* the old behaviour, so the tests below do not
assert that claim, they enumerate it — the old construction is reimplemented
here and compared tensor-for-tensor against the new one across many delays.

``test_soft_len_zero_*`` are the equivalence tests. If Soft RTC is later removed
or defaulted differently, they are what stops the change from silently altering
existing checkpoints' training or sampling.
"""

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import (
    FlowMatchingS1Model,
    rtc_prefix_weights,
)

CHUNK = 12


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d", list(range(0, CHUNK + 1)))
def test_soft_len_zero_weights_are_the_binary_mask(d):
    """L=0 must reproduce 1[j < d] exactly, for every delay including the ends."""
    w = rtc_prefix_weights(torch.tensor([d]), CHUNK, soft_len=0, soft_hmax=8)
    expected = (torch.arange(CHUNK) < d).float().unsqueeze(0)
    assert torch.equal(w, expected), f"d={d}: {w} != {expected}"


def test_dropped_prefix_disables_conditioning_entirely():
    """d=0 is the 'first chunk of an episode' case: ordinary flow matching."""
    w = rtc_prefix_weights(torch.tensor([0]), CHUNK, soft_len=4, soft_hmax=8)
    assert torch.equal(w, torch.zeros(1, CHUNK))


@pytest.mark.parametrize("d,L", [(1, 2), (2, 2), (2, 4), (3, 3), (5, 4)])
def test_soft_window_is_a_monotone_ramp_strictly_inside_zero_one(d, L):
    w = rtc_prefix_weights(torch.tensor([d]), CHUNK, soft_len=L, soft_hmax=CHUNK)[0]
    e = min(d + L, CHUNK)

    assert torch.equal(w[:d], torch.ones(d)), "committed prefix must be fully clamped"
    assert torch.equal(w[e:], torch.zeros(CHUNK - e)), "free tail must be unconditioned"

    win = w[d:e]
    assert (win > 0).all() and (win < 1).all(), (
        "soft tokens must be strictly editable and strictly informed; an endpoint "
        "of exactly 1 would zero their loss weight, which is the defect Soft RTC exists to fix"
    )
    assert torch.all(win[:-1] >= win[1:]), "schedule must be monotone decreasing"


def test_soft_window_is_capped_by_hmax():
    w = rtc_prefix_weights(torch.tensor([2]), CHUNK, soft_len=100, soft_hmax=5)[0]
    assert torch.equal(w[5:], torch.zeros(CHUNK - 5))


def test_weights_are_batched_per_sample():
    delays = torch.tensor([0, 1, 3])
    w = rtc_prefix_weights(delays, CHUNK, soft_len=2, soft_hmax=8)
    assert w.shape == (3, CHUNK)
    assert torch.equal(w[0], torch.zeros(CHUNK))
    assert w[1, 0] == 1.0 and w[2, 2] == 1.0


# --------------------------------------------------------------------------
# training tensors: new construction vs the original one
# --------------------------------------------------------------------------


def _legacy_training_tensors(actions, noise, t_flow, delays):
    """The pre-Soft-RTC construction, verbatim in behaviour.

    x_t noised with a scalar per-sample time, then prefix positions overwritten
    with clean actions, their timestep forced to 0, and their loss weight zeroed.
    """
    B, T, _ = actions.shape
    t_expand = t_flow[:, None, None]
    x_t = t_expand * noise + (1 - t_expand) * actions
    per_pos_t = t_flow[:, None].expand(B, T).clone()
    loss_mask = torch.ones(B, T, 1)
    for b in range(B):
        d = int(delays[b])
        if d > 0:
            x_t[b, :d] = actions[b, :d]
            per_pos_t[b, :d] = 0.0
            loss_mask[b, :d] = 0.0
    return x_t, per_pos_t, loss_mask


def _soft_training_tensors(actions, noise, t_flow, delays, soft_len, soft_hmax):
    """The Soft RTC construction as implemented in FlowMatchingS1Model.forward."""
    B, T, _ = actions.shape
    omega = rtc_prefix_weights(delays, T, soft_len, soft_hmax)
    per_pos_t = (1.0 - omega) * t_flow[:, None]
    t_expand = per_pos_t[..., None]
    x_t = t_expand * noise + (1 - t_expand) * actions
    loss_mask = (1.0 - omega).unsqueeze(-1)
    return x_t, per_pos_t, loss_mask


def test_soft_len_zero_reproduces_legacy_training_tensors():
    """Enumerate delays 0..8 against the original construction, bit for bit."""
    torch.manual_seed(0)
    B, T, A = 9, CHUNK, 7
    actions = torch.randn(B, T, A)
    noise = torch.randn(B, T, A)
    t_flow = torch.rand(B)
    delays = torch.arange(B)  # 0..8, covers dropout and the full delay range

    old = _legacy_training_tensors(actions, noise, t_flow, delays)
    new = _soft_training_tensors(actions, noise, t_flow, delays, 0, 8)

    for name, a, b in zip(("x_t", "per_pos_t", "loss_mask"), old, new, strict=True):
        assert torch.allclose(a, b, atol=0, rtol=0), f"{name} differs at soft_len=0"


def test_soft_window_tokens_are_trainable_unlike_hard_rtc():
    """The point of the change: position d carries loss weight under Soft RTC.

    Under Hard RTC the weight at the first executed action is exactly 0 on the
    samples where the delay covers it, so nothing ever teaches the model what to
    emit there given a prefix.
    """
    delays = torch.tensor([2])
    hard = rtc_prefix_weights(delays, CHUNK, 0, 8)
    soft = rtc_prefix_weights(delays, CHUNK, 3, 8)

    assert (1.0 - hard)[0, 1] == 0.0, "hard: last committed token is untrainable (expected)"
    assert (1.0 - hard)[0, 2] == 1.0, "hard: first executed token jumps straight to full weight"
    assert 0.0 < float((1.0 - soft)[0, 2]) < 1.0, "soft: first executed token is partly informed"


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _tiny_model(device, soft_len=0):
    config = FlowMatchingS1Config(
        action_dim=8,
        robot_state_feature=True,
        state_dim=8,
        chunk_size=CHUNK,
        hidden_dim=64,
        num_heads=4,
        num_decoder_layers=2,
        num_encoder_layers=1,
        dim_feedforward=128,
        num_inference_steps=5,
        rtc_soft_len=soft_len,
        rtc_soft_hmax=8,
    )
    return FlowMatchingS1Model(config).to(device).eval()


def _batch(device, B=1):
    return {
        "observation.images": [torch.randn(B, 3, 224, 224, device=device) for _ in range(4)],
        "observation.state": torch.zeros(B, 8, device=device),
        "observation.s2_latent": torch.zeros(B, 2048, device=device),
        "observation.s2_latent_age": torch.zeros(B, 1, device=device),
    }


@torch.no_grad()
def _legacy_sample(model, batch, prefix, num_steps):
    """The original sampler: inject before the loop, re-inject after every step."""
    device = next(model.parameters()).device
    B = prefix.shape[0]
    T = model.config.chunk_size
    context = model.encode_observations(batch)
    cached_kv = model.precompute_cross_attn_kv(context)
    x_t = torch.randn(B, T, model.config.action_dim, device=device)
    D = min(prefix.shape[1], T - 1)
    x_t[:, :D] = prefix[:, :D]
    dt = -1.0 / num_steps
    for _i in range(num_steps):
        t_val = 1.0 + _i * dt
        per_pos_t = torch.full((B, T), t_val, device=device)
        per_pos_t[:, :D] = 0.0
        v = model.denoise_step(x_t, context, per_pos_t, cached_kv=cached_kv)
        x_t = x_t + dt * v
        x_t[:, :D] = prefix[:, :D]
    return x_t


def test_soft_len_zero_sampler_matches_legacy_loop(device):
    """Same seed, same prefix: the rewritten sampler must be numerically identical."""
    model = _tiny_model(device, soft_len=0)
    batch = _batch(device)
    prefix = torch.randn(1, 3, 8, device=device)
    steps = 5

    torch.manual_seed(7)
    ref = _legacy_sample(model, batch, prefix, steps)
    torch.manual_seed(7)
    got = model.sample_actions(batch, num_steps=steps, action_prefix=prefix, prefix_len=3)

    assert torch.allclose(ref, got, atol=1e-6), f"max abs diff {(ref - got).abs().max().item():.3e}"


def test_committed_prefix_is_returned_exactly(device):
    """chunk[0:D] == prefix is relied on by the trace tooling; keep it true."""
    for soft_len in (0, 3):
        model = _tiny_model(device, soft_len=soft_len)
        batch = _batch(device)
        prefix = torch.randn(1, 6, 8, device=device)
        torch.manual_seed(3)
        out = model.sample_actions(batch, num_steps=4, action_prefix=prefix, prefix_len=2)
        assert torch.allclose(out[:, :2], prefix[:, :2], atol=1e-6), (
            f"soft_len={soft_len}: committed tokens must come back untouched"
        )


def test_soft_window_output_is_not_the_prior(device):
    """Soft tokens must be edited, not copied — otherwise it is just a longer pin."""
    model = _tiny_model(device, soft_len=3)
    batch = _batch(device)
    prefix = torch.randn(1, 6, 8, device=device)
    torch.manual_seed(3)
    out = model.sample_actions(batch, num_steps=4, action_prefix=prefix, prefix_len=2)
    assert not torch.allclose(out[:, 2:5], prefix[:, 2:5], atol=1e-4), (
        "soft-window tokens were returned unchanged; the blend is not being applied"
    )


def test_no_prefix_path_is_unaffected(device):
    """Soft RTC must not perturb generation when there is no prefix at all."""
    model = _tiny_model(device, soft_len=3)
    batch = _batch(device)
    torch.manual_seed(11)
    a = model.sample_actions(batch, num_steps=4)
    torch.manual_seed(11)
    b = model.sample_actions(batch, num_steps=4, action_prefix=None, prefix_len=0)
    assert torch.allclose(a, b, atol=0, rtol=0)
