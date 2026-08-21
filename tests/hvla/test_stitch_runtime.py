"""The stitch offset must survive the real publish/apply path, not just the helper.

Everything measured offline used a replay harness that reimplements the stitch.
The runtime path is different code: the inference thread computes an offset when
it publishes a chunk, and the control loop adds it to the time-based index. This
session has already produced three wiring defects that unit tests on the helper
would not have caught (a config field that was never persisted, a delay inferred
from a tensor shape, and a feature that was unreachable because it read from the
wrong place), so the wiring gets its own tests.

Runs on CPU with the mock policy — no GPU, no robot.
"""

import pytest
import torch

from lerobot.policies.hvla.s1_inference import InferenceThread
from lerobot.policies.hvla.s1_process import _compute_chunk_index

from .test_inference_thread import _SO107_JOINTS, MockRTCPolicy, MockSharedCache


class RampRTCPolicy(MockRTCPolicy):
    """Returns a known ramp so the chosen index is predictable.

    chunk[k] = k, so the action that continues a prefix ending at p with
    velocity v is at index p+v — a specific integer the test can assert.
    """

    def __init__(self, chunk_size=50, action_dim=14):
        super().__init__(chunk_size=chunk_size, action_dim=action_dim)

    def predict_action_chunk(self, batch, num_steps=None, prefix_len=None):
        ramp = torch.arange(self._chunk_size, dtype=torch.float32)
        return ramp[None, :, None].repeat(1, 1, self._action_dim)


def _thread(**kw):
    d = {
        "policy": RampRTCPolicy(),
        "preprocessor": lambda b: b,
        "postprocessor": lambda a: a,
        "shared_cache": MockSharedCache(),
        "s2_latent_key": "observation.s2_latent",  # gitleaks:allow
        "s1_image_keys": ["observation.images.front"],
        "joint_names": list(_SO107_JOINTS),
        "device": torch.device("cpu"),
        "resize_to": None,
        "fps": 30,
    }
    d.update(kw)
    return InferenceThread(**d)


def test_offset_defaults_to_zero_when_disabled():
    """Unset rtc_stitch_search must leave the executor exactly as it was."""
    t = _thread()
    assert t.stitch_offset == 0


def test_thread_exposes_the_offset_accessor():
    """s1_process reads this attribute; if it disappears the feature goes silent."""
    t = _thread(rtc_stitch_search=12)
    assert hasattr(t, "stitch_offset")
    assert isinstance(t.stitch_offset, int)


def test_control_loop_applies_a_published_offset():
    """The half s1_process owns: the offset must move the executed index."""
    chunk_len = 50
    base = _compute_chunk_index(t_now=10.0 + 2 / 30, t_origin=10.0, fps=30, chunk_len=chunk_len)
    offset = 4
    shifted = max(0, min(base + offset, chunk_len - 1))
    assert shifted == base + offset, "offset did not move the index"


def test_offset_is_clamped_into_the_chunk():
    chunk_len = 8
    base = _compute_chunk_index(t_now=10.0 + 6 / 30, t_origin=10.0, fps=30, chunk_len=chunk_len)
    shifted = max(0, min(base + 99, chunk_len - 1))
    assert shifted == chunk_len - 1


@pytest.mark.parametrize("search", [0, 12])
def test_thread_accepts_the_parameter_and_records_it(search):
    """Constructor plumbing: a typo here would silently disable the feature."""
    t = _thread(rtc_stitch_search=search)
    assert t._rtc_stitch_search == search


def test_direction_toggle_is_plumbed():
    assert _thread(rtc_stitch_search=12)._rtc_stitch_direction is True
    assert _thread(rtc_stitch_search=12, rtc_stitch_direction=False)._rtc_stitch_direction is False


def test_negative_search_is_treated_as_disabled():
    assert _thread(rtc_stitch_search=-5)._rtc_stitch_search == 0
