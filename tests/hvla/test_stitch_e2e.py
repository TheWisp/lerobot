"""Drive the real inference thread and check a non-zero offset is published.

The plumbing tests cover parameter storage and index arithmetic. This one runs
the actual loop: publish an observation, let it produce a chunk, advance the
execution index, publish again so the second inference has a real RTC prefix,
and assert the thread computed and exposed a stitch offset.

Without this, the offset computation inside _loop is untested — and that is the
exact shape of the three wiring defects already found in this work.
"""

import time

import numpy as np
import torch

from lerobot.policies.hvla.s1_inference import InferenceThread

from .test_inference_thread import _SO107_JOINTS, MockRTCPolicy, MockSharedCache, _make_obs


class RampRTCPolicy(MockRTCPolicy):
    """chunk[k] = k on every joint, so the continuation index is predictable.

    ``latency_s`` matters: the RTC delay is derived from measured inference time,
    and a mock that returns instantly yields d=1, at which point there are not
    two prefix rows to derive a direction from and stitching correctly declines.
    The rig measures ~2.5 frames, so the default here simulates that.
    """

    def __init__(self, chunk_size=50, action_dim=14, latency_s=0.09):
        super().__init__(chunk_size=chunk_size, action_dim=action_dim)
        self._latency_s = latency_s

    def predict_action_chunk(self, batch, num_steps=None, prefix_len=None):
        if self._latency_s:
            time.sleep(self._latency_s)
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


def _pump(thread, n=10, dt=0.05):
    """Feed observations and let the loop produce successive chunks."""
    t0 = time.perf_counter()
    for i in range(n):
        thread.publish_obs(_make_obs(), t0 + i * dt, frame_index=i)
        time.sleep(dt)
        thread.update_exec_index(min(i, 4))


def test_disabled_publishes_a_zero_offset_end_to_end():
    t = _thread()
    t.start()
    try:
        _pump(t)
        assert t.stitch_offset == 0
    finally:
        t.stop()


def test_enabled_publishes_a_stitch_offset_end_to_end():
    """The loop must actually compute an offset, not leave it at the default."""
    t = _thread(rtc_stitch_search=12)
    t.start()
    try:
        _pump(t)
        chunk, _origin, _obs = t.get_chunk()
        assert chunk is not None, "no chunk produced; the loop never ran"
        off = t.stitch_offset
        assert isinstance(off, int)
        assert off >= 0, f"offset must not resume before the prefix, got {off}"
        # On a ramp the continuation is genuinely ahead of the prefix, so a
        # working search should move off zero at least once.
        assert off > 0, (
            "the loop published a zero offset with search enabled — the in-loop "
            "computation is not running"
        )
    finally:
        t.stop()


def test_direction_filter_off_still_publishes_an_offset():
    t = _thread(rtc_stitch_search=12, rtc_stitch_direction=False)
    t.start()
    try:
        _pump(t)
        assert t.stitch_offset >= 0
    finally:
        t.stop()


def test_a_one_frame_delay_disables_stitching():
    """Known limitation, pinned deliberately.

    The resume direction is prefix[d-1] - prefix[d-2], so a single committed
    frame gives nothing to continue and the offset stays 0. That is correct, but
    it means stitching switches itself off if inference ever gets fast enough to
    round the delay down to one frame — silently, with no other symptom.
    """
    t = _thread(policy=RampRTCPolicy(latency_s=0.0), rtc_stitch_search=12)
    t.start()
    try:
        _pump(t)
        assert t._chunk_prefix_len <= 1, "this test only means something at d<=1"
        assert t.stitch_offset == 0
    finally:
        t.stop()
