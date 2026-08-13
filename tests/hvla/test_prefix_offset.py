"""The RTC prefix must be sliced from where the robot is actually executing.

Rig evidence: with stitching on, the executor ran index ~4.5 while the prefix
was sliced from ~2.0 on 76% of inferences — the model was conditioned on actions
that were never sent, and the stitched rollout reversed at 64% instead of the
predicted 15%. The offset the executor applies must therefore also shift the
prefix start.

Mutation check for this file: dropping `+ stitch_offset` from
_rtc_prefix_for_observation must fail test_offset_shifts_the_prefix_start.
"""

import numpy as np

from lerobot.policies.hvla.s1_inference import _rtc_prefix_for_observation

FPS = 30
CHUNK = np.arange(50 * 4, dtype=np.float32).reshape(50, 4)


def call(elapsed_frames, offset, **kw):
    args = dict(
        old_chunk=CHUNK,
        old_chunk_origin=0.0,
        observation_time=elapsed_frames / FPS,
        estimated_delay_s=2 / FPS,
        fps=FPS,
        max_delay=6,
        stitch_offset=offset,
    )
    args.update(kw)
    return _rtc_prefix_for_observation(**args)


def test_zero_offset_is_the_historical_behaviour():
    start, delay, prefix = call(5, 0)
    assert start == 5
    assert np.array_equal(prefix[0], CHUNK[5])


def test_offset_shifts_the_prefix_start():
    """The regression: executor at time+offset, prefix must start there too."""
    start, _delay, prefix = call(5, 3)
    assert start == 8, f"prefix start {start}; executor runs index 8, so 8 is required"
    assert np.array_equal(prefix[0], CHUNK[8])


def test_shifted_start_is_clamped_to_the_chunk():
    start, delay, prefix = call(48, 30)
    assert start == len(CHUNK) - 1
    assert delay >= 1


def test_negative_offset_is_clamped_at_zero():
    start, _delay, _prefix = call(1, -10)
    assert start == 0


def test_delay_still_respects_the_chunk_end_after_shifting():
    start, delay, prefix = call(40, 8)
    assert start + delay <= len(CHUNK)
    assert len(prefix) >= delay
