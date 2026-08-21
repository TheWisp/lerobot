"""Two chunks stitched together through the real functions, end to end.

Every earlier test covered one component in isolation; the rig failure lived in
their composition (the executor's index was shifted, the prefix slicer's was
not). This test runs the pipeline arithmetic across a chunk boundary with the
production functions — real stitch search, real prefix slicer, real executor
index — and checks the executed VALUE stream, which is what the arm feels.

Scenario, a 1 deg/frame straight line on every joint:

    chunk A, obs at frame 0:  A[k] = k       plays unshifted (shift 0)
    chunk B, obs at frame 4:  B[k] = k + 1   the plan trails: its value for any
                                             instant sits later in the vector
    B arrives at frame 6 (d = 2)

Hard resume would send B[2] = 3 right after A[5] = 5 — a backward jump. The
stitch must find k0 = 5 (B[5] = 6 continues 4, 5, ...), the executor must then
play 6, 7, 8, ... seamlessly, and the NEXT prefix must be sliced from the
shifted index so it equals the actions actually sent — the exact invariant the
rig run broke.
"""

import numpy as np

from lerobot.policies.hvla.s1_inference import (
    _rtc_prefix_for_observation,
    choose_stitch_index,
)
from lerobot.policies.hvla.s1_process import _executed_chunk_index

FPS = 30
D_SECONDS = 2 / FPS  # estimated inference delay: 2 frames


def _line(values):
    """A chunk whose value at index k is values[k], identical on 3 joints."""
    return np.stack([np.full(3, float(v)) for v in values])


A = _line(range(12))  # A[k] = k
B = _line([v + 1 for v in range(12)])  # B[k] = k + 1


def _prefix_for_next_chunk(old_chunk, old_origin_frame, obs_frame, shift):
    return _rtc_prefix_for_observation(
        old_chunk=old_chunk,
        old_chunk_origin=old_origin_frame / FPS,
        observation_time=obs_frame / FPS,
        estimated_delay_s=D_SECONDS,
        fps=FPS,
        max_delay=6,
        stitch_offset=shift,
    )


def _played(chunk, origin_frame, frame, shift):
    """The value the executor sends at a wall-clock frame — production function."""
    idx = _executed_chunk_index(frame / FPS, origin_frame / FPS, FPS, len(chunk), shift)
    return float(chunk[idx][0])


def test_stitch_search_finds_the_continuation_index():
    start, d, prefix = _prefix_for_next_chunk(A, 0, obs_frame=4, shift=0)
    assert (start, d) == (4, 2)
    assert prefix[0][0] == 4.0 and prefix[1][0] == 5.0
    k0 = choose_stitch_index(B, prefix, d, search=8)
    # prefix ends at 5 moving +1/frame, so the continuation value is 6 = B[5]
    assert k0 == 5


def test_stitched_handoff_is_seamless_across_two_chunks():
    """The headline: the executed value stream has no seam at the boundary."""
    _start, d, prefix = _prefix_for_next_chunk(A, 0, obs_frame=4, shift=0)
    shift_b = choose_stitch_index(B, prefix, d, search=8) - d

    sent = [_played(A, 0, f, 0) for f in (4, 5)]  # during B's inference
    sent += [_played(B, 4, f, shift_b) for f in (6, 7, 8, 9)]  # after B arrives

    assert sent == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    steps = np.diff(sent)
    assert np.allclose(steps, 1.0), f"seam in the executed stream: {sent}"


def test_hard_resume_shows_the_jump_the_stitch_removes():
    """Discriminating control: with shift 0 the same scenario reverses."""
    sent = [_played(A, 0, f, 0) for f in (4, 5)]
    sent += [_played(B, 4, f, 0) for f in (6, 7)]
    assert sent == [4.0, 5.0, 3.0, 4.0]
    assert sent[2] < sent[1], "expected the backward jump the stitch exists to remove"


def test_next_prefix_equals_what_the_executor_actually_sends():
    """The rig regression, at two-chunk scale.

    Chunk C's observation is taken at frame 8 while B plays with shift 3. The
    prefix handed to the model must be the same rows the executor sends at
    frames 8 and 9 — B[7] and B[8] — not the unshifted B[4:6].
    """
    _start, d, prefix = _prefix_for_next_chunk(A, 0, obs_frame=4, shift=0)
    shift_b = choose_stitch_index(B, prefix, d, search=8) - d
    assert shift_b == 3

    start_c, d_c, prefix_c = _prefix_for_next_chunk(B, 4, obs_frame=8, shift=shift_b)
    assert (start_c, d_c) == (7, 2)

    executed_rows = np.stack(
        [B[_executed_chunk_index(f / FPS, 4 / FPS, FPS, len(B), shift_b)] for f in (8, 9)]
    )
    assert np.array_equal(prefix_c, executed_rows), (
        "the model would be conditioned on actions the robot does not send"
    )

    # The unshifted slice — the rig bug — must disagree, or this test proves nothing.
    _s, _d, wrong = _prefix_for_next_chunk(B, 4, obs_frame=8, shift=0)
    assert not np.array_equal(wrong, executed_rows)


def test_shift_zero_reproduces_the_historical_pipeline():
    """With stitching disabled the composed pipeline is the pre-stitch one."""
    _start, d, prefix = _prefix_for_next_chunk(A, 0, obs_frame=4, shift=0)
    assert np.array_equal(prefix, A[4:6])
    for f, expect in ((6, 3.0), (7, 4.0)):
        assert _played(B, 4, f, 0) == expect
