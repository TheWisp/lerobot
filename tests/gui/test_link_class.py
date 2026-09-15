"""The one link constant both low-bandwidth paths derive from.

The Data tab's chunk profile and the Run tab's live stream size themselves
against the same class link, so a change to it moves both tabs together.
These pin the relations the design states, not the numbers: the numbers are
the constant's to change.
"""

import dataclasses

import pytest

from lerobot.gui.link_class import CLASS_LINK, LinkClass


def test_the_class_is_a_named_public_preset():
    # A preset someone else defined, so a reader can look it up; a downlink
    # faster than the uplink, and a round trip, as any mobile class has.
    assert CLASS_LINK.name
    assert CLASS_LINK.down_kbit_s > CLASS_LINK.up_kbit_s > 0
    assert CLASS_LINK.rtt_ms > 0


def test_the_stream_budget_leaves_a_reserve_of_the_downlink():
    budget = CLASS_LINK.stream_budget_kbit_s
    assert 0 < budget < CLASS_LINK.down_kbit_s
    # The reserve is what the data channel, retransmissions and the rest of
    # the page get; the design keeps about a quarter.
    assert CLASS_LINK.down_kbit_s - budget >= CLASS_LINK.down_kbit_s // 5


def test_per_camera_bitrate_splits_the_budget_evenly():
    for cameras in (1, 2, 3, 4, 6):
        per = CLASS_LINK.per_camera_kbit_s(cameras)
        assert per * cameras <= CLASS_LINK.stream_budget_kbit_s
        assert per * cameras > CLASS_LINK.stream_budget_kbit_s - cameras


def test_more_cameras_means_a_smaller_share():
    shares = [CLASS_LINK.per_camera_kbit_s(n) for n in (1, 2, 4)]
    assert shares == sorted(shares, reverse=True)
    assert shares[0] > shares[-1]


def test_no_cameras_is_an_error():
    with pytest.raises(ValueError):
        CLASS_LINK.per_camera_kbit_s(0)


def test_a_link_class_cannot_be_edited_in_place():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CLASS_LINK.down_kbit_s = 1  # type: ignore[misc]


def test_another_class_derives_the_same_way():
    faster = LinkClass(name="test", down_kbit_s=CLASS_LINK.down_kbit_s * 2, up_kbit_s=1, rtt_ms=1)
    assert faster.stream_budget_kbit_s == CLASS_LINK.stream_budget_kbit_s * 2
    assert faster.per_camera_kbit_s(4) == CLASS_LINK.per_camera_kbit_s(4) * 2


def test_the_encoder_is_asked_for_less_than_a_camera_may_send():
    """Rate control lands above its target, and what has to fit the link is
    what it sends: asking for the whole share puts the stream over it."""
    for cameras in (1, 2, 4):
        target = CLASS_LINK.encoder_target_kbit_s(cameras)
        allowed = CLASS_LINK.per_camera_kbit_s(cameras)
        assert 0 < target < allowed
        # The headroom has to cover a real overshoot, not a rounding error.
        assert target <= allowed * 0.95


def test_both_low_bandwidth_paths_scale_to_the_same_width():
    """The Data tab's recorded chunks and the Run tab's live stream are the
    same profile seen twice. Two literals would let one be re-tuned and the
    other left behind, and the tabs would disagree about what Low Bandwidth
    means."""
    from lerobot.gui.api.chunk_playback import PROFILES
    from lerobot.gui.link_class import PROFILE_WIDTH
    from lerobot.gui.live_video import pipeline

    assert PROFILES["low"]["width"] == PROFILE_WIDTH
    assert pipeline.PROFILE_WIDTH == PROFILE_WIDTH
