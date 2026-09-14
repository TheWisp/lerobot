"""The link the GUI's low-bandwidth paths are built for.

One value both tabs derive from: the Data tab's chunk profile and the Run
tab's live stream size themselves against it, the tests' emulated link runs
at it, and a change to it moves all of them together. It is a named public
preset rather than a measurement of ours, so a reader can look it up.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The pictures' share of the downlink; the rest is kept for the data channel,
#: retransmissions and the rest of the page.
STREAM_SHARE = 0.75

#: What an encoder is asked for, as a fraction of the camera's share. Rate
#: control aims at its target and lands a little above it, and what has to
#: fit the link is what it sends rather than what it was asked for.
ENCODER_HEADROOM = 0.9


@dataclass(frozen=True)
class LinkClass:
    name: str
    down_kbit_s: int
    up_kbit_s: int
    rtt_ms: int

    @property
    def stream_budget_kbit_s(self) -> int:
        """What all cameras together may use of the downlink."""
        return int(self.down_kbit_s * STREAM_SHARE)

    def per_camera_kbit_s(self, cameras: int) -> int:
        """The budget split evenly: what one camera may send.

        Precondition: at least one camera.
        """
        if cameras < 1:
            raise ValueError(f"a stream needs at least one camera, got {cameras}")
        return self.stream_budget_kbit_s // cameras

    def encoder_target_kbit_s(self, cameras: int) -> int:
        """What one camera's encoder is asked for, which is less than it may
        send: the overshoot is what would otherwise put the stream over."""
        return int(self.per_camera_kbit_s(cameras) * ENCODER_HEADROOM)


#: Lighthouse's and Chrome DevTools' "Slow 4G" preset, which Lighthouse
#: describes as the bottom quarter of 4G connections and the top quarter of 3G.
CLASS_LINK = LinkClass(name="Slow 4G", down_kbit_s=1600, up_kbit_s=750, rtt_ms=150)

#: The width every camera is scaled down to on the low-bandwidth paths, and
#: never up. It is what the Data tab's recorded chunks and the Run tab's live
#: stream both fit into the budget above, so it is one number: a change to it
#: has to be paid for in both places at once, which is only visible if there
#: is one place to change.
PROFILE_WIDTH = 320
