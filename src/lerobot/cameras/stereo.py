# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Side-by-side stereo frames, and the channel names they split into.

Stereo USB cameras (the ZED family among them) have no single-eye UVC mode:
every resolution they advertise is the two sensors concatenated along the width
of one frame. A ZED Mini's "2560x720" is two 1280x720 eyes. Consumed whole, the
frame carries the scene twice and halves the horizontal resolution that survives
into a policy's square encoder input.

Two places have to split such a frame identically — the live camera during
teleoperation and inference, and the offline transform that converts already
recorded datasets — so the operation and the resulting channel names live here
rather than being spelled out twice. A live rollout and the data it is compared
against then cannot disagree about which half is which.

Only the naming and the split are shared. The lifecycles are not: one owns a
V4L2 handle, the other decodes video files.
"""

from typing import Any, Literal

from numpy.typing import NDArray

__all__ = ["Eye", "EYES", "split_stereo_frame", "stereo_channel_key", "stereo_channel_keys"]

Eye = Literal["l", "r"]

#: Suffixes appended to a camera's key, one per eye, in left-then-right order.
#: Matches the existing ``{camera}_depth`` convention for a camera that
#: contributes more than one observation channel.
EYES: tuple[Eye, Eye] = ("l", "r")


def split_stereo_frame(frame: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Split a side-by-side stereo frame into its left and right eyes.

    Pre: ``frame`` is at least 2-D and its width (axis 1) is even.
    Post: returns two arrays of equal shape, each half the input width, in
    (left, right) order. Both are copies, so neither keeps the full frame
    alive — the live camera hands these to a reader thread that would
    otherwise pin twice the memory it appears to.
    """
    width = frame.shape[1]
    if width % 2:
        raise ValueError(f"a side-by-side stereo frame needs an even width, got {width}")
    half = width // 2
    return frame[:, :half].copy(), frame[:, half:].copy()


def stereo_channel_key(base: str, eye: Eye) -> str:
    """Name of one eye's channel for a camera named ``base``.

    Pre: ``eye`` is one of :data:`EYES`.

    >>> stereo_channel_key("top", "l")
    'top_l'
    """
    if eye not in EYES:
        raise ValueError(f"eye must be one of {EYES}, got {eye!r}")
    return f"{base}_{eye}"


def stereo_channel_keys(base: str) -> tuple[str, str]:
    """Both channel names for a camera named ``base``, in (left, right) order.

    >>> stereo_channel_keys("observation.images.top")
    ('observation.images.top_l', 'observation.images.top_r')
    """
    return stereo_channel_key(base, "l"), stereo_channel_key(base, "r")
