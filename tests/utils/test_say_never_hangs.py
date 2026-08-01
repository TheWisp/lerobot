"""A spoken cue must never decide whether a run terminates.

``say(blocking=True)`` ran ``spd-say --wait`` through ``subprocess.run`` with no
timeout. ``--wait`` blocks until the utterance completes, which never happens on
a host with no working speech-dispatcher -- a headless runner, a container, a
machine nobody has configured audio on. Both blocking call sites are end-of-run
announcements ("Stop recording", "Replaying episode"), so a finished recording
hung forever with its data already safely on disk.

Found by running the end-to-end suite under a virgin ``HOME``: recording
completed, logged "Stop recording", and never exited. Under the developer's own
``HOME`` -- where speech-dispatcher was already configured -- it always passed.
"""

from __future__ import annotations

import subprocess
import time
from unittest.mock import patch

from lerobot.utils.utils import SAY_BLOCKING_TIMEOUT_S, say


def test_blocking_say_is_bounded_when_tts_never_returns():
    """The real failure: the TTS command hangs rather than erroring."""
    slept = {}

    def hang(cmd, **kwargs):
        # Stand in for `spd-say --wait` on a host where speech never completes:
        # honour the timeout the caller passed, exactly as subprocess.run would.
        slept["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    with patch("lerobot.utils.utils.subprocess.run", side_effect=hang):
        start = time.perf_counter()
        say("Stop recording", blocking=True)  # must return, must not raise
        elapsed = time.perf_counter() - start

    assert slept["timeout"] == SAY_BLOCKING_TIMEOUT_S, (
        "blocking say passed no timeout to subprocess.run — an unavailable TTS would block the caller forever"
    )
    assert elapsed < SAY_BLOCKING_TIMEOUT_S, "say() should return as soon as the call gives up"


def test_blocking_say_survives_a_missing_tts_binary():
    """A host with no `spd-say` at all must not take the run down with it."""
    with patch("lerobot.utils.utils.subprocess.run", side_effect=FileNotFoundError("spd-say")):
        say("Stop recording", blocking=True)


def test_blocking_say_survives_a_failing_tts_binary():
    """check=True means a non-zero exit raises; that must not escape either."""
    with patch(
        "lerobot.utils.utils.subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=1, cmd=["spd-say"]),
    ):
        say("Stop recording", blocking=True)


def test_non_blocking_say_still_does_not_wait():
    """Guard the guard: the fix must not have quietly made every cue blocking.

    If ``say()`` started waiting on the common non-blocking path, every recording
    would pay the TTS latency at each episode boundary.
    """
    with (
        patch("lerobot.utils.utils.subprocess.Popen") as popen,
        patch("lerobot.utils.utils.subprocess.run") as run,
        patch("lerobot.utils.utils.platform.system", return_value="Linux"),
    ):
        say("Recording episode 0")
    assert popen.called, "non-blocking say stopped spawning the cue"
    assert not run.called, "non-blocking say became blocking"
