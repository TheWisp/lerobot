"""A spoken cue must never hang a run, kill a run, or drown the log.

Text-to-speech here is an OS package -- `spd-say` from `speech-dispatcher` on
Linux -- and *not* a Python dependency, so `uv sync` never provides it. This
repo's own `docker/Dockerfile.training` does not install it, while
`Dockerfile.user` does; CI runners and fresh machines generally do not have it
working either. "No TTS" is the normal case on a large share of hosts.

Three distinct failures, all reachable, none of which should cost a recording:

1. Binary present, daemon dead -> `spd-say --wait` blocks forever. Found by
   running the end-to-end suite under a virgin ``HOME``: recording completed,
   logged "Stop recording", and never exited. It had always passed on the
   developer's machine, where audio happened to be configured.
2. Binary absent -> ``Popen`` raises ``FileNotFoundError``. The worse one: the
   non-blocking path runs at every episode boundary, so it takes the session
   down mid-recording rather than at the end.
3. Unsupported platform -> used to raise ``RuntimeError`` outright.
"""

from __future__ import annotations

import logging
import subprocess
import time
from unittest.mock import patch

import pytest

import lerobot.utils.utils as utils_mod
from lerobot.utils.utils import SAY_BLOCKING_TIMEOUT_S, say


@pytest.fixture(autouse=True)
def _reset_warned_once():
    """The warning is once-per-process, so tests must not inherit each other's."""
    utils_mod._tts_warned = False
    yield
    utils_mod._tts_warned = False


def test_blocking_say_is_bounded_when_tts_never_returns():
    """Failure 1: the TTS command hangs rather than erroring."""
    seen = {}

    def hang(cmd, **kwargs):
        # Stand in for `spd-say --wait` where speech never completes: honour the
        # timeout the caller passed, exactly as subprocess.run would.
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    with patch("lerobot.utils.utils.subprocess.run", side_effect=hang):
        start = time.perf_counter()
        say("Stop recording", blocking=True)
        elapsed = time.perf_counter() - start

    assert seen["timeout"] == SAY_BLOCKING_TIMEOUT_S, (
        "blocking say passed no timeout to subprocess.run — an unavailable TTS would block the caller forever"
    )
    assert elapsed < SAY_BLOCKING_TIMEOUT_S, "say() should return as soon as the call gives up"


def test_non_blocking_say_survives_a_missing_binary():
    """Failure 2: the path that runs every episode, on a host with no spd-say."""
    with patch("lerobot.utils.utils.subprocess.Popen", side_effect=FileNotFoundError("spd-say")):
        say("Recording episode 0")  # must not raise — this would end the session


def test_blocking_say_survives_a_missing_binary():
    with patch("lerobot.utils.utils.subprocess.run", side_effect=FileNotFoundError("spd-say")):
        say("Stop recording", blocking=True)


def test_blocking_say_survives_a_failing_binary():
    """check=True means a non-zero exit raises; that must not escape either."""
    with patch(
        "lerobot.utils.utils.subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=1, cmd=["spd-say"]),
    ):
        say("Stop recording", blocking=True)


def test_unsupported_platform_does_not_raise():
    """Failure 3: an unknown OS is a missing nicety, not a fatal error."""
    with patch("lerobot.utils.utils.platform.system", return_value="FreeBSD"):
        say("Stop recording", blocking=True)


def test_unavailable_tts_warns_once_not_once_per_cue(caplog):
    """Warn — the right level for lost audio — but only the first time.

    say() is called at every episode boundary. Warning each time would bury the
    loop-health lines an operator reads during a recording session.
    """
    with (
        caplog.at_level(logging.WARNING),
        patch("lerobot.utils.utils.subprocess.Popen", side_effect=FileNotFoundError("spd-say")),
    ):
        for i in range(5):
            say(f"Recording episode {i}")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
    message = warnings[0].getMessage()
    assert "speech-dispatcher" in message, "the warning should say how to get audio back"
    assert "play_sounds" in message, "the warning should say how to silence itself"


def test_non_blocking_say_still_does_not_wait():
    """Guard the guard: the fix must not have quietly made every cue blocking.

    If say() started waiting on the common path, every recording would pay the
    TTS latency at each episode boundary.
    """
    with (
        patch("lerobot.utils.utils.subprocess.Popen") as popen,
        patch("lerobot.utils.utils.subprocess.run") as run,
        patch("lerobot.utils.utils.platform.system", return_value="Linux"),
    ):
        say("Recording episode 0")
    assert popen.called, "non-blocking say stopped spawning the cue"
    assert not run.called, "non-blocking say became blocking"


def test_working_tts_is_silent():
    """No warning when TTS works — otherwise the warning means nothing."""
    with (
        patch("lerobot.utils.utils.subprocess.Popen"),
        patch("lerobot.utils.utils.platform.system", return_value="Linux"),
    ):
        say("Recording episode 0")
    assert not utils_mod._tts_warned, "warned even though the cue succeeded"
