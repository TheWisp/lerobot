#!/usr/bin/env python

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

"""Tests for the GUI server's startup/shutdown lifecycle hooks.

The user-visible bug these guard against: Ctrl+C on the GUI server
hangs because the shutdown hook awaits ``subprocess.wait()`` in a way
that doesn't honor cancellation when the subprocess ignores SIGINT.
The shutdown helper is now extracted (``_terminate_active_process``)
and bounded so even a misbehaving subprocess can't wedge the server.
"""

from __future__ import annotations

import asyncio
import signal
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import lerobot.gui.api.run as run_module
from lerobot.gui.server import _terminate_active_process, shutdown_event


def _make_mock_proc(*, returncode: int | None = None, ignore_sigint: bool = False):
    """Build a fake asyncio.subprocess.Process.

    When ``ignore_sigint=True``, ``wait()`` only resolves once ``kill()``
    is called — mimics a wedged teleop subprocess that doesn't respond
    to its own SIGINT handler. The shutdown hook must bound its wait
    even in this case.
    """
    proc = MagicMock()
    proc.returncode = returncode
    killed = asyncio.Event()

    async def wait():
        if ignore_sigint:
            await killed.wait()  # only resolves after .kill()
        return proc.returncode if proc.returncode is not None else 0

    proc.wait = wait
    proc.send_signal = MagicMock()

    def kill():
        killed.set()
        proc.returncode = -9

    proc.kill = MagicMock(side_effect=kill)
    return proc


@pytest.fixture
def reset_active_process():
    """Save/restore run_module._active_process (and the loop it belongs to) per test.

    ``_active_loop`` is cleared rather than merely restored: these tests install
    a process into the global directly instead of launching one, so whatever
    loop a previous test left behind does not own it, and leaving that stale
    value in place makes the shutdown hook decline a process it should handle.
    """
    original = run_module._active_process
    original_loop = run_module._active_loop
    run_module._active_loop = None
    yield
    run_module._active_process = original
    run_module._active_loop = original_loop


# ============================================================================
# _terminate_active_process — the bounded helper
# ============================================================================


class TestTerminateActiveProcess:
    def test_returns_false_when_no_subprocess(self, reset_active_process):
        run_module._active_process = None
        result = asyncio.run(_terminate_active_process())
        assert result is False

    def test_returns_false_when_subprocess_already_exited(self, reset_active_process):
        run_module._active_process = _make_mock_proc(returncode=0)
        result = asyncio.run(_terminate_active_process())
        assert result is False

    def test_clean_subprocess_exits_on_sigint(self, reset_active_process):
        proc = _make_mock_proc(returncode=None, ignore_sigint=False)
        run_module._active_process = proc

        result = asyncio.run(_terminate_active_process(sigint_grace_s=2.0))

        assert result is True
        proc.send_signal.assert_called_once_with(signal.SIGINT)
        proc.kill.assert_not_called()

    def test_wedged_subprocess_gets_killed_within_grace(self, reset_active_process):
        """If the subprocess ignores SIGINT, the helper falls back to kill
        within ``sigint_grace_s`` seconds. The whole call must complete in
        bounded time even in the worst case — this is the regression the
        user hit when Ctrl+C silently hung."""
        proc = _make_mock_proc(returncode=None, ignore_sigint=True)
        run_module._active_process = proc

        t0 = time.perf_counter()
        result = asyncio.run(_terminate_active_process(sigint_grace_s=0.3))
        elapsed = time.perf_counter() - t0

        assert result is True
        proc.send_signal.assert_called_once_with(signal.SIGINT)
        proc.kill.assert_called_once()
        # Must return promptly after the grace period — never wedge.
        assert elapsed < 1.0, f"shutdown took {elapsed:.2f}s — should be bounded"

    def test_kill_failure_is_suppressed(self, reset_active_process):
        """If proc.kill() raises (e.g. the subprocess exited between the
        wait timeout and our kill call), the helper must still return
        cleanly so the rest of shutdown (shm sweep) can run."""
        proc = _make_mock_proc(returncode=None, ignore_sigint=True)
        proc.kill = MagicMock(side_effect=ProcessLookupError("vanished"))
        run_module._active_process = proc

        result = asyncio.run(_terminate_active_process(sigint_grace_s=0.1))
        assert result is True

    def test_lazy_global_lookup(self, reset_active_process):
        """The helper must read run_module._active_process lazily, not
        snapshot it at import time. This is the latent bug class behind
        the wedge: a top-level ``from … import _active_process`` captures
        the value at handler-definition time, not handler-invocation
        time, so a freshly-launched subprocess could be invisible to the
        cleanup path."""
        run_module._active_process = None
        assert asyncio.run(_terminate_active_process()) is False

        # Install a process AFTER the helper was imported. It must see it.
        proc = _make_mock_proc(returncode=None, ignore_sigint=False)
        run_module._active_process = proc

        assert asyncio.run(_terminate_active_process(sigint_grace_s=2.0)) is True


# ============================================================================
# shutdown_event — full handler
# ============================================================================


class TestShutdownEventEndToEnd:
    """The actual @app.on_event('shutdown') handler with mocks for the
    debug-process side effect and the shm sweep redirected at tmp_path."""

    def test_idle_server_shutdown_is_fast(self, reset_active_process, tmp_path):
        """Idle server (no subprocess, no shm) must shut down in
        milliseconds — this is the common Ctrl+C scenario."""
        run_module._active_process = None

        with patch("lerobot.gui.api.run._stop_debug_process", AsyncMock(return_value=None)):
            with patch("lerobot.robots.obs_stream._SHM_DIR", str(tmp_path)):
                t0 = time.perf_counter()
                asyncio.run(shutdown_event())
                elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"idle shutdown took {elapsed:.2f}s — too slow"

    def test_wedged_subprocess_does_not_wedge_shutdown(self, reset_active_process, tmp_path):
        """The user's actual symptom: a teleop subprocess that ignores
        SIGINT must NOT wedge the GUI server's Ctrl+C path."""
        proc = _make_mock_proc(returncode=None, ignore_sigint=True)
        run_module._active_process = proc

        # Patch the helper to use a tight grace so the test is fast,
        # but we still verify the bound holds end-to-end.
        from lerobot.gui import server as server_mod

        original = server_mod._terminate_active_process

        async def fast_terminate():
            return await original(sigint_grace_s=0.2)

        with (
            patch("lerobot.gui.server._terminate_active_process", fast_terminate),
            patch("lerobot.gui.api.run._stop_debug_process", AsyncMock(return_value=None)),
            patch("lerobot.robots.obs_stream._SHM_DIR", str(tmp_path)),
        ):
            t0 = time.perf_counter()
            asyncio.run(shutdown_event())
            elapsed = time.perf_counter() - t0

        assert elapsed < 1.5
        proc.kill.assert_called_once()


class TestShutdownEventBestEffort:
    """A failure in any single cleanup step MUST NOT block the others.
    Without per-step try/except, one misbehaving callback would silently
    leak every later resource (subprocess, shm, hardware, executor)."""

    def test_first_step_failure_does_not_block_later_steps(self, reset_active_process, tmp_path):
        run_module._active_process = None

        terminate_called = False
        prefetch_shutdown_called = False
        hw_cleanup_called = False

        async def fail_stop_debug():
            raise RuntimeError("simulated debug stop failure")

        async def track_terminate():
            nonlocal terminate_called
            terminate_called = True

        def track_hw_cleanup():
            nonlocal hw_cleanup_called
            hw_cleanup_called = True

        def track_prefetch_shutdown():
            nonlocal prefetch_shutdown_called
            prefetch_shutdown_called = True

        with (
            patch("lerobot.gui.api.run._stop_debug_process", fail_stop_debug),
            patch("lerobot.gui.server._terminate_active_process", track_terminate),
            patch("lerobot.gui.api.robot.cleanup_in_process_resources", track_hw_cleanup),
            patch("lerobot.gui.api.datasets.shutdown_prefetch_executor", track_prefetch_shutdown),
            patch("lerobot.robots.obs_stream._SHM_DIR", str(tmp_path)),
        ):
            # Must not raise out of shutdown_event despite the first step failing.
            asyncio.run(shutdown_event())

        assert terminate_called, "second cleanup step skipped after first raised"
        assert hw_cleanup_called, "hardware cleanup skipped after earlier step raised"
        assert prefetch_shutdown_called, "prefetch shutdown skipped after earlier step raised"

    def test_middle_step_failure_does_not_block_later_steps(self, reset_active_process, tmp_path):
        run_module._active_process = None

        prefetch_shutdown_called = False

        def fail_cleanup_streams():
            raise RuntimeError("simulated shm sweep failure")

        def track_prefetch_shutdown():
            nonlocal prefetch_shutdown_called
            prefetch_shutdown_called = True

        with (
            patch("lerobot.gui.api.run._stop_debug_process", AsyncMock(return_value=None)),
            patch("lerobot.robots.obs_stream.cleanup_stale_streams", fail_cleanup_streams),
            patch("lerobot.gui.api.robot.cleanup_in_process_resources", MagicMock()),
            patch("lerobot.gui.api.datasets.shutdown_prefetch_executor", track_prefetch_shutdown),
            patch("lerobot.robots.obs_stream._SHM_DIR", str(tmp_path)),
        ):
            asyncio.run(shutdown_event())

        assert prefetch_shutdown_called, "final cleanup step skipped after middle step raised"


# ============================================================================
# Bonus: a real subprocess that ignores SIGINT, end-to-end verification
# ============================================================================


class TestRealSubprocessShutdown:
    """Spawn an actual python subprocess that traps SIGINT and ignores it.
    Exercises the real asyncio.subprocess path — not just MagicMocks —
    to catch any wait_for/cancel interaction bugs that mocks would miss."""

    def test_ignores_sigint_then_killed(self, reset_active_process):
        async def run():
            # A tiny inline script that installs a no-op SIGINT handler
            # and sleeps. Without the handler the default Python behavior
            # would convert SIGINT to KeyboardInterrupt and exit; we want
            # the worst-case "subprocess ignores SIGINT entirely" path.
            code = "import signal, time\nsignal.signal(signal.SIGINT, lambda *a: None)\ntime.sleep(60)\n"
            proc = await asyncio.create_subprocess_exec(
                "python3",
                "-c",
                code,
            )
            run_module._active_process = proc
            try:
                t0 = time.perf_counter()
                result = await _terminate_active_process(sigint_grace_s=0.5)
                elapsed = time.perf_counter() - t0
            finally:
                # Reap the process if for some reason it's still around.
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            return result, elapsed

        result, elapsed = asyncio.run(run())
        assert result is True
        # Grace + small overhead to get to the kill path.
        assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s — should be bounded"

    def test_an_unexplained_wait_error_does_not_escalate_to_sigkill(self, reset_active_process):
        """Only a timeout means "it is ignoring SIGINT". Anything else must not kill.

        This is the mechanism behind issue #128. ``_active_process`` is a module
        global, and the test suite boots GUI servers inside the pytest process —
        so this hook can run on one event loop while the global holds a
        subprocess created on another. Awaiting that process raises **at once**
        rather than after the grace period, and the old ``except Exception``
        read that as "it ignored SIGINT" and sent SIGKILL. The child got SIGINT
        and SIGKILL about a millisecond apart and died mid-traceback, with
        nothing left to explain why.

        The harm is not limited to tests. The comment in the hook exists because
        killing a record run before ``save_episode()`` finishes silently
        discards a completed episode; escalating on an error nobody has
        diagnosed is exactly how that would happen in the field.
        """
        import sys
        import threading

        spawned, release = threading.Event(), threading.Event()
        holder: dict[str, asyncio.subprocess.Process] = {}

        async def own_the_process():
            """A different event loop creates the subprocess and keeps running."""
            # It announces readiness *after* installing the handler: signalling a
            # process that has not reached that line yet kills it with SIGINT and
            # tests the wrong thing.
            code = (
                "import signal, sys, time\n"
                "signal.signal(signal.SIGINT, lambda *a: None)\n"
                "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
                "time.sleep(60)\n"
            )
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE
            )
            holder["proc"] = proc
            run_module._active_process = proc
            assert await proc.stdout.readline() == b"ready\n", "subprocess never armed its handler"
            spawned.set()
            while not release.is_set():
                await asyncio.sleep(0.02)
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

        owner = threading.Thread(target=lambda: asyncio.run(own_the_process()))
        owner.start()
        try:
            assert spawned.wait(timeout=10), "owner loop never started the subprocess"
            proc = holder["proc"]

            # The shutdown hook, on its own loop — as a server booted by a
            # fixture inside this process would run it.
            assert asyncio.run(_terminate_active_process(sigint_grace_s=5.0)) is True

            # The child ignores SIGINT, so it must still be alive. With the
            # escalation bug it was dead within ~1.5 ms; half a second is a
            # margin of two orders of magnitude, not a race.
            time.sleep(0.5)
            assert proc.returncode is None, (
                f"subprocess was killed (rc={proc.returncode}) by an error that was not a timeout"
            )
        finally:
            release.set()
            owner.join(timeout=15)

    def test_a_run_launched_on_another_loop_is_not_signalled_at_all(self, reset_active_process):
        """A server stops the run it launched, and no one else's.

        `_active_process` is a module global. With several GUI servers in one
        interpreter — which the test suite creates, booting them in threads —
        this hook could find a subprocess belonging to a different server's
        event loop and signal it. That is issue #128: the run a test had just
        launched was killed by the shutdown of a server that had nothing to do
        with it.

        The subprocess here does *not* trap SIGINT, so surviving proves no
        signal was sent at all, rather than that one was survived.
        """
        import sys
        import threading

        spawned, release = threading.Event(), threading.Event()
        holder: dict[str, asyncio.subprocess.Process] = {}

        async def own_the_process():
            code = "import sys, time\nsys.stdout.write('ready\\n'); sys.stdout.flush()\ntime.sleep(60)\n"
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE
            )
            holder["proc"] = proc
            run_module._active_process = proc
            run_module._active_loop = asyncio.get_running_loop()
            assert await proc.stdout.readline() == b"ready\n"
            spawned.set()
            while not release.is_set():
                await asyncio.sleep(0.02)
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

        owner = threading.Thread(target=lambda: asyncio.run(own_the_process()))
        owner.start()
        try:
            assert spawned.wait(timeout=10), "owner loop never started the subprocess"
            proc = holder["proc"]

            # A foreign server's shutdown must decline, not signal.
            assert asyncio.run(_terminate_active_process(sigint_grace_s=5.0)) is False

            time.sleep(0.3)
            assert proc.returncode is None, (
                f"another server's shutdown signalled a run it did not launch (rc={proc.returncode})"
            )
        finally:
            release.set()
            owner.join(timeout=15)
