"""End-to-end: each Run-tab flow actually launches and completes.

Every other GUI test stops at the process boundary -- it asserts which argv the
endpoint assembles, then mocks the subprocess away. That is where the bugs live.
A renamed upstream flag, a robot type one script forgot to import: both produce a
GUI that looks fine, tests that stay green, and a subprocess that dies on launch.
Recording was broken exactly this way and no test noticed.

So these actually launch ``lerobot-teleoperate`` / ``-record`` / ``-replay`` and
wait for them to exit. They are the reason the ``virtual_bi_so107`` robot exists:
it has no buses and no cameras, so a full record→replay cycle runs on any machine
with nothing plugged in and nothing to move. Costs a few seconds per flow, needs
no hardware, and fails loudly on the whole class of break that unit tests cannot
see.

Marked ``e2e_flow`` -- they spawn real processes and write real datasets (into
``tmp_path``, never the user's cache).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import signal
from pathlib import Path

import pytest

from lerobot.gui.api import run as run_api
from lerobot.gui.api.run import (
    RecordRequest,
    ReplayRequest,
    TeleoperateRequest,
    start_record,
    start_replay,
    start_teleoperate,
)

pytestmark = [
    pytest.mark.e2e_flow,
    pytest.mark.skipif(
        shutil.which("lerobot-record") is None,
        reason="lerobot console scripts not installed in this environment",
    ),
]

# No buses, no cameras: nothing to plug in and nothing that can move.
ROBOT = {"type": "virtual_bi_so107", "fields": {"id": "e2e"}}
# A teleop that generates its own trajectory, so the loop has input without a human.
TELEOP = {"type": "scripted_bimanual_ee", "fields": {"id": "e2e-leader", "shape": "circle"}}


# ── Attributing a signalled exit (see issue #128) ────────────────────────
#
# test_teleoperate_runs fails a few times a year with the launched subprocess
# dead of SIGKILL a fraction of a second after exec, having printed the first
# few lines of a traceback and nothing more. A process cannot report its own
# SIGKILL, so the evidence has to come from outside it — hence `_exit_report`.
#
# What the shape of that output already rules out, measured rather than argued:
#
#   * SIGKILL alone prints nothing. Killing the real teleop child mid-import
#     produced an empty buffer every time. So the traceback in the failure means
#     an exception was ALREADY propagating when the signal landed — two events,
#     not one asynchronous kill.
#   * An OOM kill prints nothing either. Running the child under a cgroup
#     memory cap tight enough to kill it (120M through 340M) gave -9 with zero
#     traceback lines at every cap, because the kernel's SIGKILL gives Python no
#     chance to raise. The OOM killer therefore cannot produce this failure's
#     signature, which is what the issue had assumed it did.
#   * The child is also the least plausible OOM victim in the room: it holds
#     ~45 MB at the line it dies on, against pytest's gigabytes. Squeezing a
#     whole test run into a cap, the kernel killed pytest, never the child.
#   * The buffer is not truncated by the harness. `proc.wait()` resolves only
#     after the reader tasks drain, so the output stops where the child stopped
#     writing; widening `_tail` recovers nothing.
#
# Reproducing the signature took SIGINT (so Python starts printing a traceback)
# followed by SIGKILL under a millisecond later; at 5 ms the traceback completes.
# Nothing in the codebase does that — the one SIGINT-then-SIGKILL path,
# `stop_process`, waits 5 s between them and this test never calls it. So what
# raises in the child, and what kills it immediately after, are both still open.
# The OOM counter below is what settles the memory question in situ rather than
# by analogy with a local cgroup.

# The pid of the subprocess most recently waited on, for `_exit_report`. The
# endpoint clears `_active_process` as soon as the process dies, so the pid has
# to be taken while it is still running.
_last_pid: int | None = None


async def _wait_for_exit(timeout_s: float) -> int | None:
    """Wait for the launched subprocess to exit; return its code, or None on timeout."""
    global _last_pid
    proc = run_api._active_process
    assert proc is not None, "endpoint reported success but no process is running"
    _last_pid = proc.pid
    try:
        return await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except TimeoutError:
        return None


async def _terminate() -> None:
    proc = run_api._active_process
    if proc is not None and proc.returncode is None:
        proc.terminate()
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=15)


def _tail(n: int = 25) -> str:
    return "\n".join(run_api._output_lines[-n:])


def _proc_field(path: str, key: str) -> int | None:
    """The integer on the line of ``path`` starting with ``key``; None if unreadable.

    Never raises: this runs while reporting some other failure, and a diagnostic
    that fails is worse than one that says it could not look.
    """
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return None


def _oom_kills() -> int | None:
    """The kernel's cumulative count of OOM kills since boot, or None off Linux.

    Unlike dmesg this needs no privileges, which matters because the whole point
    is to run it unattended in CI.
    """
    return _proc_field("/proc/vmstat", "oom_kill ")


def _exit_report(what: str, rc: int | None, oom_before: int | None) -> str:
    """Describe a subprocess exit, including the part the subprocess cannot describe.

    A process killed by a signal leaves no account of its own death — the output
    simply stops, mid-traceback if it was already failing, and the exit code is
    the only thing left. So when the exit was signalled, add the evidence that
    only exists outside the process. The load-bearing figure is the OOM-kill
    counter: its delta across the test says whether the kernel killed anything
    at all on this host while the child was alive, which either implicates
    memory pressure or excludes it. Without that, "exited with -9" is equally
    consistent with an OOM kill and with a bug that has nothing to do with
    memory, and the failure cannot be attributed at all.
    """
    lines = [f"{what} (rc={rc}):"]
    if rc is not None and rc < 0:
        try:
            name = signal.Signals(-rc).name
        except ValueError:
            name = f"unknown signal {-rc}"
        after = _oom_kills()
        if oom_before is None or after is None:
            oom = "OOM-kill counter unreadable, so memory pressure is neither shown nor excluded"
        elif after > oom_before:
            oom = f"the kernel OOM-killed {after - oom_before} process(es) during this test"
        else:
            oom = "the kernel OOM-killed nothing on this host during this test"
        lines += [
            f"  child pid {_last_pid} was killed by {name}; {oom}.",
            f"  MemAvailable {_proc_field('/proc/meminfo', 'MemAvailable:')} kB, "
            f"this pytest process {_proc_field('/proc/self/status', 'VmRSS:')} kB RSS.",
        ]
    # A signalled process stops mid-sentence, so every line it managed to write
    # is evidence; a clean non-zero exit has said what it wanted to say.
    lines.append(_tail(200 if rc is not None and rc < 0 else 25))
    return "\n".join(lines)


@pytest.fixture
def oom_baseline() -> int | None:
    """The kernel's OOM-kill count before the test, to difference against on failure."""
    return _oom_kills()


@pytest.mark.asyncio
async def test_teleoperate_runs(tmp_path, oom_baseline):
    """Teleop loops until stopped, so success is 'still healthy after a few seconds'."""
    await start_teleoperate(TeleoperateRequest(robot=ROBOT, teleop=TELEOP, fps=30))
    try:
        rc = await _wait_for_exit(timeout_s=10)
        assert rc is None, _exit_report("teleoperate exited early", rc, oom_baseline)
    finally:
        await _terminate()


@pytest.mark.asyncio
async def test_record_then_replay_round_trip(tmp_path: Path, oom_baseline):
    """The full data path: record an episode, then replay the episode just written.

    Round-tripping is what makes this worth its runtime -- replay reads back what
    record produced, so a dataset written in a shape replay cannot consume fails
    here rather than the next time someone opens the GUI.
    """
    repo_id = "e2e/round_trip"
    root = tmp_path / "dataset"

    await start_record(
        RecordRequest(
            robot=ROBOT,
            teleop=TELEOP,
            repo_id=repo_id,
            root=str(root),
            single_task="End-to-end round trip.",
            num_episodes=1,
            episode_time_s=2,
            reset_time_s=1,
            fps=30,
            video=True,
            # No audio: a test must not depend on the host having working TTS,
            # and must not make noise on someone's machine. say() is bounded
            # now, but this keeps the flow under test to the data path.
            play_sounds=False,
        )
    )
    rc = await _wait_for_exit(timeout_s=180)
    assert rc == 0, _exit_report("record failed", rc, oom_baseline)

    # repo_id is date-stamped at creation, so the directory is not `root` itself.
    written = list(root.parent.rglob("meta/info.json"))
    assert written, f"record exited 0 but wrote no dataset under {root.parent}"
    dataset_root = written[0].parent.parent

    await start_replay(ReplayRequest(robot=ROBOT, repo_id=repo_id, root=str(dataset_root), episode=0))
    rc = await _wait_for_exit(timeout_s=180)
    assert rc == 0, _exit_report("replay of the just-recorded episode failed", rc, oom_baseline)


class TestExitReport:
    """The post-mortem gets one shot: the failure it explains happens a few times a year.

    So it is tested here rather than trusted, and every branch has to produce a
    verdict a reader can act on — including the branch where the kernel's
    counter cannot be read, which must not be mistaken for "no OOM happened".
    """

    def test_the_kernel_publishes_its_oom_count_unprivileged(self):
        """The whole design rests on this file being readable without root."""
        if not Path("/proc/vmstat").exists():
            pytest.skip("no /proc/vmstat (not Linux)")
        count = _oom_kills()
        assert isinstance(count, int) and count >= 0, f"expected a count, got {count!r}"

    def test_an_unreadable_field_is_none_rather_than_an_exception(self):
        assert _proc_field("/proc/does-not-exist", "anything") is None
        assert _proc_field("/proc/vmstat", "no_such_key ") is None

    def test_a_signalled_exit_names_the_signal(self, monkeypatch):
        monkeypatch.setattr(run_api, "_output_lines", ["some output"])
        report = _exit_report("teleoperate exited early", -9, 0)
        assert "SIGKILL" in report
        assert "some output" in report

    def test_an_oom_kill_during_the_test_is_attributed_to_the_kernel(self, monkeypatch):
        monkeypatch.setattr(run_api, "_output_lines", [])
        monkeypatch.setitem(globals(), "_oom_kills", lambda: 4)
        report = _exit_report("teleoperate exited early", -9, 1)
        assert "OOM-killed 3 process(es)" in report

    def test_no_oom_kill_is_stated_as_a_finding(self, monkeypatch):
        """A zero delta is evidence — it excludes memory pressure, so say so."""
        monkeypatch.setattr(run_api, "_output_lines", [])
        monkeypatch.setitem(globals(), "_oom_kills", lambda: 7)
        report = _exit_report("teleoperate exited early", -9, 7)
        assert "OOM-killed nothing" in report

    def test_an_unreadable_counter_does_not_masquerade_as_no_oom(self, monkeypatch):
        """The dangerous failure: silence read as an all-clear."""
        monkeypatch.setattr(run_api, "_output_lines", [])
        monkeypatch.setitem(globals(), "_oom_kills", lambda: None)
        report = _exit_report("teleoperate exited early", -9, None)
        assert "neither shown nor excluded" in report
        assert "OOM-killed nothing" not in report

    def test_a_clean_non_zero_exit_gets_no_kernel_forensics(self, monkeypatch):
        """A process that exited normally accounted for itself; don't pad the message."""
        monkeypatch.setattr(run_api, "_output_lines", ["boom"])
        report = _exit_report("record failed", 1, 0)
        assert "killed by" not in report
        assert report.startswith("record failed (rc=1):")
