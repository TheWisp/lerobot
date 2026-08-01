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


async def _wait_for_exit(timeout_s: float) -> int | None:
    """Wait for the launched subprocess to exit; return its code, or None on timeout."""
    proc = run_api._active_process
    assert proc is not None, "endpoint reported success but no process is running"
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


@pytest.mark.asyncio
async def test_teleoperate_runs(tmp_path):
    """Teleop loops until stopped, so success is 'still healthy after a few seconds'."""
    await start_teleoperate(TeleoperateRequest(robot=ROBOT, teleop=TELEOP, fps=30))
    try:
        rc = await _wait_for_exit(timeout_s=10)
        assert rc is None, f"teleoperate exited early with code {rc}:\n{_tail()}"
    finally:
        await _terminate()


@pytest.mark.asyncio
async def test_record_then_replay_round_trip(tmp_path: Path):
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
    assert rc == 0, f"record failed (rc={rc}):\n{_tail()}"

    # repo_id is date-stamped at creation, so the directory is not `root` itself.
    written = list(root.parent.rglob("meta/info.json"))
    assert written, f"record exited 0 but wrote no dataset under {root.parent}"
    dataset_root = written[0].parent.parent

    await start_replay(ReplayRequest(robot=ROBOT, repo_id=repo_id, root=str(dataset_root), episode=0))
    rc = await _wait_for_exit(timeout_s=180)
    assert rc == 0, f"replay of the just-recorded episode failed (rc={rc}):\n{_tail()}"
