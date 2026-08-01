"""End-to-end: operator interaction sequences, not just happy paths.

``test_e2e_flows.py`` proves each flow launches and completes when left alone.
Nothing proved what happens when an operator *interacts* mid-run — and that is
where the next bug class lived: record one episode, press Stop during the
reset phase, and the dataset ended up empty. ``save_episode()`` runs after the
reset, so for the entire reset window a finished episode exists only in
memory, and the GUI's Stop was a raw SIGINT.

These tests drive the same endpoints the GUI's buttons call, against real
subprocesses on the virtual robot, and assert on what reaches disk — the only
thing the operator ultimately cares about.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from lerobot.gui.api import run as run_api
from lerobot.gui.api.run import ControlRequest, RecordRequest, send_control, start_record, stop_process

pytestmark = [
    pytest.mark.e2e_flow,
    pytest.mark.skipif(
        shutil.which("lerobot-record") is None,
        reason="lerobot console scripts not installed in this environment",
    ),
]

ROBOT = {"type": "virtual_bi_so107", "fields": {"id": "e2e-op"}}
# static_hold with a huge waypoint budget: the trajectory must outlive the whole
# test, or is_exhausted ends episodes on its own and the timing assertions lie.
TELEOP = {
    "type": "scripted_bimanual_ee",
    "fields": {"id": "e2e-op-leader", "shape": "static_hold", "n_waypoints": 100_000},
}


async def _wait_for_phase(predicate, timeout_s: float) -> str | None:
    """Poll the same phase state the GUI's status endpoint reports."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        proc = run_api._active_process
        if proc is None or proc.returncode is not None:
            return None
        phase = run_api._active_phase
        if phase and predicate(phase):
            return phase
        await asyncio.sleep(0.2)
    return None


def _dataset_info(search_root: Path) -> dict:
    infos = list(search_root.rglob("meta/info.json"))
    return json.loads(infos[0].read_text()) if infos else {}


def _episodes_on_disk(search_root: Path) -> int:
    info = _dataset_info(search_root)
    return info.get("total_episodes", 0) if info else -1


@pytest.mark.asyncio
async def test_stop_during_reset_keeps_the_finished_episode(tmp_path: Path):
    """The field failure, replayed exactly: record → reset begins → Stop.

    The finished episode is only buffered during reset; Stop must let the
    recorder save it (graceful stop_recording) rather than SIGINT it away.
    """
    await start_record(
        RecordRequest(
            robot=ROBOT,
            teleop=TELEOP,
            repo_id="e2e/stop_mid_reset",
            root=str(tmp_path / "ds"),
            single_task="Operator stops during reset.",
            num_episodes=3,
            episode_time_s=3,
            # Long on purpose: the operator must be able to stop well inside
            # the reset window, exactly like the real session that lost data.
            reset_time_s=120,
            fps=30,
            video=True,
            play_sounds=False,
        )
    )

    # Runs open with an initial reset of the full reset_time_s; the operator
    # presses Next to start recording — replicate that.
    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=60), (
        "record never reached the initial reset phase"
    )
    await send_control(ControlRequest(cmd="exit_early"))
    phase = await _wait_for_phase(lambda p: p.startswith("recording episode"), timeout_s=30)
    assert phase is not None, "record never reached the recording phase"
    phase = await _wait_for_phase(lambda p: p == "resetting", timeout_s=30)
    assert phase is not None, "episode 0 never ended into the reset phase"

    result = await stop_process()

    assert result["graceful"], (
        "Stop fell back to SIGINT during the reset phase — the buffered episode was discarded"
    )
    assert _episodes_on_disk(tmp_path) == 1, (
        "the episode the operator finished before pressing Stop did not reach disk"
    )


@pytest.mark.asyncio
async def test_stop_before_any_episode_is_clean_and_the_dataset_is_empty(tmp_path: Path):
    """Stop during the initial reset: nothing recorded, nothing to save.

    The risk here is the opposite failure — a graceful stop that tries to save
    an empty buffer would crash the recorder on exit (the add_frame ValueError).
    """
    await start_record(
        RecordRequest(
            robot=ROBOT,
            teleop=TELEOP,
            repo_id="e2e/stop_before_first",
            root=str(tmp_path / "ds"),
            single_task="Operator aborts before recording.",
            num_episodes=3,
            episode_time_s=30,
            reset_time_s=120,
            fps=30,
            video=True,
            play_sounds=False,
        )
    )
    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=60)

    proc = run_api._active_process  # stop_process clears the global
    result = await stop_process()

    assert result["graceful"], "stop before the first episode did not exit within grace"
    assert proc.returncode == 0, f"recorder crashed on an empty-buffer stop (rc={proc.returncode})"
    assert _episodes_on_disk(tmp_path) == 0, "an empty dataset grew a phantom episode"


@pytest.mark.asyncio
async def test_stop_mid_episode_keeps_the_partial_take(tmp_path: Path):
    """Stop while frames are landing: the partial take reaches disk.

    Matches the recorder's Esc semantics — the frames captured so far are
    saved, not discarded. An operator who wants to drop the take has
    Re-record for that.
    """
    await start_record(
        RecordRequest(
            robot=ROBOT,
            teleop=TELEOP,
            repo_id="e2e/stop_mid_episode",
            root=str(tmp_path / "ds"),
            single_task="Operator stops mid-episode.",
            num_episodes=3,
            episode_time_s=30,
            reset_time_s=120,
            fps=30,
            video=True,
            play_sounds=False,
        )
    )
    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=60)
    await send_control(ControlRequest(cmd="exit_early"))
    assert await _wait_for_phase(lambda p: p.startswith("recording episode"), timeout_s=30)
    await asyncio.sleep(2.0)  # let frames land

    result = await stop_process()

    assert result["graceful"], "mid-episode stop fell back to SIGINT"
    assert _episodes_on_disk(tmp_path) == 1, "the partial take was lost"


@pytest.mark.asyncio
async def test_rerecord_discards_the_bad_take_and_saves_the_redo(tmp_path: Path):
    """Re-record mid-take: the bad take is discarded, the redo is what's saved.

    The discriminating assertion is the frame count. Take 1 is cut at ~1.5s
    (~45 frames at 30fps) by pressing Re-record; take 2 runs its full 6s
    (~180 frames). If the dataset's single episode is short, the recorder
    saved the take the operator rejected.
    """
    fps = 30
    await start_record(
        RecordRequest(
            robot=ROBOT,
            teleop=TELEOP,
            repo_id="e2e/rerecord",
            root=str(tmp_path / "ds"),
            single_task="Operator vetoes a take.",
            num_episodes=2,
            episode_time_s=6,
            reset_time_s=120,
            fps=fps,
            video=True,
            play_sounds=False,
        )
    )

    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=60)
    await send_control(ControlRequest(cmd="exit_early"))
    assert await _wait_for_phase(lambda p: p == "recording episode 0", timeout_s=30)

    await asyncio.sleep(1.5)  # a short bad take
    await send_control(ControlRequest(cmd="rerecord_episode"))
    # Re-record runs the reset phase first, then discards and redoes the take.
    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=30)
    await send_control(ControlRequest(cmd="exit_early"))
    assert await _wait_for_phase(lambda p: p == "recording episode 0", timeout_s=30), (
        "re-record did not restart the same episode index"
    )
    # The "re-recording" phase is instantaneous (discard and restart happen
    # back-to-back), so poll-based phase watching can miss it; the output
    # buffer records it deterministically.
    assert any("Re-record episode" in ln for ln in run_api._output_lines), "the re-record branch never ran"
    # Let the redo run to its natural 6s end, then stop during the reset.
    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=30)
    result = await stop_process()

    assert result["graceful"], "stop after the redo fell back to SIGINT"
    info = _dataset_info(tmp_path)
    assert info.get("total_episodes") == 1, (
        f"expected exactly the redo on disk, got {info.get('total_episodes')} episodes"
    )
    frames = info.get("total_frames", 0)
    assert frames >= 4 * fps, (
        f"episode has {frames} frames (~{frames / fps:.1f}s) — that is the rejected "
        f"short take, not the full-length redo"
    )


@pytest.mark.asyncio
async def test_next_episode_skips_reset_and_the_episode_survives(tmp_path: Path):
    """The other half of the session: Next during reset, then Stop mid-episode 1.

    exit_early during reset must start the next episode (not end the run), and
    stopping while episode 1 is still open must keep the completed episode 0.
    """
    await start_record(
        RecordRequest(
            robot=ROBOT,
            teleop=TELEOP,
            repo_id="e2e/next_then_stop",
            root=str(tmp_path / "ds"),
            single_task="Operator skips reset.",
            num_episodes=3,
            episode_time_s=30,
            reset_time_s=120,
            fps=30,
            video=True,
            play_sounds=False,
        )
    )

    # Skip the initial reset, as the operator does.
    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=60)
    await send_control(ControlRequest(cmd="exit_early"))
    assert await _wait_for_phase(lambda p: p == "recording episode 0", timeout_s=30)
    # End episode 0 early — the "Next episode" button while recording.
    await send_control(ControlRequest(cmd="exit_early"))
    assert await _wait_for_phase(lambda p: p == "resetting", timeout_s=30)
    # Skip the rest of the reset — the "Next episode" button while resetting.
    await send_control(ControlRequest(cmd="exit_early"))
    phase = await _wait_for_phase(lambda p: p == "recording episode 1", timeout_s=30)
    assert phase is not None, "exit_early during reset did not start the next episode"

    result = await stop_process()
    assert result["graceful"], "Stop during episode 1 fell back to SIGINT"
    # Episode 0 must be on disk. Episode 1 was cut short by Stop; whether the
    # partial take is kept is the recorder's policy — the invariant here is
    # that the finished one is never lost.
    assert _episodes_on_disk(tmp_path) >= 1, "the completed episode 0 was lost"
