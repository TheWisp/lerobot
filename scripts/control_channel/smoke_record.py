#!/usr/bin/env python
"""End-to-end smoke test for the control channel against a real ``lerobot-record`` run.

Spawns ``lerobot-record`` with a virtual robot (``virtual_bi_so107``) and
a scripted Cartesian-EE teleop (no hardware needed), then feeds the
exact JSON-line commands the GUI's ``POST /api/run/control`` endpoint
writes — verifying the record loop transitions correctly through
episode → reset → episode → reset → episode → stop.

This is the regression baseline for the
:doc:`Control Channel Roadmap <../../src/lerobot/common/CONTROL_CHANNEL>`
phases. Each future phase (P3 intervene, P5 leader-listener deletion,
P6 QuestControllerSource, ...) should leave this test green; adding a
new verb adds a new phase to the driver.

Usage:

  python scripts/control_channel/smoke_record.py

Requires the conda env's lerobot install (or ``PYTHONPATH=src``) to
pick up this branch's ``common/control_channel.py``. Also requires
``pin-pink`` for the Cartesian-IK transform the scripted teleop
drives — install with ``pip install pin-pink
qpsolvers[open_source_solvers]`` if missing.

Expected output (timestamps will differ):

  [rec] Reset the environment              # initial reset (start_with_reset=True)
  [rec] Recording episode 0
  >>> DRIVER: sent exit_early
  [rec] Control channel: exit_early from stdin
  [rec] Reset the environment              # advance: record -> reset
  >>> DRIVER: sent exit_early
  [rec] Control channel: exit_early from stdin
  [rec] Recording episode 1                # advance: reset -> next episode
  ... (cycle repeats for episode 2) ...
  >>> DRIVER: sent stop_recording
  [rec] Stop recording                     # clean exit
  === ALL PASS — record loop exited with code 0 ===

Exits 0 on success, 1 on timeout / failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def send(proc: subprocess.Popen, cmd: str) -> None:
    """Write the same JSON line the GUI's POST /api/run/control writes."""
    poll = proc.poll()
    if poll is not None:
        print(f"\n!!! DRIVER: subprocess exited (rc={poll}); cannot send {cmd}\n", flush=True)
        return
    payload = json.dumps({"v": 1, "cmd": cmd}).encode() + b"\n"
    try:
        proc.stdin.write(payload)
        proc.stdin.flush()
        print(f"\n>>> DRIVER: sent {cmd}\n", flush=True)
    except BrokenPipeError as e:
        print(f"\n!!! DRIVER: BrokenPipe sending {cmd} (subprocess poll={proc.poll()}): {e}\n", flush=True)


def wait_for(proc: subprocess.Popen, pattern: str, *, timeout: float = 30.0) -> bool:
    """Read lines from proc.stdout until ``pattern`` matches or timeout fires."""
    deadline = time.monotonic() + timeout
    pat = re.compile(pattern)
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        text = line.decode(errors="replace").rstrip()
        print(f"  [rec] {text}", flush=True)
        if pat.search(text):
            return True
    print(f"\n!!! DRIVER: TIMEOUT waiting for {pattern!r}\n", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke lerobot-record with (default: this script's interpreter).",
    )
    ap.add_argument(
        "--dataset-dir",
        default="/tmp/control_channel_smoke_dataset",  # nosec B108 — dev script for /tmp output
        help="Throwaway dataset path. Wiped at the start of every run.",
    )
    ap.add_argument("--episode-time-s", type=float, default=15.0)
    ap.add_argument("--reset-time-s", type=float, default=10.0)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if dataset_dir.exists():
        # safe-destruct: our own /tmp throwaway dataset, recreated every run
        shutil.rmtree(dataset_dir)

    env = {
        **os.environ,
        # Make sure the subprocess picks up THIS branch's control_channel,
        # not whatever editable install the conda env points at.
        "PYTHONPATH": str(SRC),
        # Enables the channel's stdin source — same env var the GUI's
        # subprocess launcher sets.
        "LEROBOT_CONTROL_CHANNEL_STDIN": "1",
    }

    cmd = [
        args.python,
        "-u",
        "-m",
        "lerobot.scripts.lerobot_record",
        "--robot.type=virtual_bi_so107",
        "--teleop.type=scripted_bimanual_ee",
        "--teleop.shape=static_hold",
        "--teleop.size_m=0.0",
        "--teleop.n_waypoints=10000",
        "--teleop.ramp_ticks=5",
        "--teleop.loop_hz=30",
        "--dataset.repo_id=local/control_channel_smoke",
        f"--dataset.root={dataset_dir}",
        "--dataset.single_task=control_channel_smoke",
        "--dataset.fps=30",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time_s}",
        "--dataset.num_episodes=3",
        "--dataset.video=false",
        "--dataset.push_to_hub=false",
        "--play_sounds=false",
    ]

    print(f"DRIVER: launching: {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        # Phase 1: initial reset (start_with_reset=True default) then ep 0 starts.
        print("=== Phase 1: wait for episode 0 to start ===", flush=True)
        if not wait_for(proc, r"Recording episode 0", timeout=60):
            return 1
        time.sleep(2)  # let a few frames record into the buffer

        # Phase 2: advance episode 0 -> reset phase.
        print("\n=== Phase 2: send exit_early -> expect reset ===", flush=True)
        send(proc, "exit_early")
        if not wait_for(proc, r"Reset the environment", timeout=15):
            return 1
        time.sleep(2)

        # Phase 3: advance reset -> episode 1.
        print("\n=== Phase 3: send exit_early -> expect episode 1 ===", flush=True)
        send(proc, "exit_early")
        if not wait_for(proc, r"Recording episode 1", timeout=15):
            return 1
        time.sleep(2)

        # Phase 4: advance episode 1 -> reset (the "reset the follower" step).
        print("\n=== Phase 4: send exit_early -> expect reset ===", flush=True)
        send(proc, "exit_early")
        if not wait_for(proc, r"Reset the environment", timeout=15):
            return 1
        time.sleep(1)

        # Phase 5: advance reset -> episode 2 (the "advance to another episode").
        print("\n=== Phase 5: send exit_early -> expect episode 2 ===", flush=True)
        send(proc, "exit_early")
        if not wait_for(proc, r"Recording episode 2", timeout=15):
            return 1
        time.sleep(1)

        # Phase 6: stop the whole thing cleanly.
        print("\n=== Phase 6: send stop_recording -> expect clean exit ===", flush=True)
        send(proc, "stop_recording")
        if not wait_for(proc, r"Stop recording", timeout=30):
            return 1

        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("DRIVER: subprocess did not exit within 60s of stop_recording")
            return 1

        print(f"\n=== ALL PASS — record loop exited with code {rc} ===", flush=True)
        return 0 if rc in (0, None) else 1
    finally:
        if proc.poll() is None:
            print("\nDRIVER: terminating leftover subprocess", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
