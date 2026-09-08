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
"""One GUI server per host: a second one refuses to start and names the first.

Two servers on one machine shared the GPU and the fixed-name shared-memory
segments, and each destroyed the other's -- both SAM3 workers bound to one
obs-stream inode, and Play refused on the server whose frame buffer the other
had just swept. The lock is held by a real second process here, because an
in-process second acquire would test the file object, not the host.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from lerobot.gui import single_instance

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="advisory file locks are POSIX")

HOLDER = """
import sys, time
from pathlib import Path
from lerobot.gui.single_instance import acquire
f = acquire("127.0.0.1", 9200, Path(sys.argv[1]))
print("held", flush=True)
time.sleep(60)
"""


@pytest.fixture
def holder(tmp_path):
    """A separate process holding the lock, the way a running server does."""
    lock = tmp_path / "gui.lock"
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", HOLDER, str(lock)], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout.readline().strip() == "held", "the holder never took the lock"
    yield lock, proc
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


def test_a_second_server_is_refused_and_told_who_runs(holder):
    lock, _proc = holder
    with pytest.raises(single_instance.AnotherServerRunningError) as e:
        single_instance.acquire("127.0.0.1", 9201, lock)
    assert "port 9200" in str(e.value), str(e.value)
    assert "One server per host" in str(e.value)


def test_the_lock_dies_with_its_holder(holder):
    """A crashed server must not lock the host forever: the kernel releases an
    flock with the process, so no stale-lock handling is needed or wanted."""
    lock, proc = holder
    proc.kill()
    proc.wait(timeout=10)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            f = single_instance.acquire("127.0.0.1", 9201, lock)
            break
        except single_instance.AnotherServerRunningError:
            time.sleep(0.1)
    else:
        pytest.fail("the lock outlived its holder")
    assert lock.read_text().startswith("pid ")
    f.close()


def test_run_server_refuses_before_it_touches_the_host(holder, monkeypatch):
    """The refusal has to come before the startup sweep and before any spawn:
    a second server that got as far as sweeping would already have destroyed
    the first one's segments."""
    from lerobot.gui import server as gui_server

    lock, _proc = holder
    monkeypatch.setattr(single_instance, "LOCK_PATH", lock)
    touched = []
    monkeypatch.setattr(gui_server, "_mount_mcp", lambda **kw: touched.append("mcp"))
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: touched.append("uvicorn"))
    with pytest.raises(SystemExit) as e:
        gui_server.run_server(host="127.0.0.1", port=9201)
    assert "port 9200" in str(e.value)
    assert touched == [], f"the refused server went on to {touched}"


def test_the_first_server_holds_the_lock_for_its_lifetime(tmp_path, monkeypatch):
    """run_server keeps the lock object on the app, so it is not collected --
    and released -- the moment the function moves on."""
    from lerobot.gui import server as gui_server

    lock = tmp_path / "gui.lock"
    monkeypatch.setattr(single_instance, "LOCK_PATH", lock)
    monkeypatch.setattr(gui_server, "_mount_mcp", lambda **kw: None)
    monkeypatch.setattr("lerobot.gui.mdns.advertise", lambda **kw: None)
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
    gui_server.run_server(host="127.0.0.1", port=9202)
    try:
        assert lock.read_text().strip().endswith("port 9202 host 127.0.0.1")
        with pytest.raises(single_instance.AnotherServerRunningError):
            single_instance.acquire("127.0.0.1", 9203, lock)
    finally:
        gui_server.app.state.instance_lock.close()
