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
"""One GUI server per host.

Every GUI server on a host shares one GPU with its SAM3 worker and one set of
shared-memory segments -- the observation stream and the overlay buffers, under
fixed names that a server sweeps at startup, replaces when it publishes and
unlinks when it stops, without knowing another server exists. Two servers on
one machine (2026-09-08) had both SAM3 workers reading a single obs-stream
inode, and one server's Play was refused because the other had just swept its
frame buffer. Partitioning all of that per server would still leave two
workers contending for the GPU, so a host runs one GUI server: a second one
refuses to start and names the first.

The lock is an advisory ``flock`` on a file in the temp dir, held for the life
of the process. The kernel releases it when the holder exits, however it
exits, so a crashed server never leaves a stale lock behind.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

LOCK_PATH = Path(tempfile.gettempdir()) / "lerobot-gui.lock"


class AnotherServerRunningError(RuntimeError):
    """A GUI server already holds this host; the message names it."""


def acquire(host: str, port: int, path: Path | None = None):
    """Take the host's GUI-server lock for this process.

    Pre: nothing of this server exists yet -- no subprocess spawned, no segment
    created or swept. Post: returns the open lock file, which must stay
    referenced (closing it releases the lock), with the holder's pid, port and
    host written into it; or raises ``AnotherServerRunningError`` naming the holder.
    Returns None where advisory file locks are unavailable (not POSIX).
    """
    try:
        import fcntl
    except ImportError:  # not POSIX: no lock, no second-server protection
        return None
    path = LOCK_PATH if path is None else path
    f = open(path, "a+")  # noqa: SIM115  # held open on purpose: the lock lives as long as this file object
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "holder unknown"
        f.close()
        raise AnotherServerRunningError(
            f"another lerobot-gui is already running on this host ({holder}). "
            "One server per host: use it, or stop it first."
        ) from None
    f.seek(0)
    f.truncate()
    f.write(f"pid {os.getpid()} port {port} host {host}\n")
    f.flush()
    return f
