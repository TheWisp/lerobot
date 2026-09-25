"""Every shared-memory segment a test process creates is named after that process.

Tests create real ``/dev/shm`` segments -- observation streams, overlay and aux
buffers -- and every GUI app a test starts sweeps its family's names on the way
up, without asking whose they are. Under the production names, or under any
name that only extends them, one xdist worker's app deletes another worker's
live segments, and a suite run beside a developer's GUI deletes that GUI's.
With the process id in front, each sweep matches only its own process's names.

Applied at configure, before the test modules are imported, so a module that
copies a prefix when it is imported copies the tagged one; the one module that
may already have copied it is updated here. ``tests/test_shm_namespace.py``
fails if any loaded module still holds a production name.

Not reached: a subprocess a test starts imports the modules afresh and uses
the production names. No test shares a segment across that boundary; one that
does has to hand its child the name.
"""

import glob
import os
import sys


def _tag() -> str:
    return f"t{os.getpid()}_"


def pytest_configure(config):
    import lerobot.overlays.aux_ipc as aux_ipc
    import lerobot.overlays.overlay_ipc as overlay_ipc
    import lerobot.robots.obs_stream as obs_stream

    tag = _tag()
    if obs_stream.SHM_PREFIX.startswith(tag):
        return  # configured twice in one process: the names are already ours
    obs_stream.SHM_PREFIX = tag + obs_stream.SHM_PREFIX
    overlay_ipc._PREFIX = tag + overlay_ipc._PREFIX
    overlay_ipc._CONTROL_SHM = f"/dev/shm/{overlay_ipc._PREFIX}control"  # nosec B108  # POSIX shm path
    aux_ipc._PREFIX = tag + aux_ipc._PREFIX
    standalone = sys.modules.get("lerobot.overlays.standalone")
    if standalone is not None:
        standalone.SHM_PREFIX = obs_stream.SHM_PREFIX


def pytest_unconfigure(config):
    """Remove what this process left behind: no sweep but its own matches it."""
    for path in glob.glob(f"/dev/shm/{_tag()}lerobot_*"):  # nosec B108  # POSIX shm path
        try:
            os.unlink(path)  # safe-destruct: this process's own tagged segments, at its exit
        except OSError:
            pass
