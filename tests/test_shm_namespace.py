"""A test process names every shared-memory segment after itself.

Every GUI app a test starts sweeps its family's names on the way up. The
overlay tests failed on CI with "overlay seq=1" because an app in another
xdist worker deleted their buffers: they were named ``lerobot_overlay_t<pid>_``,
which that sweep's ``lerobot_overlay_*`` matches. ``tests/fixtures/shm_namespace.py``
puts the pid first; these check nothing escapes it.
"""

import glob
import os
import re
import sys

import pytest

_PRODUCTION = re.compile(r"^(/dev/shm/)?lerobot_(obs|overlay|aux)_")


def test_every_family_is_named_after_this_process():
    import lerobot.overlays.aux_ipc as aux_ipc
    import lerobot.overlays.overlay_ipc as overlay_ipc
    import lerobot.robots.obs_stream as obs_stream

    tag = f"t{os.getpid()}_"
    for prefix in (obs_stream.SHM_PREFIX, overlay_ipc._PREFIX, aux_ipc._PREFIX):
        assert prefix.startswith(tag), prefix
    assert f"/dev/shm/{overlay_ipc._PREFIX}control" == overlay_ipc._CONTROL_SHM


def test_an_overlay_buffer_is_out_of_every_other_sweeps_reach():
    """The failure itself: what the buffer creates is found under this
    process's name and under no name another process's sweep matches."""
    import lerobot.overlays.overlay_ipc as overlay_ipc

    buffer = overlay_ipc.SharedOverlayBuffer(cameras={"front": (8, 8)}, model="test", create=True)
    try:
        created = [os.path.basename(p) for p in glob.glob(f"/dev/shm/{overlay_ipc._PREFIX}*")]
        assert created, "the buffer created nothing under this process's name"
        assert not [n for n in created if _PRODUCTION.match(n)], created
    finally:
        buffer.cleanup()


def test_no_loaded_module_holds_a_production_name():
    """A module that copied a prefix before configure ran would still create
    and sweep the production names; ``_FAMILY_PREFIX`` only lists taps."""
    pytest.importorskip("fastapi")
    import lerobot.gui.server  # noqa: F401  # loads the modules that name segments
    import lerobot.overlays.standalone  # noqa: F401  # copies the tap's prefix when imported

    held = [
        f"{name}.{attr} = {value!r}"
        for name, module in list(sys.modules.items())
        if name.startswith("lerobot")
        for attr, value in list(vars(module).items())
        if attr != "_FAMILY_PREFIX" and isinstance(value, str) and _PRODUCTION.match(value)
    ]
    assert not held, held
