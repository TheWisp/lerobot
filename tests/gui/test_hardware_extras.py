# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What a chosen robot needs, answered before anything is launched.

Every hardware dependency is declared and guarded where it is used, which
means the answer arrives inside a subprocess, at connect time, after the GUI
has committed to a launch. The GUI knows the robot's type the moment the
profile is chosen, so it can ask the same question then — with no hardware
attached and nothing spawned.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from lerobot.utils.hardware_extras import (
    IMPORT_FOR_EXTRA,
    extras_for_robot,
    install_command,
    missing_extras,
)

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture(autouse=True)
def _registered():
    """The registry answers for a type only once its module is imported,
    which is the GUI's job before it asks."""
    from lerobot.gui.api.robot import _ensure_configs_loaded

    _ensure_configs_loaded()


def _declared_extras() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return set(data["project"].get("optional-dependencies") or {})


def test_every_extra_named_here_exists_in_pyproject():
    """Parity, not a list someone maintains: an extra renamed in pyproject
    must not leave this table pointing at nothing, because the message it
    produces is an install command a person will paste."""
    unknown = sorted(set(IMPORT_FOR_EXTRA) - _declared_extras())
    assert not unknown, f"named here but not an extra: {unknown}"


def test_every_hardware_guard_in_the_tree_is_covered():
    """The guards are the source of truth for what is needed. Anything they
    can raise about on the hardware path has to be answerable here, or a
    robot can still die in a subprocess for a reason nothing predicted."""
    import re

    root = Path(__file__).resolve().parents[2] / "src" / "lerobot"
    pattern = re.compile(r"require_package\([^)]*extra=\"([^\"]+)\"", re.S)
    guarded: set[str] = set()
    for area in ("robots", "cameras", "motors", "teleoperators"):
        for path in (root / area).rglob("*.py"):
            guarded |= set(pattern.findall(path.read_text()))

    uncovered = sorted(guarded - set(IMPORT_FOR_EXTRA))
    assert not uncovered, f"hardware guards can raise about {uncovered}, which this table cannot answer for"


def test_a_robot_reports_the_extras_its_motors_need():
    """An SO follower speaks to Feetech servos over a serial bus, and none of
    that is a core dependency."""
    extras = extras_for_robot("so107_follower", cameras={}, teleop_type=None)
    assert "feetech" in extras
    assert "deepdiff-dep" in extras, "every motor bus needs it; it is not the arm's own"


def test_a_realsense_camera_is_counted_as_well_as_the_arm():
    """The camera set is chosen per profile, so its dependencies are the
    profile's too — this is the one that went missing in practice."""
    extras = extras_for_robot(
        "so107_follower",
        cameras={"front": {"type": "intelrealsense"}, "wrist": {"type": "opencv"}},
        teleop_type=None,
    )
    assert "intelrealsense" in extras
    assert "feetech" in extras


def test_an_unknown_type_is_not_an_error():
    """A robot this table says nothing about must launch as it always did:
    the guard in its constructor is still there to catch it."""
    assert extras_for_robot("no_such_robot", cameras={}, teleop_type=None) == set()


def test_what_is_installed_here_is_not_reported_missing(monkeypatch):
    """Otherwise the check refuses every launch.

    Pinned to a module the standard library always has, rather than asserting
    that this machine happens to have an optional extra installed — which is
    a fact about the machine and would make the test say different things in
    different places.
    """
    from lerobot.utils import hardware_extras

    monkeypatch.setitem(hardware_extras.IMPORT_FOR_EXTRA, "deepdiff-dep", ("json",))
    assert missing_extras({"deepdiff-dep"}) == set()


def test_something_absent_is_reported_with_a_command_to_fix_it(monkeypatch):
    """Also pinned: this used to skip wherever the driver happened to be
    installed, which is exactly the machine where it most needs to run."""
    from lerobot.utils import hardware_extras

    monkeypatch.setitem(hardware_extras.IMPORT_FOR_EXTRA, "intelrealsense", ("a_package_no_machine_has",))
    absent = missing_extras({"intelrealsense"})
    assert absent == {"intelrealsense"}
    command = install_command(absent)
    assert "--extra intelrealsense" in command
    assert command.startswith("uv sync")


class TestTheLaunchRefusesRatherThanDying:
    """The point of asking early: the operator is told before a subprocess is
    spawned, not after it has exited.

    Both cases pin the import table rather than reading whatever this machine
    happens to have installed — otherwise the same test asserts opposite
    things on two developers' laptops.
    """

    ROBOT = {"type": "so107_follower", "cameras": {"front": {"type": "opencv"}}}

    def test_a_missing_dependency_refuses_the_launch_with_the_command(self, monkeypatch):
        from fastapi import HTTPException

        from lerobot.gui.api.run import _refuse_if_deps_missing
        from lerobot.utils import hardware_extras

        monkeypatch.setitem(hardware_extras.IMPORT_FOR_EXTRA, "feetech", ("a_package_no_machine_has",))
        with pytest.raises(HTTPException) as caught:
            _refuse_if_deps_missing(self.ROBOT, {"type": "no_input"})

        assert caught.value.status_code == 400
        assert "feetech" in caught.value.detail
        assert "uv sync --extra feetech" in caught.value.detail

    def test_hardware_that_can_be_driven_here_is_not_refused(self, monkeypatch):
        """The complement: a check that refused everything would pass the test
        above and stop the tab working."""
        from lerobot.gui.api.run import _refuse_if_deps_missing
        from lerobot.utils import hardware_extras

        # Every extra this robot declares, answerable by a module that is
        # always importable.
        for extra in ("feetech", "pyserial-dep", "deepdiff-dep"):
            monkeypatch.setitem(hardware_extras.IMPORT_FOR_EXTRA, extra, ("json",))

        _refuse_if_deps_missing(self.ROBOT, {"type": "no_input"})

    def test_the_endpoint_refuses_before_anything_is_launched(self, monkeypatch):
        """Through the endpoint, not the helper: the helper being right says
        nothing about it being called, and the whole value of asking early is
        that no subprocess is spawned."""
        import asyncio
        from unittest.mock import patch

        from fastapi import HTTPException

        from lerobot.gui.api.run import TeleoperateRequest, start_teleoperate
        from lerobot.utils import hardware_extras

        monkeypatch.setitem(hardware_extras.IMPORT_FOR_EXTRA, "feetech", ("a_package_no_machine_has",))
        launched = []

        async def never(*args, **kwargs):
            launched.append(args)

        async def run():
            req = TeleoperateRequest(robot=self.ROBOT, teleop={"type": "no_input", "fields": {}})
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._release_preview_cameras", never),
                patch("lerobot.gui.api.run._launch_subprocess", never),
            ):
                await start_teleoperate(req)

        with pytest.raises(HTTPException) as caught:
            asyncio.run(run())
        assert caught.value.status_code == 400
        assert not launched, "the launch went ahead anyway"
