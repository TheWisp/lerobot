"""Every robot the GUI can launch must be launchable by all three scripts.

draccus resolves ``--robot.type`` from whatever subclasses happen to be imported,
so a robot is silently unavailable to a script that forgets to import it. The GUI
offers one robot dropdown for teleoperate, record and replay alike, so a robot
registered by only some of them turns into "invalid choice" at launch -- which is
how ``virtual_bi_so107``, the project's own hardware-free testbench, ended up
usable for teleoperate and record but not replay.

That gap mattered beyond one broken button: the virtual robot exists so these
flows can be verified without hardware, and replay was the one flow it could not
verify.

Parity is the invariant worth pinning, not a hardcoded robot list -- a new robot
added to one script and forgotten in another fails here, with no test to update.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import lerobot
from lerobot.robots.config import RobotConfig

# The scripts the GUI's Run tab can launch with a user-selected robot profile.
LAUNCH_SCRIPTS = [
    "lerobot.scripts.lerobot_teleoperate",
    "lerobot.scripts.lerobot_record",
    "lerobot.scripts.lerobot_replay",
]


def _declared_robot_imports(module_name: str) -> set[str]:
    """Robot names a script imports, read from source.

    Deliberately static. Registration is global and cumulative, so importing the
    scripts and reading ``RobotConfig.get_known_choices()`` would let whichever
    script imported first register a robot on behalf of the others -- hiding the
    exact asymmetry under test. Reading source also keeps this runnable in envs
    where a script's optional policy dependencies won't import.
    """
    src_root = Path(lerobot.__file__).parent
    path = src_root / Path(*module_name.split(".")[1:]).with_suffix(".py")
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "lerobot.robots":
            imported.update(alias.name for alias in node.names)
    assert imported, f"{module_name} imports nothing from lerobot.robots — parser drifted?"
    return imported


@pytest.fixture(scope="module")
def per_script_choices() -> dict[str, set[str]]:
    return {name: _declared_robot_imports(name) for name in LAUNCH_SCRIPTS}


def test_launch_scripts_register_the_same_robots(per_script_choices):
    reference_name, reference = next(iter(per_script_choices.items()))
    for module_name, imported in per_script_choices.items():
        missing = reference - imported
        extra = imported - reference
        assert not missing and not extra, (
            f"{module_name} imports a different robot set than {reference_name}. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}. "
            "A robot imported by only some launch scripts is 'invalid choice' at launch."
        )


@pytest.mark.parametrize("module_name", LAUNCH_SCRIPTS)
def test_virtual_robot_is_launchable_everywhere(module_name):
    """The hardware-free testbench must reach every flow, or it can't verify them."""
    assert "virtual_bi_so107" in _declared_robot_imports(module_name), (
        f"{module_name} cannot launch the virtual robot, so that flow has no hardware-free end-to-end path."
    )


def test_virtual_robot_type_name_is_what_the_scripts_import():
    """Guard the guard: the imported module name must be the registered type name.

    Every assertion above matches on the string ``virtual_bi_so107``. If the
    package and its registered ``--robot.type`` ever diverge, those assertions
    would keep passing while the CLI rejected the type.
    """
    import lerobot.robots.virtual_bi_so107.config_virtual_bi_so107  # noqa: F401

    assert "virtual_bi_so107" in RobotConfig.get_known_choices()
