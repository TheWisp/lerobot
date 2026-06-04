# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hotkeys binding store — ``~/.config/lerobot/hotkeys.json`` load + save.

The :mod:`lerobot.common.actions` manifest declares *which* actions
exist; this module declares *which input on which source* fires each
action. Defaults ship in the manifest's ``default_keyboard_keys``;
the JSON file stores user overrides. Loading produces a flat list of
``(action, source, binding)`` rows — the same shape the GUI's settings
page renders and the browser's keyboard handler consumes.

Schema (versioned for forward migrations):

.. code-block:: json

    {
      "version": 1,
      "bindings": [
        {"action": "exit_early", "source": "keyboard", "binding": "ArrowRight"},
        {"action": "rlt_success", "source": "quest", "binding": "right:5"},
        ...
      ]
    }

Source-specific binding conventions:

* ``keyboard`` — the ``KeyboardEvent.key`` string the browser emits
  (``"ArrowRight"``, ``"Escape"``, ``"r"``, ...). Matches what
  ``hotkeys.js`` sees on a ``keydown`` event.
* ``quest`` — ``"<hand>:<button_index>"`` (``"right:5"`` = right B,
  ``"left:5"`` = left Y, ``"right:3"`` = right joystick click, ...).
  Hand is ``left`` or ``right``; button index follows the WebXR
  gamepad mapping the Quest browser exposes.
* ``gamepad`` — the button index as a string (``"0"`` = A, ``"1"`` = B,
  ``"2"`` = X, ``"3"`` = Y, ...). Web Gamepad API standardised order.

Cross-source duplicates are expected and meaningful — keyboard ``r``
AND Quest ``right:5`` both mapping to ``rlt_success`` is the design
(operator can fire from either device). Intra-source duplicates fire
N actions on one press (legacy compound: keyboard ``ArrowLeft`` to
both ``rerecord_episode`` and ``exit_early``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from lerobot.common.actions import all_known_actions

logger = logging.getLogger(__name__)


# Sources the bindings store knows about. Used to validate the
# ``source`` field on save and to enumerate sections in the GUI.
KNOWN_SOURCES: tuple[str, ...] = ("keyboard", "quest", "gamepad")

# Current schema version. Bump + add a migration entry in
# ``_MIGRATIONS`` when the format changes.
SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class Binding:
    """One row in the bindings file: action ← (source, binding)."""

    action: str
    source: str
    binding: str


def default_bindings_path() -> Path:
    """The canonical location for ``hotkeys.json``.

    Respects ``XDG_CONFIG_HOME`` per XDG basedir; falls back to
    ``~/.config/lerobot/hotkeys.json`` otherwise. Mirrors how the
    existing GUI profiles (``robots/*.json``, ``teleops/*.json``)
    are located.
    """
    base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(base) / "lerobot" / "hotkeys.json"


def _migrate(raw: dict) -> dict:
    """Apply schema migrations until ``raw["version"] == SCHEMA_VERSION``.

    Raises ``ValueError`` if the version is newer than this code knows.
    Allows older versions to load by migrating in-place; downstream
    fork or older-saved files keep working across upgrades.
    """
    version = raw.get("version", 1)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"hotkeys.json version {version} is newer than this lerobot "
            f"supports (max {SCHEMA_VERSION}); please upgrade lerobot or "
            "delete the file to regenerate from defaults."
        )
    # Future: while version < SCHEMA_VERSION: raw = _MIGRATIONS[version](raw)
    return raw


def default_bindings() -> list[Binding]:
    """Synthesise bindings from the action manifest's ``default_keyboard_keys``.

    Used as the starting point when ``hotkeys.json`` doesn't exist —
    every action with a default keyboard binding gets a row, so the
    first run feels identical to the pre-channel keyboard behaviour
    without the user having to configure anything.

    Maps the action manifest's pynput-style key names (``"right"``,
    ``"esc"``) to the browser's ``KeyboardEvent.key`` strings
    (``"ArrowRight"``, ``"Escape"``) so the same row works for both
    sources without duplication.
    """
    out: list[Binding] = []
    for action in all_known_actions():
        for raw_key in action.default_keyboard_keys:
            out.append(
                Binding(
                    action=action.name,
                    source="keyboard",
                    binding=_pynput_to_browser_key(raw_key),
                )
            )
    # First-cut Quest defaults (mirror the hardcoded map in the Quest
    # VR teleop's button-edge dispatcher). Once the user saves an
    # override these stop applying — but until then they ship.
    out.extend(
        [
            Binding(action="rlt_success", source="quest", binding="right:5"),
            Binding(action="rlt_abort", source="quest", binding="left:5"),
            Binding(action="rlt_ignore", source="quest", binding="right:3"),
            Binding(action="rlt_toggle_engage", source="quest", binding="left:3"),
        ]
    )
    return out


def _pynput_to_browser_key(raw: str) -> str:
    """Translate a pynput-style key string (the manifest's lingua franca)
    into the matching ``KeyboardEvent.key`` value the browser sees.

    The manifest uses pynput's short names because that's what the
    CLI's ``init_keyboard_listener`` consumes; the browser's
    ``keydown`` handler sees ``ArrowRight`` etc. This mapping is the
    only place the two namespaces are reconciled.
    """
    return _PYNPUT_TO_BROWSER.get(raw, raw)


_PYNPUT_TO_BROWSER: dict[str, str] = {
    "right": "ArrowRight",
    "left": "ArrowLeft",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "esc": "Escape",
    "space": " ",
    "enter": "Enter",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    # Single-character keys (``"r"``, ``"e"``) pass through unchanged.
}


def load_bindings(path: Path | None = None) -> list[Binding]:
    """Read bindings from ``path`` (default :func:`default_bindings_path`).

    On a missing file, returns :func:`default_bindings` so the first
    run feels configured. Malformed JSON or a wrong-shape payload
    raises ``ValueError`` rather than silently falling back — a
    user-edited file with a typo deserves visibility.
    """
    p = path if path is not None else default_bindings_path()
    if not p.exists():
        logger.info("hotkeys: %s missing, using defaults", p)
        return default_bindings()

    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"hotkeys.json at {p} is not valid JSON: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(f"hotkeys.json at {p} must be a JSON object, got {type(raw).__name__}")

    raw = _migrate(raw)
    rows = raw.get("bindings", [])
    if not isinstance(rows, list):
        raise ValueError(f"hotkeys.json at {p}: 'bindings' must be a list")

    out: list[Binding] = []
    known_actions = {a.name for a in all_known_actions()}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"hotkeys.json at {p}: bindings[{i}] must be an object")
        for field in ("action", "source", "binding"):
            if field not in row or not isinstance(row[field], str):
                raise ValueError(f"hotkeys.json at {p}: bindings[{i}] is missing string field {field!r}")
        action, source, binding = row["action"], row["source"], row["binding"]
        if action not in known_actions:
            logger.warning(
                "hotkeys: bindings[%d] references unknown action %r — kept as-is (lerobot might add it later)",
                i,
                action,
            )
        if source not in KNOWN_SOURCES:
            logger.warning("hotkeys: bindings[%d] references unknown source %r — kept as-is", i, source)
        out.append(Binding(action=action, source=source, binding=binding))
    return out


def save_bindings(bindings: list[Binding], path: Path | None = None) -> None:
    """Persist ``bindings`` atomically to ``path``.

    Atomic write via ``tmp + os.replace`` so a crash mid-write leaves
    the previous file intact instead of a half-written one. Mirrors
    the existing GUI persistence pattern.
    """
    p = path if path is not None else default_bindings_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SCHEMA_VERSION,
        "bindings": [{"action": b.action, "source": b.source, "binding": b.binding} for b in bindings],
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)
    logger.info("hotkeys: wrote %d bindings to %s", len(bindings), p)


def bindings_to_dict(bindings: list[Binding]) -> dict[str, list[dict[str, str]]]:
    """Group bindings by source for frontend consumption.

    The frontend's keyboard handler does ``bindings["keyboard"]``
    lookup on the ``KeyboardEvent.key`` value, then forwards each
    bound action. Grouping by source up-front makes the per-event
    dispatch a single dict lookup instead of a list filter.
    """
    out: dict[str, list[dict[str, str]]] = {s: [] for s in KNOWN_SOURCES}
    for b in bindings:
        out.setdefault(b.source, []).append({"action": b.action, "binding": b.binding})
    return out
