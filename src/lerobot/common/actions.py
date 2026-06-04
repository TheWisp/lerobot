# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Action manifest — single source of truth for the channel's vocabulary.

The :mod:`lerobot.common.control_channel` is semantics-free: it knows
how to register names + dispatch sources, but it doesn't know that
``rlt_success`` means a +1 terminal reward. The vocabulary itself —
which names exist, what they're called by humans, which workflow
they belong to — lives here, so:

* The orchestrator subprocess imports this module at startup, calls
  :func:`actions_for_workflow` for its workflow, and registers each
  returned action on its channel.
* The GUI process imports the same module to render the hotkeys
  settings page. No log-parsing, no cross-process protocol: a
  type-checked function call on both sides.

When both processes resolve the same workflow + flags, they produce
identical action sets — by construction, since they're calling the
same function with the same args.

See :doc:`/CONTROL_CHANNEL` ("Action manifest — shared Python module")
for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionDecl:
    """Declarative description of a flow-control action.

    Subprocess uses :attr:`name` to register on the channel,
    :attr:`default_keyboard_keys` to seed the pynput source's
    bindings (CLI mode only — GUI mode bindings come from
    ``hotkeys.json``). The GUI uses :attr:`name` + :attr:`description`
    to render rows in the settings page.

    ``name`` is the channel registry key — bare for globals
    (``"intervene"``) or dotted for scoped (``"hvla.rlt.success"``).
    Scopes are not enforced at the channel layer; they exist for
    conflict-detection in the future hotkeys UI.
    """

    name: str
    description: str
    default_keyboard_keys: tuple[str, ...] = field(default_factory=tuple)


# ── Globals — fire in any workflow ───────────────────────────────────────

GLOBAL_ACTIONS: tuple[ActionDecl, ...] = ()


# ── Record / replay loop actions ────────────────────────────────────────

RECORD_ACTIONS: tuple[ActionDecl, ...] = (
    ActionDecl(
        name="exit_early",
        description="End the current episode or reset phase and advance.",
        default_keyboard_keys=("right", "left", "esc"),
    ),
    ActionDecl(
        name="rerecord_episode",
        description="Discard the current episode and re-record it.",
        default_keyboard_keys=("left",),
    ),
    ActionDecl(
        name="stop_recording",
        description="End the recording session cleanly.",
        default_keyboard_keys=("esc",),
    ),
)


# ── HVLA inference loop actions ─────────────────────────────────────────

# Same flow-control verbs as record — HVLA's loop polls them identically.
HVLA_ACTIONS: tuple[ActionDecl, ...] = RECORD_ACTIONS


# ── HVLA RLT (online RL) actions ────────────────────────────────────────

HVLA_RLT_ACTIONS: tuple[ActionDecl, ...] = (
    ActionDecl(
        name="rlt_success",
        description="Episode SUCCESS (+1 terminal reward; ends episode).",
        default_keyboard_keys=("r",),
    ),
    ActionDecl(
        name="rlt_abort",
        description="Episode ABORT (negative terminal reward; ends episode; trajectory saved).",
        default_keyboard_keys=("left",),
    ),
    ActionDecl(
        name="rlt_ignore",
        description="Discard this episode as OOD (rolls back the dataset).",
        default_keyboard_keys=("down",),
    ),
    ActionDecl(
        name="rlt_toggle_engage",
        description="Toggle RL actor on/off (does NOT end the episode).",
        default_keyboard_keys=("e",),
    ),
)


# ── Public API ──────────────────────────────────────────────────────────


def actions_for_workflow(workflow: str, *, rlt_mode: bool = False) -> list[ActionDecl]:
    """Return the actions ``workflow`` should register on its channel.

    Both the subprocess (registering) and the GUI process (rendering
    the settings page) call this. Given the same arguments, both
    produce the same list — no inter-process protocol needed.

    Args:
        workflow: One of ``"record"`` / ``"hvla"``. Unknown workflows
            return only :data:`GLOBAL_ACTIONS` so the channel still
            registers something coherent rather than raising at
            startup.
        rlt_mode: When True and ``workflow="hvla"``, include the
            RLT-specific terminal-signal actions
            (``rlt_success``, ``rlt_abort``, etc).

    Returns:
        A flat list of :class:`ActionDecl`. Duplicates within
        a workflow are possible by construction (e.g. ``exit_early``
        appears via the multi-key compound on ``left``) and the
        channel handles them — re-registering replaces bindings.
    """
    actions: list[ActionDecl] = list(GLOBAL_ACTIONS)
    if workflow == "record":
        actions.extend(RECORD_ACTIONS)
    elif workflow == "hvla":
        actions.extend(HVLA_ACTIONS)
        if rlt_mode:
            actions.extend(HVLA_RLT_ACTIONS)
    return actions


def all_known_actions() -> list[ActionDecl]:
    """Return the union of all actions across all known workflows.

    Used by the GUI's hotkeys settings page so the user can configure
    bindings for actions across every workflow at once, regardless of
    which subprocess (if any) is currently running.
    """
    seen: dict[str, ActionDecl] = {}
    for workflow in ("record", "hvla"):
        for rlt in (False, True):
            for action in actions_for_workflow(workflow, rlt_mode=rlt):
                seen.setdefault(action.name, action)
    return list(seen.values())


def register_actions(channel, actions: list[ActionDecl]) -> None:
    """Register each ``ActionDecl`` on ``channel`` with its default keys.

    A common pattern lives here so consumers don't reinvent the loop.
    Re-registering an existing action replaces its bindings — same
    semantics as ``channel.register`` itself.
    """
    for action in actions:
        if action.default_keyboard_keys:
            channel.register(action.name, keyboard_keys=action.default_keyboard_keys)
        else:
            channel.register(action.name)
