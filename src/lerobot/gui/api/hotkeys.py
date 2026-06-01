# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hotkeys REST API — manifest, bindings, connection status.

Sits on top of :mod:`lerobot.common.actions` (the action vocabulary)
and :mod:`lerobot.common.hotkeys` (the bindings store). Three thin
endpoints:

* ``GET /api/hotkeys/actions``  — full union of all known actions.
* ``GET /api/hotkeys/bindings`` — current ``(action, source, binding)`` rows.
* ``POST /api/hotkeys/bindings`` — replace the bindings list.
* ``GET /api/hotkeys/status``   — per-source live connection state.

The browser's hotkey handler hits these to render the settings page
and to know which keys map to which actions. Subprocess never queries
this API — it derives its own bindings from the shared action manifest
imported in-process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lerobot.common.actions import all_known_actions
from lerobot.common.hotkeys import (
    KNOWN_SOURCES,
    Binding,
    bindings_to_dict,
    load_bindings,
    save_bindings,
)

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hotkeys", tags=["hotkeys"])

_app_state: AppState = None  # type: ignore


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


# ── Request / response shapes ────────────────────────────────────────────


class BindingRow(BaseModel):
    """One ``(action, source, binding)`` row over the wire."""

    action: str
    source: str
    binding: str


class BindingsPayload(BaseModel):
    """``POST /bindings`` body — full replacement of the current set."""

    bindings: list[BindingRow]


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/actions")
async def get_actions() -> dict:
    """Return the union of all actions across all known workflows.

    The settings page renders one row per action regardless of which
    workflow is currently running, so the user can configure bindings
    for actions that aren't live yet.
    """
    return {
        "actions": [
            {
                "name": a.name,
                "description": a.description,
                "default_keyboard_keys": list(a.default_keyboard_keys),
            }
            for a in all_known_actions()
        ],
        "sources": list(KNOWN_SOURCES),
    }


@router.get("/bindings")
async def get_bindings() -> dict:
    """Return the current bindings, grouped by source for fast frontend lookup."""
    try:
        bindings = load_bindings()
    except ValueError as e:
        # The user-edited ``hotkeys.json`` is malformed. Surface the
        # error rather than silently falling back to defaults — that
        # would mask the typo and confuse the next save attempt.
        raise HTTPException(500, str(e)) from e
    return {
        "by_source": bindings_to_dict(bindings),
        "flat": [{"action": b.action, "source": b.source, "binding": b.binding} for b in bindings],
    }


@router.post("/bindings")
async def post_bindings(payload: BindingsPayload) -> dict:
    """Replace the bindings file with ``payload``.

    Validates source names against :data:`KNOWN_SOURCES`; unknown
    actions are logged and kept so a forward-incompatible save
    from an upgraded GUI doesn't lose data.
    """
    for i, row in enumerate(payload.bindings):
        if row.source not in KNOWN_SOURCES:
            raise HTTPException(
                400, f"bindings[{i}] has unknown source {row.source!r}; expected one of {KNOWN_SOURCES}"
            )
    bindings = [Binding(action=r.action, source=r.source, binding=r.binding) for r in payload.bindings]
    save_bindings(bindings)
    return {"saved": len(bindings)}


@router.get("/status")
async def get_status() -> dict:
    """Per-source live connection state for the page's badges.

    Keyboard is always considered connected (the browser captures
    locally; there's nothing to detect). Gamepad and Quest are
    inherently dynamic — they require the browser to do the
    detection (``navigator.getGamepads()`` for gamepad, an active
    WebXR session for Quest). The server has no view into either;
    this endpoint just declares the schema and the per-source
    "detection lives on the client" contract.
    """
    return {
        "sources": {
            "keyboard": {"connected": True, "detected_by": "server"},
            "gamepad": {"connected": None, "detected_by": "client"},
            "quest": {"connected": None, "detected_by": "client"},
        }
    }
