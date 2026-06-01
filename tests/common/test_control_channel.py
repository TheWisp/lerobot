# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for :mod:`lerobot.common.control_channel`.

Covers the registry semantics and the stdin JSON-lines source. The
pynput keyboard source is hard to test in CI (needs a display) and is
exercised by the existing CLI integration tests via
:func:`init_keyboard_listener` whenever a record run happens.
"""

from __future__ import annotations

import io
import time

import pytest

from lerobot.common.control_channel import ControlChannel, init_control_channel


def _wait_for(predicate, timeout: float = 1.0, interval: float = 0.01) -> None:
    """Spin until ``predicate()`` returns truthy or timeout fires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"predicate did not become truthy within {timeout}s")


# ── Registry semantics ───────────────────────────────────────────────────


def test_register_adds_events_key():
    """Registering an action makes ``events[name]`` queryable from the
    same tick — no race where the consumer polls a key that doesn't
    exist yet."""
    ch = ControlChannel()
    ch.register("next_episode", keyboard_key="right")
    assert "next_episode" in ch.events
    assert ch.events["next_episode"] is False


def test_register_is_idempotent_and_replaces_bindings(caplog):
    """Re-registering the same name replaces its key binding and logs
    the change. Catches the accidental double-registration that would
    otherwise silently broaden a hotkey."""
    import logging

    caplog.set_level(logging.INFO, logger="lerobot.common.control_channel")
    ch = ControlChannel()
    ch.register("intervene", keyboard_key="space")
    ch.register("intervene", keyboard_key="b")
    # Still one entry — not two — and the new binding wins.
    assert ch.registered_names().count("intervene") == 1
    assert ch._actions["intervene"].keyboard_keys == ("b",)
    assert any("re-registering" in r.message for r in caplog.records)


def test_register_rejects_both_singular_and_plural_keys():
    """``keyboard_key`` and ``keyboard_keys`` are mutually exclusive —
    accidentally passing both would silently lose one binding."""
    ch = ControlChannel()
    with pytest.raises(ValueError, match="not both"):
        ch.register("oops", keyboard_key="a", keyboard_keys=("b",))


def test_register_supports_multi_key_actions():
    """Modelling the legacy compound behavior: ``exit_early`` is
    triggered by right OR left OR esc."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_keys=("right", "left", "esc"))
    assert ch._actions["exit_early"].keyboard_keys == ("right", "left", "esc")


def test_emit_unregistered_name_is_dropped(caplog):
    """Public ``emit`` API validates against the registry — a future
    teleop calling ``channel.emit("foo")`` without registering first
    gets a warning, not a silent dict-growth that masks the typo."""
    ch = ControlChannel()
    ch.emit("never_registered", source="test")
    assert "never_registered" not in ch.events
    assert any("unknown action" in r.message for r in caplog.records)


def test_emit_registered_name_flips_event():
    ch = ControlChannel()
    ch.register("intervene", keyboard_key="space")
    ch.emit("intervene", source="test")
    assert ch.events["intervene"] is True


# ── Stdin source ─────────────────────────────────────────────────────────


def test_stdin_emits_registered_action():
    """Headline contract: a well-formed JSON line for a registered
    action mutates the events dict the orchestrator polls."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_key="right")
    stream = io.StringIO('{"v": 1, "cmd": "exit_early"}\n')
    ch.attach_stdin(stream)
    try:
        _wait_for(lambda: ch.events["exit_early"])
    finally:
        ch.stop()


def test_stdin_drops_unregistered_action(caplog):
    """An action sent from a stale GUI client (e.g. a button that was
    later renamed) must not silently leak into ``events``."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_key="right")
    stream = io.StringIO('{"v": 1, "cmd": "intervene"}\n')  # not registered
    ch.attach_stdin(stream)
    try:
        _wait_for(lambda: any("unknown action" in r.message for r in caplog.records))
        assert "intervene" not in ch.events
    finally:
        ch.stop()


def test_stdin_drops_malformed_json(caplog):
    """A torn write / non-JSON line must not crash the reader thread."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_key="right")
    stream = io.StringIO('not-json\n{"v": 1, "cmd": "exit_early"}\n')
    ch.attach_stdin(stream)
    try:
        _wait_for(lambda: ch.events["exit_early"])
        assert any("malformed JSON" in r.message for r in caplog.records)
    finally:
        ch.stop()


def test_stdin_drops_unsupported_version(caplog):
    """``v`` is the schema-evolution lever. An older subprocess seeing
    a future ``v: 2`` must drop the message rather than mis-dispatch."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_key="right")
    stream = io.StringIO('{"v": 99, "cmd": "exit_early"}\n')
    ch.attach_stdin(stream)
    try:
        _wait_for(lambda: any("unsupported message version" in r.message for r in caplog.records))
        assert ch.events["exit_early"] is False
    finally:
        ch.stop()


def test_stdin_handles_blank_lines():
    """Trailing newlines / keep-alives shouldn't trip the parser."""
    ch = ControlChannel()
    ch.register("stop_recording", keyboard_key="esc")
    stream = io.StringIO('\n\n{"v": 1, "cmd": "stop_recording"}\n\n')
    ch.attach_stdin(stream)
    try:
        _wait_for(lambda: ch.events["stop_recording"])
    finally:
        ch.stop()


def test_stdin_handles_multiple_lines_in_sequence():
    """Several commands back-to-back all apply — proves the reader
    drains the stream rather than treating one line as terminal."""
    ch = ControlChannel()
    ch.register("rerecord_episode", keyboard_key="left")
    ch.register("exit_early", keyboard_keys=("right", "left", "esc"))
    stream = io.StringIO('{"v": 1, "cmd": "rerecord_episode"}\n{"v": 1, "cmd": "exit_early"}\n')
    ch.attach_stdin(stream)
    try:
        _wait_for(lambda: ch.events["rerecord_episode"] and ch.events["exit_early"])
    finally:
        ch.stop()


# ── Init / env var handling ──────────────────────────────────────────────


def test_init_control_channel_returns_drop_in_shape():
    """``init_control_channel`` returns ``(channel, events)`` so call
    sites that did ``listener, events = init_keyboard_listener()`` are
    one-character edits. Events is the same dict reference, not a copy."""
    ch, events = init_control_channel(pynput=False, stdin=False)
    assert events is ch.events
    ch.stop()


def test_init_control_channel_pre_registers_nothing():
    """Action vocabulary is consumer-owned — the channel doesn't bake
    in any defaults. ``init_keyboard_listener`` adds legacy verbs on
    top; that's its job, not the channel's."""
    ch, events = init_control_channel(pynput=False, stdin=False)
    assert events == {}
    assert ch.registered_names() == []
    ch.stop()


def test_init_control_channel_env_var_drives_stdin(monkeypatch):
    """``stdin=None`` (default) consults LEROBOT_CONTROL_CHANNEL_STDIN.
    The GUI sets this on subprocess launch; CLI runs leave it unset.
    Misreading the env var means GUI-launched processes can't be
    controlled, so pin the behavior."""
    monkeypatch.delenv("LEROBOT_CONTROL_CHANNEL_STDIN", raising=False)
    ch_no_env, _ = init_control_channel(pynput=False, stdin=None)
    assert not any(s.__class__.__name__ == "_StdinSource" for s in ch_no_env._sources)
    ch_no_env.stop()

    monkeypatch.setenv("LEROBOT_CONTROL_CHANNEL_STDIN", "1")
    ch_env, _ = init_control_channel(pynput=False, stdin=None)
    assert any(s.__class__.__name__ == "_StdinSource" for s in ch_env._sources)
    ch_env.stop()


def test_stop_is_idempotent():
    """Cleanup paths in shutdown may call .stop() twice — must be safe."""
    ch = ControlChannel()
    ch.attach_stdin(io.StringIO(""))
    ch.stop()
    ch.stop()


# ── Legacy contract preserved ────────────────────────────────────────────


def test_init_keyboard_listener_preserves_three_verb_contract():
    """``init_keyboard_listener`` is the legacy entry point — it must
    leave ``events`` with at least the three legacy keys defined, all
    initially False. Otherwise the record loop's
    ``events["exit_early"]`` poll raises KeyError on the first tick."""
    from lerobot.common.control_utils import init_keyboard_listener

    # pynput will silently fail in CI (no display) — that's expected
    # and exercised by the headless fallback. Stdin is also off here
    # because the env var isn't set.
    channel, events = init_keyboard_listener()
    try:
        assert events["exit_early"] is False
        assert events["rerecord_episode"] is False
        assert events["stop_recording"] is False
    finally:
        channel.stop()
