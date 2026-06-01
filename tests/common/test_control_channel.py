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


def test_one_keyboard_key_fires_multiple_actions():
    """The legacy compound: pressing ``left`` had to fire BOTH
    ``rerecord_episode`` AND ``exit_early`` (the inner record-loop polls
    exit_early to break; the outer loop polls rerecord_episode to redo).
    Multi-key bindings on the channel side preserve this; the pynput
    source's on_press delegates here so it's exercised by the same code
    path the tests cover."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_keys=("right", "left", "esc"))
    ch.register("rerecord_episode", keyboard_key="left")
    ch.register("stop_recording", keyboard_key="esc")

    # left -> BOTH exit_early AND rerecord_episode.
    fired = ch.emit_for_keyboard_key("left")
    assert set(fired) == {"exit_early", "rerecord_episode"}
    assert ch.events["exit_early"] is True
    assert ch.events["rerecord_episode"] is True
    assert ch.events["stop_recording"] is False

    # Reset and try esc -> BOTH exit_early AND stop_recording.
    ch.events["exit_early"] = False
    ch.events["rerecord_episode"] = False
    fired = ch.emit_for_keyboard_key("esc")
    assert set(fired) == {"exit_early", "stop_recording"}
    assert ch.events["exit_early"] is True
    assert ch.events["stop_recording"] is True

    # right -> just exit_early (no compound).
    ch.events["exit_early"] = False
    fired = ch.emit_for_keyboard_key("right")
    assert fired == ["exit_early"]


def test_emit_for_keyboard_key_unbound_key_is_noop():
    """A key with no binding emits nothing — the pynput source can
    forward every keypress without worrying about polluting events."""
    ch = ControlChannel()
    ch.register("intervene", keyboard_key="space")
    fired = ch.emit_for_keyboard_key("z")
    assert fired == []
    assert ch.events["intervene"] is False


# ── HVLA RLT compat (legacy listener.on_press chaining) ─────────────────


class _FakeKey:
    """Minimal stand-in for ``pynput.keyboard.Key`` / ``KeyCode``.

    Has a ``name`` attribute for special keys ("right", "esc") and a
    ``char`` attribute for letter keys. Matches the duck-typed surface
    ``_pynput_key_to_str`` expects so the compat tests don't need a
    real display.
    """

    def __init__(self, *, name: str | None = None, char: str | None = None) -> None:
        if name is not None:
            self.name = name
        if char is not None:
            self.char = char


def test_on_press_default_dispatches_via_registry():
    """``channel.on_press`` defaults to a callable that runs the registry
    dispatch — so a legacy caller doing
    ``_orig = listener.on_press; _orig(key)`` gets the same behaviour as
    if the pynput source called the default."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_keys=("right", "left", "esc"))
    ch.register("rerecord_episode", keyboard_key="left")
    handler = ch.on_press
    assert callable(handler)

    handler(_FakeKey(name="left"))
    assert ch.events["exit_early"] is True
    assert ch.events["rerecord_episode"] is True


def test_on_press_setter_chains_with_captured_original():
    """The HVLA RLT pattern: capture the current handler, install a
    wrapper that does extra work then chains through. This pins the
    contract — replacing ``on_press`` does NOT silently drop the
    registry dispatch unless the new handler explicitly skips the
    chain (which RLT does for its own special keys)."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_key="right")
    ch.register("rlt_success", keyboard_key="r")

    _orig = ch.on_press  # registry dispatcher

    side_effects: list[str] = []

    def _rlt_wrapper(key):
        # RLT's special-key short-circuit: handle 'r' itself, never
        # chain through.
        if hasattr(key, "char") and getattr(key, "char", None) == "r":
            side_effects.append("rlt_success_signaled")
            ch.events["exit_early"] = True
            return
        # For everything else, fall through to the registry.
        _orig(key)

    ch.on_press = _rlt_wrapper

    # "r" triggers the wrapper's side effect; registry dispatch is
    # short-circuited so ``rlt_success`` does NOT get emitted as an event
    # (RLT signals reward via its own state, not via the events dict).
    ch.on_press(_FakeKey(char="r"))
    assert side_effects == ["rlt_success_signaled"]
    assert ch.events["exit_early"] is True
    assert ch.events["rlt_success"] is False  # wrapper handled, no chain

    # "right" falls through to the registry dispatch via _orig.
    ch.events["exit_early"] = False
    ch.on_press(_FakeKey(name="right"))
    assert ch.events["exit_early"] is True
    assert side_effects == ["rlt_success_signaled"]  # no extra RLT side effect


def test_on_press_setter_none_restores_default():
    """Setting ``on_press = None`` reverts to the registry dispatcher —
    cleanup path for code that installs a temporary wrapper."""
    ch = ControlChannel()
    ch.register("exit_early", keyboard_key="right")

    def _temp_wrapper(key):
        ch.events["exit_early"] = True  # unconditional — silly but observable

    ch.on_press = _temp_wrapper
    assert ch.on_press is _temp_wrapper

    # Reset to default. Bound-method identity isn't preserved across
    # attribute accesses (Python's descriptor protocol creates a fresh
    # bound method each time), so assert behavioural equivalence: the
    # default dispatches via the registry, and the wrapper does not
    # fire any more.
    ch.on_press = None
    assert ch.on_press is not _temp_wrapper

    ch.events["exit_early"] = False
    ch.on_press(_FakeKey(name="right"))
    assert ch.events["exit_early"] is True  # registry dispatched again


def test_emit_for_keyboard_key_picks_up_actions_registered_after_attach():
    """Pynput's on_press iterates the live registry, so an action
    registered AFTER ``attach_pynput()`` is picked up the next time the
    bound key is pressed. Critical for callers that want to ``register``
    incrementally during startup."""
    ch = ControlChannel()
    # Simulate "pynput attached but no actions yet" — at this point a
    # SPACE press would be a no-op.
    assert ch.emit_for_keyboard_key("space") == []

    # Later: orchestrator registers an action.
    ch.register("intervene", keyboard_key="space")

    # Now SPACE fires — same emit path, no re-attach needed.
    assert ch.emit_for_keyboard_key("space") == ["intervene"]
    assert ch.events["intervene"] is True


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
