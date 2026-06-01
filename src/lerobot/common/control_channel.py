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

"""Pluggable flow-control channel for orchestrator loops.

A thin in-process bus that maps **action names** (``"next_episode"``,
``"intervene"``, ...) to **triggers** (a keyboard key, an HTTP POST, a
controller button). The consumer (record loop / HVLA / HIL gym) just
registers the action names it cares about, optionally with a default
keyboard binding, and polls an ``events`` dict each tick:

    ch, events = init_control_channel()
    ch.register("next_episode",     keyboard_key="right")
    ch.register("rerecord_episode", keyboard_key="left")
    ch.register("stop_recording",   keyboard_key="esc")
    # No semantics in the channel — the consumer integrates edges
    # into whatever state machine it owns:
    if events["next_episode"]:
        events["next_episode"] = False
        ...

Why this shape: the legacy ``init_keyboard_listener`` baked the three
verbs and their key bindings into one function. Adding a new trigger
source (GUI button, Quest controller) meant either inventing a parallel
listener or hacking pynput. With a registry, **any source** plugs into
the same name space — the consumer doesn't know or care whether
``events["next_episode"]`` was set by a keyboard, a GUI POST, or a
controller. New verbs are one ``register()`` call.

Sources today:

* **pynput** keyboard listener (legacy CLI behavior). On each keypress,
  walks the action registry and emits any action whose
  ``keyboard_key`` matches. Headless-safe.
* **stdin JSON lines** ``{"v": 1, "cmd": "<name>"}``, gated on
  ``LEROBOT_CONTROL_CHANNEL_STDIN=1`` (set by the GUI's subprocess
  launcher). Unregistered names / malformed JSON / unsupported ``v``
  are dropped and logged — bumping ``v`` is how a future schema change
  avoids mis-dispatch on an older subprocess.

Adding a third source (e.g. a controller-button callback running in a
teleop's input thread) is a one-method addition: read the registry,
call ``channel.emit(name, source="...")`` when a binding fires.

Why one dict shared between sources rather than per-source queues:
the orchestrator polls ``events["next_episode"]`` and resets it after
acting — same shape ``init_keyboard_listener`` returned for years. A
queue would force every loop to migrate. Keeping the dict means
swapping the channel underneath legacy callers is a one-line edit.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import IO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Action:
    """A registered flow-control action.

    ``name`` is the public identifier sources emit and consumers poll
    (the events dict key). ``keyboard_keys`` is a tuple of optional
    default bindings the pynput source uses — empty means "no keyboard
    trigger; only stdin / controller sources can fire this." Values
    match pynput's ``Key.name`` for special keys (``"right"``,
    ``"space"``, ``"esc"``) or a single-character string for letter
    keys (``"a"``). A tuple rather than a single value because the
    legacy contract had multi-key actions — pressing ``left`` fired
    both ``rerecord_episode`` AND ``exit_early``, so ``exit_early`` is
    naturally bound to ``("right", "left", "esc")``.
    """

    name: str
    keyboard_keys: tuple[str, ...] = ()


class ControlChannel:
    """Registry of named flow-control actions, fed by any number of sources.

    Preconditions:
        * Consumer registers action names via :meth:`register` before
          polling — unregistered names raise from :meth:`emit` and are
          dropped by source readers.
        * Consumer treats ``events`` as the authoritative dict and
          resets flags after acting on them (consumer owns all state
          beyond "did this action just fire").

    Postconditions:
        * Any source's emit of a registered name sets the matching
          ``events[name] = True``.
        * Reader threads are daemon — a forgotten ``.stop()`` leaks at
          most one short-lived thread and never blocks process exit.
    """

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}
        self.events: dict[str, bool] = {}
        self._sources: list[_Source] = []
        self._stop_evt = threading.Event()
        # Legacy ``pynput.keyboard.Listener.on_press`` compat. Code in
        # HVLA RLT mode does ``_orig = listener.on_press; listener.on_press
        # = my_wrapper`` to chain its own key handlers without losing the
        # registry dispatch. Default is ``_default_keyboard_handler``
        # which forwards into ``emit_for_keyboard_key`` — so the getter
        # returns a non-None callable that can be invoked from a wrapper.
        # Setting it REPLACES the registry dispatch with the caller's
        # function (i.e. the caller's wrapper is now the only thing the
        # pynput listener calls); if the wrapper still wants registry
        # behaviour it should chain through the captured original.
        self._on_press_hook = self._default_keyboard_handler

    # ── Registry ──────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        *,
        keyboard_key: str | None = None,
        keyboard_keys: tuple[str, ...] | None = None,
    ) -> None:
        """Register an action name and optionally bind one or more default keys.

        ``keyboard_key`` is the common-case convenience: ``register(
        "next_episode", keyboard_key="right")``. ``keyboard_keys`` is
        the tuple form for multi-key bindings: ``register("exit_early",
        keyboard_keys=("right", "left", "esc"))``. Pass at most one.

        Idempotent: re-registering the same name replaces its bindings
        (logs the change so a debug session can spot accidental
        double-registration). Adding an action also adds a False entry
        in ``events`` so a consumer polling immediately sees a defined
        key.
        """
        if keyboard_key is not None and keyboard_keys is not None:
            raise ValueError(f"register({name!r}): pass keyboard_key OR keyboard_keys, not both")
        if keyboard_key is not None:
            keys: tuple[str, ...] = (keyboard_key,)
        elif keyboard_keys is not None:
            keys = tuple(keyboard_keys)
        else:
            keys = ()
        if name in self._actions:
            prev = self._actions[name].keyboard_keys
            logger.info(
                "Control channel: re-registering %r (keyboard_keys %r -> %r)",
                name,
                prev,
                keys,
            )
        self._actions[name] = Action(name=name, keyboard_keys=keys)
        self.events.setdefault(name, False)

    def is_registered(self, name: str) -> bool:
        return name in self._actions

    def registered_names(self) -> list[str]:
        return list(self._actions)

    def emit(self, name: str, *, source: str) -> None:
        """Mark ``name`` as triggered. Public for non-stdin in-process sources.

        Validated against the registry — emitting an unregistered name
        logs a warning and is a no-op rather than silently growing the
        events dict (which would mask a typo). Callers that don't know
        the registry should use :meth:`is_registered` to check first.
        """
        if name not in self._actions:
            logger.warning(
                "Control channel: emit unknown action %r from %s — register it first",
                name,
                source,
            )
            return
        self.events[name] = True
        logger.info("Control channel: %s from %s", name, source)

    def _default_keyboard_handler(self, key) -> None:
        """Default callback installed under :attr:`on_press`.

        Resolves the pynput ``Key`` to a string and dispatches via the
        registry. Captured by callers (HVLA RLT) that want to chain
        the registry dispatch after their own key handling.
        """
        key_str = _pynput_key_to_str(key)
        if key_str is None:
            return
        self.emit_for_keyboard_key(key_str)

    @property
    def on_press(self):
        """The currently installed keyboard callback (legacy
        ``pynput.keyboard.Listener.on_press`` compat).

        Returns the function the pynput source invokes on every press;
        defaults to the registry dispatcher. Callers can capture the
        current value, install a wrapper that does pre-processing,
        then chain through the captured value to keep registry dispatch.
        Used by HVLA RLT mode to add ``r`` / ``Key.left`` / ``Key.down``
        / ``e`` handling on top of the legacy three verbs.
        """
        return self._on_press_hook

    @on_press.setter
    def on_press(self, hook) -> None:
        if hook is None:
            self._on_press_hook = self._default_keyboard_handler
        else:
            self._on_press_hook = hook

    def emit_for_keyboard_key(self, key_str: str, *, source: str = "keyboard") -> list[str]:
        """Emit every action whose ``keyboard_keys`` contains ``key_str``.

        Returns the list of action names that fired. Public so the
        legacy compound contract — ``left`` fires both ``rerecord_episode``
        AND ``exit_early`` — is testable without spinning up a real
        pynput Listener: a unit test constructs a Channel, registers
        actions with overlapping keys, calls this method with a single
        key string, and asserts the expected names fired.

        The pynput source's ``on_press`` callback delegates to this
        method so its behaviour is the same code path the tests exercise.
        """
        fired: list[str] = []
        for action in self._actions.values():
            if key_str in action.keyboard_keys:
                self.emit(action.name, source=source)
                fired.append(action.name)
        return fired

    # ── Sources ───────────────────────────────────────────────────────────

    def attach_pynput(self) -> bool:
        """Start a pynput keyboard listener as one source.

        The listener walks the registry on each keypress to find an
        action whose ``keyboard_key`` matches — so actions registered
        AFTER ``attach_pynput()`` still bind without re-attaching.
        Returns True if the listener actually started, False on a
        headless environment / missing pynput.
        """
        from lerobot.common.control_utils import is_headless

        if is_headless():
            logger.warning("Control channel: headless environment — keyboard input disabled")
            return False
        try:
            from pynput import keyboard
        except ImportError:
            logger.warning("Control channel: pynput unavailable — keyboard input disabled")
            return False

        def on_press(key) -> None:
            try:
                # Invoke the currently-installed on_press hook (defaults
                # to ``_default_keyboard_handler`` which dispatches via
                # the registry). Callers may replace ``channel.on_press``
                # to chain their own logic — same legacy ``pynput.Listener
                # .on_press = X`` mutation point HVLA RLT uses.
                self._on_press_hook(key)
            except Exception:
                logger.exception("Control channel: keyboard on_press failed")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        self._sources.append(_PynputSource(listener))
        logger.info("Control channel: pynput keyboard source attached")
        return True

    def attach_stdin(self, stream: IO[str] | None = None) -> bool:
        """Start a background reader on ``stream`` (default ``sys.stdin``).

        Each line is parsed as ``{"v": 1, "cmd": "<action_name>"}``.
        Unregistered names / malformed JSON / unsupported ``v`` are
        dropped and logged (EOF stops the reader thread).
        """
        target = stream if stream is not None else sys.stdin
        if target is None:
            logger.warning("Control channel: no stdin to attach")
            return False

        thread = threading.Thread(
            target=self._stdin_loop,
            args=(target,),
            name="control_channel_stdin",
            daemon=True,
        )
        thread.start()
        self._sources.append(_StdinSource(thread))
        logger.info("Control channel: stdin JSON-lines source attached")
        return True

    def _stdin_loop(self, stream: IO[str]) -> None:
        while not self._stop_evt.is_set():
            try:
                line = stream.readline()
            except (ValueError, OSError):
                return
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Control channel: malformed JSON line: %r", line[:200])
                continue
            if not isinstance(msg, dict):
                logger.warning("Control channel: expected JSON object, got %s", type(msg).__name__)
                continue
            v = msg.get("v", 1)
            if v != 1:
                logger.warning("Control channel: unsupported message version %r (expected 1)", v)
                continue
            cmd = msg.get("cmd")
            if not isinstance(cmd, str):
                logger.warning("Control channel: missing or non-string 'cmd' in %r", msg)
                continue
            self.emit(cmd, source="stdin")

    def stop(self) -> None:
        """Tear down all sources. Idempotent."""
        if self._stop_evt.is_set():
            return
        self._stop_evt.set()
        for src in self._sources:
            try:
                src.stop()
            except Exception:
                logger.exception("Control channel: source teardown raised")


class _Source:
    """Marker base class — keeps polymorphism trivial."""

    def stop(self) -> None: ...


class _PynputSource(_Source):
    def __init__(self, listener) -> None:
        self._listener = listener

    def stop(self) -> None:
        self._listener.stop()


class _StdinSource(_Source):
    def __init__(self, thread: threading.Thread) -> None:
        self._thread = thread

    def stop(self) -> None:
        # Daemon reader exits on EOF / stop_evt — don't block on join.
        pass


def _pynput_key_to_str(key) -> str | None:
    """Normalise a pynput key into a string usable as :attr:`Action.keyboard_key`.

    Special keys (``Key.right``, ``Key.esc``, ``Key.space``) have a
    ``.name`` attribute that matches the string we expect at
    registration. Letter keys are ``KeyCode`` with ``.char`` — we use
    that. Modifier-bare ``KeyCode(None)`` returns None and is ignored.
    """
    name = getattr(key, "name", None)
    if name is not None:
        return name
    char = getattr(key, "char", None)
    return char if isinstance(char, str) else None


def init_control_channel(
    *,
    pynput: bool = True,
    stdin: bool | None = None,
) -> tuple[ControlChannel, dict[str, bool]]:
    """Build a :class:`ControlChannel` with the default sources attached.

    Returns ``(channel, events)``. The events dict is empty until the
    caller :meth:`~ControlChannel.register` actions — channel doesn't
    pre-register anything because action vocabulary is consumer-owned.

    Args:
        pynput: Attach the pynput keyboard source. Default True for
            parity with the legacy ``init_keyboard_listener``. Pass
            False for headless or GUI-only sessions where keyboard
            capture would be invasive.
        stdin: Attach the stdin JSON-lines source. ``None`` (default)
            consults ``LEROBOT_CONTROL_CHANNEL_STDIN`` — the GUI sets
            it on subprocess launch; CLI runs leave it unset. Pass
            True/False to force either way.
    """
    if stdin is None:
        stdin = os.environ.get("LEROBOT_CONTROL_CHANNEL_STDIN") == "1"

    channel = ControlChannel()
    if pynput:
        channel.attach_pynput()
    if stdin:
        channel.attach_stdin()
    return channel, channel.events
