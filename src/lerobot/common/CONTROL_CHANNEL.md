# Control Channel Roadmap

A multi-phase refactor of LeRobot's flow-control input handling. P0
ships a thin abstraction; P1–P8 migrate the existing pynput-listener-
per-feature sprawl onto it.

Related: the existing **Unified intervention contract** entry in
`src/lerobot/gui/TODO.md` (Architecture section) — that entry frames
the problem from the consumer side; this doc is the implementation
roadmap. They should be cross-linked once this branch reaches `main`.

## Motivation

Today flow-control inputs (advance episode, rerecord, stop, toggle
intervention) are wired as **independent global input listeners**,
scattered across teleops and orchestrators:

| Location                                                  | Listener            | Triggers               |
| --------------------------------------------------------- | ------------------- | ---------------------- |
| `common/control_utils.py::init_keyboard_listener`         | pynput              | right / left / esc     |
| `teleoperators/so_leader/so_leader.py::_start_keyboard_*` | pynput              | SPACE for intervention |
| `teleoperators/bi_so107_leader/...` + highrate variants   | pynput (via leader) | SPACE                  |
| `teleoperators/keyboard/teleop_keyboard.py`               | pynput              | arrows + events        |
| `teleoperators/gamepad/teleop_gamepad.py`                 | pygame              | A / B / X / Y buttons  |

Plus consumers — `scripts/lerobot_record.py`, `policies/hvla/s1_process.py`,
`rl/gym_manipulator.py` — that each separately poll
`teleop.get_teleop_events()` for the same flag.

Three problems:

1. **GUI process inherits global keystroke capture.** Each pynput
   Listener captures keys system-wide, regardless of which window has
   focus. Typing in the GUI's search box can flip intervention or
   end an episode. A GUI-launched record session today spawns at
   least two independent pynput Listeners in the same process.
2. **The toggle source is not pluggable.** A Quest controller's B
   button or a gamepad's Y button can't drive intervention because
   each teleop owns its own listener as state. Adding a new source
   means modifying every consumer.
3. **Teleop responsibility creep.** A teleop's job is producing per-
   tick actions. Holding the SPACE listener, debounce timer, and
   transition lock makes it the de-facto state machine for
   intervention as well.

## End state

A thin in-process bus:

```python
ch, events = init_control_channel()
ch.register("next_episode")
ch.register("intervene")

# Each source owns its own bindings:
PynputSource(ch).bind("right", "next_episode").bind("space", "intervene").attach()
QuestControllerSource(ch).bind(controller="right", button="B", action="intervene").attach()
StdinSource(ch).attach()  # no bindings — accepts any registered name

# Orchestrator polls (consumer-owned state):
if events["intervene"]:
    events["intervene"] = False
    is_intervention = not is_intervention
```

- **Channel knows nothing about intervention, episodes, or recording.**
  Pure name registry + per-name bool. Adding `mark_success`, `freeze`,
  `terminate_episode` is one `register()` call.
- **Sources own their own bindings.** Keyboard, Quest, gamepad, stdin
  each have a source class with its own binding map. Adding a new
  input type doesn't touch existing sources.
- **Bindings are user-overridable.** `~/.config/lerobot/hotkeys.json`
  consulted at registration; a GUI settings page surfaces the
  registry.
- **Teleops become pure action sources.** No `_intervention_active`,
  no `_start_keyboard_listener`, no `get_teleop_events`. The
  orchestrator owns all flow-control state and integrates `events`
  edges into whatever state machine it needs.

## Phases

### P0 — Foundation `[done]`

**Demo recording**: [`docs/control_channel/gui_demo.mp4`](docs/control_channel/gui_demo.mp4)
— Playwright drives the GUI through a real `lerobot-record` run
(virtual robot + scripted Cartesian-EE teleop, no hardware) and
clicks the new **Next Episode** button to advance through the
episode → reset → episode → reset → episode → stop cycle. Each click
is a `POST /api/run/control` → subprocess stdin → events dict edge,
visible in the GUI's terminal panel as the `Control channel:
exit_early from stdin` log line followed by the loop's phase
transition.

**Regression test**: `scripts/control_channel/smoke_record.py` —
same scenario driven via raw subprocess stdin (no GUI). Reusable
baseline for future phases; each new verb adds a phase to the
driver.

- `common/control_channel.py`: `ControlChannel` registry + `Action`
  dataclass + pynput / stdin sources. Bindings currently live on
  `Action.keyboard_keys` — P1 moves them into the source class.
- `common/control_utils.py::init_keyboard_listener` delegates to the
  new channel and pre-registers the three legacy verbs with their
  legacy keys (preserves the existing contract for
  `scripts/lerobot_record.py` and `policies/hvla/s1_process.py` — no
  caller changes).
- `gui/api/run.py`: subprocess launched with `stdin=PIPE` + the
  `LEROBOT_CONTROL_CHANNEL_STDIN=1` env var; new
  `POST /api/run/control {cmd: name}` endpoint writes JSON lines.
- GUI: Run-tab Next Episode + Rerecord buttons next to Stop.

Verified end-to-end via `tests/common/test_control_channel.py` (17
cases) and a smoke test that drives a fake orchestrator's stdin and
observes the events dict mutate (including a `register("intervene",
...)` round-trip to prove the registry extends without channel-side
code changes).

### P1 — Sources-own-bindings refactor

Move `keyboard_keys` off `Action` into `PynputSource`. Each source
class gains a `.bind(input, action_name)` method. Sources track their
own binding map; the channel just routes emits.

```python
# Before (P0):
ch.register("exit_early", keyboard_keys=("right", "left", "esc"))

# After (P1):
ch.register("exit_early")
pynput_source.bind("right", "exit_early")
pynput_source.bind("left", "exit_early")
pynput_source.bind("esc", "exit_early")
```

One file, all source classes update together. Mechanical migration
of the existing `keyboard_keys`. `Action` becomes name + optional
description metadata for the settings UI.

### P2 — User-overridable bindings + GUI

**Lands together with P1 / P6 / P7** so all sources arrive with their
own UI, registry, and config plumbing in one cohesive change. The
visual UI is the headline; the JSON config + per-source code are what
the UI displays.

#### Config: `~/.config/lerobot/hotkeys.json`

Flat list of `(action, source, binding)` rows. One file, all sources,
trivially diffable. Versioned for forward migrations.

```json
{
  "version": 1,
  "bindings": [
    { "action": "exit_early", "source": "keyboard", "binding": "right" },
    { "action": "exit_early", "source": "keyboard", "binding": "left" },
    { "action": "exit_early", "source": "keyboard", "binding": "esc" },
    { "action": "rerecord_episode", "source": "keyboard", "binding": "left" },
    { "action": "intervene", "source": "keyboard", "binding": "space" },
    { "action": "intervene", "source": "quest", "binding": "right_grip" },
    { "action": "rlt_success", "source": "quest", "binding": "right_B" },
    { "action": "rlt_abort", "source": "quest", "binding": "left_Y" },
    { "action": "mark_success", "source": "gamepad", "binding": "Y" }
  ]
}
```

Each source class loads the rows that match its `source` field at
`attach()` time, overriding any code-side default bindings. Bindings
are **static** — they live in the JSON whether or not the source is
currently connected. Connection state is a runtime concern shown
separately in the UI.

#### GUI: one page, three per-source views

New tab (or Robot-tab sub-section — design call):

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Hotkeys                                                                  │
│                                                                          │
│ Connected:  [⌨ Keyboard ●]  [🎮 Gamepad ●]  [🥽 Quest ○ no session]      │
│                                                                          │
│ ┌─ Keyboard ─────────────────────────────────────────────────────────┐  │
│ │  Action            Binding                                          │  │
│ │  exit_early        [right]  [left]  [esc]      [+ Add key]          │  │
│ │  rerecord_episode  [left]                      [+ Add key]          │  │
│ │  intervene         [space]                     [+ Add key]          │  │
│ │  rlt_success       [r]                         [+ Add key]          │  │
│ │  ...                                                                │  │
│ └─────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│ ┌─ Quest controllers ──────────────────────────────────────────────────┐ │
│ │                                                                       │ │
│ │  [SVG: left Touch Plus]            [SVG: right Touch Plus]            │ │
│ │   X — reset (reserved)              A — reset (reserved)              │ │
│ │   Y — rlt_abort                     B — rlt_success                   │ │
│ │   grip — clutch (reserved)          grip — intervene                  │ │
│ │   trigger — gripper (reserved)      trigger — gripper (reserved)      │ │
│ │   thumbstick — rlt_toggle_engage    thumbstick — rlt_ignore           │ │
│ │   menu — (unbound)                                                    │ │
│ │                                                                       │ │
│ │  (Click any non-reserved button to remap)                             │ │
│ └─────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌─ Gamepad ────────────────────────────────────────────────────────────┐ │
│ │  [SVG: Xbox-style gamepad with button labels]                         │ │
│ │  (Click any button to remap)                                          │ │
│ └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

Per-section interaction model:

- **Keyboard**: action-by-action list; each row has chips for the
  currently-bound keys plus an "+ Add key" affordance that captures
  the next keypress (or accepts a typed key name). Reflects the fact
  that keyboard keys aren't spatial — a list is more efficient than
  a keyboard image.
- **Quest controllers**: side-by-side SVGs of the left + right Touch
  Plus controllers. Each button shows its current binding as an
  overlay label. Reserved buttons (grip, trigger, A/X — set by the
  Quest VR teleop itself for clutch/gripper/reset) are greyed and
  non-editable; the rest open a "Bind to action" picker on click.
- **Gamepad**: single SVG of a generic Xbox-style layout. pygame
  normalizes button indices across DualSense / Xbox / generic
  gamepads, so one diagram covers the class. Same click-to-remap as
  Quest.

Conflict detection: client-side warning if a binding is reused across
actions within the same source (e.g. binding `B` on the right Quest
controller to both `rlt_success` and `intervene` — last write wins,
warn the user). Cross-source conflicts are fine (`space` on keyboard
and `right_grip` on Quest both mapping to `intervene` is the _point_).

#### Connection-status badges

Top of the page. Dynamic, derived from a `GET /api/hotkeys/status`
endpoint that the server populates per-source:

- **Keyboard** — always connected (pynput is always loadable; headless
  environments grey it out).
- **Gamepad** — pygame's `joystick.get_count() > 0` polled every few
  seconds.
- **Quest** — `WebXR session active` if the Quest VR teleop's WebSocket
  has at least one connected client.

When a source disconnects, its section in the UI doesn't disappear —
it gets a subtle "no device" overlay so the user can still configure
bindings for next time. (Bindings survive disconnects; the JSON is
the source of truth, not the live device.)

#### Action registry shape (what the GUI fetches)

`GET /api/hotkeys/actions` returns the channel's
`registered_names()`, paired with a one-line description from
`Action.description` (P1 adds this field). Each known source publishes
its own bindable-button list:

```json
{
  "actions": [
    { "name": "exit_early", "description": "End the current episode / reset phase and advance." },
    { "name": "intervene", "description": "Toggle intervention mode (operator takes over)." },
    { "name": "rlt_success", "description": "End the episode with positive terminal reward (RLT only)." },
    ...
  ],
  "sources": {
    "keyboard": { "kind": "list" },
    "gamepad": { "kind": "diagram", "buttons": ["A", "B", "X", "Y", "LB", "RB", ...] },
    "quest":   { "kind": "diagram", "buttons": ["left_A", "left_B", "left_X", "left_Y", "left_grip", ...] }
  }
}
```

The GUI is purely a view of this registry — adding a new action in
code shows up as a new row in the table without any frontend edit.

#### Build cost

| Component                                                                                                                              | LoC     | Notes                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------- |
| **P1**: `Action` loses `keyboard_keys`; each source class owns its own bindings map                                                    | ~80     | Mechanical; touches the channel + tests + `init_keyboard_listener`. |
| **P2 backend**: `hotkeys.json` load/save + `/api/hotkeys/{actions,status,bindings}` endpoints + source classes consult on attach       | ~150    | New module `lerobot.common.hotkeys`; GUI API thin wrapper.          |
| **P2 frontend**: page markup, SVG embeds, click-to-remap dialog, status polling                                                        | ~250    | Mostly HTML/JS; SVG assets one-time effort.                         |
| **SVG assets**: Xbox gamepad + Quest Touch Plus left + Quest Touch Plus right                                                          | one-off | Public-domain diagrams exist; or trace from photos. ~3 files.       |
| **P6**: `QuestControllerSource` class + Quest VR teleop integration + `set_control_channel` on Teleoperator base                       | ~100    | New source class; Quest VR `_on_frame` adds button-edge emit.       |
| **P7**: `GamepadSource` class + gamepad teleop refactor (drops `get_teleop_events`)                                                    | ~100    | Same shape as P6.                                                   |
| **Tests**: hotkeys.json round-trip; per-source attach honours JSON; conflict warning; channel emit chain through Quest/Gamepad sources | ~120    | Unit-level — no GUI E2E needed at first cut.                        |

**Total**: ~800 LoC + 3 SVG assets. One cohesive PR, big enough to
warrant a per-section commit history but small enough to land
together.

#### Design decisions to settle before code

- **Where does the page live**: top-level "Hotkeys" tab vs sub-section
  of Robot tab vs new "Settings" tab. Hotkeys are a global concern,
  not per-robot-profile, so a top-level "Settings" tab probably wins
  long-term (also a natural home for the persistence-policy item in
  `gui/TODO.md`).
- **Default-bindings policy**: ship sensible defaults in code; `hotkeys.json`
  stores overrides only (smaller diff, "reset to defaults" works
  trivially) vs always-explicit (every binding written; defaults exist
  as a Python constant the GUI reads to produce a fresh JSON on first
  open). Defaults-as-overrides is probably right.
- **Reserved-button policy**: should the Quest VR teleop's
  hardware-reserved buttons (grip / trigger / A-X) be configurable, or
  hard-coded to their semantic role? Hard-coded keeps the UX simple but
  removes ergonomic flexibility (e.g. a user wanting the trigger to do
  intervention instead of gripper). For first cut: hard-coded with a
  follow-up to make them configurable.
- **Conflict resolution**: client-side warning only, or server-side
  reject on save? Warning is friendlier; reject is safer. Probably
  warning + visual indicator on the conflicting cell.
- **Versioning**: `hotkeys.json` schema version field — on bump, the
  loader either migrates or raises. Need a migration table next to
  the loader.

### P3 — Intervention migration in record loop

Add `ch.register("intervene", keyboard_key="space")` at record-loop
startup. Read `events["intervene"]` and **OR** with the existing
`teleop.get_teleop_events()` call — both sources work in parallel
during transition. Adds an Intervene button to the GUI Run tab;
server-side allowlist gains the verb. No deletion yet — leader's
pynput listener stays.

### P4 — HVLA RLT migration `[done]`

Done in this PR. Shows the shape of a real consumer adopting the
channel beyond the legacy three verbs.

**Before** — [`s1_process.py`](../policies/hvla/s1_process.py) wrapped
`listener.on_press` to grab raw pynput `Key` objects and dispatch
per-key inline:

```python
_orig_on_press = listener.on_press

def _rlt_on_press(key, *args):
    if hasattr(key, "char") and key.char == "r":
        rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
        events["exit_early"] = True
        # ... 18 lines of inline sound playback ...
        return
    if key == _kb.Key.left:
        # ... 30 more lines for abort ...
    if hasattr(key, "char") and key.char == "e":
        # ... toggle engage ...
    if key == _kb.Key.down:
        # ... 30 more lines for ignore ...
    if _orig_on_press is not None:
        _orig_on_press(key)

listener.on_press = _rlt_on_press
```

**After** — register one action per intent, drain on the main thread:

```python
# _register_rlt_actions(listener):
listener.register("rlt_success", keyboard_key="r")
listener.register("rlt_abort", keyboard_key="left")
listener.register("rlt_ignore", keyboard_key="down")
listener.register("rlt_toggle_engage", keyboard_key="e")
# Multi-key: r and down also fire exit_early (so the loop breaks).
listener.register("exit_early", keyboard_keys=("right", "left", "esc", "r", "down"))
# Critically: REBIND rerecord_episode to drop "left" (RLT wants the
# aborted trajectory STORED as a negative-reward terminal) and add
# "down" (IGNORE wants the dataset rolled back).
listener.register("rerecord_episode", keyboard_key="down")

# Main-loop tick: _drain_rlt_events(events, rlt_state, infer_thread)
if events.get("rlt_success"):
    events["rlt_success"] = False
    rlt_state["lifecycle"].signal_terminal(TerminalKind.SUCCESS)
    _play_rlt_tone([(800, 0.3, 1.0), (1200, 0.15, 1.0)])
    # exit_early was set in the same emit_for_keyboard_key call
if events.get("rlt_abort"): ...      # symmetric
if events.get("rlt_ignore"): ...     # also rolls back via rerecord_episode multi-key
if events.get("rlt_toggle_engage"): ...
```

Wins from the migration:

- **No pynput-thread side effects.** Audio playback, log lines, and
  overlay prints all run on the main loop now, sequenced with the
  loop's own logging instead of racing with it.
- **Sound playback factored.** Three near-identical 18-line blocks
  collapse to one `_play_rlt_tone(segments)` helper. Distinct cues
  (ascending success / descending abort / two-beep ignore) are one-
  liner segment lists.
- **No legacy compat shim.** The `on_press` getter/setter that was
  added to keep this pattern alive is now removed — RLT no longer
  depends on monkey-patching pynput.
- **Same operator UX.** The 4 keys (r / left / down / e) trigger the
  same lifecycle signals + cues as before. The legacy compound
  ("left = abort, save trajectory" vs "down = ignore, discard
  trajectory") is preserved via channel rebinding, not via
  short-circuit returns in a handler.

`test_rlt_actions_register_with_expected_bindings` in
[`tests/common/test_control_channel.py`](../../../tests/common/test_control_channel.py)
pins the bindings + the dispatch behaviour for each of the four keys.

### P4 (cont.) — HIL gym migration

`rl/gym_manipulator.py` reads `IS_INTERVENTION` from a teleop's
`get_teleop_events()`. Same migration shape: register an `intervene`
action, poll `events["intervene"]`, OR with the legacy call during
transition. Independent commit.

### P5 — Delete leader's pynput SPACE listener

Once P3 + P4 prove channel-driven intervention works in production:

- Delete `_start_keyboard_listener`, `_intervention_active`,
  `_intervention_transition_lock`, `_intervention_debounce_s`,
  `_try_toggle_intervention` from `so_leader.py`.
- Delete `get_teleop_events` from `so_leader`, `bi_so107_leader`,
  bi_so107_leader_highrate, etc.
- Move transition lock + debounce into the orchestrator (whichever
  owns the swap state machine — probably a small
  `InterventionState` class composed in).

Behavior-changing — the SPACE handler moves from the leader's
listener to the channel's pynput source. User-visible: identical
(SPACE still toggles intervention), but the keystroke now goes
through the same path as GUI buttons. Risk: any downstream consumer
of `get_teleop_events` outside this repo breaks.

### P6 — QuestControllerSource

New source class. Quest VR teleop instantiates and binds B / Y face
buttons to actions:

```python
quest_source = QuestControllerSource(ch)
quest_source.bind(controller="right", button="B", action="intervene")
quest_source.bind(controller="left",  button="Y", action="mark_success")
# On WebXR frame, button edges call quest_source.on_button_edge(...)
```

Quest VR teleop's existing button parsing in `_on_frame` calls into
the source. No new wire format — same in-process emit path the
pynput source uses.

### P7 — GamepadSource

Same pattern. Replaces the gamepad teleop's `get_teleop_events` +
pygame button polling for flow-control. The gamepad's joint-control
loop stays; only the flow-control buttons move.

### P8 — Delete `get_teleop_events` from the Teleoperator protocol

Last consumer removed. Type annotations updated. The teleop's job is
producing actions, full stop.

## Open questions

- **Settings UX shape.** Where in the GUI does the hotkey settings
  page live (Run tab, Robot tab, new Settings tab)? JSON schema
  versioning. Hotkey conflict rules (refuse / warn / first-wins).
- **Quest button binding granularity.** Per-controller (left vs right
  hand) or unified ("any B button")? Affects schema.
- **Backwards compat during P5 / P8.** Downstream forks may have
  custom teleops implementing `get_teleop_events`. Deprecate-then-
  delete with a release-cycle gap, or break clean and document.
- **State machine ownership for intervention.** The orchestrator takes
  over `_intervention_active`. The transition lock probably belongs
  in a small `InterventionState` class composed into each
  orchestrator that wants intervention — not in the channel.

## Status

| Phase                            | State   | Branch / commit        |
| -------------------------------- | ------- | ---------------------- |
| P0 — Foundation                  | done    | `feat/control-channel` |
| P1 — Source-owned bindings       | pending | —                      |
| P2 — User-overridable bindings   | pending | —                      |
| P3 — Intervention in record loop | pending | —                      |
| P4 — HVLA RLT migration          | done    | `feat/control-channel` |
| P4 (cont.) — HIL gym migration   | pending | —                      |
| P5 — Delete leader's listener    | pending | —                      |
| P6 — QuestControllerSource       | pending | —                      |
| P7 — GamepadSource               | pending | —                      |
| P8 — Delete `get_teleop_events`  | pending | —                      |
