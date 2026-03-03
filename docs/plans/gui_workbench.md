# LeRobot GUI Workbench

High-level plan for extending the dataset GUI into a full robotics workbench.

The existing dataset GUI plan lives at [dataset_gui_tool.md](dataset_gui_tool.md) and remains the source of truth for the **Data** tab. This document covers the overall multi-tab architecture and the new tabs.

---

## Tab Overview

### 1. Data (existing)

Everything in [dataset_gui_tool.md](dataset_gui_tool.md). Current status:
- **Phase 1 (read-only)** — complete: browse, play, timeline, frame cache, prefetch
- **Phase 2 (basic editing)** — complete: delete, trim, pending-edits persistence
- **Phase 3 (advanced editing)** — not started: duplicate, copy/move, reorder, merge
- **Phase 4 (polish)** — not started: undo/redo, HF Hub sync

### 2. Model

Manage local and HuggingFace robotics models.

- **Browse**: list local checkpoints and HF Hub models
- **Inspect**: view config, training curves, metadata for a selected model
- **Train**: launch training jobs
  - local: wraps `lerobot train` — the GUI presents training config fields (batch size, learning rate, etc.) instead of requiring command-line arguments
  - cloud: Nebius / other GPU provider integration (submit, monitor, pull results)
- **Download / Upload**: pull from or push to HF Hub

### 3. Run

Operate a robot in real time. The Run tab is where you **choose** which robot, teleop, model, and dataset to use (those are configured/managed in their respective tabs). Mode depends on what is selected:

| Selection         | Mode              | Description |
|-------------------|-------------------|-------------|
| Robot + Teleop    | Teleoperation     | Manual control, optional recording to a dataset |
| Robot + Policy    | Policy evaluation | Run inference, optional recording |
| Robot + Dataset   | Replay            | Replay recorded actions on hardware |

- Live camera feeds in the same multi-camera grid used by the Data tab
- Start / stop / pause controls

### 4. Robot

Robot and teleop hardware setup. Replaces `chop.py`'s `--set-robot`, `--set-teleop`, `identify-ports`, and `find-cameras` commands.

Robots and teleop devices are **separate profiles** — a leader arm, game controller, or keyboard are all teleop devices, and any teleop can be paired with any robot at run time (in the Run tab).

#### Robot Profiles
- List of saved robot profiles (from `~/.config/lerobot/robots/`)
- Editable form per profile: robot type (dropdown of registered Draccus types like `so100_follower`, `bi_so107_follower`, etc.), robot ID, arm port(s)
- Cameras belong to the robot profile (they're mounted on the robot)
- Create / delete profiles

#### Teleop Profiles
- List of saved teleop profiles (from `~/.config/lerobot/teleops/`)
- Editable form: teleop type (`so100_leader`, `bi_so107_leader`, game controller, keyboard, etc.), ID, port(s), options (e.g. gripper bounce)
- Create / delete profiles

#### Camera Detection
- "Detect cameras" button → calls `OpenCVCamera.find_cameras()` + `RealSenseCamera.find_cameras()`
- Shows all detected cameras as cards (type, id/serial, resolution)
- Opens live preview of all cameras simultaneously (MJPEG stream from backend, reusing camera grid UI from Data tab)
- Assign each camera a role name (front, left_wrist, right_wrist, top, or custom) → saves into the selected robot profile

#### Port Identification
- "Scan ports" → lists `/dev/ttyACM*` devices
- "Identify arms" → the wiggle test from `chop.py identify-ports`, but with GUI buttons ("Which arm just moved?" → click Left Leader / Right Leader / Left Follower / Right Follower) instead of typing in a terminal
- Writes identified ports into the selected robot and teleop profiles

#### Calibration
- TODO: guided calibration wizard (currently `robot.calibrate()` is interactive via stdin prompts — needs to be adapted for GUI-driven step-by-step flow)

---

## Settings & Config Storage

Currently `chop.py` stores one robot config at `~/.config/chop/robot_config.json`. The workbench needs broader config. Since `chop.py` functionality is being absorbed into the GUI, we use a single shared config location:

```
~/.config/lerobot/
  settings.json          # app-level prefs (theme, default dirs, HF token ref, etc.)
  robots/
    white.json           # one file per robot profile
    black.json
  teleops/
    blue.json            # leader arm
    gamepad.json         # game controller
    keyboard.json        # keyboard teleop
  training/
    defaults.json        # default training config
  cloud/
    nebius.json          # cloud provider credentials / project refs
```

- Robots and teleops are separate — any teleop can pair with any robot (chosen in Run tab)
- One-file-per-device lets users manage multiple profiles cleanly (current `chop.py` only supports one of each)
- App settings separated from hardware configs
- Secrets (HF token, cloud creds) stored as references to env vars or keyring, not plaintext
- Calibration data stays where LeRobot already stores it: `~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}/` — the GUI reads from there, doesn't duplicate it


---

## Architecture Changes

### Backend

The current backend is a single FastAPI app (single process) with routers for datasets, edits, and playback. Extend with new routers:

```
src/lerobot/gui/
  server.py              # FastAPI app, mount all routers
  state.py               # AppState (extend with robot/model/run state)
  frame_cache.py         # (unchanged)
  api/
    datasets.py          # (existing)
    edits.py             # (existing)
    playback.py          # (existing)
    models.py            # NEW — model listing, training launch, HF sync
    robot.py             # NEW — robot profiles, port/camera detection, calibration
    run.py               # NEW — teleop, policy eval, replay sessions
```

- `run.py` manages long-lived teleop/inference loops, exposed as WebSocket streams (similar to existing playback WebSocket)
- Training jobs (local) are background subprocesses; the API tracks PID, logs, status

### Frontend

Current frontend is vanilla JS in a single `app.js`. Before adding three new tabs:
- Split `app.js` into per-tab modules (still vanilla JS, minimal risk)
- Add a top-level tab bar for navigation; each tab lazy-loads its content
- Consider React/Vite migration later if complexity warrants it

### Shared UI Patterns

Several UI patterns are reusable across tabs (exact widget style TBD per context):
- Multi-camera grid (Data playback, Run live feeds, Robot diagnostics)
- Picking a dataset, model, or robot (appears in Run tab and potentially elsewhere)

---

## Implementation Order

1. Tab shell + settings infrastructure + config storage (`~/.config/lerobot/`)
2. Robot tab — replaces `chop.py` port/camera detection, unblocks Run tab
3. Run tab (teleop first, then policy eval and replay)
4. Model tab (browse/inspect first, then training, then cloud)
5. Continue Data tab Phase 3 & 4 in parallel
