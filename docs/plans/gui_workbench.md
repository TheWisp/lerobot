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

Operate a robot in real time. The Run tab is where you **choose** which robot, model, and dataset to use (those are configured/managed in their respective tabs). Mode depends on what is selected:

| Selection         | Mode              | Description |
|-------------------|-------------------|-------------|
| Robot only        | Teleoperation     | Manual control, optional recording to a dataset |
| Robot + Policy    | Policy evaluation | Run inference, optional recording |
| Robot + Dataset   | Replay            | Replay recorded actions on hardware |

- Live camera feeds in the same multi-camera grid used by the Data tab
- Start / stop / pause controls

### 4. Robot

Robot hardware setup and calibration.

- **Profiles**: create / edit / delete robot configs (currently `robot_config.json`)
  - Robot type, arm ports, camera definitions
  - One robot at a time is sufficient
  - A dual-arm robot (with leader for teleop) is still one robot profile
- **Camera wizard**: detect available cameras (OpenCV indices, RealSense serials), preview, assign names
- **Port scanner**: detect serial ports, identify arms
- **Calibration**: guided calibration flow per arm, store/load calibration data
- **Live diagnostics**: real-time joint positions, camera feeds, latency indicators

---

## Settings & Config Storage

Currently `chop.py` stores one robot config at `~/.config/chop/robot_config.json`. The workbench needs broader config. Since `chop.py` functionality is being absorbed into the GUI, we use a single shared config location:

```
~/.config/lerobot/
  settings.json          # app-level prefs (theme, default dirs, HF token ref, etc.)
  robots/
    white.json           # one file per robot profile (migrated from chop's robot_config.json)
    black.json
  training/
    defaults.json        # default training config
  cloud/
    nebius.json          # cloud provider credentials / project refs
```

- One-file-per-robot lets users manage multiple robot profiles cleanly (current `chop.py` only supports one)
- App settings separated from robot hardware configs
- Secrets (HF token, cloud creds) stored as references to env vars or keyring, not plaintext
- Migration path: on first launch, detect `~/.config/chop/robot_config.json` and offer to import

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
