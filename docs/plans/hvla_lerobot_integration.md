# HVLA ↔ LeRobot Integration Plan

Status: Implemented (2026-03-22)

## Context

HVLA (Hierarchical VLA) is a dual-system policy: S2 (PaliGemma VLM) provides high-level scene understanding, S1 (flow matching action expert) provides low-level motor commands. Currently HVLA is hardcoded for the SO107 bimanual robot (14-DOF, 4 cameras, Feetech motors). It also runs outside LeRobot's standard policy/inference pipeline — it has its own launch script, control loop, and IPC.

This plan covers two tracks:
1. **Robot-agnostic**: remove SO107-specific hardcoding so HVLA works with any LeRobot robot
2. **GUI integration**: make HVLA discoverable and runnable from the LeRobot GUI

---

## Track 1: Robot-Agnostic HVLA

### 1.1 Current Hardcoding Inventory

| Category | Hardcoded Value | Files |
|----------|----------------|-------|
| Joint names | 14 SO107 names | s1_process.py:20-25 |
| Action/state dim | 14 | config.py:37,87 / s1_process.py:191 / ipc.py:254 |
| Gripper indices | 6 (left), 13 (right) | s1_process.py:103-104 (drop detection only — generalize to any-joint) |
| Arm split | index 7 boundary | s1_process.py:50-51 (drop detection only — generalize) |
| Camera names | front/top/left_wrist/right_wrist | s1_process.py:28-33 / config.py:53-58 |
| S2 camera key map | base_0_rgb, left_wrist_0_rgb, etc. | s1_process.py / launch.py / s2_standalone.py |
| Raw image resolution | 1280×720 | ipc.py:215 |
| State buffer | 32 slots, truncate to 14 | ipc.py:230,254 |
| Motor attributes | left_arm, right_arm | s1_process.py:672,708 |
| Torque limit | 1000 (Feetech) | s1_process.py:696,725 |

### 1.2 Use LeRobot's Robot as Source of Truth

LeRobot already has `Robot` (base class) and `RobotConfig` that provide everything HVLA needs. No new config class required.

From a connected `Robot` instance:
- **`robot.action_features`** → `{"left_shoulder_pan.pos": float, ...}` → joint names, action dim
- **`robot.observation_features`** → joint names + camera keys with `(H, W, 3)` shapes
- **`robot.cameras`** → camera configs with height, width, fps
- **`robot.left_arm`, `robot.right_arm`** (bimanual) or `robot.arm` (single) → motor buses
- **`robot.send_action(action_dict)`** → send named joint actions
- **`robot.get_observation()`** → read named joint positions + camera frames

What HVLA derives at init from the robot:
```python
# From robot.action_features:
joint_names = list(robot.action_features.keys())     # ["left_shoulder_pan.pos", ...]
action_dim = len(joint_names)                          # 14 for bimanual SO107
state_dim = action_dim

# From robot.observation_features (camera entries have tuple values):
camera_keys = [k for k, v in robot.observation_features.items() if isinstance(v, tuple)]
# → ["front", "top", "left_wrist", "right_wrist"]

# From observation_features — camera entries have (H, W, 3) tuple values:
camera_resolutions = {
    k: (v[0], v[1]) for k, v in robot.observation_features.items() if isinstance(v, tuple)
}
# → {"front": (720, 1280), "left_wrist": (480, 640), ...}
# SharedImageBuffer allocates per-camera buffers at each camera's resolution

# No gripper-specific assumptions — drop/oscillation detection should
# work on any joint (detect large sudden jumps generically)
```

The only thing not in the Robot API: **S2 camera key mapping** (e.g. `"front" → "base_0_rgb"`). This is S2-model-specific, not robot-specific. It stays as a CLI arg or S2 config.

### 1.3 Propagation Plan

| File | What changes |
|------|-------------|
| **s1/flow_matching/config.py** | `action_dim`, `state_dim`, `image_features` populated from robot at init, not hardcoded |
| **s1_process.py** | Remove `JOINT_NAMES`, `S2_CAM_KEY_MAP` constants. Accept `Robot` instance. Read joint names from `robot.action_features`. Grip drop uses `gripper_indices` derived from names. Use `robot.send_action()` / `robot.get_observation()`. |
| **ipc.py** | `SharedImageBuffer` height/width from robot's camera config. State buffer size = `action_dim`. |
| **s1_inference.py** | No changes (already generic) |
| **launch.py** | Accept LeRobot robot profile (same `--robot.*` args as `lerobot-record`), instantiate `Robot`, pass through |
| **s2_standalone.py** | Camera keys already CLI arg — no change needed |
| **s1/flow_matching/train.py** | `action_dim`, `chunk_size` from config (already parameterized, just need to not default to 14) |

### 1.4 Motor Control

Currently s1_process.py writes raw motor registers:
```python
bus.write("Torque_Limit", motor_name, 1000, normalize=False)
```

The motor bus already has `enable_torque()` / `disable_torque()` methods (used by the leader-inverse-follow branch). HVLA should use those instead of raw writes:

```python
# Current (hardcoded Feetech register + value):
bus.write("Torque_Limit", motor_name, 1000, normalize=False)

# Fixed (generic motor bus API):
robot.left_arm.bus.disable_torque()   # reset phase: arms go limp
robot.left_arm.bus.enable_torque()    # resume: arms active
```

This matches the pattern in `feature/leader-inverse-follow` where `teleop.disable_torque()` / `teleop.enable_torque()` delegate to `self.bus.disable_torque()` / `self.bus.enable_torque()`. Uses `hasattr` guard for robots that don't support it.

### 1.5 Analysis Scripts

Leave hardcoded — they're diagnostic tools for this specific robot/task, not reusable policy code. Add a note that they assume SO107.

---

## Track 2: GUI Integration

### 2.1 Model Tab — HVLA Discovery

**Current state**: The Models tab scans `outputs/` for directories with `checkpoints/*/pretrained_model/config.json`. HVLA checkpoints don't match because HVLA's `train.py` is a standalone script that doesn't use LeRobot's `Trainer` — it saves weights directly without the `pretrained_model/` wrapper or `config.json`.

Standard LeRobot: `checkpoint/pretrained_model/{config.json, model.safetensors, train_config.json, ...}`
HVLA S1: `checkpoint/{model.safetensors, norm_stats.pt, optimizer.pt}` (no config.json, no pretrained_model/)

**Fix**: Make HVLA's train.py save in the standard LeRobot checkpoint format (or at minimum emit a `config.json` with `type: "hvla_flow_s1"`). This is better than teaching the GUI to scan a nonstandard format.

- [ ] HVLA train.py: save checkpoints in `pretrained_model/` subdirectory with `config.json`
- [ ] Include S2 checkpoint path used for latent extraction in the config/metadata
- [ ] S2 checkpoints (converted from JAX) should also have a `config.json` identifying them

### 2.2 Run Tab — HVLA Inference

**Current state**: The Run tab launches `lerobot-record` with `--policy.path=...`, which uses LeRobot's standard `Policy.from_pretrained()` → `policy.select_action()` loop. HVLA doesn't fit this — it needs two processes (S1 + S2) with shared memory IPC, its own control loop, and two checkpoint paths.

**Approach: Policy-type-driven dispatch.** When the user selects an HVLA S1 checkpoint (identified by `config.json` with `type: "hvla_flow_s1"`), the Run tab switches to HVLA-specific UI and launches `python -m lerobot.policies.hvla.launch` instead of `lerobot-record`. This keeps the standard policy flow unchanged while supporting HVLA as a special case.

The robot profile dropdown works the same — the selected robot profile is converted to `--robot.*` CLI args for the HVLA launch command, same as for `lerobot-record`.

### 2.3 Run Tab — HVLA-Specific UI

When an HVLA checkpoint is selected, the policy form shows additional fields:
- [ ] S2 checkpoint path — text field (filepath, not from Models tab — S2 is a special dependency)
- [ ] Task prompt — text field
- [ ] Decode subtask toggle
- [ ] Record evaluation toggle (reuse existing dataset selector)
- [ ] Number of episodes, episode time, reset time (reuse existing fields)

The GUI converts these settings into `python -m lerobot.policies.hvla.launch --s1-checkpoint ... --s2-checkpoint ... --robot.type=... --task "..." [etc]`.

### 2.5 Dataset Recording

**Current state**: HVLA has its own `_create_recording_dataset()` / `_add_frame_to_dataset()` in s1_process.py that writes to LeRobotDataset format.

**Changes needed**:
- [ ] Verify recorded datasets appear correctly in the Data tab
- [ ] Verify camera keys in recorded dataset match what the GUI expects for playback
- [ ] Verify episode boundaries are correctly marked

---

## Track 3: Training Integration (Future)

Not in scope now, but for completeness:

- [ ] S1 training via GUI (launch `python -m lerobot.policies.hvla.s1.flow_matching.train` with args)
- [ ] S2 latent extraction via GUI (launch `python scripts/extract_s2_latents_hvla.py`)
- [ ] Training progress in Models tab (loss curve from wandb or log parsing)

---

## Implementation Order

1. [x] **Track 2.1**: HVLA train.py saves standard checkpoint format — `pretrained_model/config.json` + `training_state/`
2. [x] **Track 1.2-1.3**: Derive from LeRobot Robot — joint names, cameras, dims from `robot.action_features` / `robot.observation_features`
3. [x] **Track 2.2**: Policy-type-driven dispatch — HVLA checkpoint (`hvla_flow_s1`) triggers `/api/run/hvla` endpoint
4. [x] **Track 2.3**: HVLA-specific form fields — S2 path, task prompt, decode subtask toggle, episode controls
5. [x] **Track 1.4**: Motor control via `bus.enable_torque()` / `bus.disable_torque()` — no more raw register writes
