# Quest 3 → SO-107 teleop (prototype)

WebXR-based Cartesian teleop. Quest 3 controller pose streams to PC over
WebSocket; PC runs IK and drives the arm (sim only in this directory; physical
hookup is a separate step).

## Files

- `webxr_teleop.html` — page served to the Quest. Opens an `immersive-vr` session,
  streams both controllers' 6DOF pose + buttons at the headset's frame rate.
- `latency_receiver.py` — pure latency probe (no IK, no robot). Measures RTT.
- `sim_receiver.py` — full teleop pipeline with simulated arm rendered in rerun.
  Clutch-mode (right index trigger), position + orientation, workspace + joint
  limits, frame-stall + tracking-dropout watchdogs, RTT instrumentation.

## Setup (one-time)

1. Install deps the usual way (`uv sync` etc.). The receivers depend on
   `aiohttp`, `scipy`, `pinocchio`, `rerun` — all already in the project.
2. Make sure `openssl` is on PATH (`which openssl`). Pre-installed on Ubuntu
   and macOS. On Windows, ships with Git for Windows.
3. **Open port 8443 on the PC's firewall.** On Ubuntu:
   `sudo ufw allow 8443/tcp`.
4. Train the IK model (or copy a pre-trained one to `/tmp/so107_ik_model_action.pt`):
   ```bash
   .venv/bin/python -m lerobot.robots.so107_description.learned_ik.dataset_extractor
   .venv/bin/python -m lerobot.robots.so107_description.learned_ik.train \
       --data /tmp/so107_ik_train.npz --out /tmp/so107_ik_model_action.pt --hidden 256,256,256
   ```
5. Quest 3 setup:
   - Connect Quest to the **same WiFi** as the PC. 5 GHz / WiFi 6 recommended.
   - **Disable the proximity sensor** if you want to wear the headset on your
     chest/neck during teleop (standard pattern for collocated robot teleop —
     otherwise the immersive session pauses the moment the headset comes off
     your face and the arm stops responding). Options ranked by friction:
     - **Tape** over the sensor on the inside of the headset, top-center between
       the lenses. Universal, firmware-update-proof, what most teleop people do.
     - **SideQuest app** (Linux/macOS/Windows): Advanced Settings → toggle
       Proximity Sensor off. May auto-re-enable after disconnect timeout.

## Running

First run generates `cert.pem` + `key.pem` (self-signed, gitignored, used to
satisfy WebXR's secure-context requirement). They are regenerated automatically
if deleted; never commit them.

### Latency probe (network sanity check, no robot, no sim)

```bash
.venv/bin/python -m lerobot.robots.so107_description.teleop_quest.latency_receiver
```

- Prints URL like `https://192.168.x.x:8443/`.
- Open that URL in the Quest's built-in browser, tap **Advanced → Proceed**
  on the self-signed-cert warning (one-time per device).
- Tap **Connect WebSocket**, then **Enter VR**.
- Receiver prints RTT stats once per second. Typical good WiFi: mean 5–15 ms,
  p99 < 50 ms.

### Sim teleop (rendered arm in rerun, no physical robot)

```bash
.venv/bin/python -m lerobot.robots.so107_description.teleop_quest.sim_receiver
```

- Same Quest flow (URL → Connect → Enter VR).
- A rerun viewer opens automatically on the PC.
- **Squeeze the right index trigger to engage** (clutch). Release to freeze.
- Move/rotate your right hand → simulated arm follows.

The HUD on the Quest page and the stdout on the PC both show live state.
Per-session logs are written to `/tmp/quest_sim_HHMMSS.log` (sim) or
`/tmp/quest_latency_HHMMSS.csv` (latency probe).

## Coordinate mapping

```
Quest local stage frame:  +x = right (user), +y = up, +z = toward user
Robot base frame:         +x = forward, +y = left, +z = up
  robot_x = -quest_z      push controller away → robot reaches forward
  robot_y = -quest_x      controller right     → robot's own right (-y in URDF)
  robot_z =  quest_y      controller up        → robot up
```

Same 3×3 matrix is applied to both positions (deltas) and rotations
(similarity transform). See `QUEST_TO_ROBOT_M` in `sim_receiver.py`.

## Safety controls in sim_receiver

- **Workspace clamp**: target xyz clipped to the training-data workspace box.
- **IK reachable check**: if `ik_err > 10 mm`, motor state is not updated
  (avoids garbage joint commands accumulating).
- **Joint limits**: clamped to physical motor ranges from calibration.
- **Per-tick caps**: 2 cm position step, 5° rotation step, 3°/tick joint step.
- **Gripper excluded from IK** (not EE-tracked, would drift on bias). Sits at
  home until we wire it to a trigger/grip button.

## Known issues / future work

- Gripper not yet wired to controller buttons.
- Single-arm only — bimanual needs duplicating the Sim for the left controller.
- No physical-robot path here yet; the natural next step is to swap rerun-render
  for `robot.send_action` after we add safety supervision (limits, stop button,
  ramp-on engage).
- Self-signed cert means a one-time browser warning to click through.
