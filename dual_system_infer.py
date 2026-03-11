#!/usr/bin/env python
"""Dual-System VLA Inference: S2 (Pi0.5) + S1 (ACTWithVLM)

S2 coroutine: Async WebSocket query to Pi0.5 at ~1Hz for scene-understanding latent.
S1 loop: Synchronous ACTWithVLM at ~30Hz for reactive action chunking.

The S2 latent is shared via a thread-safe cache. If S2 disconnects or is slow,
S1 continues with the last cached latent (graceful degradation).

Controls:
  Escape  ->  abort (soft landing)

Usage:
  # Terminal 1: Start Pi0.5 server
  cd ~/Documents/openpi_subtask && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python -u \
    scripts/async_pi05/async_pi05_websocket_server.py \
    --config soarm_pi05_flow_lora \
    --checkpoint ~/.cache/openpi/checkpoints/soarm-pi05-state-11997 \
    --gpu-id 0 --port 8765

  # Terminal 2: Run dual-system inference
  python dual_system_infer.py \
    --s1-checkpoint outputs/act_vlm_cylinder_ring \
    --task "assemble cylinder into ring" \
    --s2-server ws://localhost:8765
"""

import argparse
import asyncio
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch

# Register draccus ChoiceRegistry subclasses before any decode calls
from lerobot.robots import bi_so107_follower  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say

try:
    import websockets
except ImportError:
    raise ImportError("websockets is required: pip install websockets")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("dual_system")


# --------------- Constants ---------------

FPS = 30

JOINT_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_forearm_roll.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_forearm_roll.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]

# S2 uses all 4 cameras (full scene understanding)
S2_IMAGE_KEYS = ["front", "top", "left_wrist", "right_wrist"]

CONFIG_DIR = Path.home() / ".config" / "chop"
ROBOT_CONFIG_FILE = CONFIG_DIR / "robot_config.json"


# --------------- S2 Latent Cache ---------------

@dataclass
class S2LatentCache:
    """Thread-safe cache for the S2 (VLM) latent vector."""
    latent: np.ndarray | None = None
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    update_count: int = 0

    def update(self, latent: np.ndarray):
        with self.lock:
            self.latent = latent
            self.timestamp = time.time()
            self.update_count += 1

    def get(self) -> tuple[np.ndarray | None, float]:
        with self.lock:
            return self.latent, self.timestamp

    @property
    def age_ms(self) -> float:
        with self.lock:
            if self.timestamp == 0:
                return float("inf")
            return (time.time() - self.timestamp) * 1000


# --------------- Helpers ---------------

def soft_land(robot, duration_s=4.0, steps=20):
    """Gradually reduce torque so the robot lowers gently instead of dropping."""
    if not robot.is_connected:
        return

    buses = []
    if hasattr(robot, "left_arm"):
        buses.append(robot.left_arm.bus)
    if hasattr(robot, "right_arm"):
        buses.append(robot.right_arm.bus)
    if not buses:
        return

    try:
        obs = robot.get_observation()
        hold_action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        if hold_action:
            robot.send_action(hold_action)

        step_delay = duration_s / steps
        for i in range(steps):
            torque_value = int(1000 * (1.0 - (i + 1) / steps))
            for bus in buses:
                for motor_name in bus.motors:
                    try:
                        bus.write("Torque_Limit", motor_name, torque_value, normalize=False)
                    except Exception:
                        pass
            time.sleep(step_delay)

        logger.info("Soft landing complete")
    except Exception as e:
        logger.warning(f"Error during soft landing: {e}")


def restore_torque(robot):
    """Restore full torque (in case a previous soft_land left it at 0)."""
    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(robot, arm_name, None)
        if arm is not None:
            for motor_name in arm.bus.motors:
                try:
                    arm.bus.write("Torque_Limit", motor_name, 1000, normalize=False)
                except Exception:
                    pass


def build_s2_request(robot_obs: dict, task: str) -> dict:
    """Build extract_latent request for S2 (Pi0.5), using shared memory."""
    state = [float(robot_obs[name]) for name in JOINT_NAMES]

    images = {}
    for key in S2_IMAGE_KEYS:
        if key in robot_obs:
            img = np.asarray(robot_obs[key], dtype=np.uint8)
            shm_path = f"/dev/shm/dual_s2_{key}.npy"
            np.save(shm_path, img)
            images[key] = {"shm_path": shm_path}

    return {
        "mode": "extract_latent",
        "images": images,
        "high_level_prompt": task,
        "state": state,
    }


def obs_to_s1_batch(
    robot_obs: dict,
    s1_image_keys: list[str],
    s2_cache: S2LatentCache,
    s2_latent_key: str,
    device: torch.device,
) -> dict:
    """Convert robot observation to S1 input batch.

    Images are converted to [1, C, H, W] float tensors normalized to [0, 1].
    State is [1, 14] float tensor.
    S2 latent is pulled from cache (or omitted for zero fallback).
    """
    batch = {}

    # Images → [1, C, H, W] float tensors
    for key in s1_image_keys:
        # key is like "observation.images.front" → extract camera name
        cam_name = key.split(".")[-1]
        if cam_name in robot_obs:
            img = np.asarray(robot_obs[cam_name], dtype=np.uint8)  # HWC
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # CHW
            batch[key] = img_tensor.unsqueeze(0).to(device)

    # State → [1, 14]
    state = [float(robot_obs[name]) for name in JOINT_NAMES]
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    # S2 latent from cache
    latent, _ = s2_cache.get()
    if latent is not None:
        batch[s2_latent_key] = torch.from_numpy(latent).unsqueeze(0).to(device)

    return batch


# --------------- S2 Async Worker ---------------

async def s2_worker(
    robot,
    cache: S2LatentCache,
    server_uri: str,
    task: str,
    running: threading.Event,
    robot_lock: threading.Lock,
):
    """Async S2 loop: queries Pi0.5 for latent extraction at ~1Hz."""
    retry_delay = 1.0

    while running.is_set():
        try:
            async with websockets.connect(
                server_uri,
                ping_interval=60,
                ping_timeout=60,
                max_size=50 * 1024 * 1024,
            ) as ws:
                metadata = json.loads(await ws.recv())
                logger.info("S2 connected to %s (%s)", server_uri, metadata.get("model", "?"))
                retry_delay = 1.0

                while running.is_set():
                    start = time.perf_counter()

                    with robot_lock:
                        obs = robot.get_observation()
                    request = build_s2_request(obs, task)

                    await ws.send(json.dumps(request))
                    response = json.loads(await ws.recv())

                    if response.get("status") == "success":
                        latent = np.array(response["s2_latent"], dtype=np.float32)
                        prev_latent, _ = cache.get()
                        cache.update(latent)
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        timing = response.get("timing", {})
                        latent_norm = float(np.linalg.norm(latent))
                        if prev_latent is not None:
                            latent_diff = float(np.linalg.norm(latent - prev_latent))
                            diff_str = f" | Δlatent={latent_diff:.3f}"
                        else:
                            diff_str = ""
                        logger.info(
                            "S2 #%d: %.0fms (prefix: %.0fms) | norm=%.2f%s | first5=%s",
                            cache.update_count,
                            elapsed_ms,
                            timing.get("prefix_ms", 0),
                            latent_norm,
                            diff_str,
                            np.round(latent[:5], 3).tolist(),
                        )
                    else:
                        logger.warning("S2 error: %s", response.get("error"))

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            if running.is_set():
                logger.warning("S2 disconnected (%s), retrying in %.0fs...", e, retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)


def run_s2_thread(robot, cache, server_uri, task, running, robot_lock):
    """Run S2 worker in a dedicated thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(s2_worker(robot, cache, server_uri, task, running, robot_lock))
    except Exception as e:
        logger.error("S2 thread crashed: %s", e)
    finally:
        loop.close()


# --------------- Main ---------------

def control_loop(robot, events, args):
    """S1 control loop running at target FPS with cached S2 latent."""
    from lerobot.policies.act_vlm.modeling_act_vlm import ACTWithVLMPolicy, S2_LATENT_KEY
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device(args.device)

    # Load S1 policy
    logger.info("Loading S1 policy from %s...", args.s1_checkpoint)
    policy = ACTWithVLMPolicy.from_pretrained(pretrained_name_or_path=args.s1_checkpoint)
    policy.to(device)
    policy.eval()
    policy.reset()

    s1_image_keys = list(policy.config.image_features.keys())
    logger.info("S1 loaded. Image keys: %s | Device: %s", s1_image_keys, device)

    # Load preprocessor (normalization) and postprocessor (unnormalization)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.s1_checkpoint,
    )

    # Shared lock to prevent S1 and S2 from calling robot.get_observation() simultaneously
    robot_lock = threading.Lock()

    # Start S2 thread
    s2_cache = S2LatentCache()
    s2_running = threading.Event()
    s2_running.set()
    s2_thread = threading.Thread(
        target=run_s2_thread,
        args=(robot, s2_cache, f"ws://{args.s2_host}:{args.s2_port}", args.task, s2_running, robot_lock),
        daemon=True,
    )
    s2_thread.start()

    # Wait for first S2 latent
    logger.info("Waiting for first S2 latent...")
    wait_start = time.time()
    while s2_cache.latent is None and (time.time() - wait_start) < 15.0:
        time.sleep(0.1)
    if s2_cache.latent is not None:
        logger.info("Got first S2 latent (dim=%d)", s2_cache.latent.shape[0])
    else:
        logger.warning("No S2 latent after 15s, starting S1 with zero latent")

    # S1 control loop
    logger.info("Starting S1 loop at %d FPS", args.fps)
    step_count = 0
    action_queue = []
    s1_times = []
    prev_action_np = None  # for computing action diff

    try:
        while True:
            loop_start = time.perf_counter()

            # Check abort
            if events.get("stop_recording"):
                logger.info("Escape pressed — aborting")
                break

            # Refill action queue when empty
            if not action_queue:
                with robot_lock:
                    obs = robot.get_observation()
                batch = obs_to_s1_batch(obs, s1_image_keys, s2_cache, S2_LATENT_KEY, device)

                # Preprocess (normalization)
                batch = preprocessor(batch)

                with torch.no_grad():
                    actions = policy.select_action(batch)  # [1, action_dim], normalized
                    actions = postprocessor(actions)        # [1, action_dim], unnormalized

                # Convert to list of joint-name dicts
                actions_np = actions.cpu().numpy()
                for a in actions_np:
                    action_dict = {}
                    for i, name in enumerate(JOINT_NAMES):
                        if i < len(a):
                            action_dict[name] = float(a[i])
                    action_queue.append(action_dict)

                # Log action values and diff from previous chunk
                a = actions_np[0]
                if prev_action_np is not None:
                    action_diff = float(np.linalg.norm(a - prev_action_np))
                    diff_str = f" | Δaction={action_diff:.4f}"
                else:
                    diff_str = ""
                logger.info(
                    "S1 action (step %d)%s | %s",
                    step_count,
                    diff_str,
                    " ".join(f"{n.split('.')[0][:3]}={v:.3f}" for n, v in zip(JOINT_NAMES, a)),
                )
                prev_action_np = a

                infer_ms = (time.perf_counter() - loop_start) * 1000
                s1_times.append(infer_ms)

            # Execute next action
            action = action_queue.pop(0)
            with robot_lock:
                robot.send_action(action)
            step_count += 1

            if step_count % 100 == 0 and s1_times:
                avg_s1 = np.mean(s1_times[-20:])
                logger.info(
                    "Step %d | S1 infer: %.1fms | S2 age: %.0fms | S2 updates: %d | queue: %d",
                    step_count,
                    avg_s1,
                    s2_cache.age_ms,
                    s2_cache.update_count,
                    len(action_queue),
                )

            # FPS control
            dt = time.perf_counter() - loop_start
            sleep_s = max(1.0 / args.fps - dt, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        s2_running.clear()

        if s1_times:
            logger.info(
                "Done. %d steps | S1 avg infer: %.1fms | S2 updates: %d",
                step_count,
                np.mean(s1_times),
                s2_cache.update_count,
            )


def main():
    parser = argparse.ArgumentParser(description="Dual-System VLA Inference (S2: Pi0.5 + S1: ACTWithVLM)")
    parser.add_argument("--s1-checkpoint", required=True, help="Path to trained ACTWithVLM checkpoint")
    parser.add_argument("--task", required=True, help="High-level task prompt (for S2)")
    parser.add_argument("--s2-host", default="localhost", help="Pi0.5 server host")
    parser.add_argument("--s2-port", type=int, default=8765, help="Pi0.5 server port")
    parser.add_argument("--fps", type=int, default=FPS, help="S1 control loop FPS (default: 30)")
    parser.add_argument("--device", default="cuda", help="S1 device")
    parser.add_argument("--robot-config", type=str, default=str(ROBOT_CONFIG_FILE),
                        help="Path to robot config JSON")
    args = parser.parse_args()

    init_logging()

    # Load robot (same as openpi_infer.py)
    logger.info("Loading robot config from %s", args.robot_config)
    with open(args.robot_config) as f:
        robot_raw = json.load(f)
    robot_cfg = draccus.decode(RobotConfig, robot_raw)
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    restore_torque(robot)

    listener, events = init_keyboard_listener()

    try:
        control_loop(robot, events, args)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        log_say("Stopping", True)
        soft_land(robot)
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting robot: %s", e)
        if listener:
            listener.stop()
        log_say("Done", True)


if __name__ == "__main__":
    main()
