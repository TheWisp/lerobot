"""S1 process: loads action policy, reads shared latent, runs robot control loop.

Runs in the main process (needs direct robot/camera access). S1 is latency-critical (~30Hz).
Adapted from dual_system_infer.py.
"""

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

from lerobot.policies.hvla.ipc import SharedLatentCache, SharedImageBuffer

logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_forearm_roll.pos", "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_forearm_roll.pos", "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
]

# Map S1 robot obs camera names → S2 camera keys
S2_CAM_KEY_MAP = {
    "front": "base_0_rgb",
    "top": "base_1_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}


def obs_to_s1_batch(
    robot_obs: dict,
    s1_image_keys: list[str],
    shared_cache: SharedLatentCache,
    s2_latent_key: str,
    device: torch.device,
    resize_to: tuple[int, int] | None = None,
) -> dict:
    """Convert robot observation to S1 input batch.

    Images are resized on CPU before GPU transfer (0.6MB vs 10.5MB per image).
    """
    batch = {}

    for key in s1_image_keys:
        cam_name = key.split(".")[-1]
        if cam_name in robot_obs:
            img = np.asarray(robot_obs[cam_name], dtype=np.uint8)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # CHW uint8
            if resize_to is not None:
                img_tensor = TF.resize(img_tensor, list(resize_to),
                                       interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            batch[key] = img_tensor.unsqueeze(0).float().div_(255.0).to(device)

    state = [float(robot_obs[name]) for name in JOINT_NAMES]
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    latent, age_seconds = shared_cache.read_with_age()
    batch[s2_latent_key] = latent.unsqueeze(0).to(device)
    batch["observation.s2_latent_age"] = torch.tensor([[age_seconds]], dtype=torch.float32, device=device)

    return batch


def _warmup_s1(policy, preprocessor, s1_image_keys, device, resize_to):
    """Run dummy forward passes to trigger torch.compile kernel compilation.

    Uses fake data — no robot needed. After this, the control loop runs
    at full compiled speed with no compilation stalls.
    """
    import time as _time

    h, w = resize_to if resize_to else (224, 224)
    dummy_batch = {}
    for key in s1_image_keys:
        dummy_batch[key] = torch.randn(1, 3, h, w, device=device)
    dummy_batch["observation.state"] = torch.zeros(1, 14, device=device)
    dummy_batch["observation.s2_latent"] = torch.zeros(1, 2048, device=device)
    dummy_batch["observation.s2_latent_age"] = torch.zeros(1, 1, device=device)

    dummy_batch = preprocessor(dummy_batch)

    t0 = _time.perf_counter()
    with torch.no_grad():
        for i in range(3):
            policy.select_action(dummy_batch)
            if i == 0:
                logger.info("S1: First compiled forward done (%.1fs)", _time.perf_counter() - t0)
    policy.reset()  # clear any state from warmup
    logger.info("S1: Warmup complete (%.1fs total)", _time.perf_counter() - t0)


def run_s1(
    s1_checkpoint: str,
    shared_cache: SharedLatentCache,
    shared_images: SharedImageBuffer,
    task: str,
    robot_config_path: str | None = None,
    fps: int = 30,
    device: str = "cuda",
    resize_images: tuple[int, int] | None = (224, 224),
    temporal_ensemble_coeff: float | None = None,
    n_action_steps: int | None = None,
    compile_s1: bool = False,
    stop_event=None,
):
    """S1 control loop with robot. Runs in main process."""
    # Main process logging should already be configured by launch.py,
    # but ensure it works even if run standalone.
    from lerobot.policies.hvla.logging_utils import setup_process_logging
    if not logging.getLogger().handlers:
        setup_process_logging()

    import json
    import draccus
    from lerobot.policies.act_vlm.modeling_act_vlm import ACTWithVLMPolicy, S2_LATENT_KEY
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.config import RobotConfig
    from lerobot.utils.control_utils import init_keyboard_listener

    # Register robot/camera types for draccus
    from lerobot.robots import bi_so107_follower  # noqa: F401
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

    device = torch.device(device)

    # Load S1 policy
    logger.info("S1: Loading policy from %s", s1_checkpoint)
    policy = ACTWithVLMPolicy.from_pretrained(pretrained_name_or_path=s1_checkpoint)

    if temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
        policy.config.n_action_steps = 1
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, policy.config.chunk_size)
        logger.info("S1: Temporal ensembling (coeff=%.3f)", temporal_ensemble_coeff)
    elif n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
        logger.info("S1: n_action_steps=%d", n_action_steps)

    policy.to(device)
    policy.eval()
    policy.reset()

    s1_image_keys = list(policy.config.image_features.keys())
    logger.info("S1: Policy loaded. Image keys: %s", s1_image_keys)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=s1_checkpoint,
    )

    # torch.compile is opt-in via --compile-s1 flag.
    # DINOv2 attention layers are not compatible with torch.compile out of the box.
    # TODO: investigate torch.compile with fullgraph=False or compiling only the
    # ACT encoder/decoder (excluding DINOv2 backbone).
    if compile_s1:
        logger.info("S1: Compiling model with torch.compile (mode=default)...")
        policy.model = torch.compile(policy.model, mode="default")
        logger.info("S1: Warming up compiled model...")
        _warmup_s1(policy, preprocessor, s1_image_keys, device, resize_images)

    # Load robot
    config_path = robot_config_path or str(Path.home() / ".config" / "chop" / "robot_config.json")
    logger.info("S1: Loading robot from %s", config_path)
    with open(config_path) as f:
        robot_raw = json.load(f)
    robot_cfg = draccus.decode(RobotConfig, robot_raw)
    robot = make_robot_from_config(robot_cfg)
    logger.info("S1: Connecting to robot...")
    robot.connect()
    logger.info("S1: Robot connected")

    # Keyboard listener for abort
    listener, events = init_keyboard_listener()

    # Note: S2 wait happens after inference thread starts (below),
    # because the inference thread publishes images that S2 needs.

    logger.info("S1: Starting control loop at %d FPS", fps)

    # --- Pipelined inference: background thread computes chunks, main loop executes at fixed rate ---
    #
    # Design:
    # - Inference thread: captures obs → preps batch → runs model → publishes (chunk, t_obs)
    # - Main loop: at each tick, computes index = ceil((now - t_obs) * fps) into chunk, sends action
    # - Time-based indexing means the main loop always picks the action corresponding to "now",
    #   regardless of when the chunk was produced or how long inference took.
    # - Thread safety: chunk is an immutable np.ndarray reference swap (atomic in CPython).

    import threading

    # --- Pipelined S1 ---
    # Main loop: captures obs → publishes to S2 + inference thread → sends action at fixed rate
    # Inference thread: reads obs from buffer → preps → infers → publishes (chunk, t_obs)
    # Main loop owns ALL robot I/O (cameras + motors not thread-safe).

    import threading

    _chunk_data = None       # np.ndarray [chunk_size, action_dim]
    _chunk_t_obs = 0.0       # perf_counter when obs was captured for current chunk
    _chunk_lock = threading.Lock()
    _chunk_ready = threading.Event()

    # Obs buffer: main loop writes, inference thread reads
    _obs_data = None
    _obs_time = 0.0
    _obs_lock = threading.Lock()
    _obs_ready = threading.Event()

    _infer_running = threading.Event()
    _infer_running.set()
    s1_infer_times = []
    inference_delays = []

    # Smoothness tracking
    prev_action_np = None
    action_deltas = []
    loop_intervals = []
    last_send_time = None

    def _inference_thread():
        """Reads obs from buffer → preps → infers → publishes chunk. No robot access."""
        nonlocal _chunk_data, _chunk_t_obs

        while _infer_running.is_set():
            if not _obs_ready.wait(timeout=0.5):
                continue
            _obs_ready.clear()

            with _obs_lock:
                obs = _obs_data
                t_obs = _obs_time

            if obs is None:
                continue

            # Prepare batch (CPU resize + GPU transfer)
            batch = obs_to_s1_batch(obs, s1_image_keys, shared_cache, S2_LATENT_KEY, device,
                                    resize_to=resize_images)
            batch = preprocessor(batch)

            # Inference
            t_infer = time.perf_counter()
            with torch.no_grad():
                actions = policy.predict_action_chunk(batch)  # [1, chunk_size, action_dim]
                actions = postprocessor(actions)
            infer_ms = (time.perf_counter() - t_infer) * 1000
            s1_infer_times.append(infer_ms)
            inference_delays.append(time.perf_counter() - t_obs)

            chunk_np = actions.cpu().numpy()[0]

            with _chunk_lock:
                _chunk_data = chunk_np
                _chunk_t_obs = t_obs
                _chunk_ready.set()

    # Start inference thread (no robot access — only GPU)
    infer_thread = threading.Thread(target=_inference_thread, daemon=True)
    infer_thread.start()

    # Capture initial obs, publish to both S2 and inference thread
    logger.info("S1: Capturing initial observation...")
    init_obs = robot.get_observation()
    shared_images.write_images(init_obs, S2_CAM_KEY_MAP, JOINT_NAMES)
    with _obs_lock:
        _obs_data = init_obs
        _obs_time = time.perf_counter()
    _obs_ready.set()

    # Wait for S2
    logger.info("S1: Waiting for first S2 latent (up to 120s)...")
    if not shared_cache.wait_for_first(timeout=120.0):
        logger.warning("S1: No S2 latent after 120s, starting with zero latent")
    else:
        logger.info("S1: Got first S2 latent (count=%d)", shared_cache.count)

    # Wait for first chunk
    logger.info("S1: Waiting for first action chunk...")
    _chunk_ready.wait(timeout=60.0)
    if _chunk_data is None:
        logger.error("S1: No action chunk in 60s, aborting")
        _infer_running.clear()
        return

    logger.info("S1: First chunk ready, starting at %d FPS", fps)
    step_count = 0

    try:
        while stop_event is None or not stop_event.is_set():
            loop_start = time.perf_counter()

            if events.get("stop_recording"):
                logger.info("S1: Escape pressed — aborting")
                break

            # 1. Capture observation (main loop owns robot)
            obs = robot.get_observation()
            t_now = time.perf_counter()

            # 2. Publish to inference thread + S2
            with _obs_lock:
                _obs_data = obs
                _obs_time = t_now
            _obs_ready.set()
            shared_images.write_images(obs, S2_CAM_KEY_MAP, JOINT_NAMES)

            # 3. Read latest chunk
            with _chunk_lock:
                chunk = _chunk_data
                t_obs = _chunk_t_obs

            if chunk is None:
                time.sleep(1.0 / fps)
                continue

            # 4. Index chunk and send action — computed RIGHT BEFORE send for minimum staleness.
            # t_obs = when the inference thread captured the obs that produced this chunk.
            # elapsed = total time from that obs capture to now (includes: inference thread
            # waiting for obs, prep, model inference, chunk publish, AND main loop's own
            # obs capture + S2 publish overhead in this iteration).
            # The chunk predicts action[i] for time t_obs + i/fps. We want the action
            # for "right now", so idx = ceil(elapsed * fps).
            t_before_send = time.perf_counter()
            elapsed = t_before_send - t_obs
            idx = math.ceil(elapsed * fps)
            idx = max(0, min(idx, len(chunk) - 1))
            action_np = chunk[idx]

            action_dict = {name: float(action_np[i]) for i, name in enumerate(JOINT_NAMES) if i < len(action_np)}
            robot.send_action(action_dict)
            t_after_send = time.perf_counter()

            # Smoothness tracking (after send, not on critical path)
            if prev_action_np is not None:
                action_deltas.append(np.linalg.norm(action_np - prev_action_np))
            prev_action_np = action_np.copy()

            if last_send_time is not None:
                loop_intervals.append((t_after_send - last_send_time) * 1000)
            last_send_time = t_after_send
            step_count += 1

            # Periodic logging
            if step_count % 100 == 0:
                smooth_str = ""
                if action_deltas:
                    r = action_deltas[-20:]
                    smooth_str += f" | Δaction: {np.mean(r):.3f}/{np.max(r):.3f}"
                if loop_intervals:
                    r = loop_intervals[-20:]
                    smooth_str += f" | interval: {np.mean(r):.1f}±{np.std(r):.1f}ms"
                if s1_infer_times:
                    smooth_str += f" | infer: {np.mean(s1_infer_times[-10:]):.0f}ms"
                smooth_str += f" | chunk_age: {elapsed*1000:.0f}ms"

                logger.info(
                    "S1 step %d | chunk[%d/%d] | S2 age: %.0fms | S2 #%d%s",
                    step_count, idx, len(chunk),
                    shared_cache.age_ms, shared_cache.count, smooth_str,
                )

            # Fixed-rate sleep
            dt = time.perf_counter() - loop_start
            sleep_s = max(1.0 / fps - dt, 0.0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("S1: Interrupted by user")
    finally:
        _infer_running.clear()
        infer_thread.join(timeout=3.0)
        if s1_infer_times:
            logger.info("S1: Done. %d steps | avg infer: %.1fms | S2 updates: %d",
                        step_count, np.mean(s1_infer_times), shared_cache.count)
        _soft_land(robot)
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning("Robot disconnect error (non-fatal): %s", e)
        if listener is not None:
            listener.stop()


def _soft_land(robot, duration_s=4.0, steps=20):
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
        # Hold current position first
        obs = robot.get_observation()
        hold_action = {k: v for k, v in obs.items() if k.endswith(".pos")}
        if hold_action:
            robot.send_action(hold_action)

        # Gradually reduce torque
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

        # Disable torque completely so next run can re-enable
        for bus in buses:
            for motor_name in bus.motors:
                try:
                    bus.write("Torque_Enable", motor_name, 0)
                except Exception:
                    pass

        # Restore torque limit so next run isn't stuck at 0
        for bus in buses:
            for motor_name in bus.motors:
                try:
                    bus.write("Torque_Limit", motor_name, 1000, normalize=False)
                except Exception:
                    pass

        logger.info("S1: Soft landing complete")
    except Exception as e:
        logger.warning("S1: Soft landing error: %s", e)
