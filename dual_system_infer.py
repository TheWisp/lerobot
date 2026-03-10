#!/usr/bin/env python
"""Dual-System VLA Inference: S2 (Pi0.5) + S1 (ACTWithVLM)

S2 thread: Async coroutine queries Pi0.5 at ~1Hz for scene-understanding latent.
S1 loop: Synchronous main loop runs ACTWithVLM at ~30-140Hz for reactive control.

The S2 latent is injected into S1 via a thread-safe shared cache.
If S2 disconnects, S1 continues with the last cached latent (graceful degradation).

Usage:
    python dual_system_infer.py \
        --s1-checkpoint /path/to/act_vlm_checkpoint \
        --s2-server ws://localhost:8765 \
        --high-level-prompt "pick up the cup" \
        --robot-port /dev/ttyUSB0
"""

import argparse
import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("dual_system")


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


class S2Worker:
    """Async S2 worker that queries Pi0.5 for latent extraction."""

    def __init__(
        self,
        server_uri: str,
        cache: S2LatentCache,
        high_level_prompt: str,
        image_keys: list[str],
    ):
        self.server_uri = server_uri
        self.cache = cache
        self.high_level_prompt = high_level_prompt
        self.image_keys = image_keys
        self._running = False

    async def run(self, get_observation_fn):
        """Main S2 loop. Runs until stopped.

        Args:
            get_observation_fn: Callable that returns (images_dict, state_array).
                images_dict maps model keys to numpy uint8 HWC arrays.
        """
        self._running = True
        retry_delay = 1.0

        while self._running:
            try:
                async with websockets.connect(
                    self.server_uri, max_size=50 * 1024 * 1024
                ) as ws:
                    # Read metadata
                    metadata = json.loads(await ws.recv())
                    logger.info("S2 connected to %s (%s)", self.server_uri, metadata.get("model", "?"))
                    retry_delay = 1.0

                    while self._running:
                        start = time.time()

                        # Get current observation
                        images, state = get_observation_fn()

                        # Encode images
                        images_data = {}
                        for key, img in images.items():
                            raw = img.tobytes()
                            images_data[key] = {
                                "base64": base64.b64encode(raw).decode("ascii"),
                                "shape": list(img.shape),
                            }

                        request = {
                            "mode": "extract_latent",
                            "images": images_data,
                            "high_level_prompt": self.high_level_prompt,
                        }
                        if state is not None:
                            request["state"] = state.tolist()

                        await ws.send(json.dumps(request))
                        response = json.loads(await ws.recv())

                        if response.get("status") == "success":
                            latent = np.array(response["s2_latent"], dtype=np.float32)
                            self.cache.update(latent)
                            elapsed_ms = (time.time() - start) * 1000
                            logger.debug(
                                "S2 update #%d: %.0fms (prefix: %.0fms)",
                                self.cache.update_count,
                                elapsed_ms,
                                response["timing"].get("prefix_ms", 0),
                            )
                        else:
                            logger.warning("S2 error: %s", response.get("error"))

            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                if self._running:
                    logger.warning("S2 disconnected (%s), retrying in %.0fs...", e, retry_delay)
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30.0)

    def stop(self):
        self._running = False


class DualSystemController:
    """Orchestrates S1 (ACTWithVLM) + S2 (Pi0.5) for real-time robot control."""

    def __init__(
        self,
        s1_checkpoint: str,
        s2_server_uri: str,
        high_level_prompt: str,
        s1_device: str = "cuda",
        n_action_steps: int | None = None,
    ):
        self.s2_server_uri = s2_server_uri
        self.high_level_prompt = high_level_prompt
        self.s1_device = s1_device
        self.s2_cache = S2LatentCache()

        # Load S1 policy
        logger.info("Loading S1 policy from %s...", s1_checkpoint)
        from lerobot.policies.act_vlm.configuration_act_vlm import ACTWithVLMConfig
        from lerobot.policies.act_vlm.modeling_act_vlm import ACTWithVLMPolicy, S2_LATENT_KEY

        self.S2_LATENT_KEY = S2_LATENT_KEY
        self.s1_policy = ACTWithVLMPolicy.from_pretrained(pretrained_name_or_path=s1_checkpoint)
        self.s1_policy.to(s1_device)
        self.s1_policy.eval()
        logger.info("S1 loaded. Device: %s", s1_device)

        if n_action_steps is not None:
            self.s1_policy.config.n_action_steps = n_action_steps

    def run_s1_step(self, images: list[torch.Tensor], state: torch.Tensor) -> np.ndarray:
        """Run one S1 inference step.

        Args:
            images: List of [1, C, H, W] image tensors (one per camera)
            state: [1, state_dim] state tensor

        Returns:
            action: [action_dim] numpy array
        """
        batch = {}

        # Add images
        for i, (key, _) in enumerate(self.s1_policy.config.image_features.items()):
            batch[key] = images[i].to(self.s1_device)
        batch["observation.state"] = state.to(self.s1_device)

        # Add S2 latent from cache
        latent, ts = self.s2_cache.get()
        if latent is not None:
            batch[self.S2_LATENT_KEY] = torch.from_numpy(latent).unsqueeze(0).to(self.s1_device)
        # If no latent yet, the model handles it with zeros

        with torch.no_grad():
            action = self.s1_policy.select_action(batch)

        return action.squeeze(0).cpu().numpy()

    async def run(self, robot, get_observation_fn, duration_s: float = 60.0):
        """Main dual-system control loop.

        Args:
            robot: Robot interface with send_action(action) method.
            get_observation_fn: Returns (images_dict_for_s2, state_np,
                                         images_tensors_for_s1, state_tensor_for_s1)
            duration_s: Max duration in seconds.
        """
        # Start S2 worker
        s2_worker = S2Worker(
            server_uri=self.s2_server_uri,
            cache=self.s2_cache,
            high_level_prompt=self.high_level_prompt,
            image_keys=[],
        )

        def s2_obs_fn():
            """Observation getter for S2 (returns raw images + state)."""
            obs = get_observation_fn()
            return obs[0], obs[1]  # images_dict, state_np

        s2_task = asyncio.create_task(s2_worker.run(s2_obs_fn))

        # Wait for first S2 latent (with timeout)
        logger.info("Waiting for first S2 latent...")
        wait_start = time.time()
        while self.s2_cache.latent is None and (time.time() - wait_start) < 10.0:
            await asyncio.sleep(0.1)
        if self.s2_cache.latent is not None:
            logger.info("Got first S2 latent (dim=%d)", self.s2_cache.latent.shape[0])
        else:
            logger.warning("No S2 latent after 10s, starting S1 with zero latent")

        # Main S1 loop
        logger.info("Starting S1 control loop (target: ~%.0fHz)", 1000 / 15)
        self.s1_policy.reset()
        start_time = time.time()
        step_count = 0
        s1_times = []

        try:
            while (time.time() - start_time) < duration_s:
                step_start = time.time()

                # Get observation for S1
                obs = get_observation_fn()
                images_s1 = obs[2]  # list of [1, C, H, W] tensors
                state_s1 = obs[3]  # [1, state_dim] tensor

                # S1 inference
                action = self.run_s1_step(images_s1, state_s1)

                # Send to robot
                robot.send_action(action)

                step_count += 1
                s1_ms = (time.time() - step_start) * 1000
                s1_times.append(s1_ms)

                if step_count % 100 == 0:
                    avg_s1 = np.mean(s1_times[-100:])
                    logger.info(
                        "Step %d | S1: %.1fms (%.0fHz) | S2 age: %.0fms | S2 updates: %d",
                        step_count,
                        avg_s1,
                        1000 / max(avg_s1, 1),
                        self.s2_cache.age_ms,
                        self.s2_cache.update_count,
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            s2_worker.stop()
            s2_task.cancel()
            try:
                await s2_task
            except asyncio.CancelledError:
                pass

            total_time = time.time() - start_time
            if s1_times:
                logger.info(
                    "Done. %d steps in %.1fs (avg %.1fHz). S2 updates: %d",
                    step_count,
                    total_time,
                    step_count / total_time,
                    self.s2_cache.update_count,
                )


def main():
    parser = argparse.ArgumentParser(description="Dual-System VLA Inference")
    parser.add_argument("--s1-checkpoint", required=True, help="Path to ACTWithVLM checkpoint")
    parser.add_argument("--s2-server", default="ws://localhost:8765", help="Pi0.5 WebSocket server URI")
    parser.add_argument("--high-level-prompt", default="do the task", help="Task prompt for S2")
    parser.add_argument("--device", default="cuda", help="S1 device")
    parser.add_argument("--duration", type=float, default=60.0, help="Max duration in seconds")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Override n_action_steps")
    args = parser.parse_args()

    controller = DualSystemController(
        s1_checkpoint=args.s1_checkpoint,
        s2_server_uri=args.s2_server,
        high_level_prompt=args.high_level_prompt,
        s1_device=args.device,
        n_action_steps=args.n_action_steps,
    )

    # Placeholder: real robot and observation function would be injected here.
    # For now, print usage instructions.
    print(
        "\nDualSystemController initialized.\n"
        "To use with a real robot, call:\n"
        "  asyncio.run(controller.run(robot, get_observation_fn, duration_s=60))\n"
        "\n"
        "get_observation_fn should return:\n"
        "  (images_dict_s2, state_np, images_tensors_s1, state_tensor_s1)\n"
        "\n"
        "See the DualSystemController.run() docstring for details.\n"
    )


if __name__ == "__main__":
    main()
