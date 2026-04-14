"""Background inference thread for S1 policy.

Encapsulates the pipelined inference loop: receives observations from the main
control loop, runs model inference on GPU, and publishes action chunks.

Supports pause/resume for human intervention: when paused, the thread stops
consuming observations and producing chunks, freeing the GPU.
"""

import logging
import threading
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


class InferenceThread:
    """Background GPU inference with pause/resume support.

    The main loop publishes observations via publish_obs().
    The inference thread consumes them, runs the model, and publishes chunks.
    The main loop reads chunks via get_chunk().

    For intervention: pause() blocks the inference loop (GPU idle),
    resume() resets the policy and unblocks.
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        shared_cache,
        s2_latent_key: str,
        s1_image_keys: list[str],
        joint_names: list[str],
        device: torch.device,
        resize_to: tuple[int, int] | None,
        fps: int,
        num_denoise_steps: int | None = None,
        query_interval_steps: int = 0,
        grip_drop_save_dir: str | None = None,
        # RLT parameters (all None when RLT is disabled)
        rl_token_encoder=None,
        rlt_actor=None,
        rlt_agent=None,
        rlt_state: dict | None = None,
        rlt_replay=None,
    ):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._shared_cache = shared_cache
        self._s2_latent_key = s2_latent_key
        self._s1_image_keys = s1_image_keys
        self._joint_names = joint_names
        self._device = device
        self._resize_to = resize_to
        self._fps = fps
        self._num_denoise_steps = num_denoise_steps
        self._query_interval_s = query_interval_steps / fps if query_interval_steps > 0 else 0.0
        self._grip_drop_save_dir = grip_drop_save_dir

        # RLT components
        self._rl_token_encoder = rl_token_encoder
        self._rlt_actor = rlt_actor
        self._rlt_agent = rlt_agent  # TD3Agent (for gradient updates)
        self._rlt_state = rlt_state
        self._rlt_replay = rlt_replay
        # When True, inference thread runs actor + stores transitions + runs gradients.
        # Set False during: intervention, reset, pre-episode.
        # Main thread controls this flag.
        self._rlt_active = False
        self._rlt_prev: dict | None = None  # previous transition for replay buffer
        self._rlt_step_count: int = 0

        # RTC config
        self._supports_rtc = getattr(policy, "supports_rtc", False)
        self._rtc_max_delay = getattr(policy, "rtc_prefix_length", 5) if self._supports_rtc else 0

        # Chunk state (written by inference thread, read by main loop)
        self._chunk_data: np.ndarray | None = None
        self._chunk_t_obs: float = 0.0
        self._chunk_t_origin: float = 0.0
        self._chunk_prefix_len: int = 0
        self._chunk_lock = threading.Lock()
        self._chunk_ready = threading.Event()

        # Obs buffer (written by main loop, read by inference thread)
        self._obs_data: dict | None = None
        self._obs_time: float = 0.0
        self._obs_lock = threading.Lock()
        self._obs_ready = threading.Event()

        # RTC: main loop reports which chunk index it's executing
        self._main_loop_chunk_idx: int = 0
        self._main_loop_chunk_idx_lock = threading.Lock()

        # Lifecycle
        self._running = threading.Event()
        self._running.set()
        self._paused = threading.Event()
        self._paused.set()  # starts unpaused

        # Timing
        self.infer_times: list[float] = []
        self.inference_delays: list[float] = []

        self._thread: threading.Thread | None = None

    # --- RLT methods ---

    def _rlt_inference_step(self, batch, actions, z_rl, obs):
        """Run RLT actor (if past warmup) and store transition.
        Called inside inference loop — one call per S1 chunk.
        Only runs when _rlt_active is True (main thread controls this).

        Paper Algorithm 1:
        - Line 7: S1 produces ref chunk (already done before this call)
        - Line 8: Form RL state x = (z_rl, s^p)
        - Line 9: Actor produces action (or VLA ref during warmup)
        - Line 12: Store transition in replay buffer
        """
        cfg = self._rlt_state["config"]
        C = cfg.rl_chunk_length
        is_warmup = self._rlt_state["episode"] <= cfg.warmup_episodes

        # Normalize state
        state_t = batch.get("observation.state")
        if state_t is None:
            state_np = np.array(
                [float(obs.get(j, 0)) for j in self._joint_names],
                dtype=np.float32,
            )
            state_t = torch.from_numpy(state_np).unsqueeze(0).to(self._device)
        state_norm = state_t.float()
        if self._policy._state_mean is not None:
            state_norm = (state_t.float() - self._policy._state_mean.to(self._device)) / self._policy._state_std.to(self._device)

        # Normalize ref chunk (first C actions from S1)
        ref_chunk = actions[:, :C, :].float()
        ref_norm = ref_chunk
        if self._policy._action_mean is not None:
            ref_norm = (ref_chunk - self._policy._action_mean.to(self._device)) / self._policy._action_std.to(self._device)

        # Actor forward (Paper Alg 1 line 9)
        is_deploy = self._rlt_state.get("deploy", False)
        if is_warmup and not is_deploy:
            # Warmup: execute VLA reference, actor not called
            actor_norm = ref_norm
        else:
            actor_norm = self._rlt_actor(
                z_rl.float(), state_norm, ref_norm,
                deterministic=is_deploy,  # no exploration noise in deploy mode
            )
            # Denormalize and replace first C actions in output
            if self._policy._action_mean is not None:
                actor_denorm = actor_norm * self._policy._action_std.to(self._device) + self._policy._action_mean.to(self._device)
            else:
                actor_denorm = actor_norm
            actions[:, :C, :] = actor_denorm

        # Store transition in replay buffer (Paper Alg 1 line 12)
        # Guard: _rlt_active may have been cleared by main thread during this call
        # (race between main thread setting flag and inference thread mid-method).
        # If cleared, skip storage but don't crash — the actor already modified
        # the actions tensor which is fine (it'll be overwritten next inference).
        if not self._rlt_active:
            logger.warning("RLT: _rlt_active cleared mid-inference, skipping transition storage")
            self._rlt_prev = None
            return

        prev = self._rlt_prev
        if prev is not None and self._rlt_replay is not None:
            done = self._rlt_state.get("reward_triggered", False)
            self._rlt_replay.add(
                z_rl=prev["z_rl"], state=prev["state"],
                action_chunk=prev["action"], ref_chunk=prev["ref"],
                reward=1.0 if done else 0.0,
                next_z_rl=z_rl.squeeze(0).float().detach(),
                next_state=state_norm.squeeze(0).detach(),
                next_ref_chunk=ref_norm.squeeze(0).detach(),
                done=done,
            )
            self._rlt_state["total_transitions"] += 1

        self._rlt_prev = {
            "z_rl": z_rl.squeeze(0).float().detach(),
            "state": state_norm.squeeze(0).detach(),
            "action": actor_norm.squeeze(0).detach(),
            "ref": ref_norm.squeeze(0).detach(),
        }

        # Metrics + logging
        self._rlt_step_count += 1
        actor_delta = (actor_norm - ref_norm).abs().mean().item()

        from lerobot.policies.hvla.rlt.metrics import get_metrics, save_metrics_to_file
        is_deploy = self._rlt_state.get("deploy", False)
        mode = "DEPLOY" if is_deploy else ("WARMUP" if is_warmup else "RL")
        get_metrics().record_step(
            step=self._rlt_step_count, delta=actor_delta,
            buffer_size=len(self._rlt_replay) if self._rlt_replay else 0,
            total_updates=self._rlt_state["total_updates"], mode=mode,
        )
        if self._rlt_step_count % 100 == 0:
            logger.info(
                "RLT step %d [%s] | delta=%.3f | buf=%d | updates=%d",
                self._rlt_step_count, mode, actor_delta,
                len(self._rlt_replay) if self._rlt_replay else 0,
                self._rlt_state["total_updates"],
            )
            save_metrics_to_file()

    def _rlt_check_config_overrides(self):
        """Check for GUI config overrides (beta, sigma sliders).
        Called each gradient step — cost is one stat() syscall (~1us)."""
        import json as _json
        override_path = self._rlt_state["output_dir"] / "rlt_overrides.json"
        try:
            mtime = override_path.stat().st_mtime
        except OSError:
            return
        last = getattr(self, "_rlt_override_mtime", 0)
        if mtime <= last:
            return
        self._rlt_override_mtime = mtime
        try:
            with open(override_path) as f:
                overrides = _json.load(f)
            cfg = self._rlt_state["config"]
            for key in ("beta", "actor_sigma"):
                if key in overrides:
                    old = getattr(cfg, key)
                    new = float(overrides[key])
                    if old != new:
                        setattr(cfg, key, new)
                        logger.info("RLT config override: %s %.4f → %.4f", key, old, new)
        except Exception as e:
            logger.warning("RLT config override read failed: %s", e)

    def _rlt_gradient_updates(self):
        """Run UTD gradient steps during query interval idle time.
        Paper: UTD=5, two critic updates per actor update.
        Paper Alg 1 lines 13-18: runs every step including warmup.
        """
        import time as _time
        t0 = _time.perf_counter()

        self._rlt_check_config_overrides()
        cfg = self._rlt_state["config"]
        C = cfg.rl_chunk_length
        A_flat = self._rlt_replay._action.shape[1]
        A = A_flat // C

        critic_sum, actor_sum = 0.0, 0.0
        n_critic, n_actor = 0, 0

        for _ in range(cfg.utd_ratio):
            # 2 critic updates (Paper Appendix B)
            for _ in range(2):
                b = self._rlt_replay.sample(256)
                cl = self._rlt_agent.update_critic(
                    b["z_rl"], b["state"],
                    b["action"].reshape(-1, C, A), b["ref"].reshape(-1, C, A),
                    b["reward"], b["next_z_rl"], b["next_state"],
                    b["next_ref"].reshape(-1, C, A), b["done"],
                )
                critic_sum += cl
                n_critic += 1
            # 1 actor update (Paper Alg 1 line 17)
            b = self._rlt_replay.sample(256)
            al = self._rlt_agent.update_actor(
                b["z_rl"], b["state"], b["ref"].reshape(-1, C, A),
            )
            actor_sum += al
            n_actor += 1
            self._rlt_state["total_updates"] += 1

        elapsed = (_time.perf_counter() - t0) * 1000
        avg_c = critic_sum / n_critic if n_critic else 0
        avg_a = actor_sum / n_actor if n_actor else 0

        # Log Q values for monitoring (Paper's primary metric for critic quality)
        with torch.no_grad():
            b = self._rlt_replay.sample(min(256, len(self._rlt_replay)))
            qs = self._rlt_agent.critic.min_q(
                b["z_rl"], b["state"],
                b["action"].reshape(-1, C, A),
            )
            q_mean = qs.mean().item()
            q_min = qs.min().item()
            q_max = qs.max().item()

        if self._rlt_state["total_updates"] % 10 < cfg.utd_ratio:
            logger.info(
                "RLT grad | critic=%.4f actor=%.4f | Q: mean=%.4f min=%.4f max=%.4f | updates=%d | %.0fms",
                avg_c, avg_a, q_mean, q_min, q_max,
                self._rlt_state["total_updates"], elapsed,
            )

        # Update metrics with Q values
        from lerobot.policies.hvla.rlt.metrics import get_metrics
        get_metrics().record_step(
            step=self._rlt_step_count, delta=0,
            buffer_size=len(self._rlt_replay) if self._rlt_replay else 0,
            total_updates=self._rlt_state["total_updates"],
            mode="TRAIN", critic_loss=avg_c, actor_loss=avg_a,
            q_mean=q_mean, q_min=q_min, q_max=q_max,
        )

    def start(self) -> None:
        """Start the background inference thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal thread to stop and join."""
        self._running.clear()
        self._paused.set()  # unblock if paused
        self._obs_ready.set()  # unblock if waiting for obs
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def pause(self) -> None:
        """Pause inference (for intervention). Thread blocks until resume()."""
        self._paused.clear()
        logger.info("S1 inference: paused")

    def resume(self) -> None:
        """Resume inference after intervention. Resets policy state."""
        self._policy.reset()
        # Clear stale obs so we get a fresh one
        self._obs_ready.clear()
        self._paused.set()
        logger.info("S1 inference: resumed")

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def publish_obs(self, obs: dict, t_now: float) -> None:
        """Main loop publishes observation for the inference thread."""
        with self._obs_lock:
            self._obs_data = obs
            self._obs_time = t_now
        self._obs_ready.set()

    def get_chunk(self) -> tuple[np.ndarray | None, float, float]:
        """Read latest (chunk, t_origin, t_obs). Thread-safe."""
        with self._chunk_lock:
            return self._chunk_data, self._chunk_t_origin, self._chunk_t_obs

    def update_exec_index(self, idx: int) -> None:
        """Main loop reports which chunk index it just executed (for RTC prefix)."""
        with self._main_loop_chunk_idx_lock:
            self._main_loop_chunk_idx = idx

    def wait_for_first_chunk(self, timeout: float = 60.0) -> bool:
        """Block until first chunk is ready. Returns False on timeout."""
        return self._chunk_ready.wait(timeout=timeout)

    # --- Internal ---

    def _loop(self) -> None:
        """Background loop: wait for obs → prep batch → infer → publish chunk."""
        from lerobot.policies.hvla.s1.protocol import ACTION_PREFIX_KEY
        from lerobot.policies.hvla.s1_process import obs_to_s1_batch, _save_infer_drop

        while self._running.is_set():
            # Pause gate: block here during intervention
            if not self._paused.wait(timeout=0.5):
                continue

            if not self._running.is_set():
                break

            # Wait for fresh obs from main loop
            if not self._obs_ready.wait(timeout=0.5):
                continue
            self._obs_ready.clear()

            with self._obs_lock:
                obs = self._obs_data
                t_obs = self._obs_time

            if obs is None:
                continue

            # Prepare batch (CPU resize + GPU transfer)
            batch = obs_to_s1_batch(
                obs, self._s1_image_keys, self._shared_cache,
                self._s2_latent_key, self._device, resize_to=self._resize_to,
            )
            batch = self._preprocessor(batch)

            # RTC prefix
            current_prefix_len = 0
            exec_idx = None
            expected_d = 0
            if self._supports_rtc:
                with self._chunk_lock:
                    old_chunk = self._chunk_data
                with self._main_loop_chunk_idx_lock:
                    exec_idx = self._main_loop_chunk_idx
                    t_exec_idx = time.perf_counter()

                if old_chunk is not None and exec_idx < len(old_chunk):
                    if self.inference_delays:
                        expected_d = round(np.mean(self.inference_delays[-10:]) * self._fps)
                    else:
                        expected_d = 3
                    expected_d = max(1, min(expected_d, self._rtc_max_delay,
                                           len(old_chunk) - exec_idx))

                    prefix = old_chunk[exec_idx : exec_idx + expected_d]
                    batch[ACTION_PREFIX_KEY] = torch.from_numpy(prefix).unsqueeze(0).to(self._device)
                    current_prefix_len = prefix.shape[0]

            # Inference
            t_infer_start = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # RLT: extract z_rl from S1 context tokens
                z_rl_out = None
                if self._rl_token_encoder is not None:
                    s1_config = self._policy.config
                    if s1_config.image_features:
                        batch["observation.images"] = [
                            batch[key] for key in s1_config.image_features
                        ]
                    context = self._policy.model.encode_observations(batch)
                    z_rl_out = self._rl_token_encoder(context.float()).detach()

                # S1 flow matching decoder → reference chunk
                actions = self._policy.predict_action_chunk(batch, num_steps=self._num_denoise_steps)
                actions = self._postprocessor(actions)

                # RLT: actor refines chunk + store transition
                if self._rlt_active and self._rlt_state is not None and z_rl_out is not None:
                    self._rlt_inference_step(batch, actions, z_rl_out, obs)

            t_infer_end = time.perf_counter()
            infer_ms = (t_infer_end - t_infer_start) * 1000
            self.infer_times.append(infer_ms)
            total_delay = t_infer_end - t_obs
            self.inference_delays.append(total_delay)

            chunk_np = actions.cpu().numpy()[0]

            # RTC diagnostics (periodic)
            if self._supports_rtc and (len(self.infer_times) <= 5 or len(self.infer_times) % 50 == 0):
                actual_d_frames = round((t_infer_end - t_infer_start) * self._fps)
                obs_to_infer_ms = (t_infer_start - t_obs) * 1000
                prefix_drift = None
                if current_prefix_len > 0:
                    inner_model = self._policy.model if hasattr(self._policy, 'model') else self._policy
                    prefix_drift = getattr(inner_model, '_last_prefix_drift', None)
                logger.info(
                    "S1 RTC diag | expected_d=%d actual_d=%d prefix_len=%d "
                    "| obs→infer=%.0fms infer=%.0fms total=%.0fms "
                    "| prefix_drift=%s exec_idx=%s",
                    expected_d, actual_d_frames, current_prefix_len,
                    obs_to_infer_ms, infer_ms, total_delay * 1000,
                    f"{prefix_drift:.4f}" if prefix_drift is not None else "N/A",
                    exec_idx,
                )

            # RTC alignment diagnostic
            with self._chunk_lock:
                old_chunk_for_align = self._chunk_data

            if self._supports_rtc and old_chunk_for_align is not None and exec_idx is not None:
                scan_center = exec_idx
                best_offset = 0
                best_err = float("inf")
                errors = []
                for offset in range(max(0, scan_center - 3), min(len(old_chunk_for_align) - 1, scan_center + 8)):
                    n = min(8, len(chunk_np), len(old_chunk_for_align) - offset)
                    if n <= 0:
                        continue
                    err = np.mean(np.abs(chunk_np[:n] - old_chunk_for_align[offset:offset + n]))
                    errors.append(f"{offset}={err:.1f}")
                    if err < best_err:
                        best_err = err
                        best_offset = offset
                if len(errors) > 0 and (len(self.infer_times) <= 5 or len(self.infer_times) % 50 == 0):
                    logger.info(
                        "S1 RTC align | exec_idx=%d best=%d (err=%.2f) | %s",
                        exec_idx, best_offset, best_err, " ".join(errors),
                    )

            # Grip drop diagnostics
            if self._grip_drop_save_dir:
                _save_infer_drop(chunk_np, obs, len(self.infer_times), self._grip_drop_save_dir)
                infer_count = len(self.infer_times)
                if infer_count % 50 == 0:
                    import os
                    ctrl_dir = os.path.join(self._grip_drop_save_dir, f"control_{infer_count}")
                    os.makedirs(ctrl_dir, exist_ok=True)
                    state_arr = np.array([float(obs.get(j, 0)) for j in self._joint_names])
                    np.save(os.path.join(ctrl_dir, "state.npy"), state_arr)
                    np.save(os.path.join(ctrl_dir, "chunk.npy"), chunk_np)

            # Publish chunk
            with self._chunk_lock:
                self._chunk_data = chunk_np
                self._chunk_t_obs = t_obs
                if self._supports_rtc and exec_idx is not None and 't_exec_idx' in dir():
                    self._chunk_t_origin = t_exec_idx
                else:
                    self._chunk_t_origin = t_obs
                self._chunk_prefix_len = current_prefix_len
                self._chunk_ready.set()

            # Query interval: use idle time for RLT gradient updates
            if self._query_interval_s > 0 and self._running.is_set():
                if (self._rlt_state is not None
                        and self._rlt_replay is not None
                        and len(self._rlt_replay) >= 256):
                    self._rlt_gradient_updates()
                    # Sleep remaining time if updates were faster than query interval
                    elapsed_since_infer = time.perf_counter() - t_infer_end
                    remaining = self._query_interval_s - elapsed_since_infer
                    if remaining > 0:
                        time.sleep(remaining)
                else:
                    time.sleep(self._query_interval_s)
