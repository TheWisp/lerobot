"""FlashDaggerSystem — top-level orchestrator for online flash-DAgger.

Owns:
  - the policy reference (with LoRA attached)
  - the per-episode intervention buffer
  - the session-scoped flashed pool (past corrections)
  - replay/forget index pools sampled once at construction
  - baseline forget loss (cached)
  - metrics logger

Lifecycle hooks called by the host loop (s1_process.py for HVLA):
  on_intervention_start()         operator entered intervention
  on_tick(obs, action)            per host-loop tick (filtered internally)
  on_intervention_end()           operator released intervention
  on_episode_end(episode, success) host loop hands off; system runs fit if buffer non-empty
  shutdown()                      end-of-session save + summary log

For v0 the loss/collate functions are bound to HVLA's flow-matching S1 at
construction. ACT/PI0 generalization is a future PR — replace
`_make_hvla_loss_fn` and `_make_hvla_collate_fn` with injected callables.
"""

from __future__ import annotations

import logging
import random

import torch
from torch.utils.data import DataLoader

from lerobot.policies.hvla.flash_dagger.buffer import (
    FlashedEpisodePool,
    InterventionFrameBuffer,
)
from lerobot.policies.hvla.flash_dagger.config import FlashDaggerConfig
from lerobot.policies.hvla.flash_dagger.fitter import (
    InterventionChunkDataset,
    evaluate_loss,
    fit_step_loop,
)
from lerobot.policies.hvla.flash_dagger.lora import (
    apply_lora_to_decoder,
    extract_lora_state_dict,
    load_lora_state_dict,
    lora_layer_diagnostics,
)
from lerobot.policies.hvla.flash_dagger.metrics import CycleMetrics, MetricsLogger, now_seconds
from lerobot.policies.hvla.flash_dagger.mix import ThreeWayMixDataset
from lerobot.policies.hvla.flash_dagger.persistence import save_lora

logger = logging.getLogger(__name__)


def _make_hvla_loss_fn(s1_config):
    """Bind context-based per-sample loss to a fixed S1 config.

    Online flash-DAgger feeds *pre-encoded* context to the loss (not raw
    images) — see FlashDaggerSystem._pre_encode_dataset_indices and
    ._encode_live_obs_batch. The encoder is frozen so the encode is wasted
    compute every fit step; we encode once per pool entry instead.

    Limitation: this design forecloses fine-tuning the encoder. Decoder-only
    LoRA is the recipe validated by phases B–F.
    """
    from lerobot.policies.hvla.scripts.flash_dagger_phase_a_rank import (
        compute_per_sample_loss_from_context,
    )

    def _loss(policy, batch):
        return compute_per_sample_loss_from_context(policy, batch, s1_config)

    return _loss


def _context_collate(samples: list[dict]) -> dict:
    """Stack a list of {context, action, action_is_pad} samples into a batch.

    All three keys are tensors, so default torch collate (stack along dim 0)
    works. We avoid the dataset's image-feature-aware collate since after
    encoding there are no per-camera image keys left.
    """
    out = {}
    keys = samples[0].keys()
    for k in keys:
        v0 = samples[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples])
        else:
            out[k] = [s[k] for s in samples]
    return out


class FlashDaggerSystem:
    """Orchestrator. One instance per teleop session.

    Construction is non-trivial:
      - applies LoRA to `policy` in-place
      - samples replay + forget index pools from `train_dataset`
      - computes baseline forget loss (one held-out eval pass)
    so it should be created once at session start, not per episode.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        s1_config,
        train_dataset,
        config: FlashDaggerConfig,
        device: torch.device,
        *,
        # Args needed to format live raw obs for encode_observations. These
        # come from the host loop's `obs_to_s1_batch` plumbing — see
        # s1_process.run_s1 for source values.
        s1_image_keys: list[str],
        resize_to: tuple[int, int] | None,
        shared_cache=None,
        s2_latent_key: str = "observation.s2_latent",
    ):
        self.policy = policy
        self.s1_config = s1_config
        self.train_dataset = train_dataset
        self.config = config
        self.device = device
        self.s1_image_keys = list(s1_image_keys)
        self.resize_to = resize_to
        self.shared_cache = shared_cache
        self.s2_latent_key = s2_latent_key

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = MetricsLogger(self.config.output_dir)

        # Attach LoRA in place. Idempotent only if not previously applied —
        # caller is responsible for not double-attaching.
        n_lora, n_total = apply_lora_to_decoder(
            policy,
            rank=config.rank,
            alpha=config.alpha,
            ffn=config.apply_to_ffn,
        )
        logger.info(
            "[flash-DAgger] LoRA attached: %d / %d params (%.2f%%)",
            n_lora,
            n_total,
            100.0 * n_lora / max(n_total, 1),
        )

        # The fit collate is for already-encoded (context, action, pad) samples;
        # the dataset collate (with image-feature aware key filtering) is only
        # used during pre-encoding from the HF dataset.
        from lerobot.policies.hvla.scripts.flash_dagger_phase_b import make_collate_fn

        self._dataset_collate = make_collate_fn(s1_config)
        self.collate_fn = _context_collate
        self.loss_fn = _make_hvla_loss_fn(s1_config)

        # Sample replay (for "old" slot) and forget-val (tripwire eval)
        # indices once. Frozen across the session for comparability.
        rng = random.Random(config.seed)
        n_train = len(train_dataset)
        if n_train == 0:
            raise ValueError("train_dataset is empty; flash-DAgger needs prior data")
        replay_indices = rng.sample(range(n_train), min(config.replay_pool_size, n_train))
        forget_val_indices = sorted(rng.sample(range(n_train), min(config.forget_val_size, n_train)))

        # Pre-encode both pools once. Subsequent fit cycles never re-encode
        # them — the encoder is frozen so the contexts are stable. Trade-off:
        # ~30s startup cost (5000 frames at batch 64 ≈ 80 forwards). Memory
        # is bounded by context tensor size (much smaller than image tensors).
        t0 = now_seconds()
        logger.info("[flash-DAgger] pre-encoding replay pool (%d frames)…", len(replay_indices))
        self.replay_samples: list[dict] = self._pre_encode_dataset_indices(replay_indices)
        logger.info(
            "[flash-DAgger] pre-encoding forget-val pool (%d frames)…",
            len(forget_val_indices),
        )
        self.forget_val_samples: list[dict] = self._pre_encode_dataset_indices(forget_val_indices)
        logger.info("[flash-DAgger] pre-encoding done in %.1fs", now_seconds() - t0)

        # Baseline forget loss: how good is the un-flashed policy on the
        # held-out training-set sample? Denominator for drift % each cycle.
        self.baseline_loss_old = self._eval_samples(self.forget_val_samples)
        logger.info("[flash-DAgger] baseline forget loss = %.4f", self.baseline_loss_old)

        # Per-session state
        self.intervention_buffer = InterventionFrameBuffer()
        self.flashed_pool = FlashedEpisodePool()
        self.intervention_active = False
        self.cycle_count = 0

    # ────────────────────────── lifecycle hooks ──────────────────────────

    def on_intervention_start(self) -> None:
        """Operator just entered intervention mode. Begin a new buffer segment.

        Multiple interventions in one episode each get their own segment so
        the chunked dataset (built at episode end) never crosses a boundary.
        """
        self.intervention_active = True
        self.intervention_buffer.begin_segment()

    def on_intervention_end(self) -> None:
        """Operator released intervention. Close the current segment."""
        self.intervention_active = False
        self.intervention_buffer.end_segment()
        logger.info(
            "[flash-DAgger] intervention ended; %d frames across %d segment(s) this episode",
            len(self.intervention_buffer),
            self.intervention_buffer.num_segments(),
        )

    def on_tick(self, obs: dict, action: torch.Tensor) -> None:
        """Per host-loop tick. Buffers iff intervention is active.

        `obs` should be a dict with the policy's expected observation keys
        (state, image features) — same shape the policy receives at inference.
        Tensors are moved to CPU before storing.
        `action` is the action actually executed at this tick (shape [action_dim]).
        """
        if not self.intervention_active:
            return
        cpu_obs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
        cpu_action = action.detach().cpu() if isinstance(action, torch.Tensor) else action
        self.intervention_buffer.append({"obs": cpu_obs, "action": cpu_action})

    def on_episode_end(self, episode: int, success: bool) -> None:
        """Episode boundary. Trigger fit if conditions met; clear buffer on exit."""
        try:
            n_frames = len(self.intervention_buffer)
            if n_frames == 0:
                return
            if not success:
                logger.info(
                    "[flash-DAgger] episode %d aborted; discarding %d intervention frames",
                    episode,
                    n_frames,
                )
                return
            chunk = self.s1_config.chunk_size
            if n_frames < max(self.config.min_intervention_frames, chunk + 1):
                logger.info(
                    "[flash-DAgger] episode %d: %d frames < min %d (chunk_size=%d); skipping fit",
                    episode,
                    n_frames,
                    max(self.config.min_intervention_frames, chunk + 1),
                    chunk,
                )
                return
            self._run_cycle(episode)
        finally:
            self.intervention_buffer.clear()

    def shutdown(self) -> None:
        """End-of-session: persist final summary log line."""
        logger.info(
            "[flash-DAgger] shutdown: %d cycles, %d corrections in flashed pool",
            self.cycle_count,
            len(self.flashed_pool),
        )

    # ────────────────────────── fit cycle internals ──────────────────────

    def _run_cycle(self, episode: int) -> CycleMetrics | None:
        cycle = self.cycle_count
        timings: dict[str, float] = {}

        # Snapshot per-intervention segments. Multiple interventions in one
        # episode produce multiple segments; we never let chunks span them.
        segments_raw = self.intervention_buffer.snapshot()
        n_total = sum(len(seg) for seg in segments_raw)

        # Encode raw obs → context once per captured tick. Encoder is frozen;
        # this is the only encode pass for the new pool (fit steps are
        # decoder-only). Done per-segment so segment metadata is preserved.
        #
        # Action normalization: live captures store the raw leader command
        # (joint angles in degrees), but the loss expects normalized actions
        # to match the dataset (which override_norm_stats z-scored at
        # dataset-build time). Apply (a - mean) / std here so the new pool's
        # actions sit on the same scale as old/flashed.
        #
        # Detection mirrors _encode_live_obs_batch: dataset-format obs (used
        # by smoke) means actions came from dataset[i] and are *already*
        # normalized; skip to avoid double-normalizing.
        t0 = now_seconds()
        is_dataset_format = bool(
            segments_raw
            and segments_raw[0]
            and (set(segments_raw[0][0]["obs"].keys()) & set(self.s1_image_keys))
        )
        action_mean = getattr(self.policy, "_action_mean", None)
        action_std = getattr(self.policy, "_action_std", None)
        do_normalize_action = not is_dataset_format and action_mean is not None and action_std is not None
        if do_normalize_action:
            mean_cpu = action_mean.detach().cpu()
            std_cpu = action_std.detach().cpu()
        segments_encoded: list[list[dict]] = []
        for seg in segments_raw:
            obs_list = [f["obs"] for f in seg]
            if do_normalize_action:
                actions = [(f["action"] - mean_cpu) / std_cpu for f in seg]
            else:
                actions = [f["action"] for f in seg]
            contexts = self._encode_live_obs_batch(obs_list)
            segments_encoded.append(
                [{"context": c, "action": a} for c, a in zip(contexts, actions, strict=True)]
            )
        timings["encode_live_seconds"] = now_seconds() - t0

        # Per-segment temporal train/val split: last val_pct of each segment
        # held out as val (contiguous tail). Random within-segment shuffle
        # would destroy temporal structure the chunk windowing relies on.
        #
        # A val tail must be at least chunk_size+1 long to yield ≥1 sliding-
        # window chunk, otherwise the eval would be empty. If a segment is
        # too short to support both train AND val with that minimum, we use
        # it entirely as train and accept losing val for that segment — the
        # alternative (skipping the whole cycle) is worse.
        chunk_size = self.s1_config.chunk_size
        min_chunkable = chunk_size + 1
        train_segments: list[list[dict]] = []
        val_segments: list[list[dict]] = []
        for seg in segments_encoded:
            n = len(seg)
            if n >= 2 * min_chunkable:
                # Long enough for both. Val tail = max(pct·n, min_chunkable)
                # so very long segments still cap their val at val_pct, while
                # borderline segments get exactly min_chunkable in val.
                n_val = max(int(n * self.config.val_pct), min_chunkable)
                train_segments.append(seg[: n - n_val])
                val_segments.append(seg[n - n_val :])
            elif n >= min_chunkable:
                # Too short to split usefully; keep entire segment as train.
                train_segments.append(seg)
            # else: segment shorter than chunk_size+1 — contributes nothing.

        new_pool_ds = InterventionChunkDataset(train_segments, chunk_size)
        new_val_ds = InterventionChunkDataset(val_segments, chunk_size)
        if len(new_pool_ds) == 0:
            logger.info(
                "[flash-DAgger] cycle %d: train chunks empty after split "
                "(%d segments, longest=%d, chunk=%d); skipping",
                cycle,
                len(segments_encoded),
                max((len(s) for s in segments_encoded), default=0),
                chunk_size,
            )
            return None
        if len(new_val_ds) == 0:
            logger.info(
                "[flash-DAgger] cycle %d: val chunks empty (segments too short "
                "for val tail >= chunk_size+1); proceeding without new_val metric",
                cycle,
            )

        new_pool_samples = [new_pool_ds[i] for i in range(len(new_pool_ds))]
        new_val_samples = [new_val_ds[i] for i in range(len(new_val_ds))]

        # Snapshot for revert on tripwire
        pre_lora_state = extract_lora_state_dict(self.policy)

        # ── pre-eval ────────────────────────────────────────────────────
        t_eval_pre = now_seconds()
        pre_new = self._eval_samples(new_val_samples)
        pre_old = self._eval_samples(self.forget_val_samples)
        pre_flashed = self._eval_flashed_avg() if len(self.flashed_pool) > 0 else None
        timings["pre_eval_seconds"] = now_seconds() - t_eval_pre

        # ── fit ─────────────────────────────────────────────────────────
        flashed_train_lists = [
            self.flashed_pool.train_pool(cid) for cid in self.flashed_pool.correction_ids()
        ]
        flashed_train_lists = [pool for pool in flashed_train_lists if len(pool) > 0]

        mix_ds = ThreeWayMixDataset(
            old_samples=self.replay_samples,
            flashed_pools=flashed_train_lists,
            new_pool=new_pool_samples,
            old_pct=self.config.old_pct,
            flashed_pct=self.config.flashed_pct,
            length=self.config.steps * self.config.batch_size,
        )
        loader = DataLoader(
            mix_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # samples are pre-encoded tensors; no need for workers
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        t_fit = now_seconds()
        curve = fit_step_loop(
            self.policy,
            loader,
            self.loss_fn,
            steps=self.config.steps,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            grad_clip=self.config.grad_clip,
            cosine_eta_min=self.config.lr * self.config.cosine_eta_min_frac,
            device=self.device,
        )
        timings["fit_seconds"] = now_seconds() - t_fit

        # ── post-eval ───────────────────────────────────────────────────
        t_eval_post = now_seconds()
        post_new = self._eval_samples(new_val_samples)
        post_old = self._eval_samples(self.forget_val_samples)
        post_flashed = self._eval_flashed_avg() if len(self.flashed_pool) > 0 else None
        timings["post_eval_seconds"] = now_seconds() - t_eval_post

        # ── swap-or-revert decision ─────────────────────────────────────
        baseline = max(self.baseline_loss_old, 1e-6)
        drift_pct = 100.0 * (post_old - baseline) / baseline
        swap_accepted = drift_pct <= self.config.forget_drift_abort_pct
        reject_reason = ""
        if not swap_accepted:
            reject_reason = f"forget drift {drift_pct:+.0f}% > {self.config.forget_drift_abort_pct:.0f}%"
            load_lora_state_dict(self.policy, pre_lora_state, strict_hash=False)
            post_old_after_revert = self._eval_samples(self.forget_val_samples)
            logger.warning(
                "[flash-DAgger] cycle %d: REVERTED — %s; old_val %.4f → %.4f after revert",
                cycle,
                reject_reason,
                post_old,
                post_old_after_revert,
            )

        # ── flashed-pool registration + persistence ─────────────────────
        # Flashed pool stores already-encoded chunk samples (matches the
        # mix dataset's expected sample shape; no further encoding needed).
        correction_id = -1
        t_save = now_seconds()
        if swap_accepted:
            correction_id = self.flashed_pool.add(new_pool_samples, new_val_samples)
            save_lora(
                self.policy,
                self.config.output_dir,
                cycle=cycle,
                rank=self.config.rank,
                alpha=self.config.alpha,
                apply_to_ffn=self.config.apply_to_ffn,
            )
        timings["save_seconds"] = now_seconds() - t_save

        # ── Layer B diagnostics (always logged for visibility) ──────────
        diag = lora_layer_diagnostics(self.policy)
        frob_max = max((r["frobenius"] for r in diag), default=0.0)
        eff_max = max((r["effective_rank"] for r in diag), default=0)

        m = CycleMetrics(
            cycle=cycle,
            episode=episode,
            correction_id=correction_id,
            n_intervention_frames=n_total,
            n_train_frames=len(new_pool_samples),
            n_val_frames=len(new_val_samples),
            n_steps=len(curve),
            wall_seconds=sum(timings.values()),
            loss_new_train_final=curve[-1] if curve else float("nan"),
            loss_new_val_pre=pre_new,
            loss_new_val_post=post_new,
            loss_old_val_pre=pre_old,
            loss_old_val_post=post_old,
            loss_flashed_val_pre=pre_flashed,
            loss_flashed_val_post=post_flashed,
            swap_accepted=swap_accepted,
            swap_reject_reason=reject_reason,
            n_lora_layers=len(diag),
            frobenius_max=frob_max,
            effective_rank_max=eff_max,
            encode_live_seconds=timings["encode_live_seconds"],
            pre_eval_seconds=timings["pre_eval_seconds"],
            fit_seconds=timings["fit_seconds"],
            post_eval_seconds=timings["post_eval_seconds"],
            save_seconds=timings["save_seconds"],
            n_segments=len(segments_encoded),
        )
        self.metrics_logger.write_cycle(m)
        self.metrics_logger.write_curve(cycle, curve)
        self.metrics_logger.write_layer_diag(cycle, diag)

        self.cycle_count += 1
        return m

    # ─────────────────────────── eval helpers ───────────────────────────

    def _eval_samples(self, samples: list[dict]) -> float:
        """Mean loss over a list of pre-encoded {context, action, action_is_pad} samples."""
        if not samples:
            return float("nan")
        return evaluate_loss(
            self.policy,
            samples,
            self.loss_fn,
            self.collate_fn,
            batch_size=self.config.batch_size,
            device=self.device,
            passes=1,
        )

    def _eval_flashed_avg(self) -> float:
        """Average val loss across all previously-flashed corrections."""
        cids = self.flashed_pool.correction_ids()
        if not cids:
            return float("nan")
        total = 0.0
        n = 0
        for cid in cids:
            val_samples = self.flashed_pool.val_pool(cid)
            if not val_samples:
                continue
            total += self._eval_samples(val_samples)
            n += 1
        return total / max(n, 1)

    # ──────────────────────── encoding helpers ──────────────────────────

    def _pre_encode_dataset_indices(self, indices: list[int]) -> list[dict]:
        """Encode a slice of the HF training dataset → list of training samples.

        Each sample is {"context": Tensor[N_ctx, D], "action": Tensor[T, A],
        "action_is_pad": Tensor[T] bool}. The encoder runs once per frame at
        startup; subsequent fit cycles never re-encode this pool.

        Memory: each context tensor is ~hundreds of KB; 5000 frames is in the
        low GB range. Stored on CPU; moved to device per-batch at fit time.
        """
        from torch.utils.data import DataLoader, Subset

        if not indices:
            return []
        subset = Subset(self.train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=self._dataset_collate,
        )
        samples: list[dict] = []
        was_training = self.policy.training
        self.policy.eval()
        try:
            for batch in loader:
                batch_dev = {
                    k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                # Dataset frames are ALREADY normalized at dataset build time
                # (override_norm_stats z-scores state and action). Calling
                # prepare_batch_for_encode_observations would re-normalize the
                # state. Mirror compute_per_sample_loss instead — just pack
                # OBS_IMAGES from the per-camera keys and pass to encode.
                if self.s1_config.image_features:
                    batch_dev = dict(batch_dev)
                    batch_dev["observation.images"] = [batch_dev[k] for k in self.s1_config.image_features]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ctx = self.policy.model.encode_observations(batch_dev)  # [B, N, D]
                action = batch_dev["action"]  # [B, T, A], already normalized
                pad = batch_dev.get(
                    "action_is_pad",
                    torch.zeros(action.shape[:2], dtype=torch.bool, device=action.device),
                )
                ctx_cpu = ctx.float().cpu()
                action_cpu = action.cpu()
                pad_cpu = pad.cpu()
                for i in range(ctx_cpu.shape[0]):
                    samples.append(
                        {
                            "context": ctx_cpu[i],
                            "action": action_cpu[i],
                            "action_is_pad": pad_cpu[i],
                        }
                    )
        finally:
            if was_training:
                self.policy.train()
        return samples

    def _encode_live_obs_batch(self, obs_dicts: list[dict]) -> list[torch.Tensor]:
        """Encode a list of raw robot-obs dicts → list of context tensors (CPU).

        Each input dict is the host loop's per-tick `obs` (camera images,
        raw joint positions). We mirror the InferenceThread's preprocessing:
        ``obs_to_s1_batch`` for image resize + key remap + state tensor
        construction, then ``prepare_batch_for_encode_observations`` for
        z-scoring, then ``encode_observations`` for the encoder forward.
        Batched along dim 0 for throughput.
        """
        from lerobot.policies.hvla.s1_process import obs_to_s1_batch

        if not obs_dicts:
            return []
        # Detect whether the obs dicts are already in dataset-format
        # (image keys like "observation.images.front") — happens in the
        # offline smoke path. Real on-robot captures use the host loop's
        # raw format (camera-name keys like "front") and need obs_to_s1_batch.
        first_keys = set(obs_dicts[0].keys())
        is_dataset_format = bool(first_keys & set(self.s1_image_keys))

        contexts: list[torch.Tensor] = []
        was_training = self.policy.training
        self.policy.eval()
        try:
            bs = self.config.batch_size
            for start in range(0, len(obs_dicts), bs):
                chunk_obs = obs_dicts[start : start + bs]
                if is_dataset_format:
                    # Dataset-format: state is already normalized (dataset's
                    # override_norm_stats), so we MUST NOT call prepare_batch
                    # _for_encode_observations (would re-normalize). Just add
                    # OBS_IMAGES list — same as compute_per_sample_loss does.
                    stacked = self._dataset_collate(list(chunk_obs))
                    stacked = {
                        k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                        for k, v in stacked.items()
                    }
                    if self.s1_config.image_features:
                        stacked["observation.images"] = [stacked[k] for k in self.s1_config.image_features]
                    prepared = stacked
                else:
                    # Raw robot-format: per-frame obs_to_s1_batch + stack,
                    # then prepare_batch_for_encode_observations to z-score
                    # state and pack OBS_IMAGES.
                    per_frame_batches = [
                        obs_to_s1_batch(
                            o,
                            self.s1_image_keys,
                            self.shared_cache,
                            self.s2_latent_key,
                            self.device,
                            resize_to=self.resize_to,
                        )
                        for o in chunk_obs
                    ]
                    stacked = {}
                    for k in per_frame_batches[0]:
                        if isinstance(per_frame_batches[0][k], torch.Tensor):
                            stacked[k] = torch.cat([b[k] for b in per_frame_batches], dim=0)
                    prepared = self.policy.prepare_batch_for_encode_observations(stacked)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ctx = self.policy.model.encode_observations(prepared)
                ctx_cpu = ctx.float().cpu()
                for i in range(ctx_cpu.shape[0]):
                    contexts.append(ctx_cpu[i])
        finally:
            if was_training:
                self.policy.train()
        return contexts
