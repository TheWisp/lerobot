"""Launch dual-system HVLA inference: spawns S2 process + runs S1 in main process.

S2 (VLM, ~4-15Hz) extracts scene latent → shared memory.
S1 (action policy, ~22-30Hz) reads latent, controls robot.

Usage:
    python -m lerobot.policies.hvla.launch \
        --s1-checkpoint outputs/act_vlm/checkpoint-80000 \
        --s2-checkpoint ~/.cache/lerobot/converted/soarm-pi05-pytorch/model.safetensors \
        --task "assemble cylinder into ring" \
        --resize-images 224x224 \
        --temporal-ensemble-coeff 0.01
"""

import argparse
import logging
import multiprocessing
import os
import signal

from lerobot.policies.hvla.ipc import SharedLatentCache, SharedImageBuffer
from lerobot.policies.hvla.logging_utils import setup_process_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HVLA dual-system inference (local, no server)")
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", default=None,
                        help="Path to S2 checkpoint (not needed with --attach-s2 or --zero-s2)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot-config", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-images", default="224x224")
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--decode-subtask", action="store_true")
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--s2-image-keys", nargs="+",
                        default=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb"])
    parser.add_argument("--zero-s2", action="store_true")
    parser.add_argument("--compile-s1", action="store_true",
                        help="torch.compile S1 model (experimental, may not work with DINOv2)")
    parser.add_argument("--s1-type", choices=["act", "flow"], default="act",
                        help="S1 policy type: 'act' (ACTWithVLM, default) or 'flow' (flow matching with RTC)")
    parser.add_argument("--s2-throttle-ms", type=int, default=100,
                        help="Sleep ms after each S2 query to yield GPU to S1 (0=no throttle)")
    parser.add_argument("--osc-skip", action="store_true",
                        help="Enable both-arms oscillation skip: when both arms are flat, "
                             "jump ahead in chunk to where movement starts")
    parser.add_argument("--denoise-steps", type=int, default=None,
                        help="Number of flow matching denoising steps (default: 5 from config)")
    parser.add_argument("--s1-query-interval", type=int, default=5,
                        help="Number of action steps to wait before re-querying S1. "
                             "E.g. 20 = execute 20 actions (~660ms at 30fps) from current "
                             "chunk before next inference. 0 = query as fast as possible (default).")
    parser.add_argument("--max-step-delta", type=float, default=None,
                        help="Max degrees any joint can change per frame (e.g. 10). Prevents sudden jumps.")
    parser.add_argument("--save-grip-drops", type=str, default=None,
                        help="Directory to save observations when grip drops are detected (for offline analysis)")
    args = parser.parse_args()

    # s2-checkpoint only needed if no existing S2 process is found
    # (validated later, after attempting to attach)

    setup_process_logging()

    resize = tuple(int(x) for x in args.resize_images.split("x")) if args.resize_images else None
    s2_image_keys = tuple(args.s2_image_keys)

    # Use 'spawn' context — child process gets fresh CUDA context
    ctx = multiprocessing.get_context("spawn")

    # Shared memory IPC — try to attach to existing S2, fall back to spawning
    s2_attached = False
    if not args.zero_s2:
        # Try to attach to an existing S2 process (started by s2_standalone.py)
        try:
            shared_cache = SharedLatentCache(create=False, name=SharedLatentCache.SHM_NAME)
            shared_images = SharedImageBuffer(camera_keys=s2_image_keys, create=False)
            s2_attached = True
            logger.info("Attached to existing S2 (latent count=%d, age=%.0fms)",
                        shared_cache.count, shared_cache.age_ms)
        except FileNotFoundError:
            # No existing S2 — need to spawn one
            if args.s2_checkpoint is None:
                logger.error("No S2 process found and no --s2-checkpoint provided.\n"
                             "Either start S2 first:\n"
                             "  python -m lerobot.policies.hvla.s2_standalone --checkpoint ... --task ...\n"
                             "Or provide --s2-checkpoint to spawn one.")
                return
            logger.info("No existing S2 found, will spawn from checkpoint")
            shared_cache = SharedLatentCache(create=True)
            shared_images = SharedImageBuffer(camera_keys=s2_image_keys, create=True)
    else:
        shared_cache = SharedLatentCache(create=True)
        shared_images = SharedImageBuffer(camera_keys=s2_image_keys, create=True)

    stop_event = ctx.Event()

    # Ctrl+C handling: first = graceful, second = force quit
    _ctrl_c_count = 0

    def signal_handler(sig, frame):
        nonlocal _ctrl_c_count
        _ctrl_c_count += 1
        if _ctrl_c_count >= 2:
            logger.warning("Force quit (second Ctrl+C)")
            os._exit(1)
        logger.info("Ctrl+C — shutting down (press again to force quit)...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start S2 process (unless already attached or using zero stub)
    s2_proc = None
    if s2_attached:
        logger.info("Using existing S2 process (no spawn)")
    elif args.zero_s2:
        import torch
        logger.info("--zero-s2: S2 disabled, using zero latent (ablation mode)")
        shared_cache.write(torch.zeros(2048))
    else:
        from lerobot.policies.hvla.s2_process import run_s2

        s2_proc = ctx.Process(
            target=run_s2,
            kwargs={
                "checkpoint_path": args.s2_checkpoint,
                "shared_cache": shared_cache,
                "shared_images": shared_images,
                "task": args.task,
                "device": args.device,
                "image_keys": s2_image_keys,
                "decode_subtask": args.decode_subtask,
                "norm_stats_path": args.norm_stats,
                "stop_event": stop_event,
                "throttle_ms": args.s2_throttle_ms,
            },
            daemon=True,
        )
        s2_proc.start()
        logger.info("S2 process started (PID %d)", s2_proc.pid)

    # Run S1 in main process
    from lerobot.policies.hvla.s1_process import run_s1

    try:
        run_s1(
            s1_checkpoint=args.s1_checkpoint,
            shared_cache=shared_cache,
            shared_images=shared_images,
            task=args.task,
            robot_config_path=args.robot_config,
            fps=args.fps,
            device=args.device,
            resize_images=resize,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            n_action_steps=args.n_action_steps,
            compile_s1=args.compile_s1,
            s1_type=args.s1_type,
            stop_event=stop_event,
            osc_skip=args.osc_skip,
            query_interval_steps=args.s1_query_interval,
            num_denoise_steps=args.denoise_steps,
            max_step_delta=args.max_step_delta,
            grip_drop_save_dir=args.save_grip_drops,
        )
    finally:
        stop_event.set()
        if s2_proc is not None:
            s2_proc.join(timeout=5.0)
            if s2_proc.is_alive():
                logger.warning("S2 process didn't exit cleanly, terminating")
                s2_proc.terminate()
        shared_images.cleanup()
        shared_cache.cleanup()
        logger.info("HVLA shutdown complete")


if __name__ == "__main__":
    main()
