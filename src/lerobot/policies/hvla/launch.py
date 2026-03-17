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
    parser.add_argument("--s2-checkpoint", required=True)
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
    args = parser.parse_args()

    setup_process_logging()

    resize = tuple(int(x) for x in args.resize_images.split("x")) if args.resize_images else None
    s2_image_keys = tuple(args.s2_image_keys)

    # Use 'spawn' context — child process gets fresh CUDA context
    ctx = multiprocessing.get_context("spawn")

    # Shared memory IPC (named blocks, survive spawn/pickle)
    shared_cache = SharedLatentCache(latent_dim=2048)
    shared_images = SharedImageBuffer(camera_keys=s2_image_keys)
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

    # Start S2 process
    s2_proc = None
    if args.zero_s2:
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
