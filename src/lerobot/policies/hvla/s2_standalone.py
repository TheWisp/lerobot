"""Standalone S2 process — keeps VLM hot between S1 restarts.

Usage:
    # Terminal 1: start S2 (stays running)
    python -m lerobot.policies.hvla.s2_standalone \
        --checkpoint ~/.cache/lerobot/converted/soarm-pi05-state-11997-pytorch/model.safetensors \
        --task "assemble cylinder into ring"

    # Terminal 2: start S1 (connects to existing S2 shared memory)
    python -m lerobot.policies.hvla.launch \
        --s1-type flow --s1-checkpoint outputs/flow_s1_hvla_v3/checkpoint-50000/model.safetensors \
        --task "assemble cylinder into ring" --resize-images 224x224 \
        --attach-s2
"""

import argparse
import logging
import signal
import sys
import threading

from lerobot.policies.hvla.ipc import SharedLatentCache, SharedImageBuffer
from lerobot.policies.hvla.logging_utils import setup_process_logging
from lerobot.policies.hvla.s2_process import run_s2

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Standalone S2 VLM process (persistent)")
    parser.add_argument("--checkpoint", required=True, help="Path to Pi0.5 model.safetensors")
    parser.add_argument("--task", required=True, help="High-level task prompt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--s2-image-keys", nargs="+",
                        default=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb"])
    parser.add_argument("--decode-subtask", action="store_true")
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--throttle-ms", type=int, default=100)
    args = parser.parse_args()

    setup_process_logging()

    s2_image_keys = tuple(args.s2_image_keys)

    # Create shared memory with well-known names (S1 will attach by name)
    logger.info("Creating shared memory (latent: '%s', images: '%s*')...",
                SharedLatentCache.SHM_NAME, SharedImageBuffer.SHM_PREFIX)
    shared_cache = SharedLatentCache(create=True, name=SharedLatentCache.SHM_NAME)
    shared_images = SharedImageBuffer(camera_keys=s2_image_keys, create=True)

    logger.info("Shared memory ready. S1 can connect with --attach-s2")
    logger.info("S2 will stay running until Ctrl+C.")

    stop_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("Ctrl+C — shutting down S2...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run_s2(
            checkpoint_path=args.checkpoint,
            shared_cache=shared_cache,
            shared_images=shared_images,
            task=args.task,
            device=args.device,
            image_keys=s2_image_keys,
            decode_subtask=args.decode_subtask,
            norm_stats_path=args.norm_stats,
            stop_event=stop_event,
            throttle_ms=args.throttle_ms,
        )
    except Exception:
        logger.exception("S2 standalone crashed")
    finally:
        logger.info("Cleaning up shared memory...")
        try:
            shared_cache.cleanup()
        except Exception:
            pass
        try:
            shared_images.cleanup()
        except Exception:
            pass
        logger.info("S2 standalone shutdown complete.")


if __name__ == "__main__":
    main()
