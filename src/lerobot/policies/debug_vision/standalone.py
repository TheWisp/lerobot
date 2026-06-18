"""Standalone debug-vision process — runs a representation model alongside teleop.

Reads live camera frames from the always-on ObservationStream (any process can
attach read-only), runs a DebugVisionAdapter, and publishes a per-camera RGBA
overlay to a SharedOverlayBuffer that the GUI backend serves as PNG.

Usage:
    python -m lerobot.policies.debug_vision.standalone \
        --model grounding_dino --prompt "cup . bottle . robot arm ." \
        --cameras observation.images.top
"""

import argparse
import contextlib
import logging
import signal
import threading
import time

import numpy as np

from lerobot.policies.debug_vision.adapters import ADAPTERS, build_adapter
from lerobot.policies.debug_vision.overlay_ipc import SharedOverlayBuffer
from lerobot.robots.obs_stream import ObservationStreamReader

logger = logging.getLogger(__name__)


def _set_overlay(text: str, color: str = "#39d353") -> None:
    """Emit the GUI text-overlay protocol (parsed by run.py _append_output)."""
    print(f"##OVERLAY:{text}:{color}##", flush=True)


def _wait_for_obs_stream(stop: threading.Event) -> ObservationStreamReader | None:
    logger.info("waiting for observation stream (start teleop to begin)...")
    while not stop.is_set():
        try:
            return ObservationStreamReader()
        except (FileNotFoundError, RuntimeError):
            time.sleep(0.5)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone debug-vision overlay process")
    parser.add_argument("--model", required=True, choices=list(ADAPTERS))
    parser.add_argument(
        "--cameras", nargs="*", default=None, help="Camera keys to overlay (default: all in the obs stream)"
    )
    parser.add_argument("--prompt", default=None, help="Initial text prompt (Grounding DINO)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--throttle-ms", type=int, default=33, help="Min ms between inference passes (default ~30Hz)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    # Build the model FIRST so load failures (e.g. SAM3 gated weights) surface
    # immediately on Load — before we block waiting for teleop's obs stream.
    _set_overlay(f"loading {args.model} ...", "#e3b341")
    try:
        adapter = build_adapter(args.model, device=args.device)
    except Exception as e:
        # Surface load failures (e.g. SAM3 gated weights) to the GUI: a red
        # camera overlay + the actionable message (incl. any URL) in the Output
        # panel, instead of a silent crash.
        logger.error("failed to load debug-vision model '%s': %s", args.model, e)
        _set_overlay(f"{args.model}: failed to load — see Output panel", "#f85149")
        print(f"ERROR loading '{args.model}': {e}", flush=True)
        return
    if args.prompt:
        adapter.set_control({"prompt": args.prompt})
    logger.info("model '%s' ready", args.model)

    reader = _wait_for_obs_stream(stop)
    if reader is None:
        return
    logger.info("attached to obs stream; cameras available: %s", list(reader.image_keys))

    all_cams = list(reader.image_keys)
    if args.cameras:
        # Substring match so "top" selects "observation.images.top".
        cams = [c for c in all_cams if any(sub.lower() in c.lower() for sub in args.cameras)]
        if not cams:
            logger.warning("no camera matched %s; overlaying all of %s", args.cameras, all_cams)
            cams = all_cams
    else:
        cams = all_cams
    dims = {c: (int(reader.image_keys[c][0]), int(reader.image_keys[c][1])) for c in cams}
    logger.info("overlaying %s", dims)

    overlay = SharedOverlayBuffer(cameras=dims, model=args.model, create=True)
    _set_overlay(f"{adapter.label} — live", "#39d353")

    throttle = max(0.0, args.throttle_ms / 1000.0)
    try:
        while not stop.is_set():
            t0 = time.perf_counter()
            control = overlay.read_control()
            if control:
                adapter.set_control(control)
            for cam in cams:
                result = reader.read_image(cam)
                if result is None:
                    continue
                frame, _ts = result
                try:
                    rgba = adapter.infer(np.ascontiguousarray(frame))
                    overlay.write_overlay(cam, rgba)
                except Exception:
                    logger.exception("inference failed for camera %s", cam)
            dt = time.perf_counter() - t0
            if throttle > dt:
                time.sleep(throttle - dt)
    except Exception:
        logger.exception("debug-vision crashed")
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            overlay.cleanup()
        with contextlib.suppress(Exception):
            reader.close()
        logger.info("debug-vision shutdown complete")


if __name__ == "__main__":
    main()
