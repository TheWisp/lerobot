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
import collections
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


def _resolve_active(filter_names, all_cams: list[str]) -> set[str]:
    """Camera keys to actually run inference on, from a requested filter.

    Substring-matches each requested name against the obs-stream keys (so "top"
    selects "observation.images.top"). Empty/None, or a filter that matches nothing,
    falls back to all cameras. Used both at launch (``--cameras``) and live (the
    ``cameras`` control), so the active set can change without recreating the buffer.
    """
    if not filter_names:
        return set(all_cams)
    matched = {c for c in all_cams if any(str(s).lower() in c.lower() for s in filter_names)}
    return matched or set(all_cams)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone debug-vision overlay process")
    parser.add_argument("--model", required=True, choices=list(ADAPTERS))
    parser.add_argument(
        "--cameras", nargs="*", default=None, help="Camera keys to overlay (default: all in the obs stream)"
    )
    parser.add_argument("--prompt", default=None, help="Initial text prompt (Grounding DINO / SAM3)")
    parser.add_argument(
        "--objects",
        default=None,
        help='Initial monitored objects, JSON [{"name","color":[r,g,b],"sign":"+"}]',
    )
    parser.add_argument(
        "--background",
        default=None,
        help='Background fill JSON {"color":[r,g,b]} (null/absent = transparent)',
    )
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
    import json

    init_control: dict = {}
    if args.objects:
        try:
            init_control["objects"] = json.loads(args.objects)
        except Exception:
            logger.warning("ignoring malformed --objects: %s", args.objects)
    elif args.prompt:
        init_control["prompt"] = args.prompt
    if args.background:
        try:
            init_control["background"] = json.loads(args.background)
        except Exception:
            logger.warning("ignoring malformed --background: %s", args.background)
    if init_control:
        adapter.set_control(init_control)
    logger.info("model '%s' ready", args.model)

    reader = _wait_for_obs_stream(stop)
    if reader is None:
        return
    logger.info("attached to obs stream; cameras available: %s", list(reader.image_keys))

    all_cams = list(reader.image_keys)
    # The overlay buffer covers ALL cameras so the active filter can change live (via
    # the `cameras` control) without recreating shared memory. Disabled cameras are
    # skipped in the loop — no inference, so the filter doubles as a compute dial.
    dims = {c: (int(reader.image_keys[c][0]), int(reader.image_keys[c][1])) for c in all_cams}
    active = _resolve_active(args.cameras, all_cams)
    logger.info("cameras=%s active=%s", all_cams, sorted(active))

    overlay = SharedOverlayBuffer(cameras=dims, model=args.model, create=True)
    _set_overlay(f"{adapter.label} — live", "#39d353")

    throttle = max(0.0, args.throttle_ms / 1000.0)
    last_active = set(active)
    fps_window: collections.deque = collections.deque(maxlen=20)  # rolling iteration times
    fps_prev = time.perf_counter()
    fps_last_emit = fps_prev
    try:
        while not stop.is_set():
            t0 = time.perf_counter()
            control = overlay.read_control()
            if control:
                if "cameras" in control:
                    active = _resolve_active(control.get("cameras"), all_cams)
                adapter.set_control(control)
            for cam in all_cams:
                if cam not in active:
                    continue
                result = reader.read_image(cam)
                if result is None:
                    continue
                frame, _ts = result
                try:
                    rgba = adapter.infer(np.ascontiguousarray(frame))
                    overlay.write_overlay(cam, rgba)
                except Exception:
                    logger.exception("inference failed for camera %s", cam)
            # Clear overlays for cameras just switched off so a stale mask doesn't linger.
            for cam in last_active - active:
                h, w = dims[cam]
                with contextlib.suppress(Exception):
                    overlay.write_overlay(cam, np.zeros((h, w, 4), dtype=np.uint8))
            last_active = set(active)
            dt = time.perf_counter() - t0
            if throttle > dt:
                time.sleep(throttle - dt)
            # Achieved overlay rate (incl. throttle) → ##FPS## for the GUI panel, ~1 Hz.
            now = time.perf_counter()
            fps_window.append(now - fps_prev)
            fps_prev = now
            if now - fps_last_emit >= 1.0 and fps_window:
                print(f"##FPS:{len(fps_window) / sum(fps_window):.1f}##", flush=True)
                fps_last_emit = now
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
