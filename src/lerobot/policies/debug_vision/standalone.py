"""Standalone debug-vision process — runs a representation model alongside teleop.

Reads live camera frames from the always-on ObservationStream (any process can
attach read-only), runs a DebugVisionAdapter, and publishes a per-camera RGBA
overlay to a SharedOverlayBuffer that the GUI backend serves as PNG.

Usage:
    python -m lerobot.policies.debug_vision.standalone \
        --model sam3_track --prompt "robot arm . cylinder . green ring" \
        --cameras observation.images.top
"""

import argparse
import atexit
import contextlib
import logging
import signal
import threading
import time

import numpy as np

from lerobot.policies.debug_vision.adapters import ADAPTERS, build_adapter
from lerobot.policies.debug_vision.overlay_ipc import OverlayStatus, SharedOverlayBuffer
from lerobot.robots.obs_stream import ObservationStreamReader

logger = logging.getLogger(__name__)


def _vram_gb() -> float:
    """The loaded model's own VRAM footprint in GB — this process's live CUDA allocation
    (so it stays flat once warmed and climbs if the model leaks). 0.0 without CUDA."""
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    return 0.0


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


def _try_reattach(old_reader, expected_cams: list[str]) -> ObservationStreamReader | None:
    """Re-attach to the obs stream if the publisher was restarted.

    teleop stop+start creates a *fresh* shm segment with the same name; the existing
    reader still maps the dead one and never sees new frames again. When idle, probe
    for a replacement: same camera keys but a reset (lower) sequence counter. Returns
    the new reader on a genuine restart, else None — a merely paused stream maps to the
    same live segment (same high seq) and must NOT be swapped.
    """
    try:
        new = ObservationStreamReader()
    except (FileNotFoundError, RuntimeError):
        return None
    if set(new.image_keys) != set(expected_cams):
        new.close()  # different stream shape — don't silently swap (restart the overlay instead)
        return None
    old_max = max((old_reader.image_seq(c) for c in expected_cams), default=0)
    new_max = max((new.image_seq(c) for c in expected_cams), default=0)
    if new_max < old_max:
        logger.info("obs stream replaced (seq %d -> %d): re-attaching", old_max, new_max)
        return new
    new.close()  # same live segment (paused) — release the probe, keep the current reader
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
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        _set_overlay("requires an NVIDIA GPU (CUDA) — unavailable", "#f85149")
        print("ERROR: Overlays require an NVIDIA GPU (CUDA), which isn't available on this host.", flush=True)
        raise SystemExit(1)
    _set_overlay(f"loading {args.model} ...", "#e3b341")
    # The standalone owns the lifecycle phase: 'loading' now (created BEFORE the model load),
    # flipped to 'active' once loaded. The GUI reads this to drive the badge state machine.
    status = OverlayStatus(create=True)
    atexit.register(status.cleanup)  # free the status shm on every exit path (return / SystemExit / SIGTERM)
    try:
        adapter = build_adapter(args.model, device=args.device)
    except Exception as e:
        # Surface load failures (e.g. SAM3 gated weights) to the GUI: a red
        # camera overlay + the actionable message (incl. any URL) in the Output
        # panel, and exit non-zero so the badge shows "error" — never a crash.
        logger.error("failed to load debug-vision model '%s': %s", args.model, e)
        _set_overlay(f"{args.model}: failed to load — see Output panel", "#f85149")
        print(f"ERROR loading '{args.model}': {e}", flush=True)
        raise SystemExit(1) from e
    # transformers (imported by build_adapter) clears the root logging handlers, which would
    # silence every INFO below — including the per-second "live: N infer/s" activity line, so
    # a WORKING stream would log nothing. Re-assert our config so the loop is actually visible.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
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
    status.write("active", 0.0, _vram_gb())  # loaded; the GUI badge goes active (fps 0 until frames flow)

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
    # Clear every camera up front so a reused shm segment can't surface a
    # previous run's stale overlays on cameras this run won't touch.
    for _cam, (_h, _w) in dims.items():
        overlay.write_overlay(_cam, np.zeros((_h, _w, 4), dtype=np.uint8))
    _set_overlay(f"{adapter.label} — live", "#39d353")

    throttle = max(0.0, args.throttle_ms / 1000.0)
    last_active = set(active)
    last_seq: dict[str, int] = {}  # per-camera obs-stream seq — gate inference on new frames
    infer_loops = 0  # productive iterations since the last emit (0 while the stream is idle)
    idle_secs = 0  # consecutive ~1s windows with no new frames (drives the stale-stream warning)
    stalled = False  # whether we've surfaced the "no frames" state to the GUI (toggles on transitions)
    last_generation = None  # control `generation`; a bump = new stream (scrub/episode/wrap) -> reseed
    n_infer = 0  # inferences in the current ~1s window (for the latency average)
    compute_ms_sum = 0.0  # adapter.infer() time this window
    ipc_ms_sum = 0.0  # write_overlay (shm) time this window
    fps_last_emit = time.perf_counter()
    try:
        while not stop.is_set():
            t0 = time.perf_counter()
            control = overlay.read_control()
            if control:
                if "cameras" in control:
                    new_active = _resolve_active(control.get("cameras"), all_cams)
                    if new_active != active:
                        logger.info("camera filter -> %s (was %s)", sorted(new_active), sorted(active))
                        active = new_active
                # A new stream (data scrub / episode / wrap bumps `generation`) means a stateful step
                # must drop its memory and reseed; live never sets it (reattach handles restarts).
                gen = control.get("generation")
                if gen is not None and gen != last_generation:
                    last_generation = gen
                    for c in active:
                        adapter.set_camera(c)
                        adapter.reset()
                # The protocol owns `generation` / `cameras`; the rest is the step's opaque config.
                adapter.set_control(control.get("config", control))
            did_infer = False
            for cam in all_cams:
                if cam not in active:
                    continue
                # Event-driven: only infer when the frame actually advanced. A frozen
                # stream (teleop paused) must NOT re-infer the same frame and burn the
                # GPU — skip until the obs-stream sequence counter changes.
                seq = reader.image_seq(cam)
                if seq == last_seq.get(cam):
                    continue
                result = reader.read_image(cam)
                if result is None:
                    continue
                last_seq[cam] = seq
                frame, _ts = result
                try:
                    adapter.set_camera(cam)  # scope per-camera tracking state
                    tc = time.perf_counter()
                    rgba = adapter.infer(np.ascontiguousarray(frame))
                    compute_ms_sum += (time.perf_counter() - tc) * 1000.0
                    ti = time.perf_counter()
                    overlay.write_overlay(cam, rgba)
                    ipc_ms_sum += (time.perf_counter() - ti) * 1000.0
                    n_infer += 1
                    did_infer = True
                except Exception:
                    logger.exception("inference failed for camera %s", cam)
            if did_infer:
                infer_loops += 1
            # Clear overlays for cameras just switched off so a stale mask doesn't linger.
            for cam in last_active - active:
                h, w = dims[cam]
                last_seq.pop(cam, None)
                with contextlib.suppress(Exception):
                    overlay.write_overlay(cam, np.zeros((h, w, 4), dtype=np.uint8))
            last_active = set(active)
            dt = time.perf_counter() - t0
            if throttle > dt:
                time.sleep(throttle - dt)
            # Publish the actual INFERENCE rate (0 while idle/gated) + VRAM, ~1 Hz, and log
            # obs-stream activity so a frozen / stale / wrong stream is diagnosable from the
            # standalone log (which camera seqs are advancing vs stuck).
            now = time.perf_counter()
            if now - fps_last_emit >= 1.0:
                fps = infer_loops / (now - fps_last_emit)
                overlay.write_fps(fps)  # -> GUI meta block
                overlay.write_latency(
                    {
                        "compute_ms": round(compute_ms_sum / n_infer, 1) if n_infer else 0.0,
                        "ipc_ms": round(ipc_ms_sum / n_infer, 2) if n_infer else 0.0,
                        "n": n_infer,
                    }
                )
                seqs = {c: reader.image_seq(c) for c in sorted(active)}
                if infer_loops:
                    idle_secs = 0
                    logger.info(
                        "live: %.1f infer/s · compute %.1fms · ipc %.2fms · active=%s · seqs=%s",
                        fps,
                        compute_ms_sum / max(1, n_infer),
                        ipc_ms_sum / max(1, n_infer),
                        sorted(active),
                        seqs,
                    )
                    if stalled:  # frames came back — clear the warning state
                        _set_overlay(f"{adapter.label} — live", "#39d353")
                        stalled = False
                else:
                    idle_secs += 1
                    logger.warning(
                        "live: no new frames for %ds · active=%s · seqs=%s — obs stream not advancing "
                        "(publisher paused, stopped, or a stale stream)",
                        idle_secs,
                        sorted(active),
                        seqs,
                    )
                    # Don't keep flashing a green "live" badge over an idle feed — but don't cry
                    # error either: we can't tell paused from stale yet (see umbrella-registry TODO).
                    # Show a neutral 'idle' state.
                    if idle_secs >= 3 and not stalled:
                        _set_overlay(f"{adapter.label} — idle (no input frames)", "#c9a94a")
                        stalled = True
                    # The publisher may have RESTARTED (teleop stop+start), leaving us mapped
                    # to the dead segment forever. Probe for a fresh one and resume on it; the
                    # next productive window flips the badge back to live.
                    if idle_secs >= 3 and idle_secs % 3 == 0:
                        fresh = _try_reattach(reader, all_cams)
                        if fresh is not None:
                            with contextlib.suppress(Exception):
                                reader.close()
                            reader = fresh
                            last_seq.clear()  # the new segment's low seqs count as new frames
                            idle_secs = 0
                model_gb = _vram_gb()
                if model_gb:
                    overlay.write_vram(model_gb)
                status.write("active", fps, model_gb)  # the live badge's single source of truth
                infer_loops = 0
                n_infer = 0
                compute_ms_sum = 0.0
                ipc_ms_sum = 0.0
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
