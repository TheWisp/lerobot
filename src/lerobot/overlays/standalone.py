"""Standalone debug-vision process — runs a representation model alongside teleop.

Reads live camera frames from the always-on ObservationStream (any process can
attach read-only), runs a DebugVisionAdapter, and publishes a per-camera RGBA
overlay to a SharedOverlayBuffer that the GUI backend serves as PNG.

Usage:
    python -m lerobot.overlays.standalone \
        --model sam3_track --prompt "robot arm . cylinder . green ring" \
        --cameras observation.images.top
"""

import argparse
import atexit
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from lerobot.overlays.adapters import ADAPTERS, ConceptMaskAdapter, _concept_color, build_adapter
from lerobot.overlays.overlay_ipc import OverlayStatus, SharedOverlayBuffer
from lerobot.robots.obs_stream import _SHM_DIR, SHM_PREFIX, ObservationStreamReader

logger = logging.getLogger(__name__)


def _vram_gb() -> float:
    """The model's live tensor allocations in GB (``torch.cuda.memory_allocated``) — flat once
    warmed, climbs on a leak; that's the signal this badge is for. NOT the process's full GPU
    footprint: the CUDA context + allocator cache are excluded, so ``nvidia-smi`` reads ~1-1.5 GB
    higher (measured 3.7 vs 5.0 on the SAM3 worker). 0.0 without CUDA."""
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    return 0.0


def _set_overlay(text: str, color: str = "#39d353") -> None:
    """Emit the GUI text-overlay protocol (parsed by run.py _append_output)."""
    print(f"##OVERLAY:{text}:{color}##", flush=True)


def _build_identity() -> str:
    """WHICH code this process is actually running: the loaded lerobot path + git SHA.

    The antidote to 'stale deployment' — a stale long-lived worker, or the wrong checkout on
    PYTHONPATH (this repo vs the conda-editable one). Logged at startup so one grep answers
    'am I running the fix I just committed?' instead of post-hoc log forensics."""
    import lerobot

    path = lerobot.__file__
    sha = "no-git"
    with contextlib.suppress(Exception):
        root = os.path.dirname(path)
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if subprocess.call(["git", "-C", root, "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0:
            sha += "-dirty"
    return f"lerobot @ {path} | git {sha}"


def _wait_for_obs_stream(stop: threading.Event) -> ObservationStreamReader | None:
    logger.info("waiting for observation stream (start teleop to begin)...")
    while not stop.is_set():
        try:
            return ObservationStreamReader()
        except (FileNotFoundError, RuntimeError):
            time.sleep(0.5)
    return None


def _stream_identity() -> int | None:
    """The obs stream's IDENTITY = the inode of its meta segment.

    A run start unlinks + recreates the segment under the same name (fresh inode); a
    pause keeps the same inode; a stopped+swept publisher leaves none. So the inode lets
    a consumer tell "I'm holding a *replaced* (orphaned) stream" from "same stream, just
    paused" — the distinction the seq counter alone couldn't make — without any
    producer-side token. Returns None when the segment is absent.

    Inode reuse can't bite us here: while we still map the old segment the kernel keeps
    that inode pinned, so a freshly created segment necessarily gets a different one.
    """
    with contextlib.suppress(OSError):
        return os.stat(os.path.join(_SHM_DIR, f"{SHM_PREFIX}meta")).st_ino
    return None


def _reader_inode(reader: ObservationStreamReader) -> int | None:
    """The inode of the segment `reader` is ACTUALLY bound to, via its own fd.

    Race-free, unlike statting the path: the path resolves to whatever segment currently
    owns the name, which can differ from what the reader mapped (a restart between attach
    and a path-stat, or between a stat and an open). fstat-ing the reader's live fd always
    names the segment it holds, so `held_ino` tracks our binding rather than a coincidental
    path value — closing the startup-race and TOCTOU windows. Falls back to the path stat
    if the reader internals aren't reachable (degrades to path-stat behavior, never thrashes).
    """
    with contextlib.suppress(AttributeError, OSError, ValueError):
        return os.fstat(reader._meta._shm._fd).st_ino
    return _stream_identity()


def _try_reattach(
    expected_cams: list[str], held_ino: int | None
) -> tuple[ObservationStreamReader, int | None] | None:
    """Re-attach iff the obs stream's backing segment was REPLACED.

    teleop/policy stop+start unlinks + recreates a fresh segment under the same name; the
    existing reader still maps the dead orphan and never sees frames again. Compare the
    segment the NAME currently resolves to (`_stream_identity()`) against the one we're
    actually bound to (`held_ino`, from our reader's fd): a *different* inode means a
    genuinely new stream (swap to it); the *same* inode means the same segment — paused or
    stopped, do NOT swap; a missing one (either side) means we can't act. Identity-based, so
    it holds even when the new run's seq has already passed the stale one's. Returns
    (new_reader, new_ino) — new_ino read from the new reader's own fd, not a pre-open stat —
    on a real replacement, else None.
    """
    cur_ino = _stream_identity()
    if cur_ino is None or held_ino is None or cur_ino == held_ino:
        return None
    try:
        new = ObservationStreamReader()
    except (FileNotFoundError, RuntimeError):
        return None
    if set(new.image_keys) != set(expected_cams):
        new.close()  # different stream shape — don't silently swap (restart the overlay instead)
        return None
    new_ino = _reader_inode(new)  # the inode the NEW reader actually bound to
    logger.info("obs stream segment replaced (inode %s -> %s): re-attaching", held_ino, new_ino)
    return new, new_ino


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


# ── Detection chrome (data-tab WYSIWYG mode) ──────────────────────────────────
# A live-tile-only affordance so the user can verify segmentation: a thin accent
# glow outline + a tiny label on each detected object. It is UI chrome — never a
# fill, never part of any written dataset — so it cannot be mistaken for a treatment
# (that was the old confusion, when objects were colour-FILLED). One accent colour;
# the labels disambiguate objects. Precedent: SAM demos glow the mask.
_CHROME_ACCENT = (79, 195, 247)  # RGB — the panel accent (#4fc3f7)
_IDLE_POLL_S = 0.003  # idle obs-stream seq poll (~3 ms) — low scrub-pickup latency, cheap shm reads


def _obj_treatments(objs) -> dict:
    """``{object_name: {key, params}}`` from an objects list (or its JSON string)."""
    if isinstance(objs, str):
        try:
            objs = json.loads(objs)
        except Exception:
            return {}
    out: dict = {}
    for o in objs or []:
        name = str(o.get("name", "")).strip()
        if name:
            out[name] = o.get("treatment") or {"key": "none"}
    return out


_LABEL_FONT_CACHE: dict = {}


def _label_font(size: int):
    """A cached bold TrueType font at ``size`` (antialiased — far more legible than
    cv2's Hershey bitmap fonts, which is what made the labels unreadable)."""
    f = _LABEL_FONT_CACHE.get(size)
    if f is None:
        from PIL import ImageFont

        for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
            try:
                f = ImageFont.truetype(name, size)
                break
            except Exception:
                f = None
        if f is None:
            f = ImageFont.load_default()
        _LABEL_FONT_CACHE[size] = f
    return f


def _draw_labels(rgb: np.ndarray, labels: list, accent) -> np.ndarray:
    """Draw each ``(text, (x, y_top))`` as an antialiased label pill sitting just above
    the object's top edge. Font size scales with the frame so it stays readable after the
    tile downscales it. Returns a new RGB array."""
    if not labels:
        return rgb
    from PIL import Image, ImageDraw

    size = max(14, int(rgb.shape[0] * 0.032))  # ~23 px at 720p
    font = _label_font(size)
    pad = max(2, size // 5)
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil)
    fill = tuple(int(c) for c in accent)
    for text, (x, y_top) in labels:
        ty = max(pad, int(y_top) - size - 2 * pad)  # pill above the object
        tx = int(x)
        x0, y0, x1, y1 = d.textbbox((tx, ty), text, font=font)
        d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=(18, 22, 28))
        d.text((tx, ty), text, font=font, fill=fill)
    return np.asarray(pil)


def _draw_detection_chrome(rgb: np.ndarray, masks_by_name: dict) -> np.ndarray:
    """Return a COPY of ``rgb`` with a soft glow outline + a tiny label on each detected
    object, in that concept's STABLE auto-assigned color (never user-chosen — the color
    carries identity only, so it cannot be mistaken for a committed treatment).
    Display-only chrome — NEVER committed. RGB in, RGB out."""
    import cv2

    out = rgb.copy()
    glow = np.zeros_like(out)
    h = rgb.shape[0]
    outline_w = max(1, h // 360)  # scale line weight with resolution so it's visible
    glow_w = max(4, h // 110)
    concepts = list(masks_by_name)
    per_obj: list[tuple[str, tuple, list]] = []  # (name, color, contours)
    for name, mask in masks_by_name.items():
        m = mask.astype(np.uint8) if mask is not None else None
        if m is None or not m.any():
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        per_obj.append((name, _concept_color(name, concepts), contours))
        cv2.drawContours(glow, contours, -1, _concept_color(name, concepts), thickness=glow_w)
    if per_obj:
        glow = cv2.GaussianBlur(glow, (0, 0), max(2, glow_w // 2))
        out = cv2.addWeighted(out, 1.0, glow, 0.75, 0)
        for _name, color, contours in per_obj:  # crisp outline over the glow
            cv2.drawContours(out, contours, -1, color, thickness=outline_w, lineType=cv2.LINE_AA)
        for name, color, contours in per_obj:
            x, y, _bw, _bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            out = _draw_labels(out, [(name, (x, y))], color)
    return out


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
        help='Initial monitored objects, JSON [{"name","sign":"+","treatment":{...}}]',
    )
    parser.add_argument(
        "--background-treatment",
        default=None,
        help='Data-editing background treatment JSON {"key","params"} — its presence renders the '
        "per-region WYSIWYG composite + detection chrome instead of debug contours",
    )
    parser.add_argument("--style", default=None, help="Initial render style (policy_saliency)")
    parser.add_argument(
        "--method",
        default=None,
        help="Initial saliency method (gradient|rollout) — seeded into the control block for the POLICY",
    )
    parser.add_argument(
        "--smooth", type=float, default=None, help="Initial smoothing sigma (policy_saliency)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="SAM inference resolution preset (load-time; see ConceptMaskAdapter.RESOLUTIONS). "
        "None = the adapter default. Ignored by non-segmenter models.",
    )
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
    crashed = False
    try:
        adapter = build_adapter(args.model, device=args.device, resolution=args.resolution)
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
    # Stamp the build AFTER the re-assert: transformers clears the root handlers during model load, so
    # a stamp logged before it is silently dropped — which is exactly what swallowed it the first time.
    logger.info("BUILD: %s | pid %d", _build_identity(), os.getpid())

    init_control: dict = {}
    if args.objects:
        try:
            init_control["objects"] = json.loads(args.objects)
        except Exception:
            logger.warning("ignoring malformed --objects: %s", args.objects)
    elif args.prompt:
        init_control["prompt"] = args.prompt
    if args.style:
        init_control["style"] = args.style
    if args.smooth is not None:
        init_control["smooth"] = args.smooth
    if init_control:
        adapter.set_control(init_control)
    logger.info("model '%s' ready", args.model)
    status.write("active", 0.0, _vram_gb())  # loaded; the GUI badge goes active (fps 0 until frames flow)

    reader = _wait_for_obs_stream(stop)
    if reader is None:
        return
    held_ino = _reader_inode(reader)  # the segment we're bound to (from our fd); change == restart
    logger.info("attached to obs stream (inode %s); cameras available: %s", held_ino, list(reader.image_keys))

    all_cams = list(reader.image_keys)
    # The overlay buffer covers ALL cameras so the active filter can change live (via
    # the `cameras` control) without recreating shared memory. Disabled cameras are
    # skipped in the loop — no inference, so the filter doubles as a compute dial.
    dims = {c: (int(reader.image_keys[c][0]), int(reader.image_keys[c][1])) for c in all_cams}
    active = _resolve_active(args.cameras, all_cams)
    logger.info("cameras=%s active=%s", all_cams, sorted(active))

    overlay = SharedOverlayBuffer(cameras=dims, model=args.model, create=True)
    if args.method:
        # Seed the POLICY-read method into the control block the moment it exists — the GUI can't
        # (its write would race a segment that appears only now); a later GUI control write
        # replaces the whole config including method, so this is only the initial value.
        overlay.write_control({"config": {"method": args.method}})
    # Clear every camera up front so a reused shm segment can't surface a
    # previous run's stale overlays on cameras this run won't touch.
    for _cam, (_h, _w) in dims.items():
        overlay.write_overlay(_cam, np.zeros((_h, _w, 4), dtype=np.uint8))
    _set_overlay(f"{adapter.label} — live", "#39d353")

    # Data-editing WYSIWYG mode: when the config carries a ``background_treatment``,
    # the data tab renders the per-region composite (each object + the background
    # carry a treatment) PLUS the always-on detection chrome, instead of the debug
    # contour overlay. Randomized treatments are sampled once per (camera, generation)
    # so the look is stable within an episode — the same per-episode rule the batch
    # pass uses, so what you preview is what gets committed (chrome excluded).
    from lerobot.overlays.effects import build_and_sample_regions, composite_regions

    bg_treatment: dict | None = None
    if args.background_treatment:  # seed at spawn so the first inference already composites
        try:
            bg_treatment = json.loads(args.background_treatment)
        except Exception:
            logger.warning("ignoring malformed --background-treatment: %s", args.background_treatment)
    obj_treatments: dict = _obj_treatments(args.objects)
    region_caches: dict = {}  # (cam, generation) -> per-region sampled cache
    treat_rng = np.random.default_rng(0)

    throttle = max(0.0, args.throttle_ms / 1000.0)
    last_active = set(active)
    last_seq: dict[str, int] = {}  # per-camera obs-stream seq — gate inference on new frames
    infer_loops = 0  # productive iterations since the last emit (0 while the stream is idle)
    idle_secs = 0  # consecutive ~1s windows with no new frames (drives the stale-stream warning)
    stalled = False  # whether we've surfaced the "no frames" state to the GUI (toggles on transitions)
    last_generation = None  # control `generation`; a bump = new stream (scrub/episode/wrap) -> reseed
    rgba_bufs: dict[str, np.ndarray] = {}  # per-camera reusable RGBA output (data/WYSIWYG mode)
    n_infer = 0  # inferences in the current ~1s window (for the latency average)
    compute_ms_sum = 0.0  # full per-camera pass (model + effects) this window
    seg_ms_sum = 0.0  # model part only (adapter.segment/infer) — compute minus seg = effects/compose
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
                    region_caches.clear()  # new episode -> redraw randomized treatments
                    for c in active:
                        adapter.set_camera(c)
                        adapter.reset()
                # The protocol owns `generation` / `cameras`; the rest is the step's opaque config.
                cfg = control.get("config", control)
                adapter.set_control(cfg)
                if isinstance(cfg, dict):
                    # A treatment change (bg or per-object) must re-render the parked frame,
                    # so clear last_seq to let the inference gate fire without a scrub.
                    changed = False
                    if "background_treatment" in cfg and cfg["background_treatment"] != bg_treatment:
                        bg_treatment = cfg["background_treatment"]
                        changed = True
                    if "objects" in cfg:
                        new_objt = _obj_treatments(cfg["objects"])
                        if new_objt != obj_treatments:
                            obj_treatments = new_objt
                            changed = True
                    if changed:
                        last_seq.clear()
            did_infer = False
            # Gather every active camera whose frame ADVANCED this sweep. A frozen
            # stream (teleop paused) must NOT re-infer the same frame and burn the
            # GPU — the seq gate skips it until the obs-stream counter changes.
            frames_by_cam: dict[str, np.ndarray] = {}
            for cam in all_cams:
                if cam not in active:
                    continue
                seq = reader.image_seq(cam)
                if seq == last_seq.get(cam):
                    continue
                result = reader.read_image(cam)
                if result is None:
                    continue
                last_seq[cam] = seq
                frames_by_cam[cam] = np.ascontiguousarray(result[0])
            if frames_by_cam and isinstance(adapter, ConceptMaskAdapter):
                # WYSIWYG data mode: ONE segmentation pass for the whole sweep — the
                # adapter shares the vision encode across cameras when batching is on
                # — then the
                # per-region composite per camera. The composite is the COMMITTED
                # result; the detection chrome is drawn on top for the LIVE tile only.
                try:
                    tc = time.perf_counter()
                    masks_by_cam = adapter.segment_many(frames_by_cam)
                    seg_total_ms = (time.perf_counter() - tc) * 1000.0
                    seg_ms_sum += seg_total_ms
                    compute_ms_sum += seg_total_ms  # fx per camera is added below
                    for cam, frgb in frames_by_cam.items():
                        tfx = time.perf_counter()
                        h, w = frgb.shape[:2]
                        masks_by_name = masks_by_cam[cam]
                        cache = region_caches.setdefault((cam, last_generation), {})
                        regions, sampled = build_and_sample_regions(
                            masks_by_name, obj_treatments, bg_treatment, h, w, treat_rng, cache
                        )
                        composed = composite_regions(frgb, regions, sampled)
                        display = _draw_detection_chrome(composed, masks_by_name)
                        # TRANSPARENT-DIFF overlay: opaque only where a treatment painted
                        # (union of treated regions' feathered alphas) or chrome drew
                        # (pixels the chrome pass changed). Untreated areas stay
                        # transparent, so the run tab's live feed shows through at full
                        # rate and the data tab's PNGs shrink to the changed pixels.
                        buf = rgba_bufs.get(cam)
                        if buf is None or buf.shape[:2] != (h, w):
                            buf = np.empty((h, w, 4), dtype=np.uint8)
                            rgba_bufs[cam] = buf
                        alpha = np.zeros((h, w), dtype=np.float32)
                        for (ralpha, treatment), _samp in zip(regions, sampled, strict=True):
                            if ((treatment or {}).get("key") or "none") != "none":
                                np.maximum(alpha, ralpha, out=alpha)
                        chrome_px = (display != composed).any(axis=2)
                        buf[..., :3] = display
                        buf[..., 3] = np.maximum(alpha * 255.0, chrome_px * 255).astype(np.uint8)
                        rgba = buf
                        compute_ms_sum += (time.perf_counter() - tfx) * 1000.0
                        ti = time.perf_counter()
                        overlay.write_overlay(cam, rgba)
                        ipc_ms_sum += (time.perf_counter() - ti) * 1000.0
                        n_infer += 1
                        did_infer = True
                except Exception:
                    logger.exception("inference failed for cameras %s", sorted(frames_by_cam))
            else:
                for cam, frgb in frames_by_cam.items():
                    try:
                        adapter.set_camera(cam)  # scope per-camera tracking state
                        tc = time.perf_counter()
                        rgba = adapter.infer(frgb)  # run-tab debug contour / saliency overlay
                        seg_ms_sum += (time.perf_counter() - tc) * 1000.0  # no separate effects stage
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
            # Pace the loop. When we inferred, honour the throttle as a max-rate cap (it
            # rarely binds — SAM > throttle). When IDLE (no new frame), poll the obs-stream
            # seq at a much finer cadence so a scrubbed frame is picked up in a few ms, not
            # up to `throttle`. The seq gate above already blocks redundant inference, so the
            # fast idle poll is just cheap shm-header reads, not extra GPU work or a busy spin.
            if did_infer:
                if throttle > dt:
                    time.sleep(throttle - dt)
            else:
                time.sleep(_IDLE_POLL_S)
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
                        "seg_ms": round(seg_ms_sum / n_infer, 1) if n_infer else 0.0,
                        "ipc_ms": round(ipc_ms_sum / n_infer, 2) if n_infer else 0.0,
                        "n": n_infer,
                    }
                )
                seqs = {c: reader.image_seq(c) for c in sorted(active)}
                if infer_loops:
                    idle_secs = 0
                    logger.info(
                        "live: %.1f infer/s · compute %.1fms (seg %.1f + fx %.1f) · ipc %.2fms · active=%s · seqs=%s",
                        fps,
                        compute_ms_sum / max(1, n_infer),
                        seg_ms_sum / max(1, n_infer),
                        (compute_ms_sum - seg_ms_sum) / max(1, n_infer),
                        ipc_ms_sum / max(1, n_infer),
                        sorted(active),
                        seqs,
                    )
                    if stalled:  # frames came back — clear the warning state
                        _set_overlay(f"{adapter.label} — live", "#39d353")
                        stalled = False
                else:
                    idle_secs += 1
                    # Re-bind FIRST: a restarted run unlinks+recreates the segment under the
                    # same name (fresh inode), leaving us mapped to the dead orphan. The inode
                    # check is definitive, so probe every idle second and swap to the live one.
                    swapped = _try_reattach(all_cams, held_ino)
                    if swapped is not None:
                        # Segment was replaced — bind the live one. _try_reattach logs the swap;
                        # the badge is left to the next productive window (the `if stalled` path
                        # flips it back to live once frames actually flow on the new segment).
                        new_reader, held_ino = swapped
                        with contextlib.suppress(Exception):
                            reader.close()
                        reader = new_reader
                        last_seq.clear()  # the new segment's low seqs count as new frames
                        idle_secs = 0
                        # A fresh segment is a new run/scene: drop stateful adapter memory (e.g.
                        # sam3_track's tracking) so it re-seeds against the new stream, exactly as
                        # a `generation` bump resets it on a data-tab discontinuity.
                        for c in active:
                            adapter.set_camera(c)
                            adapter.reset()
                    else:
                        # No live segment to swap to. Name the reason instead of the old catch-all:
                        # the segment is gone (publisher exited + swept) vs frozen in place (paused,
                        # or the writer died without cleanup — same orphan either way).
                        gone = _stream_identity() is None
                        logger.warning(
                            "live: no new frames for %ds · active=%s · seqs=%s — %s",
                            idle_secs,
                            sorted(active),
                            seqs,
                            "publisher gone (segment removed)"
                            if gone
                            else "stream not advancing (paused or stalled)",
                        )
                        if idle_secs >= 3 and not stalled:
                            if gone:
                                _set_overlay(f"{adapter.label} — stale (publisher gone)", "#d29922")
                            else:
                                _set_overlay(f"{adapter.label} — idle (no input frames)", "#c9a94a")
                            stalled = True
                model_gb = _vram_gb()
                if model_gb:
                    overlay.write_vram(model_gb)
                status.write("active", fps, model_gb)  # the live badge's single source of truth
                infer_loops = 0
                n_infer = 0
                compute_ms_sum = 0.0
                seg_ms_sum = 0.0
                ipc_ms_sum = 0.0
                fps_last_emit = now
    except Exception:
        logger.exception("debug-vision crashed")
        # Exit non-zero: a crash that exits 0 reads as a clean self-stop to the GUI's
        # process observer, which would show "inactive" instead of the error badge.
        crashed = True
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            overlay.cleanup()
        with contextlib.suppress(Exception):
            reader.close()
        logger.info("debug-vision shutdown complete")
    if crashed:
        sys.exit(1)


if __name__ == "__main__":
    main()
