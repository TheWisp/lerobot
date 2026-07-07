"""Debug-vision model adapters: frame (HxWx3 RGB uint8) -> overlay (HxWx4 RGBA uint8).

Each adapter loads ONE representation model and renders a frame-sized RGBA
overlay (transparent where nothing is drawn). Weights are public Hugging Face
checkpoints, fetched into the standard HF cache on first load — nothing is
bundled or pulled from a temp location, so a fresh environment reproduces it.

Adding a model = one subclass that declares its id + key + control needs and
implements infer(). The GUI discovers controls from the class attributes.
"""

from __future__ import annotations

import colorsys
import contextlib
import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

_IMPORT_HINT = (
    "transformers is required for debug-vision models. Install with "
    "`uv sync --extra debug-vision` (or `pip install transformers==5.3.0`)."
)


def _color_for(label: str) -> tuple[int, int, int]:
    """Stable RGB color per label string."""
    h = int(hashlib.md5(label.encode(), usedforsecurity=False).hexdigest(), 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


# Distinct, legible colors assigned per concept in prompt order (hash fallback for
# extras), so each concept keeps a stable color across frames — color alone tells
# the masks apart, no on-image labels needed.
_CONCEPT_PALETTE = [
    (239, 68, 68),
    (34, 197, 94),
    (59, 130, 246),
    (234, 179, 8),
    (168, 85, 247),
    (20, 184, 166),
    (249, 115, 22),
    (236, 72, 153),
]


def _parse_objects(control: dict, max_objects: int):
    """Pull monitored objects from a control dict (the universal concept selector).

    Returns ``(names, colors, signs)`` — ``names`` deduped and ``<= max_objects``;
    ``colors`` maps name -> (r, g, b) (omitted when transparent/unset → palette
    fallback); ``signs`` maps name -> "+"/"-" (default "+", "-" = exclude). Returns
    ``(None, None, None)`` when there are no usable objects so the caller keeps state.
    """
    objs = control.get("objects")
    if not isinstance(objs, list) or not any(str(o.get("name", "")).strip() for o in objs):
        return None, None, None
    names: list[str] = []
    colors: dict[str, tuple[int, int, int]] = {}
    signs: dict[str, str] = {}
    for o in objs[:max_objects]:
        name = str(o.get("name", "")).strip()
        if not name:
            continue
        names.append(name)
        c = o.get("color")
        if isinstance(c, (list, tuple)) and len(c) == 3:
            colors[name] = (int(c[0]), int(c[1]), int(c[2]))
        signs[name] = "-" if o.get("sign") == "-" else "+"
    return list(dict.fromkeys(names)), colors, signs


def _concept_color(concept, concepts, colors):
    """Stable color for a concept: user-chosen if set, else palette by position, else hashed."""
    if concept in colors:
        return colors[concept]
    if concept in concepts:
        return _CONCEPT_PALETTE[concepts.index(concept) % len(_CONCEPT_PALETTE)]
    return _color_for(concept)


_BG_UNSET = object()  # sentinel: this control dict didn't mention the background


def _parse_background(control: dict):
    """Background fill color from a control dict.

    Returns ``(r, g, b)`` to fill the inverse region, ``None`` for transparent
    (don't paint), or ``_BG_UNSET`` when the key is absent (keep current state).
    """
    if "background" not in control:
        return _BG_UNSET
    bg = control.get("background") or {}
    c = bg.get("color") if isinstance(bg, dict) else None
    if isinstance(c, (list, tuple)) and len(c) == 3:
        return (int(c[0]), int(c[1]), int(c[2]))
    return None  # transparent


def _composite_concepts(h, w, masks_by_concept, concepts, colors, signs, bg_color, cv2, fill_alpha=130):
    """Paint an RGBA overlay from per-concept boolean masks.

    ``+`` concepts are drawn in their color; ``-`` concepts are carved out of the
    positive masks (and excluded from the detected region). ``bg_color`` (when not
    None) fills everything NOT covered by a positive detection — the inverse region.
    Background is painted first so positive fills + contours sit on top.
    """
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Fast path: with no negative concepts and a transparent background (the common
    # case) we skip the per-mask full-frame ops (carve + detected-union) entirely —
    # they cost ~2 HxW boolean passes PER mask, which dominates when a concept like
    # "object" returns dozens of instances.
    has_neg = any(signs.get(c, "+") == "-" for c in concepts)
    need_detected = bg_color is not None
    neg = np.zeros((h, w), dtype=bool)
    if has_neg:
        for c in concepts:
            if signs.get(c, "+") == "-":
                for m in masks_by_concept.get(c, []):
                    neg |= m
    detected = np.zeros((h, w), dtype=bool) if need_detected else None
    draw: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for c in concepts:
        if signs.get(c, "+") == "-":
            continue
        col = _concept_color(c, concepts, colors)
        for m in masks_by_concept.get(c, []):
            mm = (m & ~neg) if has_neg else m
            if has_neg and not mm.any():
                continue
            draw.append((mm, col))
            if need_detected:
                detected |= mm
    if need_detected:
        rgba[~detected] = (*bg_color, fill_alpha)
    for mm, col in draw:
        rgba[mm] = (*col, fill_alpha)
        cnts, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgba, cnts, -1, (*col, 255), 2)
    return rgba


class DebugVisionAdapter:
    """Base adapter. Subclasses load a model and render an RGBA overlay.

    Class attributes describe the model to the GUI:
      key       — stable identifier used in the API / dropdown
      label     — human-readable name
      controls  — list of control specs the GUI should render, e.g.
                  [{"type": "text", "key": "prompt", "label": "Prompt"}]
    """

    key = "base"
    label = "base"
    controls: list[dict] = []

    def __init__(self, device: str = "cuda"):
        self.device = device

    def set_control(self, control: dict) -> None:
        """Apply a control update (prompt text, thresholds, ...). Idempotent."""

    def set_camera(self, cam: str | None) -> None:
        """Tell the adapter which camera the next infer() frame comes from. No-op for
        stateless adapters; stateful ones (video tracking) override to scope per-camera
        state so multiple views don't share one temporal memory."""

    def reset(self) -> None:
        """Drop temporal memory for the current camera (set_camera) so the next infer()
        starts fresh. The scheduler fires this on a DISCONTINUITY — first frame, scrub
        jump, episode switch, playback wrap. No-op for stateless adapters; stateful
        trackers override to clear that camera's session."""

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return an HxWx4 RGBA uint8 overlay sized to frame_rgb. Precondition:
        frame_rgb is contiguous HxWx3 uint8 RGB."""
        raise NotImplementedError


class Sam3TrackByDetectionAdapter(DebugVisionAdapter):
    """SAM3 LOCKED-OBJECT tracking (tracking-by-detection). Two tiers sharing one encoder:

    - Tier 1 — ``Sam3Model`` image detector: text -> mask. Used only to SEED a new object
      and to RE-DETECT one after it's lost (heavy occlusion).
    - Tier 2 — ``Sam3TrackerVideoModel`` geometric tracker: propagates each seeded object
      frame-to-frame purely from spatial-temporal memory (no per-frame re-detection), so the
      masks lock onto specific objects instead of the concept model's proliferating instances.

    Indefinite-stream memory is bounded by REBUILD, not editing: every ``FLUSH_EVERY`` frames
    (and on recovery) the tracker session is destroyed and reseeded from the current mask —
    flat GPU forever, and it never desyncs the memory bank (which is what pruning did). Each
    period-separated concept is locked to one instance in its own color.

    Architecture per a SAM3 maintainer; see SAM3_VIDEO_STREAMING_OOM.md. GATED weights —
    accept the Meta SAM License at https://huggingface.co/facebook/sam3 (+ ``hf auth login``).
    """

    key = "sam3_track"
    label = "SAM3"
    controls = [
        {
            "type": "text",
            "key": "prompt",
            "label": "Objects",
            "placeholder": "robot arm . cylinder . green ring",
            "hint": "Period-separated objects; each is detected once then locked + tracked in its "
            "own color (legend, top-left). Changing this restarts tracking.",
        }
    ]
    SAM3_ID = "facebook/sam3"
    DEFAULT_PROMPT = "object"
    MAX_OBJECTS = 6  # cap monitored objects (shared encoder keeps multi-object cheap)
    FLUSH_EVERY = 150  # rebuild each tracker session every N frames -> flat GPU memory
    LOST_THRESH = 0.30  # sigmoid(object_score_logits) below this = track lost -> Tier-1 recover
    RECOVER_EVERY = 5  # throttle Tier-1 re-detection attempts (frames) while an object is lost

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from PIL import Image
            from transformers import (
                Sam3Model,
                Sam3Processor,
                Sam3TrackerVideoModel,
                Sam3TrackerVideoProcessor,
            )
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        self._Image = Image
        logger.info("loading %s (detector + geometric tracker) ...", self.SAM3_ID)
        try:
            self.det_proc = Sam3Processor.from_pretrained(self.SAM3_ID)
            self.det = Sam3Model.from_pretrained(self.SAM3_ID, dtype=torch.float16).to(device).eval()
            self.trk_proc = Sam3TrackerVideoProcessor.from_pretrained(self.SAM3_ID)
            self.trk = (
                Sam3TrackerVideoModel.from_pretrained(self.SAM3_ID, dtype=torch.float16).to(device).eval()
            )
        except Exception as e:
            raise RuntimeError(
                f"SAM3 weights are gated — accept the Meta SAM License at "
                f"https://huggingface.co/{self.SAM3_ID} and run `hf auth login`, then reload. ({type(e).__name__})"
            ) from e
        # Do NOT share the encoder (trk.vision_encoder = det.vision_encoder). Despite the same
        # PE-ViT-L+ architecture, the detector's and tracker's encoder WEIGHTS differ — feeding
        # the tracker the detector's features silently corrupts tracking: it drifts off the
        # seeded object onto distractors while still reporting a high score. Measured on real
        # frames: with the share the ring track jumped to the gripper by frame ~14; without it,
        # it holds. The ~0.9 GB saved is not worth a broken tracker.
        self.prompt = self.DEFAULT_PROMPT
        self._concepts: list[str] = []
        self._colors: dict[str, tuple[int, int, int]] = {}  # user-chosen color per concept
        self._signs: dict[str, str] = {}
        self._bg_color: tuple[int, int, int] | None = None
        self._det_threshold = 0.5
        self._cam: str | None = None
        self._tracks: dict[str | None, dict] = {}  # per-camera tracker state (session + masks)

    def _parse_concepts(self) -> list[str]:
        parts = (c.strip() for c in self.prompt.replace(",", ".").split("."))
        names = list(dict.fromkeys(c for c in parts if c))[: self.MAX_OBJECTS]
        return names or [self.DEFAULT_PROMPT]

    def set_camera(self, cam: str | None) -> None:
        self._cam = cam  # which camera's tracker state infer() should use

    def reset(self) -> None:
        # Discontinuity: drop this camera's session so the next infer() re-seeds from
        # scratch instead of propagating a stale memory bank across a scrub/episode/wrap.
        self._tracks.pop(self._cam, None)

    def set_control(self, control: dict) -> None:
        # Structured monitored objects (preferred). Color/sign/background are display-only;
        # only an object-NAME change restarts tracking.
        names, colors, signs = _parse_objects(control, self.MAX_OBJECTS)
        if names is not None:
            self._colors = colors
            self._signs = signs
            new_prompt = " . ".join(names)
            if new_prompt and new_prompt != self.prompt:
                self.prompt = new_prompt
                self._tracks = {}  # restart tracking on every camera with the new objects
        else:
            p = control.get("prompt")
            if isinstance(p, str) and p.strip() and p.strip() != self.prompt:
                self.prompt = p.strip()
                self._tracks = {}
        bg = _parse_background(control)
        if bg is not _BG_UNSET:
            self._bg_color = bg

    # ---------------- Tier 1: image detector (text -> one mask per concept) ----------------
    def _detect(self, frame_rgb: np.ndarray, concept: str, h: int, w: int) -> np.ndarray | None:
        """Largest instance mask for ``concept`` on this single frame, or None."""
        torch = self._torch
        inp = self.det_proc(images=self._Image.fromarray(frame_rgb), text=concept, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            out = self.det(**inp)
        res = self.det_proc.post_process_instance_segmentation(
            out, threshold=self._det_threshold, target_sizes=[(h, w)]
        )[0]
        masks = res.get("masks", [])
        if len(masks) == 0:
            return None
        arrs = [(m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)) > 0 for m in masks]
        best = max(arrs, key=lambda a: int(a.sum()))
        assert best.shape == (h, w), f"detector mask {best.shape} != frame {(h, w)}"
        return best if int(best.sum()) > 50 else None

    # ---------------- Tier 2: geometric video tracker ----------------
    def _pv(self, frame_rgb: np.ndarray):
        inp = self.trk_proc(images=self._Image.fromarray(frame_rgb), return_tensors="pt")
        return inp["pixel_values"][0].to(self.device, self._torch.float16)

    def _seed(self, track: dict, seeds: dict[str, np.ndarray], pv, h: int, w: int) -> None:
        """REBUILD: drop the old session, init a fresh one, seed obj-per-concept from
        ``seeds`` (current masks), run frame 0. Rebuilding (never editing the memory bank)
        is what keeps GPU flat without desyncing the tracker's frame indices.

        A degenerate seed mask (passes the area gate but collapses at the tracker's low
        conditioning resolution) makes SAM3 reject the whole frame with "maskmem_features
        ... cannot be empty when not is_initial_conditioning_frame", which would kill every
        co-seeded object too. So seed all objects, and on a conditioning failure drop the
        smallest-area one and retry — one bad detection no longer takes the rest down."""
        torch = self._torch
        old = track.get("session")
        # Largest masks first so the smallest (most likely degenerate) is dropped first.
        items = sorted(seeds.items(), key=lambda kv: -int(np.asarray(kv[1]).sum()))
        while items:
            sess = self.trk_proc.init_video_session(
                video=None,
                inference_device=self.device,
                inference_state_device=self.device,
                dtype=torch.float16,
            )
            fidx = sess.add_new_frame(pv)
            objs = {}
            for i, (concept, mask) in enumerate(items, start=1):
                self.trk_proc.process_new_mask_for_video_frame(
                    inference_session=sess, frame_idx=fidx, obj_ids=[i], input_masks=mask.astype(np.uint8)
                )
                objs[concept] = i
            # process_new_mask_for_video_frame REPLACES the session's "new input" set each
            # call (instead of adding to it), so after seeding N objects only the LAST is
            # flagged. The tracker then conditions only that one and treats the rest as
            # already-tracked frames -> "maskmem_features ... cannot be empty". Re-flag every
            # seeded object so they ALL get conditioned on this initial frame. (This is the
            # real cause of multi-object / +- carving failing; the drop-retry below is now
            # only a fallback for a genuinely degenerate single mask.)
            with contextlib.suppress(Exception):
                sess.obj_with_new_inputs = type(sess.obj_with_new_inputs)(range(1, len(items) + 1))
            try:
                with torch.inference_mode():
                    out = self.trk(inference_session=sess, frame_idx=fidx)
            except Exception as e:
                del sess
                if len(items) > 1:
                    logger.warning(
                        "tracker seed failed (%s: %s); dropping smallest object %r and retrying",
                        type(e).__name__,
                        e,
                        items[-1][0],
                    )
                    items = items[:-1]
                    continue
                logger.warning(
                    "tracker seed failed for %r (%s: %s); no track this frame",
                    items[0][0],
                    type(e).__name__,
                    e,
                )
                track["session"] = None
                return
            track["session"], track["objs"], track["since_flush"] = sess, objs, 0
            self._read_output(track, out, h, w)
            if old is not None:
                del old  # free the previous session's memory bank
            return

    def _read_output(self, track: dict, out, h: int, w: int) -> None:
        """Update per-concept full-res mask + score from a tracker forward output."""
        torch = self._torch
        id_to_concept = {oid: c for c, oid in track["objs"].items()}
        track["masks"], track["scores"] = {}, {}
        ids = list(out.object_ids or [])
        if out.pred_masks is None or not ids:
            return
        pm = out.pred_masks
        pm = pm.reshape(pm.shape[0], 1, *pm.shape[-2:])  # -> (num_obj, 1, low_h, low_w)
        # post_process_masks wants a LIST of per-image mask batches -> upscale to frame size
        full = self.trk_proc.post_process_masks([pm], original_sizes=[(h, w)])[0]  # (num_obj, 1, H, W)
        logits = out.object_score_logits.reshape(-1) if out.object_score_logits is not None else None
        for k, oid in enumerate(ids):
            concept = id_to_concept.get(oid)
            if concept is None:
                continue
            fm = full[k]
            fm = fm.cpu().numpy() if hasattr(fm, "cpu") else np.asarray(fm)
            track["masks"][concept] = fm.squeeze().astype(bool)  # (H, W)
            track["scores"][concept] = float(torch.sigmoid(logits[k])) if logits is not None else 1.0

    def _live_masks(self, track: dict) -> dict[str, list[np.ndarray]]:
        """Per-concept mask list for compositing — only objects currently held (score ok)."""
        return {
            c: (
                [track["masks"][c]]
                if track["scores"].get(c, 0.0) >= self.LOST_THRESH and c in track["masks"]
                else []
            )
            for c in self._concepts
        }

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        assert frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3, (
            f"infer expects HxWx3 RGB, got {frame_rgb.shape}"
        )
        torch = self._torch
        h, w = frame_rgb.shape[:2]
        cam = self._cam
        self._concepts = self._parse_concepts()
        track = self._tracks.get(cam)
        if track is None or track.get("shape") != (h, w):
            track = {
                "session": None,
                "objs": {},
                "masks": {},
                "scores": {},
                "since_flush": 0,
                "since_recover": 0,
                "shape": (h, w),
            }
            self._tracks[cam] = track
        pv = self._pv(frame_rgb)

        if track["session"] is None:
            # No track yet — Tier 1 detects each object to seed Tier 2.
            seeds = {c: m for c in self._concepts if (m := self._detect(frame_rgb, c, h, w)) is not None}
            # Visibility: what the detector found vs missed on the seed frame, and what we
            # hand the tracker. Periodic (seed / rebuild / recover), not per-frame.
            missing = [c for c in self._concepts if c not in seeds]
            logger.info(
                "seed[%s]: detected %s%s",
                self._cam or "?",
                {c: int(np.asarray(m).sum()) for c, m in seeds.items()} or "nothing",
                f" · NOT detected {missing}" if missing else "",
            )
            if seeds:
                self._seed(track, seeds, pv, h, w)
        else:
            try:
                with torch.inference_mode():
                    out = self.trk(inference_session=track["session"], frame=pv)
            except Exception as e:
                # The tracker can degrade mid-stream (e.g. after a re-seed) and throw
                # "maskmem_features ... empty"; drop the session so the next frame
                # re-seeds rather than failing on every frame forever.
                logger.warning(
                    "tracker step failed (%s: %s); resetting session to re-seed", type(e).__name__, e
                )
                track["session"], track["masks"], track["scores"] = None, {}, {}
            else:
                self._read_output(track, out, h, w)
                track["since_flush"] += 1
                track["since_recover"] += 1
                lost = [c for c in self._concepts if track["scores"].get(c, 0.0) < self.LOST_THRESH]
                # Rebuild on the rolling-window flush, OR to recover a lost object (throttled).
                if track["since_flush"] >= self.FLUSH_EVERY or (
                    lost and track["since_recover"] >= self.RECOVER_EVERY
                ):
                    seeds = {}
                    for c in self._concepts:
                        if track["scores"].get(c, 0.0) >= self.LOST_THRESH and c in track["masks"]:
                            seeds[c] = track["masks"][c]  # healthy: reseed from current mask
                        elif (m := self._detect(frame_rgb, c, h, w)) is not None:
                            seeds[c] = m  # lost: Tier-1 re-detect
                    why = "flush" if track["since_flush"] >= self.FLUSH_EVERY else "recover"
                    logger.info(
                        "%s[%s]: lost %s · re-seeding %s", why, self._cam or "?", lost or "none", list(seeds)
                    )
                    track["since_recover"] = 0
                    if seeds:
                        self._seed(track, seeds, pv, h, w)

        masks_by_concept = self._live_masks(track)
        rgba = _composite_concepts(
            h,
            w,
            masks_by_concept,
            self._concepts,
            self._colors,
            self._signs,
            self._bg_color,
            self._cv2,
        )
        assert rgba.shape == (h, w, 4), f"overlay {rgba.shape} != frame {(h, w, 4)}"
        return rgba


def _import_cv2():
    try:
        import cv2

        return cv2
    except ImportError as e:  # opencv is a core dep, but fail loudly if absent
        raise RuntimeError("opencv (cv2) is required for debug-vision overlays") from e


def _blue_yellow_lut(cv2) -> np.ndarray:
    """A vivid blue→yellow 256×1×3 BGR LUT for ``cv2.applyColorMap``. CIVIDIS' low end is near-black
    navy (reads as 'dark', not blue); this stays a saturated blue→teal→yellow so cool regions read."""
    stops = [(0.0, (40, 90, 235)), (0.45, (40, 190, 200)), (1.0, (250, 230, 45))]  # RGB
    xs = np.array([s[0] for s in stops])
    t = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((256, 1, 3), np.uint8)
    for ch in range(3):  # cv2 LUTs are BGR
        lut[:, 0, 2 - ch] = np.interp(t, xs, [s[1][ch] for s in stops]).astype(np.uint8)
    return lut


class PolicySaliencyAdapter(DebugVisionAdapter):
    """Live attention map of the running policy, per camera — where the upcoming action DEPENDS on
    each camera's pixels (input-gradient by default, attention rollout as the routing lens).

    Unlike every other step this draws the POLICY's own internals, not a separate vision model's
    output. The policy process publishes the per-camera grid it already computed for the action it
    just took to a ``SharedAuxBuffer``; this adapter attaches read-only and colorizes the latest
    grid onto the camera tile. It runs no model of its own and never re-runs the policy.

    Run-path only: the data tab scrubs a dataset with no live policy, so there is no aux to read
    and it draws nothing (transparent). Attaches lazily — transparent until the policy starts
    publishing, and re-attaches after a policy restart. Render STYLES are switchable at RUNTIME via
    the overlay ``style`` control (``set_control``), so the look is A/B'd live without a restart.
    """

    key = "policy_saliency"
    label = "Attention map"
    MAX_ALPHA = 180  # peak-saliency opacity; the heatmap fades to transparent at low attention
    ALPHA_FLOOR = 0.25  # below this normalized saliency the tile stays clear (background not tinted)

    # name -> (colormap, alpha-mode, lo_pct, hi_pct). 'ramped'/'full' keep the COOL (blue) end visible;
    # 'gated' shows only hotspots (scene stays clear). The GUI 'style' select picks one live.
    STYLES = {
        "blue_yellow": (
            "vivid",
            "gated",
            50.0,
            99.0,
        ),  # vivid blue->yellow, hotspots only (scene stays clear)
        "blue_yellow_field": ("vivid", "ramped", 10.0, 99.5),  # the old full-field tint (kept for A/B)
        "cividis": ("cividis", "ramped", 10.0, 99.5),  # perceptually-uniform navy->yellow
        "spotlight": ("cividis", "gated", 50.0, 99.0),  # hotspots only, scene stays clear
        "heatmap": ("vivid", "full", 10.0, 99.5),  # full blue field + yellow hot, scene dimmed
        "inferno": ("inferno", "gated", 50.0, 99.0),  # the original golden glow
    }
    DEFAULT_STYLE = "blue_yellow"
    SMOOTH_SIGMA = 1.2  # grid-space gaussian — smooths the 64x64 blockiness on upscale (0 = off, raw)

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._cv2 = _import_cv2()
        self._cam: str | None = None
        self._aux = None  # SharedAuxBuffer reader, attached lazily once the policy publishes
        self._style = self.DEFAULT_STYLE
        self._smooth = float(self.SMOOTH_SIGMA)
        # Perceptually-monotonic colormaps so the brightest pixel IS the peak — TURBO/JET dip in
        # lightness at both ends, making low and high both read dark and the hotspot ambiguous.
        self._cmaps = {
            "vivid": _blue_yellow_lut(self._cv2),
            "cividis": getattr(self._cv2, "COLORMAP_CIVIDIS", self._cv2.COLORMAP_VIRIDIS),
            "inferno": getattr(self._cv2, "COLORMAP_INFERNO", self._cv2.COLORMAP_JET),
        }

    def _ensure_aux(self) -> None:
        if self._aux is not None:
            return
        try:
            from lerobot.overlays.aux_ipc import SharedAuxBuffer

            self._aux = SharedAuxBuffer(create=False)
        except FileNotFoundError:
            self._aux = None  # writer (policy) not up yet — retry next frame

    def set_camera(self, cam: str | None) -> None:
        self._cam = cam

    def reset(self) -> None:
        # Drop the reader so a restarted policy (new aux segment / camera set) reattaches cleanly.
        if self._aux is not None:
            with contextlib.suppress(Exception):
                self._aux.cleanup()
            self._aux = None

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        self._ensure_aux()
        # Throttled diagnostics (~1/s/cam): log WHICH branch makes the overlay transparent, so a
        # blank overlay is never a guess — no aux / read=None / grid<=0 / actually drawn.
        self._dbg_n = getattr(self, "_dbg_n", 0) + 1
        dbg = self._dbg_n % 30 == 1
        if self._aux is None or self._cam is None:
            if dbg:
                logger.info(
                    "[saliency-adapter] cam=%s: no aux reader / no live policy -> transparent", self._cam
                )
            return rgba  # no live policy (e.g. the data tab) -> draw nothing
        try:
            got = self._aux.read_saliency(self._cam)
        except Exception:
            self._aux = None  # stale segment (policy restarted) — reattach next frame
            if dbg:
                logger.info(
                    "[saliency-adapter] cam=%s: read_saliency raised -> reattach next frame", self._cam
                )
            return rgba
        if got is None:
            if dbg:
                logger.info(
                    "[saliency-adapter] cam=%s: read_saliency=None (no published grid for this cam) -> transparent",
                    self._cam,
                )
            return rgba
        grid, _ = got
        if not grid.size or float(grid.max()) <= 0.0 or not np.isfinite(grid).all():
            if dbg:
                logger.info(
                    "[saliency-adapter] cam=%s: grid empty/<=0/nonfinite (size=%d max=%s) -> transparent",
                    self._cam,
                    grid.size,
                    float(grid.max()) if grid.size else None,
                )
            return rgba
        if dbg:
            logger.info(
                "[saliency-adapter] cam=%s: DRAWN |grid|max=%.2e mean=%.2e",
                self._cam,
                float(grid.max()),
                float(grid.mean()),
            )
        return self._render(grid, w, h)

    def set_control(self, control: dict) -> None:
        """Pick the render style + smoothing at runtime. ``style`` must be a ``STYLES`` key; ``smooth``
        is the grid-space gaussian sigma (>=0, 0 = raw 64x64). Unknown/missing values leave the current
        setting unchanged (idempotent)."""
        control = control or {}
        style = control.get("style")
        if style in self.STYLES:
            self._style = style
        smooth = control.get("smooth")
        if smooth is not None:
            with contextlib.suppress(TypeError, ValueError):
                self._smooth = max(0.0, float(smooth))

    def _render(self, grid: np.ndarray, w: int, h: int) -> np.ndarray:
        cmap, mode, lo_pct, hi_pct = self.STYLES.get(self._style, self.STYLES[self.DEFAULT_STYLE])
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # The grid is a coarse 64x64 (area-pooled from the 224px pixel gradient); a plain upscale shows
        # that blockiness. Smooth in GRID space (so it scales with the upscale) + bicubic interpolate.
        g = grid.astype(np.float32)
        if self._smooth > 0:
            g = self._cv2.GaussianBlur(g, (0, 0), self._smooth)
        lo, hi = np.percentile(g, (lo_pct, hi_pct))
        n = np.clip((g - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        up = np.clip(self._cv2.resize(n, (w, h), interpolation=self._cv2.INTER_CUBIC), 0.0, 1.0)
        heat = self._cv2.applyColorMap((up * 255).astype(np.uint8), self._cmaps[cmap])  # BGR
        rgba[..., 0] = heat[..., 2]
        rgba[..., 1] = heat[..., 1]
        rgba[..., 2] = heat[..., 0]
        if mode == "gated":  # hotspots only — cool end fully transparent, scene clear
            a = up.copy()
            a[up < self.ALPHA_FLOOR] = 0.0
            a = a * (self.MAX_ALPHA / 255.0)
        elif mode == "ramped":  # cool end visible (floor alpha), hot strongest, scene still readable
            a = 0.30 + 0.50 * up
        else:  # "full" — blue field everywhere + yellow hot, scene dimmed
            a = np.maximum(0.42, 0.30 + 0.45 * up)
        rgba[..., 3] = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        return rgba


ADAPTERS: dict[str, type[DebugVisionAdapter]] = {
    Sam3TrackByDetectionAdapter.key: Sam3TrackByDetectionAdapter,
    PolicySaliencyAdapter.key: PolicySaliencyAdapter,
}


def build_adapter(key: str, device: str = "cuda") -> DebugVisionAdapter:
    if key not in ADAPTERS:
        raise ValueError(f"unknown debug-vision model '{key}'; have {list(ADAPTERS)}")
    return ADAPTERS[key](device=device)
