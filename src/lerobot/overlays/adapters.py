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


# The keys that make up one click/box gesture. Shared with gui.api.overlays, which rides the
# latest op along on every control write.
_CLICK_OP_KEYS = frozenset({"clicks", "boxes", "click_name", "clicks_remove", "clicks_rename", "click_seq"})


def _parse_objects(control: dict, max_objects: int):
    """Pull monitored objects from a control dict (the universal concept selector).

    Returns ``(names, signs)`` — ``names`` deduped and ``<= max_objects``; ``signs``
    maps name -> "+"/"-" (default "+", "-" = exclude). Returns ``(None, None)`` when
    there are no usable objects so the caller keeps state.
    """
    objs = control.get("objects")
    if not isinstance(objs, list) or not any(str(o.get("name", "")).strip() for o in objs):
        return None, None
    names: list[str] = []
    signs: dict[str, str] = {}
    for o in objs[:max_objects]:
        name = str(o.get("name", "")).strip()
        if not name:
            continue
        # Clicked objects carry a treatment and sign but never join the prompt — no text
        # finds them, which is why they were clicked.
        signs[name] = "-" if o.get("sign") == "-" else "+"
        if not o.get("clicked"):
            names.append(name)
    return list(dict.fromkeys(names)), signs


# Colour per concept NAME, for the life of the worker. Assigned on first sight and never
# reassigned, so it does not depend on anything that moves.
_COLOR_BY_CONCEPT: dict[str, tuple[int, int, int]] = {}


def _concept_color(concept: str) -> tuple[int, int, int]:
    """Stable colour for a concept, used by the detection chrome to tell objects apart.

    It used to be the concept's INDEX in the list passed in — and that list is the set
    currently VISIBLE, so removing one object's row, or an object briefly dropping out of
    view, renumbered everything after it and recoloured objects nobody had touched. Assign
    once, preferring a palette entry no live concept holds; hash past the palette.
    """
    if concept not in _COLOR_BY_CONCEPT:
        used = set(_COLOR_BY_CONCEPT.values())
        free = [c for c in _CONCEPT_PALETTE if c not in used]
        _COLOR_BY_CONCEPT[concept] = free[0] if free else _color_for(concept)
    return _COLOR_BY_CONCEPT[concept]


def _release_concept_color(concept: str) -> None:
    """Return a deleted concept's palette entry to the pool.

    Assign-and-never-free keeps colours stable, but the palette is 8 long and a session of
    adding and removing objects exhausts it — after which every new object falls back to a
    hash and they stop being reliably distinguishable. Only an explicit deletion frees; an
    object merely lost for a few frames must keep its colour, which is the point of the map.
    """
    _COLOR_BY_CONCEPT.pop(concept, None)


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


class DeviceFrame:
    """One frame, which may already live on the GPU.

    The tracker runs on EVERY frame and needs only normalised pixel values,
    which can be produced on-device. The detector runs rarely (seeding and
    recovery) and wants numpy. Holding both lazily lets the per-frame path skip
    the device->host->device round trip -- measured at ~15 ms of a ~49 ms frame,
    with the GPU idle at 6% while it happened -- without changing the detector
    or any caller that still passes numpy.

    Pre: exactly one of ``tensor`` (uint8 CHW on any device) or ``array``
    (uint8 HWC) is given. Post: ``shape`` is HWC in both cases, and
    :meth:`numpy` returns the HWC array, copying from the device at most once.
    """

    __slots__ = ("_array", "_tensor", "shape")

    def __init__(self, *, tensor=None, array=None):
        assert (tensor is None) != (array is None), "give exactly one representation"
        self._tensor = tensor
        self._array = array
        if tensor is not None:
            c, h, w = (int(x) for x in tensor.shape)
            assert c == 3, f"expected CHW with 3 channels, got {tuple(tensor.shape)}"
            self.shape = (h, w, c)
        else:
            self.shape = tuple(int(x) for x in array.shape)

    @property
    def tensor(self):
        """The device tensor, or None when this frame only exists as numpy."""
        return self._tensor

    def numpy(self) -> np.ndarray:
        if self._array is None:
            self._array = np.ascontiguousarray(self._tensor.permute(1, 2, 0).cpu().numpy())
        return self._array


def as_device_frame(frame) -> DeviceFrame:
    """Accept numpy HWC, a uint8 CHW tensor, or an existing DeviceFrame."""
    if isinstance(frame, DeviceFrame):
        return frame
    if isinstance(frame, np.ndarray):
        return DeviceFrame(array=frame)
    return DeviceFrame(tensor=frame)


class ConceptMaskAdapter(DebugVisionAdapter):
    """Shared base for text-prompted concept->mask adapters (the SAM3 family).

    Owns everything that is NOT model-specific: the control contract (objects /
    signs / colors / background / multi_instance), the ``+``/``-`` carving in
    :meth:`segment`, and the RGBA compositing in :meth:`infer`. Subclasses load a
    model and implement :meth:`_infer_masks` (frame -> per-concept mask lists) plus
    :meth:`_restart_tracking` (drop all temporal state when the concept set changes).

    Resolution: SAM3's vision encoders accept any patch-multiple input size, but the
    global-attention RoPE tables are built from the model config at load time — so
    the inference resolution is a LOAD-TIME knob (``resolution`` ctor arg), applied
    to the model config AND every processor call. ``RESOLUTIONS`` are the supported
    presets; lower = quadratically less encoder work. Measured on real robot frames
    (5090, fp16): 672 is ~1.8x faster than 1008 with equal-or-better masks.
    """

    RESOLUTIONS = (1008, 672, 504)  # supported presets (GUI: Full / Balanced / Fast)
    DEFAULT_RESOLUTION = 672  # measured equal-or-better masks than 1008 at ~1.8x the speed
    MAX_OBJECTS = 6  # cap monitored objects (shared encoder keeps multi-object cheap)
    DEFAULT_PROMPT = "object"

    def __init__(self, device: str = "cuda", resolution: int | None = None):
        super().__init__(device)
        resolution = int(resolution) if resolution else self.DEFAULT_RESOLUTION
        assert resolution % 14 == 0 and resolution >= 224, (
            f"resolution must be a multiple of the ViT patch (14) and >= 224, got {resolution}"
        )
        self.resolution = resolution
        self._proc_size = {"height": resolution, "width": resolution}
        self._cv2 = _import_cv2()
        self.prompt = self.DEFAULT_PROMPT
        self._concepts: list[str] = []
        self._signs: dict[str, str] = {}
        # Data editing unions all instances of a concept (protect every match, e.g.
        # both arms); the debug overlay keeps the single-largest lock. Set via
        # set_control({"multi_instance": ...}); default False = the debug lock.
        self._seed_multi = False
        self._cam: str | None = None
        self._init_click_state()

    def _init_click_state(self) -> None:
        """Initialise the click-to-segment state.

        Separate from ``__init__`` because it is one cohesive group with one lifetime, and
        because ``_infer_masks`` reads all of it on every frame: the bare adapters the unit
        tests build with ``object.__new__`` need exactly this to be drivable, and a single
        call keeps them working when a field is added. Post: every field below exists and is
        empty/default — never shared between instances, which is why these are instance
        attributes rather than class-level defaults.
        """
        # Gestures awaiting the next frame (seeding needs a frame), and the names they
        # created. From here on a clicked object is an ordinary concept — same session,
        # chrome and compositing; only its seed is a point rather than a text detection.
        self._pending_clicks: dict[str | None, list[tuple[float, float, int]]] = {}
        self._pending_boxes: dict[str | None, list[tuple[float, float, float, float]]] = {}
        self._click_names: dict[str | None, list[str]] = {}
        # False when nothing is typed. Otherwise the adapter falls back to DEFAULT_PROMPT
        # and outlines whatever "object" matches, next to the thing you clicked.
        self._text_detection = True
        # "tracker" = promptable segmenter ("the object here"); "exemplar" = Meta's
        # image-predictor route, box as a visual prompt to the detector. Both are exposed
        # because they fail differently in clutter — see gui/docs/click_to_segment.md.
        self._box_method = "tracker"
        self._pending_click_name: dict[str | None, str] = {}
        self._last_click_seq = 0  # highest click-event id applied (the channel replays)
        self._last_click_fingerprint = ""  # so a REPLAY stays quiet but a dropped op is loud

    def _parse_concepts(self) -> list[str]:
        if not self._text_detection:
            return []  # click-only: the clicked objects are the whole concept list
        parts = (c.strip() for c in self.prompt.replace(",", ".").split("."))
        names = list(dict.fromkeys(c for c in parts if c))[: self.MAX_OBJECTS]
        return names or [self.DEFAULT_PROMPT]

    def set_camera(self, cam: str | None) -> None:
        self._cam = cam  # which camera's tracking state _infer_masks() should use

    def _restart_tracking(self) -> None:
        """Drop ALL cameras' temporal state (the concept set / seed policy changed)."""
        raise NotImplementedError

    def set_control(self, control: dict) -> None:
        # Structured monitored objects (preferred). Color/sign/background are display-only;
        # only an object-NAME change restarts tracking.
        names, signs = _parse_objects(control, self.MAX_OBJECTS)
        if names is not None:
            self._signs = signs
            new_prompt = " . ".join(names)
            if new_prompt and new_prompt != self.prompt:
                for gone in set(self._parse_concepts()) - set(names):
                    _release_concept_color(gone)  # the row was deleted: free its colour
                self.prompt = new_prompt
                self._restart_tracking()  # restart tracking on every camera with the new objects
        else:
            p = control.get("prompt")
            if isinstance(p, str) and p.strip() and p.strip() != self.prompt:
                self.prompt = p.strip()
                self._restart_tracking()
        # "Segment all instances of each concept" (both arms) vs the single largest.
        # Absent = keep the current value (default False: the debug overlay's lock;
        # the data-editing paths send True). A change restarts tracking so the next
        # frame re-seeds under the new policy instead of waiting for a flush.
        if "multi_instance" in control:
            mv = bool(control["multi_instance"])
            if mv != self._seed_multi:
                self._seed_multi = mv
                self._restart_tracking()
        # {"clicks": {cam: [[x, y, label], ...]}} in FRAME pixels, label 1 = positive.
        # Queued, not applied: seeding needs a frame, which arrives in the inference loop.
        if "box_method" in control:
            bm = str(control["box_method"] or "tracker").lower()
            if bm in ("tracker", "exemplar"):
                self._box_method = bm
        if "text_detection" in control:
            want = bool(control["text_detection"])
            if want != self._text_detection:
                self._text_detection = want
                self._restart_tracking()
        # The control block is re-read every frame and never consumed. That is harmless for
        # idempotent settings, but a gesture is an event: replaying it re-seeded the same
        # object ~13x/second and resurrected deleted ones. Apply each id at most once.
        seq = control.get("click_seq")
        fingerprint = repr([(k, control[k]) for k in sorted(_CLICK_OP_KEYS) if k in control])
        if seq is not None and seq <= self._last_click_seq:
            # The applied op rides along on every control write, so seeing it again is normal
            # and stays quiet. A DIFFERENT op that cannot advance the counter is a lost
            # gesture, and it is otherwise indistinguishable from one that did nothing —
            # which is how removals were silently dropped for a whole session.
            if fingerprint != self._last_click_fingerprint:
                logger.warning(
                    "click op DROPPED: seq %s <= applied %s — %s",
                    seq,
                    self._last_click_seq,
                    sorted(k for k in control if k in _CLICK_OP_KEYS and k != "click_seq"),
                )
            return
        if seq is not None:
            self._last_click_seq = seq
            self._last_click_fingerprint = fingerprint
            # Pairs with the API's "click op" line: together they say whether a gesture
            # crossed the process boundary, which no other log can distinguish from a no-op.
            logger.info(
                "click op seq %s: %s",
                seq,
                sorted(k for k in control if k in _CLICK_OP_KEYS and k != "click_seq"),
            )
        names_in = control.get("click_name")
        if isinstance(names_in, dict):
            self._pending_click_name = dict(names_in)
        # Removing ONE clicked object (the row's x), as opposed to clearing the camera.
        removals = control.get("clicks_remove")
        if isinstance(removals, dict):
            for cam, victims in removals.items():
                names = self._click_names.get(cam)
                track = self._tracks.get(cam) or {}
                for victim in victims or []:
                    while names and victim in names:  # duplicates would survive their own deletion
                        names.remove(victim)
                    for key in ("masks", "scores", "objs"):
                        d = track.get(key)
                        if isinstance(d, dict):
                            d.pop(victim, None)
                # Rebuild from the masks we still hold. Nulling the session instead would
                # take the detect path, which re-seeds only text concepts — dropping every
                # other clicked object along with this one.
                track["reseed"] = True
                for victim in victims or []:
                    still_used = victim in self._parse_concepts() or any(
                        victim in v for v in self._click_names.values()
                    )
                    if not still_used:
                        _release_concept_color(victim)
                logger.info("click[%s]: removed %s", cam, victims)
        clicks = control.get("clicks")
        if isinstance(clicks, dict):
            for cam, pts in clicks.items():
                if not pts:  # empty list = forget this camera's clicked objects
                    self._pending_clicks.pop(cam, None)
                    self._pending_boxes.pop(cam, None)
                    for gone in self._click_names.pop(cam, []):
                        if gone not in self._parse_concepts() and not any(
                            gone in v for v in self._click_names.values()
                        ):
                            _release_concept_color(gone)
                    # Drop the session too, or we keep paying for the cleared masklets.
                    self._tracks.pop(cam, None)
                    continue
                self._pending_clicks.setdefault(cam, []).extend(
                    (float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1) for p in pts
                )
        # {"boxes": {cam: [[x0, y0, x1, y1], ...]}} in FRAME pixels, corners ordered.
        # Same queue-then-seed path as clicks, behind the same seq gate.
        boxes = control.get("boxes")
        if isinstance(boxes, dict):
            for cam, bxs in boxes.items():
                if not bxs:
                    self._pending_boxes.pop(cam, None)
                    continue
                self._pending_boxes.setdefault(cam, []).extend(
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in bxs
                )
        # Relabel the tracked mask; never re-detect. The name is a key, not a query — no
        # text finds this object, so searching for the new name would lose it.
        renames = control.get("clicks_rename")
        if isinstance(renames, dict):
            for cam, mapping in renames.items():
                names = self._click_names.get(cam)
                if not names or not isinstance(mapping, dict):
                    continue
                track = self._tracks.get(cam) or {}
                for old, new in mapping.items():
                    new = str(new).strip()
                    if not new or old not in names or new in names:
                        continue
                    names[names.index(old)] = new
                    for key in ("masks", "scores", "objs"):
                        d = track.get(key)
                        if isinstance(d, dict) and old in d:
                            d[new] = d.pop(old)
                    logger.info("click[%s]: renamed %r -> %r (mask kept)", cam, old, new)

    def segment_many(self, frames_by_cam: dict) -> dict[str, dict[str, np.ndarray]]:
        """:meth:`segment` for several cameras' frames of the SAME timestep.

        Serial per-camera loop -- tracking is sequential within a camera by
        construction, so this cannot be batched. Pre: each frame is HxWx3 uint8
        numpy, or a uint8 CHW tensor already on the device. Post:
        ``{cam: {concept: mask}}``, exactly one entry per input camera.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for cam, frame in frames_by_cam.items():
            self.set_camera(cam)
            out[cam] = self.segment(frame)
        return out

    def _infer_masks(self, frame_rgb: np.ndarray) -> tuple[dict[str, list[np.ndarray]], int, int]:
        """Run the model for one frame -> ``(masks_by_concept, h, w)``. Mutates the
        current camera's tracking state — the caller must have selected the camera
        via :meth:`set_camera`."""
        raise NotImplementedError

    def segment(self, frame_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Per-concept boolean masks for this frame (positive concepts only).

        Runs the same tracking pipeline as :meth:`infer` but returns the raw
        ``{concept: HxW bool mask}`` instead of an RGBA overlay — what an
        offline pixel-editing pass (background replacement, recolor) needs.
        Negative (``-``) concepts are carved out of the positives, exactly as
        the overlay compositor does, so the returned masks are the region that
        an effect should KEEP as foreground. Precondition: ``frame_rgb`` is
        contiguous HxWx3 uint8 RGB; call :meth:`set_camera` / :meth:`reset` to
        scope and reseed per-camera tracking just like the live loop. Whether a
        concept yields all its instances (both arms) or just the largest is set via
        ``set_control({"multi_instance": ...})`` — same knob the overlay + batch share.
        """
        frame_rgb = as_device_frame(frame_rgb)
        masks_by_concept, _h, _w = self._infer_masks(frame_rgb)
        # Apply the same +/- carving the compositor does, so a caller gets the
        # final kept region per positive concept without re-deriving the logic.
        neg = None
        has_neg = any(self._signs.get(c, "+") == "-" for c in self._concepts)
        if has_neg:
            neg = np.zeros(frame_rgb.shape[:2], dtype=bool)
            for c in self._concepts:
                if self._signs.get(c, "+") == "-":
                    for m in masks_by_concept.get(c, []):
                        neg |= m
        out: dict[str, np.ndarray] = {}
        for c in self._concepts:
            if self._signs.get(c, "+") == "-":
                continue
            ms = masks_by_concept.get(c, [])
            if not ms:
                continue
            union = np.zeros(frame_rgb.shape[:2], dtype=bool)
            for m in ms:
                union |= m
            if neg is not None:
                union &= ~neg
            out[c] = union
        return out


def _from_cache_first(cls, model_id: str, **kwargs):
    """``from_pretrained`` that trusts the local Hub cache before the network.

    Every ``from_pretrained`` re-validates its config files against
    huggingface.co even when the weights are already on disk. That is six
    loads here, each several round trips, and this rig tunnels all Hub traffic
    (~320 ms per trip): a mask save measured 9 s of its 36 s on HEAD requests
    for files that had not changed. Loading from the cache first removes that.

    Pre: ``cls`` has a ``from_pretrained`` accepting ``local_files_only``.
    Post: the cache is used if complete; a miss falls back to the network, so a
    first run still downloads. The tradeoff is deliberate — an upstream change
    to the weights is picked up when the cache is cleared, not silently
    mid-project, which is the behaviour worth having while masking a dataset.
    """
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_id, **kwargs)


class Sam3TrackByDetectionAdapter(ConceptMaskAdapter):
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
    FLUSH_EVERY = 150  # rebuild each tracker session every N frames -> flat GPU memory
    LOST_THRESH = 0.30  # sigmoid(object_score_logits) below this = track lost -> Tier-1 recover
    RECOVER_EVERY = 5  # throttle Tier-1 re-detection attempts (frames) while an object is lost

    def __init__(self, device: str = "cuda", resolution: int | None = None):
        super().__init__(device, resolution)
        try:
            import torch
            from PIL import Image
            from transformers import (
                Sam3Config,
                Sam3Model,
                Sam3Processor,
                Sam3TrackerVideoConfig,
                Sam3TrackerVideoModel,
                Sam3TrackerVideoProcessor,
            )
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._Image = Image
        logger.info("loading %s (detector + geometric tracker) at %d px ...", self.SAM3_ID, self.resolution)
        try:
            # Inference resolution is a load-time knob: the global-attention layers build
            # their RoPE tables from config.image_size, so it must be set BEFORE loading
            # (pos-embeds tile to any size at runtime; a processor-only resize crashes).
            det_cfg = _from_cache_first(Sam3Config, self.SAM3_ID)
            det_cfg.vision_config.image_size = self.resolution
            trk_cfg = _from_cache_first(Sam3TrackerVideoConfig, self.SAM3_ID)
            trk_cfg.image_size = self.resolution
            trk_cfg.memory_attention_rope_feat_sizes = [self.resolution // 14] * 2
            self.det_proc = _from_cache_first(Sam3Processor, self.SAM3_ID)
            self.det = (
                _from_cache_first(Sam3Model, self.SAM3_ID, config=det_cfg, dtype=torch.float16)
                .to(device)
                .eval()
            )
            self.trk_proc = _from_cache_first(Sam3TrackerVideoProcessor, self.SAM3_ID)
            # Read the preprocessing constants off the processor rather than
            # restating them: a checkpoint that changes them must change this
            # path too, and a silent mismatch is a quietly wrong mask.
            _inner = getattr(self.trk_proc, "image_processor", self.trk_proc)
            _mean = getattr(_inner, "image_mean", (0.5, 0.5, 0.5))
            _std = getattr(_inner, "image_std", (0.5, 0.5, 0.5))
            self._pp_rescale = float(getattr(_inner, "rescale_factor", 1.0 / 255.0))
            self._pp_mean = self._torch.tensor(_mean, device=self.device).view(3, 1, 1)
            self._pp_std = self._torch.tensor(_std, device=self.device).view(3, 1, 1)
            self.trk = (
                _from_cache_first(Sam3TrackerVideoModel, self.SAM3_ID, config=trk_cfg, dtype=torch.float16)
                .to(device)
                .eval()
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
        self._det_threshold = 0.5
        self._tracks: dict[str | None, dict] = {}  # per-camera tracker state (session + masks)
        # Per-concept detector text features (deterministic per string -> lifetime cache);
        # lets _detect_many skip the text encoder entirely on seed/recover.
        self._text_cache: dict[str, tuple] = {}

    def reset(self) -> None:
        """Discontinuity — scrub, episode change, wrap. Drop the memory bank, which describes
        frames that no longer precede this one.

        A TEXT concept is re-detected from its name, so nothing needs keeping. A CLICKED one
        has no such path, and deleting it was a self-fulfilling decision: it was dropped
        because "it cannot be recovered", which was only true because the mask-based re-seed
        below was never wired up. The tracker is memory-conditioned, not a mask-copier, so
        handing it the last good mask on the new frame is a real query — find this thing
        here — and it is exactly what the periodic flush already does every FLUSH_EVERY
        frames. Keep the clicked masks, drop the session, re-seed from them next frame.
        """
        self._pending_clicks.pop(self._cam, None)  # queued for a frame that is now gone
        self._pending_boxes.pop(self._cam, None)
        track = self._tracks.get(self._cam)
        if track is None:
            return
        clicked = set(self._click_names.get(self._cam, []))
        keep = {
            c: m
            for c, m in track.get("masks", {}).items()
            if c in clicked and track.get("scores", {}).get(c, 0.0) >= self.LOST_THRESH
        }
        if not keep:
            self._tracks.pop(self._cam, None)
            self._click_names.pop(self._cam, None)
            return
        track["masks"] = dict(keep)
        track["scores"] = {c: track["scores"][c] for c in keep}
        track["session"] = None  # the memory bank is stale; the masks are not
        track["objs"] = {}
        track["since_flush"] = 0
        track["reseed"] = True

    def _restart_tracking(self) -> None:
        self._tracks = {}

    # ---------------- Tier 1: image detector (text -> mask per concept) ----------------
    def _select_instances(self, res: dict, h: int, w: int) -> np.ndarray | None:
        """Turn one post-processed detection result into a seed mask, or None.

        Debug overlay (``_seed_multi`` False): the single largest instance — the
        tracker then locks onto that one object. Data editing (``_seed_multi``
        True): the UNION of every instance, so a concept like "robot arm" protects
        BOTH arms, not just the biggest (SAM3 returns them as separate instances;
        taking the largest silently dropped the second). Instances under the small
        area gate are noise, not objects."""
        masks = res.get("masks", [])
        if len(masks) == 0:
            return None
        arrs = [(m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)) > 0 for m in masks]
        arrs = [a for a in arrs if int(a.sum()) > 50]
        if not arrs:
            return None
        if self._seed_multi:
            union = np.zeros((h, w), dtype=bool)
            for a in arrs:
                union |= a
            assert union.shape == (h, w), f"detector mask {union.shape} != frame {(h, w)}"
            return union
        best = max(arrs, key=lambda a: int(a.sum()))
        assert best.shape == (h, w), f"detector mask {best.shape} != frame {(h, w)}"
        return best

    def _detect_many(
        self, frame_rgb: np.ndarray, concepts: list[str], h: int, w: int
    ) -> dict[str, np.ndarray | None]:
        """Seed masks for ``concepts`` on ONE frame with ONE vision encode.

        The detector's vision backbone is ~73% of a full forward and depends only
        on the frame; ``Sam3Model.forward`` takes precomputed ``vision_embeds`` /
        ``text_embeds`` for exactly this reuse, so N concepts cost one encode plus
        N cheap fusion/decode passes instead of N full forwards. Text features are
        deterministic per concept string and cached for the adapter's lifetime.
        Pre: ``frame_rgb`` is HxWx3 uint8. Post: one entry per concept (None =
        nothing detected)."""
        if not concepts:
            return {}
        torch = self._torch
        inp = self.det_proc(
            images=self._Image.fromarray(frame_rgb),
            size=self._proc_size,  # match the load-time model resolution
            return_tensors="pt",
        ).to(self.device)
        out: dict[str, np.ndarray | None] = {}
        with torch.inference_mode():
            vision_embeds = self.det.vision_encoder(inp["pixel_values"])
            texts, attns = [], []
            for concept in concepts:
                cached = self._text_cache.get(concept)
                if cached is None:
                    tok = self.det_proc(text=concept, return_tensors="pt").to(self.device)
                    feats = self.det.get_text_features(
                        input_ids=tok["input_ids"],
                        attention_mask=tok.get("attention_mask"),
                        return_dict=True,
                    ).pooler_output
                    cached = (feats, tok.get("attention_mask"))
                    self._text_cache[concept] = cached
                texts.append(cached[0])
                attns.append(cached[1])
            # ONE decode for all concepts, batched along the text dimension. The
            # fusion/decode passes are launch-bound (profiled: ~1,233 kernel
            # launches per frame), so N serial passes cost far more than one
            # batch-N pass — measured 6 concepts at 672px: 175 ms -> 21 ms, with
            # per-concept masks equal to serial to within fp16 boundary noise
            # (XOR <= 33 px on 180k-px masks, zero on small ones; far inside the
            # 5 px feather every composite applies). The encoder output is
            # batch-1; expand() broadcasts it as views, no copy.
            batch = len(concepts)
            text_embeds = torch.cat(texts, dim=0)
            attn = torch.cat(attns, dim=0) if attns[0] is not None else None
            fields = {}
            for key, value in vision_embeds.items():
                if torch.is_tensor(value) and value.shape[:1] == (1,):
                    fields[key] = value.expand(batch, *value.shape[1:])
                elif isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
                    fields[key] = type(value)(
                        v.expand(batch, *v.shape[1:]) if v.shape[:1] == (1,) else v for v in value
                    )
                else:
                    fields[key] = value
            fwd = self.det(
                vision_embeds=type(vision_embeds)(**fields), text_embeds=text_embeds, attention_mask=attn
            )
            results = self.det_proc.post_process_instance_segmentation(
                fwd, threshold=self._det_threshold, target_sizes=[(h, w)] * batch
            )
            # One result per concept, by construction of the batched decode:
            # a length mismatch is a contract break, not something to truncate.
            for concept, res in zip(concepts, results, strict=True):
                out[concept] = self._select_instances(res, h, w)
        return out

    # ---------------- Tier 2: geometric video tracker ----------------
    def _pv(self, frame):
        """Normalised pixel values for the tracker, on the device.

        When the frame is already a device tensor this resizes and normalises
        there. The processor's own path costs a device->host copy, a PIL
        conversion and a CPU resize -- ~15 ms of a ~49 ms frame, measured,
        while the GPU sat at 6% -- so on the batch path, where frames arrive
        from NVDEC, none of that has to happen. Numpy input keeps the
        processor's path, unchanged, for the live overlay and every other
        caller. The two agree within the tolerance asserted by
        tests/overlays/test_device_preprocess.py.
        """
        frame = as_device_frame(frame)
        if frame.tensor is None or not str(frame.tensor.device).startswith("cuda"):
            inp = self.trk_proc(
                images=self._Image.fromarray(frame.numpy()), size=self._proc_size, return_tensors="pt"
            )
            return inp["pixel_values"][0].to(self.device, self._torch.float16)

        from torchvision.transforms import v2

        torch = self._torch
        x = frame.tensor.to(torch.float32)
        # antialias matches PIL's downscaling filter, which is what the
        # processor uses (resample=BILINEAR); without it a 1280->672 downscale
        # aliases and the masks move.
        x = v2.functional.resize(
            x,
            [self._proc_size["height"], self._proc_size["width"]],
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        )
        x = x * self._pp_rescale
        x = (x - self._pp_mean) / self._pp_std
        return x.to(self.device, torch.float16)

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

    # Prompt coordinates normalise against this, while _pv() feeds the configured preset —
    # so an unscaled point lands (native / resolution) out. Measured 1.47-1.52x at 672.
    # Hard to spot: the misplaced mask looks plausible rather than broken.
    PROMPT_NATIVE_RES = 1008

    def _mask_from_prompt(
        self,
        pv,
        h: int,
        w: int,
        *,
        points: list[tuple[float, float, int]] | None = None,
        box: tuple[float, float, float, float] | None = None,
    ):
        """Segment what the prompt points at, on the frame ``pv`` came from.

        Pre: exactly one of ``points`` (x, y, label; >=1 positive) or ``box`` (ordered
        x0, y0, x1, y1), both in FRAME pixels. Post: an HxW bool mask, or None.

        Runs in a throwaway session; the caller re-seeds the real one from the mask, which
        is how a clicked object joins the session text concepts live in.
        """
        assert (points is None) != (box is None), "exactly one prompt kind"
        torch = self._torch
        k = self.resolution / self.PROMPT_NATIVE_RES
        sess = self.trk_proc.init_video_session(
            video=None,
            inference_device=self.device,
            inference_state_device=self.device,
            dtype=torch.float16,
        )
        fidx = sess.add_new_frame(pv)
        prompt_kw = (
            {
                "input_points": [[[[p[0] * k, p[1] * k] for p in points]]],
                "input_labels": [[[int(p[2]) for p in points]]],
            }
            if points is not None
            else {"input_boxes": [[[box[0] * k, box[1] * k, box[2] * k, box[3] * k]]]}
        )
        self.trk_proc.process_new_points_or_boxes_for_video_frame(
            inference_session=sess,
            frame_idx=fidx,
            obj_ids=[1],
            original_size=(h, w),
            **prompt_kw,
        )
        probe = {"session": sess, "objs": {"_click": 1}, "masks": {}, "scores": {}, "since_flush": 0}
        try:
            with torch.inference_mode():
                out = self.trk(inference_session=sess, frame_idx=fidx)
            self._read_output(probe, out, h, w)
        except Exception as e:
            logger.warning("prompt seed failed (%s: %s)", type(e).__name__, e)
            return None
        m = probe["masks"].get("_click")
        return m if m is not None and m.any() else None

    def _apply_clicks(self, track: dict, pv, frame_rgb: np.ndarray, h: int, w: int) -> bool:
        """Turn this camera's queued clicks/boxes into tracked concepts. Returns True if
        the session was rebuilt (the caller should not also seed this frame)."""
        pending = self._pending_clicks.pop(self._cam, None)
        if not pending:
            return self._apply_boxes(track, pv, frame_rgb, h, w)
        mask = self._mask_from_prompt(pv, h, w, points=pending)
        if mask is None:
            logger.info("click[%s]: no mask at %s", self._cam, [(p[0], p[1]) for p in pending])
            return False
        names = self._click_names.setdefault(self._cam, [])
        # Clicking the same thing again refines it. Otherwise every click adds a
        # near-duplicate masklet, each costing memory attention on every frame.
        positives = [(p[0], p[1]) for p in pending if p[2] == 1]
        for existing in names:
            prev = track.get("masks", {}).get(existing)
            if prev is None or not prev.any():
                continue
            inside = any(
                0 <= int(y) < prev.shape[0] and 0 <= int(x) < prev.shape[1] and prev[int(y), int(x)]
                for x, y in positives
            )
            if inside:
                logger.info(
                    "click[%s]: refined %r (%d -> %d px)",
                    self._cam,
                    existing,
                    int(prev.sum()),
                    int(mask.sum()),
                )
                seeds = {
                    c: track["masks"][c]
                    for c in track.get("masks", {})
                    if track["scores"].get(c, 0.0) >= self.LOST_THRESH and c != existing
                }
                seeds[existing] = mask
                self._seed(track, seeds, pv, h, w)
                return True
        return self._admit_clicked_mask(track, mask, pv, h, w)

    def _mask_from_exemplar(self, frame_rgb: np.ndarray, box: tuple, h: int, w: int):
        """Segment via the DETECTOR, with the box as a visual exemplar (Meta's image
        predictor route). Pre: ``box`` is (x0, y0, x1, y1) in frame pixels. Post: an HxW
        bool mask, or None.

        The box conditions a whole-image detection rather than deciding the mask, so this
        returns several scored instances. Keep the one that best overlaps the drawn
        rectangle — max-score would return a confident object elsewhere in the frame.

        The BOX is the only prompt. An earlier version passed the typed concepts as text
        alongside it, because that scored better in clutter, but those words belong to other
        object rows: a gesture would then mean different things depending on rows it has
        nothing to do with, and boxing a dowel while "green ring" sat in another row biased
        the result toward the ring. The two box modes differ in which model reads the box,
        and in nothing else.
        """
        torch = self._torch
        x0, y0, x1, y1 = box
        text = ""
        try:
            inp = self.det_proc(
                images=self._Image.fromarray(frame_rgb), size=self._proc_size, return_tensors="pt"
            ).to(self.device)
            tok = self.det_proc(text=text, return_tensors="pt").to(self.device)
            boxes = self._torch.tensor(
                [[[(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]]],
                dtype=next(self.det.parameters()).dtype,
                device=self.device,
            )
            labels = self._torch.ones(1, 1, dtype=self._torch.long, device=self.device)
            with torch.inference_mode():
                feats = self.det.get_text_features(
                    input_ids=tok["input_ids"],
                    attention_mask=tok.get("attention_mask"),
                    return_dict=True,
                ).pooler_output
                fwd = self.det(
                    vision_embeds=self.det.vision_encoder(inp["pixel_values"]),
                    text_embeds=feats,
                    attention_mask=tok.get("attention_mask"),
                    input_boxes=boxes,
                    input_boxes_labels=labels,
                )
                res = self.det_proc.post_process_instance_segmentation(
                    fwd, threshold=self._det_threshold, target_sizes=[(h, w)]
                )[0]
        except Exception as e:
            logger.warning("exemplar box failed (%s: %s)", type(e).__name__, e)
            return None
        masks = res["masks"] if isinstance(res, dict) else res.masks
        if masks is None or not len(masks):
            logger.info("box[%s]: exemplar detected nothing in %s", self._cam, [round(v) for v in box])
            return None
        drawn = np.zeros((h, w), bool)
        drawn[max(0, int(y0)) : int(y1), max(0, int(x0)) : int(x1)] = True
        best, best_iou = None, -1.0
        for m in masks:
            mm = np.asarray(m.cpu() if hasattr(m, "cpu") else m).astype(bool)
            union = np.logical_or(mm, drawn).sum()
            iou = float(np.logical_and(mm, drawn).sum()) / float(union) if union else 0.0
            if iou > best_iou:
                best, best_iou = mm, iou
        logger.info(
            "box[%s]: detector kept 1 of %d instance(s) (box IoU %.2f)", self._cam, len(masks), best_iou
        )
        return best if best is not None and best.any() else None

    def _apply_boxes(self, track: dict, pv, frame_rgb: np.ndarray, h: int, w: int) -> bool:
        """Turn ONE queued box into a new tracked concept; re-queue the rest, since each
        admit re-seeds the session. A box always creates a new object and never refines:
        enclosing a mask is how you select the thing next to it, so overlap proves nothing
        the way a point inside a mask does."""
        boxes = self._pending_boxes.pop(self._cam, None)
        if not boxes:
            return False
        box, rest = boxes[0], boxes[1:]
        if rest:
            self._pending_boxes[self._cam] = rest
        if self._box_method == "exemplar":
            mask = self._mask_from_exemplar(frame_rgb, box, h, w)
        else:
            mask = self._mask_from_prompt(pv, h, w, box=box)
        if mask is None:
            logger.info("box[%s]: no mask in %s", self._cam, [round(v) for v in box])
            return False
        return self._admit_clicked_mask(track, mask, pv, h, w)

    def _admit_clicked_mask(self, track: dict, mask: np.ndarray, pv, h: int, w: int) -> bool:
        """Admit a probed mask as a new tracked object: cap check, placeholder name, and a
        re-seed of the real session so it is tracked like every text concept."""
        names = self._click_names.setdefault(self._cam, [])
        # One cap over both kinds, because the session carries both. Capping them
        # separately would allow twice MAX_OBJECTS masklets.
        text = self._parse_concepts()
        if len(text) + len(names) >= self.MAX_OBJECTS:
            logger.warning(
                "click[%s]: at the %d-object cap (%s) — remove one before adding another",
                self._cam,
                self.MAX_OBJECTS,
                text + names,
            )
            return False
        # A clicked object has no name by construction, so it gets a slot label to rename,
        # not a guess at what it is.
        given = self._pending_click_name.pop(self._cam, None)
        name = str(given).strip() if given else f"object_{len(names) + 1}"
        if name in names:
            # Reusing a live name would replace a different object's mask, and its row
            # would start showing this object instead.
            base, n = name, 2
            while name in names:
                name = f"{base} ({n})"
                n += 1
            logger.warning("click[%s]: %r is taken — using %r instead", self._cam, base, name)
        names.append(name)
        seeds = {
            c: track["masks"][c]
            for c in track.get("masks", {})
            if track["scores"].get(c, 0.0) >= self.LOST_THRESH
        }
        seeds[name] = mask
        logger.info(
            "click[%s]: seeded %r (%d px) alongside %s", self._cam, name, int(mask.sum()), sorted(seeds)
        )
        self._seed(track, seeds, pv, h, w)
        return True

    def _live_masks(self, track: dict) -> dict[str, list[np.ndarray]]:
        """Per-concept mask list for compositing — only objects currently held (score ok).

        Logs the held->lost transition. An object leaving this set simply stops being drawn,
        which for a CLICKED object is permanent — nothing re-seeds it. That happened with no
        trace in any log, so "my objects disappeared" was indistinguishable from a reset, a
        dropped gesture, or a bug, and cost several wrong diagnoses.
        """
        out = self._live_masks_now(track)
        held = {c for c, m in out.items() if m}
        prev = track.get("held")
        if prev is not None and held != prev:
            clicked = set(self._click_names.get(self._cam, []))
            gone = sorted(prev - held)
            if gone:
                logger.info(
                    "track[%s]: lost %s (score < %.2f)%s",
                    self._cam,
                    gone,
                    self.LOST_THRESH,
                    " — clicked, so it cannot come back" if any(g in clicked for g in gone) else "",
                )
            if held - prev:
                logger.info("track[%s]: recovered %s", self._cam, sorted(held - prev))
        track["held"] = held
        return out

    def _live_masks_now(self, track: dict) -> dict[str, list[np.ndarray]]:
        return {
            c: (
                [track["masks"][c]]
                if track["scores"].get(c, 0.0) >= self.LOST_THRESH and c in track["masks"]
                else []
            )
            for c in self._concepts
        }

    def _infer_masks(self, frame_rgb: np.ndarray) -> tuple[dict[str, list[np.ndarray]], int, int]:
        """Drive the tracker for one frame and return ``(masks_by_concept, h, w)``.

        The body shared by :meth:`infer` (which composites an RGBA overlay) and
        :meth:`segment` (which returns the raw masks). Mutates per-camera tracker
        state — the caller must have selected the camera via :meth:`set_camera`.
        """
        torch = self._torch
        frame_rgb = as_device_frame(frame_rgb)
        h, w = frame_rgb.shape[:2]
        cam = self._cam
        # Append clicked objects so everything downstream treats them identically. They are
        # not in the prompt, so recovery simply finds nothing for a lost one.
        self._concepts = self._parse_concepts() + self._click_names.get(cam, [])
        track = self._tracks.get(cam)
        if track is None or track.get("shape") != (h, w):
            track = {
                "session": None,
                "objs": {},
                "masks": {},
                "scores": {},
                "since_flush": 0,
                # RECOVER_EVERY so the FIRST frame after a reset/scrub probes immediately;
                # a failed probe resets it, throttling subsequent attempts (below).
                "since_recover": self.RECOVER_EVERY,
                "shape": (h, w),
            }
            self._tracks[cam] = track
        pv = self._pv(frame_rgb)
        if track.pop("reseed", False):
            keep = {
                c: track["masks"][c]
                for c in track.get("masks", {})
                if track["scores"].get(c, 0.0) >= self.LOST_THRESH
            }
            if keep:
                self._seed(track, keep, pv, h, w)
            else:
                track["session"] = None
        if self._apply_clicks(track, pv, frame_rgb.numpy(), h, w):
            # _apply_clicks appended a name and _live_masks filters by the list, so
            # re-read it — otherwise the new object is invisible for one frame.
            self._concepts = self._parse_concepts() + self._click_names.get(cam, [])
            return self._live_masks(track), h, w

        if track["session"] is None:
            # Nothing to detect: every region here was clicked, or none exists yet. The
            # picker alone starts the worker, so cameras you never touch land here.
            if not self._concepts:
                return self._live_masks(track), h, w
            # No track yet — Tier 1 detects each object to seed Tier 2. Throttled like
            # recovery: with no objects in view, an unthrottled probe re-runs the detector
            # for EVERY concept on EVERY frame (measured ~30 ms/concept/frame — it
            # dominated a live run's per-camera cost during empty-scene stretches).
            track["since_recover"] += 1
            if track["since_recover"] < self.RECOVER_EVERY:
                return self._live_masks(track), h, w
            track["since_recover"] = 0
            # Never text-detect a clicked object, exactly as the recover path does not: its
            # name is the user's label, not a description of anything.
            clicked = set(self._click_names.get(cam, []))
            detected = self._detect_many(
                frame_rgb.numpy(), [c for c in self._concepts if c not in clicked], h, w
            )
            seeds = {c: m for c, m in detected.items() if m is not None}
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
                    to_detect = []
                    clicked = set(self._click_names.get(cam, []))
                    for c in self._concepts:
                        if track["scores"].get(c, 0.0) >= self.LOST_THRESH and c in track["masks"]:
                            seeds[c] = track["masks"][c]  # healthy: reseed from current mask
                        elif c in clicked:
                            # Never re-detect a clicked object from its label: the name is
                            # the user's, not a description, so 'stick' would jump onto some
                            # other stick. Lost stays lost until clicked again.
                            continue
                        else:
                            to_detect.append(c)  # lost: Tier-1 re-detect (one shared encode)
                    recovered = 0
                    for c, m in self._detect_many(frame_rgb, to_detect, h, w).items():
                        if m is not None:
                            seeds[c] = m
                            recovered += 1
                    is_flush = track["since_flush"] >= self.FLUSH_EVERY
                    if not is_flush and not recovered:
                        # Recover attempt found nothing new: rebuilding the session would
                        # recondition the SAME healthy masks it already tracks — pure cost
                        # (~40-70 ms) paid every RECOVER_EVERY frames for as long as an
                        # object stays undetectable. Keep the live session; the detect
                        # probes above are the only work a persistent loss needs.
                        track["since_recover"] = 0
                        return self._live_masks(track), h, w
                    why = "flush" if is_flush else "recover"
                    logger.info(
                        "%s[%s]: lost %s · re-seeding %s", why, self._cam or "?", lost or "none", list(seeds)
                    )
                    track["since_recover"] = 0
                    if seeds:
                        self._seed(track, seeds, pv, h, w)

        return self._live_masks(track), h, w


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

# Text-prompted segmenters — the models valid for concept masking (data editing);
# excludes overlay-only adapters like policy saliency.
SEGMENTER_KEYS: tuple[str, ...] = tuple(
    k for k, cls in ADAPTERS.items() if issubclass(cls, ConceptMaskAdapter)
)


def build_adapter(key: str, device: str = "cuda", resolution: int | None = None) -> DebugVisionAdapter:
    """Instantiate an adapter by key. ``resolution`` (a ``ConceptMaskAdapter.RESOLUTIONS``
    preset) applies only to concept-mask adapters — it is a LOAD-TIME knob; changing it
    means rebuilding the adapter. None = the adapter's default. Non-segmenter adapters
    ignore it."""
    if key not in ADAPTERS:
        raise ValueError(f"unknown debug-vision model '{key}'; have {list(ADAPTERS)}")
    cls = ADAPTERS[key]
    if issubclass(cls, ConceptMaskAdapter):
        return cls(device=device, resolution=resolution)
    return cls(device=device)
