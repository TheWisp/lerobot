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
        names.append(name)
        signs[name] = "-" if o.get("sign") == "-" else "+"
    return list(dict.fromkeys(names)), signs


def _concept_color(concept, concepts):
    """Stable color for a concept: palette by position, hashed fallback. Auto-assigned
    (never user-chosen), used by the detection chrome to tell objects apart."""
    if concept in concepts:
        return _CONCEPT_PALETTE[concepts.index(concept) % len(_CONCEPT_PALETTE)]
    return _color_for(concept)  # transparent


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
        # Batch the per-frame vision encode across cameras (segment_many). Default on;
        # runtime-togglable via set_control({"batch_cameras": ...}) — an EXPERIMENTAL
        # perf option: batched cuDNN kernels differ numerically from batch-1, so
        # borderline tracker scores can take a different (equally valid) trajectory.
        # The same flag drives preview AND commit, so preview == commit per setting.
        # Default OFF: the batched vision encode is numerically different enough to
        # collapse tracking holds on some scenes (measured: front-cam dowel 4/199
        # batched vs 24/199 serial, ring 4/199 vs 89/199, merged_raw ep157). Opt-in.
        self._batch_cams = False
        self._cam: str | None = None

    def _parse_concepts(self) -> list[str]:
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
        if "batch_cameras" in control:
            self._batch_cams = bool(control["batch_cameras"])  # runtime toggle, no restart needed

    def segment_many(self, frames_by_cam: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
        """:meth:`segment` for several cameras' frames of the SAME timestep.

        Base implementation is the serial loop (always correct); adapters that can
        share work across cameras (one batched vision encode) override
        :meth:`_prime_batch` — the serial per-camera calls then reuse the primed
        state. Honors the ``batch_cameras`` control flag. Pre: each frame is HxWx3
        uint8. Post: ``{cam: {concept: mask}}``, exactly one entry per input camera.
        """
        if self._batch_cams and len(frames_by_cam) > 1:
            self._prime_batch(frames_by_cam)
        out: dict[str, dict[str, np.ndarray]] = {}
        for cam, frame in frames_by_cam.items():
            self.set_camera(cam)
            out[cam] = self.segment(frame)
        return out

    def _prime_batch(self, frames_by_cam: dict[str, np.ndarray]) -> None:
        """Optional hook: do the cross-camera shared work (e.g. one batched encode)
        before the per-camera :meth:`segment` calls. Base: nothing to share."""

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
            det_cfg = Sam3Config.from_pretrained(self.SAM3_ID)
            det_cfg.vision_config.image_size = self.resolution
            trk_cfg = Sam3TrackerVideoConfig.from_pretrained(self.SAM3_ID)
            trk_cfg.image_size = self.resolution
            trk_cfg.memory_attention_rope_feat_sizes = [self.resolution // 14] * 2
            self.det_proc = Sam3Processor.from_pretrained(self.SAM3_ID)
            self.det = (
                Sam3Model.from_pretrained(self.SAM3_ID, config=det_cfg, dtype=torch.float16).to(device).eval()
            )
            self.trk_proc = Sam3TrackerVideoProcessor.from_pretrained(self.SAM3_ID)
            self.trk = (
                Sam3TrackerVideoModel.from_pretrained(self.SAM3_ID, config=trk_cfg, dtype=torch.float16)
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
        # Per-sweep preprocessed frames from _prime_batch (consumed by _infer_masks).
        self._pv_cache: dict[str | None, object] = {}

    def reset(self) -> None:
        # Discontinuity: drop this camera's session so the next infer() re-seeds from
        # scratch instead of propagating a stale memory bank across a scrub/episode/wrap.
        self._tracks.pop(self._cam, None)

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
                text_embeds, attn = cached
                fwd = self.det(vision_embeds=vision_embeds, text_embeds=text_embeds, attention_mask=attn)
                res = self.det_proc.post_process_instance_segmentation(
                    fwd, threshold=self._det_threshold, target_sizes=[(h, w)]
                )[0]
                out[concept] = self._select_instances(res, h, w)
        return out

    # ---------------- Tier 2: geometric video tracker ----------------
    def _pv(self, frame_rgb: np.ndarray):
        inp = self.trk_proc(
            images=self._Image.fromarray(frame_rgb), size=self._proc_size, return_tensors="pt"
        )
        return inp["pixel_values"][0].to(self.device, self._torch.float16)

    def _prime_batch(self, frames_by_cam: dict[str, np.ndarray]) -> None:
        """One batched tracker vision encode for all cameras with a LIVE session,
        pre-seeded into each session's feature cache — the tracker's frame path
        checks the cache before encoding, so the per-camera steps skip their own
        encode (measured 1.31x per 2-cam sweep). Cameras without a session (about
        to seed/re-seed at frame 0) are left out — their encode happens in _seed.
        The preprocessed tensors are also cached for :meth:`_infer_masks` via
        ``_pv_cache`` so the frame isn't preprocessed twice."""
        torch = self._torch
        ready = []
        for cam, frame in frames_by_cam.items():
            track = self._tracks.get(cam)
            if (
                track is not None
                and track.get("session") is not None
                and track.get("shape") == frame.shape[:2]
            ):
                ready.append(cam)
        self._pv_cache = {cam: self._pv(frames_by_cam[cam]) for cam in frames_by_cam}
        if len(ready) < 2:
            return  # nothing to share
        with torch.inference_mode():
            stack = torch.stack([self._pv_cache[cam] for cam in ready])
            out = self.trk.get_image_features(stack, return_dict=True)
        for k, cam in enumerate(ready):
            sess = self._tracks[cam]["session"]
            fidx = len(sess.processed_frames or {})  # the index forward() will assign this frame
            feats = out.fpn_hidden_states[k : k + 1]  # keep batch dim = 1 (the cached shape)
            pes = [pe[k : k + 1] for pe in out.fpn_position_encoding]
            sess.cache.cache_vision_features(fidx, {"vision_feats": feats, "vision_pos_embeds": pes})

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

    def _infer_masks(self, frame_rgb: np.ndarray) -> tuple[dict[str, list[np.ndarray]], int, int]:
        """Drive the tracker for one frame and return ``(masks_by_concept, h, w)``.

        The body shared by :meth:`infer` (which composites an RGBA overlay) and
        :meth:`segment` (which returns the raw masks). Mutates per-camera tracker
        state — the caller must have selected the camera via :meth:`set_camera`.
        """
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
                # RECOVER_EVERY so the FIRST frame after a reset/scrub probes immediately;
                # a failed probe resets it, throttling subsequent attempts (below).
                "since_recover": self.RECOVER_EVERY,
                "shape": (h, w),
            }
            self._tracks[cam] = track
        pv = self._pv_cache.pop(cam, None)  # primed by _prime_batch (batched sweeps)
        if pv is None:
            pv = self._pv(frame_rgb)

        if track["session"] is None:
            # No track yet — Tier 1 detects each object to seed Tier 2. Throttled like
            # recovery: with no objects in view, an unthrottled probe re-runs the detector
            # for EVERY concept on EVERY frame (measured ~30 ms/concept/frame — it
            # dominated a live run's per-camera cost during empty-scene stretches).
            track["since_recover"] += 1
            if track["since_recover"] < self.RECOVER_EVERY:
                return self._live_masks(track), h, w
            track["since_recover"] = 0
            detected = self._detect_many(frame_rgb, self._concepts, h, w)
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
                    for c in self._concepts:
                        if track["scores"].get(c, 0.0) >= self.LOST_THRESH and c in track["masks"]:
                            seeds[c] = track["masks"][c]  # healthy: reseed from current mask
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


class Sam3VideoUnifiedAdapter(ConceptMaskAdapter):
    """SAM3 unified video pipeline (``Sam3VideoModel``): detection + tracking +
    ASSOCIATION in one model, per frame.

    Unlike :class:`Sam3TrackByDetectionAdapter` (seed once, propagate, hand-rolled
    re-seed on loss), this runs Meta's own detect-every-frame pipeline with built-in
    masklet association and keep-alive — so hard/occluded objects recover without our
    re-seed churn, and an object that leaves the frame is genuinely ABSENT (the
    two-tier tracker keeps propagating a stale mask). Measured on a real episode: the
    "wooden dowel" our two-tier lost on 56% of frames is held on EVERY frame it is
    visible, at ~49 ms/frame (672 px, 5090). Costs the detector every frame — the
    two-tier's steady-state (tracker-only) is cheaper when nothing is ever lost.

    Same weights + gating as the two-tier (``facebook/sam3``). The SAM 3.1 multiplex
    checkpoint is this architecture's successor but is not yet loadable in
    transformers (new tracker modules, no conversion) — when it is, it plugs in here
    as a checkpoint swap.

    Streaming session per camera; a session accumulates temporal memory, so
    :meth:`reset` (scrub jump / episode switch) drops the current camera's session.

    Memory bound (the reason a naive ``Sam3VideoModel`` step was banned before): the
    transformers session RETAINS every streamed frame (``processed_frames``) and every
    frame's tracker outputs, forever. Bounded here the same way the two-tier is: the
    raw frame is evicted right after its forward (nothing ever re-reads it), and the
    session is REBUILT every ``FLUSH_EVERY`` frames to drop the residual per-frame
    output growth — the pipeline re-detects + re-associates in a single frame, so a
    flush is near-seamless (unlike the two-tier, whose flush re-seeds from masks).
    """

    key = "sam3_video"
    label = "SAM3 video (unified)"
    controls = [
        {
            "type": "text",
            "key": "prompt",
            "label": "Objects",
            "placeholder": "robot arm . cylinder . green ring",
            "hint": "Period-separated objects; each is detected + associated on every frame "
            "(built-in recovery, true absence when out of view). Changing this restarts tracking.",
        }
    ]
    SAM3_ID = "facebook/sam3"
    FLUSH_EVERY = 120  # rebuild each session every N frames -> flat GPU memory on long streams
    # The pipeline's default masklet cap is 10000 (sized for 100+ object benchmarks), so a
    # scene cut can churn hundreds of masklets into VRAM between flushes (measured: OOM at
    # ~250 masklets from an N^2 mask-IoU). We track <= MAX_OBJECTS concepts — cap masklets.
    # The cap must come WITH empty-masklet decay (set at load): by default an empty (junk /
    # out-of-view) masklet's keep-alive never decreases, so junk saturates a small cap and
    # then REAL detections get rejected — measured as holds collapsing under cap 24.
    MAX_MASKLETS = 48
    EMPTY_MASKLET_DECAY = True  # reap masklets whose mask stays empty (junk); see note above

    def __init__(self, device: str = "cuda", resolution: int | None = None):
        super().__init__(device, resolution)
        try:
            import torch
            from PIL import Image
            from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._Image = Image
        logger.info("loading %s (unified video pipeline) at %d px ...", self.SAM3_ID, self.resolution)
        try:
            # Same load-time resolution contract as the two-tier adapter, applied to BOTH
            # sub-configs: the detector's global-attn RoPE and the tracker's prompt-encoder
            # grid + memory-attention RoPE are all built from config at load.
            cfg = Sam3VideoConfig.from_pretrained(self.SAM3_ID)
            cfg.detector_config.vision_config.image_size = self.resolution
            cfg.tracker_config.image_size = self.resolution
            cfg.tracker_config.memory_attention_rope_feat_sizes = [self.resolution // 14] * 2
            cfg.image_size = self.resolution
            cfg.max_num_objects = self.MAX_MASKLETS
            cfg.decrease_trk_keep_alive_for_empty_masklets = self.EMPTY_MASKLET_DECAY
            self.proc = Sam3VideoProcessor.from_pretrained(self.SAM3_ID)
            self.model = (
                Sam3VideoModel.from_pretrained(self.SAM3_ID, config=cfg, dtype=torch.float16)
                .to(device)
                .eval()
            )
        except Exception as e:
            raise RuntimeError(
                f"SAM3 weights are gated — accept the Meta SAM License at "
                f"https://huggingface.co/{self.SAM3_ID} and run `hf auth login`, then reload. ({type(e).__name__})"
            ) from e
        self._sessions: dict[str | None, dict] = {}  # per-camera {"sess", "shape"}

    def reset(self) -> None:
        # Discontinuity (scrub jump / episode switch / wrap): the session's temporal
        # memory assumes contiguous frames — drop it so the next frame starts fresh.
        self._sessions.pop(self._cam, None)

    def _restart_tracking(self) -> None:
        self._sessions = {}  # prompts live in the session -> rebuild with the new concept set

    def _session_for(self, h: int, w: int) -> dict:
        entry = self._sessions.get(self._cam)
        if entry is None or entry["shape"] != (h, w):
            sess = self.proc.init_video_session(inference_device=self.device, dtype=self._torch.float16)
            self.proc.add_text_prompt(sess, list(self._concepts))
            entry = {"sess": sess, "shape": (h, w), "frames": 0}
            self._sessions[self._cam] = entry
            logger.info("session[%s]: new, prompts=%s", self._cam or "?", self._concepts)
        return entry

    def _infer_masks(self, frame_rgb: np.ndarray) -> tuple[dict[str, list[np.ndarray]], int, int]:
        torch = self._torch
        h, w = frame_rgb.shape[:2]
        self._concepts = self._parse_concepts()
        entry = self._session_for(h, w)
        pv = self.proc(images=self._Image.fromarray(frame_rgb), size=self._proc_size, return_tensors="pt")[
            "pixel_values"
        ][0].to(self.device, torch.float16)
        with torch.inference_mode():
            out = self.model(inference_session=entry["sess"], frame=pv)
        res = self.proc.postprocess_outputs(entry["sess"], out, original_sizes=[[h, w]])
        # Bound session memory by REBUILD only — do NOT evict streamed frames from the
        # session: the tracker sizes its memory attention from len(processed_frames)
        # (num_frames -> max_object_pointers_to_use), so eviction silently lobotomises
        # it — measured as per-frame masklet churn + lost holds. A session therefore
        # grows (raw frames + per-frame outputs + masklet memory) until the rolling
        # flush drops it whole; the pipeline re-detects + re-associates in one frame.
        entry["frames"] += 1
        if entry["frames"] >= self.FLUSH_EVERY:
            self._sessions.pop(self._cam, None)
        oids = list(res.get("object_ids", []))
        masks = res.get("masks")
        p2o = res.get("prompt_to_obj_ids", {})
        masks_by_concept: dict[str, list[np.ndarray]] = {}
        for c in self._concepts:
            got: list[np.ndarray] = []
            for oid in p2o.get(c, []):
                oid = int(oid)
                if oid in oids:
                    m = masks[oids.index(oid)]
                    m = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).squeeze().astype(bool)
                    assert m.shape == (h, w), f"postprocessed mask {m.shape} != frame {(h, w)}"
                    if m.any():
                        got.append(m)
            if got and not self._seed_multi:
                got = [max(got, key=lambda a: int(a.sum()))]  # debug-lock semantics: largest only
            masks_by_concept[c] = got
        return masks_by_concept, h, w


class Sam31MultiplexAdapter(ConceptMaskAdapter):
    """SAM 3.1 (Object Multiplex) via Meta's ``sam3`` repo — the SIDECAR model.

    The multiplex tracker is not loadable in ``transformers`` (new architecture,
    no conversion), so this adapter drives Meta's own implementation. The worker
    process must therefore run in the sidecar env (``LEROBOT_SAM31_PYTHON``,
    default ``~/.cache/sam31/venv``) — a venv overlaid on the main env's
    site-packages with ``facebookresearch/sam3`` installed; the server picks that
    interpreter when spawning this model's worker.

    Incremental sessions over their offline API: Meta's OSS release only loads
    whole video files, but the model layer is per-frame — a session is
    bootstrapped from a 1-frame dummy image, its ``images`` swapped for a growing
    feeder list (their own AsyncImageFrameLoader proves loader objects are
    supported), every per-frame state list grown in lockstep, and each new frame
    consumed via a single-frame ``propagate_in_video`` call — reusing their
    pipeline, heuristics and output formatting verbatim.

    One session per (camera, concept): the OSS session stores a SINGLE
    ``text_prompt``. Cost therefore scales with cameras x concepts at the model's
    native 1008 px (measured ~13 fps for one stream on the 5090) — a QUALITY
    option (post-detection holds measured perfect on objects the two-tier loses);
    cross-session backbone sharing is the known perf follow-up. Memory is bounded
    the same way as the other stateful adapters: rolling session rebuild.
    """

    key = "sam3_1"
    label = "SAM 3.1 (multiplex, sidecar)"
    controls = [
        {
            "type": "text",
            "key": "prompt",
            "label": "Objects",
            "placeholder": "robot arm . cylinder . green ring",
            "hint": "Meta's SAM 3.1 multiplex tracker (native 1008 px). One session per object "
            "per camera — highest quality, cost scales with objects x cameras.",
        }
    ]
    FLUSH_EVERY = 120  # rebuild each (cam, concept) session -> bounded feeder + state growth
    PROB_THRESH = 0.35  # their default 0.5 detected our thin dowel ~80 frames late
    EPISODE_CHUNK = 450  # batch-session frame cap (full-video sessions of this size fit the 5090)

    def __init__(self, device: str = "cuda", resolution: int | None = None):
        # resolution is accepted for interface parity but IGNORED: Meta's builder
        # hardcodes 1008 (positional encodings precompute at that size).
        super().__init__(device, resolution)
        try:
            import torch
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError as e:
            raise RuntimeError(
                "SAM 3.1 needs the sidecar env (Meta's sam3 repo). Expected interpreter: "
                "$LEROBOT_SAM31_PYTHON (default ~/.cache/sam31/venv/bin/python) with "
                "`pip install -e ~/.cache/sam31/sam3`. See memory: reference_sam31_sidecar."
            ) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        logger.info("loading SAM 3.1 multiplex (Meta repo, SDPA) ...")
        # use_fa3=False: the fp8 FlashAttention-3 path needs flash-attn-3 (Hopper-first).
        predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
        self.model = predictor.model
        self._img_size = int(getattr(self.model, "image_size", 1008))
        mean = getattr(self.model, "image_mean", (0.5, 0.5, 0.5))
        std = getattr(self.model, "image_std", (0.5, 0.5, 0.5))
        self._mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self._sessions: dict[tuple[str | None, str], dict] = {}  # (cam, concept) -> entry
        self._batch: dict[str | None, dict] = {}  # cam -> {"masks": [...], "cursor": int}

    def reset(self) -> None:
        for key in [k for k in self._sessions if k[0] == self._cam]:
            self._sessions.pop(key, None)
        self._batch.pop(self._cam, None)

    def _restart_tracking(self) -> None:
        self._sessions = {}
        self._batch = {}

    def _preprocess(self, frame_rgb: np.ndarray):
        torch = self._torch
        img = self._cv2.resize(
            frame_rgb, (self._img_size, self._img_size), interpolation=self._cv2.INTER_LINEAR
        )
        arr = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - self._mean) / self._std
        return torch.from_numpy(arr)

    def _bootstrap(self, h: int, w: int) -> dict:
        """A fresh incremental session: init from a 1-frame dummy, swap in a growing
        feeder, and remember which per-frame state lists must grow with it."""
        import tempfile

        from PIL import Image as PILImage

        if not hasattr(self, "_dummy_path"):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                self._dummy_path = f.name
            PILImage.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(self._dummy_path)
        import copy

        state = self.model.init_state(resource_path=self._dummy_path)
        # Per-frame bookkeeping lists were sized for the 1-frame dummy; reset them to
        # empty and record (container, key, per-frame default) so append() can grow
        # them. The per-frame lists live at the top level AND inside
        # tracker_metadata / rank0_metadata — scan one nested level.
        grow: list[tuple[dict, str, object]] = []
        containers = [state]
        tm = state.get("tracker_metadata")
        if isinstance(tm, dict):
            containers.append(tm)
            r0 = tm.get("rank0_metadata")
            if isinstance(r0, dict):
                containers.append(r0)
        for cont in containers:
            for k, v in list(cont.items()):
                if isinstance(v, list) and len(v) == 1:
                    grow.append((cont, k, v[0]))
                    cont[k] = []
        feeder: list = []
        state["images"] = feeder
        state["num_frames"] = 0
        state["orig_height"], state["orig_width"] = h, w
        # input_batch was also sized for the dummy: swap its image container for the
        # same growing feeder (their NestedTensor indexes whatever backs .tensors) and
        # keep frame 0's FindStage as a template to clone per appended frame.
        ib = state["input_batch"]
        template = copy.deepcopy(ib.find_inputs[0])
        ib.img_batch = type(ib.img_batch)(tensors=feeder, mask=None)
        ib.find_inputs = []
        ib.find_targets = []
        ib.find_metadatas = []
        return {
            "state": state,
            "feeder": feeder,
            "grow": grow,
            "template": template,
            "frames": 0,
            "prompted": False,
            "shape": (h, w),
        }

    def _append(self, entry: dict, pv) -> int:
        import copy

        torch = self._torch
        st = entry["state"]
        idx = len(entry["feeder"])
        entry["feeder"].append(pv.to(st.get("device", self.device), torch.float16))
        st["num_frames"] = len(entry["feeder"])
        for cont, k, default in entry["grow"]:
            cont[k].append(copy.deepcopy(default))
        # SAM2 sub-states snapshot num_frames at creation; keep them in step (their
        # own per-frame outputs are dict-keyed, so only the scalar needs syncing).
        for sub in st.get("sam2_inference_states") or []:
            if isinstance(sub, dict) and "num_frames" in sub:
                sub["num_frames"] = st["num_frames"]
        ib = st["input_batch"]
        stage = copy.deepcopy(entry["template"] if not ib.find_inputs else ib.find_inputs[-1])
        # Point the cloned stage at THIS frame (img_ids may be tensor or list post-convert).
        if hasattr(stage.img_ids, "fill_"):
            stage.img_ids.fill_(idx)
        else:
            stage.img_ids = [idx]
        stage.img_ids_np = np.array([idx])
        ib.find_inputs.append(stage)
        ib.find_targets.append(None)
        ib.find_metadatas.append(None)
        return idx

    def _step(self, entry: dict, concept: str, idx: int):
        """Run this frame through the session; returns the formatted outputs dict."""
        torch = self._torch
        st = entry["state"]
        with torch.inference_mode():
            if not entry["prompted"]:
                out = self.model.add_prompt(st, idx, text_str=concept, output_prob_thresh=self.PROB_THRESH)
                entry["prompted"] = True
            else:
                # Their action-history parser assumes the offline flow (one prompt,
                # one full propagation) and downgrades later calls to "fetch cached
                # results" — which don't exist for a frame we just appended. Keep only
                # the prompt record so every incremental call parses as a fresh full
                # propagation of its one-frame window.
                hist = st.get("action_history")
                if isinstance(hist, list) and len(hist) > 1:
                    del hist[1:]
                out = None
                for _fidx, o in self.model.propagate_in_video(
                    inference_state=st,
                    start_frame_idx=idx,
                    max_frame_num_to_track=1,
                    reverse=False,
                    output_prob_thresh=self.PROB_THRESH,
                ):
                    out = o
        if isinstance(out, tuple):  # some paths return (frame_idx, outputs)
            out = out[-1]
        return out or {}

    def _parse_out(self, out, h: int, w: int) -> list[np.ndarray]:
        """Instance masks from one frame's formatted outputs (largest-only unless multi)."""
        m = out.get("out_binary_masks") if isinstance(out, dict) else None
        got: list[np.ndarray] = []
        if m is not None:
            arr = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            for inst in arr:
                inst = inst.astype(bool)
                assert inst.shape == (h, w), f"sam3.1 mask {inst.shape} != frame {(h, w)}"
                if inst.any():
                    got.append(inst)
        if got and not self._seed_multi:
            got = [max(got, key=lambda a: int(a.sum()))]
        return got

    def process_episode(self, frames_by_cam: dict[str, list[np.ndarray]]) -> None:
        """Offline batch mode: run Meta's native one-shot propagation per (camera,
        concept) over the whole episode and cache per-frame masks; the next
        ``len(frames)`` ``segment()``/``segment_many()`` calls per camera consume the
        cache in order. This is the fast path — one amortized propagation instead of
        per-frame calls — and matches their offline flow exactly, so tracking quality
        equals the reference implementation.

        Preconditions: control (objects) already set; frames are RGB uint8 HxWx3, all
        the same shape per camera. Episodes longer than ``EPISODE_CHUNK`` run in
        chunks with a fresh session each (a tracking reseed at each seam)."""
        torch = self._torch
        self._concepts = self._parse_concepts()
        self._batch = {}
        for cam, frames in frames_by_cam.items():
            n = len(frames)
            per_frame: list[dict[str, list[np.ndarray]]] = [{} for _ in range(n)]
            if n:
                h, w = frames[0].shape[:2]
                for concept in self._concepts:
                    for c0 in range(0, n, self.EPISODE_CHUNK):
                        chunk = frames[c0 : c0 + self.EPISODE_CHUNK]
                        try:
                            with torch.inference_mode():
                                entry = self._bootstrap(h, w)
                                for rgb in chunk:
                                    self._append(entry, self._preprocess(np.ascontiguousarray(rgb)))
                                st = entry["state"]
                                self.model.add_prompt(
                                    st, 0, text_str=concept, output_prob_thresh=self.PROB_THRESH
                                )
                                for fidx, out in self.model.propagate_in_video(
                                    inference_state=st,
                                    start_frame_idx=0,
                                    max_frame_num_to_track=None,
                                    reverse=False,
                                    output_prob_thresh=self.PROB_THRESH,
                                ):
                                    per_frame[c0 + fidx][concept] = self._parse_out(out, h, w)
                        except RuntimeError as e:
                            # Their propagate crashes on zero-object edge states (B=0
                            # expand); this chunk yields no masks for the concept.
                            logger.warning("sam3.1 batch chunk failed for %r: %s", concept, e)
                        logger.info(
                            "sam3.1 batch: cam=%s concept=%r frames %d-%d done",
                            cam,
                            concept,
                            c0,
                            c0 + len(chunk) - 1,
                        )
            self._batch[cam] = {"masks": per_frame, "cursor": 0}

    def _infer_masks(self, frame_rgb: np.ndarray) -> tuple[dict[str, list[np.ndarray]], int, int]:
        h, w = frame_rgb.shape[:2]
        self._concepts = self._parse_concepts()
        batch = self._batch.get(self._cam)
        if batch is not None and batch["cursor"] < len(batch["masks"]):
            cached = batch["masks"][batch["cursor"]]
            batch["cursor"] += 1
            return {c: cached.get(c, []) for c in self._concepts}, h, w
        pv = self._preprocess(np.ascontiguousarray(frame_rgb))
        masks_by_concept: dict[str, list[np.ndarray]] = {}
        for concept in self._concepts:
            key = (self._cam, concept)
            entry = self._sessions.get(key)
            if entry is None or entry["shape"] != (h, w) or entry["frames"] >= self.FLUSH_EVERY:
                entry = self._bootstrap(h, w)
                self._sessions[key] = entry
            idx = self._append(entry, pv)
            entry["frames"] += 1
            try:
                out = self._step(entry, concept, idx)
            except RuntimeError as e:
                # Their propagate crashes on zero-object edge states (B=0 expand);
                # rebuild next frame rather than failing the sweep.
                logger.warning("sam3.1 step failed (%s); session rebuilds next frame", e)
                self._sessions.pop(key, None)
                masks_by_concept[concept] = []
                continue
            masks_by_concept[concept] = self._parse_out(out, h, w)
        return masks_by_concept, h, w


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
    Sam3VideoUnifiedAdapter.key: Sam3VideoUnifiedAdapter,
    Sam31MultiplexAdapter.key: Sam31MultiplexAdapter,
    PolicySaliencyAdapter.key: PolicySaliencyAdapter,
}

# Text-prompted segmenters — the models valid for concept masking (data editing);
# excludes overlay-only adapters like policy saliency.
SEGMENTER_KEYS: tuple[str, ...] = tuple(
    k for k, cls in ADAPTERS.items() if issubclass(cls, ConceptMaskAdapter)
)


def python_for_model(key: str) -> str:
    """Interpreter for the subprocess that will load ``key``. SAM 3.1 lives only in
    Meta's repo, installed in a sidecar venv overlaid on the main env
    (``LEROBOT_SAM31_PYTHON``, default ``~/.cache/sam31/venv/bin/python``); every
    other model runs in the server's own interpreter.

    Raises RuntimeError (with the setup recipe) if the sidecar is required but
    missing, so callers can surface an actionable error before spawning."""
    import os
    import sys

    if key != Sam31MultiplexAdapter.key:
        return sys.executable
    py = os.environ.get("LEROBOT_SAM31_PYTHON") or os.path.expanduser("~/.cache/sam31/venv/bin/python")
    if not os.path.exists(py):
        raise RuntimeError(
            f"SAM 3.1 needs its sidecar env; interpreter not found at {py}. Create it with: "
            "python -m venv --system-site-packages ~/.cache/sam31/venv && "
            "git clone https://github.com/facebookresearch/sam3 ~/.cache/sam31/sam3 && "
            "~/.cache/sam31/venv/bin/pip install -e ~/.cache/sam31/sam3 pycocotools "
            "(or set LEROBOT_SAM31_PYTHON)."
        )
    return py


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
