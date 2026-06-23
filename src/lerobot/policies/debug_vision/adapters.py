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
import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Public HF checkpoints. Pinned by id so a fresh env fetches the same weights.
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"
DINOV2_ID = "facebook/dinov2-base"

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

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return an HxWx4 RGBA uint8 overlay sized to frame_rgb. Precondition:
        frame_rgb is contiguous HxWx3 uint8 RGB."""
        raise NotImplementedError


class GroundingDinoAdapter(DebugVisionAdapter):
    """Open-vocabulary detection: free-text prompt -> boxes + labels."""

    key = "grounding_dino"
    label = "Grounding DINO — open-vocab boxes"
    controls = [
        {
            "type": "text",
            "key": "prompt",
            "label": "Prompt",
            "placeholder": "cup . bottle . hand .",
            "hint": "Lowercase, period-separated. Only list objects actually in view — "
            "Grounding DINO grounds every phrase onto something, so spurious phrases get mislabeled.",
        }
    ]
    # Keep neutral: listing objects that aren't present makes the model force the
    # label onto the nearest blob. Users edit this per scene.
    DEFAULT_PROMPT = "cup . bottle . hand ."

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        logger.info("loading %s ...", GROUNDING_DINO_ID)
        self.processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID).to(device).eval()
        self.prompt = self.DEFAULT_PROMPT
        self.box_threshold = 0.30
        self.text_threshold = 0.25
        self._colors: dict[str, tuple[int, int, int]] = {}  # per-concept color (objects selector)

    def set_control(self, control: dict) -> None:
        names, colors, signs = _parse_objects(control, 6)
        if names is not None:
            # Grounding DINO grounds one period-separated string in a single pass.
            # Negative (-) concepts are simply dropped from the prompt (not detected).
            pos = [n for n in names if signs.get(n, "+") == "+"]
            self.prompt = (" . ".join(pos) + " .") if pos else "."
            self._colors = {n.lower(): c for n, c in colors.items()}
        else:
            p = control.get("prompt")
            if isinstance(p, str) and p.strip():
                self.prompt = p.strip()
        if "box_threshold" in control:
            self.box_threshold = float(control["box_threshold"])
        if "text_threshold" in control:
            self.text_threshold = float(control["text_threshold"])

    def _post_process(self, out, input_ids, hw):
        # transformers renamed box_threshold -> threshold across versions.
        try:
            return self.processor.post_process_grounded_object_detection(
                out,
                input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[hw],
            )[0]
        except TypeError:
            return self.processor.post_process_grounded_object_detection(
                out,
                input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[hw],
            )[0]

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch, cv2 = self._torch, self._cv2
        h, w = frame_rgb.shape[:2]
        text = self.prompt.lower()
        if not text.endswith("."):
            text += " ."
        inputs = self.processor(images=Image.fromarray(frame_rgb), text=text, return_tensors="pt").to(
            self.device
        )
        # autocast (not a blanket .half()): GroundingDINO's BERT text encoder
        # errors with mat1/mat2 dtype mismatch under model.half().
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = self.model(**inputs)
        res = self._post_process(out, inputs["input_ids"], (h, w))

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        boxes = res["boxes"].cpu().numpy()
        labels = res.get("text_labels") or res.get("labels") or ["?"] * len(boxes)
        scores = res["scores"].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores, strict=False):
            x0, y0, x1, y1 = (int(v) for v in box)
            col = self._colors.get(str(label).strip().lower(), _color_for(str(label)))
            cv2.rectangle(rgba, (x0, y0), (x1, y1), (*col, 255), 2)
            txt = f"{label} {score:.2f}"
            ty = max(12, y0 - 4)
            cv2.putText(rgba, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(rgba, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (*col, 255), 1, cv2.LINE_AA)
        return rgba


class DinoFeatureAdapter(DebugVisionAdapter):
    """Frozen DINOv2 patch features -> PCA-RGB heatmap (opaque whole-frame repaint)."""

    key = "dino_features"
    label = "DINOv2 — feature heatmap (PCA)"
    controls = [
        {
            "type": "range",
            "key": "alpha",
            "label": "Overlay opacity",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 1.0,
        }
    ]
    _PATCH = 14  # DINOv2 patch size

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        logger.info("loading %s ...", DINOV2_ID)
        self.processor = AutoImageProcessor.from_pretrained(DINOV2_ID)
        self.model = AutoModel.from_pretrained(DINOV2_ID).to(device).half().eval()
        self.alpha = 255  # 0..255; opaque repaint by default (dense heatmap, not a tint)

    def set_control(self, control: dict) -> None:
        if "alpha" in control:
            a = float(control["alpha"])
            self.alpha = int(a * 255) if a <= 1.0 else int(a)

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch, cv2 = self._torch, self._cv2
        h, w = frame_rgb.shape[:2]
        inp = self.processor(images=Image.fromarray(frame_rgb), return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].half()
        with torch.inference_mode():
            out = self.model(**inp)
        ph = inp["pixel_values"].shape[-2] // self._PATCH
        pw = inp["pixel_values"].shape[-1] // self._PATCH
        tokens = out.last_hidden_state[0, 1 : 1 + ph * pw].float()  # drop CLS
        _, _, v = torch.pca_lowrank(tokens, q=3)
        proj = tokens @ v
        proj = (proj - proj.min(0).values) / (proj.max(0).values - proj.min(0).values + 1e-6)
        grid = (proj.reshape(ph, pw, 3).cpu().numpy() * 255).astype(np.uint8)
        heat = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)  # RGB
        alpha = np.full((h, w, 1), self.alpha, dtype=np.uint8)
        return np.concatenate([heat, alpha], axis=2)


class DepthAnythingAdapter(DebugVisionAdapter):
    """Depth Anything V2 monocular depth → colormapped whole-frame heatmap."""

    key = "depth_anything"
    label = "Depth Anything V2 — depth heatmap"
    controls = [
        {
            "type": "range",
            "key": "alpha",
            "label": "Overlay opacity",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 1.0,
        }
    ]
    DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        logger.info("loading %s ...", self.DEPTH_ID)
        self.processor = AutoImageProcessor.from_pretrained(self.DEPTH_ID)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(self.DEPTH_ID, dtype=torch.float16).to(device).eval()
        )
        self.alpha = (
            255  # opaque: repaint the frame with the depth map; lower the slider to blend RGB back in
        )
        self._dmin: float | None = None  # EMA'd depth range (anti-flicker)
        self._dmax: float | None = None

    def set_control(self, control: dict) -> None:
        if "alpha" in control:
            a = float(control["alpha"])
            self.alpha = int(a * 255) if a <= 1.0 else int(a)

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch, cv2 = self._torch, self._cv2
        h, w = frame_rgb.shape[:2]
        inp = self.processor(images=Image.fromarray(frame_rgb), return_tensors="pt").to(
            self.device, torch.float16
        )
        with torch.inference_mode():
            out = self.model(**inp)
        depth = self.processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]["predicted_depth"]
        depth = depth.float().cpu().numpy()
        lo, hi = float(depth.min()), float(depth.max())
        # EMA the min/max so the colormap doesn't flicker frame-to-frame.
        self._dmin = lo if self._dmin is None else 0.9 * self._dmin + 0.1 * lo
        self._dmax = hi if self._dmax is None else 0.9 * self._dmax + 0.1 * hi
        d = np.clip((depth - self._dmin) / (self._dmax - self._dmin + 1e-6), 0, 1)
        bgr = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        alpha = np.full((h, w, 1), self.alpha, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=2)


class Sam2MaskAdapter(DebugVisionAdapter):
    """SAM2.1 promptable segmentation. No click channel yet → auto center-point mask."""

    key = "sam2_mask"
    label = "SAM2.1 — segment (center point)"
    controls: list[dict] = []
    SAM2_ID = "facebook/sam2.1-hiera-small"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        logger.info("loading %s ...", self.SAM2_ID)
        self.processor = Sam2Processor.from_pretrained(self.SAM2_ID)
        self.model = Sam2Model.from_pretrained(self.SAM2_ID).to(device).eval()

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch, cv2 = self._torch, self._cv2
        h, w = frame_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        inp = self.processor(
            images=Image.fromarray(frame_rgb),
            input_points=[[[[cx, cy]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            out = self.model(**inp)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])  # (num, H, W)
        scores = out.iou_scores.reshape(-1)
        mask = masks[int(scores.argmax())].numpy() > 0
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask] = (*_color_for("seg"), 120)
        cv2.drawMarker(rgba, (cx, cy), (255, 255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        return rgba


class Sam3TrackerAdapter(DebugVisionAdapter):
    """SAM 3.1 tracker — geometric (click/box) promptable segmentation, SAM2 lineage.

    Unlike :class:`Sam3Adapter` (text concepts), this is the geometric head: a point
    or box seeds a single high-quality mask. The adapter contract has no click channel
    yet, so we seed the frame CENTER (same stopgap as :class:`Sam2MaskAdapter`); this
    is the SAM 3.1 upgrade of that adapter and the backbone for the planned click
    picker. Stateless per frame — the video-memory variant (``Sam3TrackerVideoModel``)
    is a later step once a click can be seeded once and propagated.

    GATED weights — accept the Meta SAM License at https://huggingface.co/facebook/sam3
    (+ ``hf auth login``) first.
    """

    key = "sam3_tracker"
    label = "SAM 3.1 tracker — segment (center point)"
    controls: list[dict] = []
    # The published standalone tracker checkpoints (danelcsb/sam3_tracker.1_*,
    # facebook/sam3_tracker.1-*) referenced in the transformers docs do not exist on
    # the Hub yet (404). SAM3 is a unified model, so the tracker head ships inside the
    # gated facebook/sam3 weights — Sam3TrackerModel's checkpoint-conversion mapping
    # pulls it out (strips the tracker_model. prefix, ignores the memory/detector keys).
    SAM3_ID = "facebook/sam3"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        logger.info("loading %s (tracker) ...", self.SAM3_ID)
        try:
            self.processor = Sam3TrackerProcessor.from_pretrained(self.SAM3_ID)
            self.model = Sam3TrackerModel.from_pretrained(self.SAM3_ID, dtype=torch.float16).to(device).eval()
        except Exception as e:
            raise RuntimeError(
                f"SAM3 weights are gated — accept the Meta SAM License at "
                f"https://huggingface.co/{self.SAM3_ID} and run `hf auth login`, then reload. ({type(e).__name__})"
            ) from e

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch, cv2 = self._torch, self._cv2
        h, w = frame_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        inp = self.processor(
            images=Image.fromarray(frame_rgb),
            input_points=[[[[cx, cy]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to(self.device)
        # fp16 model: match the float inputs (pixel_values + point coords). Coords here
        # are << 2048, so fp16 represents them exactly — no localization error.
        for k, v in list(inp.items()):
            if torch.is_tensor(v) and v.dtype == torch.float32:
                inp[k] = v.half()
        with torch.inference_mode():
            out = self.model(**inp, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])  # (num, H, W)
        scores = out.iou_scores.reshape(-1)
        mask = masks[int(scores.argmax())].numpy() > 0
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        col = _color_for("seg")
        rgba[mask] = (*col, 120)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgba, cnts, -1, (*col, 255), 2)
        cv2.drawMarker(rgba, (cx, cy), (255, 255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        return rgba


class Sam3Adapter(DebugVisionAdapter):
    """SAM3 text/concept-promptable instance segmentation. GATED weights — accept the
    Meta SAM License at https://huggingface.co/facebook/sam3 (+ `hf auth login`) first."""

    key = "sam3"
    label = "SAM3 — text-prompt masks (gated)"
    controls = [
        {
            "type": "text",
            "key": "prompt",
            "label": "Concept",
            "placeholder": "red cube",
            "hint": "A short noun phrase; returns masks for ALL matching instances.",
        }
    ]
    SAM3_ID = "facebook/sam3"
    DEFAULT_PROMPT = "object"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        logger.info("loading %s ...", self.SAM3_ID)
        try:
            self.processor = Sam3Processor.from_pretrained(self.SAM3_ID)
            self.model = Sam3Model.from_pretrained(self.SAM3_ID, dtype=torch.float16).to(device).eval()
        except Exception as e:
            raise RuntimeError(
                f"SAM3 weights are gated — accept the Meta SAM License at "
                f"https://huggingface.co/{self.SAM3_ID} and run `hf auth login`, then reload. ({type(e).__name__})"
            ) from e
        self.prompt = self.DEFAULT_PROMPT
        self.threshold = 0.5
        self._concepts: list[str] = [self.DEFAULT_PROMPT]
        self._colors: dict[str, tuple[int, int, int]] = {}
        self._signs: dict[str, str] = {}
        self._bg_color: tuple[int, int, int] | None = None

    def set_control(self, control: dict) -> None:
        names, colors, signs = _parse_objects(control, 6)
        if names is not None:
            self._concepts = names
            self._colors = colors
            self._signs = signs
            self.prompt = names[0]  # keep self.prompt in sync for masks() / fallback
        else:
            p = control.get("prompt")
            if isinstance(p, str) and p.strip():
                self.prompt = p.strip()
                self._concepts = [self.prompt]
                self._signs = {self.prompt: "+"}
        bg = _parse_background(control)
        if bg is not _BG_UNSET:
            self._bg_color = bg
        if "threshold" in control:
            self.threshold = float(control["threshold"])

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch, cv2 = self._torch, self._cv2
        h, w = frame_rgb.shape[:2]
        # Encode the image ONCE, then run each concept reusing the vision embeds: the
        # ~1008² ViT (the dominant cost) runs once no matter how many objects.
        img_inp = self.processor(images=Image.fromarray(frame_rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vision_embeds = self.model.get_vision_features(pixel_values=img_inp["pixel_values"])
        masks_by_concept: dict[str, list[np.ndarray]] = {}
        for concept in self._concepts:
            txt = self.processor(text=concept, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(
                    vision_embeds=vision_embeds,
                    input_ids=txt["input_ids"],
                    attention_mask=txt.get("attention_mask"),
                )
            res = self.processor.post_process_instance_segmentation(
                out, threshold=self.threshold, target_sizes=[(h, w)]
            )[0]
            masks_by_concept[concept] = [
                (mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)) > 0
                for mask in res.get("masks", [])
            ]
        return _composite_concepts(
            h, w, masks_by_concept, self._concepts, self._colors, self._signs, self._bg_color, cv2
        )

    def masks(self, frame_rgb: np.ndarray, prompt: str | None = None) -> list[dict]:
        """Raw instance masks for ``prompt`` (defaults to the current prompt).

        Returns ``[{"mask": (H,W) bool, "score": float}, ...]`` for every matching
        instance, highest score first ([] if nothing matches). The reconstruction
        path needs the boolean masks, not the RGBA overlay :meth:`infer` draws.
        """
        from PIL import Image

        torch = self._torch
        if prompt is not None and prompt.strip():
            self.prompt = prompt.strip()
        h, w = frame_rgb.shape[:2]
        inp = self.processor(images=Image.fromarray(frame_rgb), text=self.prompt, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            out = self.model(**inp)
        res = self.processor.post_process_instance_segmentation(
            out, threshold=self.threshold, target_sizes=[(h, w)]
        )[0]
        masks = res.get("masks", [])
        scores = res.get("scores", [1.0] * len(masks))
        items = []
        for m, s in zip(masks, scores, strict=True):
            arr = (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)) > 0
            items.append({"mask": arr, "score": float(s.cpu() if hasattr(s, "cpu") else s)})
        items.sort(key=lambda it: it["score"], reverse=True)
        return items


class CoTracker3Adapter(DebugVisionAdapter):
    """CoTracker3 online point tracking — auto grid, streaming. Points + short trails.

    Stateful: buffers step*2 frames, seeds a regular grid once (no clicks), then
    advances the online window. Code + weights come from torch.hub (standard cache).
    """

    key = "cotracker3"
    label = "CoTracker3 — point tracks (auto grid)"
    controls = [
        {"type": "range", "key": "grid", "label": "Grid size", "min": 5, "max": 20, "step": 1, "default": 10}
    ]

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        from collections import deque

        import torch

        self._torch = torch
        self._cv2 = _import_cv2()
        self._deque = deque
        logger.info("loading CoTracker3 (online) via torch.hub ...")
        # torch.hub fetches code + weights into the standard cache. For strict
        # fresh-env reproducibility, pin the cotracker pip dep instead (see extra).
        self.model = (
            torch.hub.load("facebookresearch/co-tracker", "cotracker3_online", trust_repo=True)
            .to(device)
            .eval()
        )
        self.step = int(getattr(self.model, "step", 4))
        self.grid_size = 10
        self._buf: list = []
        self._since = 0
        self._seeded = False
        self._tracks = None
        self._vis = None
        self._trails: dict = {}

    def set_control(self, control: dict) -> None:
        if "grid" in control:
            g = int(float(control["grid"]))
            if g != self.grid_size:
                self.grid_size = g
                self._seeded = False  # reseed at the new density
                self._buf.clear()
                self._trails.clear()

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = frame_rgb.shape[:2]
        t = torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).float().to(self.device)
        self._buf.append(t)
        win = self.step * 2
        if len(self._buf) > win:
            self._buf.pop(0)
        self._since += 1
        if len(self._buf) == win and (not self._seeded or self._since >= self.step):
            chunk = torch.stack(self._buf)[None]  # (1, win, C, H, W)
            with torch.inference_mode():
                if not self._seeded:
                    self.model(chunk, is_first_step=True, grid_size=self.grid_size)
                    self._seeded = True
                else:
                    tracks, vis = self.model(chunk)
                    self._tracks = tracks[0, -1].cpu().numpy()
                    self._vis = vis[0, -1].cpu().numpy().reshape(-1) > 0.5
            self._since = 0
        return self._draw(h, w)

    def _draw(self, h: int, w: int) -> np.ndarray:
        cv2 = self._cv2
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if self._tracks is None:
            return rgba
        for i, (x, y) in enumerate(self._tracks):
            if self._vis is not None and i < len(self._vis) and not self._vis[i]:
                continue
            xi, yi = int(x), int(y)
            trail = self._trails.setdefault(i, self._deque(maxlen=8))
            trail.append((xi, yi))
            col = _color_for(f"pt{i % 16}")
            pts = list(trail)
            for j in range(1, len(pts)):
                cv2.line(rgba, pts[j - 1], pts[j], (*col, 160), 1, cv2.LINE_AA)
            cv2.circle(rgba, (xi, yi), 3, (*col, 255), -1)
        return rgba


class Sam3VideoAdapter(DebugVisionAdapter):
    """SAM3 VIDEO tracking — concept masks that PERSIST across frames (hold through
    partial occlusion), each concept in its own stable color with an on-mask label
    and a legend. The prompt is period-separated ("robot arm . cylinder . green ring")
    and every concept is tracked independently.

    Stateful: keeps a streaming session, so unlike the per-frame ``sam3`` model the
    masks don't flicker and survive occlusion. Changing the prompt (or the camera
    resolution) restarts tracking; like CoTracker3 it assumes a single camera stream.
    GATED weights — accept the Meta SAM License at https://huggingface.co/facebook/sam3
    (+ ``hf auth login``) first.
    """

    key = "sam3_video"
    label = "SAM3 video — tracked masks (gated)"
    controls = [
        {
            "type": "text",
            "key": "prompt",
            "label": "Concepts",
            "placeholder": "robot arm . cylinder . green ring",
            "hint": "Period-separated concepts; each is tracked across frames in its own color "
            "(legend, top-left). Changing this restarts tracking.",
        }
    ]
    SAM3_ID = "facebook/sam3"
    DEFAULT_PROMPT = "object"
    MAX_OBJECTS = 6  # cap monitored concepts (shared encoder keeps multi-concept cheap)

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        try:
            import torch
            from PIL import Image
            from transformers import Sam3VideoModel, Sam3VideoProcessor
        except ImportError as e:
            raise RuntimeError(_IMPORT_HINT) from e
        self._torch = torch
        self._cv2 = _import_cv2()
        self._Image = Image
        logger.info("loading %s (video) ...", self.SAM3_ID)
        try:
            self.processor = Sam3VideoProcessor.from_pretrained(self.SAM3_ID)
            self.model = Sam3VideoModel.from_pretrained(self.SAM3_ID, dtype=torch.float16).to(device).eval()
        except Exception as e:
            raise RuntimeError(
                f"SAM3 weights are gated — accept the Meta SAM License at "
                f"https://huggingface.co/{self.SAM3_ID} and run `hf auth login`, then reload. ({type(e).__name__})"
            ) from e
        self.prompt = self.DEFAULT_PROMPT
        self._concepts: list[str] = []
        self._colors: dict[str, tuple[int, int, int]] = {}  # user-chosen color per concept
        self._signs: dict[str, str] = {}
        self._bg_color: tuple[int, int, int] | None = None
        # One tracking session (memory bank) PER camera. Sharing a single session across
        # views interleaves incompatible viewpoints into one memory bank and corrupts
        # tracking — the whole point of the video model is per-stream temporal memory.
        self._cam: str | None = None
        self._sessions: dict[str | None, object] = {}
        self._shapes: dict[str | None, tuple[int, int]] = {}

    def _parse_concepts(self) -> list[str]:
        parts = (c.strip() for c in self.prompt.replace(",", ".").split("."))
        names = list(dict.fromkeys(c for c in parts if c))[: self.MAX_OBJECTS]
        return names or [self.DEFAULT_PROMPT]

    def _start_session(self, cam: str | None, h: int, w: int) -> None:
        self._concepts = self._parse_concepts()
        session = self.processor.init_video_session(inference_device=self.device, dtype=self._torch.float16)
        self.processor.add_text_prompt(session, self._concepts)
        self._sessions[cam] = session
        self._shapes[cam] = (h, w)

    def set_camera(self, cam: str | None) -> None:
        self._cam = cam  # which camera's tracking session infer() should use

    def set_control(self, control: dict) -> None:
        # Structured monitored objects (preferred). Color/sign/background are display-
        # only (applied at composite time); only a concept-NAME change restarts tracking.
        names, colors, signs = _parse_objects(control, self.MAX_OBJECTS)
        if names is not None:
            self._colors = colors
            self._signs = signs
            new_prompt = " . ".join(names)
            if new_prompt and new_prompt != self.prompt:
                self.prompt = new_prompt
                self._sessions = {}  # restart tracking on every camera with the new concepts
        else:
            p = control.get("prompt")
            if isinstance(p, str) and p.strip() and p.strip() != self.prompt:
                self.prompt = p.strip()
                self._sessions = {}
        bg = _parse_background(control)
        if bg is not _BG_UNSET:
            self._bg_color = bg

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = frame_rgb.shape[:2]
        cam = self._cam
        if self._sessions.get(cam) is None or self._shapes.get(cam) != (h, w):
            self._start_session(cam, h, w)
        session = self._sessions[cam]
        inp = self.processor(images=self._Image.fromarray(frame_rgb), return_tensors="pt")
        pv = inp["pixel_values"][0].to(self.device, torch.float16)
        with torch.inference_mode():
            out = self.model(inference_session=session, frame=pv)
        res = self.processor.postprocess_outputs(session, out, original_sizes=[[h, w]])
        obj_to_concept = {oid: p for p, oids in res["prompt_to_obj_ids"].items() for oid in oids}
        masks = res["masks"]
        masks_by_concept: dict[str, list[np.ndarray]] = {c: [] for c in self._concepts}
        for k, oid in enumerate(res["object_ids"].tolist()):
            arr = masks[k]
            m = (arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)) > 0
            masks_by_concept.setdefault(obj_to_concept.get(oid, "?"), []).append(m)
        return _composite_concepts(
            h, w, masks_by_concept, self._concepts, self._colors, self._signs, self._bg_color, self._cv2
        )


def _import_cv2():
    try:
        import cv2

        return cv2
    except ImportError as e:  # opencv is a core dep, but fail loudly if absent
        raise RuntimeError("opencv (cv2) is required for debug-vision overlays") from e


ADAPTERS: dict[str, type[DebugVisionAdapter]] = {
    GroundingDinoAdapter.key: GroundingDinoAdapter,
    DinoFeatureAdapter.key: DinoFeatureAdapter,
    DepthAnythingAdapter.key: DepthAnythingAdapter,
    Sam2MaskAdapter.key: Sam2MaskAdapter,
    Sam3TrackerAdapter.key: Sam3TrackerAdapter,
    Sam3Adapter.key: Sam3Adapter,
    Sam3VideoAdapter.key: Sam3VideoAdapter,
    CoTracker3Adapter.key: CoTracker3Adapter,
}


def build_adapter(key: str, device: str = "cuda") -> DebugVisionAdapter:
    if key not in ADAPTERS:
        raise ValueError(f"unknown debug-vision model '{key}'; have {list(ADAPTERS)}")
    return ADAPTERS[key](device=device)
