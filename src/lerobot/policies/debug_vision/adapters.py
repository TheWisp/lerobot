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

    def set_control(self, control: dict) -> None:
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
            col = _color_for(str(label))
            cv2.rectangle(rgba, (x0, y0), (x1, y1), (*col, 255), 2)
            txt = f"{label} {score:.2f}"
            ty = max(12, y0 - 4)
            cv2.putText(rgba, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(rgba, txt, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (*col, 255), 1, cv2.LINE_AA)
        return rgba


class DinoFeatureAdapter(DebugVisionAdapter):
    """Frozen DINOv2 patch features -> PCA-RGB heatmap (whole-frame tint)."""

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
            "default": 0.55,
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
        self.alpha = 140  # 0..255

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


def _import_cv2():
    try:
        import cv2

        return cv2
    except ImportError as e:  # opencv is a core dep, but fail loudly if absent
        raise RuntimeError("opencv (cv2) is required for debug-vision overlays") from e


ADAPTERS: dict[str, type[DebugVisionAdapter]] = {
    GroundingDinoAdapter.key: GroundingDinoAdapter,
    DinoFeatureAdapter.key: DinoFeatureAdapter,
}


def build_adapter(key: str, device: str = "cuda") -> DebugVisionAdapter:
    if key not in ADAPTERS:
        raise ValueError(f"unknown debug-vision model '{key}'; have {list(ADAPTERS)}")
    return ADAPTERS[key](device=device)
