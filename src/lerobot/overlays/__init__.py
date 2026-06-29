"""Debug-vision: run representation-layer models (open-vocab detection,
feature visualization, ...) alongside teleop and stream per-frame visual
overlays back to the GUI camera view.

Frame ingress reuses the always-on ObservationStream (any process can
attach read-only); overlay egress is a dedicated RGBA SharedOverlayBuffer
served by the GUI backend as PNG and composited on a <canvas> over each
camera <img>. Each model is one DebugVisionAdapter (frame -> RGBA overlay).

Model weights are public Hugging Face checkpoints fetched into the standard
HF cache on first load (no bundled/temp artifacts) — see adapters.py.
"""
