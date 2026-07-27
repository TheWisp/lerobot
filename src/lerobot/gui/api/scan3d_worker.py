"""Scan worker: an RGB frame + a text prompt → SAM3 mask → SAM 3D Objects mesh →
the scan3d cache, so the URDF viewer picks it up.

Run as a subprocess in the lerobot env (it loads SAM3 + shells out to the isolated
``~/.cache/sam3d`` venv for SAM 3D Objects + trimesh, which aren't in the main env):

    python -m lerobot.gui.api.scan3d_worker --image frame.png --prompt "cylinder"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SAM3D_PY = str(Path.home() / ".cache/sam3d/venv/bin/python")
RUN_SAM3D = str(Path.home() / ".cache/sam3d/run_sam3d.py")
SCAN_DIR = Path.home() / ".cache/huggingface/lerobot/gui/scan3d"
# Canonical SAM 3D output isn't metric; normalize the long axis to this until the
# monocular-depth (Depth-Anything) / known-size anchoring lands.
DISPLAY_LONG_AXIS_M = 0.12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    frame = np.array(Image.open(args.image).convert("RGB"))

    print(f"[scan] SAM3 segmenting {args.prompt!r} ...", flush=True)
    from lerobot.overlays.adapters import build_adapter

    # The landed overlays keep only the productionized "sam3_track" adapter; its Tier-1 detector
    # exposes masks() (added for scan3d / FoundationPose seeding) — same SAM3 model, raw bool masks.
    sam3 = build_adapter("sam3_track", device="cuda")
    sam3._det_threshold = args.threshold
    masks = sam3.masks(frame, args.prompt)
    if not masks:
        print("[scan] ERROR: nothing matched the prompt", flush=True)
        sys.exit(2)
    mask = masks[0]["mask"]
    print(f"[scan] mask {int(mask.sum())} px (score {masks[0].get('score', 0):.2f})", flush=True)

    rgb_png = SCAN_DIR / "in_rgb.png"
    mask_png = SCAN_DIR / "in_mask.png"
    Image.fromarray(frame).save(rgb_png)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_png)

    raw_glb = SCAN_DIR / "sam3d_raw.glb"
    print("[scan] SAM 3D Objects (geometry, ~seconds) ...", flush=True)
    subprocess.run(
        [SAM3D_PY, RUN_SAM3D, "--image", str(rgb_png), "--mask", str(mask_png), "--out", str(raw_glb)],
        check=True,
    )

    # Normalize to a display size + center (trimesh lives in the sam3d venv).
    out_glb = SCAN_DIR / "object.glb"
    norm = (
        "import trimesh,numpy as np;"
        f"m=trimesh.load(r'{raw_glb}',force='mesh');"
        f"m.apply_scale({DISPLAY_LONG_AXIS_M}/max(m.extents));"
        "m.apply_translation(-m.bounds.mean(0));"
        f"m.export(r'{out_glb}');"
        "print('[scan] normalized extents(m):',np.round(m.extents,3))"
    )
    subprocess.run([SAM3D_PY, "-c", norm], check=True)
    print(f"[scan] DONE -> {out_glb}", flush=True)


if __name__ == "__main__":
    main()
