"""Regenerate ``coco_reference.json`` — ground truth from pycocotools itself.

Run with a throwaway environment; pycocotools is deliberately NOT a dependency
of this repo (it is a C extension built at install time, which is the trade the
codec's module docstring records):

    python -m venv /tmp/cocoenv && /tmp/cocoenv/bin/pip install pycocotools numpy
    /tmp/cocoenv/bin/python tests/datasets/regen_coco_reference.py

The codec is hand-rolled to be COCO byte-for-byte, and interop is the stated
reason for that. Every other assertion in the codec's tests is encode -> our own
decode, which an encoder and a decoder wrong in the same way both pass. This
produces the external oracle: the exact strings the reference implementation
emits.

Masks are stored by DESCRIPTION, not as pixel lists. A 720x1280 case written as
one integer per pixel is 11 MB of fixture for one mask; as the rectangles it is
made of, it is one line. The description is plain numpy slicing, so nothing
about how the mask is built borrows from the encoding under test.
"""

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

CASES: dict[str, dict] = {}


def build(spec: dict) -> np.ndarray:
    """Materialise a case's mask from its description. Mirrored in the test."""
    h, w = spec["h"], spec["w"]
    m = np.zeros((h, w), bool)
    kind = spec["fill"]
    if kind == "pixels":
        m = np.array(spec["pixels"], dtype=bool).reshape(h, w)
    elif kind == "rects":
        for r0, r1, c0, c1 in spec["rects"]:
            m[r0:r1, c0:c1] = True
    elif kind == "checker":
        m = np.indices((h, w)).sum(axis=0) % 2 == 0
    else:
        raise ValueError(kind)
    return m


def add(name: str, spec: dict) -> None:
    m = build(spec)
    enc = coco.encode(np.asfortranarray(m.astype(np.uint8)))
    CASES[name] = {**spec, "counts": enc["counts"].decode("ascii")}


h, w = 12, 9
rng = np.random.default_rng(0)
add("empty", {"h": h, "w": w, "fill": "rects", "rects": []})
add("full", {"h": h, "w": w, "fill": "rects", "rects": [[0, h, 0, w]]})
add("first_pixel", {"h": h, "w": w, "fill": "rects", "rects": [[0, 1, 0, 1]]})
add("first_row", {"h": h, "w": w, "fill": "rects", "rects": [[0, 1, 0, w]]})
add("first_column", {"h": h, "w": w, "fill": "rects", "rects": [[0, h, 0, 1]]})
add("checker", {"h": h, "w": w, "fill": "checker"})
add("blob", {"h": h, "w": w, "fill": "rects", "rects": [[3, 9, 2, 7]]})
add(
    "random",
    {"h": h, "w": w, "fill": "pixels", "pixels": (rng.random((h, w)) > 0.5).astype(int).ravel().tolist()},
)
add("two_blobs", {"h": h, "w": w, "fill": "rects", "rects": [[0, 3, 0, 2], [7, 11, 5, 9]]})
# Runs too long for a single count character -- the classic break, invisible on
# small masks.
add("wide_uniform", {"h": 720, "w": 1280, "fill": "rects", "rects": [[0, 720, 0, 1280]]})
add("wide_blob", {"h": 720, "w": 1280, "fill": "rects", "rects": [[100, 600, 200, 900]]})

out = Path(__file__).parent / "coco_reference.json"
out.write_text(json.dumps(CASES, indent=1, sort_keys=True) + "\n")
print(f"wrote {len(CASES)} cases to {out}")
