"""Regenerate ``mask_codec_fixture.json`` from the Python encoder.

The fixture exists so the JS decoder is checked against the encoder that
actually writes datasets, rather than against a second hand-written spec. That
only holds while the fixture is generated — a hand-edited one would let the two
drift apart without any test going red. Run this after changing the encoding:

    python tests/gui/regen_mask_fixture.py

Cases are chosen for what breaks run-length coders, not for coverage of typical
masks: uniform masks produce no value changes yet must still cover every pixel,
a checkerboard makes every pixel its own run, and first_row / first_column are
indistinguishable unless the column-major convention is right.
"""

import json
from pathlib import Path

import numpy as np

from lerobot.datasets.mask_codec import encode_mask


def _case(name: str, mask: np.ndarray) -> dict:
    return {
        "name": name,
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "counts": encode_mask(mask),
        "flat_rowmajor": mask.astype(int).ravel().tolist(),
    }


def build() -> list[dict]:
    rng = np.random.default_rng(7)
    first_column = np.zeros((6, 5), bool)
    first_column[:, 0] = True
    first_row = np.zeros((6, 5), bool)
    first_row[0, :] = True
    blob = np.zeros((9, 7), bool)
    blob[2:5, 1:4] = True
    return [
        _case("empty", np.zeros((6, 5), bool)),
        _case("full", np.ones((6, 5), bool)),
        _case("first_column", first_column),
        _case("first_row", first_row),
        _case("checker", np.indices((6, 5)).sum(0) % 2 == 0),
        _case("blob", blob),
        _case("random", rng.random((13, 11)) > 0.5),
    ]


if __name__ == "__main__":
    out = Path(__file__).parent / "mask_codec_fixture.json"
    out.write_text(json.dumps(build()))
    print(f"wrote {len(build())} fixtures to {out}")
