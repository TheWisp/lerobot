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

from lerobot.datasets.mask_codec import decode_frame, encode_frame, encode_mask


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


def build_frames() -> list[dict]:
    """Frame-level rows, for the enabled flag rather than the run lengths.

    `expect_drawn` is derived from ``decode_frame``'s own answer, not written
    out by hand -- so the JS decision is compared against what the trainer
    actually composites, which is the disagreement that matters.
    """
    labels = ["tray", "ball"]
    tray = np.zeros((6, 5), bool)
    tray[1:4, 1:4] = True
    ball = np.zeros((6, 5), bool)
    ball[5, 0:2] = True
    both = {"tray": tray, "ball": ball}

    def _case(name, row):
        return {
            "name": name,
            "row": json.loads(row),
            "expect_drawn": sorted(labels.index(n) for n in decode_frame(row, labels, (6, 5))),
        }

    cases = [
        _case("both_enabled", encode_frame(both, labels)),
        _case("one_disabled", encode_frame(both, labels, disabled=["tray"])),
        _case("all_disabled", encode_frame(both, labels, disabled=labels)),
        _case("found_nothing", encode_frame({}, labels)),
    ]
    # Forms this encoder never writes but every reader must accept: an explicit
    # enabled flag, as an int and as a JSON bool. Hand-built for that reason.
    explicit = json.loads(encode_frame(both, labels))
    cases.append({"name": "explicit_one", "row": [[e[0], e[1], 1] for e in explicit], "expect_drawn": [0, 1]})
    cases.append(
        {"name": "explicit_true", "row": [[e[0], e[1], True] for e in explicit], "expect_drawn": [0, 1]}
    )
    return cases


if __name__ == "__main__":
    here = Path(__file__).parent
    # Trailing newline, or the end-of-files hook adds one and every regen shows
    # a spurious diff against the file it just wrote.
    (here / "mask_codec_fixture.json").write_text(json.dumps(build()) + "\n")
    (here / "mask_frame_fixture.json").write_text(json.dumps(build_frames()) + "\n")
    print(f"wrote {len(build())} mask and {len(build_frames())} frame fixtures to {here}")
