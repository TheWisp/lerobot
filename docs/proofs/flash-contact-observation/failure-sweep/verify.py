"""Check saved experiment completeness, scoring integrity and gap-control parity."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = Path.home() / ".cache/flash-failure-sweep-20261029"
with gzip.open(HERE / "results.json.gz", "rt", encoding="utf-8") as stream:
    cases = json.load(stream)["cases"]
audit = {x["sequence"]: x for x in json.loads((HERE / "input_audit.json").read_text())}
assert len(cases) == 10
config_hash = hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest()
assert sum(c["frames_count"] for c in cases if not c["sequence"].endswith("-sam3")) == 1629
for case in cases:
    directory = ROOT / case["sequence"]
    meta = json.loads((directory / "provenance.json").read_text())
    manifest = json.loads((directory / "evidence/manifest.json").read_text())
    assert manifest["complete"] and manifest["frames_completed"] == 181
    assert meta["config_sha256"] == case["config_sha256"] == config_hash
    assert meta["annotated_poses_used_by_tracker"] is False
    assert sum(case["states"].values()) == 181 and len(case["frames"]) == 181
    assert len(audit[case["sequence"]]["frames"]) == 181
    for i, row in enumerate(case["frames"]):
        evidence = json.loads(gzip.decompress((directory / "evidence" / f"{i:06}.json.gz").read_bytes()))
        assert row["frame"] == i and row["source_index"] == meta["start"] + i
        assert row["state"] == evidence["estimates"]["object"]["evidence"]
        assert (row["published"] is None) == (evidence["estimates"]["object"]["pose"] is None)
        assert (row["raw"] is None) == (evidence["proposals"]["object"]["raw_pose"] is None)
        if i == 0:
            assert row["raw"]["point_mm"] < 1e-8 and row["raw"]["angle_deg"] < 1e-8
    # The scorer must exclude initialization, even though it is always observed.
    observed_score = case["errors"]["observed"]["published"]["point_mm"]
    assert (0 if observed_score is None else observed_score["n"]) == case["states"].get("observed", 0) - 1
probe = json.loads((HERE / "gap_probe.json").read_text())
for old, changed in zip(probe[0]["frames"][:-1], probe[1]["frames"][:-1], strict=True):
    assert old["frame"] == changed["frame"] and old["inliers"] == changed["inliers"]
    np.testing.assert_allclose(old["base_motion_from_seed"], changed["base_motion_from_seed"], atol=1e-12)
assert probe[0]["frames"][-1]["error"] == "causal point tracks lost a camera-frame interval"
assert probe[1]["frames"][-1]["inliers"] == 48
assert probe[1]["frames"][-1]["label_origin_error_mm"] < 2
assert probe[1]["frames"][-1]["label_angle_error_deg"] < 2
print("Complete: 9 frozen intervals + SAM3 comparison; labels scored after inference; gap control isolated.")
