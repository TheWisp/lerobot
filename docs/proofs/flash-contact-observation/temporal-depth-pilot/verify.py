"""Check saved comparisons against withheld input data, not tracker acceptance."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache/flash-depth-completion-pilot-20261029"
identities = {}
for suffix in ["", "_dense_prompt", "_lingbot_r9", "_lingbot_r0"]:
    report = json.loads((HERE / f"depth_results{suffix}.json").read_text())
    assert len(report["cases"]) == 5
    for row in report["cases"]:
        a = np.load(CACHE / f"{row['case']}{suffix}.npz")
        depth, mask, holdout = a["depth"], a["mask"], a["holdout"]
        digest = hashlib.sha256(a["rgb"].tobytes() + depth.tobytes() + mask.tobytes()).hexdigest()
        assert digest == row["input_sha256"]
        assert identities.setdefault(row["case"], digest) == digest
        assert np.all(a["withheld"][holdout] == 0) and np.all(depth[holdout] > 0)
        assert np.array_equal(a["withheld"][~holdout], depth[~holdout])
        assert np.array_equal(a["hybrid"][depth > 0], depth[depth > 0])
        idx = distance_transform_edt(a["withheld"] == 0, return_distances=False, return_indices=True)
        assert np.array_equal(a["nearest"], a["withheld"][tuple(idx)])
        valid = np.isfinite(a["withheld_prediction"]) & (a["withheld_prediction"] > 0)
        assert int(np.sum(holdout & ~valid)) == row.get("withheld_invalid_predictions", 0)
        for key, prediction, selection in [
            ("withheld_model", a["withheld_prediction"], holdout & valid),
            ("withheld_nearest", a["nearest"], holdout),
        ]:
            errors = np.abs(prediction - depth)[selection]
            assert len(errors) == row[key]["count"]
            np.testing.assert_allclose(np.median(errors) * 1000, row[key]["median_mm"], rtol=1e-6)
            np.testing.assert_allclose(np.percentile(errors, 90) * 1000, row[key]["p90_mm"], rtol=1e-6)
        assert row["forward_median_ms"] > 0
temporal = json.loads((HERE / "temporal.json").read_text())
assert [r["frame"] for r in temporal] == list(range(250, 311))
print(
    "Verified identical inputs, held-out readings, nearest baseline, sensor preservation, reported errors, and temporal completeness."
)
