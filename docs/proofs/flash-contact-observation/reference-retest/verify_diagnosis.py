"""Cross-check numerical results against independent saved baseline runs.

Run after the experiments; reads diagnostic outputs only. It does not touch the
user's datasets, tracker settings, GUI state, or robot.
"""

import json
from pathlib import Path

import numpy as np

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
current = json.loads((out / "clock_objectives.json").read_text())
baseline = json.loads((out / "clock_fits.json").read_text())
np.testing.assert_allclose(
    current["poses"]["0.0"],
    [row["fixed_pose"] for row in baseline["frames"]],
    atol=1e-9,
    rtol=1e-9,
)
for pair in current["pairs"]:
    for solution in pair["solutions"]:
        np.testing.assert_allclose(
            sum(solution["shape_terms"].values()),
            solution["cost"],
            atol=1e-10,
            rtol=1e-10,
        )
scores = json.loads((out / "clock_objective_scores.json").read_text())
assert len(scores["full_clip"]) == len(current["variants"]) == 7
assert all(len(row["frames"]) == len(baseline["frames"]) == 203 for row in scores["full_clip"].values())
assert all(len(row["frames"]) == 13 for row in scores["selected_pair_tests"].values())
assert len(current["pairs"]) == 61
assert sum(len(pair["solutions"]) for pair in current["pairs"]) == 413
for path in Path(__file__).parent.glob("*.json"):

    def reject_nonfinite(value, source_path=path):
        raise ValueError(f"{source_path}: non-finite JSON number {value}")

    json.loads(path.read_text(), parse_constant=reject_nonfinite)
print("Verified baseline equivalence, score decomposition, complete comparisons and finite stored results.")
