"""Keep compact reproducible results; source videos and large fit traces stay local."""

import gzip
import json
from collections import Counter
from pathlib import Path

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
demos = Path.home() / ".cache/huggingface/lerobot/demos"
scores = json.loads((out / "clock_scores.json").read_text())
result = {"code_revision": "4d364d228", "production_tracker_changed": False, "clock_frames": 203}
result["clock_complete_tracks"] = {
    key: {"summary": scores[key]["summary"], "frames": scores[key]["frames"]}
    for key in ["original", "fixed_first_all_patches", "previous_frame_all_patches", "rgb_motion_chain"]
}
result["clock_selected_frame_pair_search"] = [
    {
        "target": v["target"],
        "score_selected_source": v["chosen_source"],
        "score_selected": {k: v["chosen"][k] for k in ["start", "point_error_mm", "angle_error_deg", "cost"]},
        "evaluation_only_best_source": v["oracle_source"],
        "evaluation_only_best": {
            k: v["oracle"][k] for k in ["start", "point_error_mm", "angle_error_deg", "cost"]
        },
    }
    for v in scores["target_tests"]
]
result["clock_searched_pairs"] = len(scores["pair_tests"])
result["clock_searched_candidates"] = sum(len(v["solutions"]) for v in scores["pair_tests"])
result["dish_selection"] = json.loads((out / "dish.json").read_text())
result["complete_pipeline_counts"] = {}
for name, role, path in [
    ("dish_original", "object_b", Path("/tmp/flash_dish_click_evidence_20261029")),  # nosec B108: offline scratch artifact
    ("dish_all_patches", "object_b", out / "dish_allpatch_evidence"),
    ("dish_all_patches_no_depth_span_cutoff", "object_b", out / "dish_no_span_evidence"),
    (
        "mug_original",
        "object_a",
        demos / "flash_benchmark_caryx_002/flash/v1/observation-evidence/episode_000000",
    ),
    ("mug_all_patches", "object_a", out / "mug_allpatch_evidence"),
]:
    rows = [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(path.glob("*.json.gz"))]
    result["complete_pipeline_counts"][name] = {
        "frames": len(rows),
        "states": dict(Counter(v["estimates"][role]["evidence"] for v in rows)),
        "first_frame_support": rows[0]["proposals"][role]["visible_support"],
        "second_frame_support": rows[1]["proposals"][role]["visible_support"],
    }
Path(__file__).with_name("results.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    "saved results; searched",
    result["clock_searched_pairs"],
    "pairs,",
    result["clock_searched_candidates"],
    "candidates",
)
