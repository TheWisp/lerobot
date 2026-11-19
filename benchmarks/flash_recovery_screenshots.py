"""Export measured geometry for explanatory figures from the original full history."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from benchmarks.flash_motion_recovery import read_recorded_frame
from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter


def export(recording, output):
    cv2.setNumThreads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recording = Path(recording)
    settings = json.loads((recording / "config.json").read_text())["pipeline_settings"]
    wanted = {2741, 4759, 5381, 5512}
    source_rows = [json.loads(f.read_text()) for f in sorted(recording.glob("*.overlay.json"))]
    context, arrays, records = {}, {}, {}
    original = ImageConditionedFit.correct_surface_first
    correct = PoseKalmanFilter.correct

    def fit(self, proposal, previous, **kwargs):
        pose, detail = original(self, proposal, previous, **kwargs)
        n = context["frame"]
        if n in wanted:
            arrays.update(
                {
                    f"reference_{n}": self.reference.copy(),
                    f"seed_{n}": proposal.copy(),
                    f"prior_{n}": previous.copy(),
                    f"target_{n}": self.target.copy(),
                }
            )
            if pose is not None:
                arrays[f"fit_{n}"] = pose.copy()
            for kind, candidate in [("prior", previous.copy()), ("center", previous.copy())]:
                if kind == "center":
                    from lerobot.policies.flash.surface_observation import move

                    candidate[:3, 3] += np.median(self.observation["obs"]["points"], 0) - np.median(
                        move(self.reference, candidate), 0
                    )
                alternative, _ = original(self, candidate, previous, **kwargs)
                if alternative is not None:
                    arrays[f"alternative_{kind}_{n}"] = alternative
        return pose, detail

    def fuse(self, *args, **kwargs):
        result = correct(self, *args, **kwargs)
        if context["frame"] in wanted:
            arrays[f"fused_{context['frame']}"] = self.pose.copy()
        return result

    with (
        SearchPool(settings["config"]["shape"]["batch_query_workers"]) as pool,
        patch.object(ImageConditionedFit, "correct_surface_first", fit),
        patch.object(PoseKalmanFilter, "correct", fuse),
    ):
        for ref, last in [(2607, 2741), (4173, 4759), (5118, 5512)]:
            source = type("RecordedMasks", (), {"prompts": {"object": "selected object"}})()
            live = LiveSurfaceSession(settings, source, kernels=compiled_kernels(), search_pool=pool)
            try:
                for saved in source_rows:
                    n = saved["frame"]
                    if saved["objects"]["object"]["diagnostics"].get("reference_frame") != ref or n > last:
                        continue
                    context["frame"] = n
                    frame, masks = read_recorded_frame(recording, n)
                    row, matches, _ = live.step(frame, measured_masks=(n, masks, 0.0))
                    if n in wanted:
                        arrays.update(
                            {f"rgb_{n}": frame.rgb, f"k_{n}": frame.intrinsics, f"mask_{n}": masks["object"]}
                        )
                        records[str(n)] = {"row": row, "matches": matches, "saved": saved}
                        print("captured figure evidence", n, flush=True)
            finally:
                live.close()
    np.savez_compressed(output / "figure-evidence.npz", **arrays)
    (output / "figure-evidence.json").write_text(json.dumps(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.recording, args.output)
