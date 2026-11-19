"""Replay group membership using the pre-fix fitter, without changing working-tree code."""

import argparse
import ast
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from benchmarks.flash_motion_recovery import read_recorded_frame
from lerobot.policies.flash import surface_grounding
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.object_group_association import ObjectGroupAssociation
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(
        description="Inspect retained object members in the baseline recovery replay."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.recording
    settings = json.loads((root / "config.json").read_text())["pipeline_settings"]
    old = ast.parse(
        subprocess.check_output(
            ["git", "show", "9ff25cf80:src/lerobot/policies/flash/surface_grounding.py"], text=True
        )
    )
    cls = next(x for x in old.body if isinstance(x, ast.ClassDef) and x.name == "WholeSurfaceFit")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "fit_once")
    ns = {}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "<baseline-fit-once>", "exec"),
        vars(surface_grounding),
        ns,
    )  # Execute only the fixed local baseline method, not recording content.
    query = ObjectGroupAssociation.query
    report = []

    def audit(self, snapshot, frame, scene, mask, reference_frame, **kwargs):
        hint, details = query(self, snapshot, frame, scene, mask, reference_frame, **kwargs)
        if frame.frame_id >= 5365:
            g = snapshot.groups.get(self._group)
            if g:
                known = {int(i) for i in g.ids if int(i) in self.anchors}
                measured = {int(i) for i, b in zip(g.ids, g.measured, strict=True) if b}
                scene_ids = set(scene.ids.tolist())
                rows = []
                for i in sorted(known):
                    j = np.flatnonzero(scene.ids == i)
                    rows.append(
                        {
                            "id": i,
                            "scene_measurement": bool(len(j)),
                            "group_inlier": i in measured,
                            "uv": scene.xy[j[0]].tolist() if len(j) else None,
                        }
                    )
                r = {
                    "frame": frame.frame_id,
                    "details": details,
                    "known_points": rows,
                    "known_with_scene_measurements": len(known & scene_ids),
                    "known_group_inliers": len(known & measured),
                }
                report.append(r)
                print(
                    frame.frame_id,
                    "known",
                    len(known),
                    "measured in scene",
                    len(known & scene_ids),
                    "group inliers",
                    len(known & measured),
                    flush=True,
                )
        return hint, details

    source = type("SavedMasks", (), {"prompts": {"object": "selected object"}})()
    with (
        SearchPool(settings["config"]["shape"]["batch_query_workers"]) as pool,
        patch.object(surface_grounding.WholeSurfaceFit, "fit_once", ns["fit_once"]),
        patch.object(ObjectGroupAssociation, "query", audit),
    ):
        live = LiveSurfaceSession(settings, source, kernels=compiled_kernels(), search_pool=pool)
        try:
            for p in sorted(root.glob("*.overlay.json")):
                saved = json.loads(p.read_text())
                n = saved["frame"]
                if saved["objects"]["object"]["diagnostics"].get("reference_frame") != 5118 or n > 5381:
                    continue
                f, m = read_recorded_frame(root, n)
                live.step(f, measured_masks=(n, m, 0.0))
        finally:
            live.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
