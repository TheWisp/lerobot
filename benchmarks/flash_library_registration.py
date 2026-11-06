"""Run the shared pipeline with an optional public-library surface matcher.

Replace every local solve, not merely its initializer. Keep the production
candidate search, full mask/depth validation, covariance/Kalman and map gates.
The library objective differs: UFM/group evidence supplies seeds and candidate
ranking, but is not a residual inside GICP/ICP. No custom SciPy refinement runs.
Colored ICP uses only reference points whose color was measured in a retained
RGB-D view; all reference points remain in the downstream geometry checks.
"""

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from lerobot.policies.flash.surface_grounding import SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_registration import (
    LibraryConfig,
    LibraryMatcher,
    measured_colors as measured_colors,
)


class LibraryTrial(LibraryMatcher):
    def __enter__(self):
        self._patch = ExitStack()
        self._patch.enter_context(patch.object(WholeSurfaceFit, "_solve", self.solve_method()))
        if self.config.backend == "open3d_color":
            from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceSession

            original = GrowingSurfaceSession.update

            def update(session, *args, **kwargs):
                view = None
                if session.connected_observation is not None:
                    view = SurfaceFitView(
                        session.connected_observation, session.connected_intrinsics, session.connected_pose
                    )
                self._local.color_view = view
                try:
                    return original(session, *args, **kwargs)
                finally:
                    self._local.color_view = None

            self._patch.enter_context(patch.object(GrowingSurfaceSession, "update", update))
        return self

    def solve_method(self):
        def solve(fitter, initial, max_evaluations):
            pose, detail = self.refine(fitter, initial, max_evaluations=max_evaluations)
            return pose, detail, [(fitter.score(p), p) for p in self._local.proposals]

        return solve

    def __exit__(self, *args):
        self._patch.close()

    def summary(self):
        return {
            "config": asdict(self.config),
            "library_version": self.version,
            "refinements": len(self.calls),
            "refinements_with_library_solve": sum(row["solved_levels"] > 0 for row in self.calls),
            "mean_per_refinement": {
                key: float(np.mean([r[key] for r in self.calls])) for key in self.calls[0]
            }
            if self.calls
            else {},
            "objective_change": "Library geometry/color replaces all local solves; UFM/groups remain seeds and ranking evidence.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("small_gicp", "open3d_plane", "open3d_color"), required=True)
    parser.add_argument("--voxel-sizes", type=float, nargs="+", default=[0.006, 0.003])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("runtime_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from benchmarks.flash_runtime_benchmark import main as run_runtime

    runtime_args = args.runtime_arguments
    if runtime_args and runtime_args[0] == "--":
        runtime_args = runtime_args[1:]
    output = Path(runtime_args[runtime_args.index("--output") + 1])
    config = LibraryConfig(args.backend, tuple(args.voxel_sizes), args.iterations, threads=args.threads)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "registration.json"
        path.write_text(json.dumps(asdict(config)))
        with patch.object(
            sys, "argv", ["flash_runtime_benchmark", *runtime_args, "--registration-config", str(path)]
        ):
            run_runtime()
    (output / "library.json").write_text(
        json.dumps({"config": asdict(config), "shared_pipeline": True}, indent=2) + "\n"
    )
    (output / "code/benchmarks/flash_library_registration.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
