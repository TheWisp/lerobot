"""Read-only Point2Pose TSDF/registration capture for the insertion demo.

Pass this as ``--worker-bridge`` to ``run_p2p_external_mask.py`` with
``--mask-source internal``. The sealed bridge and pinned P2P checkout remain
unmodified; this wrapper captures selected outputs after each normal step.
Set ``FLASH_P2P_SDF_PROBE_OUTPUT`` to an explicit output directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

DEMO_FIRST_FRAME = 100
CAPTURE_FRAMES = frozenset(
    (100, 252, 279, 295, 305, 311, 316, 340, 369, 495, 500, 501, 505, 514, 530, 540, 541, 550)
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-bridge", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("sealed_p2p_bridge_sdf_probe", args.sealed_bridge)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load sealed Point2Pose bridge")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    output_name = os.environ.get("FLASH_P2P_SDF_PROBE_OUTPUT")
    if not output_name:
        raise ValueError("FLASH_P2P_SDF_PROBE_OUTPUT is required")
    output = Path(output_name)
    output.mkdir(parents=True, exist_ok=True)
    ordinary_answer = bridge.Session._answer

    def capture_answer(session, frame, milliseconds):
        answer = ordinary_answer(session, frame, milliseconds)
        demo_frame = frame.id + DEMO_FIRST_FRAME
        if demo_frame not in CAPTURE_FRAMES:
            return answer
        obj = session.pipe.objects[0]
        frontend = getattr(session, "_fe", None)
        stats = {} if frontend is None else (frontend.reg_stats.get(0) or {})
        sdf = getattr(obj, "sdf", None) or {}
        volume = getattr(obj, "sdf_volume", None)
        selected_idx = stats.get("best_cluster_idx")
        candidates = stats.get("clusters") or []
        selected = candidates[selected_idx] if selected_idx is not None and candidates else {}
        summary = {
            str(k): v for k, v in selected.items() if isinstance(v, (int, float, str, bool, np.number))
        }
        if selected.get("T") is not None:
            summary["T"] = np.asarray(selected["T"]).tolist()
        metadata = {
            "frame": demo_frame,
            "pose_lost": bool(obj.lost),
            "num_integrated": int(getattr(obj, "sdf_num_integrated", 0)),
            "num_candidates": len(candidates),
            "best_cluster_idx": selected_idx,
            "selected_candidate": summary,
            "sdf_refine": stats.get("sdf_refine"),
            "final_inlier_gate": stats.get("final_inlier_gate"),
            "return_refined_pose": bool(
                getattr(session.pipe.frontend.register, "_return_refined_pose", False)
            ),
        }
        weights = np.asarray(getattr(volume, "_weight_vol_cpu", np.zeros(0)), dtype=np.float32)
        if getattr(volume, "gpu_mode", False) and hasattr(volume, "_weight_vol_gpu"):
            import pycuda.driver as cuda

            weights = weights.copy()
            cuda.memcpy_dtoh(weights, volume._weight_vol_gpu)
        stem = f"{demo_frame:06d}"
        np.savez_compressed(
            output / f"{stem}.npz",
            tsdf=np.asarray(sdf.get("tsdf", np.zeros(0)), dtype=np.float32),
            weight=weights,
            bounds=np.asarray(sdf.get("vol_bnds", np.zeros((3, 2))), dtype=np.float32),
            origin=np.asarray(sdf.get("vol_origin", np.zeros(3)), dtype=np.float32),
            voxel_size=np.float32(sdf.get("voxel_size", 0)),
            pose=np.asarray(answer["delta"], dtype=np.float64),
        )
        (output / f"{stem}.json").write_text(
            json.dumps(
                metadata,
                default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value),
                indent=2,
            )
        )
        return answer

    bridge.Session._answer = capture_answer
    sys.argv = [str(args.sealed_bridge), "--repo", str(args.repo), "--config", str(args.config)]
    bridge.main()


if __name__ == "__main__":
    main()
