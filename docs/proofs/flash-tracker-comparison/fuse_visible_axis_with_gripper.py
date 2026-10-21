"""Ablate a held-pose update that uses visible shaft axis, not a hidden tip.

The gripper prior supplies the object center, end identity and axial gauge.
Each current RGB-D shaft fit changes only the two observable tilt components.
This diagnostic does not gate weak geometry or claim the contact end was seen.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).parent
    parser.add_argument("--recording", type=Path, default=root / "pose_traces.npz")
    parser.add_argument("--gripper-prior", type=Path, default=root / "gripper_dowel_seed340_340_500.npz")
    parser.add_argument("--visible-axis", type=Path, default=root / "gripper_visible_end_300_500.csv")
    parser.add_argument("--output", type=Path, default=root / "gripper340_visible_axis_340_500.npz")
    args = parser.parse_args()

    with np.load(args.recording) as recording:
        camera_from_base = np.linalg.inv(recording["base_from_camera"])
        gripper_poses = recording["demo_ee_pose_base"].copy()
        axis_object = recording["dowel_reference_shaft_camera"].copy()
        center_object = recording["dowel_reference_center_camera_m"].copy()
    with np.load(args.gripper_prior) as prior:
        frames = prior["frames"].copy()
        poses = prior["poses"].copy()
        lost = prior["lost"].copy()
    with args.visible_axis.open(newline="", encoding="utf-8") as stream:
        observed = {int(row["frame"]): row for row in csv.DictReader(stream)}

    corrected = []
    available = []
    for frame, pose, missing in zip(frames, poses, lost, strict=True):
        row = observed.get(int(frame))
        if missing or row is None:
            corrected.append(np.eye(4))
            available.append(False)
            continue
        axis_in_ee = np.array([float(row[f"visible_axis_ee_{coordinate}"]) for coordinate in "xyz"])
        camera_from_ee = camera_from_base @ gripper_poses[int(frame)]
        observed_axis = camera_from_ee[:3, :3] @ axis_in_ee
        prior_axis = pose[:3, :3] @ axis_object
        if observed_axis @ prior_axis < 0:
            observed_axis = -observed_axis
        swing = Rotation.align_vectors(observed_axis[None], prior_axis[None])[0].as_matrix()
        center = pose[:3, :3] @ center_object + pose[:3, 3]
        updated = np.eye(4)
        updated[:3, :3] = swing @ pose[:3, :3]
        updated[:3, 3] = center - updated[:3, :3] @ center_object
        corrected.append(updated)
        available.append(True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=frames,
        poses=np.stack(corrected),
        lost=~np.array(available),
        metadata=np.asarray(
            json.dumps(
                {
                    "method": "gripper_center_prior_plus_same_frame_visible_axis",
                    "gripper_prior": str(args.gripper_prior),
                    "visible_axis": str(args.visible_axis),
                }
            )
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()
