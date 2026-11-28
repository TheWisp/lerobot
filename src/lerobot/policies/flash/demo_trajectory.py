"""Display the demonstrated tool poses used by demo adaptation."""

import numpy as np

from lerobot.robots.cartesian_model import CartesianModel

from .rollout_trajectory import fit_trajectory


def read_demo_trajectory(dataset, episode, camera):
    info = dataset.meta.info
    model = CartesianModel.from_description(info["robot_cartesian"], info["robot_geometry"])
    geometry = info["robot_geometry"]["cameras"][camera.removeprefix("observation.images.")]
    camera_from_base = np.linalg.inv(np.asarray(geometry["T_base_cam"]))
    feature = dataset.meta.features[camera]
    calibration = feature["info"]["camera"]["intrinsics"]
    height, width = feature["shape"][:2]
    if (height, width) != (calibration["height"], calibration["width"]):
        raise ValueError("Recorded camera size differs from its calibration")
    names = dataset.meta.features["observation.state"]["names"]
    if not all(name in names for name in model.joint_names):
        raise ValueError("Recorded observations do not contain the robot joints")
    points, segment = [], 0
    # Match compile_demo's demonstrated tool pose, before object-relative
    # adaptation. The action column contains compensated servo goals instead.
    for frame, values in enumerate(dataset.episode_column("observation.state", episode)):
        state = dict(zip(names, values, strict=True))
        if not np.isfinite([state[name] for name in model.joint_names]).all():
            segment += 1
            continue
        tool = camera_from_base @ model.forward(state)
        points.append(
            {
                "time_s": frame / dataset.fps,
                "frame": frame,
                "position": tool[:3, 3].tolist(),
                "segment": segment,
            }
        )
    return {
        "origin_s": 0,
        "points": points,
        "spans": fit_trajectory(points),
        "camera": {
            "width": width,
            "height": height,
            "intrinsics": [
                [calibration["fx"], 0, calibration["cx"]],
                [0, calibration["fy"], calibration["cy"]],
                [0, 0, 1],
            ],
        },
    }
