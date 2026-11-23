"""Record optional stereo inputs through the ordinary LeRobot writer.

The robot owns its cameras. Recording peeks one complete cached frameset and
uses that RGB/depth pair too, rather than attaching newer IR to older RGB.
IR and sensor depth are lossless TIFF image channels; RGB keeps the configured
video encoder. Frame sequence and host frameset-arrival time expose skipped or
repeated camera frames independently of the dataset's nominal control-loop timestamp.
"""

import numpy as np

from lerobot.datasets.io_utils import write_info
from lerobot.utils.constants import OBS_IMAGES


def stereo_cameras(robot):
    names = getattr(robot, "camera_feature_names", {}) or {}
    for name, camera in getattr(robot, "cameras", {}).items():
        if getattr(getattr(camera, "config", None), "enable_infrared", False):
            yield names.get(name, name), camera


def add_stereo_features(features, robot):
    """Declare lossless channels before creating the dataset, without connecting."""
    for name, camera in stereo_cameras(robot):
        if not camera.config.width or not camera.config.height or not camera.config.fps:
            raise ValueError("Stereo recording requires explicit camera width, height and FPS")
        if getattr(camera, "post_grab_processor", None) is not None:
            raise ValueError(
                "Stereo recording needs raw camera data: enable the robot's record_depth setting"
            )
        color_key = f"{OBS_IMAGES}.{name}"
        if color_key not in features:
            raise ValueError(f"Stereo camera {name} has no RGB observation feature")
        for suffix in ("infrared_left", "infrared_right"):
            features[f"{color_key}_{suffix}"] = {
                "dtype": "image",
                "shape": camera.infrared_shape,
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": False, "modality": "infrared"},
            }
        features[f"{color_key}_depth"] = {
            "dtype": "image",
            "shape": (*features[color_key]["shape"][:2], 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "depth_unit": "mm"},
        }
        # The dataset tensor reader defaults Python floats to float32. Integer
        # nanoseconds preserve capture intervals even after days of host uptime.
        for suffix in ("capture_time_ns", "frame_sequence"):
            features[f"observation.cameras.{name}.{suffix}"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }
    return features


def capture_stereo(robot):
    """Return image observations and scalar dataset fields from the same capture."""
    images, timing = {}, {}
    for name, camera in stereo_cameras(robot):
        rgb, depth, left, right, stamp, sequence = camera.read_latest_stereo()
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError("Stereo infrared images must be uint8")
        if left.shape != camera.infrared_shape[:2] or right.shape != left.shape:
            raise ValueError("Stereo infrared dimensions do not match the configured profile")
        images.update(
            {
                name: rgb,
                f"{name}_depth": depth,
                f"{name}_infrared_left": left[..., None],
                f"{name}_infrared_right": right[..., None],
            }
        )
        timing[f"observation.cameras.{name}.capture_time_ns"] = np.array([round(stamp * 1e9)], dtype=np.int64)
        timing[f"observation.cameras.{name}.frame_sequence"] = np.array([sequence], dtype=np.int64)
    return images, timing


def attach_stereo_calibration(meta, robot):
    changed = False
    for name, camera in stereo_cameras(robot):
        color_key = f"{OBS_IMAGES}.{name}"
        calibration = {
            **camera.stereo_calibration(),
            "left_feature": f"{color_key}_infrared_left",
            "right_feature": f"{color_key}_infrared_right",
            "capture_time_feature": f"observation.cameras.{name}.capture_time_ns",
            "capture_time_unit": "ns",
            "sequence_feature": f"observation.cameras.{name}.frame_sequence",
        }
        info = meta.features[color_key].setdefault("info", {})
        if "stereo" in info and info["stereo"] != calibration:
            raise ValueError(f"Stereo calibration for {name} differs from the recording")
        info["stereo"] = calibration
        changed = True
    if changed:
        write_info(meta.info, meta.root)
