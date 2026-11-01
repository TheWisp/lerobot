"""Read recorded RGB-D and masks without importing saved pose estimates.

Adapters reuse the frozen tracking suite. Nothing writes to a source dataset.
Pose labels are deliberately read by the scorer, separately from this module.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.perception import RGBDFrame


def path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def image(file: Path, *, color: bool = False) -> np.ndarray:
    data = cv2.imread(str(file), cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(file)
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB) if color else data


def decode_mask(runs: list, shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    end = 0
    for start, length in runs:
        if type(start) is not int or type(length) is not int or start < end or length < 1:
            raise ValueError("invalid mask runs")
        end = start + length
        if end > flat.size:
            raise ValueError("mask extends outside its image")
        flat[start:end] = True
    return flat.reshape(shape)


def _ycb(case: dict):
    root = path(case["source_root"])
    start, count = case["start_frame"], case["expected_frames"]
    names = sorted(p.name for p in (root / "rgb").glob("*.png"))[start : start + count]
    if len(names) != count:
        raise ValueError("missing YCB images")
    ns = np.asarray([int(Path(n).stem) for n in names], dtype=np.int64)
    times = (ns - ns[0]) * 1e-9
    k = np.loadtxt(root / "cam_K.txt").reshape(3, 3)
    for i, (name, t) in enumerate(zip(names, times, strict=True)):
        frame = RGBDFrame(
            i,
            image(root / "rgb" / name, color=True),
            image(root / "depth" / name),
            k,
            0.001,
            np.eye(4),
            float(t),
            float(t),
            "fixed-camera",
        )
        yield frame, {"object": image(root / "gt_mask" / name) > 0}


def _recording(case: dict):
    root = path(case["artifact_root"]) / "recording"
    evidence = path(case["mask_evidence_dir"])
    manifest = json.loads((evidence / "manifest.json").read_text())
    times = np.atleast_1d(np.loadtxt(root / "times.txt"))
    times = times - times[0]
    if (
        not manifest["complete"]
        or manifest["frames_completed"] != len(times)
        or len(times) != case["expected_frames"]
    ):
        raise ValueError("demo masks and RGB-D have different frame counts")
    k = np.loadtxt(root / "cam_K.txt").reshape(3, 3)
    offset = None
    for i, t in enumerate(times):
        rgb = image(root / "rgb" / f"{i:06d}.{case.get('rgb_extension', 'jpg')}", color=True)
        depth = image(root / "depth" / f"{i:06d}.png")
        with gzip.open(evidence / f"{i:06d}.json.gz", "rt") as stream:
            row = json.load(stream)
        if row["frame_index"] != i or row["image_size"] != list(depth.shape):
            raise ValueError("demo mask belongs to a different image")
        if offset is None:
            offset = row["source_time_s"]
        # The recorded clock may have a constant origin offset. Its intervals must match.
        if not np.isclose(row["source_time_s"] - offset, t, atol=1e-6, rtol=0):
            raise ValueError("demo mask time differs from recorded RGB-D time")
        masks = {role: decode_mask(row["masks_rle"].get(role, []), depth.shape) for role in case["roles"]}
        yield (
            RGBDFrame(
                i,
                rgb,
                depth,
                k,
                0.001,
                np.asarray(row["base_from_camera"]),
                float(t),
                float(t),
                row["calibration_version"],
            ),
            masks,
        )


def _lerobot(case: dict):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.flash.observation_dataset import (
        camera_feature_keys,
        depth_scale_m,
        episode_camera_samples,
        scene_from_camera,
    )

    root = path(case["artifact_root"])
    source = json.loads((root / "flash/v1/source.json").read_text())
    dataset = LeRobotDataset(source["repo_id"], root=root)
    episode = case.get("episode", 0)
    rgb_key, depth_key = camera_feature_keys(dataset.meta.features, source)
    samples, _ = episode_camera_samples(root, dataset, source, episode)
    for i, sample in enumerate(samples):
        row = dataset[sample.row_index]
        rgb = (row[rgb_key].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        depth = row[depth_key].numpy().squeeze().astype(np.float32)
        camera, version = scene_from_camera(row, dataset.meta.features, source)
        masks = {
            role: image(root / "flash/v1/imported-masks" / role / f"{i:06d}.png") > 0
            for role in case["roles"]
        }
        yield (
            RGBDFrame(
                i,
                rgb,
                depth,
                np.asarray(source["camera_intrinsics"]),
                depth_scale_m(source),
                camera,
                sample.source_time_s,
                sample.source_time_s,
                version,
            ),
            masks,
        )


def samples(case: dict):
    reader = {
        "ycb_annotated_pose": _ycb,
        "flash_prepared_demo": _recording,
        "recorded_rgbd": _recording,
        "lerobot_rgbd": _lerobot,
    }[case["kind"]]
    previous_time = -np.inf
    count = 0
    for frame, masks in reader(case):
        if (
            frame.frame_id != count
            or not np.isfinite(frame.source_time_s)
            or frame.source_time_s <= previous_time
        ):
            raise ValueError("RGB-D frame IDs or timestamps are not increasing")
        if frame.rgb.shape[:2] != frame.depth.shape or any(
            m.shape != frame.depth.shape for m in masks.values()
        ):
            raise ValueError("RGB, depth and masks do not align")
        if not np.isfinite(frame.intrinsics).all() or np.any(np.diag(frame.intrinsics)[:2] <= 0):
            raise ValueError("invalid camera intrinsics")
        yield frame, masks
        previous_time = frame.source_time_s
        count += 1
    if count != case["expected_frames"]:
        raise ValueError("source count differs from the case manifest")
