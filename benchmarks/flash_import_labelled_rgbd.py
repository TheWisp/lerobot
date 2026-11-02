"""Import consecutive DexYCB/BOP frames for the shared offline tracker.

Uses supplied visible masks to isolate tracking from segmentation quality.
Object pose annotations are written separately for evaluation, never inference.
No downloads, rescaling, depth filling, or tracking-specific parameter changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from benchmarks.flash_experiment_cache import file_hash


class CalibrationLoader(yaml.SafeLoader):
    """Official DexYCB YAML uses a tuple for its unused depth/RGB extrinsics."""


CalibrationLoader.add_constructor("tag:yaml.org,2002:python/tuple", CalibrationLoader.construct_sequence)


def import_clip(kind, source, output, *, object_id, start, count, fps, intrinsics=None, name=None):
    if count < 2 or start < 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Need at least two consecutive frames and a positive nominal rate")
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("rgb", "depth", "masks/object", "evaluation"):
        (output / directory).mkdir(parents=True)
    hashes = {}

    def record(file):
        hashes[str(file)] = file_hash(file)
        return file

    if kind == "bop":
        cameras = json.loads(record(source / "scene_camera.json").read_text())
        annotations = json.loads(record(source / "scene_gt.json").read_text())
        extension = "png"
    elif kind == "dexycb":
        metadata = yaml.safe_load(record(source.parent / "meta.yml").read_text())
        object_index = metadata["ycb_ids"].index(object_id)
        calibration = yaml.load(record(Path(intrinsics)).read_text(), Loader=CalibrationLoader)["color"]
        k = np.array(
            [
                [calibration["fx"], 0, calibration["ppx"]],
                [0, calibration["fy"], calibration["ppy"]],
                [0, 0, 1],
            ]
        )
        extension = "jpg"
    else:
        raise ValueError(f"Unsupported source {kind}")
    poses, ks, scales, worlds, indices = [], [], [], [], []
    for i, original in enumerate(range(start, start + count)):
        world = np.eye(4)
        pose = np.eye(4)
        if kind == "bop":
            camera = cameras[str(original)]
            candidates = [
                (j, row) for j, row in enumerate(annotations[str(original)]) if row["obj_id"] == object_id
            ]
            if len(candidates) != 1:
                raise ValueError("Select an object with a unique labelled instance throughout the clip")
            object_index, annotation = candidates[0]
            k = np.asarray(camera["cam_K"]).reshape(3, 3)
            scale = camera["depth_scale"] / 1000
            if "cam_R_w2c" in camera:
                world[:3, :3] = np.asarray(camera["cam_R_w2c"]).reshape(3, 3)
                world[:3, 3] = np.asarray(camera["cam_t_w2c"]) / 1000
                world = np.linalg.inv(world)
            pose[:3, :3] = np.asarray(annotation["cam_R_m2c"]).reshape(3, 3)
            pose[:3, 3] = np.asarray(annotation["cam_t_m2c"]) / 1000
            rgb = source / "rgb" / f"{original:06d}.png"
            depth = source / "depth" / f"{original:06d}.png"
            mask = cv2.imread(
                str(record(source / "mask_visib" / f"{original:06d}_{object_index:06d}.png")),
                cv2.IMREAD_UNCHANGED,
            )
        else:
            with np.load(record(source / f"labels_{original:06d}.npz"), allow_pickle=False) as data:
                pose[:3] = data["pose_y"][object_index]
                mask = (data["seg"] == object_id).astype(np.uint8) * 255
            scale = 0.001  # DexYCB aligned_depth_to_color is uint16 millimetres.
            rgb = source / f"color_{original:06d}.jpg"
            depth = source / f"aligned_depth_to_color_{original:06d}.png"
        if mask is None:
            raise ValueError(f"Missing visible mask for frame {original}")
        shutil.copyfile(record(rgb), output / "rgb" / f"{i:06d}.{extension}")
        shutil.copyfile(record(depth), output / "depth" / f"{i:06d}.png")
        cv2.imwrite(str(output / "masks/object" / f"{i:06d}.png"), mask)
        poses.append(pose.copy())
        ks.append(k.copy())
        scales.append(scale)
        worlds.append(world)
        indices.append(original)
    np.savez_compressed(
        output / "calibration.npz",
        times_s=np.arange(count) / fps,
        intrinsics=ks,
        depth_m_per_unit=scales,
        world_from_camera=worlds,
    )
    pose_file = output / "evaluation/poses.npy"
    np.save(pose_file, poses)
    case = {
        "id": output.name,
        "display_name": name or output.name,
        "kind": "rgbd_sequence",
        "source_root": str(output),
        "roles": ["object"],
        "expected_frames": count,
        "rgb_extension": extension,
        "mask_source": "Dataset visible-mask annotation (SAM not evaluated)",
        "evaluation": {"object": {"format": "poses_npy", "path": str(pose_file)}},
    }
    # Validate through the actual consumers before publishing a case.
    from benchmarks.flash_experiment_inputs import samples
    from benchmarks.flash_experiment_labels import load_pose_labels

    rows = [{"frame": frame.frame_id, "time_s": frame.source_time_s} for frame, _ in samples(case)]
    load_pose_labels(case, rows)
    provenance = {
        "dataset": kind,
        "source_directory": str(source),
        "object_id": object_id,
        "source_frame_indices": indices,
        "nominal_fps": fps,
        "timing": "Consecutive source indices divided by nominal rate; no original capture timestamps",
        "input_files_sha256": hashes,
        "pose_convention": "object to camera; metres",
        "pose_labels": "evaluation only",
        "mask_source": case["mask_source"],
    }
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("bop", "dexycb"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--intrinsics", type=Path, help="DexYCB camera calibration YAML")
    parser.add_argument("--name")
    args = vars(parser.parse_args())
    print(json.dumps(import_clip(**args), indent=2))


if __name__ == "__main__":
    main()
