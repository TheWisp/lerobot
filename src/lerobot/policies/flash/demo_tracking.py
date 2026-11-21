"""Read an ordinary LeRobot episode into the live/experiment frame session.

This adapter owns neither a camera nor a second tracking algorithm. Output
poses remain in the recorded camera frame; robot calibration is a later input.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np

from .demo_document import DemoDocument, atomic_json, load_document
from .live_surface_session import LIVE_METHOD, LiveSurfaceSession
from .perception import RGBDFrame
from .pipeline_version import PIPELINE_VERSION
from .surface_session_config import defaults, validate


class DemoFrames:
    def __init__(self, dataset, episode: int, document: DemoDocument):
        self.dataset, self.document = dataset, document
        self.first, self.length = dataset.episode_rows(episode)
        features = dataset.meta.features
        camera = features[document.camera]
        calibration = camera.get("info", {}).get("camera", {}).get("intrinsics")
        if calibration is None:
            raise ValueError("This camera has no recorded intrinsics. Record with a calibrated RGB-D camera.")
        aligned = [
            key
            for key, feature in features.items()
            if feature.get("info", {}).get("is_depth_map")
            and feature.get("info", {}).get("camera", {}).get("aligned_to") == document.camera
        ]
        if len(aligned) != 1:
            raise ValueError("Select a color camera with exactly one recorded, color-aligned depth channel")
        self.depth_key = aligned[0]
        self.intrinsics = np.array(
            [
                [calibration["fx"], 0, calibration["cx"]],
                [0, calibration["fy"], calibration["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.scale = {"m": 1.0, "mm": 0.001}[dataset.depth_output_unit]
        self.shape = (calibration["height"], calibration["width"])
        if (
            tuple(camera["shape"][:2]) != self.shape
            or tuple(features[self.depth_key]["shape"][:2]) != self.shape
        ):
            raise ValueError("Recorded camera calibration, RGB and depth sizes disagree")
        self.stereo = camera.get("info", {}).get("stereo")
        self.stereo_geometry = None
        if self.stereo:
            intr = self.stereo["left_intrinsics"]
            self.stereo_geometry = (
                np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]),
                np.asarray(self.stereo["color_from_left"]),
                self.stereo["baseline_m"],
            )
            for field in ("left_feature", "right_feature", "capture_time_feature", "sequence_feature"):
                if self.stereo[field] not in features:
                    raise ValueError(f"Missing recorded stereo channel: {self.stereo[field]}")
            self.capture_origin = int(dataset[self.first][self.stereo["capture_time_feature"]].item())

    def require_depth_source(self, source):
        from .depth_enhancement import needs_stereo_pair

        if needs_stereo_pair(source) and self.stereo_geometry is None:
            raise ValueError("This demo has no stereo IR inputs. Record a new demo with stereo enabled.")

    def __iter__(self):
        for index in range(self.document.reference_frame, self.length):
            row = self.dataset[self.first + index]
            rgb = (
                np.rint(row[self.document.camera].numpy().transpose(1, 2, 0) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            # Shared learned-depth output uses float metres. Normalize the
            # dataset's mm/m option here so model output cannot inherit a mm scale.
            depth = row[self.depth_key].numpy().squeeze().astype(np.float32) * self.scale
            timestamp = float(row["timestamp"].item())
            infrared = {}
            if self.stereo:
                timestamp = (
                    int(row[self.stereo["capture_time_feature"]].item()) - self.capture_origin
                ) * 1e-9
                infrared = {
                    f"infrared_{side}": np.rint(row[self.stereo[f"{side}_feature"]].numpy().squeeze() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                    for side in ("left", "right")
                }
            yield RGBDFrame(
                index,
                rgb,
                depth,
                self.intrinsics,
                1.0,
                np.eye(4),
                timestamp,
                timestamp,
                "recorded-camera",
                **infrared,
            )


def track_demo(
    dataset, episode, document, destination: Path, *, settings=None, mask_source=None, depth_source="sensor"
):
    """Persist measured masks and published poses using the shared session.

    ``mask_source`` is also used by deterministic RGB-D tests, while production
    uses the same Sam3MaskSource as live camera inspection.
    """
    from lerobot.gui.flash_group_overlay import SceneGroupOverlay
    from lerobot.gui.flash_live_scene import live_scene_packet

    from .appearance import Sam3MaskSource
    from .depth_enhancement import make_depth_source, tracking_frame_with_depth_source
    from .vision_pipeline import VisionPipelineConfig

    frames = DemoFrames(dataset, episode, document)
    frames.require_depth_source(depth_source)
    if not document.objects:
        raise ValueError("Select at least one object")
    settings = validate(settings or defaults())
    config = VisionPipelineConfig.from_dict(settings["config"])
    prompts = {obj.id: obj.name for obj in document.objects}
    source = mask_source or Sam3MaskSource(
        prompts,
        camera_id=document.camera,
        resolution=config.sam_resolution,
        seed_points_by_role={obj.id: obj.point for obj in document.objects if obj.point is not None},
        detection_correction=config.detection_correction.control(offline=True),
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(
        destination / "input.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "document": document.model_dump(mode="json"),
            "settings": settings,
            "depth_source": depth_source,
        },
    )
    depth_model = make_depth_source(depth_source)
    session = LiveSurfaceSession(settings, source)
    group_overlay = SceneGroupOverlay()
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "total": frames.length - document.reference_frame,
        "done": 0,
    }
    atomic_json(destination / "status.json", {**metadata, "state": "running"})
    previous_time = None
    result = None
    try:
        for frame in frames:
            if frame.source_time_s == previous_time:
                # The robot loop may sample one cached camera frameset twice.
                # Keep both robot rows, but estimate this observation only once.
                repeated = {**result, "frame": frame.frame_id}
                atomic_json(destination / "frames" / f"{frame.frame_id:06d}.json", repeated)
                metadata["done"] += 1
                continue
            previous_time = frame.source_time_s
            frame = tracking_frame_with_depth_source(frame, depth_model, frames.stereo_geometry)
            row, _, masks = session.step(frame)
            if frame.frame_id == document.reference_frame:
                # Exact processed observation that defined the demo's object
                # coordinates. Apply primes the SAME session with this frame,
                # then measures new frames; it must not invent a new origin.
                np.savez_compressed(
                    destination / "reference.npz",
                    rgb=frame.rgb,
                    depth=frame.metric_depth(),
                    intrinsics=frame.intrinsics,
                    **masks,
                )
            # Use the same display evidence as live inspection. RGB stays in
            # the LeRobot video; the sidecar preserves exact processed depth
            # and this frame's surface, never the final surface back-projected.
            packet = live_scene_packet(
                frame,
                session,
                row,
                masks,
                group_overlay=group_overlay,
                depth_source=depth_source,
                encoded_rgb=b"",
            )
            header_size = struct.unpack("<I", packet[:4])[0]
            evidence = json.loads(packet[4 : 4 + header_size])
            scenes = destination / "scenes"
            scenes.mkdir(exist_ok=True)
            (scenes / f"{frame.frame_id:06d}.bin.gz").write_bytes(gzip.compress(packet, compresslevel=1))
            result = {
                "pipeline_version": PIPELINE_VERSION,
                "frame": frame.frame_id,
                "observation_frame": frame.frame_id,
                "time_s": frame.source_time_s,
                "depth_source": depth_source,
                "intrinsics": frames.intrinsics.tolist(),
                "shape": list(frames.shape),
                "camera": evidence["camera"],
                "objects": evidence["objects"],
                "scene_group_tracks": evidence["scene_group_tracks"],
                "masks_rle": evidence["masks_rle"],
            }
            atomic_json(destination / "frames" / f"{frame.frame_id:06d}.json", result)
            metadata["done"] += 1
            atomic_json(
                destination / "status.json", {**metadata, "state": "running", "frame": frame.frame_id}
            )
        tracks = session.engine.candidates[LIVE_METHOD]
        np.savez_compressed(
            destination / "surfaces.npz", **{role: track.store.points for role, track in tracks.items()}
        )
        atomic_json(
            destination / "status.json",
            {
                **metadata,
                "state": "complete",
                "reference_frames": {role: track.reference_frame for role, track in tracks.items()},
                "display_schema": 1,
            },
        )
    except BaseException as exc:
        atomic_json(
            destination / "status.json", {**metadata, "state": "failed", "reason": str(exc) or "Stopped"}
        )
        raise
    finally:
        session.close()


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--revision", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--depth-source", default="sensor")
    args = parser.parse_args()
    dataset = LeRobotDataset(args.root.name, root=args.root)
    document, _ = load_document(dataset, args.episode)
    if document.revision != args.revision:
        raise ValueError("The demo changed before processing started; process the current edit")
    import json

    settings = json.loads(args.settings.read_text()) if args.settings else None
    track_demo(
        dataset, args.episode, document, args.output, settings=settings, depth_source=args.depth_source
    )


if __name__ == "__main__":
    main()
