"""Run one bounded FLASH attempt with LeRobot's SO-107 and aligned RealSense API.

The prepared skill and robot bindings are explicit inputs. Run with
``--preflight`` to check camera calibration and motor state without moving.
Each moving attempt creates a normal LeRobot dataset plus a FLASH trace.
"""

# hardcode-lint: ignore-file - This launcher binds generic FLASH to the calibrated SO-107/RealSense rig.

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import numpy as np

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.appearance import AppearanceBackendFactory
from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.kinematics import SelectedJointKinematics
from lerobot.policies.flash.planner import PlannerLimits
from lerobot.policies.flash.planning_coordinator import PlanBufferConfig
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.recording import FlashEpisodeWriter, RecorderWorker
from lerobot.policies.flash.rgbd_source import LeRobotRGBDSource
from lerobot.policies.flash.runtime import AttachmentRuntimeLimits, FlashRuntime
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
from lerobot.robots.so107_follower_gravity import SO107FollowerGravity, SO107FollowerGravityRobotConfig


class StrictMotorState:
    """Use LeRobot for commands; reject failed reads instead of its cached/zero fallback."""

    def __init__(self, robot: SO107FollowerGravity) -> None:
        self.robot = robot

    def get_state_observation(self) -> dict[str, float]:
        positions = self.robot.bus.sync_read("Present_Position", num_retry=1)
        if set(positions) != set(self.robot.bus.motors) or not all(
            np.isfinite(value) for value in positions.values()
        ):
            raise RuntimeError("incomplete or non-finite SO-107 motor state")
        return {f"{name}.pos": float(value) for name, value in positions.items()}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return self.robot.send_action(action)


def _intrinsics(camera: RealSenseCamera) -> np.ndarray:
    import pyrealsense2 as rs

    values = camera.rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return np.array([[values.fx, 0, values.ppx], [0, values.fy, values.ppy], [0, 0, 1]], dtype=float)


def _features(height: int, width: int) -> dict:
    joints = [f"{name}.pos" for name in MOTOR_NAMES]
    return {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": joints},
        "action": {"dtype": "float32", "shape": (7,), "names": joints},
        "observation.images.overhead": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.overhead_depth": {
            "dtype": "image",
            "shape": (height, width, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
    }


def _new_attempt_settings(settings: dict, suffix: str) -> dict:
    """Give each trial its own ordinary LeRobot dataset without editing the rig profile."""
    output_root = Path(settings["output_root"]).expanduser()
    return {
        **settings,
        "output_root": str(output_root.with_name(f"{output_root.name}-{suffix}")),
        "output_repo_id": f"{settings['output_repo_id']}-{suffix}",
    }


def trial(
    bundle_path: Path,
    config_path: Path,
    *,
    preflight: bool,
    auto_output: bool = False,
    source_dataset_root: Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict:
    bundle = PreparedFlashSkill.load(bundle_path)
    settings = json.loads(config_path.read_text())
    if (
        source_dataset_root is not None
        and Path(settings["source_dataset_root"]).expanduser().resolve()
        != source_dataset_root.expanduser().resolve()
    ):
        raise ValueError("trial config source dataset differs from the selected FLASH demo dataset")
    if auto_output and not preflight:
        settings = _new_attempt_settings(settings, datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f"))
    profile = json.loads(Path(settings["robot_profile"]).expanduser().read_text())
    if profile["type"] != "so107_follower_gravity":
        raise ValueError("FLASH SO-107 launcher requires the gravity robot profile")
    robot = SO107FollowerGravity(SO107FollowerGravityRobotConfig(**profile["fields"]))
    camera = RealSenseCamera(
        RealSenseCameraConfig(
            serial_number_or_name=settings["camera_serial"],
            width=settings["camera_width"],
            height=settings["camera_height"],
            fps=settings["fps"],
            use_depth=True,
            enable_decimation=False,
            enable_spatial=False,
            enable_temporal=False,
            enable_hole_filling=False,
        )
    )
    calibration = json.loads(Path(settings["calibration_path"]).expanduser().read_text())
    arm = profile["fields"]["arm"]
    if arm != settings["robot_arm"]:
        raise ValueError("robot arm and calibrated arm differ")
    try:
        camera.connect()
        robot.connect(calibrate=False)
        strict_robot = StrictMotorState(robot)
        state = strict_robot.get_state_observation()
        rgb, depth, captured_s = camera.read_color_and_aligned_depth_with_timestamp()
        camera_k = _intrinsics(camera)
        demonstrated_k = np.loadtxt(Path(settings["source_dataset_root"]) / "recording/cam_K.txt")
        if rgb.shape[:2] != depth.shape or not np.allclose(camera_k, demonstrated_k, atol=2.0, rtol=0):
            raise ValueError("live aligned camera geometry differs from the demonstrated calibration")
        report = {
            "camera_shape": list(rgb.shape),
            "camera_capture_age_s": time.perf_counter() - captured_s,
            "motor_state": state,
            "camera_k": camera_k.tolist(),
            "preflight": preflight,
        }
        if preflight:
            return report
        full = make_calibrated_so107_kinematics(calibration, arm=arm)
        arm_names = tuple(name for name in MOTOR_NAMES if name != "gripper")
        kin = SelectedJointKinematics(full, tuple(MOTOR_NAMES), arm_names, {"gripper": state["gripper.pos"]})
        source = LeRobotRGBDSource(
            camera,
            intrinsics=camera_k,
            depth_m_per_unit=0.001,
            base_from_camera_at=lambda _time_s: np.asarray(calibration["camera"]["T_base_cam"]),
            calibration_version=bundle.references[0].snippets[0].calibration_version,
            color_order="rgb",
            max_capture_age_s=settings["max_capture_age_s"],
        )
        backend = AppearanceBackendFactory(
            bundle.appearance_models,
            bundle.prompts_by_role,
            settings["camera_serial"],
            settings["min_similarity"],
            settings["registration_threshold_m"],
            settings["registration_min_inliers"],
            settings["registration_trials"],
            sam_resolution=settings["sam_resolution"],
            dino_model_id=bundle.descriptor_model_id,
            axial_symmetries=bundle.axial_symmetries,
            seed_points_by_role=settings.get("seed_points_by_role"),
        )
        output_root = Path(settings["output_root"]).expanduser()
        if output_root.exists():
            raise FileExistsError(f"attempt dataset already exists: {output_root}")
        dataset = LeRobotDataset.create(
            repo_id=settings["output_repo_id"],
            fps=settings["fps"],
            features=_features(*rgb.shape[:2]),
            root=output_root,
            robot_type="so107_follower_gravity",
            use_videos=False,
            record_images=True,
            image_writer_processes=0,
            image_writer_threads=4,
        )
        recorder = RecorderWorker(FlashEpisodeWriter(dataset, 0, settings["task"]))
        runtime = FlashRuntime(
            skill=bundle.skill,
            robot=strict_robot,
            source=source,
            pose_backend_factory=backend,
            references=bundle.references,
            kinematics=kin,
            arm_joint_names=arm_names,
            gripper_joint_name="gripper",
            planner_limits=PlannerLimits(
                tuple(settings["joint_speed_deg_s"]),
                settings["position_error_m"],
                np.deg2rad(settings["rotation_error_deg"]),
                settings["max_pose_age_s"],
                allow_predicted_reference=True,
                ik_refinement_attempts=settings["ik_refinement_attempts"],
                max_attachment_age_s=settings["max_attachment_age_s"],
            ),
            buffer_config=PlanBufferConfig(
                settings["lead_ticks"],
                settings["horizon_ticks"],
                settings["refill_at_ticks"],
                settings["max_reference_shift_m"],
                np.deg2rad(settings["max_reference_turn_deg"]),
                max_wait_for_evidence_s=settings.get("max_wait_for_evidence_s", 2.0),
            ),
            attachment_limits=AttachmentRuntimeLimits(
                settings["history_capacity"],
                settings["max_motor_sample_gap_s"],
                settings["max_attachment_age_s"],
                settings["max_update_shift_m"],
                np.deg2rad(settings["max_update_turn_deg"]),
                settings["max_pregrasp_age_s"],
            ),
            recorder=recorder,
            rgb_feature="overhead",
            depth_feature="overhead_depth",
            max_camera_age_s=settings["max_camera_age_s"],
            fps=settings["fps"],
            causal_tracking=(
                CausalTrackingConfig(**settings["causal_tracking"])
                if settings.get("causal_tracking") is not None
                else None
            ),
        )
        result = runtime.run(
            max_duration_s=settings["max_duration_s"],
            pose_start_timeout_s=settings["pose_start_timeout_s"],
            first_pose_timeout_s=settings["first_pose_timeout_s"],
            stop_requested=stop_requested,
        )
        report.update(
            {
                "outcome": result.outcome,
                "reason": result.reason,
                "phase": result.phase.value,
                "motor_ticks": result.motor_ticks,
                "dataset_root": str(output_root),
                "motor_metrics": {
                    name: value
                    for name, value in vars(runtime.session.motor.metrics).items()
                    if not isinstance(value, list)
                },
                "pose_metrics": vars(runtime.perception.worker.metrics).copy(),
                "causal_metrics": (
                    {
                        "frames": runtime.causal.frames,
                        "anchors": runtime.causal.anchors,
                        "missed_anchors": runtime.causal.missed_anchors,
                        "published": runtime.causal.published,
                    }
                    if runtime.causal is not None
                    else None
                ),
                "planner_last_result": (
                    {
                        "revision": runtime.planner.last_result.revision,
                        "issued_actions": runtime.planner.last_result.issued_actions,
                        "error": runtime.planner.last_result.error,
                        "failure": runtime.planner.last_result.failure,
                        "connection_duration_s": (
                            runtime.planner.last_result.connection.duration_s
                            if runtime.planner.last_result.connection is not None
                            else None
                        ),
                        "connection_start_time_s": (
                            runtime.planner.last_result.connection.start_time_s
                            if runtime.planner.last_result.connection is not None
                            else None
                        ),
                        "connection_start_xyz_m": (
                            runtime.planner.last_result.connection.start_pose[:3, 3].tolist()
                            if runtime.planner.last_result.connection is not None
                            else None
                        ),
                        "connection_goal_xyz_m": (
                            runtime.planner.last_result.connection.goal_pose[:3, 3].tolist()
                            if runtime.planner.last_result.connection is not None
                            else None
                        ),
                    }
                    if runtime.planner.last_result is not None
                    else None
                ),
                "planner_superseded": runtime.planner.superseded,
                "planner_pending_revision": runtime.coordinator._pending_revision,
                "attachment": {
                    "active_role": runtime.attachment.active_role,
                    "last_event": runtime.attachment.last_event,
                    "state_available": runtime.attachment.state(runtime.session.motor.clock()) is not None,
                    "roles": {
                        role: {
                            "has_postgrasp_observation": updater.has_postgrasp_observation,
                            "last_rejection": updater.last_rejection,
                        }
                        for role, updater in runtime.attachment.updaters.items()
                    },
                },
            }
        )
        return report
    finally:
        if robot.is_connected:
            robot.disconnect()
        if camera.is_connected:
            camera.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--auto-output", action="store_true", help="write a new dataset for this attempt")
    parser.add_argument(
        "--source-dataset-root", type=Path, help="require this demo dataset in the trial config"
    )
    args = parser.parse_args()
    stop = Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    report = trial(
        args.bundle,
        args.config,
        preflight=args.preflight,
        auto_output=args.auto_output,
        source_dataset_root=args.source_dataset_root,
        stop_requested=stop.is_set,
    )
    print(json.dumps(report, indent=2))
    if report.get("outcome") == "failure" or (report.get("outcome") == "interrupted" and not stop.is_set()):
        print(f"FLASH {report['outcome']}: {report['reason']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
