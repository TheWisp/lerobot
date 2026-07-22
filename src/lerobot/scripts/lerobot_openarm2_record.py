"""Record OpenArm 2 Quest demonstrations through LeRobot's native recorder."""

import argparse
from pathlib import Path

from lerobot.configs.openarm2_standard import (
    make_openarm2_quest_config,
    make_openarm2_standard_robot_config,
)
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=60.0)
    parser.add_argument("--reset-seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quest-port", type=int, default=8443)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    cfg = RecordConfig(
        robot=make_openarm2_standard_robot_config(
            control_fps=args.fps,
            enable_torque_on_connect=True,
        ),
        teleop=make_openarm2_quest_config(port=args.quest_port),
        dataset=DatasetRecordConfig(
            repo_id=args.repo_id,
            single_task=args.task,
            root=args.root,
            fps=args.fps,
            episode_time_s=args.episode_seconds,
            reset_time_s=args.reset_seconds,
            num_episodes=args.episodes,
            video=False,
            record_images=False,
            push_to_hub=args.push_to_hub,
        ),
        display_data=False,
        play_sounds=False,
        start_with_reset=False,
        latency_monitor=True,
    )
    record(cfg)


if __name__ == "__main__":
    main()
