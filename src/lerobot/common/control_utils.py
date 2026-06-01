# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

########################################################################################
# Utilities
########################################################################################
import json
import logging
import os
import sys
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.utils.import_utils import _deepdiff_available, require_package

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.types import PolicyAction


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener():
    """
    Initializes a flow-control channel pre-registered with the legacy three verbs.

    Despite the name, the underlying mechanism is now a pluggable
    :class:`~lerobot.common.control_channel.ControlChannel` (a registry of action
    names mapped to triggers). This helper preserves the legacy contract — the
    returned events dict has ``exit_early`` / ``rerecord_episode`` /
    ``stop_recording`` keys, all initially False, with default keyboard bindings
    on right / left / esc — so callers like ``lerobot_record.py`` see no change.

    Stdin JSON-lines is auto-attached when ``LEROBOT_CONTROL_CHANNEL_STDIN=1``
    (set by the GUI's subprocess launcher), so a GUI-launched record run is
    controllable via ``POST /api/run/control`` without code changes here.

    The returned ``listener`` is a :class:`ControlChannel` whose ``.stop()``
    tears down all sources — drop-in for the old ``pynput.keyboard.Listener``.

    Returns:
        A tuple of (channel, events_dict).
    """
    from lerobot.common.control_channel import init_control_channel

    channel, events = init_control_channel()
    # Legacy bindings — match the pre-channel ``init_keyboard_listener``
    # behavior exactly. The pynput source already attached above will
    # pick these up on the next keypress because it walks the registry
    # at emit time, not at attach time.
    #
    # ``exit_early`` is multi-key because the legacy listener set it on
    # every flow-control keypress: right alone advances; left also
    # rerecords; esc also stops. Modelled here as "exit_early is
    # triggered by any of {right, left, esc}", which keeps the inner
    # episode loop's `if events["exit_early"]: break` working unchanged
    # whether the user wanted advance, rerecord, or stop.
    channel.register("exit_early", keyboard_keys=("right", "left", "esc"))
    channel.register("rerecord_episode", keyboard_key="left")
    channel.register("stop_recording", keyboard_key="esc")
    return channel, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    require_package("deepdiff", extra="deepdiff-dep")

    from lerobot.utils.constants import DEFAULT_FEATURES

    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


def _read_train_dataset_repo_id(pretrained_path: str | Path) -> str | None:
    """Read the training dataset repo_id from a saved policy's ``train_config.json``.

    Returns ``None`` if the file is missing or unreadable. Pure JSON parse —
    avoids importing the full draccus train config so this is cheap and side-effect-free.
    """
    path = Path(pretrained_path)
    if not path.is_dir():
        return None
    train_config = path / "train_config.json"
    if not train_config.is_file():
        return None
    try:
        with open(train_config) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        return None
    repo_id = dataset.get("repo_id")
    return repo_id if isinstance(repo_id, str) else None


def _read_dataset_robot_type(repo_id: str) -> str | None:
    """Read ``robot_type`` from a locally cached dataset's ``meta/info.json``.

    Returns ``None`` when the dataset isn't cached or the field is missing.
    Intentionally avoids any hub download or full
    :class:`LeRobotDatasetMetadata` construction — we just want a string.
    """
    from lerobot.utils.constants import HF_LEROBOT_HOME

    info_path = HF_LEROBOT_HOME / repo_id / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        with open(info_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    robot_type = data.get("robot_type")
    return robot_type if isinstance(robot_type, str) else None


def warn_on_policy_robot_type_mismatch(
    pretrained_path: str | Path | None,
    robot: Robot,
    *,
    interactive: bool | None = None,
) -> str | None:
    """Surface a warning when the runtime robot's ``robot_type`` differs from the one the policy was trained on.

    The check is advisory — inference is NOT blocked. The mismatch may be
    intentional (e.g. running a policy trained on ``bi_so107_follower`` with the
    predictive variant ``bi_so107_follower_predictive`` to add inference-time
    lookahead). The system has no way to verify embodiment compatibility beyond
    string equality, so we log loudly and let the operator confirm.

    Preconditions: ``robot`` is a constructed :class:`Robot` (need not be connected).

    Args:
        pretrained_path: Local path to the saved policy. If ``None``, only a
            hub repo, or missing ``train_config.json``, the check is skipped
            with an INFO-level note.
        robot: The runtime robot instance.
        interactive: If ``True``, prompt the user to confirm via stdin when a
            mismatch is detected. If ``False``, never prompt. If ``None``
            (default), prompt only when ``stdin`` is a TTY AND
            ``LEROBOT_NONINTERACTIVE`` env var is unset.

    Returns:
        The training dataset's ``robot_type`` string when it could be
        determined, else ``None``.
    """
    if pretrained_path is None:
        logging.info(
            "Inference: skipping policy/robot embodiment check (no pretrained_path). Runtime robot_type=%s.",
            robot.robot_type,
        )
        return None

    train_repo_id = _read_train_dataset_repo_id(pretrained_path)
    if train_repo_id is None:
        logging.info(
            "Inference: could not determine the policy's training dataset "
            "(no train_config.json at %s). Runtime robot_type=%s. Skipping "
            "embodiment compatibility check.",
            pretrained_path,
            robot.robot_type,
        )
        return None

    trained_robot_type = _read_dataset_robot_type(train_repo_id)
    if trained_robot_type is None:
        logging.info(
            "Inference: policy trained on dataset %r is not in the local cache; "
            "cannot verify embodiment. Runtime robot_type=%s.",
            train_repo_id,
            robot.robot_type,
        )
        return None

    if trained_robot_type == robot.robot_type:
        logging.info(
            "Inference: policy/robot embodiment match: robot_type=%s (trained on %s).",
            robot.robot_type,
            train_repo_id,
        )
        return trained_robot_type

    banner = "=" * 72
    logging.warning(
        "\n%s\n"
        "EMBODIMENT MISMATCH WARNING\n"
        "  Policy was trained on robot_type=%s (dataset: %s)\n"
        "  Runtime robot_type=%s\n"
        "\n"
        "  The system will NOT block inference — observation/action shapes are\n"
        "  still validated by the policy. But behaviour is undefined if the new\n"
        "  embodiment differs in kinematics, calibration, or control semantics.\n"
        "\n"
        "  Common intentional case: running a non-predictive-trained policy on\n"
        "  the *_predictive variant of the same hardware. Verify that:\n"
        "    - Joint names and order match\n"
        "    - State and action units are the same\n"
        "    - Cameras produce equivalent observations\n"
        "%s",
        banner,
        trained_robot_type,
        train_repo_id,
        robot.robot_type,
        banner,
    )

    if interactive is None:
        interactive = (
            sys.stdin is not None and sys.stdin.isatty() and not os.environ.get("LEROBOT_NONINTERACTIVE")
        )

    if interactive:
        try:
            reply = input(
                f"Continue with robot_type={robot.robot_type} "
                f"(policy trained on {trained_robot_type})? [y/N]: "
            )
        except (EOFError, KeyboardInterrupt):
            reply = ""
        if reply.strip().lower() not in {"y", "yes"}:
            raise SystemExit(
                "Aborted by user due to robot_type mismatch. Re-run with "
                "LEROBOT_NONINTERACTIVE=1 to bypass the prompt."
            )

    return trained_robot_type
