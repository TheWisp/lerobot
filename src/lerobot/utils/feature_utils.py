#!/usr/bin/env python

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
"""Lightweight feature-manipulation utilities.

These functions are intentionally kept free of heavy dependencies (e.g. the
HuggingFace ``datasets`` library) so that they can be imported from anywhere
in the codebase – including modules that are part of the *minimal* install –
without triggering the ``lerobot.datasets`` package guard.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PolicyFeature

from .constants import ACTION, DEFAULT_FEATURES, OBS_ENV_STATE, OBS_IMAGES, OBS_STR


def _validate_feature_names(features: dict[str, dict]) -> None:
    """Validate that feature names do not contain invalid characters and that
    optional flags (e.g. ``per_episode``) are only set on supported feature
    shapes.

    Args:
        features (dict): The LeRobot features dictionary.

    Raises:
        ValueError: If any feature name contains '/' or carries an invalid
            ``per_episode`` declaration (e.g. on an image/video/vector feature).
    """
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")

    # Late import to avoid a circular through datasets.feature_utils → utils.feature_utils.
    from lerobot.datasets.feature_utils import validate_per_episode_flag

    errors = []
    for name, ft in features.items():
        err = validate_per_episode_flag(name, ft)
        if err:
            errors.append(err)
    if errors:
        raise ValueError("Invalid per_episode declaration:\n" + "".join(errors))


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """Convert hardware-specific features to a LeRobot dataset feature dictionary.

    This function takes a dictionary describing hardware outputs (like joint states
    or camera image shapes) and formats it into the standard LeRobot feature
    specification. Single-channel cameras (shape ``(H, W, 1)``) are flagged as depth
    maps via ``info["is_depth_map"] = True``; three-channel cameras ``(H, W, 3)`` are
    treated as RGB.

    Args:
        hw_features (dict): Dictionary mapping feature names to their type (float for
            joints) or shape (tuple for images).
        prefix (str): The prefix to add to the feature keys (e.g., "observation"
            or "action").
        use_video (bool): If True, image features are marked as "video", otherwise "image".

    Returns:
        dict: A LeRobot features dictionary. Depth cameras carry ``info["is_depth_map"] = True``.
    """
    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    # TODO(CarolinePascal): we should not rely on the shape to determine if a feature is a camera !
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        dtype = "video" if use_video else "image"
        if len(shape) == 3 and shape[2] in (1, 3):
            features[f"{prefix}.images.{key}"] = {
                "dtype": dtype,
                "shape": shape,
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": shape[2] == 1},
            }
        else:
            raise ValueError(
                f"Camera feature '{key}' has shape {shape}. "
                f"Expected a 3-tuple (H, W, C), e.g. (480, 640, 3) for RGB or (480, 640, 1) for depth."
            )

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """Construct a single data frame from raw values based on dataset features.

    A "frame" is a dictionary containing all the data for a single timestep,
    formatted as numpy arrays according to the feature specification.

    Args:
        ds_features (dict): The LeRobot dataset features dictionary.
        values (dict): A dictionary of raw values from the hardware/environment.
        prefix (str): The prefix to filter features by (e.g., "observation"
            or "action").

    Returns:
        dict: A dictionary representing a single frame of data.
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset features to policy features.

    This function transforms the dataset's feature specification into a format
    that a policy can use, classifying features by type (e.g., visual, state,
    action) and ensuring correct shapes (e.g., channel-first for images).

    Args:
        features (dict): The LeRobot dataset features dictionary.

    Returns:
        dict: A dictionary mapping feature keys to `PolicyFeature` objects.

    Raises:
        ValueError: If an image feature does not have a 3D shape.
    """
    # TODO(aliberts): Implement "type" in dataset features and simplify this
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            else:
                names = ft["names"]
                # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0 for ported datasets.
                if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                    shape = (shape[2], shape[0], shape[1])
        elif key == OBS_ENV_STATE:
            type = FeatureType.ENV
        elif key.startswith(OBS_STR):
            type = FeatureType.STATE
        elif key.startswith(ACTION):
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


CAMERA_DTYPES = ("image", "video")


def camera_keys_from_features(features: dict[str, dict]) -> list[str]:
    """Return the visual feature keys of a LeRobot features dict, in declaration order.

    A camera is identified by its dtype, not by its name. Naming is not uniform:
    ``lerobot/pusht`` keys its only camera ``observation.image``, and this repo's
    own dataset fixtures use bare names with no prefix at all, while SO-101 and
    ALOHA datasets use ``observation.images.<name>``. Keying off the plural prefix
    finds no cameras in the first two cases, and reports it as a dataset that has
    none rather than as an error.
    """
    return [key for key, ft in features.items() if ft.get("dtype") in CAMERA_DTYPES]


def camera_name(key: str) -> str:
    """Return the short, user-facing name of a camera feature key.

    ``observation.images.top`` -> ``top``. A key that does not carry the plural
    prefix has no shorter form and is returned unchanged, so ``observation.image``
    names itself.
    """
    return key.removeprefix(f"{OBS_IMAGES}.")


def resolve_camera_keys(features: dict[str, dict], cameras: Sequence[str] | None) -> list[str]:
    """Resolve a user-facing camera selection into dataset feature keys.

    Preconditions:
        ``features`` is a LeRobot features dict. Each entry of ``cameras`` is either
        a full feature key (``observation.images.top``) or the short name that
        :func:`camera_name` produces (``top``). ``None`` means "every camera".

    Postconditions:
        The result is a subset of :func:`camera_keys_from_features` in the dataset's
        own feature order, never empty, and never contains duplicates — so callers
        can compare it against the full list to decide whether a restriction applies.

    Raises:
        ValueError: If a name matches no camera, or if the selection resolves to
            nothing. Both are refused rather than silently ignored: a typo would
            otherwise train on more cameras than asked for, which is invisible in
            the loss and only shows up as an input-shape mismatch at deployment.
    """
    available = camera_keys_from_features(features)
    if cameras is None:
        return available

    # A short name can collide with another camera's full key when a dataset mixes
    # naming shapes (a bare ``top`` alongside ``observation.images.top``). Rare, but
    # resolving it by insertion order would silently train on the wrong camera — so
    # collisions are recorded and refused on use, not on sight: an ambiguous pair
    # must not stop you selecting some third, unambiguous camera.
    by_alias: dict[str, list[str]] = {}
    for key in available:
        for alias in {key, camera_name(key)}:
            by_alias.setdefault(alias, []).append(key)

    selected: list[str] = []
    unknown: list[str] = []
    for name in cameras:
        claimants = by_alias.get(name, [])
        if not claimants:
            unknown.append(name)
        elif len(claimants) > 1:
            raise ValueError(
                f"Ambiguous camera name {name!r}: it names {sorted(claimants)}. "
                "Select these cameras by their full feature keys."
            )
        elif claimants[0] not in selected:
            selected.append(claimants[0])

    if unknown:
        raise ValueError(
            f"Unknown camera(s): {sorted(unknown)}. "
            f"This dataset has: {sorted(camera_name(k) for k in available)}"
        )
    if not selected:
        # This assumes an explicit empty selection is a mistake, which holds only while
        # every run wants vision. A dataset with no cameras is already fine under the
        # default (None yields an empty list, no error), so [] can only mean "drop the
        # cameras this dataset does have" -- a state-only run on a camera-carrying
        # dataset would want exactly that. Nothing asks for it today, and reading [] as
        # "every camera" is the worse of the two failures. Relax here and in
        # DatasetConfig.__post_init__ if that case turns up.
        raise ValueError(
            "The camera selection is empty. Pass None to train on every camera; "
            "an explicit empty selection would leave the policy with no visual input."
        )

    return [key for key in available if key in selected]


def combine_feature_dicts(*dicts: dict) -> dict:
    """Merge LeRobot grouped feature dicts.

    - For 1D numeric specs (dtype not image/video/string) with "names": we merge the names and recompute the shape.
    - For others (e.g. `observation.images.*`), the last one wins (if they are identical).

    Args:
        *dicts: A variable number of LeRobot feature dictionaries to merge.

    Returns:
        dict: A single merged feature dictionary.

    Raises:
        ValueError: If there's a dtype mismatch for a feature being merged.
    """
    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )

            if is_vector:
                # Initialize or retrieve the accumulating dict for this feature key
                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})
                # Ensure consistent data types across merged entries
                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")

                # Merge feature names: append only new ones to preserve order without duplicates
                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)
                # Recompute the shape to reflect the updated number of features
                target["shape"] = (len(target["names"]),)
            else:
                # For images/videos and non-1D entries: override with the latest definition
                out[key] = value
    return out
