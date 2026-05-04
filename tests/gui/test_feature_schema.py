# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for ``DatasetInfo.features_schema`` and ``_build_features_schema``.

The schema dict is what powers the GUI's renderer registry — the front-end
dispatches widgets on (dtype, ndim) and uses ``names`` for vector-component
labels. These tests pin the JSON shape so a frontend that reads
``features_schema[name].dtype`` etc. can rely on it.
"""

from __future__ import annotations

from lerobot.gui.api.datasets import FeatureSchema, _build_features_schema


def test_scalar_features() -> None:
    features = {
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "success": {"dtype": "bool", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    schema = _build_features_schema(features)
    assert set(schema.keys()) == {"reward", "success", "frame_index"}
    assert schema["reward"].dtype == "float32"
    assert schema["reward"].shape == [1]  # tuple coerced to list for JSON
    assert schema["reward"].names is None
    assert schema["success"].dtype == "bool"


def test_vector_with_component_names() -> None:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "gripper",
                "j8",
                "j9",
                "j10",
                "j11",
                "j12",
                "j13",
                "j14",
            ],
        }
    }
    schema = _build_features_schema(features)
    assert schema["observation.state"].shape == [14]
    assert schema["observation.state"].names is not None
    assert len(schema["observation.state"].names) == 14
    assert schema["observation.state"].names[0] == "joint1"


def test_image_feature() -> None:
    features = {
        "observation.images.front": {
            "dtype": "image",
            "shape": [3, 480, 640],
            "names": None,
        }
    }
    schema = _build_features_schema(features)
    assert schema["observation.images.front"].dtype == "image"
    assert schema["observation.images.front"].shape == [3, 480, 640]


def test_string_feature() -> None:
    features = {"subtask": {"dtype": "string", "shape": (1,), "names": None}}
    schema = _build_features_schema(features)
    assert schema["subtask"].dtype == "string"
    # No coercion of names for scalars/strings — stays None.
    assert schema["subtask"].names is None


def test_dict_names_collapsed_to_list() -> None:
    """Some upstream features encode names as ``{"motors": [...]}`` —
    flatten to a single list so the GUI doesn't have to handle two shapes."""
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": ["m1", "m2", "m3", "m4", "m5", "m6"]},
        }
    }
    schema = _build_features_schema(features)
    assert schema["action"].names == ["m1", "m2", "m3", "m4", "m5", "m6"]


def test_dict_names_unflattenable_drops_to_none() -> None:
    features = {
        "weird": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"a": "scalar", "b": "scalar"},  # non-list values inside dict
        }
    }
    schema = _build_features_schema(features)
    assert schema["weird"].names is None


def test_missing_dtype_or_shape_doesnt_crash() -> None:
    """``info.json`` for older datasets may have partial entries —
    we should produce a valid (if degraded) FeatureSchema."""
    features = {"weird": {}}
    schema = _build_features_schema(features)
    assert schema["weird"].dtype == ""
    assert schema["weird"].shape == []
    assert schema["weird"].names is None


def test_feature_schema_round_trips_through_json() -> None:
    features = {
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "observation.state": {"dtype": "float32", "shape": [14], "names": ["a"] * 14},
    }
    schema = _build_features_schema(features)
    # Pydantic round-trip — confirms the response is JSON-serializable.
    payload = {name: fs.model_dump() for name, fs in schema.items()}
    re_parsed = {name: FeatureSchema.model_validate(d) for name, d in payload.items()}
    assert re_parsed["reward"].dtype == "float32"
    assert re_parsed["observation.state"].shape == [14]


# ── Subtask synthesis: subtask_index + meta/subtasks.parquet → "subtask" string ──


def test_subtask_synthesis_replaces_storage_with_string_display() -> None:
    """When ``subtask_synthesis=True``, ``subtask_index`` is removed from the
    schema and replaced by a synthetic ``subtask`` (string) entry.

    The user always thinks in terms of strings. Backend rule: synthesize iff
    BOTH ``subtask_index`` is in ``features`` AND the dataset has a lookup
    table (``meta/subtasks.parquet``) — caller passes ``subtask_synthesis=True``
    when that condition is met.
    """
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
        "reward": {"dtype": "float32", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, subtask_synthesis=True)
    assert "subtask_index" not in schema, "storage feature must be hidden"
    assert "subtask" in schema, "display feature must be synthesized"
    sub = schema["subtask"]
    assert sub.dtype == "string"
    assert sub.shape == [1]
    assert sub.names is None
    # Reward is unrelated; should pass through.
    assert "reward" in schema
    assert schema["reward"].dtype == "float32"


def test_subtask_synthesis_disabled_keeps_storage_name() -> None:
    """``subtask_synthesis=False`` (the default for datasets without the
    lookup table) leaves ``subtask_index`` in the schema as-is."""
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, subtask_synthesis=False)
    assert "subtask" not in schema
    assert "subtask_index" in schema
    assert schema["subtask_index"].dtype == "int64"


def test_subtask_synthesis_when_no_subtask_index_is_noop() -> None:
    """Passing ``subtask_synthesis=True`` for a dataset that doesn't have
    ``subtask_index`` should NOT invent a fake ``subtask`` entry."""
    features = {"reward": {"dtype": "float32", "shape": [1], "names": None}}
    schema = _build_features_schema(features, subtask_synthesis=True)
    assert "subtask" not in schema
    assert "reward" in schema


def test_subtask_synthesis_inherits_per_episode_flag() -> None:
    """If ``subtask_index`` is detected as per-episode-broadcast, the
    synthetic ``subtask`` inherits the flag — coercion happens at the
    storage layer, but the user-facing card needs to know to render the
    'whole episode' note."""
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, per_episode={"subtask_index"}, subtask_synthesis=True)
    assert "subtask" in schema
    assert schema["subtask"].is_per_episode is True


def test_subtask_synthesis_does_not_set_per_episode_when_storage_isnt() -> None:
    features = {
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    schema = _build_features_schema(features, per_episode=set(), subtask_synthesis=True)
    assert schema["subtask"].is_per_episode is False
