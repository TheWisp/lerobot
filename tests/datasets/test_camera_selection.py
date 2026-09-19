# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Camera selection: a training run consuming a subset of a dataset's cameras.

The load-bearing claim is that restricting ``meta.features`` is enough — that
decode, download and policy-feature derivation all follow from it without any
per-policy support. Each of those is pinned separately below, because a change
that filters only the returned sample would still pass a naive shape assertion
while paying the full decode cost this exists to remove.
"""

from unittest.mock import patch

import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.utils.feature_utils import (
    camera_keys_from_features,
    camera_name,
    dataset_to_policy_features,
    resolve_camera_keys,
)
from tests.fixtures.constants import DUMMY_REPO_ID

# Keys used by the pure-resolver tests, which build their own features dicts.
LAPTOP_KEY = "observation.images.laptop"
PHONE_KEY = "observation.images.phone"

# The shared dataset fixture stores its two cameras under bare keys, with no
# ``observation.images.`` prefix at all — a third naming shape in the wild, and
# the reason the resolver keys off dtype rather than any prefix.
LAPTOP = "laptop"
PHONE = "phone"


def _features(*keys: str, dtype: str = "video") -> dict[str, dict]:
    """A minimal features dict carrying the given camera keys plus a state vector."""
    features: dict[str, dict] = {"observation.state": {"dtype": "float32", "shape": (6,), "names": ["a"] * 6}}
    for key in keys:
        features[key] = {"dtype": dtype, "shape": (3, 64, 96), "names": ["channels", "height", "width"]}
    return features


# --------------------------------------------------------------------------
# resolve_camera_keys
# --------------------------------------------------------------------------


def test_none_selects_every_camera():
    assert resolve_camera_keys(_features(LAPTOP_KEY, PHONE_KEY), None) == [LAPTOP_KEY, PHONE_KEY]


def test_short_names_and_full_keys_resolve_to_the_same_thing():
    features = _features(LAPTOP_KEY, PHONE_KEY)
    assert resolve_camera_keys(features, ["laptop"]) == [LAPTOP_KEY]
    assert resolve_camera_keys(features, [LAPTOP_KEY]) == [LAPTOP_KEY]
    assert resolve_camera_keys(features, ["laptop", PHONE_KEY]) == [LAPTOP_KEY, PHONE_KEY]


def test_result_follows_dataset_order_not_the_order_asked_for():
    """Feature order decides channel order downstream, so it must not depend on argv."""
    assert resolve_camera_keys(_features(LAPTOP_KEY, PHONE_KEY), ["phone", "laptop"]) == [
        LAPTOP_KEY,
        PHONE_KEY,
    ]


def test_a_repeated_name_is_not_a_repeated_camera():
    assert resolve_camera_keys(_features(LAPTOP_KEY, PHONE_KEY), ["laptop", "laptop"]) == [LAPTOP_KEY]


def test_an_unknown_name_is_refused_and_names_what_is_available():
    with pytest.raises(ValueError, match="Unknown camera") as exc:
        resolve_camera_keys(_features(LAPTOP_KEY, PHONE_KEY), ["labtop"])
    # The message has to carry the alternatives: this is the only place a typo is
    # catchable, since every downstream stage would just see one fewer camera.
    assert "labtop" in str(exc.value)
    assert "laptop" in str(exc.value)
    assert "phone" in str(exc.value)


def test_an_empty_selection_is_refused_rather_than_meaning_everything():
    with pytest.raises(ValueError, match="empty"):
        resolve_camera_keys(_features(LAPTOP_KEY, PHONE_KEY), [])


def test_a_name_claimed_by_two_cameras_is_refused():
    """A dataset mixing naming shapes can make one short name mean two things."""
    features = _features("top", LAPTOP_KEY)
    features["observation.images.top"] = dict(features["top"])

    with pytest.raises(ValueError, match="Ambiguous camera name"):
        resolve_camera_keys(features, ["top"])

    # The full keys stay unambiguous, which is what the message tells you to use.
    assert resolve_camera_keys(features, ["observation.images.top"]) == ["observation.images.top"]

    # And an unrelated camera is still selectable — the ambiguity is refused where
    # it is used, not merely where it exists.
    assert resolve_camera_keys(features, ["laptop"]) == [LAPTOP_KEY]


def test_a_camera_outside_the_plural_prefix_is_still_a_camera():
    """PushT-style datasets key a single camera as ``observation.image``.

    Identifying cameras by dtype rather than by the ``observation.images.``
    prefix is what makes this work for those datasets at all.
    """
    features = _features("observation.image")
    assert camera_keys_from_features(features) == ["observation.image"]
    assert resolve_camera_keys(features, ["observation.image"]) == ["observation.image"]
    assert camera_name("observation.image") == "observation.image"


def test_image_dtype_cameras_are_cameras_too():
    features = _features(LAPTOP_KEY, dtype="image")
    assert camera_keys_from_features(features) == [LAPTOP_KEY]


# --------------------------------------------------------------------------
# The metadata view
# --------------------------------------------------------------------------


def test_the_full_selection_returns_the_same_object(lerobot_dataset_factory, tmp_path):
    meta = lerobot_dataset_factory(root=tmp_path / "full").meta
    assert meta.restricted_to_cameras(None) is meta
    assert meta.restricted_to_cameras(sorted(meta.camera_keys)) is meta


def test_the_view_drops_only_the_unselected_cameras(lerobot_dataset_factory, tmp_path):
    meta = lerobot_dataset_factory(root=tmp_path / "view").meta
    view = meta.restricted_to_cameras(["laptop"])

    assert view.camera_keys == [LAPTOP]
    assert view.video_keys == [LAPTOP]
    assert PHONE not in view.features
    # Non-camera features are untouched.
    assert "state" in view.features
    assert "action" in view.features
    assert "timestamp" in view.features
    assert view.fps == meta.fps
    assert view.total_episodes == meta.total_episodes


def test_the_view_does_not_mutate_the_metadata_it_came_from(lerobot_dataset_factory, tmp_path):
    meta = lerobot_dataset_factory(root=tmp_path / "immutable").meta
    before = sorted(meta.camera_keys)
    meta.restricted_to_cameras(["laptop"])
    assert sorted(meta.camera_keys) == before


def test_writing_through_a_restricted_view_is_refused(lerobot_dataset_factory, tmp_path):
    """Persisting a view would write an info.json that has silently lost cameras."""
    view = lerobot_dataset_factory(root=tmp_path / "readonly", cameras=["laptop"]).meta

    with pytest.raises(RuntimeError, match="camera-restricted"):
        view.save_episode(
            episode_index=0,
            episode_length=1,
            episode_tasks=["t"],
            episode_stats={},
            episode_metadata={},
        )
    with pytest.raises(RuntimeError, match="camera-restricted"):
        view.update_chunk_settings(chunks_size=10)

    # An unrestricted dataset's metadata is still writable — the guard is on the
    # view, not on the class.
    unrestricted = lerobot_dataset_factory(root=tmp_path / "writable").meta
    unrestricted.update_chunk_settings(chunks_size=10)


# --------------------------------------------------------------------------
# End to end through LeRobotDataset
# --------------------------------------------------------------------------


def test_an_unselected_camera_is_absent_from_the_sample(lerobot_dataset_factory, tmp_path):
    dataset = lerobot_dataset_factory(root=tmp_path / "sample", cameras=["laptop"])
    item = dataset[0]

    assert LAPTOP in item
    assert PHONE not in item
    assert dataset.cameras == [LAPTOP]


def test_an_unselected_camera_is_never_decoded(lerobot_dataset_factory, tmp_path):
    """The point of the whole design: the frames are not produced, not discarded.

    Filtering the sample after ``__getitem__`` would satisfy the test above while
    still paying decode and shared-memory collation for every camera — which is
    the cost this feature exists to remove.
    """
    from lerobot.datasets import dataset_reader

    dataset = lerobot_dataset_factory(root=tmp_path / "decode", cameras=["laptop"])
    real_decode = dataset_reader.decode_video_frames
    decoded_paths = []

    def _spy(video_path, *args, **kwargs):
        decoded_paths.append(str(video_path))
        return real_decode(video_path, *args, **kwargs)

    with patch.object(dataset_reader, "decode_video_frames", _spy):
        dataset[0]

    assert decoded_paths, "expected the selected camera to be decoded"
    assert all("laptop" in path for path in decoded_paths), decoded_paths
    assert not any("phone" in path for path in decoded_paths), decoded_paths


def test_the_policy_sees_exactly_the_selected_cameras(lerobot_dataset_factory, tmp_path):
    """make_policy derives input_features from ds_meta, so this is what lands in the checkpoint."""
    dataset = lerobot_dataset_factory(root=tmp_path / "policy", cameras=["laptop"])
    features = dataset_to_policy_features(dataset.meta.features)

    assert LAPTOP in features
    assert PHONE not in features


def test_no_selection_leaves_every_camera(lerobot_dataset_factory, tmp_path):
    dataset = lerobot_dataset_factory(root=tmp_path / "all")
    item = dataset[0]

    assert LAPTOP in item
    assert PHONE in item
    assert sorted(dataset.cameras) == [LAPTOP, PHONE]


def test_a_typo_fails_at_construction(lerobot_dataset_factory, tmp_path):
    with pytest.raises(ValueError, match="Unknown camera"):
        lerobot_dataset_factory(root=tmp_path / "typo", cameras=["labtop"])


# --------------------------------------------------------------------------
# The training config
# --------------------------------------------------------------------------


def test_dataset_config_refuses_an_explicitly_empty_selection():
    with pytest.raises(ValueError, match="at least one camera"):
        DatasetConfig(repo_id=DUMMY_REPO_ID, cameras=[])


def test_dataset_config_refuses_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id=DUMMY_REPO_ID, cameras=["laptop", "laptop"])


def test_dataset_config_defaults_to_every_camera():
    assert DatasetConfig(repo_id=DUMMY_REPO_ID).cameras is None


# --------------------------------------------------------------------------
# What a policy is actually built with
# --------------------------------------------------------------------------


def test_no_registered_policy_can_opt_out_of_the_selection():
    """The mechanism is derivation, and this checks every policy still derives.

    No policy declares a camera-selection field, and none needs to: each config's
    ``image_features`` is computed from ``input_features``, which the factory fills
    from ``ds_meta.features``. Restricting the metadata therefore narrows every
    registered policy at once. A policy that hardcoded its camera list, or cached
    ``image_features`` at construction, would silently ignore the selection — and
    would fail here rather than at someone's deployment.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.gui.api.training import _ensure_policy_configs_loaded

    # Registration is lazy: the decorators only run once each policy package is
    # imported. Reuse the GUI's discovery so this covers exactly the set the
    # catalog offers, rather than a hand-written list that would go stale.
    _ensure_policy_configs_loaded()

    # Prefixed keys: gaussian_actor identifies its image features by name rather
    # than by FeatureType, so a bare key would be invisible to it for reasons that
    # have nothing to do with camera selection.
    both = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        LAPTOP_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
        PHONE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 96)),
    }
    one = {key: ft for key, ft in both.items() if key != PHONE_KEY}

    choices = PreTrainedConfig.get_known_choices()
    assert choices, "expected registered policies"

    for name, cls in sorted(choices.items()):
        # Only the property is exercised — no weights, no optional extras.
        cfg = cls.__new__(cls)
        cfg.input_features = dict(both)
        assert PHONE_KEY in cfg.image_features, f"{name} does not see both cameras to begin with"

        cfg.input_features = dict(one)
        assert PHONE_KEY not in cfg.image_features, f"{name} ignores the camera selection"
        assert LAPTOP_KEY in cfg.image_features, f"{name} lost the selected camera"


def test_a_policy_is_built_with_only_the_selected_cameras(lerobot_dataset_factory, tmp_path):
    """The whole claim, exercised through make_policy rather than around it.

    The test above pins the derivation for every registered policy from the config
    side; this one runs it end to end through ``make_policy`` for a policy that
    builds on the base install, so the two halves are not both theory.
    """
    from lerobot.policies.factory import make_policy, make_policy_config

    # ACT, because it is the one registered policy that builds on the base install
    # in well under a second; pi0 takes ~47s and diffusion needs the extra.
    dataset = lerobot_dataset_factory(root=tmp_path / "act", cameras=["laptop"])
    policy = make_policy(make_policy_config("act"), ds_meta=dataset.meta)

    assert LAPTOP in policy.config.image_features
    assert PHONE not in policy.config.image_features
    # And the checkpoint records it, so inference asks the robot for this camera only.
    assert PHONE not in policy.config.input_features


def test_a_policy_trains_on_the_narrowed_batch(lerobot_dataset_factory, tmp_path):
    """Configured for one camera is not the same claim as runs on one camera.

    The test above proves ``make_policy`` builds ACT with a single camera. This one
    takes a step with a batch that carries only that camera, so a model that had
    quietly kept a second vision tower — or that indexed the dropped key — fails
    here rather than at the first real training run.
    """
    import torch

    from lerobot.policies.factory import make_policy, make_policy_config

    dataset = lerobot_dataset_factory(root=tmp_path / "fwd", cameras=["laptop"])
    policy = make_policy(make_policy_config("act", device="cpu"), ds_meta=dataset.meta)
    policy.train()

    batch_size, horizon = 2, policy.config.chunk_size
    height, width = dataset.meta.features[LAPTOP]["shape"][:2]
    batch = {
        "observation.state": torch.zeros(batch_size, len(dataset.meta.features["state"]["names"])),
        LAPTOP: torch.zeros(batch_size, 3, height, width),
        "action": torch.zeros(batch_size, horizon, len(dataset.meta.features["action"]["names"])),
        "action_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool),
    }
    # PHONE is deliberately absent: the batch a narrowed dataset produces does not
    # contain it, so the model must not reach for it.
    assert PHONE not in batch

    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss), "a step on the narrowed batch must produce a usable loss"


def test_a_dataset_with_no_cameras_needs_no_selection(lerobot_dataset_factory, tmp_path):
    """State-only training is untouched by any of this.

    The refusal of an explicit empty selection could be read as "a run must have a
    camera". It is not: a dataset that declares none works under the default, which
    is the case that actually occurs. Only ``cameras=[]`` on a dataset that *has*
    cameras is refused, and that restriction is a choice recorded in
    :func:`resolve_camera_keys`, not a requirement of the design.
    """
    dataset = lerobot_dataset_factory(root=tmp_path / "nocam", camera_features={})

    assert dataset.cameras == []
    assert resolve_camera_keys(dataset.meta.features, None) == []

    item = dataset[0]
    assert "state" in item, "a state-only run must still get its inputs"


def test_every_method_that_persists_info_refuses_a_restricted_view():
    """The guard list is derived, not remembered.

    A restricted view's ``info`` has lost cameras, so persisting it would rewrite
    ``meta/info.json`` without them — the dataset would lose those cameras on disk
    while every other process still expected them. Three methods write ``self.info``
    today and all three refuse a view, but that was true because someone grepped
    once. A fourth added later would silently reintroduce the hazard, so the
    correspondence is checked here instead of asserted in a comment.

    ``create`` is excluded on purpose: it is a classmethod that writes a freshly
    built object's ``info``, which cannot be a view of anything.
    """
    import ast
    import inspect

    from lerobot.datasets import dataset_metadata

    source = inspect.getsource(dataset_metadata)
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeRobotDatasetMetadata"
    )

    unguarded = []
    for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
        body = ast.get_source_segment(source, fn) or ""
        called = {
            getattr(call.func, "id", getattr(call.func, "attr", None))
            for call in ast.walk(fn)
            if isinstance(call, ast.Call)
        }
        persists_own_info = "write_info" in called and "self.info" in body
        if persists_own_info and "_refuse_if_camera_restricted" not in called:
            unguarded.append(fn.name)

    assert not unguarded, (
        f"{unguarded} persist self.info without refusing a camera-restricted view. "
        "Writing through a view rewrites meta/info.json with cameras missing. "
        "Call self._refuse_if_camera_restricted('<what it does>') first."
    )


def test_restricting_a_view_again_still_names_everything_missing(lerobot_dataset_factory, tmp_path):
    """A second restriction narrows relative to the first, so the guard must accumulate."""
    meta = lerobot_dataset_factory(root=tmp_path / "twice").meta
    once = meta.restricted_to_cameras([LAPTOP, PHONE])  # same object: nothing dropped yet
    twice = once.restricted_to_cameras([LAPTOP])

    assert twice.camera_keys == [LAPTOP]
    with pytest.raises(RuntimeError, match=PHONE):
        twice.update_chunk_settings(chunks_size=10)
