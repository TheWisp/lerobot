# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What a stored mask row means depends on a label's POSITION, so the
vocabulary can only ever grow at the end.

A row is `[[label_id, rle], ...]` and `label_id` indexes `mask_labels`. Move a
label and every row already written means something else: masks recorded for
the ball are read as the tray, across every episode, with nothing to notice.
Only appending is safe.

Two mechanisms claim to protect this — a `ValueError` in `generate_episode_masks`
and a 409 `mask_labels_differ` from the save endpoint. **Neither can fire.**
The writer normalises the request to `stored + new` before its own check, and
the endpoint compares a list it built as `have + [...]` against `have`, which
is a prefix of itself by construction. What actually happens is silent
normalisation, pinned below.

The invariant that matters does hold: a stored id never moves. What is lost is
the operator's intent — a rename becomes an append, and a reorder is discarded
without a word. Whether that should stay silent is a product decision, not a
defect in the storage; these tests describe today's behaviour so that changing
it is a deliberate act with a failing test attached.
"""

import pytest


# The normalisation the writer applies, transcribed from
# generate_episode_masks, where it is inline in a function that needs a whole
# dataset to call. A transcription asserts nothing about the code on its own,
# so `test_the_real_writer_normalises_the_same_way` drives the actual writer
# over the same cases and compares; these helpers only make the table above
# readable.
def _normalise(stored: list[str], requested: list[str]) -> list[str]:
    if stored and requested[: len(stored)] != stored:
        return stored + [name for name in requested if name not in stored]
    return requested


def _writer_would_raise(stored: list[str], requested: list[str]) -> bool:
    effective = _normalise(stored, requested)
    return effective[: len(stored)] != stored


def _endpoint_would_refuse(have: list[str], requested: list[str]) -> bool:
    merged = have + [name for name in requested if name not in have]
    return merged[: len(have)] != have


STORED = ["ball", "tray"]


@pytest.mark.parametrize(
    ("case", "requested", "effective"),
    [
        # Appending is the supported edit: stored ids keep their meaning.
        ("append", ["ball", "tray", "cup"], ["ball", "tray", "cup"]),
        # Naming a subset is explicitly allowed -- a pass may re-run one object
        # and leave the others' rows alone -- so it is restored, not refused.
        ("subset", ["ball"], ["ball", "tray"]),
        # A reorder is discarded. The data is safe; the request is ignored.
        ("reorder", ["tray", "ball"], ["ball", "tray"]),
        # A rename becomes an APPEND: the old label survives and the new one is
        # added, so the operator ends with three objects, not two renamed.
        ("rename", ["ball", "table"], ["ball", "tray", "table"]),
        ("replace", ["cup"], ["ball", "tray", "cup"]),
    ],
)
def test_the_vocabulary_only_ever_grows_at_the_end(case, requested, effective):
    assert _normalise(STORED, requested) == effective, case
    assert _normalise(STORED, requested)[: len(STORED)] == STORED, (
        f"{case} moved a stored label; every row written before this now means something else"
    )


@pytest.mark.parametrize(
    "requested",
    [["ball", "tray", "cup"], ["ball"], ["tray", "ball"], ["ball", "table"], ["cup"]],
)
def test_neither_documented_refusal_can_fire(requested):
    """Pins the dead code as dead.

    If either mechanism is ever made real, this test fails and says so -- which
    is the point: the refusals are documented in `generate_episode_masks` and in
    the save endpoint's docstring, and a reader has no way to tell they are
    unreachable.
    """
    assert not _writer_would_raise(STORED, requested), (
        "the writer's ValueError is reachable now; its normalisation must have changed"
    )
    assert not _endpoint_would_refuse(STORED, requested), (
        "the endpoint's 409 mask_labels_differ is reachable now; update the docstrings that "
        "describe it and remove the NOT IMPLEMENTED annotations"
    )


def test_the_endpoint_check_is_a_tautology():
    """Why the 409 cannot fire, stated once rather than inferred from the cases.

    `merged` is built as `have + [...]`, so its first `len(have)` entries are
    `have` for any input at all -- including inputs that plainly should be
    refused.
    """
    for requested in ([], ["z"], ["tray", "ball"], ["completely", "different"]):
        merged = STORED + [n for n in requested if n not in STORED]
        assert merged[: len(STORED)] == STORED


# ── the same rule, through the code that actually runs ──────────────────────
# Everything above is a transcription. These drive `generate_episode_masks`
# itself, so a change to the writer that the transcription no longer describes
# fails here rather than passing quietly.

from tests.datasets.test_saved_masks_training import (  # noqa: E402
    masked_dataset_root,  # noqa: F401
)


def _stored_labels(root) -> list[str]:
    import json

    info = json.loads((root / "meta" / "info.json").read_text())
    return next(v["mask_labels"] for v in info["features"].values() if v.get("mask_encoding") == "coco_rle")


@pytest.mark.parametrize(
    ("case", "requested_names", "expected"),
    [
        ("append", ["tray", "ball", "cup"], ["tray", "ball", "cup"]),
        ("subset", ["tray"], ["tray", "ball"]),
        ("reorder", ["ball", "tray"], ["tray", "ball"]),
        ("rename", ["tray", "cup"], ["tray", "ball", "cup"]),
    ],
)
def test_the_real_writer_normalises_the_same_way(
    masked_dataset_root,  # noqa: F811
    case,
    requested_names,
    expected,
):
    """Re-run a masked episode with a different vocabulary and read what the
    dataset ends up declaring. The fixture's stored vocabulary is
    ``["tray", "ball"]``."""
    from lerobot.datasets.dataset_postprocess import generate_episode_masks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tests.datasets.test_saved_masks_training import _StripeAdapter

    root, repo_id = masked_dataset_root
    assert _stored_labels(root) == ["tray", "ball"], "fixture changed; update the cases"

    ds = LeRobotDataset(repo_id, root=root)
    treatment = {"key": "none"}
    result = generate_episode_masks(
        ds,
        episode=0,
        objects=[{"name": n, "sign": "+", "treatment": dict(treatment)} for n in requested_names],
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=_StripeAdapter(),
    )
    assert not result.get("cancelled")
    assert _stored_labels(root) == expected, (
        f"{case}: the writer's effective vocabulary is not what the transcription predicts"
    )
    # The invariant the whole scheme rests on, checked against the real file.
    assert _stored_labels(root)[:2] == ["tray", "ball"], (
        "a stored label moved; every mask row written before this now means something else"
    )


def test_the_real_writer_never_raises_for_these(masked_dataset_root):  # noqa: F811
    """The documented ValueError, against the running code rather than a copy
    of its condition."""
    from lerobot.datasets.dataset_postprocess import generate_episode_masks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tests.datasets.test_saved_masks_training import _StripeAdapter

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    generate_episode_masks(
        ds,
        episode=0,
        objects=[{"name": n, "sign": "+", "treatment": {"key": "none"}} for n in ("ball", "tray")],
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=_StripeAdapter(),
    )  # a reorder: documented as refused, in fact normalised


# ── one vocabulary across cameras, through the real writer ──────────────────


def _labels_per_column(root) -> dict[str, list[str]]:
    import json

    info = json.loads((root / "meta" / "info.json").read_text())
    return {k: v["mask_labels"] for k, v in info["features"].items() if v.get("mask_encoding") == "coco_rle"}


def test_a_one_camera_pass_does_not_split_the_vocabulary(masked_dataset_root):  # noqa: F811
    """The defect this exists for. Segmenting a new object with ONE camera
    selected used to append the name only to that camera's column, after which
    a rename or a treatment applied by name reached some views and not others
    -- with every mask still decoding correctly, because rows reference
    positions and no position moved."""
    from lerobot.datasets.dataset_postprocess import generate_episode_masks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tests.datasets.test_saved_masks_training import _StripeAdapter

    root, repo_id = masked_dataset_root
    before = _labels_per_column(root)
    assert len(before) == 2, "this test needs two masked cameras"
    assert len(set(map(tuple, before.values()))) == 1, "fixture starts consistent"

    ds = LeRobotDataset(repo_id, root=root)
    one_camera = sorted(before)[0].replace("masks.", "observation.images.")
    result = generate_episode_masks(
        ds,
        episode=0,
        objects=[{"name": n, "sign": "+", "treatment": {"key": "none"}} for n in ("tray", "banana")],
        cameras=[one_camera],
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=_StripeAdapter(),
    )
    assert not result.get("cancelled")

    after = _labels_per_column(root)
    assert "banana" in next(iter(after.values())), "the new object was not declared at all"
    assert len(set(map(tuple, after.values()))) == 1, (
        f"cameras drifted apart: {after} -- the same object is now named per view"
    )
    for labels in after.values():
        assert labels[: len(before[next(iter(before))])] == before[next(iter(before))], (
            "a stored label moved; rows written before this now mean something else"
        )


def test_a_one_camera_pass_writes_rows_only_where_it_looked(masked_dataset_root):  # noqa: F811
    """The complement: declaring a label everywhere must not fabricate
    coverage. Without this, "the vocabularies match" would also be satisfied by
    a writer that segmented every camera regardless of the selection."""
    from lerobot.datasets.dataset_postprocess import generate_episode_masks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import coverage
    from tests.datasets.test_saved_masks_training import _StripeAdapter

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    cams = sorted(_labels_per_column(root))
    picked = cams[0].replace("masks.", "observation.images.")
    other = cams[1].replace("masks.", "observation.images.")
    before_other = coverage(ds, 0, other)

    generate_episode_masks(
        ds,
        episode=0,
        objects=[{"name": "banana", "sign": "+", "treatment": {"key": "none"}}],
        cameras=[picked],
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=_StripeAdapter(),
    )
    ds2 = LeRobotDataset(repo_id, root=root)
    assert coverage(ds2, 0, other) == before_other, "the unselected camera gained rows"


def test_a_save_does_not_reset_stored_treatments(masked_dataset_root):  # noqa: F811
    """Treatments are edited in the Inspector; a segmentation pass has no
    opinion about them.

    The caller still sends a treatment per object, defaulting to "none", so
    preferring it would reset every label the pass named -- segment again to add
    one object, and every other object's effect is silently gone.
    """
    import json

    from lerobot.datasets.dataset_postprocess import generate_episode_masks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tests.datasets.test_saved_masks_training import _StripeAdapter

    root, repo_id = masked_dataset_root
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for name, feat in info["features"].items():
        if name.startswith("masks.") and "mask_labels" in feat:
            feat["mask_treatments"] = {"tray": {"key": "blur", "params": {"strength": 0.7}}}
    info_path.write_text(json.dumps(info, indent=2))

    ds = LeRobotDataset(repo_id, root=root)
    generate_episode_masks(
        ds,
        episode=0,
        # Names "tray" with the default treatment the panel sends.
        objects=[{"name": n, "sign": "+", "treatment": {"key": "none"}} for n in ("tray", "ball")],
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=_StripeAdapter(),
    )

    after = json.loads(info_path.read_text())["features"]
    stored = next(v["mask_treatments"] for v in after.values() if v.get("mask_encoding") == "coco_rle")
    assert stored["tray"]["key"] == "blur", f"the save reset a stored treatment: {stored}"


# ── the write rule ──────────────────────────────────────────────────────────
# Masks cost ~8 h to compute for a large dataset and nothing to delete, so a
# pass fills gaps and leaves what is there alone. Without this a re-run
# silently replaced hours of segmentation, and muting a bad detection was
# pointless because the next pass put it straight back.


def _run(ds, labels, adapter=None):
    from lerobot.datasets.dataset_postprocess import generate_episode_masks
    from tests.datasets.test_saved_masks_training import _StripeAdapter

    return generate_episode_masks(
        ds,
        episode=0,
        objects=[{"name": n, "sign": "+", "treatment": {"key": "none"}} for n in labels],
        cameras=None,
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=adapter or _StripeAdapter(),
    )


def test_a_second_pass_does_not_replace_what_the_first_stored(masked_dataset_root):  # noqa: F811
    """A re-run adds; it does not repair by replacement."""
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import read_frame

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    _run(ds, ["tray"])
    ds = LeRobotDataset(repo_id, root=root)
    first = read_frame(ds, 0, 0, "observation.images.top")["tray"].copy()

    # A second pass whose detection for the SAME label is a different shape.
    class Shifted:
        def set_camera(self, cam):
            pass

        def reset(self):
            pass

        def set_control(self, c):
            pass

        def segment(self, rgb):
            m = np.zeros(rgb.shape[:2], bool)
            m[-3:, -3:] = True
            return {"tray": m}

    _run(ds, ["tray"], adapter=Shifted())
    ds = LeRobotDataset(repo_id, root=root)
    after = read_frame(ds, 0, 0, "observation.images.top")["tray"]
    assert np.array_equal(after, first), "the second pass overwrote a stored mask"


def test_a_pass_fills_a_label_that_was_absent(masked_dataset_root):  # noqa: F811
    """The complement, and the documented repair loop: delete, then run again.

    Without it "leaves what is there alone" would be satisfied by a writer that
    does nothing at all.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import delete_label_range, read_frame

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    _run(ds, ["tray", "ball"])

    ds = LeRobotDataset(repo_id, root=root)
    delete_label_range(ds, 0, "observation.images.top", "ball")
    assert "ball" not in read_frame(ds, 0, 0, "observation.images.top"), "the delete did not land"

    _run(ds, ["tray", "ball"])
    ds = LeRobotDataset(repo_id, root=root)
    assert "ball" in read_frame(ds, 0, 0, "observation.images.top"), (
        "the gap left by a delete was not refilled"
    )


def test_a_pass_leaves_a_disabled_mask_disabled(masked_dataset_root):  # noqa: F811
    """The case muting exists for. A muted mask that came back enabled would
    silently rejoin training, and the next pass would undo every repair."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import set_label_enabled, states

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    _run(ds, ["tray"])
    ds = LeRobotDataset(repo_id, root=root)
    set_label_enabled(ds, 0, "observation.images.top", "tray", False, frames=range(0, 2))
    assert states(ds, 0, "observation.images.top")[0]["tray"] is False

    _run(ds, ["tray"])
    ds = LeRobotDataset(repo_id, root=root)
    assert states(ds, 0, "observation.images.top")[0]["tray"] is False, (
        "a pass re-enabled a muted mask; the mute cannot survive a re-run"
    )


# ── the boundary: a stopped run keeps what it computed ──────────────────────


def test_cancelling_keeps_the_frames_already_segmented(masked_dataset_root):  # noqa: F811
    """Cancelling used to return having written nothing, discarding every frame
    already computed.

    The loss is invisible, which is what makes it serious: the tracks show
    those frames as absent, exactly as if they had never been segmented, so an
    operator who stopped a long run would see no sign the work existed.
    """
    import numpy as np

    from lerobot.datasets.dataset_postprocess import generate_episode_masks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import coverage, delete_label_range

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    cam = "observation.images.top"
    # Clear the fixture's masks so coverage counts only what this run writes.
    for name in ("tray", "ball"):
        delete_label_range(ds, 0, cam, name)
    ds = LeRobotDataset(repo_id, root=root)
    assert coverage(ds, 0, cam)[0] == 0, "the fixture was not cleared"

    length = int(ds.meta.episodes["length"][0])
    assert length >= 3, "need a few frames to stop part-way through"

    seen = {"n": 0}

    class Counting:
        """Segments normally, and asks to stop after a couple of frames."""

        def set_camera(self, c):
            pass

        def reset(self):
            pass

        def set_control(self, c):
            pass

        def segment(self, rgb):
            seen["n"] += 1
            m = np.zeros(rgb.shape[:2], bool)
            m[:4] = True
            return {"tray": m}

    stop_after = 2
    result = generate_episode_masks(
        ds,
        episode=0,
        objects=[{"name": "tray", "sign": "+", "treatment": {"key": "none"}}],
        cameras=[cam],
        model="sam3_track",
        resolution=None,
        multi_instance=True,
        background_treatment={"key": "none", "params": {}},
        adopt=False,
        device="cpu",
        adapter=Counting(),
        should_cancel=lambda: seen["n"] >= stop_after,
    )
    assert result["cancelled"] is True, result

    ds = LeRobotDataset(repo_id, root=root)
    kept, _ = coverage(ds, 0, cam)
    assert kept > 0, (
        f"the run stopped after {seen['n']} frames and kept none of them; "
        "everything computed before the stop was discarded"
    )
    assert kept <= length


# ── two rows, one name ──────────────────────────────────────────────────────


def test_a_repeated_object_name_is_stored_once(masked_dataset_root):  # noqa: F811
    """The vocabulary is positional, so a name may hold exactly one id.

    Nothing stops an operator naming two object rows the same thing — the
    overlay panel is free text and offers "+ Add object" — and the writer used
    to store the list verbatim. Two ids sharing a name means a stored row can
    decode to either, the timeline draws two identical lanes, and every by-name
    lookup silently resolves to one of them. Found in the GUI, where the label
    coverage endpoint deduped and the column did not, so the two disagreed
    about how many labels the dataset had.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import mask_columns, read_frame

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    _run(ds, ["tray", "ball", "tray"])

    ds = LeRobotDataset(repo_id, root=root)
    for key in set(mask_columns(ds).values()):
        stored = list(ds.meta.features[key]["mask_labels"])
        assert len(stored) == len(set(stored)), f"{key} stored a duplicate label: {stored}"
        # First occurrence wins, so an id already in use never moves.
        assert stored[:2] == ["tray", "ball"], stored

    # And the repeat is not simply dropped along with the object: the label is
    # still segmented, or this would pass for a writer that ignored it.
    assert "tray" in read_frame(ds, 0, 0, "observation.images.top")


def test_a_repeated_name_does_not_move_a_stored_id(masked_dataset_root):  # noqa: F811
    """A second pass repeating a name must still append only what is new.

    The dedup runs before the stored-vocabulary normalisation, so this pins that
    the two compose: a duplicate cannot push an existing label to a new index,
    which would reinterpret every row already written.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import mask_columns

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    _run(ds, ["tray", "ball"])
    ds = LeRobotDataset(repo_id, root=root)
    before = {k: list(ds.meta.features[k]["mask_labels"]) for k in set(mask_columns(ds).values())}

    _run(ds, ["tray", "tray", "ball", "cube", "cube"])
    ds = LeRobotDataset(repo_id, root=root)
    for key, was in before.items():
        now = list(ds.meta.features[key]["mask_labels"])
        assert now[: len(was)] == was, f"{key}: a stored id moved, {was} -> {now}"
        assert now == was + ["cube"], f"{key}: expected one append, got {now}"


# ── a pass with nothing to find must not run the model ───────────────────────


class _CountingAdapter:
    """Wraps the deterministic stripe adapter and counts frames segmented."""

    def __init__(self):
        from tests.datasets.test_saved_masks_training import _StripeAdapter

        self.inner = _StripeAdapter()
        self.frames = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    # Deliberately no `segment_many`: the wrapped adapter has none, and
    # advertising one sends the pass down a path the inner object cannot serve.
    def segment(self, rgb):
        self.frames += 1
        return self.inner.segment(rgb)


def test_a_pass_over_covered_ground_does_not_run_the_model(masked_dataset_root):  # noqa: F811
    """The write rule is applied AFTER segmentation, so a re-run used to segment
    every frame and discard every result -- costing exactly as much as the first
    pass. Measured on a real 294-frame two-camera episode: 18.7 s either way,
    92% of it in the model.

    Asserted by counting frames segmented rather than by timing, so it states
    what actually has to hold and cannot go flaky on a slow machine.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import delete_label_range

    root, repo_id = masked_dataset_root
    # The fixture arrives already covered, so make a real gap first -- otherwise
    # the "first" pass skips too and the comparison below is between two skips.
    ds = LeRobotDataset(repo_id, root=root)
    for cam in ds.meta.camera_keys:
        delete_label_range(ds, 0, cam, "ball")

    first = _CountingAdapter()
    _run(LeRobotDataset(repo_id, root=root), ["tray", "ball"], adapter=first)
    assert first.frames > 0, "the first pass segmented nothing; the rest proves nothing"

    second = _CountingAdapter()
    result = _run(LeRobotDataset(repo_id, root=root), ["tray", "ball"], adapter=second)
    assert result.get("skipped") is True, "a fully covered episode was not skipped"
    assert second.frames == 0, f"the model ran {second.frames} times over ground that was already covered"
    assert result["frames_done"] == result["frames_total"], "a skip must still report the episode done"


def test_a_pass_still_runs_when_one_requested_label_is_missing(masked_dataset_root):  # noqa: F811
    """The complement, and the case that decides the check is worth anything: a
    label the dataset has never seen is a real gap, so the pass must look for it
    even though every other requested label is present. Without this the test
    above passes for a pass that skips unconditionally.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, repo_id = masked_dataset_root
    _run(LeRobotDataset(repo_id, root=root), ["tray", "ball"])

    counting = _CountingAdapter()
    result = _run(LeRobotDataset(repo_id, root=root), ["tray", "ball", "cube"], adapter=counting)
    assert not result.get("skipped"), "a missing label was treated as covered"
    assert counting.frames > 0, "the pass did not look for the label that was absent"


# ── the pieces lifted out of generate_episode_masks ──────────────────────────
#
# They were inline in a 431-line function and had no tests of their own: the
# only way to reach the dedupe rule or the camera refusal was to run a whole
# segmentation pass. Extracted, they are enumerable, so the rules are pinned
# here rather than implied by a pass that happens to work.


def test_requested_vocabulary_dedupes_and_keeps_what_this_run_asked_for():
    from lerobot.datasets.dataset_postprocess import _requested_vocabulary

    objects = [
        {"name": "tray", "treatment": {"key": "tint"}},
        {"name": "  ", "treatment": {"key": "blur"}},  # blank: not a label
        {"name": "ball"},  # no treatment: none
        {"name": "tray", "treatment": {"key": "blur"}},  # repeat: first wins
    ]
    labels, requested, treatments = _requested_vocabulary(objects)
    assert labels == ["tray", "ball"], labels
    assert requested == labels and requested is not labels, "requested must be a snapshot, not an alias"
    assert treatments["ball"] == {"key": "none"}, "an object with no treatment is untreated"
    assert treatments["tray"]["key"] == "blur", "the later entry supplies the treatment for a repeat"
    assert "" not in treatments and "  " not in treatments


def test_requested_vocabulary_refuses_a_run_with_nothing_named():
    from lerobot.datasets.dataset_postprocess import _requested_vocabulary

    with pytest.raises(ValueError, match="no named objects"):
        _requested_vocabulary([{"name": "   "}, {"treatment": {"key": "tint"}}])


def test_resolve_cameras_subsets_and_derives_the_mask_column(masked_dataset_root):  # noqa: F811
    from lerobot.datasets.dataset_postprocess import _resolve_cameras
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    every, keys = _resolve_cameras(ds, None)
    assert every == list(ds.meta.camera_keys), "None means every camera"
    assert all(keys[c].startswith("masks.") for c in every), keys

    one = every[:1]
    subset, _ = _resolve_cameras(ds, one)
    assert subset == one, "a run writes the cameras it was given and no others"

    with pytest.raises(ValueError, match="no camera keys selected"):
        _resolve_cameras(ds, ["observation.images.nonexistent"])
