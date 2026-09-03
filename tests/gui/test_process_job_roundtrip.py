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

"""Every field of a process job must survive the trip to the worker.

`ProcessJobConfig` is a dataclass with a hand-written to_json/from_json that
names each field explicitly, so a field added to the dataclass alone is dropped
in transit — silently, because the worker then reads the dataclass default.

That is not hypothetical. `kind` was added to route stereo-split jobs to a
different transform, the serializer was not updated, and every split job reached
the worker as a segmentation job and died on "no camera keys selected to
process" — an error about the wrong transform entirely.

So this compares against the dataclass's own field list rather than a list
repeated here, which would rot the same way.
"""

import dataclasses

from lerobot.gui.process_jobs import ProcessJobConfig


def _config(**overrides) -> ProcessJobConfig:
    base = {
        "job_id": "abc123",
        "source_id": "/data/src",
        "source_repo_id": "owner/src",
        "source_root": "/data/src",
        "out_repo_id": "owner/out",
        "out_root": "/data/out",
        "model": "sam3_track",
        "objects": [{"name": "ball", "sign": "+", "treatment": {"key": "none"}}],
        "background_treatment": {"key": "random", "params": {}},
        "apply_mode": "per_episode",
        "variants": 2,
        "multi_instance": True,
        "cameras": ["top"],
        "episodes": [0, 3],
        "preview": False,
        "jobs_dir": "/jobs",
        "resolution": 1008,
        "kind": "split_stereo",
        # Every value here must differ from the field's default, or dropping
        # the field from to_json round-trips to the same value and the test
        # below passes without transporting anything. Pinned by
        # `test_the_fixture_uses_a_non_default_for_every_defaulted_field`.
        "adopt": True,
    }
    base.update(overrides)
    return ProcessJobConfig(**base)


def test_every_declared_field_survives_the_round_trip():
    original = _config()
    restored = ProcessJobConfig.from_json(original.to_json())
    for field in dataclasses.fields(ProcessJobConfig):
        assert getattr(restored, field.name) == getattr(original, field.name), (
            f"{field.name} was dropped by to_json/from_json"
        )


def test_the_fixture_uses_a_non_default_for_every_defaulted_field():
    """Guard the guard above: it is generic over fields, its fixture was not.

    `test_every_declared_field_survives_the_round_trip` compares each field
    before and after, so a field left at its default is invisible to it -- drop
    that field from to_json and from_json supplies the same default, and the
    comparison holds. `adopt` was exactly that: removing it from the
    serializer changed nothing the suite could see.
    """
    cfg = _config()
    missed = [
        f.name
        for f in dataclasses.fields(ProcessJobConfig)
        if f.default is not dataclasses.MISSING and getattr(cfg, f.name) == f.default
    ]
    assert not missed, (
        f"the round-trip fixture leaves {missed} at the default, so dropping "
        f"{'it' if len(missed) == 1 else 'them'} from to_json would not fail any test"
    )


def test_kind_reaches_the_worker():
    # The specific defect: a split job arriving as a segmentation job.
    restored = ProcessJobConfig.from_json(_config(kind="split_stereo").to_json())
    assert restored.kind == "split_stereo"


def test_configs_written_before_kind_existed_still_load():
    import json

    raw = json.loads(_config().to_json())
    del raw["kind"]
    restored = ProcessJobConfig.from_json(json.dumps(raw))
    assert restored.kind == "segment"


def test_episode_masks_fields_survive_the_json_round_trip():
    """kind and adopt travel through the worker's env config.

    ProcessJobConfig.to_json has silently dropped a field before (kind itself,
    which routed every split-stereo job into the segment path); this pins the
    episode-masks fields both ways so the worker cannot fall back to a pixel
    bake with the adoption consent lost.
    """
    from lerobot.gui.process_jobs import ProcessJobConfig

    cfg = ProcessJobConfig(
        job_id="j1",
        source_id="s",
        source_repo_id="r",
        source_root="/x",
        out_repo_id="r",
        out_root="/x",
        model="sam3_track",
        resolution=672,
        objects=[{"name": "tray", "sign": "+", "treatment": {"key": "none"}}],
        background_treatment={"key": "blur", "params": {}},
        apply_mode="per_episode",
        variants=1,
        multi_instance=True,
        cameras=["observation.images.top_l"],
        episodes=[3],
        preview=False,
        kind="episode_masks",
        adopt=True,
        jobs_dir="/tmp/j",
    )
    back = ProcessJobConfig.from_json(cfg.to_json())
    assert back.kind == "episode_masks"
    assert back.adopt is True
    assert back.episodes == [3]


def test_a_masks_job_reports_per_camera_coverage():
    """Two episodes of a 274-episode dataset came back with zero masks on every
    camera while the job said "complete"; a re-run filled them, so it was a
    transient seed failure. Empty rows read as "segmented, found nothing",
    which at training time turns the whole frame into background — so the
    count has to reach the client, and a terminal snapshot must not drop it.
    """
    from lerobot.gui.process_jobs import ProcessJobState

    job = ProcessJobState(
        job_id="j",
        source_id="s",
        out_repo_id="r",
        out_root="/x",
        effect="masks",
        status="running",
        started_at=0.0,
    )
    job.merge_progress(
        {
            "status": "complete",
            "stage": "done",
            "frames_done": 192,
            "frames_total": 192,
            "coverage": {"masks.top_l": 0, "masks.right_wrist": 192},
        }
    )
    assert job.coverage == {"masks.top_l": 0, "masks.right_wrist": 192}
    assert job.to_dict()["coverage"]["masks.top_l"] == 0, (
        "the editor cannot warn about an empty camera it never receives"
    )


def test_the_worker_state_carries_coverage_to_the_snapshot():
    """The worker computes coverage and used to drop it on the floor."""
    from lerobot.gui.process_worker import _WorkerState

    state = _WorkerState()
    state.coverage = {"masks.top_l": 0}
    assert state.snapshot()["coverage"] == {"masks.top_l": 0}
