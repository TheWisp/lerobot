"""FLASH recordings remain readable without changing the shared dataset writer."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash import dataset_export
from lerobot.policies.flash.execution import ActionReceipt, TimedAction
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.recording import FlashEpisodeWriter, RecordedTick
from lerobot.policies.flash.tracking_history import RollingHistory, export_episode


@pytest.mark.parametrize("producer", ["history", "recording"])
def test_flash_finalization_splits_data_and_reopens_exact_depth(tmp_path, monkeypatch, producer):
    # Scale down the byte budget, not the production writer or reader. Each
    # frame fits; the complete episode does not, even with two depth columns.
    monkeypatch.setattr(dataset_export, "_ROW_GROUP_BYTES", 5000)
    rng = np.random.default_rng(81)
    depths = rng.integers(0, 65536, (6, 2, 32, 32), dtype=np.uint16)
    original = {}
    finalize = LeRobotDataset.finalize

    def capture_finalized(dataset):
        finalize(dataset)
        path = next((dataset.root / "data").rglob("*.parquet"))
        original["table"] = pq.read_table(path)
        original["groups"] = pq.ParquetFile(path).num_row_groups
        original["metadata"] = {
            p.relative_to(dataset.root): p.read_bytes()
            for p in (dataset.root / "meta").rglob("*")
            if p.is_file()
        }

    monkeypatch.setattr(LeRobotDataset, "finalize", capture_finalized)
    if producer == "history":
        recent = tmp_path / "recent"
        recent.mkdir()
        (recent / "config.json").write_text("{}")
        history = RollingHistory(recent, reserve_bytes=0)
        try:
            for i, (depth, before) in enumerate(depths):
                history.submit_frame(
                    RGBDFrame(
                        i,
                        rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
                        depth,
                        np.eye(3),
                        0.001,
                        np.eye(4),
                        i / 30,
                        i / 30,
                        "fixture",
                        depth_before_filters=before,
                    )
                )
                history.queue.join()
        finally:
            history.close()
        root = export_episode(recent)
        depth_keys = [
            "observation.images.camera_depth",
            "observation.images.camera_depth_before_filters",
        ]
    else:
        root = tmp_path / "recorded"
        depth_keys = ["observation.images.depth", "observation.images.before"]
        features = {
            key: {
                "dtype": "image",
                "shape": (32, 32, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            }
            for key in depth_keys
        }
        features.update(
            {
                "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
                "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
            }
        )
        dataset = LeRobotDataset.create("local/flash-repack", 30, features, root=root, use_videos=False)
        writer = FlashEpisodeWriter(dataset, 0, "record depth")
        for i, pair in enumerate(depths):
            receipt = ActionReceipt(
                i,
                i / 30,
                i / 30,
                0.0,
                0.0,
                {"joint.pos": float(i)},
                TimedAction(i, {"joint.pos": float(i)}, "move"),
                {"joint.pos": float(i)},
                0,
            )
            writer.write(
                RecordedTick(
                    receipt,
                    {
                        key.removeprefix("observation.images."): depth[:, :, None]
                        for key, depth in zip(depth_keys, pair, strict=True)
                    },
                    i / 30,
                    None,
                )
            )
        writer.save("unknown", "test recording")
        manifest = json.loads((root / "flash/v1/manifest.json").read_text())
        assert manifest["episodes"]["0"]["frames"] == len(depths)

    path = next((root / "data").rglob("*.parquet"))
    assert original["groups"] == 1
    assert pq.ParquetFile(path).num_row_groups > original["groups"]
    assert pq.read_table(path).equals(original["table"], check_metadata=True)
    assert {p: (root / p).read_bytes() for p in original["metadata"]} == original["metadata"]
    reopened = LeRobotDataset("local/flash-repack", root=root)
    assert reopened.episode_rows(0) == (0, len(depths))
    for i, pair in enumerate(depths):
        row = reopened[i]
        for key, depth in zip(depth_keys, pair, strict=True):
            np.testing.assert_array_equal(np.asarray(row[key]), depth[None])


def test_repack_preserves_episode_boundaries_and_variable_size_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_export, "_ROW_GROUP_BYTES", 256)
    path = tmp_path / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    tables = [
        pa.table(
            {"episode_index": [i] * 4, "image": [{"bytes": b"x" * n} for n in (10, 160, 90, 300)]}
        ).replace_schema_metadata({b"custom": b"preserve me"})
        for i in range(2)
    ]
    with pq.ParquetWriter(path, tables[0].schema) as writer:
        for table in tables:
            writer.write_table(table)
    path.chmod(0o640)

    dataset_export.repack_episode_data(tmp_path)

    assert pq.read_table(path).equals(pa.concat_tables(tables), check_metadata=True)
    assert path.stat().st_mode & 0o777 == 0o640
    with pq.ParquetFile(path) as source:
        assert source.num_row_groups > len(tables)
        for i in range(source.num_row_groups):
            group = source.read_row_group(i)
            assert len(set(group["episode_index"].to_pylist())) == 1
            assert group.nbytes <= 256 or group.num_rows == 1


def test_repack_failure_keeps_original_file_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "data/file.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table({"frame": [1, 2]}), path)
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk failure")

    monkeypatch.setattr(dataset_export.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        dataset_export.repack_episode_data(tmp_path)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
