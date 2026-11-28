"""FLASH recordings remain readable without changing the shared dataset writer."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.policies.flash import dataset_export


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
