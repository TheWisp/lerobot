"""Temporary Parquet compatibility workaround for completed FLASH recordings."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_ROW_GROUP_BYTES = 256 * 1024**2


def repack_episode_data(root: Path) -> None:
    """Split data row groups after the dataset writer has been finalized.

    The caller must exclusively own this dataset until this function returns.
    This adds a streaming read/write pass over its data files. Rows, schema
    metadata and existing row-group boundaries are preserved; only oversized
    groups are subdivided. Videos and dataset metadata files are untouched.
    A single row larger than the target necessarily gets its own group.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # TODO: Remove this workaround and both FLASH finalization calls once the
    # shared LeRobot writer produces bounded row groups for embedded images,
    # verified by reopening multi-GiB RGB-D episodes through LeRobotDataset.
    # One episode-sized group can exceed Arrow's nested binary array limit.
    # Keep the extra I/O here so other LeRobot recording paths are unaffected.
    for path in sorted((Path(root) / "data").rglob("*.parquet")):
        fd, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with (
                os.fdopen(fd, "wb") as stream,
                pq.ParquetFile(path) as source,
                pq.ParquetWriter(
                    stream, source.schema_arrow, compression="snappy", use_dictionary=True
                ) as writer,
            ):
                for group_index in range(source.num_row_groups):
                    batches = []
                    size = 0
                    # Reading the whole old group can itself hit the Arrow
                    # limit. Measure actual decoded bytes, not compressed
                    # file size or a per-episode average of frame sizes.
                    for batch in source.iter_batches(batch_size=1, row_groups=[group_index]):
                        if batches and size + batch.nbytes > _ROW_GROUP_BYTES:
                            writer.write_table(pa.Table.from_batches(batches))
                            batches = []
                            size = 0
                        batches.append(batch)
                        size += batch.nbytes
                    if batches:
                        writer.write_table(pa.Table.from_batches(batches))
            shutil.copymode(path, temporary)
            os.replace(temporary, path)
        finally:
            # safe-destruct: only the unique temporary file owned by this rewrite.
            temporary.unlink(missing_ok=True)
