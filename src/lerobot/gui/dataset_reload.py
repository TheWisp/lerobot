"""Shared helpers for reloading LeRobotDataset metadata/data in-place.

Used by edits.py, datasets.py, and hub sync to keep reload logic DRY.
Reloads in-place (mutating the existing dataset object) so that consumers
holding references see updated state. This is important because creating
a new ``LeRobotDataset`` would re-trigger the constructor's Hub round-trip
for local-only datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def reload_metadata(dataset: "LeRobotDataset", root: Path | None = None) -> None:
    """Reload all metadata files (info, episodes, stats, tasks) in-place.

    Args:
        dataset: the LeRobotDataset to mutate.
        root: root path override. Defaults to ``dataset.root``.
    """
    from lerobot.datasets.io_utils import load_episodes, load_info, load_stats, load_tasks

    r = Path(root) if root is not None else dataset.root
    dataset.meta.info = load_info(r)
    dataset.meta.episodes = load_episodes(r)
    dataset.meta.stats = load_stats(r)
    dataset.meta.tasks = load_tasks(r)


def ensure_episodes_loaded(dataset: "LeRobotDataset", root: Path | None = None) -> None:
    """Ensure ``dataset.meta.episodes`` is populated (lazy datasets skip it).

    Cheap no-op if already loaded. Used by endpoints that need episode info
    immediately (e.g., counting, indexing) without paying for the full reload.
    """
    from lerobot.datasets.io_utils import load_episodes

    if dataset.meta.episodes is None:
        r = Path(root) if root is not None else dataset.root
        dataset.meta.episodes = load_episodes(r)


def reload_hf_dataset(dataset: "LeRobotDataset", root: Path | None = None) -> None:
    """Reload the HuggingFace Arrow dataset from parquet files in-place.

    Clears any cached Arrow files first, disables dataset caching during
    load to force fresh data, then re-enables caching. Sets
    ``_lazy_loading=False`` to prevent the lazy-load path from overwriting
    the freshly-loaded data with stale cache.

    Safe no-op if ``dataset.hf_dataset is None`` (lazy datasets).
    """
    if dataset.hf_dataset is None:
        return

    import datasets as hf_datasets

    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import hf_transform_to_torch, load_nested_dataset

    r = Path(root) if root is not None else dataset.root

    try:
        num_cleaned = dataset.hf_dataset.cleanup_cache_files()
        if num_cleaned > 0:
            logger.info("Cleaned up %d HF cache files", num_cleaned)
    except Exception as e:
        logger.warning("Could not cleanup HF cache files: %s", e)

    hf_datasets.disable_caching()
    try:
        features = get_hf_features_from_features(dataset.meta.features)
        dataset.hf_dataset = load_nested_dataset(r / "data", features=features)
        dataset.hf_dataset.set_transform(hf_transform_to_torch)
        dataset._lazy_loading = False
    finally:
        hf_datasets.enable_caching()


def reload_dataset_from_disk(dataset: "LeRobotDataset", root: Path | None = None) -> None:
    """Full in-place reload: metadata + HuggingFace dataset.

    This is what you want after edits or hub downloads — everything that
    was on disk has potentially changed, reload all of it.
    """
    reload_metadata(dataset, root=root)
    reload_hf_dataset(dataset, root=root)
