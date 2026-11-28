"""Temporary local-only reader for FLASH recordings.

Keep the normal LeRobot row decoding and episode accessors, but never enter
the dataset constructor's download/repair path while browsing a recording.
"""

from pathlib import Path

from lerobot.configs import DEFAULT_DEPTH_UNIT
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_safe_default_video_backend


class _LocalMetadata(LeRobotDatasetMetadata):
    def __init__(self, repo_id: str, root: Path):
        # An opaque revision skips Hub version resolution if metadata is absent.
        # The file's own codebase_version is still checked by _load_metadata.
        super().__init__(repo_id, root=root, revision="flash-local")

    def _pull_from_repo(self, *args, **kwargs) -> None:
        raise FileNotFoundError(f"Local FLASH metadata is incomplete: {self.root}")


class LocalDataset(LeRobotDataset):
    """Read an existing FLASH recording; missing local files are an error.

    This deliberately exposes only a repo ID and an explicit local root at
    construction. It does not add an option to the shared LeRobot API.
    """

    def __init__(self, repo_id: str, root: str | Path):
        # TODO: Replace this FLASH-only initializer with the standard dataset
        # constructor once it supports local-only metadata AND data loading.
        # Until then, compose its existing metadata/reader components here;
        # never patch shared classes or change process-wide Hub/offline state.
        self.repo_id = repo_id
        self.root = self._requested_root = Path(root)
        self.meta = _LocalMetadata(repo_id, self.root)
        self.revision = self.meta.revision
        self.episodes = None
        self.delta_timestamps = None
        self.tolerance_s = 1e-4
        self._video_backend = get_safe_default_video_backend()
        self._return_uint8 = False
        self._depth_output_unit = DEFAULT_DEPTH_UNIT
        self._batch_encoding_size = 1
        self._encoder_threads = None
        self._decode_videos = True
        self._record_images = True
        self._exclude_flags = []
        self._frame_compositor = None
        self.image_transforms = None
        self.writer = None
        self._is_finalized = True
        self.meta.rescale_depth_stats(self._depth_output_unit)
        self.reader = DatasetReader(
            meta=self.meta,
            root=self.root,
            episodes=self.episodes,
            tolerance_s=self.tolerance_s,
            video_backend=self._video_backend,
            delta_timestamps=self.delta_timestamps,
            image_transforms=self.image_transforms,
            depth_output_unit=self._depth_output_unit,
        )
        if not self.reader.try_load():
            raise FileNotFoundError(f"Local FLASH dataset is incomplete: {self.root}")
