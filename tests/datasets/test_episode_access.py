"""Reading a dataset by episode through its own accessors.

The GUI's playback path and the mask store both need three things about
an episode: where its rows are in the loaded table, a slice of one column
for those rows, and where its pictures are in a camera's video file. Each
had been derived by hand in more than one place, with two of the copies
disagreeing about what ``dataset_from_index`` means. These tests pin the
accessors on a dataset that rotates across several files and on one opened
with an episode filter, which is where a hand-rolled copy goes wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset

H, W = 32, 48
FPS = 10
CAM = "observation.images.cam"


@pytest.fixture(scope="module")
def rotated_root(tmp_path_factory):
    """Three episodes of unequal length, written with file-size ceilings small
    enough that the data and the video rotate across files."""
    root = tmp_path_factory.mktemp("rotated") / "ds"
    ds = LeRobotDataset.create(
        repo_id="tests/rotated",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (3,), "names": ["a", "b", "c"]},
            "reward": {"dtype": "float32", "shape": (1,), "names": None},
            CAM: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        },
        use_videos=True,
        data_files_size_in_mb=0.001,
        video_files_size_in_mb=0.001,
    )
    rng = np.random.default_rng(0)
    for ep, length in enumerate((12, 7, 15)):
        for i in range(length):
            ds.add_frame(
                {
                    "observation.state": np.array([ep, i, ep * 100 + i], np.float32),
                    "reward": np.array([i / 10], np.float32),
                    "task": f"task {ep}",
                    CAM: rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


def test_the_fixture_rotates_files(rotated_root):
    """The precondition the other tests rest on: episodes are not all in one file."""
    ds = LeRobotDataset("tests/rotated", root=rotated_root)
    data_files = {ds.meta.get_data_file_path(e) for e in range(3)}
    video_files = {ds.meta.get_video_file_path(e, CAM) for e in range(3)}
    assert len(data_files) > 1 and len(video_files) > 1, (data_files, video_files)


def test_episode_rows_are_the_metadata_range_when_every_episode_is_loaded(rotated_root):
    ds = LeRobotDataset("tests/rotated", root=rotated_root)
    assert ds.episode_rows(0) == (0, 12)
    assert ds.episode_rows(1) == (12, 7)
    assert ds.episode_rows(2) == (19, 15)
    first, count = ds.episode_rows(2)
    # Every row of the range is the episode's, and the absolute index matches.
    rows = ds.hf_dataset.data.slice(first, count)
    assert set(rows.column("episode_index").to_pylist()) == {2}
    assert rows.column("index").to_pylist() == list(range(19, 34))
    with pytest.raises(IndexError):
        ds.episode_rows(3)


def test_episode_rows_follow_the_filtered_table(rotated_root):
    """Opened with ``episodes=[2]``, the table holds only episode 2, at row 0,
    while its metadata still says it starts at absolute frame 19."""
    ds = LeRobotDataset("tests/rotated", root=rotated_root, episodes=[2])
    assert int(ds.meta.episodes[2]["dataset_from_index"]) == 19
    assert ds.episode_rows(2) == (0, 15)
    assert ds.hf_dataset.data.column("index").to_pylist()[:2] == [19, 20]
    with pytest.raises(IndexError):
        ds.episode_rows(0)  # loaded? no: filtered out


def test_episode_column_slices_numeric_and_string_features(rotated_root):
    ds = LeRobotDataset("tests/rotated", root=rotated_root)
    state = ds.episode_column("observation.state", 2)
    assert state.shape == (15, 3) and state.dtype == np.float64
    assert state[4].tolist() == [2.0, 4.0, 204.0]
    part = ds.episode_column("observation.state", 2, start=5, count=3)
    assert part[:, 1].tolist() == [5.0, 6.0, 7.0]
    reward = ds.episode_column("reward", 1)
    assert reward.shape == (7, 1) and abs(reward[3, 0] - 0.3) < 1e-6
    # A scalar int column and its meaning are untouched by the float cast.
    assert ds.episode_column("frame_index", 1)[:, 0].tolist() == list(range(7))
    with pytest.raises(IndexError):
        ds.episode_column("reward", 1, start=5, count=3)
    with pytest.raises(KeyError):
        ds.episode_column(CAM, 1)
    with pytest.raises(KeyError):
        ds.episode_column("no.such.feature", 1)


def test_episode_column_on_a_filtered_dataset_reads_the_right_rows(rotated_root):
    ds = LeRobotDataset("tests/rotated", root=rotated_root, episodes=[1, 2])
    assert ds.episode_column("observation.state", 2)[0].tolist() == [2.0, 0.0, 200.0]
    assert ds.episode_column("observation.state", 1)[-1].tolist() == [1.0, 6.0, 106.0]


def test_episode_column_returns_mask_cells_as_stored(rotated_root, tmp_path):
    from lerobot.datasets.mask_store import adopt, read_frame, write_episode

    ds = LeRobotDataset("tests/rotated", root=rotated_root)
    adopt(ds, [CAM], ["blob"], (H, W))
    blob = np.zeros((H, W), bool)
    blob[8:20, 10:30] = True
    write_episode(ds, 1, CAM, [{"blob": blob} for _ in range(7)])
    cells = ds.episode_column("masks.cam", 1)
    assert isinstance(cells, list) and len(cells) == 7
    assert all(isinstance(c, str) and c for c in cells), cells[:1]
    # The store's own reader, now routed through episode_rows, agrees.
    decoded = read_frame(ds, 1, 3, CAM)
    assert decoded is not None and decoded["blob"].sum() == blob.sum()


def test_episode_video_span_names_the_file_and_the_time_range(rotated_root):
    from lerobot.datasets.video_utils import get_video_duration_in_s

    ds = LeRobotDataset("tests/rotated", root=rotated_root)
    for ep, length in enumerate((12, 7, 15)):
        path, t_from, t_to = ds.meta.get_episode_video_span(ep, CAM)
        assert path == ds.meta.get_video_file_path(ep, CAM)
        assert (ds.root / path).exists()
        assert abs((t_to - t_from) - length / FPS) < 1.5 / FPS, (ep, t_from, t_to)
        assert t_to <= get_video_duration_in_s(ds.root / path) + 1.5 / FPS
    with pytest.raises(KeyError):
        ds.meta.get_episode_video_span(0, "observation.state")
    with pytest.raises(IndexError):
        ds.meta.get_episode_video_span(3, CAM)


def test_video_bitrate_is_size_over_duration(rotated_root):
    from lerobot.datasets.video_utils import get_video_bitrate_kbps, get_video_duration_in_s

    ds = LeRobotDataset("tests/rotated", root=rotated_root)
    path = ds.root / ds.meta.get_video_file_path(0, CAM)
    kbps = get_video_bitrate_kbps(path)
    assert kbps == round(path.stat().st_size * 8 / get_video_duration_in_s(path) / 1000) and kbps > 0
