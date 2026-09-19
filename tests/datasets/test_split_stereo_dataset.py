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

"""Converting an already-recorded side-by-side stereo dataset into two channels.

Datasets recorded through the plain UVC path store both eyes concatenated, and
cannot be re-recorded. The conversion has to be trustworthy in two directions:
the eyes must actually be the eyes (a swap would be invisible and would poison
every comparison against a live rollout), and everything that is not the stereo
camera must survive untouched — actions, states, the other cameras, and the
per-episode statistics.

Synthesised in tmp_path throughout: no real dataset is read or written.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.datasets.dataset_postprocess import split_stereo_cameras
from lerobot.datasets.lerobot_dataset import LeRobotDataset

H, W_FULL, N_EP, EP_LEN = 64, 128, 2, 6


def _marked_frame(i: int) -> np.ndarray:
    """Halves carry different, frame-dependent content so a swap is detectable."""
    frame = np.zeros((H, W_FULL, 3), dtype=np.uint8)
    # Stepped well apart: at CRF 30 adjacent values collapse to the same mean,
    # which would make the per-frame distinctness check fail on compression
    # rather than on the transform.
    frame[:, : W_FULL // 2, 0] = 20 + 15 * i  # left: red ramp
    frame[:, W_FULL // 2 :, 2] = 240 - 15 * i  # right: blue ramp
    return frame


@pytest.fixture
def src(tmp_path) -> LeRobotDataset:
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (H, W_FULL, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (H, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        # One-element feature: decodes to a 0-d tensor, which validate_frame
        # rejects against its declared (1,) shape. Quality flags are stored
        # exactly this way, so a conversion that skips this cannot run on the
        # labelled dataset at all.
        "quality.flags": {"dtype": "int64", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    ds = LeRobotDataset.create(
        repo_id="test/stereo_src",
        fps=10,
        features=features,
        root=tmp_path / "src",
        use_videos=True,
    )
    for ep in range(N_EP):
        for i in range(EP_LEN):
            ds.add_frame(
                {
                    "observation.images.top": _marked_frame(ep * EP_LEN + i),
                    "observation.images.wrist": np.full((H, 64, 3), 77, dtype=np.uint8),
                    "observation.state": np.array([ep, i], dtype=np.float32),
                    "action": np.array([i, ep], dtype=np.float32),
                    "quality.flags": np.array([ep * 8 + i], dtype=np.int64),
                    "task": "pick",
                }
            )
        ds.save_episode()
    ds.finalize()
    # Reopen: episode metadata is populated on load, and the real caller always
    # passes a dataset it opened rather than one it just wrote.
    return LeRobotDataset("test/stereo_src", root=tmp_path / "src")


def _convert(src, tmp_path, **kw) -> LeRobotDataset:
    res = split_stereo_cameras(
        src,
        out_repo_id="test/stereo_out",
        cameras=["top"],
        out_root=tmp_path / "out",
        **kw,
    )
    assert not res.cancelled
    return LeRobotDataset("test/stereo_out", root=tmp_path / "out")


def test_stereo_key_is_replaced_by_two_at_half_width(src, tmp_path):
    out = _convert(src, tmp_path)
    assert "observation.images.top" not in out.meta.features
    for key in ("observation.images.top_l", "observation.images.top_r"):
        assert tuple(out.meta.features[key]["shape"]) == (H, W_FULL // 2, 3)


def test_eyes_are_not_swapped(src, tmp_path):
    # A swap is invisible in shapes and would silently invalidate every
    # comparison between this data and a live rollout.
    out = _convert(src, tmp_path)
    left = out[0]["observation.images.top_l"]
    right = out[0]["observation.images.top_r"]
    assert left[0].mean() > left[2].mean(), "left eye should be the red half"
    assert right[2].mean() > right[0].mean(), "right eye should be the blue half"


def test_every_frame_is_converted_not_just_the_first(src, tmp_path):
    out = _convert(src, tmp_path)
    assert out.num_frames == N_EP * EP_LEN
    # The halves were given frame-dependent values; distinct means per frame
    # proves the loop is not re-emitting one cached frame.
    means = [float(out[i]["observation.images.top_l"][0].mean()) for i in range(N_EP * EP_LEN)]
    assert len({round(m, 4) for m in means}) == N_EP * EP_LEN


def test_other_camera_and_non_camera_data_survive(src, tmp_path):
    out = _convert(src, tmp_path)
    assert tuple(out.meta.features["observation.images.wrist"]["shape"]) == (H, 64, 3)
    for i in range(N_EP * EP_LEN):
        a, b = src[i], out[i]
        torch.testing.assert_close(a["observation.state"], b["observation.state"])
        torch.testing.assert_close(a["action"], b["action"])
        torch.testing.assert_close(
            a["observation.images.wrist"], b["observation.images.wrist"], atol=2 / 255, rtol=0
        )


def test_episode_structure_is_preserved(src, tmp_path):
    out = _convert(src, tmp_path)
    assert out.meta.total_episodes == N_EP
    assert list(out.meta.episodes["length"]) == [EP_LEN] * N_EP


def test_no_statistic_for_the_removed_key_survives(src, tmp_path):
    # A stale stat under the old key would silently mis-normalize.
    out = _convert(src, tmp_path)
    cols = set(out.meta.episodes.column_names)
    assert not any("observation.images.top/" in c for c in cols), sorted(c for c in cols if "images.top" in c)
    assert any("observation.images.top_l/" in c for c in cols)


def test_unknown_camera_is_rejected(src, tmp_path):
    with pytest.raises(ValueError, match="not cameras of this dataset"):
        split_stereo_cameras(src, out_repo_id="test/x", cameras=["nope"], out_root=tmp_path / "x")


def test_cancellation_finalizes_what_was_written(src, tmp_path):
    seen = {"n": 0}

    def cancel() -> bool:
        seen["n"] += 1
        return seen["n"] > EP_LEN + 2  # partway through the second episode

    res = split_stereo_cameras(
        src,
        out_repo_id="test/stereo_cancel",
        cameras=["top"],
        out_root=tmp_path / "cancel",
        should_cancel=cancel,
    )
    assert res.cancelled
    assert res.episodes_written == 1, "the completed episode should be kept"


def test_scalar_features_survive_the_conversion(src, tmp_path):
    # Regression: a 1-element feature decodes 0-d and was rejected by
    # validate_frame, so the conversion failed outright on the labelled dataset.
    out = _convert(src, tmp_path)
    assert tuple(out.meta.features["quality.flags"]["shape"]) == (1,)
    for i in range(N_EP * EP_LEN):
        assert int(out[i]["quality.flags"]) == int(src[i]["quality.flags"])


def test_untouched_cameras_are_carried_through_bit_identically(src, tmp_path):
    # A camera that is not being split has no reason to be re-encoded. Going
    # through the writer cost the real wrist cameras 39-40 dB against their own
    # source, on data that cannot be re-recorded. Hardlinked means the same
    # bytes, so equality here is exact rather than within a tolerance.
    out = _convert(src, tmp_path, passthrough=True)
    for i in range(N_EP * EP_LEN):
        torch.testing.assert_close(
            src[i]["observation.images.wrist"],
            out[i]["observation.images.wrist"],
            atol=0,
            rtol=0,
        )


def test_carried_video_files_share_inodes_with_the_source(src, tmp_path):
    out = _convert(src, tmp_path, passthrough=True)
    src_dir = src.root / "videos" / "observation.images.wrist"
    out_dir = out.root / "videos" / "observation.images.wrist"
    src_inodes = {p.stat().st_ino for p in src_dir.rglob("*.mp4")}
    out_inodes = {p.stat().st_ino for p in out_dir.rglob("*.mp4")}
    assert out_inodes and out_inodes <= src_inodes, "carried videos were copied, not linked"


def test_split_camera_is_still_re_encoded(src, tmp_path):
    # The passthrough must not accidentally carry the camera being split.
    out = _convert(src, tmp_path, passthrough=True)
    src_inodes = {p.stat().st_ino for p in (src.root / "videos").rglob("*.mp4")}
    for eye in ("top_l", "top_r"):
        d = out.root / "videos" / f"observation.images.{eye}"
        assert d.exists(), f"{eye} has no video directory"
        assert not {p.stat().st_ino for p in d.rglob("*.mp4")} & src_inodes


def test_subset_conversion_falls_back_to_re_encoding(src, tmp_path):
    # Carried metadata indexes the SOURCE episode numbering, so a subset would
    # leave timestamps addressing episodes the output does not contain — and the
    # symptom is silently misaligned video, not an error.
    res = split_stereo_cameras(
        src,
        out_repo_id="test/stereo_subset",
        cameras=["top"],
        out_root=tmp_path / "subset",
        episodes=[0],
        passthrough=True,
    )
    assert not res.cancelled
    out = LeRobotDataset("test/stereo_subset", root=tmp_path / "subset")
    assert "observation.images.wrist" in out.meta.features
    src_inodes = {p.stat().st_ino for p in (src.root / "videos").rglob("*.mp4")}
    out_inodes = {p.stat().st_ino for p in (out.root / "videos").rglob("*.mp4")}
    assert not (out_inodes & src_inodes), "a subset conversion must not hardlink"


def _codecs(root: Path) -> dict[str, str]:
    """Codec of every video file under a dataset, keyed by camera."""
    got: dict[str, str] = {}
    for v in sorted((root / "videos").rglob("*.mp4")):
        cam = next(
            (
                p.name.replace("observation.images.", "")
                for p in v.parents
                if p.name.startswith("observation.images.")
            ),
            "?",
        )
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(v),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        got[cam] = r.stdout.strip()
    return got


@pytest.fixture
def mixed_codec_src(src) -> LeRobotDataset:
    """The source with one camera transcoded, so its codecs genuinely differ.

    Mirrors the real dataset, which carries 52 h264 files and 8 AV1 from two
    recording eras. Asserting "one codec out" against a uniform source would
    pass even for a conversion that copied everything.
    """
    for v in (src.root / "videos" / "observation.images.wrist").rglob("*.mp4"):
        tmp = v.with_suffix(".h264.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(v),
                "-c:v",
                "libx264",
                "-crf",
                "30",
                "-pix_fmt",
                "yuv420p",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(v)
    assert len(set(_codecs(src.root).values())) > 1, "fixture failed to create a mixture"
    return src


def test_conversion_leaves_no_codec_mixture(mixed_codec_src, tmp_path):
    before = _codecs(mixed_codec_src.root)
    out = _convert(mixed_codec_src, tmp_path)
    after = _codecs(out.root)
    assert len(set(before.values())) > 1, f"source was not mixed: {before}"
    assert len(set(after.values())) == 1, f"conversion left a mixture: {after}"


def test_passthrough_preserves_the_mixture_it_carries(mixed_codec_src, tmp_path):
    # The reason passthrough is opt-in: it is bit-identical for carried cameras,
    # which necessarily means keeping whatever codec they already had.
    out = _convert(mixed_codec_src, tmp_path, passthrough=True)
    assert len(set(_codecs(out.root).values())) > 1
