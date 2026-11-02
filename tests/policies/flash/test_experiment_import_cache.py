"""Check source units/alignment and stale-result rejection through real readers."""

import json

import cv2
import numpy as np
import pytest

from benchmarks.flash_experiment_cache import case_identity, reusable, seal_result
from benchmarks.flash_experiment_inputs import sample_at, samples
from benchmarks.flash_experiment_labels import load_pose_labels
from benchmarks.flash_import_labelled_rgbd import import_clip


def imported_bop(tmp_path):
    raw = tmp_path / "bop"
    for d in ("rgb", "depth", "mask_visib"):
        (raw / d).mkdir(parents=True)
    cameras, poses = {}, {}
    for i in (7, 8):
        cameras[str(i)] = {
            "cam_K": [100, 0, 12, 0, 100, 10, 0, 0, 1],
            "depth_scale": 0.1,
            "cam_R_w2c": np.eye(3).flatten().tolist(),
            "cam_t_w2c": [i, 0, 0],
        }
        poses[str(i)] = [{"obj_id": 5, "cam_R_m2c": np.eye(3).flatten().tolist(), "cam_t_m2c": [i, 0, 1000]}]
        cv2.imwrite(str(raw / "rgb" / f"{i:06d}.png"), np.full((20, 24, 3), i, np.uint8))
        cv2.imwrite(str(raw / "depth" / f"{i:06d}.png"), np.full((20, 24), 10000, np.uint16))
        cv2.imwrite(str(raw / "mask_visib" / f"{i:06d}_000000.png"), np.full((20, 24), 255, np.uint8))
    (raw / "scene_camera.json").write_text(json.dumps(cameras))
    (raw / "scene_gt.json").write_text(json.dumps(poses))
    return import_clip("bop", raw, tmp_path / "imported", object_id=5, start=7, count=2, fps=30)


def test_bop_import_keeps_units_pose_direction_and_consecutive_frame_identity(tmp_path):
    case = imported_bop(tmp_path)
    frames = list(samples(case))
    labels, _ = load_pose_labels(case, [{"frame": i} for i in range(2)])
    for i, (frame, masks) in enumerate(frames):
        assert frame.rgb[0, 0, 0] == 7 + i
        np.testing.assert_allclose(frame.metric_depth(), 1.0)
        assert frame.base_from_camera[0, 3] == pytest.approx(-(7 + i) / 1000)
        assert labels["object"][i, 0, 3] == pytest.approx((7 + i) / 1000)
        np.testing.assert_array_equal(sample_at(case, i)[0].rgb, frame.rgb)
        assert masks["object"].all()
    # An unavailable source frame must fail, never silently become a gap in a video.
    with pytest.raises(KeyError):
        import_clip("bop", tmp_path / "bop", tmp_path / "bad", object_id=5, start=8, count=2, fps=30)


@pytest.mark.parametrize(
    "changed", ["rgb", "depth", "mask", "calibration", "labels", "config", "code", "weights"]
)
def test_cache_rejects_changes_even_with_unchanged_paths_and_frame_counts(tmp_path, changed):
    case = imported_bop(tmp_path)
    settings, execution = {"setting": 1}, {"code": "a", "weights": "a"}
    identity = case_identity(case, settings, execution)
    assert case_identity(case, settings, execution) == identity
    root = tmp_path / "imported"
    if changed in ("rgb", "depth", "mask"):
        folder = {"rgb": "rgb", "depth": "depth", "mask": "masks/object"}[changed]
        file = root / folder / "000000.png"
        image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        image[0, 0] = 0
        cv2.imwrite(str(file), image)
    elif changed == "calibration":
        with np.load(root / "calibration.npz") as data:
            fields = {key: data[key].copy() for key in data.files}
        fields["intrinsics"][0, 0, 0] += 1
        np.savez(root / "calibration.npz", **fields)
    elif changed == "labels":
        file = root / "evaluation/poses.npy"
        poses = np.load(file)
        poses[1, 0, 3] += 0.01
        np.save(file, poses)
    elif changed == "config":
        settings["setting"] = 2
    else:
        execution[changed] = "b"
    assert case_identity(case, settings, execution) != identity


def test_incomplete_or_modified_results_are_not_reused(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    seal_result(tmp_path, "key")
    assert not reusable(tmp_path, "key")
    for name in ("frames.jsonl", "reference_points.npz", "case.json", "correspondences.jsonl.gz"):
        (tmp_path / name).write_bytes(b"test fixture")
    seal_result(tmp_path, "key")
    assert reusable(tmp_path, "key")
    assert not reusable(tmp_path, "different")
    (tmp_path / "frames.jsonl").write_bytes(b"modified")
    assert not reusable(tmp_path, "key")


def test_dexycb_uses_selected_object_and_colour_camera_calibration(tmp_path):
    raw = tmp_path / "dex/camera"
    raw.mkdir(parents=True)
    (raw.parent / "meta.yml").write_text("ycb_ids: [4, 6]\n")
    calibration = tmp_path / "camera.yml"
    calibration.write_text(
        "color: {fx: 100, fy: 101, ppx: 12, ppy: 10}\ndepth: {fx: 90}\nextrinsics: !!python/tuple [1, 2, 3]\n"
    )
    for i in range(2):
        cv2.imwrite(str(raw / f"color_{i:06d}.jpg"), np.full((20, 24, 3), 80, np.uint8))
        cv2.imwrite(str(raw / f"aligned_depth_to_color_{i:06d}.png"), np.full((20, 24), 1000, np.uint16))
        poses = np.tile(np.eye(4)[:3], (2, 1, 1))
        poses[1, :, 3] = [i * 0.01, 0, 1]
        mask = np.zeros((20, 24), np.uint8)
        mask[:10] = 4
        mask[10:] = 6
        np.savez(raw / f"labels_{i:06d}.npz", pose_y=poses, seg=mask)
    case = import_clip(
        "dexycb", raw, tmp_path / "imported", object_id=6, start=0, count=2, fps=30, intrinsics=calibration
    )
    frame, masks = sample_at(case, 1)
    assert frame.intrinsics[0, 0] == 100
    assert not masks["object"][:10].any() and masks["object"][10:].all()
    assert frame.metric_depth().mean() == pytest.approx(1)
    labels, _ = load_pose_labels(case, [{"frame": 0}, {"frame": 1}])
    np.testing.assert_allclose(labels["object"][1, :3, 3], [0.01, 0, 1])
