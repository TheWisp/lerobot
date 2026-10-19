"""A demo folder and an act folder, each in the layout it is stored in today, read as one kind of recording
(lerobot.showservo.recordings): the same frames, times on the wall clock, arm, intrinsics; a demo's annotations."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from lerobot.showservo.recordings import open_recording

H, W, FPS = 480, 848, 30.0  # the rig camera's frame and rate
FX, FY, CX, CY = 604.2, 604.2, 419.2, 250.6
T0 = 1_792_000_000.0  # a wall-clock start


def _frame(k):
    """Frame ``k``: an image whose colour says its number (red 40 k, blue 200), and a depth of (400 + k) mm."""
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = (40 * k, 0, 200)
    return rgb, np.full((H, W), 400 + k, np.uint16)


def _write_frame(folder, name, k):
    rgb, depth_mm = _frame(k)
    (folder / "rgb").mkdir(parents=True, exist_ok=True)
    (folder / "depth").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(folder / "rgb" / f"{name:06d}.jpg"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    cv2.imwrite(str(folder / "depth" / f"{name:06d}.png"), depth_mm)


def _arm(n):
    t = np.arange(n) / FPS
    q = np.tile(np.arange(7, dtype=float), (n, 1)) + t[:, None]
    tips = np.tile(np.eye(4), (n, 1, 1))
    tips[:, 0, 3] = t * 0.01
    return t, q, tips


def _demo(root, n=5):
    cam = root / "recording"
    for k in range(n):
        _write_frame(cam, k, k)
    np.savetxt(cam / "times.txt", T0 + np.arange(n) / FPS, fmt="%.6f")
    np.savetxt(cam / "cam_K.txt", [[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    t, q, tips = _arm(n)
    np.savez_compressed(root / "showservo_demo.npz", t=t, t0=T0, q_obs=q, q_cmd=q + 1.0, tips=tips)
    (root / "keypoints.json").write_text(
        json.dumps({"keypoints": [{"t": 0.1, "kind": "key frame"}], "landing": "turns"})
    )
    mask = np.zeros((H, W), bool)
    mask[200:240, 100:180] = True
    symmetry = {"kind": "continuous", "axis": "long_axis", "order": 1, "end_swap": True}
    np.savez_compressed(
        root / "objects.npz",
        names=json.dumps([["dowel", 1, [140, 220], symmetry]]),
        o0_mask=mask,
        o0_deltas=np.tile(np.eye(4), (n, 1, 1)),
    )
    return mask, symmetry


def _act(root, demo_root):
    """Six recording ticks: four captured frames, one tick holding the frame before, one missed; each frame taken
    20 ms before the tick that sampled it."""
    cam = root / "camera"
    rows, image = [], 0
    for tick, status in enumerate(
        ["captured", "captured", "held_camera_frame", "captured", "missed_tick", "captured"]
    ):
        t_mono = 100.0 + tick / FPS
        row = {"slot": tick, "t_mono": t_mono, "t_wall": T0 + tick / FPS, "status": status}
        if status == "captured":
            _write_frame(cam, image, image)
            row.update(image=image, camera_mono=t_mono - 0.02)
            image += 1
        elif status == "held_camera_frame":
            row.update(image=image - 1, camera_mono=t_mono - 0.02 - 1 / FPS)
        rows.append(row)
    (cam / "timeline.json").write_text(json.dumps({"fps": FPS, "rows": rows}))
    t, q, tips = _arm(6)
    np.savez_compressed(root / "arm.npz", t=T0 + t, t_mono=100.0 + t, q_obs=q, q_cmd=q + 1.0, tip_obs=tips)
    t_bc = np.eye(4)
    t_bc[:3, 3] = (0.1, 0.2, 0.5)
    (root / "act.json").write_text(
        json.dumps(
            {
                "intr": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
                "t_bc": t_bc.tolist(),
                "demo_root": str(demo_root),
            }
        )
    )
    return t_bc


def test_a_demo_and_an_act_read_as_one_kind_of_recording(tmp_path):
    demo_root, act_root = tmp_path / "insertion", tmp_path / "insertion" / "acts" / "20261019_095417"
    mask, symmetry = _demo(demo_root)
    t_bc = _act(act_root, demo_root)
    demo, act = open_recording(demo_root), open_recording(act_root)
    assert (demo.kind, len(demo), act.kind, len(act)) == ("demo", 5, "act", 4)
    for r in (demo, act):
        np.testing.assert_allclose(r.K, [[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        for k in range(len(r)):
            assert r.color(k)[H // 2, W // 2].astype(int).tolist() == pytest.approx(
                [40 * k, 0, 200], abs=3
            ), "frame k, RGB"
            assert r.depth(k)[5, 5] == pytest.approx((400 + k) / 1000.0), "depth in metres"
        np.testing.assert_allclose(r.arm.q_cmd - r.arm.q_obs, 1.0)
    np.testing.assert_allclose(demo.times, T0 + np.arange(5) / FPS, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        demo.arm.t, T0 + np.arange(5) / FPS, rtol=0, atol=1e-6, err_msg="its arm on the wall clock"
    )
    # the act's captured frames only, each at its capture time: the tick's wall time less the 20 ms before it (times
    # near 1.8e9 s: a relative tolerance would allow minutes)
    np.testing.assert_allclose(act.times, T0 + np.array([0, 1, 3, 5]) / FPS - 0.02, rtol=0, atol=1e-6)
    assert demo.frame_at(T0 + 2.4 / FPS) == 2 and act.frame_at(T0 + 3 / FPS) == 2
    assert demo.base_from_camera is None, "a demo does not keep where the camera stood"
    np.testing.assert_allclose(act.base_from_camera, t_bc)
    assert act.annotations is None and act.demo == demo_root
    a = demo.annotations
    assert (
        a.keypoints == [{"t": 0.1, "kind": "key frame"}] and a.landing == "turns" and a.landing_axis is None
    )
    o = a.objects["dowel"]
    assert o["frame"] == 1 and o["click"] == [140, 220] and o["symmetry"] == symmetry
    np.testing.assert_array_equal(o["mask"], mask)


def test_a_folder_that_is_neither_is_said_so(tmp_path):
    with pytest.raises(ValueError, match="neither a demo nor an act"):
        open_recording(tmp_path)
