import json
import pathlib
import shutil
import sys

import cv2
import numpy as np
import pytest

INTR = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0}
H, W = 480, 848  # the rig's camera


@pytest.fixture(scope="module")
def overlay():
    bench = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(bench))
    try:
        import act_overlay

        yield act_overlay
    finally:
        sys.path.remove(str(bench))


def _act(tmp_path: pathlib.Path) -> pathlib.Path:
    """One recorded frame of an act at the rig's size: the picked object's mask and points, the cube's mask and points,
    where the act held the cube (moved 30 mm along x from the demo's view), the fingertip; and the demo the cube was
    designated in, flat at 0.45 m with the camera as the base."""
    demo = tmp_path / "demo"
    (demo / "recording" / "depth").mkdir(parents=True)
    cube = np.zeros((H, W), bool)
    cube[200:260, 300:360] = True
    np.savez(demo / "objects.npz", names=json.dumps([["cube", 0, [330, 230], 4]]), o0_mask=cube)
    cv2.imwrite(str(demo / "recording" / "depth" / "000000.png"), np.full((H, W), 450, np.uint16))
    np.savetxt(demo / "recording" / "cam_K.txt", [[600, 0, 424], [0, 600, 240], [0, 0, 1]])
    act = tmp_path / "act"
    (act / "frames").mkdir(parents=True)
    cv2.imwrite(str(act / "frames" / "000000.jpg"), np.full((H, W, 3), 60, np.uint8))
    cv2.imwrite(str(act / "frames" / "000000_depth.png"), np.full((H, W), 450, np.uint16))
    picked = np.zeros((H, W), np.uint8)
    picked[300:360, 600:700] = 255
    cv2.imwrite(str(act / "frames" / "000000_mask.png"), picked)
    held = np.eye(4)
    held[0, 3] = 0.03  # 40 px at 0.45 m
    np.savez(
        act / "frames" / "000000.npz",
        track_uv=np.array([[620.0, 320.0], [680.0, 340.0]]),
        track_vis=np.array([True, False]),
        other_track_uv_0=np.array([[330.0, 230.0]]),
        other_track_vis_0=np.array([True]),
        other_mask_0=cube,
        onto_held=held,
    )
    tip = np.eye(4)
    tip[:3, 3] = [-0.15, 0.1, 0.45]  # pixel (224, 373)
    np.savez(act / "arm.npz", t=np.array([10.0]), tip_obs=tip[None])
    frame = {
        "i": 0,
        "t_frame": 10.0,
        "step": "place",
        "used": True,
        "state": "tracking",
        "n_matches": 2,
        "n_tracks": 2,
    }
    frame["onto_held_from"] = {"find_id": "f1", "stamp": 9.0, "tracked_at": None}
    meta = {
        "demo_root": str(demo),
        "intr": INTR,
        "t_bc": np.eye(4).tolist(),
        "t_started": 9.0,
        "frames": [frame],
        "target": {"object": "cube", "delta": np.eye(4).tolist(), "at": 9.0},
        "targets": [{"t": 9.8, "step": "pre-place 2", "pose": tip.tolist()}],
    }
    (act / "act.json").write_text(json.dumps(meta))
    return act


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_an_act_replays_with_what_it_knew_drawn_on_each_frame(overlay, tmp_path):
    """To judge an act afterwards from what it saw: each recorded frame drawn with the masks and points it kept, where
    the act held the place object (as recorded), and the fingertip; and a video of them."""
    act = _act(tmp_path)
    video = overlay.render(act, 5.0)
    assert video == act / "overlay.mp4" and video.stat().st_size > 0
    img = cv2.imread(str(act / "overlay" / "000000.jpg")).astype(int)

    def near(px, colour, tol=70):
        return np.abs(img[px[1], px[0]] - np.array(colour)).max() <= tol

    b, g, r = img[255, 305]  # the cube's mask, away from what is drawn on it
    assert b > r + 40 and g > r + 40, "the cube's mask, cyan"
    b, g, r = img[355, 690]
    assert g > b + 40 and r > b + 40, "the picked object's mask, yellow"
    assert near((370, 230), overlay.MAGENTA), "where the act held the cube: its demo-view middle moved 40 px"
    assert not near((330, 230), overlay.MAGENTA), "not where the demo's view had it"
    assert near((224, 373), overlay.RED), "the fingertip"
