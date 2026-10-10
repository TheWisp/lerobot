"""Recorded acts replayed offline, to check a change to how the act trusts its views of the object placed onto against
what those acts saw.

Every act records the frames its tracker stepped on, colour and depth (``frames/NNNNNN.jpg`` and ``_depth.png`` beside
``act.json``). ``replay`` runs them through a fresh Point2Pose session following both objects, as the act's own did:
the picked object with the act's recorded mask on its first frame, the object placed onto cut by SAM at its middle
where the act's find put it. It keeps per frame what a record made before 2026-10-10 does not: the placed object's
tracks (number, pixel, seen), mask, pose, key points and fit points, one NPZ per act. ``score`` puts each replayed
frame through the server's own handling of the object placed onto (``pregrasp._apply_others``) with the depth check
off and on, with the act's recorded depth and its point groups' frames, and reports per act the views taken and how far
the pose the act would hold puts the place's last fingertip from where the object lies: the median of the views taken
with the check off, before the place begins (the object lies still until then in these acts).

Usage (from the checkout, the lerobot environment; ``replay`` needs Point2Pose installed for the worker):
  PYTHONPATH=src:benchmarks python benchmarks/act_replay_bench.py replay DEMO_DIR OUT_DIR [--from ACT] [--acts ACT ...]
  PYTHONPATH=src python benchmarks/act_replay_bench.py score DEMO_DIR OUT_DIR [--tol-mm 20]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import cv2
import numpy as np


def _demo(root: pathlib.Path):
    from lerobot.gui.api import pregrasp

    z = np.load(root / pregrasp.DEMO_FILE, allow_pickle=False)
    return pregrasp._Demo(
        name=str(z["name"]),
        concept=str(z["concept"]),
        fps=float(z["fps"]),
        t=np.asarray(z["t"], float),
        tips=np.asarray(z["tips"], float),
        grippers=np.asarray(z["grippers"], float),
        q_obs=np.asarray(z["q_obs"], float),
        q_cmd=np.asarray(z["q_cmd"], float),
        deltas=np.asarray(z["deltas"], float),
        seen=np.asarray(z["seen"]).astype(bool),
        delta0=np.asarray(z["delta0"], float),
        t0=float(z["t0"]),
        root=str(root),
        intr=json.loads(str(z["intr"])),
        keypoints=pregrasp._read_keypoints(root),
        recording=str(root / pregrasp.DEMO_RECORDING),
        objects=pregrasp._read_objects(root),
    )


def _acts(root: pathlib.Path, first: str | None, names: list[str] | None) -> list[pathlib.Path]:
    """The acts to replay: those named, or every act from ``first`` on that recorded its frames and found the object
    placed onto."""
    out = []
    for d in sorted((root / "acts").iterdir()):
        if names and d.name not in names or not names and first and d.name < first:
            continue
        f = d / "act.json"
        if (
            f.exists()
            and (d / "frames" / "000000_depth.png").exists()
            and json.loads(f.read_text()).get("target")
        ):
            out.append(d)
    return out


def _frame(act_dir: pathlib.Path, i: int):
    import pregrasp_worker as pw

    bgr = cv2.imread(str(act_dir / "frames" / f"{i:06d}.jpg"))
    depth = cv2.imread(str(act_dir / "frames" / f"{i:06d}_depth.png"), cv2.IMREAD_UNCHANGED)
    return pw._Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth.astype(np.float32) / 1000.0, f"f{i}")


def replay(args) -> None:
    import pregrasp_worker as pw

    from lerobot.gui.api import pregrasp
    from lerobot.showservo.pose import CameraIntrinsics

    root, out = pathlib.Path(args.demo), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    demo = _demo(root)
    onto = pregrasp._place_object(demo)
    assert onto is not None, "the demo places an object onto another"
    models = pw.Models("cuda", "facebook/dinov3-vits16-pretrain-lvd1689m", 1008)
    sam, _tier = models.ensure("object")
    bridge = pw.P2PBridge()
    try:
        for act_dir in _acts(root, args.first, args.acts):
            act = json.loads((act_dir / "act.json").read_text())
            t_bc, k = np.asarray(act["t_bc"], float), act["intr"]
            ref = (pregrasp._object_points(demo, onto, t_bc) - t_bc[:3, 3]) @ t_bc[:3, :3]
            d = np.asarray(act["target"]["delta"], float)
            m = (
                d[:3, :3] @ ref.mean(axis=0) + d[:3, 3]
            )  # its middle where the act's find put it, camera frame
            click = (int(round(m[0] / m[2] * k["fx"] + k["cx"])), int(round(m[1] / m[2] * k["fy"] + k["cy"])))
            frames = act["frames"]
            first = _frame(act_dir, int(frames[0]["i"]))
            picked = cv2.imread(
                str(act_dir / "frames" / f"{frames[0]['i']:06d}_mask.png"), cv2.IMREAD_GRAYSCALE
            )
            masks = {"picked": picked > 0, onto: sam.mask_at(first.rgb, *click)}
            scene = pw.Scene(lambda: bridge)
            if not scene.start(
                first, CameraIntrinsics(fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"]), masks
            ):
                print(f"{act_dir.name}: the session did not start", flush=True)
                continue
            keep = {"i": [], "t": [], "delta": [], "n_tracks": [], "n_visible": [], "lost": []}
            per_frame = {}
            for f in frames:
                fr = _frame(act_dir, int(f["i"]))
                reply = scene.step(fr.rgb, fr.depth)
                if reply is None or not reply.get("ok"):
                    continue
                s, j = scene.share(onto), scene.order.index(onto)
                keep["i"].append(int(f["i"]))
                keep["t"].append(float(f["t_frame"]) - float(act["t_started"]))
                keep["delta"].append(s["delta"])
                keep["n_tracks"].append(int(s.get("n_tracks") or 0))
                keep["n_visible"].append(int(s.get("n_visible") or 0))
                keep["lost"].append(bool(s["lost"]))
                for key in ("track_idx", "track_uv", "track_vis", "mask", "model", "fit_uv", "fit_inlier"):
                    if f"{key}_{j}" in reply:
                        per_frame[f"{key}_{f['i']}"] = np.asarray(reply[f"{key}_{j}"])
            np.savez_compressed(
                out / f"{act_dir.name}.npz", **{n: np.asarray(v) for n, v in keep.items()}, **per_frame
            )
            print(f"{act_dir.name}: {len(keep['i'])} frames, {onto} clicked at {click}", flush=True)
    finally:
        bridge.close()


def score(args) -> None:
    from lerobot.gui.api import pregrasp

    root, out = pathlib.Path(args.demo), pathlib.Path(args.out)
    demo = _demo(root)
    onto = pregrasp._place_object(demo)
    assert onto is not None, "the demo places an object onto another"
    d_f = np.asarray(demo.objects[onto]["deltas"][pregrasp._pose_frame(demo, onto, "preplace")], float)
    end = next(float(k["t"]) for k in demo.keypoints if k["kind"] == "place_end")
    tip = demo.tips[int(np.argmin(np.abs(demo.t - end)))][:, 3]
    worst = {False: [], True: []}
    for npz in sorted(out.glob("*.npz")):
        act_dir = root / "acts" / npz.stem
        act = json.loads((act_dir / "act.json").read_text())
        rep = np.load(npz)
        t_bc, find = np.asarray(act["t_bc"], float), np.asarray(act["target"]["delta"], float)
        x = np.linalg.inv(d_f) @ np.linalg.inv(t_bc) @ tip
        steps = {int(f["i"]): f["step"] for f in act["frames"]}
        frames_g = [(float(g[0]), np.asarray(g[1:], float).reshape(4, 4)) for g in (act.get("groups") or {}).get("frames", {}).get(onto, [])]  # fmt: skip
        still = {
            j for j, i in enumerate(rep["i"]) if not steps.get(int(i), "").startswith("place") and steps.get(int(i)) != "settling"
        }  # fmt: skip
        truth, line = None, []
        for check in (False, True):
            found = {"object": onto, "ok": True, "delta": find.copy(), "view": ["d", 0]}
            feed = pregrasp._GroupsFeed()
            feed.frames[onto] = collections.deque(frames_g)
            with pregrasp._state.lock:
                pregrasp._state.located = {onto: found}
                pregrasp._state.target = pregrasp._TargetTrack(
                    obj=onto, anchor=find, n_points=int(rep["n_tracks"][0])
                )
                pregrasp._state.trust_share = pregrasp.TRUST_SHARE_DEFAULT
                pregrasp._state.depth_check, pregrasp._state.depth_tol_m = check, args.tol_mm / 1000.0
                pregrasp._state.groups_with_acts = bool(frames_g)
                pregrasp._state.groups_feed = feed
                pregrasp._state.run = None
            taken, held, took = 0, [], []
            for j, i in enumerate(rep["i"]):
                i = int(i)
                if j == 0:  # the session's first frame: the find itself
                    continue
                share = {"name": onto, "ok": not bool(rep["lost"][j]), "lost": bool(rep["lost"][j]),
                         "n_visible": int(rep["n_visible"][j]), "n_tracks": int(rep["n_tracks"][j]), "session": 1}  # fmt: skip
                r = {"others": [share], "other_delta_0": rep["delta"][j]}
                for key in ("fit_uv", "fit_inlier", "model", "track_idx", "track_uv", "track_vis"):
                    if f"{key}_{i}" in rep:
                        r[f"other_{key}_0"] = rep[f"{key}_{i}"]
                depth = cv2.imread(str(act_dir / "frames" / f"{i:06d}_depth.png"), cv2.IMREAD_UNCHANGED)
                before = found["delta"].copy()
                pregrasp._apply_others(
                    r, depth.shape, depth.astype(np.float32) / 1000.0, act["intr"], stamp=act["t_started"] + float(rep["t"][j])
                )  # fmt: skip
                aim = (t_bc @ found["delta"] @ x)[:3]
                if not np.allclose(before, found["delta"]):
                    taken += 1
                    if j in still:
                        took.append(aim)
                if j in still:
                    held.append(aim)
            if truth is None:
                truth = np.median(np.array(took), axis=0) if took else held[0]
            e = np.linalg.norm(np.array(held) - truth, axis=1) * 1000 if held else np.zeros(1)
            worst[check].append(float(e.max()))
            line.append(f"check {'on' if check else 'off'}: {taken} views taken, held off median {np.median(e):.1f} mm, worst {e.max():.1f}")  # fmt: skip
        print(f"{npz.stem}: " + " | ".join(line), flush=True)
    for check in (False, True):
        w = np.array(worst[check])
        print(f"check {'on' if check else 'off'}, over {len(w)} acts: the worst held pose median {np.median(w):.1f} mm, at most {w.max():.1f}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("replay", help="replay recorded acts through Point2Pose")
    a.add_argument("demo", help="the demo's directory; its acts are under acts/")
    a.add_argument("out", help="where each act's replay is written")
    a.add_argument("--from", dest="first", default=None, help="the first act to replay, by name")
    a.add_argument("--acts", nargs="*", default=None, help="these acts only")
    a.set_defaults(fn=replay)
    b = sub.add_parser("score", help="the server's handling of the replayed views, depth check off and on")
    b.add_argument("demo", help="the demo's directory")
    b.add_argument("out", help="the replays")
    b.add_argument("--tol-mm", type=float, default=20.0, help="the depth check's tolerance")
    b.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
