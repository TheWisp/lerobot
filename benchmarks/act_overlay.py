"""Replay an act from its record: every recorded frame with what the act knew drawn on it.

    python benchmarks/act_overlay.py ACT_DIR [--fps 5]

Writes ``ACT_DIR/overlay/NNNNNN.jpg`` (one per recorded frame) and ``ACT_DIR/overlay.mp4``. Drawn from the record as
it was written, nothing re-computed but the place object's pose, which the record keeps in parts:

- the picked object: its tracker's mask (yellow tint) and points (yellow; filled seen, hollow not seen);
- the place object: its tracker's mask (cyan tint, acts recorded since it was kept) and points (cyan, the same way);
- where the act holds the place object (magenta): as the act recorded it with the frame, or, for an act recorded
  before that was kept, rebuilt from the record's parts (its last trusted view, the find before the act or a trusted
  tracked frame, moved on by the point groups since, as act_loop.md's rule says); drawn as the object's surface from
  the demo's view of it, with a cross at its middle;
- the point groups' middle of the place object (green ring);
- the fingertip: where the arm was (red dot) and, while the act walked it, where it was told to go (red ring);
- the step, and whether each object's view was trusted, with the counts it was judged on.
"""

import argparse
import json
import pathlib
import subprocess

import cv2
import numpy as np

YELLOW, CYAN, MAGENTA, GREEN, RED, WHITE = (
    (0, 220, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
)


def project(intr: dict, cam: np.ndarray) -> np.ndarray:
    cam = np.atleast_2d(cam)
    z = np.maximum(cam[:, 2], 1e-6)
    return np.stack([intr["fx"] * cam[:, 0] / z + intr["cx"], intr["fy"] * cam[:, 1] / z + intr["cy"]], 1)


def base_to_cam(t_bc: np.ndarray, p: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(t_bc)
    return inv[:3, :3] @ p + inv[:3, 3]


def text(img: np.ndarray, s: str, org: tuple[int, int], color=WHITE, scale=0.45) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def points(img: np.ndarray, uv, vis, color) -> None:
    if uv is None:
        return
    for (u, v), seen in zip(np.asarray(uv), np.asarray(vis).astype(bool), strict=True):
        cv2.circle(img, (int(u), int(v)), 3, color, -1 if seen else 1, cv2.LINE_AA)


def nearest(rows: list, t: float, key=lambda r: r["t"]):
    return min(rows, key=lambda r: abs(key(r) - t)) if rows else None


def demo_surface(demo_root: pathlib.Path, name: str) -> np.ndarray | None:
    """The place object's surface in the demo's view of it (camera frame, metres), from the frame it was designated on."""
    f = demo_root / "objects.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    rows = json.loads(str(z["names"]))
    n = next((i for i, r in enumerate(rows) if r[0] == name), None)
    if n is None:
        return None
    rec = demo_root / "recording"
    depth = cv2.imread(str(rec / "depth" / f"{int(rows[n][1]):06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    k = np.loadtxt(rec / "cam_K.txt")
    mask = cv2.erode(np.asarray(z[f"o{n}_mask"], np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ys, xs = np.nonzero(mask & (depth > 100))
    zz = depth[ys, xs] / 1000.0
    pts = np.stack([(xs - k[0, 2]) / k[0, 0] * zz, (ys - k[1, 2]) / k[1, 1] * zz, zz], 1)
    return pts[:: max(1, len(pts) // 150)]


def held_pose(meta: dict, t: float) -> tuple[np.ndarray | None, str]:
    """Where the act holds the place object at ``t`` (camera frame motion from the demo's view): its last trusted view
    moved on by the point groups since; and what that view was."""
    target = meta.get("target") or {}
    if target.get("delta") is None:
        return None, "no find of it"
    view, t_view, what = (
        np.asarray(target["delta"], dtype=float),
        float(target.get("stamp") or target["at"]),
        "find",
    )
    for e in meta.get("target_track") or []:
        if e["t"] <= t and e.get("trusted"):
            view, t_view, what = (
                np.asarray(e["delta"], dtype=float),
                e["t"],
                f"tracked view +{e['t'] - meta['t_started']:.1f}s",
            )
    rows = ((meta.get("groups") or {}).get("frames") or {}).get(target.get("object") or "", [])
    if not rows:
        return view, what
    rows = np.asarray(rows, dtype=float)

    def at(s: float) -> np.ndarray:  # the groups' pose at s; before their first frame, their first
        return rows[int(np.argmin(np.abs(rows[:, 0] - max(s, rows[0, 0])))), 1:].reshape(4, 4)

    return at(t) @ np.linalg.inv(at(t_view)) @ view, what + " + point groups"


def render(act: pathlib.Path, fps: float) -> pathlib.Path:
    meta = json.loads((act / "act.json").read_text())
    intr, t_bc, t0 = meta["intr"], np.asarray(meta["t_bc"], dtype=float), meta["t_started"]
    onto = (meta.get("target") or {}).get("object")
    surface = demo_surface(pathlib.Path(meta["demo_root"]), onto) if onto else None
    arm = np.load(act / "arm.npz") if (act / "arm.npz").exists() else None
    groups = ((meta.get("groups") or {}).get("frames") or {}).get(onto or "", [])
    groups = np.asarray(groups, dtype=float) if groups else None
    pose_targets = [x for x in meta.get("targets", []) if "pose" in x]
    out = act / "overlay"
    out.mkdir(exist_ok=True)
    for f in meta["frames"]:
        i, t = f["i"], f["t_frame"]
        img = cv2.imread(str(act / "frames" / f"{i:06d}.jpg"))
        if img is None:
            continue
        m = cv2.imread(str(act / "frames" / f"{i:06d}_mask.png"), cv2.IMREAD_UNCHANGED)
        if m is not None:
            tint = m > 0
            img[tint] = (0.6 * img[tint] + 0.4 * np.array(YELLOW)).astype(np.uint8)
        z = np.load(act / "frames" / f"{i:06d}.npz", allow_pickle=True)
        if "other_mask_0" in z.files:
            tint = np.asarray(z["other_mask_0"]).astype(bool)
            img[tint] = (0.6 * img[tint] + 0.4 * np.array(CYAN)).astype(np.uint8)
        points(img, z.get("track_uv"), z.get("track_vis"), YELLOW)
        points(img, z.get("other_track_uv_0"), z.get("other_track_vis_0"), CYAN)
        if "onto_held" in z.files:
            src = f.get("onto_held_from") or {}
            since = f" (tracker since +{src['tracked_at'] - t0:.1f}s)" if src.get("tracked_at") else ""
            pose, what = (
                np.asarray(z["onto_held"], dtype=float),
                f"recorded: {src.get('find_id') or 'find'}{since}",
            )
        else:
            pose, what = held_pose(meta, t) if onto else (None, "")
            what = f"rebuilt: {what}" if what else what
        if pose is not None and surface is not None:
            cam = surface @ pose[:3, :3].T + pose[:3, 3]
            for u, v in project(intr, cam):
                cv2.circle(img, (int(u), int(v)), 1, MAGENTA, -1)
            u, v = project(intr, cam.mean(0))[0]
            cv2.drawMarker(img, (int(u), int(v)), MAGENTA, cv2.MARKER_CROSS, 16, 2)
        if groups is not None:
            g = groups[int(np.argmin(np.abs(groups[:, 0] - t))), 1:].reshape(4, 4)
            u, v = project(intr, g[:3, 3])[0]
            cv2.circle(img, (int(u), int(v)), 9, GREEN, 2, cv2.LINE_AA)
        if arm is not None:
            tip = arm["tip_obs"][int(np.argmin(np.abs(arm["t"] - t)))][:3, 3]
            u, v = project(intr, base_to_cam(t_bc, tip))[0]
            cv2.circle(img, (int(u), int(v)), 5, RED, -1, cv2.LINE_AA)
        told = [x for x in pose_targets if x["t"] <= t]
        if told and t - told[-1]["t"] < 1.0:
            u, v = project(intr, base_to_cam(t_bc, np.asarray(told[-1]["pose"])[:3, 3]))[0]
            cv2.circle(img, (int(u), int(v)), 9, RED, 2, cv2.LINE_AA)
        tt = nearest(meta.get("target_track") or [], t)
        lines = [
            f"+{t - t0:5.1f}s  frame {i}  {f['step']}",
            f"picked: {f.get('state')}, {'used' if f.get('used') else 'not used'}  "
            f"{f.get('n_matches')}/{f.get('n_tracks')} pts  depth {f.get('depth_seen')}",
        ]
        if tt is not None and abs(tt["t"] - t) < 0.5:
            lines.append(
                f"{onto}: {'trusted' if tt['trusted'] else 'not trusted'}{' (lost)' if tt['lost'] else ''}  "
                f"seen {tt['seen']:.0%}  depth {tt.get('depth_seen')}  {tt['n_visible']}/{tt['n_tracks']} pts"
            )
        if what:
            lines.append(f"{onto} held from: {what}")
        for k, s in enumerate(lines):
            text(img, s, (8, 18 + 17 * k))
        legend = [
            ("picked: mask, points", YELLOW),
            (f"{onto} points", CYAN),
            (f"{onto} where the act holds it", MAGENTA),
            (f"{onto} point groups", GREEN),
            ("fingertip / told", RED),
        ]
        for k, (s, c) in enumerate(legend):
            text(img, s, (8, img.shape[0] - 10 - 15 * (len(legend) - 1 - k)), c, 0.4)
        cv2.imwrite(str(out / f"{i:06d}.jpg"), img)
    video = act / "overlay.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(out / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return video


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("act", type=pathlib.Path)
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()
    print(render(args.act, args.fps))
