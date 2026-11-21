"""Small real RGB-D sequences through the live recovery consumer.

Only SAM and learned depth are frozen. Image matching, motion groups, fitting,
filtering, accepted-reference changes and surface growth run sequentially.
Experimental policies are scoped to this runner; production defaults are unchanged.
"""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.image_matching import prepare_image_pair
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.shared_image_tracking import SharedFrameTracker, depth_support_mask, pixels_inside
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_correspondence import measured_pairs
from lerobot.policies.flash.surface_observation import move

FIXTURE = Path(__file__).resolve().parents[1] / "tests/policies/flash/fixtures/motion_recovery_20261117"
CASES = {
    "cube": [2735, 2741, 2747, 2752],
    "pointer": [2992, 3000, 3008],
    "ring": [4740, 4750, 4759, 4765],
}


def read_recorded_frame(recording, number):
    import zstandard

    from lerobot.policies.flash.tracking_history import decode_record, read_packet

    recording = Path(recording)
    frame, masks, _, _ = decode_record(read_packet(recording, number))
    raw = zstandard.ZstdDecompressor().decompress((recording / f"{number:010d}.model_depth.zst").read_bytes())
    depth = np.frombuffer(raw, "<f4").reshape(frame.depth.shape).copy()
    return replace(frame, depth=depth, depth_m_per_unit=1.0, depth_before_filters=None), masks


def full_recording(recording):
    """Original sequential history for a follow-up experiment, not a CI input."""
    recording = Path(recording)
    refs = {2607: "cube", 2911: "pointer", 4173: "ring", 5118: "cube_again"}
    cases = {name: [] for name in refs.values()}
    for p in sorted(recording.glob("*.overlay.json")):
        row = json.loads(p.read_text())
        ref = row["objects"]["object"]["diagnostics"].get("reference_frame")
        if ref in refs:
            cases[refs[ref]].append(row["frame"])

    class Frames:
        def __getitem__(self, n):
            return read_recorded_frame(recording, n)

    return {
        "settings": json.loads((recording / "config.json").read_text())["pipeline_settings"],
        "cases": cases,
    }, Frames()


def export(recording, destination=FIXTURE):
    recording, destination = Path(recording), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    settings = json.loads((recording / "config.json").read_text())["pipeline_settings"]
    arrays, frames = {}, {}
    for number in sorted({f for ids in CASES.values() for f in ids}):
        frame, masks = read_recorded_frame(recording, number)
        depth = frame.depth
        ok, png = cv2.imencode(".png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        assert ok
        arrays.update(
            {
                f"rgb_{number}": png,
                f"depth_{number}": depth,
                f"mask_{number}": masks["object"],
                f"k_{number}": frame.intrinsics,
                f"camera_{number}": frame.base_from_camera,
            }
        )
        frames[str(number)] = {
            "time_s": frame.source_time_s,
            "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        }
    archive = destination / "observations.npz"
    np.savez_compressed(archive, **arrays)
    manifest = {
        "source_recording": recording.name,
        "source_commit": "efdc380a59942fcf72115f698dcabca8210b9222",
        "settings": settings,
        "cases": CASES,
        "frames": frames,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "scope": "Lossless RGB, original float32 model depth, saved SAM masks and calibration. "
        "Short sequences initialize fresh, so they reproduce failure mechanisms, not the "
        "entire original map or group history. No recorded pose is an estimator input. "
        "No independent full 6D pose labels are available.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_fixture(root=FIXTURE):
    root = Path(root)
    meta = json.loads((root / "manifest.json").read_text())
    path = root / "observations.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Recovery fixture checksum mismatch")
    frames = {}
    with np.load(path, allow_pickle=False) as saved:
        for key, info in meta["frames"].items():
            n = int(key)
            rgb = cv2.cvtColor(cv2.imdecode(saved[f"rgb_{n}"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            frames[n] = (
                RGBDFrame(
                    n,
                    rgb,
                    saved[f"depth_{n}"],
                    saved[f"k_{n}"],
                    1.0,
                    saved[f"camera_{n}"],
                    info["time_s"],
                    info["time_s"],
                    "recorded",
                ),
                {"object": saved[f"mask_{n}"]},
            )
    return meta, frames


@contextmanager
def recovery_experiment(policy, min_pairs=6):
    """Test alternate recovery seeds without relaxing any geometry checks."""
    original = ImageConditionedFit.correct_surface_first
    original_pairs = SharedFrameTracker.pairs

    def fit(self, proposal, previous, **kwargs):
        pose, detail = original(self, proposal, previous, **kwargs)
        trials = [{"seed": "original", "accepted": pose is not None}]
        if pose is None and policy.startswith("recover"):
            # A diagnostic proposal, not an identity claim or acceptance rule.
            centered = previous.copy()
            centered[:3, 3] += np.median(self.observation["obs"]["points"], axis=0) - np.median(
                move(self.reference, centered), axis=0
            )
            for name, candidate in [("prior", previous), ("current_surface_center", centered)]:
                pose, alternate = original(self, candidate, previous, **kwargs)
                trials.append({"seed": name, "accepted": pose is not None})
                if pose is not None:
                    detail = alternate
                    break
        detail["recovery_experiment"] = trials
        return pose, detail

    def pairs(self, a, b, intrinsics, *, evidence=None):
        result = original_pairs(self, a, b, intrinsics, evidence=evidence)
        if not policy.startswith("rematch") or len(result[0]) >= min_pairs:
            return result
        # Fresh reference-to-current matching: use the original reference image,
        # regardless of whether any identities survived the intervening images.
        image_a, scale = self._image(a["rgb"])
        image_b, _ = self._image(b["rgb"])
        ys, xs = np.nonzero(
            a["usable"] & depth_support_mask(a["depth"], radius_px=self.config.depth_support_radius_px)
        )
        queries = np.column_stack((xs, ys))
        from lerobot.policies.flash.surface_observation import spatial_sample

        queries = queries[spatial_sample(queries, self.config.object_points)]
        config = self.config.ufm if self.config.backend == "ufm" else self.config.lk
        pair = prepare_image_pair(image_a, image_b, config, model=self.model)
        match = pair.match((queries + 0.5) * scale - 0.5)
        target = (match.target_xy + 0.5) / scale - 0.5
        pb, inside = pixels_inside(target, b["depth"].shape)
        stage = np.zeros(len(queries), np.uint8)
        stage[match.accepted & inside] = 1
        rows = np.flatnonzero(stage == 1)
        rows = rows[b["mask"][pb[rows, 1], pb[rows, 0]]]
        stage[rows] = 2
        support = depth_support_mask(b["depth"], radius_px=self.config.depth_support_radius_px)
        rows = rows[b["usable"][pb[rows, 1], pb[rows, 0]] & support[pb[rows, 1], pb[rows, 0]]]
        stage[rows] = 3
        result = measured_pairs(a, b, queries[rows], target[rows], intrinsics)
        if evidence is not None:
            evidence.clear()
            evidence.update(source_uv=queries, target_uv=target, stage=stage)
        return result

    with (
        patch.object(ImageConditionedFit, "correct_surface_first", fit),
        patch.object(SharedFrameTracker, "pairs", pairs),
    ):
        yield


def replay(case, *, policy="baseline", fixture=None, missing=False, model=None, backend="lk"):
    meta, frames = fixture or load_fixture()
    settings = deepcopy(meta["settings"])
    settings["shared_matching"]["backend"] = backend
    # Preserve the historical no-fallback baseline after the runtime default
    # changes. checked_fit exercises the production fallback without patches.
    settings["surface_experiment"]["allow_fitted_pose_fallback"] = policy == "checked_fit"
    if policy.endswith("_reset"):
        settings["surface_experiment"]["uncertainty"]["reset_on_geometry_conflict"] = True
    numbers = meta["cases"][case]
    source = type("RecordedMasks", (), {"prompts": {"object": "selected object"}})()
    # Pass saved masks through the same checked boundary as live.measure_masks.
    rows = []
    with (
        SearchPool(settings["config"]["shape"]["batch_query_workers"]) as pool,
        recovery_experiment(policy, settings["config"]["motion"]["min_ransac_pairs"]),
    ):
        live = LiveSurfaceSession(settings, source, model, compiled_kernels(), pool)
        try:
            for index, n in enumerate(numbers):
                frame, masks = frames[n]
                if missing and index == 1:
                    # Controlled total observation loss, clearly separated from
                    # the untouched recording: no image identities can survive.
                    frame = replace(frame, rgb=np.zeros_like(frame.rgb), depth=np.zeros_like(frame.depth))
                    masks = {"object": np.zeros_like(masks["object"])}
                begin = time.perf_counter()
                row, matches, _ = live.step(frame, measured_masks=(n, masks, 0.0))
                obj = row["objects"]["object"]
                track = live.engine.candidates["group_growing"]["object"]
                pose = obj["group_growing"]
                rows.append(
                    {
                        "frame": n,
                        "status": obj["group_growing_status"],
                        "pose": pose,
                        "details": obj["group_growing_details"],
                        "elapsed_ms": (time.perf_counter() - begin) * 1000,
                        "accepted_frame": track.accepted_frame,
                        "map_version": track.store.version,
                        "reference": track.reference.tolist(),
                        "surface": track.store.points.tolist(),
                        "matches": matches,
                        "shared": row["shared_matching"],
                    }
                )
        finally:
            live.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--policies", nargs="+", default=["baseline", "recover", "baseline_reset", "recover_reset"]
    )
    parser.add_argument("--backend", choices=["lk", "ufm"], default="lk")
    parser.add_argument("--full-recording", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.export:
        export(args.export)
    if args.output:
        fixture = full_recording(args.full_recording) if args.full_recording else load_fixture()
        model = None
        if args.backend == "ufm":
            from lerobot.policies.flash.surface_tracking import load_ufm

            model = load_ufm(revision="980825b59331413bb51e5773f9272bfce525bce2")
        result = {}
        for missing in [False] if args.full_recording else [False, True]:
            for policy in args.policies:
                for case in fixture[0]["cases"]:
                    name = f"{case}/{policy}/" + ("missing" if missing else "recorded")
                    rows = replay(
                        case,
                        policy=policy,
                        fixture=fixture,
                        missing=missing,
                        model=model,
                        backend=args.backend,
                    )
                    result[name] = rows
                    if args.full_recording:
                        from collections import Counter

                        print(name, dict(Counter(r["status"] for r in rows)), flush=True)
                    else:
                        print(
                            name,
                            [(r["frame"], r["status"], r["details"].get("image_pairs")) for r in rows],
                            flush=True,
                        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
