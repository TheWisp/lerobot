"""Cross the saved result → video/export → HTTP → browser playback boundary."""

import gzip
import hashlib
import json

import av
import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from benchmarks.flash_experiment_browser import export
from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_flow_paths import FlowPaths, RecordedFlow
from lerobot.gui.api import flash_experiments


def completed_run(tmp_path):
    source = tmp_path / "source"
    for name in ("rgb", "depth", "gt_mask"):
        (source / name).mkdir(parents=True)
    np.savetxt(source / "cam_K.txt", [[100, 0, 48], [0, 100, 32], [0, 0, 1]])
    for i, stamp in enumerate((1_000_000_000, 1_040_000_000, 1_200_000_000)):
        cv2.imwrite(str(source / "rgb" / f"{stamp}.png"), np.full((64, 96, 3), 40 + 60 * i, np.uint8))
        cv2.imwrite(str(source / "depth" / f"{stamp}.png"), np.full((64, 96), 1000, np.uint16))
        mask = np.zeros((64, 96), np.uint8)
        mask[15:45, 25:65] = 255
        mask[25:35, 35:45] = 0  # Preserve holes in the display mask.
        cv2.imwrite(str(source / "gt_mask" / f"{stamp}.png"), mask)
    case = {
        "id": "example",
        "kind": "ycb_annotated_pose",
        "source_root": str(source),
        "start_frame": 0,
        "expected_frames": 3,
        "roles": ["object"],
    }
    run = tmp_path / "experiment" / "case"
    run.mkdir(parents=True)
    (run / "case.json").write_text(json.dumps(case))
    rows = []
    digest = hashlib.sha256()
    for frame, masks in samples(case):
        pose = np.eye(4)
        pose[0, 3] = 0.01 * frame.frame_id
        other = pose.copy()
        other[1, 3] = 0.02
        rows.append(
            {
                "frame": frame.frame_id,
                "time_s": frame.source_time_s,
                "objects": {
                    "object": {
                        "ufm_chain": pose.tolist() if frame.frame_id != 2 else None,
                        "surface_corrected": other.tolist(),
                    }
                },
            }
        )
        for value in (
            frame.rgb,
            frame.depth,
            frame.intrinsics,
            np.asarray(frame.source_time_s),
            *masks.values(),
        ):
            a = np.asarray(value)
            digest.update(str((a.shape, a.dtype)).encode() + a.tobytes())
    (run / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (run / "summary.json").write_text(json.dumps({"frames": 3, "input_sha256": digest.hexdigest()}))
    np.savez(
        run / "reference_points.npz",
        object=np.array([[-0.1, -0.1, 1], [0.1, -0.1, 1], [-0.1, 0.1, 1], [0.1, 0.1, 1]]),
    )
    return run, source


def recorded_matches(run):
    def pair(source, target):
        return {
            "source_frame": source,
            "target_frame": target,
            "source_uv": [[30, 20], [40, 20]],
            # Deliberately disagree with the projected pose (1 px to the right).
            "target_uv": [[32, 25], [41, 20]],
            "stage": [1, 4],
            "fit_available": True,
        }

    rows = [
        {"frame": 0, "objects": {"object": {"pair": None, "retry": None}}},
        {"frame": 1, "objects": {"object": {"pair": pair(0, 1), "retry": pair(0, 1)}}},
        {"frame": 2, "objects": {"object": {"pair": pair(1, 2), "retry": pair(0, 2)}}},
    ]
    with gzip.open(run / "correspondences.jsonl.gz", "wt") as stream:
        stream.write("\n".join(json.dumps(row) for row in rows) + "\n")
    return rows


def test_export_keeps_irregular_camera_times_and_missing_poses(tmp_path):
    run, _ = completed_run(tmp_path)
    result = export(run, tmp_path / "browser")
    with av.open(str(result / "video.mp4")) as container:
        decoded = list(container.decode(video=0))
    assert [float(f.time) for f in decoded] == pytest.approx([0, 0.04, 0.2], abs=1e-6)
    for i, frame in enumerate(decoded):
        assert frame.to_ndarray(format="rgb24").mean() == pytest.approx(40 + 60 * i, abs=3)
    data = json.loads((result / "overlays.json").read_text())
    assert (data["width"], data["height"]) == (96, 64)
    assert [r["t"] for r in data["frames"]] == pytest.approx([0, 0.04, 0.2])
    first, second, last = (r["objects"]["object"] for r in data["frames"])
    assert len(first["mask"]) == 2
    assert np.array(first["mask"][1]).min(0).tolist() == [25, 15]
    a, b = np.asarray(first["poses"]["ufm_chain"]), np.asarray(second["poses"]["ufm_chain"])
    # The saved surface is at z=1 m; the three display-axis endpoints need not be.
    np.testing.assert_allclose(b[4:] - a[4:], np.tile([1, 0], (len(a) - 4, 1)), atol=0.011)
    assert last["poses"]["ufm_chain"] is None
    assert last["poses"]["surface_corrected"] is not None
    assert not data["has_correspondences"] and first["matches"] is None


def test_export_preserves_raw_2d_predictions_and_earlier_retry_source(tmp_path):
    run, _ = completed_run(tmp_path)
    evidence = recorded_matches(run)
    # Full-field samples can differ from interpolation of the sparse raw grid.
    # Preserve their identities and positions rather than reconstructing them.
    paths = [[123, 80, 30.5, 20, 34.25, 25, 1]]
    evidence[1]["objects"]["object"]["pair"].update(paths=paths, path_sampling="full_ufm_output")
    with gzip.open(run / "correspondences.jsonl.gz", "wt") as stream:
        stream.write("\n".join(json.dumps(row) for row in evidence) + "\n")
    result = export(run, tmp_path / "browser")
    data = json.loads((result / "overlays.json").read_text())
    assert data["has_correspondences"]
    pair = data["frames"][1]["objects"]["object"]["matches"]["pair"]
    assert pair["paths"] == paths
    assert pair["path_sampling"] == "full_ufm_output"
    assert pair["raw"] == [[30, 20, 32, 25], [40, 20, 41, 20]]
    assert pair["used"] == [[40, 20, 41, 20]]
    assert pair["counts"] == [0, 1, 0, 0, 1]
    assert pair["raw_stage"] == [1, 4]
    retry = data["frames"][2]["objects"]["object"]["matches"]["retry"]
    assert (retry["source_frame"], retry["target_frame"]) == (0, 2)
    assert (retry["source_t"], retry["target_t"]) == pytest.approx((0, 0.2))


@pytest.mark.parametrize("wrong", ["row_frame", "target_frame"])
def test_export_rejects_matches_from_another_frame(tmp_path, wrong):
    run, _ = completed_run(tmp_path)
    rows = recorded_matches(run)
    if wrong == "row_frame":
        rows[1]["frame"] = 0
    else:
        rows[1]["objects"]["object"]["pair"]["target_frame"] = 2
    with gzip.open(run / "correspondences.jsonl.gz", "wt") as stream:
        stream.write("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="different frame"):
        export(run, tmp_path / "browser")
    assert list((tmp_path / "browser").iterdir()) == []


def test_changed_source_mask_cannot_be_exported_against_old_poses(tmp_path):
    run, source = completed_run(tmp_path)
    cv2.imwrite(str(source / "gt_mask" / "1040000000.png"), np.zeros((64, 96), np.uint8))
    with pytest.raises(ValueError, match="changed since"):
        export(run, tmp_path / "browser")
    assert list((tmp_path / "browser").iterdir()) == []


def test_http_serves_seekable_video_but_cannot_escape_export_directory(tmp_path, monkeypatch):
    run, _ = completed_run(tmp_path)
    root = tmp_path / "browser"
    result = export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    app = FastAPI()
    app.include_router(flash_experiments.router)
    with TestClient(app) as client:
        base = "/api/run/flash/experiments"
        assert client.get(base).json()[0]["id"] == result.name
        part = client.get(f"{base}/{result.name}/video.mp4", headers={"Range": "bytes=0-63"})
        assert part.status_code == 206 and part.content == (result / "video.mp4").read_bytes()[:64]
        assert client.get(f"{base}/{result.name}/overlays.json").json()["frames"][2]["frame"] == 2
        (result / "overlays.json").unlink()
        (result / "overlays.json").symlink_to(run / "case.json")
        assert client.get(f"{base}/{result.name}/overlays.json").status_code == 404
        assert client.get(f"{base}/{result.name}/manifest.json").status_code == 404
        assert client.get(base).json() == []


@pytest.mark.requires_playwright
def test_browser_seek_method_switch_and_lost_pose_use_same_video_frame(gui_page, tmp_path, monkeypatch):
    run, _ = completed_run(tmp_path)
    recorded_matches(run)
    root = tmp_path / "browser"
    export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    page = gui_page
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.wait_for_selector('#flash-experiment-overlay[data-frame="0"]')
    page.select_option("#flash-experiment-flow-trail", "0")
    assert page.locator(".flash-experiment-item").count() == 1

    assert not page.locator("#run-launch-btn").is_visible()
    # A seek into the long interval must display frame 1, not time * a fixed FPS.
    page.evaluate("document.getElementById('flash-experiment-video').currentTime = .12")
    page.wait_for_selector('#flash-experiment-overlay[data-frame="1"]')
    assert "Next recorded image: +0.16 s" in page.locator("#flash-experiment-status").inner_text()
    current = page.locator("#flash-experiment-video").evaluate("(v) => v.currentTime")
    page.select_option("#flash-experiment-method", "surface_corrected")
    assert page.locator("#flash-experiment-overlay").get_attribute("data-frame") == "1"
    assert page.locator("#flash-experiment-video").evaluate("(v) => v.currentTime") == pytest.approx(current)
    page.uncheck("#flash-experiment-mask")
    page.uncheck("#flash-experiment-pose")
    page.select_option("#flash-experiment-matches", "off")
    assert page.locator("#flash-experiment-overlay").evaluate(
        '(c) => c.getContext("2d").getImageData(0, 0, c.width, c.height).data.every(v => v === 0)'
    )
    page.select_option("#flash-experiment-matches", "raw")
    assert page.locator("#flash-experiment-video").evaluate("(v) => v.currentTime") == pytest.approx(current)
    assert "frame 1 → 2" in page.locator("#flash-experiment-matches").get_attribute("title")
    # The rejected UFM endpoint must be drawn where UFM put it, not where the
    # final pose projects it. The separate fit-agreement view removes it.
    endpoint_alpha = '(c) => { const ctx=c.getContext("2d"), s=ctx.getTransform(); return ctx.getImageData(Math.round(32*s.a), Math.round(25*s.d), 1, 1).data[3]; }'
    assert page.locator("#flash-experiment-overlay").evaluate(endpoint_alpha) > 0
    page.select_option("#flash-experiment-matches", "used")
    assert page.locator("#flash-experiment-overlay").evaluate(endpoint_alpha) == 0
    page.select_option("#flash-experiment-matches", "off")
    page.check("#flash-experiment-mask")
    assert page.locator("#flash-experiment-overlay").evaluate(
        '(c) => c.getContext("2d").getImageData(0, 0, c.width, c.height).data.some(v => v !== 0)'
    )
    page.select_option("#flash-experiment-method", "ufm_chain")
    page.evaluate("document.getElementById('flash-experiment-video').currentTime = .2")
    page.wait_for_selector('#flash-experiment-overlay[data-frame="2"]')
    assert "No pose: object" in page.locator("#flash-experiment-status").inner_text()
    page.select_option("#flash-experiment-method", "surface_corrected")
    assert "No pose" not in page.locator("#flash-experiment-status").inner_text()
    # A new experiment can replace an export while this page remains open.
    # Refresh must select a present result instead of retaining a removed URL.
    next(root.iterdir()).rename(tmp_path / "archived-export")
    replacement_run = run.rename(run.with_name("replacement"))
    replacement = export(replacement_run, root)
    page.evaluate("loadFlashExperiments()")
    page.wait_for_function(
        "id => document.getElementById('flash-experiment-video').src.includes(id)",
        arg=replacement.name,
    )
    page.wait_for_selector('#flash-experiment-overlay[data-frame="0"]')
    page.select_option("#flash-experiment-flow-trail", "0")
    assert page.locator('.flash-experiment-item[aria-pressed="true"]').count() == 1


@pytest.mark.requires_playwright
def test_inline_flow_uses_recorded_retry_pair_without_seeking_main_player(gui_page, tmp_path, monkeypatch):
    run, _ = completed_run(tmp_path)
    matches = recorded_matches(run)
    matches[2]["objects"]["object"]["pair"]["target_uv"][0] = [60, 25]
    with gzip.open(run / "correspondences.jsonl.gz", "wt") as stream:
        stream.write("\n".join(json.dumps(row) for row in matches) + "\n")
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    for row in rows:
        obj = row["objects"]["object"]
        obj["ufm_retained_reference"] = obj["surface_corrected"]
    (run / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    root = tmp_path / "browser"
    export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    page = gui_page
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.wait_for_selector('#flash-experiment-overlay[data-frame="0"]')
    page.select_option("#flash-experiment-flow-trail", "0")
    page.evaluate("document.getElementById('flash-experiment-video').currentTime = .2")
    page.wait_for_selector('#flash-experiment-overlay[data-frame="2"]')
    page.select_option("#flash-experiment-method", "ufm_retained_reference")
    assert "frame 1 → 3" in page.locator("#flash-experiment-matches").get_attribute("title")
    assert page.locator("#flash-experiment-video").evaluate("v => v.currentTime") == pytest.approx(0.2)
    endpoint_alpha = '(c) => { const ctx=c.getContext("2d"), s=ctx.getTransform(); return ctx.getImageData(Math.round(32*s.a), Math.round(25*s.d), 1, 1).data[3]; }'
    assert page.locator("#flash-experiment-overlay").evaluate(endpoint_alpha) > 0
    # The normal path has a different pair; switching must replace the retry arrow.
    page.uncheck("#flash-experiment-mask")
    page.uncheck("#flash-experiment-pose")
    page.select_option("#flash-experiment-method", "ufm_chain")
    assert page.locator("#flash-experiment-overlay").evaluate(endpoint_alpha) == 0
    assert "frame 2 → 3" in page.locator("#flash-experiment-matches").get_attribute("title")
    # Paths that cannot be continued end, rather than leaving unrelated ghosts.
    page.select_option("#flash-experiment-flow-trail", "5")
    faded = page.locator("#flash-experiment-overlay").evaluate(endpoint_alpha)
    assert faded == 0
    page.select_option("#flash-experiment-flow-trail", "0")
    assert page.locator("#flash-experiment-overlay").evaluate(endpoint_alpha) == 0

    assert (
        page.locator("#flash-experiment-overlay").evaluate(
            '(c) => { const ctx=c.getContext("2d"), s=ctx.getTransform(); return ctx.getImageData(Math.round(60*s.a), Math.round(25*s.d), 1, 1).data[3]; }'
        )
        > 0
    )
    assert page.locator("#flash-experiment-video").evaluate("v => v.currentTime") == pytest.approx(0.2)


def grid_pair(source_frame=0, target_frame=1, delta=(0.5, 0.25)):
    y, x = np.mgrid[0:41:4, 0:41:4]
    source = np.column_stack((x.ravel(), y.ravel())).astype(float)
    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "source_uv": source.tolist(),
        "target_uv": (source + delta).tolist(),
        "stage": [4] * len(source),
        "fit_available": False,
    }


def test_flow_paths_compose_subpixel_motion_and_keep_id_when_grid_order_changes():
    paths = FlowPaths(spacing_px=8)
    first = paths.update(grid_pair())
    seed = next(p for p in first if p[2:4] == [8, 8])
    second_pair = grid_pair(1, 2)
    for name in ("source_uv", "target_uv", "stage"):
        second_pair[name].reverse()
    second = next(p for p in paths.update(second_pair) if p[0] == seed[0])
    assert second[1] == seed[1]
    np.testing.assert_allclose(second[2:4], [8.5, 8.25])
    np.testing.assert_allclose(second[4:6], [9, 8.5])
    # An explicit old-image retry follows that image's coordinates, not the last pose.
    retry = next(p for p in paths.update(grid_pair(0, 3, (3, 1))) if p[0] == seed[0])
    np.testing.assert_allclose(retry[2:6], [8, 8, 11, 9])


def test_flow_paths_end_at_missing_cells_instead_of_nearest_neighbour_joining():
    paths = FlowPaths(spacing_px=8)
    first = paths.update(grid_pair())
    seed = next(p for p in first if p[2:4] == [8, 8])
    broken = grid_pair(1, 2)
    missing = broken["source_uv"].index([12, 12])
    for name in ("source_uv", "target_uv", "stage"):
        broken[name].pop(missing)
    assert seed[0] not in {p[0] for p in paths.update(broken)}
    # A whole missing source frame also starts new IDs, without inventing a bridge.
    old_ids = {p[0] for p in first}
    assert old_ids.isdisjoint(p[0] for p in paths.update(grid_pair(8, 9)))


def test_flow_sampling_reproduces_affine_field_without_clipping_bad_endpoints():
    pair = grid_pair()
    source = np.asarray(pair["source_uv"])
    pair["target_uv"] = (source @ np.array([[1.1, 0.2], [0.3, 0.9]]) + [100, -50]).tolist()
    pair["stage"] = [1] * len(source)  # outside mask, still visible as raw predictions
    query = np.array([[7.25, 18.5], [28.75, 4.25]])
    target, stage, valid = RecordedFlow(pair).sample(query)
    np.testing.assert_allclose(target, query @ np.array([[1.1, 0.2], [0.3, 0.9]]) + [100, -50])
    assert valid.all() and (stage == 1).all()


def test_full_flow_preserves_motion_between_recorded_grid_samples():
    from benchmarks.flash_flow_paths import DenseFlow

    pair = grid_pair(delta=(0, 0))
    flow = np.zeros((2, 41, 41), dtype=np.float32)
    flow[0, :, 2::4] = 2  # missed entirely by a grid sampled every four pixels
    query = np.array([[6.25, 10.5]])
    coarse, _, _ = RecordedFlow(pair).sample(query)
    dense, _, valid = DenseFlow(pair, flow, (0, 0), 1).sample(query)
    np.testing.assert_allclose(coarse, query)
    np.testing.assert_allclose(dense, query + [1.5, 0])
    assert valid.all()


def test_rebuild_recomputes_then_publishes_without_erasing_failed_or_old_results(tmp_path, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace

    import torch

    from benchmarks import flash_experiment_cache
    from benchmarks.flash_rebuild_examples import rebuild, write_json
    from lerobot.policies.flash import surface_tracking
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    root = tmp_path / "browser"
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    config = VisionPipelineConfig(compiled_surface_fit=False, reuse_surface_search=False)
    runs = []
    for name in ("good", "bad"):
        run, source = completed_run(tmp_path / name)
        (source / "annotated_poses").mkdir()
        for i in range(3):
            np.savetxt(source / "annotated_poses" / f"{i:03d}.txt", np.eye(4))
        write_json(run.parent / "provenance.json", {"config": asdict(config)})
        runs.append((run, export(run, root)))
    items = flash_experiments._list()
    good = next(item for item in items if item["source_run"] == str(runs[0][0]))
    bad = next(item for item in items if item["source_run"] == str(runs[1][0]))
    # The broken source must fail without hiding its already-playable result.
    (runs[1][0] / "case.json").unlink()
    old = (runs[0][1] / "overlays.json").read_bytes()
    job = root / ".rebuild" / "test"
    job.mkdir(parents=True)
    write_json(job / "request.json", {"browser_root": str(root), "items": [bad, good]})
    write_json(
        job / "status.json",
        {"id": "test", "state": "starting", "total": 2, "completed": 0, "succeeded": 0, "errors": []},
    )
    # Only replace neural inference. Dataset loading, fit, score, journals,
    # exporter, atomic publication and the GUI's catalogue reader are real.
    output = SimpleNamespace(
        flow=SimpleNamespace(flow_output=torch.zeros(1, 2, 320, 320)),
        covisibility=SimpleNamespace(mask=torch.ones(1, 320, 320)),
    )
    model = SimpleNamespace(predict_correspondences_batched=lambda *_: output)
    monkeypatch.setattr(surface_tracking, "load_ufm", lambda **_: model)
    monkeypatch.setattr(
        flash_experiment_cache,
        "execution_identity",
        lambda source: {"model_revision": "fixture", "source": source},
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    rebuild(job)
    state = json.loads((job / "status.json").read_text())
    assert state["state"] == "completed_with_errors"
    assert state["completed"] == 2 and state["succeeded"] == 1 and len(state["errors"]) == 1
    assert state["frame"] == 3
    assert state["report_ready"]
    report = json.loads((job / "report.json").read_text())
    assert len(report["cases"]) == 2
    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][1]["summary"]["accuracy_labels"]
    measured = report["cases"][1]["summary"]["roles"]["object"]["methods"]["ufm_chain"]
    assert measured["available_after_initialization"] == 2
    assert measured["pose_error"]["cloud_mm_median"] < 0.001
    assert "No such file" in (job / "report.html").read_text()
    latest = flash_experiments._list()
    assert len(latest) == 2 and bad in latest
    replacement = next(item for item in latest if item.get("slot_id") == good["id"])
    assert replacement["id"] != good["id"] and replacement["source_run"] != good["source_run"]
    assert (runs[0][1] / "overlays.json").read_bytes() == old
    from pathlib import Path

    new_run = Path(replacement["source_run"])
    assert json.loads((new_run.parent / "provenance.json").read_text())["config"] == asdict(config)
    data = json.loads((root / replacement["id"] / "overlays.json").read_text())
    pair = data["frames"][1]["objects"]["object"]["matches"]["pair"]
    assert pair["path_sampling"] == "full_ufm_output"
    assert pair["paths"]
    assert (
        data["frames"][1]["objects"]["object"]["poses"]["ufm_chain"]
        != json.loads(old)["frames"][1]["objects"]["object"]["poses"]["ufm_chain"]
    )

    # Cross the real worker/cache boundary: an unchanged rerun must not even
    # load the network, while forced reruns and an altered input must do so.
    def no_model(**_):
        raise AssertionError("model loaded")

    monkeypatch.setattr(surface_tracking, "load_ufm", no_model)

    def rerun(name, force=False):
        next_job = root / ".rebuild" / name
        next_job.mkdir()
        write_json(
            next_job / "request.json", {"browser_root": str(root), "items": [replacement], "force": force}
        )
        write_json(
            next_job / "status.json",
            {"id": name, "state": "starting", "total": 1, "completed": 0, "succeeded": 0, "errors": []},
        )
        rebuild(next_job)
        return json.loads((next_job / "status.json").read_text())

    cached = rerun("cached")
    assert cached["state"] == "completed" and cached["reused"] == 1 and cached["computed"] == 0
    assert rerun("forced", force=True)["errors"][0]["error"] == "model loaded"
    depth_path = next((runs[0][0].parents[1] / "source/depth").glob("*.png"))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth[0, 0] += 1
    cv2.imwrite(str(depth_path), depth)
    assert rerun("changed")["errors"][0]["error"] == "model loaded"


def test_rebuild_lock_survives_launcher_and_detects_worker_exit(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    root = tmp_path / "browser"
    run, _ = completed_run(tmp_path)
    export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    launch = subprocess.Popen
    children = []

    def hold_lock(command, **kwargs):
        # Real process and inherited descriptor; only the expensive work is
        # replaced. This catches closing/unlocking the lock in the parent.
        assert command[1:3] == ["-m", "benchmarks.flash_rebuild_examples"]
        child = launch([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(flash_experiments.subprocess, "Popen", hold_lock)
    app = FastAPI()
    app.include_router(flash_experiments.router)
    try:
        with TestClient(app) as client:
            base = "/api/run/flash/experiments/rebuild"
            assert client.get(base).json()["state"] == "idle"
            assert client.post(base + "?force=true").status_code == 202
            job = Path(json.loads((root / ".rebuild/current.json").read_text())["job"])
            assert json.loads((job / "request.json").read_text())["force"] is True
            assert client.get(base).json()["state"] == "starting"
            assert client.post(base).status_code == 409
            assert len(children) == 1
            children[0].terminate()
            children[0].wait(timeout=5)
            assert client.get(base).json()["state"] == "failed"
            assert client.post(base).status_code == 202
            assert len(children) == 2
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_benchmark_report_serves_results_without_claiming_unlabelled_accuracy(tmp_path, monkeypatch):
    from benchmarks.flash_rebuild_examples import write_report

    root = tmp_path / "browser"
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    job_id = "rebuild-20261101-120000-1234abcd"
    job = root / ".rebuild" / job_id
    job.mkdir(parents=True)
    request = {"items": [{"id": "one", "name": "<object>", "source_run": "/example", "mask_source": "SAM"}]}
    outcomes = {
        "one": {
            "status": "completed",
            "summary": {
                "accuracy_labels": False,
                "roles": {
                    "object": {
                        "methods": {
                            "method": {
                                "available_after_initialization": 1,
                                "total_after_initialization": 2,
                            }
                        }
                    }
                },
            },
        }
    }
    write_report(job, request, {"succeeded": 1, "total": 1, "errors": []}, outcomes)
    app = FastAPI()
    app.include_router(flash_experiments.router)
    with TestClient(app) as client:
        base = f"/api/run/flash/experiments/rebuild/{job_id}"
        page = client.get(base + "/report.html")
        assert page.status_code == 200
        assert "Unmeasured" in page.text and "&lt;object&gt;" in page.text
        assert "<object>" not in page.text and "1 / 2" in page.text
        result = client.get(base + "/report.json")
        assert "attachment" in result.headers["content-disposition"]
        assert result.json()["cases"][0]["summary"] == outcomes["one"]["summary"]
        assert client.get(base + "/request.json").status_code == 404
        assert client.get(base.replace(job_id, "invalid") + "/report.html").status_code == 404


@pytest.mark.requires_playwright
def test_browser_rebuild_button_reports_progress_and_failure(gui_page, tmp_path, monkeypatch):
    run, _ = completed_run(tmp_path)
    root = tmp_path / "browser"
    export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    page = gui_page
    state = {"state": "idle"}
    posts = []

    def status(route):
        if route.request.method == "POST":
            posts.append(route.request.method)
            state.update(
                id="job",
                state="running",
                current="Example",
                phase="Tracking",
                frame=2,
                frames=20,
                completed=0,
                succeeded=0,
                total=1,
                errors=[],
            )
        route.fulfill(
            status=202 if route.request.method == "POST" else 200,
            content_type="application/json",
            body=json.dumps(state),
        )

    page.route("**/api/run/flash/experiments/rebuild", status)
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.get_by_role("button", name="Rerun all benchmarks").click()
    page.wait_for_function(
        "document.getElementById('flash-experiment-rebuild-status').textContent.includes('2/20')"
    )
    assert posts == ["POST"]
    assert page.get_by_role("button", name="Rerun all benchmarks").is_disabled()
    # Published results remain accessible even while the next report is pending.
    page.get_by_role("button", name="View results", exact=True).click()
    assert page.locator("#flash-experiment-comparison").is_visible()
    assert not page.get_by_role("link", name="Download JSON").is_visible()
    page.locator(".flash-experiment-item").click()
    assert page.locator("#flash-experiment-playback").is_visible()
    # Reloading the page reconnects to the existing job, without starting another.
    page.reload()
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.wait_for_function(
        "document.getElementById('flash-experiment-rebuild-status').textContent.includes('2/20')"
    )
    assert posts == ["POST"]
    state.update(
        state="completed_with_errors", completed=1, errors=[{"name": "Example", "error": "Missing depth"}]
    )
    page.wait_for_function(
        "document.getElementById('flash-experiment-rebuild-status').textContent.includes('1 failed')"
    )
    assert page.get_by_role("button", name="Rerun all benchmarks").is_enabled()
    page.locator("#flash-experiment-rebuild-errors summary").click()
    assert "Missing depth" in page.locator("#flash-experiment-rebuild-errors").inner_text()
    assert page.locator(".flash-experiment-item").count() == 1

    state.update(report_ready=True, id="rebuild-20261101-120000-1234abcd")
    page.wait_for_function("!document.getElementById('flash-experiment-report-download').hidden")
    assert page.get_by_role("button", name="View results", exact=True).is_visible()
    assert page.get_by_role("link", name="Download JSON").get_attribute("href").endswith("/report.json")

    # Saved Run settings can rebuild the form during a POST. Losing the old
    # controls must not fail the request or leave an unhandled JS exception.
    page.evaluate("""async () => {
        const original = window.fetch;
        let finish;
        window.fetch = (url, options) => options?.method === 'POST'
            ? new Promise(resolve => { finish = resolve; }) : original(url, options);
        const pending = rebuildFlashExamples();
        document.getElementById('flash-experiment-sidebar').replaceChildren();
        finish(new Response('{}', {status: 202}));
        try { await pending; } finally { window.fetch = original; }
    }""")


def test_growing_surface_playback_uses_frame_model_without_recentering_axes(tmp_path):
    run, _ = completed_run(tmp_path)
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    model_dir = run / "surface_models/conditioned_growing/object"
    model_dir.mkdir(parents=True)
    for i, row in enumerate(rows):
        cloud = np.array([[0.01 * i, 0, 1], [0.01 * i, 0.01, 1], [0.02, 0.01, 1]])
        np.savez(model_dir / f"{i + 1:06d}.npz", points=cloud)
        row["objects"]["object"].update(
            conditioned_growing=np.eye(4).tolist(), conditioned_growing_details={"map_version_after": i + 1}
        )
    (run / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = export(run, tmp_path / "browser")
    data = json.loads((result / "overlays.json").read_text())
    geometries = [
        np.asarray(frame["objects"]["object"]["poses"]["conditioned_growing"]) for frame in data["frames"]
    ]
    np.testing.assert_array_equal(geometries[0][:4], geometries[2][:4])
    assert geometries[0][4, 0] == 48
    assert geometries[2][4, 0] == 50
    assert data["methods"]["conditioned_growing"] == "Image + surface: growing model"


@pytest.mark.requires_playwright
def test_growing_method_does_not_show_matches_from_a_different_source_frame(gui_page, tmp_path, monkeypatch):
    run, _ = completed_run(tmp_path)
    recorded_matches(run)
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    for row in rows:
        row["objects"]["object"].update(
            conditioned_growing=np.eye(4).tolist(), conditioned_growing_details={"image_source_frame": 0}
        )
    (run / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    root = tmp_path / "browser"
    export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    page = gui_page
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.wait_for_selector('#flash-experiment-overlay[data-frame="0"]')
    page.select_option("#flash-experiment-method", "conditioned_growing")
    page.check("#flash-experiment-pose")
    assert "accumulated measured surface" in page.locator("#flash-experiment-explanation").inner_text()
    assert "undefined" not in page.locator("#flash-experiment-explanation").inner_text()
    assert page.evaluate(
        "flashExperimentPair(flashExperiments.data.frames[1].objects.object, 'conditioned_growing') !== null"
    )
    assert page.evaluate(
        "flashExperimentPair(flashExperiments.data.frames[2].objects.object, 'conditioned_growing') === null"
    )


def scored_run(tmp_path, *, labelled=True):
    """Real scorer: one exact UFM pose, one missing; corrected poses 15 mm off."""
    from benchmarks.flash_tracker_experiment import score_case

    run, source = completed_run(tmp_path)
    (source / "annotated_poses").mkdir()
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        truth = np.eye(4)
        truth[0, 3] = 0.01 * i
        np.savetxt(source / "annotated_poses" / f"{i:03d}.txt", truth)
        obj = row["objects"]["object"]
        corrected = truth.copy()
        corrected[1, 3] = 0.015
        obj["surface_corrected"] = corrected.tolist()
        obj.update(prepare_ms=5.0, finish_ms=7.0)
    (run / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    case = json.loads((run / "case.json").read_text())
    with np.load(run / "reference_points.npz") as points:
        summary = score_case(case, rows, {"object": points["object"]}, {"object": 0})
    old = json.loads((run / "summary.json").read_text())
    summary.update(input_sha256=old["input_sha256"], accuracy_labels=labelled)
    for role in summary["roles"].values():
        role["accuracy_labels"] = labelled
        if not labelled:
            for method in role["methods"].values():
                method.pop("pose_error", None)
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def test_catalogue_scores_belong_to_export_and_ignore_mismatched_legacy_source(tmp_path, monkeypatch):
    run = scored_run(tmp_path)
    root = tmp_path / "browser"
    exported = export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    saved = flash_experiments._list()[0]["metrics"]
    assert saved["roles"]["object"]["methods"]["ufm_chain"]["pose_error"]["within_10mm_10deg"] == 1
    (run / "summary.json").write_text(json.dumps({"frames": 3, "input_sha256": "different"}))
    assert flash_experiments._list()[0]["metrics"] == saved
    (exported / "summary.json").unlink()  # Older exports may rely on the original result.
    assert "metrics" not in flash_experiments._list()[0]


@pytest.mark.requires_playwright
def test_result_panel_and_scene_charts_use_real_scores_and_preserve_playback(gui_page, tmp_path, monkeypatch):
    root = tmp_path / "browser"
    labelled = export(scored_run(tmp_path / "labelled"), root)
    unlabelled = export(scored_run(tmp_path / "unlabelled", labelled=False), root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    page = gui_page
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.locator(f'.flash-experiment-item[data-id="{labelled.name}"]').click()
    page.wait_for_function("flashExperiments.data !== null")
    page.select_option("#flash-experiment-method", "ufm_chain")
    panel = page.locator("#flash-experiment-results")
    assert "50.0%" in panel.inner_text()  # Missing pose is in the denominator.
    assert "1 / 2 frames within 10 mm / 10°" in panel.inner_text()
    page.locator("#flash-experiment-video").evaluate("v => {v.currentTime=.04;}")
    page.wait_for_function('document.getElementById("flash-experiment-video").currentTime === .04')
    page.select_option("#flash-experiment-method", "surface_corrected")
    assert "0.0%" in panel.inner_text() and "15.0 / 15.0 mm" in panel.inner_text()
    assert page.locator("#flash-experiment-video").evaluate("v=>v.currentTime") == pytest.approx(0.04)
    assert page.get_by_role("button", name="Compare methods", exact=True).count() == 0
    page.locator("#flash-experiment-sidebar").get_by_role("button", name="View results", exact=True).click()
    assert "1 of 2 scenes" in page.locator("#flash-experiment-comparison-note").inner_text()
    card = page.locator(f'.flash-comparison-card[data-id="{labelled.name}"]')
    assert card.locator('[data-method="ufm_chain"] .flash-chart-value').inner_text() == "50.0"
    assert card.locator('[data-method="surface_corrected"] .flash-chart-value').inner_text() == "0.0"
    page.select_option("#flash-experiment-limits", "20")
    assert card.locator('[data-method="surface_corrected"] .flash-chart-value').inner_text() == "100.0"
    unknown = page.locator(f'.flash-comparison-card[data-id="{unlabelled.name}"]')
    assert "no reference poses connected" in unknown.inner_text()
    assert unknown.locator(".flash-chart-row").count() == 0
    unknown.get_by_role("button", name="Show tracking availability").click()
    assert unknown.locator('[data-method="surface_corrected"] .flash-chart-value').inner_text() == "100.0"
    unknown.locator('[data-method="surface_corrected"]').click()
    page.wait_for_selector("#flash-experiment-playback:visible")
    assert page.locator("#flash-experiment-playback").is_visible()
    assert page.locator("#flash-experiment-method").input_value() == "surface_corrected"
    assert "Accuracy unmeasured" in panel.inner_text()
    assert "100.0%" not in panel.locator(".flash-metric-primary").inner_text()
    page.get_by_role("button", name="View results", exact=True).click()
    page.locator(f'.flash-experiment-item[data-id="{labelled.name}"]').click()
    page.wait_for_function("flashExperiments.data !== null")
    assert page.locator("#flash-experiment-playback").is_visible()
    assert page.locator("#flash-experiment-comparison").is_hidden()


def test_cancel_stops_only_verified_worker_and_keeps_completed_exports(tmp_path, monkeypatch):
    import subprocess

    root = tmp_path / "browser"
    run, _ = completed_run(tmp_path)
    export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    module = tmp_path / "worker/benchmarks"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    (module / "flash_rebuild_examples.py").write_text("import time\ntime.sleep(30)\n")
    launch = subprocess.Popen
    children = []

    def slow_worker(command, **kwargs):
        kwargs["cwd"] = module.parent
        child = launch(command, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(flash_experiments.subprocess, "Popen", slow_worker)
    app = FastAPI()
    app.include_router(flash_experiments.router)
    try:
        with TestClient(app) as client:
            base = "/api/run/flash/experiments/rebuild"
            before = client.get("/api/run/flash/experiments").json()
            assert client.post(base).status_code == 202
            # A stale/wrong worker PID must never stop another process.
            import sys
            from pathlib import Path

            job = Path(json.loads((root / ".rebuild/current.json").read_text())["job"])
            unrelated = launch([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                (job / "worker.json").write_text(json.dumps({"pid": unrelated.pid}))
                assert client.post(base + "/cancel").status_code == 409
                assert unrelated.poll() is None and children[0].poll() is None
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)
            (job / "worker.json").write_text(json.dumps({"pid": children[0].pid}))
            result = client.post(base + "/cancel")
            assert result.status_code == 202
            children[0].wait(timeout=5)
            assert client.get(base).json()["state"] == "cancelled"
            assert client.get("/api/run/flash/experiments").json() == before
            assert client.post(base + "/cancel").json()["state"] == "cancelled"
            assert client.post(base).status_code == 202
            assert len(children) == 2
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_worker_cancel_before_model_load_writes_partial_report(tmp_path, monkeypatch):
    from benchmarks.flash_rebuild_examples import rebuild, write_json
    from lerobot.policies.flash import surface_tracking

    job = tmp_path / "job"
    job.mkdir()
    write_json(job / "request.json", {"browser_root": str(tmp_path / "browser"), "items": []})
    write_json(job / "status.json", {"state": "starting", "total": 0, "succeeded": 0, "errors": []})
    (job / "cancel").touch()

    def forbidden():
        raise AssertionError("Cancellation must be checked before loading a model")

    monkeypatch.setattr(surface_tracking, "load_ufm", forbidden)
    rebuild(job)
    state = json.loads((job / "status.json").read_text())
    assert state["state"] == "cancelled" and state["report_ready"]
    assert (job / "report.html").is_file()


def labelled_scene_run(tmp_path):
    run, source = completed_run(tmp_path)
    (source / "annotated_poses").mkdir()
    for i in range(3):
        pose = np.eye(4)
        pose[:3, 3] = [0.3, 0.02 * i, 0.8]  # Nonzero model origin must cancel.
        np.savetxt(source / "annotated_poses" / f"{i:06d}.txt", pose)
    return run, source


def decode_scene_packet(content):
    import struct

    length = struct.unpack("<I", content[:4])[0]
    data = json.loads(content[4 : 4 + length])
    start = 4 + length
    shape = (data["camera"]["height"], data["camera"]["width"])
    depth = np.frombuffer(content[start : start + data["depth_bytes"]], "<f4").reshape(shape)
    rgb = cv2.imdecode(np.frombuffer(content[start + data["depth_bytes"] :], np.uint8), cv2.IMREAD_COLOR)
    return data, depth, rgb


def test_scene_http_uses_same_frame_units_and_relative_label_origin(tmp_path, monkeypatch):
    from benchmarks.flash_experiment_inputs import sample_at

    run, _ = labelled_scene_run(tmp_path)
    case = json.loads((run / "case.json").read_text())
    # Indexed playback and sequential inference read identical source data.
    for index, (expected, _) in enumerate(samples(case)):
        actual, _ = sample_at(case, index)
        np.testing.assert_array_equal(actual.rgb, expected.rgb)
        np.testing.assert_array_equal(actual.depth, expected.depth)
        assert actual.source_time_s == expected.source_time_s
    root = tmp_path / "browser"
    output = export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    app = FastAPI()
    app.include_router(flash_experiments.router)
    with TestClient(app) as client:
        url = f"/api/run/flash/experiments/{output.name}/scene/1"
        response = client.get(url, params={"method": "surface_corrected"})
        assert response.status_code == 200
        data, depth, rgb = decode_scene_packet(response.content)
        assert data["frame"] == 1 and data["time_s"] == pytest.approx(0.04)
        np.testing.assert_array_equal(depth, np.ones((64, 96)))
        np.testing.assert_array_equal(rgb, np.full((64, 96, 3), 100))
        obj = data["objects"]["object"]
        assert np.array(obj["pose"])[:3, 3] == pytest.approx([0.01, 0.02, 0])
        assert np.array(obj["labelled_pose"])[:3, 3] == pytest.approx([0, 0.02, 0])
        assert obj["label_kind"] == "initial_observation_at_labelled_pose"
        last = client.get(url.replace("/scene/1", "/scene/2"), params={"method": "ufm_chain"})
        obj = decode_scene_packet(last.content)[0]["objects"]["object"]
        assert obj["pose"] is None and obj["labelled_pose"] is not None
        assert client.get(url, params={"method": "../unknown"}).status_code == 409
        assert (
            client.get(url.replace("/scene/1", "/scene/9"), params={"method": "ufm_chain"}).status_code == 409
        )


def test_scene_growth_never_changes_labelled_reference(tmp_path):
    from lerobot.gui.flash_experiment_scene import scene_packet

    run, _ = labelled_scene_run(tmp_path)
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        obj = row["objects"]["object"]
        obj["conditioned_growing"] = obj["surface_corrected"]
        obj["conditioned_growing_details"] = {"map_version_after": i + 1}
        file = run / "surface_models/conditioned_growing/object" / f"{i + 1:06d}.npz"
        file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(file, points=np.full((7, 3), i + 0.25))
    (run / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output = export(run, tmp_path / "browser")
    manifest = json.loads((output / "manifest.json").read_text())
    for index in (2, 0, 1):
        data, _, _ = decode_scene_packet(scene_packet(manifest, index, "conditioned_growing"))
        obj = data["objects"]["object"]
        np.testing.assert_array_equal(obj["surface"], np.full((7, 3), index + 0.25))
        assert len(obj["reference"]) == 4
        assert np.array(obj["labelled_pose"])[:3, 3] == pytest.approx([0, index * 0.02, 0])


@pytest.mark.requires_playwright
def test_scene_browser_orbits_resets_and_switches_recorded_method(gui_page, tmp_path, monkeypatch):
    import shutil

    run, _ = labelled_scene_run(tmp_path)
    root = tmp_path / "browser"
    first_export = export(run, root)
    other_run = tmp_path / "other-run"
    shutil.copytree(run, other_run)
    second_export = export(other_run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    page = gui_page
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.locator(f'.flash-experiment-item[data-id="{first_export.name}"]').click()
    page.wait_for_selector('#flash-experiment-overlay[data-frame="0"]')
    page.get_by_role("button", name="3D scene", exact=True).click()
    page.wait_for_selector('#flash-experiment-scene[data-frame="0"]')
    # Validate the actual GPU projection: off-centre intrinsics and portrait
    # source must agree with the pixel positions of object-fit: contain.
    error = page.evaluate("""async () => {
        const {calibratedProjection} = await import('/static/rgbd_scene.js?v=3');
        const T = await import('/static/vendor/three.module.js');
        const camera = new T.PerspectiveCamera(45, 1, .01, 100);
        let error = 0;
        for (const [w,h] of [[640,480],[192,256]]) {
            const fx=180,fy=190,cx=w*.42,cy=h*.57;
            const width=700,height=400;
            calibratedProjection(camera,{width:w,height:h,intrinsics:[[fx,0,cx],[0,fy,cy],[0,0,1]]},width,height);
            for (const [u,v] of [[0,0],[w-1,h-1],[cx,cy]]) {
                const q = new T.Vector3((u-cx)/fx,-(v-cy)/fy,-1).project(camera);
                const scale=Math.min(width/w,height/h);
                error=Math.max(error,Math.abs((q.x+1)*width/2-((width-w*scale)/2+u*scale)),
                    Math.abs((1-q.y)*height/2-((height-h*scale)/2+v*scale)));
            }
        }
        return error;
    }""")
    assert error < 1e-9
    initial = page.evaluate("flashSceneState.viewer.view()")
    box = page.locator("#flash-experiment-scene").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.7
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 80, y - 20, steps=6)
    page.mouse.up()
    assert page.evaluate("flashSceneState.viewer.view().position") != initial["position"]
    page.get_by_role("button", name="Camera view", exact=True).click()
    assert page.evaluate("flashSceneState.viewer.view().position") == initial["position"]
    page.evaluate("seekFlashExperiment(1)")
    page.wait_for_selector('#flash-experiment-scene[data-frame="1"]')
    page.select_option("#flash-experiment-method", "surface_corrected")
    page.wait_for_selector('#flash-experiment-scene[data-method="surface_corrected"]')
    packet = page.evaluate(
        "({frame:flashSceneState.packet.frame, pose:flashSceneState.packet.objects.object.pose})"
    )
    assert packet["frame"] == 1 and np.array(packet["pose"])[:3, 3] == pytest.approx([0.01, 0.02, 0])
    # Changing recording while the RGB video is hidden must load its first 3D
    # frame without requiring a video presentation callback or a manual toggle.
    page.locator(f'.flash-experiment-item[data-id="{second_export.name}"]').click()
    page.wait_for_selector(f'#flash-experiment-scene[data-identity="{second_export.name}"][data-frame="0"]')
    # A hidden video's media clock must still advance the 3D player.
    page.evaluate("""() => {
        const video=document.getElementById('flash-experiment-video');
        video.currentTime=flashExperiments.data.frames[1].t+0.001;
        video.dispatchEvent(new Event('timeupdate'));
    }""")
    page.wait_for_selector('#flash-experiment-scene[data-frame="1"]')
    page.get_by_role("button", name="3D scene", exact=True).click()
    assert page.locator("#flash-experiment-scene").is_hidden()
    assert page.evaluate("flashSceneState.viewer") is None


@pytest.mark.requires_playwright
def test_shared_rgbd_shader_preserves_rgb_depth_alignment_and_missing_depth(gui_page):
    result = gui_page.evaluate("""async () => {
        const {createRgbdScene} = await import('/static/rgbd_scene.js?v=3');
        const canvas=document.createElement('canvas');
        canvas.style.cssText='position:fixed;left:0;top:0;width:320px;height:240px';
        document.body.append(canvas);
        const viewer=createRgbdScene(canvas), depth=new Float32Array(32*24).fill(1), rgb=new Uint8Array(32*24*4);
        for(let v=0;v<24;v++) for(let u=0;u<32;u++) {
            const i=v*32+u;
            rgb[i*4+(u<16?0:2)]=255; rgb[i*4+3]=255;
            if(v>15) depth[i]=0;
        }
        viewer.update(depth,rgb,{width:32,height:24,intrinsics:[[30,0,13],[0,30,10],[0,0,1]]},[]);
        viewer.draw();
        const copy=document.createElement('canvas');copy.width=canvas.width;copy.height=canvas.height;
        const context=copy.getContext('2d');context.drawImage(canvas,0,0);
        const pixel=(u,v)=>Array.from(context.getImageData(u*10,v*10,1,1).data);
        const result={left:pixel(8,8),right:pixel(24,8),missing:pixel(8,20)};
        viewer.dispose();canvas.remove();return result;
    }""")
    assert result["left"][:3] == [255, 0, 0]
    assert result["right"][:3] == [0, 0, 255]
    assert result["missing"][:3] != result["left"][:3]
    assert max(result["missing"][:3]) < 80


def test_supplied_label_model_uses_its_own_object_coordinates(tmp_path):
    from lerobot.gui.flash_experiment_scene import scene_packet

    run, _ = labelled_scene_run(tmp_path)
    model = tmp_path / "label-model.npy"
    vertices = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.06, 0]])
    np.save(model, vertices)
    case = json.loads((run / "case.json").read_text())
    case["evaluation_surfaces"] = {"object": {"points_m": str(model)}}
    (run / "case.json").write_text(json.dumps(case))
    output = export(run, tmp_path / "browser")
    manifest = json.loads((output / "manifest.json").read_text())
    data, _, _ = decode_scene_packet(scene_packet(manifest, 1, "ufm_chain"))
    obj = data["objects"]["object"]
    assert obj["label_kind"] == "dataset_model"
    pose = np.array(obj["labelled_pose"])
    placed = np.array(obj["labelled_surface"]) @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(placed, vertices + [0.3, 0.02, 0.8], atol=1e-7)
    assert len(obj["reference"]) == 4  # The scored reference remains untouched.


@pytest.mark.requires_playwright
def test_scene_ignores_inflight_frame_after_method_changes(gui_page, tmp_path, monkeypatch):
    run, _ = labelled_scene_run(tmp_path)
    root = tmp_path / "browser"
    export(run, root)
    monkeypatch.setenv("LEROBOT_FLASH_EXPERIMENT_DIR", str(root))
    page = gui_page
    held = []

    def delay_first(route):
        if not held:
            response = route.fetch()
            held.append((route, response))
        else:
            route.continue_()

    page.route("**/scene/*?method=*", delay_first)
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    page.wait_for_selector('#flash-experiment-overlay[data-frame="0"]')
    page.get_by_role("button", name="3D scene", exact=True).click()
    page.wait_for_function("flashSceneState.loading")
    # Let Playwright deliver the intercepted request before changing methods.
    page.wait_for_timeout(50)
    assert held
    page.select_option("#flash-experiment-method", "surface_corrected")
    route, response = held[0]
    route.fulfill(response=response)
    page.wait_for_selector('#flash-experiment-scene[data-method="surface_corrected"]')
    assert page.evaluate("flashSceneState.packet.method") == "surface_corrected"
    assert np.asarray(page.evaluate("flashSceneState.packet.objects.object.pose"))[:3, 3] == pytest.approx(
        [0, 0.02, 0]
    )
