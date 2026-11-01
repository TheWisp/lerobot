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
    monkeypatch.setattr(surface_tracking, "load_ufm", lambda: model)
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


def test_rebuild_lock_survives_launcher_and_detects_worker_exit(tmp_path, monkeypatch):
    import subprocess
    import sys

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
            assert client.post(base).status_code == 202
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
    page.get_by_role("button", name="Run all benchmarks").click()
    page.wait_for_function(
        "document.getElementById('flash-experiment-rebuild-status').textContent.includes('2/20')"
    )
    assert posts == ["POST"]
    assert page.get_by_role("button", name="Run all benchmarks").is_disabled()
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
    assert page.get_by_role("button", name="Run all benchmarks").is_enabled()
    page.locator("#flash-experiment-rebuild-errors summary").click()
    assert "Missing depth" in page.locator("#flash-experiment-rebuild-errors").inner_text()
    assert page.locator(".flash-experiment-item").count() == 1

    state.update(report_ready=True, id="rebuild-20261101-120000-1234abcd")
    page.wait_for_function("!document.getElementById('flash-experiment-report').hidden")
    assert (
        page.get_by_role("link", name="View results")
        .get_attribute("href")
        .endswith("/rebuild/rebuild-20261101-120000-1234abcd/report.html")
    )
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
