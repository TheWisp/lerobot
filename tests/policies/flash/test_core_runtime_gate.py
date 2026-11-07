"""The performance gate consumes the replay output, including published poses."""

import json

from benchmarks import flash_core_gate as gate, flash_runtime_benchmark as runtime


def _run(root, *, time_ms=80, correct=2, geometry=True, digest="same"):
    case_id = "recorded-case"
    path = root / "000" / case_id
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "frame": frame,
            "processing_ms": (time_ms[frame - 1] if isinstance(time_ms, list) else time_ms) if frame else 10,
            "objects": {
                "object": {
                    "group_growing": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "group_growing_status": "observed_candidate",
                    "group_growing_details": {"fit": {"geometry_check": {"accepted": geometry}}},
                }
            },
        }
        for frame in range(3)
    ]
    (path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "summary.json").write_text(
        json.dumps(
            [
                {
                    "case": case_id,
                    "frames": 3,
                    "input_sha256": digest,
                    "processing": {"method": "group_growing"},
                    "roles": {
                        "object": {
                            "methods": {
                                "group_growing": {
                                    "pose_error": {"within_20mm_20deg": correct, "available": 2, "total": 2}
                                }
                            }
                        }
                    },
                }
            ]
        )
    )


def test_gate_reads_latency_labels_and_actual_geometry_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "CORE_CASES", ("recorded-case",))
    monkeypatch.setattr(gate, "LABELLED_FLOORS", {"recorded-case": ("object", 2)})
    monkeypatch.setattr(gate, "CLEAR_AVAILABILITY", {})
    _run(tmp_path)
    assert gate.evaluate(tmp_path, max_p90_ms=100, max_frame_ms=100)["pass"]

    # These three independent mutations must each make the real output fail.
    _run(tmp_path, time_ms=[80, 200])
    assert not gate.evaluate(tmp_path, max_p90_ms=100)["pass"]
    assert not gate.evaluate(tmp_path, max_stddev_ms=20)["pass"]
    _run(tmp_path, correct=1)
    assert any("correct labelled poses" in x for x in gate.evaluate(tmp_path)["failures"])
    _run(tmp_path, geometry=False)
    assert any("unvalidated pose published" in x for x in gate.evaluate(tmp_path)["failures"])


def test_comparison_requires_same_input_and_does_not_use_fit_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "CORE_CASES", ("recorded-case",))
    monkeypatch.setattr(gate, "LABELLED_FLOORS", {"recorded-case": ("object", 1)})
    monkeypatch.setattr(gate, "CLEAR_AVAILABILITY", {})
    before, after = tmp_path / "before", tmp_path / "after"
    _run(before, correct=2)
    _run(after, correct=1)
    assert any("accuracy regressed" in x for x in gate.evaluate(after, baseline=before)["failures"])
    _run(after, digest="different")
    assert any("inputs differ" in x for x in gate.evaluate(after, baseline=before)["failures"])


def test_latency_distribution_reports_tail_and_variance():
    stats = runtime.latency_distribution([10, 10, 10, 210])
    assert stats["median"] == 10
    assert stats["p90"] > 100
    assert stats["stddev"] > 0
    assert stats["over_100ms"] == 1
