"""Recompute the examples currently listed by the FLASH experiment browser.

Runs in a separate process. Old exports are immutable; a completed replacement
is published into the same list slot. Source recordings and masks are read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from html import escape
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_report(job: Path, request: dict, state: dict, outcomes: dict):
    """Keep failures and unlabelled cases visible alongside measured scores."""
    cases = [
        {
            "name": item["name"],
            "source_run": item["source_run"],
            "mask_source": item.get("mask_source", "Recorded mask"),
            **outcomes.get(item["id"], {"status": "not_completed"}),
        }
        for item in request["items"]
    ]
    write_json(job / "report.json", {"job": state, "cases": cases})
    rows = []

    def number(value):
        return "—" if value is None else f"{value:.2f}"

    for case in cases:
        name = escape(case["name"])
        if case["status"] != "completed":
            message = escape(case.get("error", "Not completed in this job"))
            rows.append(f'<tr><td>{name}</td><td colspan="7">{message}</td></tr>')
            continue
        summary = case["summary"]
        for role, result in summary["roles"].items():
            methods = result["methods"]
            if not methods:
                rows.append(
                    f'<tr><td>{name}</td><td>{escape(role)}</td><td colspan="6">No initialization</td></tr>'
                )
            for method, data in methods.items():
                errors = data.get("pose_error", {})
                labels = summary["accuracy_labels"]
                cloud = " / ".join(number(errors.get(f"cloud_mm_{q}")) for q in ("median", "p90"))
                angle = " / ".join(number(errors.get(f"angle_deg_{q}")) for q in ("median", "p90"))
                timing = result.get("timing_ms", {})
                prepare = number((timing.get("prepare_ms") or {}).get("median"))
                fit = number((timing.get("finish_ms") or {}).get("median"))
                cells = [
                    case["name"],
                    role,
                    method,
                    f"{data['available_after_initialization']} / {data['total_after_initialization']}",
                    cloud if labels else "Unmeasured",
                    angle if labels else "Unmeasured",
                    f"{prepare} / {fit}",
                    case["mask_source"],
                ]
                rows.append("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in cells) + "</tr>")
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>FLASH benchmark results</title><style>body{font:16px/1.5 system-ui;margin:24px;color:#18283e}"
        "table{border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}"
        "th{background:#eef3f8}main{overflow-x:auto}a{color:#1265a0}</style>"
        '<a href="/#/run/flash">Back to FLASH</a><h1>Benchmark results</h1>'
        f"<p>{state['succeeded']} / {state['total']} completed · {len(state['errors'])} failed.</p>"
        "<p>Pose availability is not correctness. Errors use labelled frames where a pose was returned; "
        "different methods may return poses on different frames. Initialization is excluded.</p>"
        "<p>Point error measures displacement of the same initial 3D points against annotated motion. "
        "Rotation error uses the annotated orientation without symmetry adjustment. "
        "Unlabelled examples have no accuracy score.</p>"
        "<p>Times are per-object stages shared by the compared methods, not independent end-to-end FPS. "
        "Saved masks and case configurations are reused; SAM is not rerun. "
        '<a href="report.json">Download full results (JSON)</a></p><main><table><thead><tr>'
        "<th>Example</th><th>Object</th><th>Method</th><th>Poses / frames</th>"
        "<th>Point error mm<br>median / p90</th><th>Rotation error °<br>median / p90</th>"
        "<th>Prepare / fit ms<br>medians</th><th>Mask source</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></main></html>"
    )
    temporary = job / "report.html.tmp"
    temporary.write_text(page)
    temporary.replace(job / "report.html")


def rebuild(job: Path):
    # Import model/dataset dependencies only inside this process, never the GUI.
    from benchmarks.flash_experiment_browser import export
    from benchmarks.flash_tracker_experiment import code_snapshot, run_case
    from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
    from lerobot.policies.flash.surface_tracking import load_ufm
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    request = json.loads((job / "request.json").read_text())
    state = json.loads((job / "status.json").read_text())
    outcomes = {}
    root = Path(request["browser_root"])
    state.update(state="running", phase="Loading UFM", pid=os.getpid())
    write_json(job / "status.json", state)
    try:
        provenance = code_snapshot(job)
        write_json(job / "provenance.json", provenance)
        model = load_ufm()
        kernels = None
        for index, item in enumerate(request["items"]):
            state.update(current=item["name"], frame=0, frames=item["frames"], phase="Tracking")
            write_json(job / "status.json", state)
            pool = None
            try:
                previous_run = Path(item["source_run"])
                case = json.loads((previous_run / "case.json").read_text())
                previous_provenance = json.loads((previous_run.parent / "provenance.json").read_text())
                config = VisionPipelineConfig.from_dict(previous_provenance["config"])
                case_dir = job / "runs" / f"{index:03d}"
                case_dir.mkdir(parents=True)
                write_json(
                    case_dir / "provenance.json",
                    {
                        "config": previous_provenance["config"],
                        "source_run": str(previous_run),
                        "code_snapshot": str(job),
                        "record_correspondences": True,
                    },
                )
                if config.compiled_surface_fit and kernels is None:
                    kernels = compiled_kernels()
                if config.reuse_surface_search:
                    pool = SearchPool(config.shape.batch_query_workers)
                last_report = 0.0

                def progress(frame, total):
                    nonlocal last_report
                    now = time.monotonic()
                    if now - last_report >= 0.5 or frame == total:
                        state.update(frame=frame, frames=total)
                        write_json(job / "status.json", state)
                        last_report = now

                run = case_dir / case["id"]
                run_case(
                    case,
                    run,
                    model,
                    config,
                    kernels if config.compiled_surface_fit else None,
                    pool,
                    retain_motion_reference="ufm_retained_reference" in item["methods"],
                    record_correspondences=True,
                    progress=progress,
                )
                state.update(phase="Building playback")
                write_json(job / "status.json", state)
                # Stage the whole export outside the public catalogue. Publish
                # only after adding its stable slot identity and build timestamp.
                staged = export(run, job / "exports")
                manifest = json.loads((staged / "manifest.json").read_text())
                manifest.update(
                    slot_id=item.get("slot_id", item["id"]), built_at_ns=time.time_ns(), experiment=job.name
                )
                write_json(staged / "manifest.json", manifest)
                staged.rename(root / staged.name)
                outcomes[item["id"]] = {
                    "status": "completed",
                    "run": str(run),
                    "config": previous_provenance["config"],
                    "summary": json.loads((run / "summary.json").read_text()),
                }
                state["succeeded"] += 1
            except Exception as exc:
                traceback.print_exc()
                state["errors"].append({"name": item["name"], "error": str(exc)})
                outcomes[item["id"]] = {"status": "failed", "error": str(exc)}
            finally:
                if pool is not None:
                    pool.close()
            state["completed"] += 1
            write_json(job / "status.json", state)
        state.update(state="completed" if not state["errors"] else "completed_with_errors", phase="Finished")
    except BaseException as exc:
        traceback.print_exc()
        state.update(state="failed", phase="Stopped", error=str(exc))
        raise
    finally:
        write_report(job, request, state, outcomes)
        state["report_ready"] = True
        write_json(job / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    # The GUI transfers its flock to the child so no second job can start during
    # startup or a server restart. The descriptor stays open until process exit.
    parser.add_argument("--lock-fd", type=int, required=True)
    args = parser.parse_args()
    os.fstat(args.lock_fd)
    job = args.job.resolve()
    try:
        rebuild(job)
    except Exception as exc:
        state = json.loads((job / "status.json").read_text())
        state.update(state="failed", phase="Stopped", error=str(exc))
        write_json(job / "status.json", state)
        raise


if __name__ == "__main__":
    main()
