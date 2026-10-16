"""Measure live tracking while read-only registration jobs compete for compute.

The arm must be disconnected. Does not change tracking designations or command
motion. Run with worker/start parallel_recovery false and true for an A/B.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import requests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://127.0.0.1:9100")
    ap.add_argument("--object", required=True)
    ap.add_argument("--click", required=True, help="x,y in the live RGB image")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    http = requests.Session()
    assert not http.get(args.server + "/api/jog/state", timeout=5).json()["connected"]
    state = http.get(args.server + "/api/pregrasp/state", timeout=5).json()
    assert state["track"]["on"], "start continuous tracking before benchmarking"
    rows, jobs, errors = [], [], []
    stop = threading.Event()

    def poll():
        seen = set()
        with requests.Session() as reader:
            while not stop.is_set():
                try:
                    response = reader.get(args.server + "/api/pregrasp/state", timeout=5)
                    response.raise_for_status()
                    track = response.json()["track"]
                    last = track.get("last") or {}
                    checks = last.get("pose_verification") or []
                    stamp = checks[0]["t_frame"] if checks else None
                    if stamp is not None and stamp not in seen:
                        seen.add(stamp)
                        rows.append(
                            {
                                "received": time.time(),
                                "stamp": stamp,
                                "fps": track["fps"],
                                "timing_ms": last.get("timing_ms"),
                                "render_stages_ms": last.get("render_stages_ms"),
                            }
                        )
                except Exception as exc:
                    errors.append(str(exc))
                stop.wait(0.02)

    thread = threading.Thread(target=poll)
    thread.start()
    try:
        for _ in range(args.jobs):
            started = time.time()
            response = http.post(
                args.server + "/api/pregrasp/locate",
                json={
                    "object": args.object,
                    "click": [int(x) for x in args.click.split(",")],
                    "track": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            jobs.append(
                {"started": started, "elapsed_ms": 1000 * (time.time() - started), "result": response.json()}
            )
    finally:
        stop.set()
        thread.join(timeout=6)
    assert len(rows) >= 3, "no fresh tracking results: this is not a concurrency measurement"
    keys = sorted({k for row in rows for k in (row["timing_ms"] or {})})
    summary = {}
    for key in keys:
        values = [row["timing_ms"][key] for row in rows if key in (row["timing_ms"] or {})]
        summary[key] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    result = {"frames": rows, "jobs": jobs, "errors": errors, "summary": summary}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"frames": len(rows), "jobs": len(jobs), "summary": summary, "errors": errors}))


if __name__ == "__main__":
    main()
