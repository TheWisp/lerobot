#!/usr/bin/env python
"""Aggregate chunk-cadence cells across replicate runs.

Reads run-stratified directories like outputs/chunk_cadence/safe_run{1,2,3}
and outputs/chunk_cadence/cylinder_run{1,2,3}; pairs cells by
(source, update_every, lookahead) and reports mean ± std across replicates.

Use after running ``run_sweep.sh N`` for N in {1, 2, 3, ...} and keeping
each run in its own subdirectory.

Usage:
  experiments/chunk_cadence/summarize_replicates.py outputs/chunk_cadence
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import analyze  # sibling module  # noqa: E402

_NAME_RE = re.compile(
    r"trace_N(?P<N>\d+)_L(?P<L>\d+)_bias(?P<bias>[\d.]+)"
    r"(?:_smooth(?P<sw>\d+))?"
    r"(?:_bm(?P<bm>[ou]))?"
    r"_seed(?P<seed>\d+)\.npz$"
)
# Teleop trace files: trace_teleop[_fps<f>]_L<lk>_noise<n>_seed<s>.npz
_TELEOP_RE = re.compile(
    r"trace_teleop(?:_fps(?P<fps>\d+))?_L(?P<L>\d+)_noise(?P<noise>[\d.]+)_seed(?P<seed>\d+)\.npz$"
)
_DIR_RE = re.compile(r"^(?P<source>.+?)_run(?P<run>\d+)$")


def _scan_runs(parent: Path) -> dict:
    """Walk parent dir, return {source: {(N, L, bias, seed): [run_indices: cell_dict]}}."""
    by_source: dict[str, dict] = {}
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        m = _DIR_RE.match(child.name)
        if not m:
            continue
        source = m.group("source")
        run = int(m.group("run"))
        for npz in sorted(child.glob("trace_*.npz")):
            # Teleop schema first (more specific match).
            tm = _TELEOP_RE.search(npz.name)
            if tm:
                lk = int(tm["L"])
                noise = float(tm["noise"])
                fps = int(tm["fps"]) if tm["fps"] else 30
                seed = int(tm["seed"])
                # Encode teleop into the same key shape as chunked so the
                # printer doesn't need a separate path. N=fps (overloaded
                # for teleop to carry the outer-loop rate). bias=noise.
                # sw=0. bm='teleop@<fps>' to distinguish rates in the table.
                key = (fps, lk, noise, 0, f"teleop@{fps}", seed)
                by_source.setdefault(source, {}).setdefault(key, []).append((run, analyze(npz)))
                continue
            mm = _NAME_RE.search(npz.name)
            if not mm:
                continue
            n = int(mm["N"])
            lk = int(mm["L"])
            bias = float(mm["bias"])
            sw = int(mm["sw"]) if mm["sw"] else 0
            bm = {"o": "overlap", "u": "uniform"}.get(mm["bm"] or "o", "overlap")
            seed = int(mm["seed"])
            key = (n, lk, bias, sw, bm, seed)
            by_source.setdefault(source, {}).setdefault(key, []).append((run, analyze(npz)))
    return by_source


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) <= 1:
        return float(values[0]) if values else 0.0, 0.0
    return statistics.fmean(values), statistics.stdev(values)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parent_dir", help="Directory containing <source>_run<N> subdirs")
    p.add_argument(
        "--output-json",
        default=None,
        help="Also write aggregated mean ± std to this file (one record per (source, N, L)).",
    )
    args = p.parse_args()

    by_source = _scan_runs(Path(args.parent_dir))
    json_rows = []
    for source, cells in by_source.items():
        print(f"\n== {source} (replicates: {sorted({run for v in cells.values() for run, _ in v})}) ==")
        header = (
            f"{'N':>3} {'bm':>8} {'sw':>3} {'L_ms':>5}  "
            f"{'state_lag_ms (mean±std)':>26}  "
            f"{'jitter_deg (mean±std)':>24}  "
            f"{'jump95 (mean±std)':>20}"
        )
        print(header)
        print("-" * len(header))
        for (n, lk, _bias, sw, bm, _seed), runs in sorted(cells.items()):
            lags = [-r["lag_no_lookahead_ms_median"] for _, r in runs]
            jitters = [
                r["jitter_lookahead_mean_deg"] if lk > 0 else r["jitter_no_lookahead_mean_deg"]
                for _, r in runs
            ]
            jumps = [r["boundary_jump_p95_deg"] for _, r in runs]
            lag_m, lag_s = _mean_std(lags)
            jit_m, jit_s = _mean_std(jitters)
            jmp_m, jmp_s = _mean_std(jumps)
            tag = "τ_motor" if lk == 0 else "residual"
            print(
                f"{n:>3} {bm:>8} {sw:>3d} {lk:>5d}  "
                f"{lag_m:>7.1f} ± {lag_s:>5.2f} ({tag:>8})  "
                f"{jit_m:>10.3f} ± {jit_s:>6.4f}     "
                f"{jmp_m:>8.3f} ± {jmp_s:>5.3f}"
            )
            json_rows.append(
                {
                    "source": source,
                    "update_every": n,
                    "lookahead_ms": lk,
                    "bias_threshold_deg": _bias,
                    "smooth_window": sw,
                    "bias_model": bm,
                    "n_replicates": len(runs),
                    "state_lag_ms_mean": lag_m,
                    "state_lag_ms_std": lag_s,
                    "state_lag_kind": tag,
                    "jitter_deg_mean": jit_m,
                    "jitter_deg_std": jit_s,
                    "boundary_jump_p95_deg_mean": jmp_m,
                    "boundary_jump_p95_deg_std": jmp_s,
                }
            )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(json_rows, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
