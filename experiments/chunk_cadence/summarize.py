#!/usr/bin/env python
"""Cross-cadence summary of chunk-cadence backtest cells.

Reads every trace .npz in a directory, pairs the lookahead and no-lookahead
runs per (source, update_every), and prints a comparison table.

Usage:
  experiments/chunk_cadence/summarize.py outputs/chunk_cadence/<source>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import analyze  # sibling module  # noqa: E402

_NAME_RE = re.compile(r"trace_N(\d+)_L(\d+)_bias([\d.]+)_seed(\d+)\.npz$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source_dir", nargs="+", help="One or more directories of trace files")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    all_rows = []
    for source_dir in args.source_dir:
        d = Path(source_dir)
        if not d.is_dir():
            print(f"[skip] {d} (not a directory)")
            continue
        cells = {}
        for npz in sorted(d.glob("trace_*.npz")):
            m = _NAME_RE.search(npz.name)
            if not m:
                continue
            n, lk, bias, seed = int(m[1]), int(m[2]), float(m[3]), int(m[4])
            key = (n, bias, seed)
            cells.setdefault(key, {})[lk] = analyze(npz)

        if not cells:
            continue

        print(f"\n== {d.name} ==")
        header = (
            f"{'N':>3} {'bias':>5} {'L_ms':>5}  {'state_lag_ms':>14}  {'jitter_deg':>10}  {'jump95_deg':>10}"
        )
        print(header)
        print("-" * len(header))
        for (n, bias, seed), lk_map in sorted(cells.items()):
            for lk in sorted(lk_map):
                r = lk_map[lk]
                # state_lag = how much state TRAILS motor_cmd, in ms (positive).
                # When L=0 this IS the bare motor response time τ_motor. When
                # L>0 it's the residual: the part the lookahead failed to
                # compensate (ideal would be 0; we measure τ_motor − L).
                # Take −1 × xcorr_lag to flip the convention from "motor
                # leads state by X" to "state trails motor by X".
                state_lag = -r["lag_no_lookahead_ms_median"]
                jit = r["jitter_lookahead_mean_deg"] if lk > 0 else r["jitter_no_lookahead_mean_deg"]
                jump = r["boundary_jump_p95_deg"]
                tag = "τ_motor" if lk == 0 else "residual"
                print(
                    f"{n:>3} {bias:>5.1f} {lk:>5d}  "
                    f"{state_lag:>7.1f} ms ({tag:>8})  "
                    f"{jit:>10.3f}  {jump:>10.3f}"
                )
                all_rows.append({"source": d.name, "N": n, "L": lk, "bias": bias, "seed": seed, **r})

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(all_rows, indent=2, default=str))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
