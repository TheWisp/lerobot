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

# Filename schema:
#   trace_N<update_every>_L<lookahead_ms>_bias<threshold>[_smooth<w>][_bm<o|u>]_seed<seed>.npz
_NAME_RE = re.compile(
    r"trace_N(?P<N>\d+)_L(?P<L>\d+)_bias(?P<bias>[\d.]+)"
    r"(?:_smooth(?P<sw>\d+))?"
    r"(?:_bm(?P<bm>[ou]))?"
    r"_seed(?P<seed>\d+)\.npz$"
)


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
            n = int(m["N"])
            lk = int(m["L"])
            bias = float(m["bias"])
            sw = int(m["sw"]) if m["sw"] else 0
            seed = int(m["seed"])
            key = (n, bias, sw, seed)
            cells.setdefault(key, {})[lk] = analyze(npz)

        if not cells:
            continue

        print(f"\n== {d.name} ==")
        header = (
            f"{'N':>3} {'bias':>5} {'sw':>3} {'L_ms':>5}  "
            f"{'state_lag_ms':>14}  {'jitter_deg':>10}  {'jump95_deg':>10}"
        )
        print(header)
        print("-" * len(header))
        for (n, bias, sw, seed), lk_map in sorted(cells.items()):
            for lk in sorted(lk_map):
                r = lk_map[lk]
                state_lag = -r["lag_no_lookahead_ms_median"]
                jit = r["jitter_lookahead_mean_deg"] if lk > 0 else r["jitter_no_lookahead_mean_deg"]
                jump = r["boundary_jump_p95_deg"]
                tag = "τ_motor" if lk == 0 else "residual"
                print(
                    f"{n:>3} {bias:>5.1f} {sw:>3d} {lk:>5d}  "
                    f"{state_lag:>7.1f} ms ({tag:>8})  "
                    f"{jit:>10.3f}  {jump:>10.3f}"
                )
                all_rows.append(
                    {
                        "source": d.name,
                        "N": n,
                        "L": lk,
                        "bias": bias,
                        "smooth_window": sw,
                        "seed": seed,
                        **r,
                    }
                )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(all_rows, indent=2, default=str))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
