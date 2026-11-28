"""Run the complete causal Point2Pose pose pipeline on prepared RGB-D clips.

The supplied-mask pass isolates pose registration from segmentation. The
internal-mask pass supplies the same initializer, then uses Point2Pose's SAM2
for both its first output and subsequent masks.
All paths to the sealed bridge and pinned Point2Pose environment are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(args: argparse.Namespace, case: Path, mode: str) -> Path | None:
    manifest_path = case / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first, last = int(manifest["first_frame"]), int(manifest["last_frame"])
    output = args.output_root / case.name / mode
    output.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(manifest["mask_dir"])
    if mode == "oracle_masks":
        mask_parent = output / "mask_input"
        mask_parent.mkdir(exist_ok=True)
        mask_link = mask_parent / "object"
        if not mask_link.exists():
            mask_link.symlink_to(mask_dir, target_is_directory=True)
    else:
        mask_parent = None
    result = output / "p2p_object_pose.npz"
    if result.exists() and not args.force:
        print(f"already complete: {result}", flush=True)
        return result
    prior = output / "provenance.json"
    if prior.exists() and not args.force and json.loads(prior.read_text()).get("exit_code") != 0:
        print(f"already failed: {case.name} {mode}; see {output / 'run.log'}", flush=True)
        return None
    command = [
        sys.executable,
        str(args.runner),
        "--dataset",
        str(Path(manifest["recording"])),
        "--initial-mask",
        str(mask_dir / f"{first:06d}.png"),
        "--role",
        "object",
        "--mask-source",
        "external" if mode == "oracle_masks" else "internal",
        "--honor-initial-mask" if mode == "oracle_masks" else "--save-masks",
        "--label",
        mode,
        "--python",
        str(args.p2p_python),
        "--sealed-bridge",
        str(args.sealed_bridge),
        "--p2p-repo",
        str(args.p2p_repo),
        "--p2p-config",
        str(args.p2p_config),
        "--output",
        str(output),
        "--output-file",
        str(result),
        "--first-frame",
        str(first),
        "--last-frame",
        str(last),
    ]
    if mask_parent is not None:
        command.extend(("--current-masks", str(mask_parent)))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.sealed_src) + os.pathsep + environment.get("PYTHONPATH", "")
    provenance = {
        "case": str(case),
        "mode": mode,
        "manifest_sha256": sha256(manifest_path),
        "initial_mask_sha256": sha256(mask_dir / f"{first:06d}.png"),
        "p2p_config_sha256": sha256(args.p2p_config),
        "p2p_checkout": subprocess.check_output(
            ["git", "-C", str(args.p2p_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sealed_bridge_sha256": sha256(args.sealed_bridge),
        "first_frame": first,
        "last_frame": last,
        "initial_mask_exact": mode == "oracle_masks",
        "current_masks": str(mask_dir) if mode == "oracle_masks" else "Point2Pose SAM2",
    }
    started = time.monotonic()
    with (output / "run.log").open("w") as log:
        completed = subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    provenance["wall_s"] = time.monotonic() - started
    provenance["exit_code"] = completed.returncode
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if completed.returncode:
        print(f"failed: {case.name} {mode}; see {output / 'run.log'}", flush=True)
        return None
    print(f"complete: {result} ({provenance['wall_s']:.1f}s)", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--sealed-src", type=Path, required=True)
    parser.add_argument("--sealed-bridge", type=Path, required=True)
    parser.add_argument("--p2p-python", type=Path, required=True)
    parser.add_argument("--p2p-repo", type=Path, required=True)
    parser.add_argument("--p2p-config", type=Path, required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=("oracle_masks", "internal_masks"), default=["oracle_masks"]
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for case in args.cases:
        for mode in args.modes:
            run_case(args, case, mode)


if __name__ == "__main__":
    main()
