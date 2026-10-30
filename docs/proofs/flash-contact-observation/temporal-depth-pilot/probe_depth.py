"""Isolated RGB + sensor-depth pilot. Never changes recorded depth or FLASH defaults.

Run with PromptDA's pinned source on PYTHONPATH alongside this repository's src.
Real missing pixels have no truth here. Holdout errors compare with withheld
sensor readings, not independently measured physical geometry.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache/flash-depth-completion-pilot-20261029"
MODEL = "depth-anything/prompt-depth-anything-vits"
REVISION = "bdc8715134ff95362ffd0402748c6f08071bc1e4"


def ycb(case, index=0, sam=False):
    root = Path.home() / ".cache/flash-failure-sweep-20261029" / case
    p = json.loads((root / "provenance.json").read_text())
    source = Path(p["source"])
    name = p["names"][index]
    rgb = cv2.cvtColor(cv2.imread(str(source / "rgb" / name)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(source / "depth" / name), -1).astype(np.float32) * 0.001
    if sam:
        e = json.loads(gzip.decompress((root / f"evidence/{index:06}.json.gz").read_bytes()))
        mask = np.zeros(depth.shape, bool)
        for start, count in e["masks_rle"]["object"]:
            mask.flat[start : start + count] = True
    else:
        mask = cv2.imread(str(source / "gt_mask" / name), 0) > 0
    return rgb, depth, mask, str(source / "rgb" / name)


def sources():
    yield "box_holes", *ycb("cracker_box_yalehand0-sam3", sam=True)
    yield "sugar_control", *ycb("sugar_box1")
    root = Path.home() / ".cache/huggingface/lerobot/demos/insertion/recording"
    rgb = cv2.cvtColor(cv2.imread(str(root / "rgb/000280.jpg")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(root / "depth/000280.png"), -1).astype(np.float32) * 0.001
    mask = cv2.imread(str(Path.home() / ".cache/flash-mask-depth-experiment/sam3-full/000280.png"), 0) > 0
    yield "dowel", rgb, depth, mask, str(root / "rgb/000280.jpg")
    for name, path in [
        ("dish", "/tmp/flash_case_retest_20261029/dish_000.npz"),  # nosec B108: offline scratch artifact
        ("mug", "/tmp/mug_000_audit.npz"),  # nosec B108: offline scratch artifact
    ]:
        a = np.load(path)
        yield name, a["rgb"], a["depth"].astype(np.float32), a["mask"].astype(bool), path


def errors(values):
    if not len(values):
        return {"median_mm": None, "p90_mm": None, "count": 0}
    return {
        "median_mm": float(np.median(values) * 1000),
        "p90_mm": float(np.percentile(values, 90) * 1000),
        "count": len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense-prompt",
        action="store_true",
        help="Nearest-fill raw holes before conditioning; PromptDA is trained with low-resolution dense LiDAR.",
    )
    parser.add_argument("--model", choices=["promptda", "lingbot"], default="promptda")
    parser.add_argument("--resolution-level", type=int, choices=range(10), default=9)
    args = parser.parse_args()
    suffix = "_dense_prompt" if args.dense_prompt else ""
    CACHE.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    if args.model == "promptda":
        from promptda.promptda import PromptDA

        model_id, revision = MODEL, REVISION
        source_dir = "/tmp/flash-promptda-20261029"  # nosec B108: offline scratch artifact
        model = (
            PromptDA.from_pretrained(model_id, model_kwargs={"encoder": "vits"}, revision=revision)
            .cuda()
            .eval()
        )
    else:
        from mdm.model.dinov2_rgbd.layers.block import Block
        from mdm.model.v2 import MDMModel

        model_id = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
        revision = "79204ed6b837f4fdd192cf563e59481fecfa0295"
        source_dir = "/tmp/flash-lingbot-depth-20261029"  # nosec B108: offline scratch artifact
        model = MDMModel.from_pretrained(model_id, revision=revision).cuda().eval()
        model.enable_pytorch_native_sdpa()

        # Official nested-list forward requires xFormers, absent on this host.
        # Batch one has no cross-item attention: use the same Block operations
        # on its sole tensor, preserving the official removal of missing-depth
        # tokens. Do not disable depth masking to bypass the dependency.
        def single_item_forward(block, x):
            if isinstance(x, list):
                assert len(x) == 1 and not block.training
                return [Block.forward(block, x[0])]
            return Block.forward(block, x)

        for block in model.encoder.backbone.blocks:
            block.forward = types.MethodType(single_item_forward, block)
        suffix = f"_lingbot_r{args.resolution_level}"
    report = {
        "model": model_id,
        "model_revision": revision,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "precision": "float32" if args.model == "promptda" else "official infer bfloat16 autocast",
        "parameters": sum(p.numel() for p in model.parameters()),
        "source_revision": subprocess.check_output(
            ["git", "-C", source_dir, "rev-parse", "HEAD"], text=True
        ).strip(),
        "input": "whole RGB image and whole sensor depth in metres; no SAM input to model",
        "hole_conditioning": "nearest measured pixel" if args.dense_prompt else "raw zeros",
        "resize": "RGB down to multiples of 14, maximum side 630; depth unchanged; output bilinear to original grid",
        "holdout": "25% of 8x8 image tiles, RNG 42, intersected with mask eroded by 3x3 and valid raw depth; set to zero before inference",
        "caveat": "Withheld sensor readings are not ground truth for true holes or transparency.",
        "cases": [],
    }
    if args.model == "lingbot":
        report["resize"] = (
            f"official infer internal token resize at resolution_level={args.resolution_level}; original RGB/depth supplied"
        )
        report["mask"] = "apply_mask=False to inspect all inferred depth; no output mask filter used"
        report["attention"] = (
            "official native SDPA; batch-one nested lists unwrapped through original Block.forward; depth token masking retained; xFormers parity not benchmarked"
        )
    for name, rgb, depth, mask, source in sources():
        assert rgb.shape[:2] == depth.shape == mask.shape
        assert depth.dtype == np.float32 and np.isfinite(depth).all() and depth.min() >= 0
        h, w = depth.shape
        scale = min(1.0, 630 / max(h, w))
        size = (int(w * scale) // 14 * 14, int(h * scale) // 14 * 14)
        if args.model == "lingbot":
            size = (w, h)
        rgb_t = (
            torch.from_numpy(cv2.resize(rgb.astype(np.float32) / 255, size, interpolation=cv2.INTER_AREA))
            .permute(2, 0, 1)[None]
            .cuda()
        )

        def predict(d, frame_rgb=rgb_t, frame_width=w, frame_height=h):
            if args.dense_prompt:
                indices = distance_transform_edt(d == 0, return_distances=False, return_indices=True)
                d = d[tuple(indices)]
            prompt = torch.from_numpy(d)[None, None].cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                if args.model == "promptda":
                    out = model.predict(frame_rgb, prompt)
                else:
                    out = model.infer(
                        frame_rgb,
                        depth_in=prompt[:, 0],
                        resolution_level=args.resolution_level,
                        apply_mask=False,
                    )["depth"][:, None]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            out = out[0, 0].float().cpu().numpy()
            return cv2.resize(
                out, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR
            ), elapsed * 1000

        for _ in range(3):
            predict(depth)
        times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            predicted, ms = predict(depth)
            times.append(ms)
        model_valid = np.isfinite(predicted) & (predicted > 0)
        hybrid = np.where(depth > 0, depth, np.where(model_valid, predicted, 0))
        assert np.array_equal(hybrid[depth > 0], depth[depth > 0])
        assert np.isfinite(hybrid).all() and (hybrid >= 0).all()
        rng = np.random.default_rng(42)
        tiles = rng.random(((h + 7) // 8, (w + 7) // 8)) < 0.25
        holdout = np.repeat(np.repeat(tiles, 8, 0), 8, 1)[:h, :w]
        holdout &= cv2.erode(mask.astype("uint8"), np.ones((3, 3), "uint8")) > 0
        holdout &= depth > 0
        withheld = depth.copy()
        withheld[holdout] = 0
        test, _ = predict(withheld)
        test_valid = np.isfinite(test) & (test > 0)
        indices = distance_transform_edt(withheld == 0, return_distances=False, return_indices=True)
        nearest = withheld[tuple(indices)]
        row = {
            "case": name,
            "source": source,
            "rgb_shape": list(rgb.shape),
            "network_shape": list(size[::-1]),
            "missing_mask_percent": float(100 * np.mean(depth[mask] == 0)),
            "forward_median_ms": float(np.median(times)),
            "forward_p90_ms": float(np.percentile(times, 90)),
            "torch_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
            "invalid_prediction_pixels": int(np.sum(~model_valid)),
            "invalid_prediction_object_pixels": int(np.sum(mask & ~model_valid)),
            "withheld_invalid_predictions": int(np.sum(holdout & ~test_valid)),
            "change_on_measured_mask": errors(np.abs(predicted - depth)[mask & (depth > 0) & model_valid]),
            "withheld_model": errors(np.abs(test - depth)[holdout & test_valid]),
            "withheld_nearest": errors(np.abs(nearest - depth)[holdout]),
            "input_sha256": hashlib.sha256(rgb.tobytes() + depth.tobytes() + mask.tobytes()).hexdigest(),
        }
        np.savez_compressed(
            CACHE / f"{name}{suffix}.npz",
            rgb=rgb,
            depth=depth,
            mask=mask,
            predicted=predicted,
            hybrid=hybrid,
            holdout=holdout,
            withheld=withheld,
            withheld_prediction=test,
            nearest=nearest,
        )
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
    (HERE / f"depth_results{suffix}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
