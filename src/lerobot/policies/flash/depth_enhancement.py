"""Optional learned depth sources. Sensor measurements remain separate inputs.

LingBot refines aligned RGB-D. Fast-FoundationStereo and Lite Any Stereo v2
estimate depth from a synchronized rectified infrared pair. Model outputs
remain distinct from camera measurements.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

DEPTH_SOURCES = (
    "sensor",
    "lingbot",
    "foundation_stereo",
    "lite_any_stereo_v2_s",
    "lite_any_stereo_v2_m",
    "lite_any_stereo_v2_l",
    "lite_any_stereo_v2",  # Existing H selection and saved-session identifier.
)


def needs_stereo_pair(source: str) -> bool:
    return source == "foundation_stereo" or source.startswith("lite_any_stereo_v2")


@dataclass(frozen=True)
class StereoInput:
    left: np.ndarray
    right: np.ndarray
    left_intrinsics: np.ndarray
    color_from_left: np.ndarray
    baseline_m: float


@dataclass(frozen=True)
class DepthInput:
    rgb: np.ndarray
    sensor_depth_m: np.ndarray
    color_intrinsics: np.ndarray
    stereo: StereoInput | None = None


@dataclass(frozen=True)
class DepthOutput:
    depth_m: np.ndarray
    source: str
    # Pixels with positive sensor readings retain their provenance even if a
    # selected model proposes a different value at the same pixel.
    sensor_measured: np.ndarray


def tracking_frame_with_depth_source(frame, source, stereo_geometry=None):
    """Apply the selected depth model to a captured frame for live or replay.

    The source camera frame is retained separately by recording; this returns
    only the frame to pass to the shared tracker. Missing predictions stay zero.
    """
    if source is None:
        return frame
    stereo = (
        StereoInput(frame.infrared_left, frame.infrared_right, *stereo_geometry)
        if stereo_geometry is not None
        else None
    )
    selected = source.predict(DepthInput(frame.rgb, frame.metric_depth(), frame.intrinsics, stereo))
    if np.issubdtype(frame.depth.dtype, np.integer):
        depth = np.rint(selected.depth_m / frame.depth_m_per_unit).clip(0, 65535).astype(np.uint16)
    else:
        depth = selected.depth_m.astype(np.float32)
    return replace(frame, depth=depth, depth_before_filters=frame.depth)


def _valid_output(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != shape:
        raise ValueError(f"Depth model returned {depth.shape}, expected {shape}")
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)


def _stereo_disparity_gpu(session, inputs):
    """Keep supported TensorRT/ONNX outputs on CUDA; test doubles use run()."""
    if hasattr(session, "run_gpu"):
        return session.run_gpu(None, inputs)[0]
    if not hasattr(session, "io_binding"):
        return None
    import torch

    output = session.get_outputs()[0]
    if output.name != "disparity" or output.type != "tensor(float)":
        raise ValueError("Stereo ONNX output must be FP32 disparity")
    # Explicit trial ONNX exports may have symbolic output dimensions. Let
    # ONNX Runtime allocate those and use the existing CPU projection path.
    if any(not isinstance(dimension, int) for dimension in output.shape):
        return None
    result = torch.empty(tuple(output.shape), device="cuda:0", dtype=torch.float32)
    binding = session.io_binding()
    for name, value in inputs.items():
        binding.bind_cpu_input(name, value)
    binding.bind_output("disparity", "cuda", 0, np.float32, tuple(result.shape), result.data_ptr())
    session.run_with_iobinding(binding)
    return result


def color_aligned_stereo_depth(
    left_depth_m: np.ndarray,
    left_k: np.ndarray,
    color_k: np.ndarray,
    color_from_left: np.ndarray,
    color_shape: tuple[int, int],
) -> np.ndarray:
    """Forward-project stereo depth into color pixels, keeping the nearest hit.

    Invalid stereo pixels remain holes. The 2x2 splat covers the color grid
    when the color stream has higher resolution than the IR pair.
    """
    h, w = left_depth_m.shape
    y, x = np.indices((h, w), dtype=np.float32)
    z = left_depth_m.ravel()
    good = np.isfinite(z) & (z > 0)
    if not np.any(good):
        return np.zeros(color_shape, np.float32)
    x, y, z = x.ravel()[good], y.ravel()[good], z[good]
    points = np.stack(
        ((x - left_k[0, 2]) * z / left_k[0, 0], (y - left_k[1, 2]) * z / left_k[1, 1], z), axis=0
    )
    moved = color_from_left[:3, :3] @ points + color_from_left[:3, 3:4]
    cz = moved[2]
    good = np.isfinite(cz) & (cz > 0)
    u = color_k[0, 0] * moved[0, good] / cz[good] + color_k[0, 2]
    v = color_k[1, 1] * moved[1, good] / cz[good] + color_k[1, 2]
    cz = cz[good].astype(np.float32)
    pixel_x = np.floor(u).astype(np.int32)
    pixel_y = np.floor(v).astype(np.int32)
    color_height, color_width = color_shape
    # Flatten the z-buffer so each splat reuses the same projected pixels.
    out = np.full(color_height * color_width, np.inf, np.float32)
    base_index = pixel_y * color_width + pixel_x
    for dx in (0, 1):
        x_inside = (pixel_x + dx >= 0) & (pixel_x + dx < color_width)
        for dy in (0, 1):
            inside = x_inside & (pixel_y + dy >= 0) & (pixel_y + dy < color_height)
            np.minimum.at(out, base_index[inside] + dy * color_width + dx, cz[inside])
    out = out.reshape(color_shape)
    out[~np.isfinite(out)] = 0
    return fill_projection_gaps(out)


def fill_projection_gaps(depth: np.ndarray, *, max_neighbor_spread_m: float = 0.005) -> np.ndarray:
    """Fill only one-pixel resampling gaps on locally continuous surfaces.

    Forward projection from the 848x480 IR grid to 1280x720 color leaves a
    regular pixel lattice empty even when every stereo pixel has depth. A hole
    is filled only when at least three neighboring projected hits agree within
    5 mm. Depth steps and larger gaps remain unknown.
    """
    valid = depth > 0
    kernel = np.ones((3, 3), np.uint8)
    count = cv2.boxFilter(
        valid.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    total = cv2.boxFilter(depth, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    near = cv2.erode(np.where(valid, depth, 1e6), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1e6)
    far = cv2.dilate(np.where(valid, depth, -1e6), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=-1e6)
    fill = ~valid & (count >= 3) & ((far - near) <= max_neighbor_spread_m)
    result = depth.copy()
    result[fill] = total[fill] / count[fill]
    return result


def color_aligned_stereo_depth_gpu(left_depth_m, left_k, color_k, color_from_left, color_shape):
    """Project GPU stereo depth to RGB, then copy only the final map to CPU."""
    import torch
    import torch.nn.functional as functional

    if not left_depth_m.is_cuda:
        raise ValueError("GPU stereo projection requires depth on CUDA")
    height, width = left_depth_m.shape
    color_height, color_width = color_shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=left_depth_m.device),
        torch.arange(width, device=left_depth_m.device),
        indexing="ij",
    )
    valid = torch.isfinite(left_depth_m) & (left_depth_m > 0)
    x = x_grid[valid].double()
    y = y_grid[valid].double()
    z = left_depth_m[valid].double()
    left_k = np.asarray(left_k)
    color_k = np.asarray(color_k)
    points = torch.stack(
        ((x - left_k[0, 2]) * z / left_k[0, 0], (y - left_k[1, 2]) * z / left_k[1, 1], z), dim=0
    )
    transform = torch.as_tensor(color_from_left, device=left_depth_m.device, dtype=torch.float64)
    moved = transform[:3, :3] @ points + transform[:3, 3:4]
    cz = moved[2]
    positive = torch.isfinite(cz) & (cz > 0)
    u = color_k[0, 0] * moved[0, positive] / cz[positive] + color_k[0, 2]
    v = color_k[1, 1] * moved[1, positive] / cz[positive] + color_k[1, 2]
    depth = cz[positive].float()
    pixel_x = torch.floor(u).long()
    pixel_y = torch.floor(v).long()
    base_index = pixel_y * color_width + pixel_x
    out = torch.full(
        (color_height * color_width,), float("inf"), device=left_depth_m.device, dtype=torch.float32
    )
    for dx in (0, 1):
        x_inside = (pixel_x + dx >= 0) & (pixel_x + dx < color_width)
        for dy in (0, 1):
            inside = x_inside & (pixel_y + dy >= 0) & (pixel_y + dy < color_height)
            out.scatter_reduce_(0, base_index[inside] + dy * color_width + dx, depth[inside], reduce="amin")
    out = torch.where(torch.isfinite(out), out, 0).reshape(1, 1, color_height, color_width)

    # Match fill_projection_gaps: only one-pixel sampling gaps on a locally flat surface.
    measured = out > 0
    count = functional.avg_pool2d(measured.float(), 3, stride=1, padding=1) * 9
    total = functional.avg_pool2d(out, 3, stride=1, padding=1) * 9
    near = -functional.max_pool2d(-torch.where(measured, out, 1e6), 3, stride=1, padding=1)
    far = functional.max_pool2d(torch.where(measured, out, -1e6), 3, stride=1, padding=1)
    fill = (~measured) & (count >= 3) & ((far - near) <= 0.005)
    return torch.where(fill, total / count.clamp_min(1), out)[0, 0].cpu().numpy()


class LingBotDepth:
    """LingBot-Depth-v0.5, loaded only when selected."""

    source = "lingbot"

    def __init__(self, *, resolution_level: int = 0):
        import types

        import torch
        from mdm.model.dinov2_rgbd.layers.block import Block
        from mdm.model.v2 import MDMModel

        self.torch = torch
        self.resolution_level = resolution_level
        self.model = MDMModel.from_pretrained("robbyant/lingbot-depth-pretrain-vitl-14-v0.5").cuda().eval()
        self.model.enable_pytorch_native_sdpa()

        # On hosts without xFormers, upstream's batch-one nested-list path
        # still calls it. The model's original tensor path uses native SDPA.
        def one_item(block, values):
            if isinstance(values, list):
                if len(values) != 1:
                    raise ValueError("LingBot adapter requires batch one")
                return [Block.forward(block, values[0])]
            return Block.forward(block, values)

        for block in self.model.encoder.backbone.blocks:
            block.forward = types.MethodType(one_item, block)

    def predict(self, frame: DepthInput) -> DepthOutput:
        if frame.rgb.shape[:2] != frame.sensor_depth_m.shape:
            raise ValueError("LingBot needs color-aligned sensor depth")
        torch = self.torch
        rgb = torch.from_numpy(np.ascontiguousarray(frame.rgb))
        rgb = rgb.permute(2, 0, 1)[None].cuda().float() / 255
        depth = torch.from_numpy(np.ascontiguousarray(frame.sensor_depth_m))[None].cuda()
        with torch.inference_mode():
            predicted = (
                self.model.infer(
                    rgb,
                    depth_in=depth,
                    resolution_level=self.resolution_level,
                    apply_mask=False,
                )["depth"][0]
                .float()
                .cpu()
                .numpy()
            )
        predicted = _valid_output(predicted, frame.sensor_depth_m.shape)
        return DepthOutput(predicted, self.source, frame.sensor_depth_m > 0)


class FoundationStereoDepth:
    """Fast-FoundationStereo disparity on rectified infrared images."""

    source = "foundation_stereo"

    def __init__(self, model_path: str | Path | None = None):
        path = Path(
            model_path
            or os.environ.get("LEROBOT_FOUNDATION_STEREO_MODEL")
            or os.environ.get("LEROBOT_FOUNDATION_STEREO_ONNX")
            or Path.home()
            / ".cache/flash-foundationstereo/fixed-480x864-d192-i8/fast_foundationstereo.engine"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Fast-FoundationStereo model not found: {path}. "
                "Build the fixed 480x864 TensorRT engine or set LEROBOT_FOUNDATION_STEREO_MODEL."
            )
        if path.suffix == ".engine":
            self.session = _TensorRtSession(path)
        elif path.suffix == ".onnx":
            self.session = _cuda_onnx_session(path)
        else:
            raise ValueError("Fast-FoundationStereo model must be a TensorRT .engine or .onnx file")

    @staticmethod
    def _aligned_output(frame: DepthInput, disparity: np.ndarray, source: str) -> DepthOutput:
        stereo = frame.stereo
        left_depth = np.divide(
            stereo.left_intrinsics[0, 0] * stereo.baseline_m,
            disparity,
            out=np.zeros(disparity.shape, np.float32),
            where=disparity > 0.01,
        )
        aligned = color_aligned_stereo_depth(
            left_depth,
            stereo.left_intrinsics,
            frame.color_intrinsics,
            stereo.color_from_left,
            frame.sensor_depth_m.shape,
        )
        return DepthOutput(
            _valid_output(aligned, frame.sensor_depth_m.shape), source, frame.sensor_depth_m > 0
        )

    @staticmethod
    def _aligned_output_gpu(frame: DepthInput, disparity, source: str) -> DepthOutput:
        import torch

        stereo = frame.stereo
        left_depth = torch.where(
            disparity > 0.01,
            (float(stereo.left_intrinsics[0, 0] * stereo.baseline_m) / disparity.double()).float(),
            0,
        )
        aligned = color_aligned_stereo_depth_gpu(
            left_depth,
            stereo.left_intrinsics,
            frame.color_intrinsics,
            stereo.color_from_left,
            frame.sensor_depth_m.shape,
        )
        return DepthOutput(
            _valid_output(aligned, frame.sensor_depth_m.shape), source, frame.sensor_depth_m > 0
        )

    @staticmethod
    def _require_stereo(frame: DepthInput) -> StereoInput:
        stereo = frame.stereo
        if stereo is None:
            raise ValueError("Stereo depth requires synchronized left/right IR images")
        if stereo.left.shape != stereo.right.shape:
            raise ValueError("Stereo images must share a rectified pixel grid")
        return stereo

    @staticmethod
    def _image(image: np.ndarray) -> np.ndarray:
        pad = (-image.shape[1]) % 32
        image = np.pad(image, ((0, 0), (0, pad)), mode="edge")
        rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32) / 255
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None])

    def predict(self, frame: DepthInput) -> DepthOutput:
        stereo = self._require_stereo(frame)
        inputs = {"left_image": self._image(stereo.left), "right_image": self._image(stereo.right)}
        gpu_result = _stereo_disparity_gpu(self.session, inputs)
        if gpu_result is not None:
            return self._aligned_output_gpu(
                frame, gpu_result[0, 0, : stereo.left.shape[0], : stereo.left.shape[1]], self.source
            )
        result = self.session.run(None, inputs)[0]
        disparity = result[0, 0, : stereo.left.shape[0], : stereo.left.shape[1]]
        return self._aligned_output(frame, disparity, self.source)


def _cuda_onnx_session(path: Path):
    import onnxruntime as ort

    ort.preload_dlls(directory="")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("Stereo depth model requires ONNX CUDA execution")
    return session


class _TensorRtSession:
    """The same named-input runner contract as ONNX Runtime, for a fixed engine."""

    def __init__(self, path: Path):
        import tensorrt as trt
        import torch

        self.trt, self.torch = trt, torch
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT could not load {path}; rebuild it for this GPU and TensorRT version"
            )
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        if self.inputs != ["left_image", "right_image"] or self.outputs != ["disparity"]:
            raise ValueError("TensorRT engine does not have Fast-FoundationStereo inputs and output")
        for name in self.inputs:
            if self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
                raise ValueError(f"TensorRT {name} must be FP32 normalized image input")
        output_dtype = self.engine.get_tensor_dtype("disparity")
        if output_dtype not in (trt.DataType.FLOAT, trt.DataType.HALF):
            raise ValueError("TensorRT disparity must be FP32 or FP16")
        self.output_torch_dtype = torch.float32 if output_dtype == trt.DataType.FLOAT else torch.float16

    def run(self, output_names, inputs):
        return [self.run_gpu(output_names, inputs)[0].cpu().numpy()]

    def run_gpu(self, output_names, inputs):
        if output_names not in (None, ["disparity"]):
            raise ValueError("Unsupported TensorRT output request")
        if set(inputs) != set(self.inputs):
            raise ValueError("TensorRT engine needs left_image and right_image")
        torch = self.torch
        with torch.cuda.stream(self.stream):
            gpu_inputs = {}
            for name in self.inputs:
                array = np.ascontiguousarray(inputs[name], dtype=np.float32)
                expected = tuple(self.engine.get_tensor_shape(name))
                if array.shape != expected:
                    raise ValueError(f"TensorRT {name} needs {expected}, got {array.shape}")
                gpu_inputs[name] = torch.from_numpy(array).to("cuda", non_blocking=True)
                self.context.set_tensor_address(name, int(gpu_inputs[name].data_ptr()))
            result = torch.empty(
                tuple(self.engine.get_tensor_shape("disparity")),
                device="cuda",
                dtype=self.output_torch_dtype,
            )
            self.context.set_tensor_address("disparity", int(result.data_ptr()))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT Fast-FoundationStereo inference failed")
            output = result.float()
        self.stream.synchronize()
        return [output]


class LiteAnyStereoV2Depth(FoundationStereoDepth):
    """One official LAS2 S/M/L/H checkpoint on the rectified infrared pair."""

    source = "lite_any_stereo_v2"

    def __init__(self, model_path: str | Path | None = None, *, model_size: str = "h"):
        if model_size not in ("s", "m", "l", "h"):
            raise ValueError(f"Unknown Lite Any Stereo V2 size: {model_size}")
        self.source = "lite_any_stereo_v2" if model_size == "h" else f"lite_any_stereo_v2_{model_size}"
        env_name = "LEROBOT_LAS2_ONNX" if model_size == "h" else f"LEROBOT_LAS2_{model_size.upper()}_ONNX"
        path = Path(
            model_path
            or os.environ.get(
                env_name,
                str(Path.home() / f".cache/flash-lite-any-stereo-v2/las2-{model_size}-480x864.onnx"),
            )
        )
        if not path.is_file():
            raise FileNotFoundError(f"Set {env_name} to a LAS2-{model_size.upper()} 480x864 ONNX checkpoint")
        self.session = _cuda_onnx_session(path)

    @staticmethod
    def _image(image: np.ndarray) -> np.ndarray:
        if image.shape != (480, 848):
            raise ValueError("LAS2-H ONNX export requires a 848x480 infrared pair")
        padded = np.pad(image, ((0, 0), (8, 8)), mode="edge")
        return np.ascontiguousarray(np.repeat(padded[None, None], 3, axis=1), dtype=np.float32)

    def predict(self, frame: DepthInput) -> DepthOutput:
        stereo = self._require_stereo(frame)
        inputs = {"left": self._image(stereo.left), "right": self._image(stereo.right)}
        gpu_result = _stereo_disparity_gpu(self.session, inputs)
        if gpu_result is not None:
            return self._aligned_output_gpu(frame, gpu_result[0, 0, :, 8:-8], self.source)
        result = self.session.run(None, inputs)[0]
        return self._aligned_output(frame, result[0, 0, :, 8:-8], self.source)


def make_depth_source(name: str):
    if name == "sensor":
        return None
    if name == "lingbot":
        return LingBotDepth()
    if name == "foundation_stereo":
        return FoundationStereoDepth()
    if name == "lite_any_stereo_v2":
        return LiteAnyStereoV2Depth()
    if name in ("lite_any_stereo_v2_s", "lite_any_stereo_v2_m", "lite_any_stereo_v2_l"):
        return LiteAnyStereoV2Depth(model_size=name[-1])
    raise ValueError(f"Unknown depth source: {name}")
