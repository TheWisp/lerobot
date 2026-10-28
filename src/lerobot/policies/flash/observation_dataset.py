"""Select aligned RGB and depth features from an observation-only dataset."""

from __future__ import annotations


def camera_feature_keys(features: dict, source: dict) -> tuple[str, str]:
    videos = {
        key: value
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    }
    depth = [key for key, feature in videos.items() if feature.get("info", {}).get("is_depth_map")]
    rgb = [key for key in videos if key not in depth]
    requested_rgb = source.get("rgb_feature")
    requested_depth = source.get("depth_feature")
    if requested_rgb is not None and requested_rgb not in rgb:
        raise ValueError("recorded RGB feature does not match dataset metadata")
    if requested_depth is not None and requested_depth not in depth:
        raise ValueError("recorded depth feature does not match dataset metadata")
    if requested_rgb is None and len(rgb) != 1:
        raise ValueError("choose one RGB video feature in the observation source metadata")
    if requested_depth is None and len(depth) != 1:
        raise ValueError("choose one aligned depth feature in the observation source metadata")
    rgb_key, depth_key = requested_rgb or rgb[0], requested_depth or depth[0]
    if tuple(features[rgb_key]["shape"][:2]) != tuple(features[depth_key]["shape"][:2]):
        raise ValueError("RGB and depth features have different image sizes")
    return rgb_key, depth_key
