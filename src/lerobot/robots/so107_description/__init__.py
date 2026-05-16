"""SO-107 robot description (URDF + meshes) for kinematics and visualization."""

from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def get_urdf_path() -> Path:
    """Absolute path to the SO-107 URDF file."""
    p = _PKG_DIR / "urdf" / "SO107.urdf"
    assert p.exists(), f"SO-107 URDF missing at {p}"
    return p


def get_meshes_dir() -> Path:
    """Absolute path to the directory containing the SO-107 link meshes."""
    p = _PKG_DIR / "meshes"
    assert p.is_dir(), f"SO-107 meshes dir missing at {p}"
    return p


__all__ = ["get_urdf_path", "get_meshes_dir"]
