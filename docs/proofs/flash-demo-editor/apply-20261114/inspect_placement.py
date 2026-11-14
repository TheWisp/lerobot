"""Offline bounded endpoint diagnostics, not a path or hardware guarantee."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from lerobot.gui.api.robot import _make_robot_from_profile

D = Path("docs/proofs/flash-demo-editor/apply-20261114")
p = json.loads((D / "compiled-plan.json").read_text())
e = json.loads((D / "live-execution.json").read_text())
r = _make_robot_from_profile(
    json.loads(Path("/home/feit/.cache/flash-demo-editor-preview/config/robots/white_left.json").read_text())
)
r.connect = Mock(side_effect=AssertionError("No motors"))
r.send_action = Mock(side_effect=AssertionError("No motors"))
m = r.cartesian_model()
a = np.array(e["evidence"]["base_from_reference"])
lo = np.array(m.lower_deg) + 0.006
hi = np.array(m.upper_deg) - 0.006
out = []
for frame in [570, 577, 590, 610, 650, 700, 740, 780]:
    k = next(k for k in p["keyframes"] if k["frame"] == frame)
    T = a @ np.array(k["reference_from_tool"])

    def residual(q, target=T):
        f = m.kinematics.forward_kinematics(q)
        return np.r_[
            (f[:3, 3] - target[:3, 3]) / 0.003,
            Rotation.from_matrix(target[:3, :3].T @ f[:3, :3]).as_rotvec() / np.deg2rad(3),
        ]

    def norms(q):
        e = residual(q)
        return np.array([np.linalg.norm(e[:3]), np.linalg.norm(e[3:])])

    q = m.degrees(k["state"])
    ls = least_squares(
        residual, np.clip(q, lo, hi), bounds=(lo, hi), max_nfev=250, ftol=1e-11, xtol=1e-11, gtol=1e-11
    )
    # Smallest maximum normalized pose error, independent of Pink's soft sum.
    initial = np.r_[ls.x, max(norms(ls.x))]
    s = minimize(
        lambda x: x[-1],
        initial,
        method="SLSQP",
        bounds=list(zip(lo, hi, strict=True)) + [(0, None)],
        constraints={"type": "ineq", "fun": lambda x: x[-1] - norms(x[:-1])},
        options={"ftol": 1e-9, "maxiter": 300},
    )
    n = norms(s.x[:-1])
    row = {
        "frame": frame,
        "position_mm": float(3 * n[0]),
        "angle_deg": float(3 * n[1]),
        "within_existing_tolerances": bool(max(n) <= 1),
        "solver_success": bool(s.success),
        "q_deg": s.x[:-1].tolist(),
    }
    out.append(row)
    print(row, flush=True)
(D / "placement-feasibility.json").write_text(json.dumps(out, indent=2))
r.connect.assert_not_called()
r.send_action.assert_not_called()
