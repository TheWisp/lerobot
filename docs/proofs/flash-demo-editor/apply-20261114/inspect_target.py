"""Compare IK objectives on the captured target without any motor access.

The unbounded arm is diagnostic only and must never produce robot commands.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from scipy.optimize import least_squares
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
selected = m.kinematics
mapped = selected.inner
tip = mapped._inner
pink = tip._inner
T = np.array(e["target"])
q = np.array(
    [
        2.6890405248288296,
        37.84542941176379,
        -44.25780562863578,
        5.788317476455433,
        -93.31295174073,
        27.94418327462453,
    ]
)
recorded = m.degrees(p["keyframes"][131]["state"])


def error(q):
    f = m.kinematics.forward_kinematics(q)
    return np.r_[
        ((f[:3, 3] - T[:3, 3]) / 0.003),
        Rotation.from_matrix(T[:3, :3].T @ f[:3, :3]).as_rotvec() / np.deg2rad(3),
    ]


def summarize(q):
    er = error(q)
    return {
        "q_deg": q.tolist(),
        "position_mm": float(np.linalg.norm(er[:3]) * 3),
        "rotation_deg": float(np.linalg.norm(er[3:]) * 3),
        "pass": bool(np.linalg.norm(er[:3]) <= 1 and np.linalg.norm(er[3:]) <= 1),
    }


results = {
    "exact_target": T.tolist(),
    "demo_target": p["keyframes"][131]["demo_tool"],
    "stalled": summarize(q),
    "pink": [],
    "independent": [],
}
for posture, iters, pw, ow in [
    (0.05, 50, 1, 1),
    (0.05, 500, 1, 1),
    (0, 500, 1, 1),
    (0.05, 200, 1, 0.003 / np.deg2rad(3)),
    (0, 200, 1, 0.003 / np.deg2rad(3)),
]:
    pink.posture_cost = float(posture)
    pink.max_iters = iters
    full = selected._full(q)
    urdf = mapped._motor_to_urdf(full)
    solved = pink.inverse_kinematics(
        urdf, T @ tip._tip_offset_inv, position_weight=float(pw), orientation_weight=float(ow)
    )
    out = mapped._urdf_to_motor(solved)[selected._controlled_indices]
    results["pink"].append(
        {
            "posture": posture,
            "max_iters": iters,
            "iters": pink.last_n_iters,
            "pw": pw,
            "ow": ow,
            **summarize(out),
        }
    )
lo = np.array(m.lower_deg) + 1e-6
hi = np.array(m.upper_deg) - 1e-6
rng = np.random.default_rng(413)
for name, seed in [("stalled", q), ("recorded", recorded)] + [
    (f"random{i}", rng.uniform(lo, hi)) for i in range(12)
]:
    fit = least_squares(
        error, np.clip(seed, lo, hi), bounds=(lo, hi), max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10
    )
    results["independent"].append({"seed": name, "nfev": fit.nfev, **summarize(fit.x)})
# Diagnostic only: required joints if calibrated working bounds are ignored.
a = mapped._urdf_to_motor(np.rad2deg(pink._fk_q_lo))
b = mapped._urdf_to_motor(np.rad2deg(pink._fk_q_hi))
ulo = np.minimum(a, b)[selected._controlled_indices] + 0.01
uhi = np.maximum(a, b)[selected._controlled_indices] - 0.01
fit = least_squares(
    error, np.clip(q, ulo, uhi), bounds=(ulo, uhi), max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10
)
results["ignoring_working_limits_DIAGNOSTIC_ONLY"] = summarize(fit.x)
Ad = np.array(p["keyframes"][131]["demo_base_from_reference"])
Al = np.array(e["evidence"]["base_from_reference"])
delta = Al @ np.linalg.inv(Ad)
results["transfer"] = {
    "rotation_deg": float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())),
    "tool_position_change_mm": (
        (T[:3, 3] - np.array(p["keyframes"][131]["demo_tool"])[:3, 3]) * 1000
    ).tolist(),
    "composition_error": float(np.max(np.abs(T - Al @ np.array(p["keyframes"][131]["reference_from_tool"])))),
}
(D / "target-inspection.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
r.connect.assert_not_called()
r.send_action.assert_not_called()
