// Unit test for the leave-tab camera release decision (run from pytest via
// test_camera_release_js.py, or directly with `node tests/gui/camera_release.test.js`).
//
// Locks the fix for: leaving the Robot tab called stopCameraPreview(), which stops the
// polling and hides the Stop Preview button but left the backend holding a V4L2 /
// librealsense handle per camera. Nothing on screen said so, and a run launched from
// another tab then competed with the GUI for the same /dev/video* devices — reproduced
// on hardware as a teleop-style open failing with ConnectionError while a preview was held.
const assert = require("assert");
const { shouldReleaseCameras } = require("../../src/lerobot/gui/static/camera_release.js");

// Leaving the robot tab releases, wherever you go.
for (const next of ["run", "data", "model"]) {
  assert.strictEqual(shouldReleaseCameras("robot", next), true, "robot -> " + next + " releases");
}

// Staying on the robot tab must not release — the user is still looking at the previews.
assert.strictEqual(shouldReleaseCameras("robot", "robot"), false, "robot -> robot keeps previews");

// Switches that never touched the robot tab must not POST.
assert.strictEqual(shouldReleaseCameras("data", "run"), false, "data -> run: no POST");
assert.strictEqual(shouldReleaseCameras("run", "data"), false, "run -> data: no POST");
assert.strictEqual(shouldReleaseCameras(null, "run"), false, "first render: no POST");

// The decision must NOT depend on frontend camera state. An earlier version gated on
// `detectedCameras.length`, which an end-to-end check disproved: cameras opened by any
// path other than the frontend's own detectCameras() left that array empty while the
// backend held five devices, so the release silently never fired. Only the tab
// transition is an input here — the backend knows what it holds.
assert.strictEqual(shouldReleaseCameras.length, 2, "takes exactly (previousTab, nextTab)");

// Purity: identical inputs give identical output, so it cannot desync from UI state.
assert.strictEqual(
  shouldReleaseCameras("robot", "run"),
  shouldReleaseCameras("robot", "run"),
  "pure"
);

console.log("camera_release: all assertions passed");
