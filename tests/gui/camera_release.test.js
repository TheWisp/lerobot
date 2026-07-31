// Unit test for the leave-tab camera release decision (run from pytest via
// test_camera_release_js.py, or directly with `node tests/gui/camera_release.test.js`).
//
// Locks the fix for: leaving the Robot tab called stopCameraPreview(), which stops the
// polling and hides the Stop Preview button but leaves the backend holding a V4L2 /
// librealsense handle per camera. Nothing on screen said so, and a run launched from
// another tab then competed with the GUI for the same /dev/video* devices — reproduced
// on hardware as a teleop-style open failing with ConnectionError while a preview was held.
const assert = require("assert");
const { shouldReleaseCameras } = require("../../src/lerobot/gui/static/camera_release.js");

// Leaving the robot tab with previews open must release them.
for (const tab of ["run", "data", "model"]) {
  assert.strictEqual(shouldReleaseCameras(tab, 4), true, tab + " with 4 held -> release");
}

// Staying on the robot tab must NOT release — the user is still looking at the previews.
assert.strictEqual(shouldReleaseCameras("robot", 4), false, "robot tab -> keep previews");

// Nothing held -> no release, so an ordinary tab switch does not POST on every click.
for (const held of [0, undefined, null]) {
  assert.strictEqual(shouldReleaseCameras("run", held), false, String(held) + " held -> no POST");
}

// Purity: the decision depends only on its inputs, so it cannot desync from UI state.
assert.strictEqual(shouldReleaseCameras("run", 2), shouldReleaseCameras("run", 2), "pure");

console.log("camera_release: all assertions passed");
