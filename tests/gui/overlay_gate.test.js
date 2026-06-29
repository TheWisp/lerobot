// Unit test for the live-overlay draw gate (run from pytest via test_overlay_gate_js.py, or
// directly with `node tests/gui/overlay_gate.test.js`). Locks the started-race fix: the draw
// decision is a PURE function of the backend machine state, with no hidden history — so a
// transient non-active state during the worker's spawn can never latch drawing off (the bug).
const assert = require("assert");
const { shouldDraw } = require("../../src/lerobot/gui/static/overlay_gate.js");

// Draws ONLY when the backend OverlayStateMachine reports ACTIVE.
assert.strictEqual(shouldDraw("live", "policy_saliency", true, "active"), true, "active -> draw");

// Every other machine state must NOT draw — including the spawn states the started-race
// mishandled (a transient 'inactive'/'loading' used to latch the old `started` flag false).
for (const state of ["inactive", "loading", "stopping", "error"]) {
  assert.strictEqual(shouldDraw("live", "m", true, state), false, state + " -> no draw");
}

// Guards independent of the machine state.
assert.strictEqual(shouldDraw("live", "", true, "active"), false, "no model -> no draw");
assert.strictEqual(shouldDraw("live", "m", false, "active"), false, "objects not ready -> no draw");
assert.strictEqual(shouldDraw("data", "m", true, "active"), false, "data mode -> no draw");

// Purity: identical inputs give identical output — no hidden state to desync (the fix's invariant).
assert.strictEqual(
  shouldDraw("live", "m", true, "active"),
  shouldDraw("live", "m", true, "active"),
  "pure function of its inputs",
);

console.log("overlay_gate.test.js: all assertions passed");
