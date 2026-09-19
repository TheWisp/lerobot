// Regression tests for the two defects that desynced the SAM3 preview from
// playback. Each fails if its fix is reverted.
//
// Run from pytest via test_live_layer_arbitration_js.py, or directly with
// `node tests/gui/live_layer_arbitration.test.js`.
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const STATIC = path.join(__dirname, "..", "..", "src", "lerobot", "gui", "static");
const load = (name) => fs.readFileSync(path.join(STATIC, name), "utf8");

// ---- 1. "the live layer owns the tiles" must include the starting window ----
//
// masks.js refreshes stills when it leaves composited mode, and loadAllFrames
// treats any call as a scrub and STOPS the stream. Guarding on `streaming`
// alone misses the window where the worker is active but the stream has not
// started -- exactly when enabling SAM3 flips that mode -- so enabling SAM3
// killed the stream it had just started.
function liveActiveWith({ streaming, badgeClass, badgeText }) {
  const badge = badgeClass === null ? null : { className: badgeClass, textContent: badgeText || "" };
  global.window = { OverlayStream: { streaming } };
  global.document = { getElementById: (id) => (id === "overlays-badge" ? badge : null) };
  new Function(load("masks.js"))();
  return global.window.MaskOverlay.liveLayerActive();
}

assert.strictEqual(liveActiveWith({ streaming: true, badgeClass: null }), true, "streaming -> live");
assert.strictEqual(liveActiveWith({ streaming: false, badgeClass: "badge loading" }), true,
  "REGRESSION: a worker that is still loading owns the tiles; stills here stop the stream it is starting");
assert.strictEqual(liveActiveWith({ streaming: false, badgeClass: "badge ok" }), true, "active worker -> live");
assert.strictEqual(liveActiveWith({ streaming: false, badgeClass: "badge idle", badgeText: "0 fps" }), true,
  "idle worker still owns the tiles");
assert.strictEqual(liveActiveWith({ streaming: false, badgeClass: "badge idle", badgeText: "busy: other tab" }), false,
  "held by another client -> not ours");
assert.strictEqual(liveActiveWith({ streaming: false, badgeClass: null }), false, "no worker -> the app owns the tiles");

// ---- 2. the stream must move the APP's transport state, not just the label ----
//
// togglePlay() delegates to the stream and returns before touching isPlaying,
// so a stream that only rewrites the button label left the app believing it was
// paused; any re-render from isPlaying then offered "Play" over moving video.
let labelWrites = 0;
const btn = { set innerHTML(v) { labelWrites += 1; }, get innerHTML() { return ""; }, textContent: "", addEventListener() {} };
const calls = [];
global.window = {
  __streamSetPlaying: (p) => calls.push(p),
  addEventListener() {},
};
global.document = {
  readyState: "complete",
  getElementById: (id) => (id === "play-btn" ? btn : null),
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => ({ style: {}, classList: { add() {} }, addEventListener() {}, appendChild() {} }),
  addEventListener() {},
  body: { appendChild() {} },
};
new Function(load("overlay_stream.js"))();
global.window.OverlayStream._setPlayBtn(true);
assert.deepStrictEqual(calls, [true],
  "REGRESSION: the stream must set the app's isPlaying, not only the button label");
assert.strictEqual(labelWrites, 0, "with the app hook present the stream must not write the label itself");

console.log("live_layer_arbitration.test.js: ok");
