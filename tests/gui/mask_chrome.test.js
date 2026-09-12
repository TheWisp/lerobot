// The mask layer's chrome -- outline and name, hidden labels -- is one decision
// for both picture paths (design: R6). Driven in node on the layer itself.
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const STATIC = path.join(__dirname, "..", "..", "src", "lerobot", "gui", "static");

global.window = { addEventListener() {}, datasets: {}, currentDataset: null, currentEpisode: null };
global.document = {
  readyState: "complete", getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
  addEventListener() {}, createElement: () => ({ style: {}, getContext: () => null }), body: { appendChild() {} },
};
global.fetch = () => new Promise(() => {});
new Function(fs.readFileSync(path.join(STATIC, "masks.js"), "utf8"))();
const M = global.window.MaskOverlay;
assert.ok(M.chromeOptions, "the layer exports its chrome decision");

// Saved masks on the camera: outline and names. None: a fill, no names.
let o = M.chromeOptions({ hasAny: true, labels: ["ball"], scaleTo: [240, 480] });
assert.strictEqual(o.outline, true);
assert.deepStrictEqual(o.labels, ["ball"]);
assert.deepStrictEqual(o.scaleTo, [240, 480]);
o = M.chromeOptions({ hasAny: false, labels: [], scaleTo: null });
assert.strictEqual(o.outline, false);

// The hidden set is the layer's own, live: hiding a label shows in the next decision.
M.setLabelHidden(2, true);
assert.ok(M.chromeOptions({ hasAny: true, labels: [] }).hidden.has(2));
M.setLabelHidden(2, false);
assert.ok(!M.chromeOptions({ hasAny: true, labels: [] }).hidden.has(2));

// Outlines can be switched off for the layer as a whole; the decision follows.
M.setOutlines(false);
assert.strictEqual(M.chromeOptions({ hasAny: true, labels: [] }).outline, false, "outlines off: a fill even with saved masks");
M.setOutlines(true);
assert.strictEqual(M.chromeOptions({ hasAny: true, labels: [] }).outline, true);

console.log("mask_chrome: ok");
