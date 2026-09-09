// What a finished mask pass claims about itself has to be true.
//
// The completion toast said "Saved, but nothing was found" whenever ANY camera
// came back with no masks. On a three-camera rig where the object sits outside
// one camera's view that camera is empty on every run, so the message fired
// every time, over a timeline visibly full of masks, advising a re-run that
// could never change the outcome.
//
// Run from pytest via test_mask_coverage_report_js.py, or directly with
// `node tests/gui/mask_coverage_report.test.js`.
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const STATIC = path.join(__dirname, "..", "..", "src", "lerobot", "gui", "static");

function coverageReport() {
  global.window = { addEventListener() {} };
  global.document = {
    readyState: "complete",
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => ({ style: {}, classList: { add() {} }, addEventListener() {}, appendChild() {} }),
    addEventListener() {},
    body: { appendChild() {} },
  };
  new Function(fs.readFileSync(path.join(STATIC, "overlay_stream.js"), "utf8"))();
  return global.window.OverlayStream._coverageReport;
}

const report = coverageReport();

// ---- the case that must stay silent, or every assertion below is vacuous ----
assert.strictEqual(report({ "masks.top": 854, "masks.right_wrist": 829 }), null,
  "a pass where every camera found masks has nothing to report");
assert.strictEqual(report({}), null, "no cameras, nothing to say");
assert.strictEqual(report(null), null, "no coverage, nothing to say");

// ---- the regression: one camera of three cannot see the object -------------
const partial = report({ "masks.left_wrist": 0, "masks.right_wrist": 829, "masks.top": 854 });
assert.ok(partial, "a camera with no masks is still worth saying: it composites as background");
assert.ok(!/nothing was found/i.test(partial.title),
  "REGRESSION: two of three cameras were written; claiming nothing was found is false");
assert.ok(!/seed failure|Re-running/i.test(partial.message),
  "REGRESSION: a camera that cannot see the object is not fixed by running it again");
assert.strictEqual(partial.type, "info", "a normal outcome must not be styled as an error");
// Split on the dash so the two halves are checked for the RIGHT cameras: naming
// both lists is not enough, since swapping them would still name both.
const [lacks, wrote] = partial.message.split("\u2014");
assert.ok(/left_wrist/.test(lacks) && !/right_wrist|top/.test(lacks),
  "the empty camera, and only it, is reported as having no masks");
assert.ok(/right_wrist/.test(wrote) && /top/.test(wrote) && !/left_wrist/.test(wrote),
  "the cameras that were written, and only those, are reported as written");

// ---- the case the toast was written for: a seed failure --------------------
const none = report({ "masks.left_wrist": 0, "masks.right_wrist": 0, "masks.top": 0 });
assert.ok(none, "a pass that found nothing anywhere must still say so");
assert.ok(/nothing was found/i.test(none.title), "the seed-failure wording survives where it is true");
assert.strictEqual(none.type, "error", "nothing found on any camera is a failed outcome");
assert.ok(/seed failure/i.test(none.message), "keeps the advice that actually applies here");

// A single camera selected, and it found nothing, is also "nothing was found".
const solo = report({ "masks.top": 0 });
assert.ok(/nothing was found/i.test(solo.title), "one camera, empty, is every camera empty");

console.log("mask_coverage_report: ok");
