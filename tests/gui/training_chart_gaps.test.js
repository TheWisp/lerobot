// Unit tests for the "is this metric present?" decision in training.js
// (run from pytest via test_training_chart_gaps_js.py, or directly with
// `node tests/gui/training_chart_gaps.test.js`).
//
// The decision is one expression — Number.isFinite(Number(sample[key])) — used
// both to choose between a canvas and the "Not logged by this run" placeholder,
// and to punch gaps in a drawn line. It is load-bearing for the resource
// telemetry in DESIGN.md, whose central requirement is that an unreadable
// source reads as *absent*, never as a genuine 0%: a chart claiming the GPU sat
// idle is worse than a chart admitting it does not know.
//
// The hazard is JS coercion, not logic. Number(undefined) is NaN, but
// Number(null), Number(""), Number([]) and Number(false) are all a finite 0 —
// so the wrong flavour of "missing" silently becomes a real-looking data point.

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "../../src/lerobot/gui/static/training.js"),
  "utf8",
);

const noop = () => {};
const context = vm.createContext({
  console,
  window: {},
  setTimeout: noop,
  setInterval: noop,
  fetch: noop,
  document: {
    addEventListener: noop,
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
  },
  localStorage: { getItem: () => null, setItem: noop },
});
vm.runInContext(source, context);

const { trainingMetricsCardHtml } = context;
assert.strictEqual(typeof trainingMetricsCardHtml, "function", "card renderer must be reachable");

// A chart is "present" iff its canvas is rendered; otherwise the card says so.
const charted = (html, key) => html.includes(`id="training-chart-${key}"`);
const PLACEHOLDER = "Not logged by this run";

// ── Baseline: a real metric charts, an absent one is declared absent ────────
{
  const html = trainingMetricsCardHtml([{ step: 1, loss: 0.5 }], true);
  assert.ok(charted(html, "loss"), "a logged metric must chart");
  assert.ok(!charted(html, "grdn"), "an unlogged metric must not chart");
  assert.ok(html.includes(PLACEHOLDER), "an unlogged metric must say so");
}

// ── The contract: every flavour of missing must read as missing ─────────────
//
// undefined is the only one JS already gets right. The rest coerce to a finite
// 0 and would render a chart asserting the value really was zero.
for (const [name, missing] of [
  ["undefined", undefined],
  ["null", null],
  ["empty string", ""],
  ["false", false],
  ["empty array", []],
]) {
  const html = trainingMetricsCardHtml([{ step: 1, loss: 0.5, grdn: missing }], true);
  assert.ok(
    !charted(html, "grdn"),
    `${name} must read as missing, not as a genuine 0 (Number(${name}) is ${Number(missing)})`,
  );
}

// ── ...and a real zero must still chart ────────────────────────────────────
// The mirror of the above: suppressing 0 to dodge the coercion trap would hide
// a genuinely idle GPU, which is a real and interesting reading.
{
  const html = trainingMetricsCardHtml([{ step: 1, loss: 0.5, grdn: 0 }], true);
  assert.ok(charted(html, "grdn"), "a measured 0 is data and must chart");
}

// A single real sample among missing ones is enough to chart the line.
{
  const html = trainingMetricsCardHtml(
    [
      { step: 1, loss: 0.5, grdn: null },
      { step: 2, loss: 0.4, grdn: 1.25 },
    ],
    true,
  );
  assert.ok(charted(html, "grdn"), "one real reading is enough to chart");
}

console.log("training_chart_gaps.test.js: all assertions passed");
