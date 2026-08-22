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

// ── Resource tiles: the three states, never two ────────────────────────────
//
// stat 0 measured · 1 not present on this machine · 2 present but unreadable.
// Collapsing 1 and 2 together is the failure that matters: a broken sampler
// would render identically to a machine that simply has no GPU.

const sample = (extra) => [{ step: 1, loss: 0.5, ...extra }];

{
  // Measured: the tile charts.
  const html = trainingMetricsCardHtml(
    sample({ cpu: 71.2, cpu_max: 88, pcpu: 68, rq: 4, cores: 32, cpu_stat: 0 }),
    true,
  );
  assert.ok(charted(html, "cpu-host"), "host CPU charts when measured");
  assert.ok(charted(html, "cpu-run"), "run CPU is its own tile");
  assert.ok(charted(html, "runqueue"), "run queue is its own tile");
}

{
  // Not present: no tile at all, rather than an empty one implying a problem.
  const html = trainingMetricsCardHtml(sample({ cpu_stat: 1 }), true);
  assert.ok(!charted(html, "cpu-host"), "an absent resource draws no chart");
  assert.ok(!html.includes("CPU — whole machine"), "an absent resource renders no tile");
}

{
  // Present but unreadable: the tile appears and says so.
  const html = trainingMetricsCardHtml(sample({ cpu_stat: 2 }), true);
  assert.ok(html.includes("CPU — whole machine"), "an unreadable resource still renders a tile");
  assert.ok(!charted(html, "cpu-host"), "...but no chart");
  assert.ok(
    html.includes("CPU counters could not be read"),
    "...and it says why, rather than reading as idle",
  );
}

{
  // A run from before telemetry existed reports no status at all. It must fall
  // back to the ordinary not-logged path, not to an error.
  const html = trainingMetricsCardHtml(sample({}), true);
  assert.ok(html.includes("CPU — whole machine"), "tile still listed for an old run");
  assert.ok(!html.includes("could not be read"), "a silent run is not a broken one");
  assert.ok(html.includes("Not logged by this run"), "it reads as simply unlogged");
}

// ── GPU tiles are derived from the data, one group per device ──────────────
{
  const html = trainingMetricsCardHtml(
    sample({ g0sm: 75, g0busy: 95, g0pw: 351, g0mem: 4e9, g0_stat: 0 }),
    true,
  );
  assert.ok(charted(html, "gpu0-occupancy"), "occupancy tile");
  assert.ok(charted(html, "gpu0-power"), "power is its own tile, not mixed into percent");
  assert.ok(charted(html, "gpu0-memory"), "memory is its own tile, in its own unit");
  assert.ok(!charted(html, "gpu1-occupancy"), "no tile for a device the run never saw");
}

{
  // Two devices give two groups, and one failing does not suppress the other.
  const html = trainingMetricsCardHtml(
    sample({ g0busy: 95, g0_stat: 0, g1_stat: 2 }),
    true,
  );
  assert.ok(charted(html, "gpu0-occupancy"), "healthy device still charts");
  assert.ok(
    html.includes("GPU 1 counters could not be read"),
    "the failing device reports its own failure",
  );
}

console.log("training_chart_gaps.test.js: all assertions passed");
