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

const {
  trainingGeneralizationCardHtml,
  trainingGeneralizationSeries,
  trainingLatestMetricValue,
  trainingMetricsCardHtml,
} = context;
assert.strictEqual(typeof trainingMetricsCardHtml, "function", "card renderer must be reachable");
assert.strictEqual(
  typeof trainingGeneralizationCardHtml,
  "function",
  "generalization card renderer must be reachable",
);
assert.strictEqual(
  typeof trainingLatestMetricValue,
  "function",
  "latest-value lookup must be reachable",
);

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

// ── Sparse generalization evaluations stay separate from dense metrics ────
{
  const dense = [
    { step: 100, loss: 0.5 },
    {
      step: 200,
      loss: 0.4,
      generation_train_ratio: 0.31,
      generation_held_out_ratio: 0.57,
      generation_ratio_gap: 0.26,
    },
    { step: 300, loss: 0.3 },
  ];
  const evaluations = trainingGeneralizationSeries(dense);
  assert.strictEqual(evaluations.length, 1, "only complete evaluation samples belong in the curve");
  assert.strictEqual(evaluations[0].step, 200, "the evaluation keeps its actual training step");
}

{
  const partial = [
    { step: 100, samples_per_s: 19.2, mem_gb: 7.4 },
    { step: 150, generation_train_ratio: 0.31, generation_held_out_ratio: 0.57 },
  ];
  assert.strictEqual(
    trainingLatestMetricValue(partial, "samples_per_s"),
    19.2,
    "an eval-only record must not blank the latest throughput summary",
  );
  assert.strictEqual(
    trainingLatestMetricValue(partial, "mem_gb"),
    7.4,
    "an eval-only record must not blank the latest memory summary",
  );
}

{
  assert.strictEqual(
    trainingGeneralizationCardHtml([{ step: 100, loss: 0.5 }]),
    "",
    "old and non-HVLA runs must keep their existing layout",
  );
}

{
  const html = trainingGeneralizationCardHtml([
    {
      step: 2000,
      generation_train_ratio: 0.316,
      generation_held_out_ratio: 0.569,
      generation_ratio_gap: 0.253,
    },
  ]);
  assert.ok(html.includes("Train ratio"), "one evaluation shows the latest summary");
  assert.ok(html.includes("+0.253"), "the overfitting gap is signed");
  assert.ok(html.includes("Evaluation history"), "the exact evaluation is available in the table");
  assert.ok(!html.includes("<details class=\"training-generalization-history\" open"), "history is collapsed by default");
  assert.ok(!html.includes('id="training-generalization-chart"'), "one point must not create a blank chart");
}

{
  const html = trainingGeneralizationCardHtml([
    { step: 2000, generation_train_ratio: 0.316, generation_held_out_ratio: 0.569 },
    { step: 4000, generation_train_ratio: 0.206, generation_held_out_ratio: 0.507 },
  ]);
  assert.ok(html.includes('id="training-generalization-chart"'), "two evaluations render a trend chart");
  assert.ok(html.indexOf(">4000<") < html.indexOf(">2000<"), "history lists the latest evaluation first");
}

console.log("training_chart_gaps.test.js: all assertions passed");
