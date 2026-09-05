// Which run's data the detail pane is allowed to show (run from pytest via
// test_training_run_switch_js.py, or directly with
// `node tests/gui/training_run_switch.test.js`).
//
// The pane is filled by an async fetch of one run's snapshot. That fetch reads
// the run's files through its host — one SSH round trip per file, measured at
// 3.4 s against a rig 226 ms away — so "in flight" is a state the operator
// spends real time in, and can click through several times.
//
// Two ways that goes wrong, both reported from the GUI:
//
//   * the outgoing run stays painted while the new one loads, so the pane
//     shows one run's numbers under another's selection;
//   * a response that arrives after the selection moved on paints anyway,
//     which puts them back in the same state having clicked away from it.
//
// Neither is visible on a local host, where the fetch returns in ~30 ms.

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(path.join(__dirname, "../../src/lerobot/gui/static/training.js"), "utf8");

const noop = () => {};

// Timers are fired by hand: the placeholder is deliberately delayed, so the
// test has to be able to stand at both sides of that delay.
let _timerSeq = 0;
const timers = new Map();
const setTimeoutFake = (fn, ms) => {
  const id = ++_timerSeq;
  timers.set(id, { fn, ms });
  return id;
};
const clearTimeoutFake = (id) => timers.delete(id);
const fireTimers = () => {
  const pending = [...timers.values()];
  timers.clear();
  pending.forEach((t) => t.fn());
};
const makeEl = () => ({
  innerHTML: "",
  style: {},
  scrollTop: 0,
  scrollHeight: 0,
  clientHeight: 0,
  querySelector: () => null,
  querySelectorAll: () => [],
});

const els = {};
const getElementById = (id) => {
  if (!els[id]) els[id] = makeEl();
  return els[id];
};

// Fetches are resolved by hand so a response can be delivered after the
// selection has moved on — the race the guard exists for.
const inFlight = [];
const context = vm.createContext({
  console,
  window: {},
  setTimeout: setTimeoutFake,
  clearTimeout: clearTimeoutFake,
  setInterval: noop,
  clearInterval: noop,
  fetch: (url) => new Promise((resolve) => inFlight.push({ url, resolve })),
  document: {
    addEventListener: noop,
    getElementById,
    querySelector: () => null,
    querySelectorAll: () => [],
  },
  localStorage: { getItem: () => null, setItem: noop },
});
vm.runInContext(source, context);

// Stub the parts that are not under test: what the renderer produces, and the
// sidebar refresh (which would issue fetches of its own).
context.trainingRenderDetailHtml = (snap) => `PANE:${snap.run.run_id}`;
context.trainingDrawDetailCharts = noop;
context.trainingRefreshRuns = noop;
context.trainingHideStateTooltip = noop;

const pane = () => getElementById("training-detail").innerHTML;
const settle = () => new Promise((resolve) => setImmediate(resolve));
const respond = (entry, runId) =>
  entry.resolve({ ok: true, json: async () => ({ run: { run_id: runId }, events: [] }) });

async function main() {
  // ── A wait long enough to mislead clears the outgoing run ────────────────
  {
    context.trainingSelectRun("run-a");
    respond(inFlight.shift(), "run-a");
    await settle();
    assert.strictEqual(pane(), "PANE:run-a", "the selected run's snapshot should be painted");

    context.trainingSelectRun("run-b");
    // Nothing has answered for run-b yet: this is the whole 3.4 s window.
    fireTimers();
    assert.ok(!pane().includes("run-a"), "the outgoing run must not survive the wait");
    assert.ok(pane().includes("run-b"), "the pane must name the run it is loading");
  }

  // ── A host that answers in time never blanks the pane ────────────────────
  //
  // The workstation returns the same snapshot in ~29 ms. Clearing there buys
  // nothing and costs a blink plus a collapse of the pane's height, on every
  // switch, for the host most runs use.
  {
    inFlight.length = 0;
    context.trainingSelectRun("run-quick");
    respond(inFlight.shift(), "run-quick");
    await settle();
    assert.strictEqual(pane(), "PANE:run-quick", "a fast answer paints straight through");

    fireTimers(); // the placeholder was scheduled; it must have been cancelled
    assert.strictEqual(pane(), "PANE:run-quick", "a cancelled placeholder must not land");
  }

  // ── A late response for a run no longer selected must not paint ──────────
  {
    inFlight.length = 0; // drop the previous block's unanswered fetch
    context.trainingSelectRun("run-slow");
    const slow = inFlight.shift();
    context.trainingSelectRun("run-fast");
    const fast = inFlight.shift();

    respond(fast, "run-fast");
    await settle();
    assert.strictEqual(pane(), "PANE:run-fast", "the current selection should paint");

    respond(slow, "run-slow"); // arrives late, from the abandoned selection
    await settle();
    assert.strictEqual(pane(), "PANE:run-fast", "a superseded response must not paint");
  }

  // ── The same applies to a failure: it belongs to its own run ─────────────
  {
    inFlight.length = 0;
    context.trainingSelectRun("run-doomed");
    const doomed = inFlight.shift();
    context.trainingSelectRun("run-live");
    const live = inFlight.shift();

    respond(live, "run-live");
    await settle();

    doomed.resolve({ ok: false, status: 500 });
    await settle();
    assert.strictEqual(pane(), "PANE:run-live", "a superseded error must not replace the pane");
  }

  console.log("training_run_switch: all assertions passed");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
