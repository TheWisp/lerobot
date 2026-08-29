// The worker-count box is frozen when the GPU image pipeline is chosen
// (run from pytest via test_worker_lock_js.py, or directly with
// `node tests/gui/worker_lock.test.js`).
//
// On that pipeline the loader is pinned to one worker: the workers no longer
// decode video, and one produces batches far faster than training consumes them
// — 1246/s at batch 4 against 4.75 consumed. A box that still accepts a number
// the run will ignore is worse than no box, so it is disabled and relabelled.
//
// Only `gpu` locks it. Under `auto` the path is not decided until the run probes
// the dataset, and greying a field out on a guess would misreport what happens.

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
  CSS: { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => "\\" + c) },
});
vm.runInContext(source, context);

const { trainingBindWorkerLock } = context;
assert.strictEqual(typeof trainingBindWorkerLock, "function", "the binder must be reachable");

const ORIGINAL_HINT = "Parallel data loading; affects input throughput.";

function harness(pipelineValue) {
  const hint = { textContent: ORIGINAL_HINT };
  const workers = {
    id: "training-arg-num_workers",
    value: "4",
    disabled: false,
    closest: () => ({ querySelector: () => hint }),
  };
  const listeners = [];
  const pipeline = {
    id: "training-arg-data_path",
    value: pipelineValue,
    addEventListener: (_evt, fn) => listeners.push(fn),
  };
  const container = {
    querySelector: (sel) => (sel.includes("data_path") ? pipeline : sel.includes("num_workers") ? workers : null),
  };
  return { container, pipeline, workers, hint, fire: () => listeners.forEach((f) => f()) };
}

// ── gpu locks the box, pins the value, and explains itself ──────────────────
{
  const h = harness("gpu");
  trainingBindWorkerLock(h.container);
  assert.strictEqual(h.workers.disabled, true, "gpu must disable the worker box");
  assert.strictEqual(h.workers.value, "1", "gpu must pin the value the run will use");
  assert.notStrictEqual(h.hint.textContent, ORIGINAL_HINT, "the hint must say why it is fixed");
}

// ── auto leaves it alone: the path is not known yet ─────────────────────────
{
  const h = harness("auto");
  trainingBindWorkerLock(h.container);
  assert.strictEqual(h.workers.disabled, false, "auto must not lock the box");
  assert.strictEqual(h.workers.value, "4", "auto must not overwrite the chosen count");
  assert.strictEqual(h.hint.textContent, ORIGINAL_HINT, "auto must keep the original hint");
}

// ── cpu likewise ────────────────────────────────────────────────────────────
{
  const h = harness("cpu");
  trainingBindWorkerLock(h.container);
  assert.strictEqual(h.workers.disabled, false, "cpu must not lock the box");
}

// ── and it reacts to a change, not only to the initial render ───────────────
{
  const h = harness("cpu");
  trainingBindWorkerLock(h.container);
  h.pipeline.value = "gpu";
  h.fire();
  assert.strictEqual(h.workers.disabled, true, "switching to gpu must lock it");
  h.pipeline.value = "cpu";
  h.fire();
  assert.strictEqual(h.workers.disabled, false, "switching back must release it");
  assert.strictEqual(h.hint.textContent, ORIGINAL_HINT, "the original hint must come back");
}

console.log("worker_lock.test.js: all assertions passed");
