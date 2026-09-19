// Unit tests for the flag picker's cost annotation in training.js
// (run from pytest via test_flag_cost_js.py, or directly with
// `node tests/gui/flag_cost.test.js`).
//
// The figure is a share of the supervision a chunk length defines, so the chunk
// length it was computed with has to be the one the form will train with. Every
// policy offering this picker names that field differently and three offer none,
// and the server's own default is 50 -- which is the default of the one policy
// this started from, so a dropped parameter looks exactly like a correct answer.
//
// The second hazard is that the annotation is asynchronous over a box reused
// across renders: a fetch started for one dataset can land after the picker has
// been rebuilt for another, and appendChild does not care.
//
// Phases run in sequence and never concurrently. They share one vm context and
// each installs its own fetch double, so overlapping them would let one phase's
// double answer another's call -- which, the first time this was written, left
// two assertions permanently pending and passing vacuously.

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
    createElement: () => ({ className: "", textContent: "" }),
  },
  localStorage: { getItem: () => null, setItem: noop },
  URLSearchParams,
  WeakMap,
  WeakSet,
  CSS: { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => "\\" + c) },
});
vm.runInContext(source, context);

const { trainingActionWindow, annotateFlagCost } = context;
for (const [name, fn] of Object.entries({ trainingActionWindow, annotateFlagCost })) {
  assert.strictEqual(typeof fn, "function", `${name} must be reachable`);
}

// A form exposing exactly the named inputs, matched on the `name$=` selectors
// the production code uses.
function formWith(inputs) {
  return {
    querySelector(selector) {
      const m = /input\[name\$="([^"]+)"\]/.exec(selector);
      if (!m) return null;
      const hit = Object.keys(inputs).find((n) => n.endsWith(m[1]));
      return hit === undefined ? null : { name: hit, value: String(inputs[hit]) };
    },
  };
}
const useForm = (inputs) => {
  context.document.getElementById = () => formWith(inputs);
};

function theWindowComesFromTheForm() {
  // HVLA's prefix is empty; every draccus policy prefixes with `policy.`.
  assert.strictEqual(trainingActionWindow(formWith({ chunk_size: 50 })), 50);
  assert.strictEqual(trainingActionWindow(formWith({ "policy.chunk_size": 100 })), 100);
  // The diffusion family names it `horizon`.
  assert.strictEqual(trainingActionWindow(formWith({ "policy.horizon": 16 })), 16);
  // Whatever the operator typed, not what the policy defaults to: an edited
  // value is exactly the case a stale figure would misreport.
  assert.strictEqual(trainingActionWindow(formWith({ "policy.chunk_size": 7 })), 7);

  // n_action_steps is how many of the window's actions get executed, not how
  // many are supervised. Pricing against it would divide by the wrong total.
  assert.strictEqual(
    trainingActionWindow(formWith({ "policy.n_action_steps": 25 })),
    null,
    "n_action_steps must not stand in for the supervised window",
  );

  // gaussian_actor, lingbot_va and vqbet offer the picker and no horizon field.
  for (const absent of [{}, { "policy.chunk_size": "" }, { "policy.chunk_size": 0 }, { "policy.chunk_size": "abc" }]) {
    assert.strictEqual(trainingActionWindow(formWith(absent)), null, JSON.stringify(absent));
  }
}

async function theWindowReachesTheRequest() {
  const asked = [];
  context.fetch = (url) => {
    asked.push(url);
    return Promise.resolve({ ok: true, json: async () => ({ total_positions: 0, labels: [] }) });
  };
  const box = { querySelectorAll: () => [] };

  useForm({ "policy.horizon": 16 });
  await annotateFlagCost(box, { root: "/ds" });
  assert.strictEqual(asked.length, 1, "a form with a window should be priced");
  assert.ok(
    asked[0].includes("chunk_size=16"),
    `the request must carry the form's window, got ${asked[0]}`,
  );

  asked.length = 0;
  useForm({ "policy.n_action_steps": 25 });
  await annotateFlagCost(box, { root: "/ds" });
  assert.strictEqual(
    asked.length,
    0,
    "with no window in the form there is no denominator, so nothing may be fetched",
  );
}

async function anOvertakenFetchWritesNothing() {
  const label = {
    appended: [],
    querySelector: () => ({ value: "blurry" }),
    appendChild(node) {
      this.appended.push(node);
    },
  };
  const box = { querySelectorAll: () => [label] };

  // Held open so the two calls can be resolved in the opposite order -- the
  // interleaving a second dataset click produces. The root's length stands in
  // for "which dataset", so the figure written says which fetch produced it.
  const pending = [];
  context.fetch = (url) => {
    const frames = new URLSearchParams(url.split("?")[1]).get("root").length;
    return new Promise((resolve) =>
      pending.push(() =>
        resolve({
          ok: true,
          json: async () => ({
            total_positions: 1000,
            labels: [
              { label: "blurry", frames, episodes: 1, per_episode: false, positions_lost: frames },
            ],
          }),
        }),
      ),
    );
  };
  useForm({ chunk_size: 50 });

  const first = annotateFlagCost(box, { root: "/a" }); //  frames = 2
  const second = annotateFlagCost(box, { root: "/bb" }); // frames = 3
  assert.strictEqual(pending.length, 2, "both renders must have reached the fetch");
  pending[1]();
  pending[0]();
  await Promise.all([first, second]);

  assert.strictEqual(
    label.appended.length,
    1,
    `the overtaken fetch must not append a second figure (got ${label.appended.length})`,
  );
  assert.ok(
    label.appended[0].textContent.includes("3 fr"),
    `the surviving figure must be the newer selection's, got ${label.appended[0].textContent}`,
  );
}

async function main() {
  theWindowComesFromTheForm();
  await theWindowReachesTheRequest();
  await anOvertakenFetchWritesNothing();
  console.log("flag_cost.test.js: all assertions passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
