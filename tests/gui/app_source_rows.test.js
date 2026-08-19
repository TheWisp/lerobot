// Unit tests for the pure row decisions in app.js.
//
// The dataset tree is built inside DOM code, so these rules were reachable only
// by driving a browser — which is how a placeholder shipped that was named
// unlike its neighbours, sorted after every dataset, and absent entirely from a
// source that had not been scanned yet.
//
// app.js is not a module and references helpers from sibling files, so loading
// it throws partway. Function declarations hoist, so the pure helpers are
// defined regardless; the stubs below only need to carry it far enough.

const assert = require("assert");

// Arrays built inside the VM carry that realm's Array prototype, which
// deepStrictEqual rejects however equal the contents. Copy into a host array.
const host = (a) => Array.from(a);
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const src = fs.readFileSync(
  path.join(__dirname, "../../src/lerobot/gui/static/app.js"),
  "utf8",
);

const noop = () => {};
const el = new Proxy(
  {},
  {
    get: (t, k) =>
      k === "style"
        ? {}
        : k === "classList"
          ? { add: noop, remove: noop, contains: () => false }
          : noop,
  },
);
const context = vm.createContext({
  console,
  window: {},
  setTimeout: noop,
  setInterval: noop,
  clearTimeout: noop,
  clearInterval: noop,
  fetch: noop,
  localStorage: { getItem: () => null, setItem: noop },
  location: { href: "", hash: "", host: "test" },
  navigator: {},
  document: {
    addEventListener: noop,
    getElementById: () => el,
    querySelector: () => el,
    querySelectorAll: () => [],
    createElement: () => el,
    body: el,
    documentElement: el,
  },
});
try {
  vm.runInContext(src, context);
} catch {
  // Expected: app.js reaches a helper defined in a sibling file. The hoisted
  // function declarations under test are already in place.
}

const { sourceRowsFor, duplicateNameFor } = context;
assert.strictEqual(typeof sourceRowsFor, "function", "sourceRowsFor must be reachable");
assert.strictEqual(typeof duplicateNameFor, "function", "duplicateNameFor must be reachable");

const SRC = "/hf";
const scanned = [
  { name: "o/alpha", root: "/hf/o/alpha", total_episodes: 2 },
  { name: "o/zulu", root: "/hf/o/zulu", total_episodes: 5 },
];

// ── Naming ──────────────────────────────────────────────────────────────────
// A copy is named the way a scanned row is: relative to the source root. The
// bare folder name reads as a different kind of thing in a tree of "owner/name".
{
  const pending = new Map([["/hf/o/mike_copy", { source: "/hf/o/mike" }]]);
  const row = sourceRowsFor(SRC, [], pending)[0];
  assert.strictEqual(row.name, "o/mike_copy");
  assert.strictEqual(row.root, "/hf/o/mike_copy");
  assert.strictEqual(row.copying, true);
}

// ── Ordering ────────────────────────────────────────────────────────────────
// Sorted among the scanned rows, not appended after them: in a source with 150
// datasets a trailing row is effectively invisible.
{
  const pending = new Map([["/hf/o/mike_copy", { source: "/hf/o/mike" }]]);
  const names = host(sourceRowsFor(SRC, scanned, pending).map((r) => r.name));
  assert.deepStrictEqual(names, ["o/alpha", "o/mike_copy", "o/zulu"]);
}

// ── Scope ───────────────────────────────────────────────────────────────────
// Only copies landing under this source, and a prefix match must not treat a
// sibling directory with the same leading characters as being inside it.
{
  const pending = new Map([
    ["/hf/o/mine", { source: "/hf/o/x" }],
    ["/other/o/theirs", { source: "/other/o/y" }],
    ["/hf2/o/nope", { source: "/hf2/o/z" }],
  ]);
  const names = host(sourceRowsFor(SRC, [], pending).map((r) => r.name));
  assert.deepStrictEqual(names, ["o/mine"]);
}

// ── Empty and missing inputs ────────────────────────────────────────────────
// A copy into a source with nothing scanned yet is exactly when the row matters.
{
  const pending = new Map([["/hf/o/first", { source: "/hf/o/src" }]]);
  assert.strictEqual(sourceRowsFor(SRC, [], pending).length, 1);
  assert.strictEqual(sourceRowsFor(SRC, undefined, pending).length, 1);
  assert.strictEqual(sourceRowsFor(SRC, scanned, new Map()).length, 2);
  assert.strictEqual(sourceRowsFor(SRC, scanned, undefined).length, 2);
}

// Scanned rows are passed through untouched — the tree still needs their counts.
{
  const row = sourceRowsFor(SRC, scanned, new Map())[0];
  assert.strictEqual(row.total_episodes, 2);
  assert.ok(!row.copying, "a scanned row must not be marked as copying");
}

// ── Suggested copy name ─────────────────────────────────────────────────────
assert.strictEqual(duplicateNameFor("/a/b/pick_place"), "pick_place_copy");
assert.strictEqual(duplicateNameFor("/a/b/pick_place/"), "pick_place_copy");
assert.strictEqual(duplicateNameFor(""), "dataset_copy");
assert.strictEqual(duplicateNameFor(undefined), "dataset_copy");

// ── Opened-panel label ──────────────────────────────────────────────────────
// Rows there are labelled `owner/name` (the repo_id), so a placeholder showing
// the bare folder reads as a different kind of thing beside its neighbours.
{
  const { openedLabelFor } = context;
  assert.strictEqual(typeof openedLabelFor, "function");
  assert.strictEqual(openedLabelFor("/hf/thewisp/run_05_copy"), "thewisp/run_05_copy");
  assert.strictEqual(openedLabelFor("/hf/thewisp/run_05_copy/"), "thewisp/run_05_copy");
  assert.strictEqual(openedLabelFor("solo"), "solo");
  assert.strictEqual(openedLabelFor(""), "");
}

// ── Scroll preservation ─────────────────────────────────────────────────────
// Rebuilding a panel's innerHTML scrolls it to the top however small the data
// change was, which is what made a copy or a delete feel like a full refresh.
{
  const { _withScrollPreserved } = context;
  assert.strictEqual(typeof _withScrollPreserved, "function");

  const scroller = { scrollTop: 250, className: "sources-section" };
  const node = { closest: (sel) => (sel.includes("sources-section") ? scroller : null) };

  // A rebuild that resets the offset has it put back.
  _withScrollPreserved(node, () => {
    scroller.scrollTop = 0;
  });
  assert.strictEqual(scroller.scrollTop, 250, "scroll offset must survive a rebuild");

  // A render that scrolls deliberately is not fought — only a reset is undone.
  _withScrollPreserved(node, noop);
  assert.strictEqual(scroller.scrollTop, 250);

  // No scrolling ancestor is not an error.
  assert.doesNotThrow(() => _withScrollPreserved({ closest: () => null }, noop));
  assert.doesNotThrow(() => _withScrollPreserved(null, noop));
}

console.log("app_source_rows.test.js: all assertions passed");
