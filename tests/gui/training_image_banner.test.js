// Unit tests for the image-status banner in training.js (run from pytest via
// test_training_image_banner_js.py, or directly with
// `node tests/gui/training_image_banner.test.js`).
//
// The banner is the only place a run says which image it is about to train on.
// Its renderer is a switch with an empty default, so an event the switch does
// not name renders as nothing at all — indistinguishable from a run that had
// no image step. That is how `image_refresh_failed` was first added invisible: the
// orchestrator emitted it, and the UI silently dropped it.

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(path.join(__dirname, "../../src/lerobot/gui/static/training.js"), "utf8");
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

const { trainingImageStatusHtml } = context;
assert.strictEqual(typeof trainingImageStatusHtml, "function", "banner renderer must be reachable");

const run = { state: "running" };
const banner = (...events) => trainingImageStatusHtml(run, events);
const IMG = "ghcr.io/thewisp/lerobot-training:latest";

// ── The regression: a refresh that failed must reach the operator ───────────
{
  const html = banner(
    { type: "image_pull_started", image: IMG },
    { type: "image_refresh_failed", image: IMG, error: "no route to host" },
  );
  assert.notStrictEqual(html, "", "a failed refresh must not render as nothing");
  assert.ok(/out of date/i.test(html), "it must say the image may be stale, not merely that a pull failed");
  assert.ok(html.includes("no route to host"), "the cause belongs on screen");
}

// ── It must not read as a normal, healthy pull ──────────────────────────────
{
  const refreshed = banner({ type: "image_refresh_failed", image: IMG, error: "timeout" });
  const pulled = banner({ type: "image_pulled", image: IMG, duration_s: 3, size_bytes: 10 });
  assert.notStrictEqual(refreshed, pulled, "a stale copy must not render like a fresh pull");
  assert.ok(!/banner ok/.test(refreshed), "it is not an ok state");
}

// ── A missing error tail must not print "undefined" at the operator ─────────
{
  const html = banner({ type: "image_refresh_failed", image: IMG });
  assert.ok(!html.includes("undefined"), `no undefined in the banner: ${html}`);
}

// ── The flows that already worked still do ─────────────────────────────────
{
  assert.ok(/cache hit/i.test(banner({ type: "image_cache_hit", image: IMG })));
  assert.ok(/pulled in/i.test(banner({ type: "image_pulled", image: IMG, duration_s: 3, size_bytes: 10 })));
  assert.ok(/pull failed/i.test(banner({ type: "image_pull_failed", image: IMG, error: "denied" })));
  assert.strictEqual(banner({ type: "image_wat", image: IMG }), "", "an unknown image event still renders empty");
  assert.strictEqual(trainingImageStatusHtml(run, []), "", "a run with no image step renders nothing");
}

console.log("training image banner: all assertions passed");
