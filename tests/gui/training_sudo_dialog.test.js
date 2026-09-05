// The sudo-password dialog's submit path (run from pytest via
// test_training_sudo_dialog_js.py, or directly with
// `node tests/gui/training_sudo_dialog.test.js`).
//
// Submitting is not one request. It reads the failed run's configuration back,
// then posts a new run carrying the password — and it is offered precisely on
// hosts where each of those is a slow SSH round trip. So the dialog sits there
// looking idle for seconds after the click, which is exactly how a person is
// led to click again or hold Enter.
//
// Every extra submit would start another run, each provisioning the same host
// as root with the same password.

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(path.join(__dirname, "../../src/lerobot/gui/static/training.js"), "utf8");

const noop = () => {};
const els = {
  "sudo-password-overlay": { dataset: { runId: "run-1" }, style: {} },
  "sudo-password-input": { value: "hunter2", focus: noop },
  "sudo-password-error": { textContent: "" },
};

const posts = [];
const inFlight = [];
const context = vm.createContext({
  console,
  window: {},
  setTimeout: noop,
  setInterval: noop,
  clearInterval: noop,
  fetch: (url, opts) => {
    if (opts && opts.method === "POST") posts.push(JSON.parse(opts.body));
    return new Promise((resolve) => inFlight.push({ url, opts, resolve }));
  },
  document: {
    addEventListener: noop,
    getElementById: (id) => els[id] || null,
    querySelector: () => null,
    querySelectorAll: () => [],
  },
  localStorage: { getItem: () => null, setItem: noop },
});
vm.runInContext(source, context);
context.trainingSelectRun = noop;
context.trainingRefreshRuns = noop;

const settle = () => new Promise((resolve) => setImmediate(resolve));

async function main() {
  // ── A second submit while the first is in flight must not start a run ────
  {
    context.trainingSubmitSudoPassword();
    await settle();
    assert.strictEqual(inFlight.length, 1, "the config read should be in flight");

    context.trainingSubmitSudoPassword();
    context.trainingSubmitSudoPassword();
    await settle();
    assert.strictEqual(inFlight.length, 1, "further submits must not issue more requests");

    // Let the first one through: config read, then exactly one POST.
    inFlight[0].resolve({ ok: true, json: async () => ({ run: { run_id: "run-1", host_id: "h", recipe_name: "r", dataset_id: "d", args: {} } }) });
    await settle();
    await settle();
    assert.strictEqual(posts.length, 1, "exactly one run may be started");
    assert.strictEqual(posts[0].sudo_password, "hunter2", "the password must ride on the retry");
  }

  // ── The guard releases, so a failed attempt can be retried ───────────────
  {
    const post = inFlight.find((f) => f.opts && f.opts.method === "POST");
    post.resolve({ ok: false, status: 500 });
    await settle();
    await settle();

    const before = inFlight.length;
    context.trainingSubmitSudoPassword();
    await settle();
    assert.ok(inFlight.length > before, "a submit after a failure must be allowed through");
  }

  // ── The password is never put in the run's args ──────────────────────────
  {
    assert.ok(
      !Object.keys(posts[0].args || {}).some((k) => /sudo|password/i.test(k)),
      "args are copied onto the Run and written to run.json — the password must not be in them",
    );
  }

  console.log("training_sudo_dialog: all assertions passed");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
