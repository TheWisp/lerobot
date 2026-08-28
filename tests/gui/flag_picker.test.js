// Unit tests for the flag picker's read-back rule in training.js
// (run from pytest via test_flag_picker_js.py, or directly with
// `node tests/gui/flag_picker.test.js`).
//
// The flag picker is the camera picker's mirror image, and the mirroring is the
// thing worth pinning. Cameras are an inclusion list, so *all* ticked is the
// default and submits nothing; flags are an exclusion list, so *none* ticked is
// the default and submits nothing. Both defaults must be the absent value,
// because that is what a recipe recorded before either field existed replays as.
//
// The hazard specific to this direction: submitting [] instead of undefined.
// DatasetConfig refuses an empty exclude_flags precisely so a run cannot report
// itself filtered while training on every frame — so if the read-back ever
// returned [], every default run would be refused at container start, minutes
// after the user pressed the button.

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

const { formValue, trainingFormKey, fieldHtml } = context;
for (const [name, fn] of Object.entries({ formValue, trainingFormKey, fieldHtml })) {
  assert.strictEqual(typeof fn, "function", `${name} must be reachable`);
}

function formWith(boxes) {
  return {
    querySelectorAll: () => boxes,
    querySelector: () => boxes[0] || null,
  };
}
const box = (value, checked) => ({ value, checked, type: "checkbox" });
// Arrays built inside the vm carry that realm's prototype; copy before comparing.
const selection = (v) => (Array.isArray(v) ? Array.from(v) : v);
const FIELD = { type: "flags", key: "dataset.exclude_flags" };
const emptyFd = { get: () => null };

// ── The args-dict key ──────────────────────────────────────────────────────
{
  const draccus = { arg_key_prefix: "policy." };
  assert.strictEqual(
    trainingFormKey(draccus, { name: "exclude_flags", arg_key: "dataset.exclude_flags" }),
    "dataset.exclude_flags",
    "arg_key must override the entry's prefix — the selection restricts the dataset, not the policy",
  );
  assert.strictEqual(
    trainingFormKey({ arg_key_prefix: "" }, { name: "exclude_flags" }),
    "exclude_flags",
    "HVLA's empty prefix leaves the bare name its argparse wants",
  );
}

// ── Reading the selection back ─────────────────────────────────────────────
{
  const none = formWith([box("blurry", false), box("fumble", false)]);
  assert.strictEqual(
    formValue(emptyFd, none, FIELD),
    undefined,
    "nothing ticked must submit nothing, not []",
  );
}

{
  const one = formWith([box("blurry", false), box("fumble", true)]);
  assert.deepStrictEqual(
    selection(formValue(emptyFd, one, FIELD)),
    ["fumble"],
    "a subset must submit exactly that subset",
  );
}

{
  const all = formWith([box("blurry", true), box("fumble", true)]);
  assert.deepStrictEqual(
    selection(formValue(emptyFd, all, FIELD)),
    ["blurry", "fumble"],
    "every flag ticked is a real selection, unlike the camera picker where it is the default",
  );
}

{
  // A dataset declaring no flags renders no checkboxes; that is absence.
  assert.strictEqual(
    formValue(emptyFd, formWith([]), FIELD),
    undefined,
    "no checkboxes at all must read as absent",
  );
}

{
  // Order follows the rendered boxes, which follow declaration (bit) order.
  const boxes = formWith([box("a", true), box("b", false), box("c", true)]);
  assert.deepStrictEqual(selection(formValue(emptyFd, boxes, FIELD)), ["a", "c"]);
}

// ── The two pickers must not read back the same way ────────────────────────
{
  // Stated as a single assertion because the whole design rests on it: the same
  // physical state — every box ticked — means "no restriction" for cameras and
  // "exclude everything" for flags.
  const all = formWith([box("x", true), box("y", true)]);
  assert.strictEqual(formValue(emptyFd, all, { type: "cameras", key: "dataset.cameras" }), undefined);
  assert.deepStrictEqual(selection(formValue(emptyFd, all, FIELD)), ["x", "y"]);
}

// ── Rendering ──────────────────────────────────────────────────────────────
{
  const html = fieldHtml({
    type: "flags",
    key: "dataset.exclude_flags",
    label: "Flags to exclude",
    description: "pick some",
  });
  assert.ok(
    html.includes('data-flags-field="dataset.exclude_flags"'),
    "the holder must carry the form key so the refresh can find it",
  );
  assert.ok(html.includes("training-flags-box"), "the box the choices land in must exist");
  assert.ok(
    html.includes("Select a dataset to see its flags"),
    "with no dataset chosen the picker must say why it is empty",
  );
  assert.ok(
    !html.startsWith("\n      <label"),
    "must not be a <label>: it wraps several checkboxes and a label may own only one",
  );
}

// ── The default state the user is shown ────────────────────────────────────
//
// "None ticked" and "submits nothing" are two halves of one contract, and only
// the second half is checked above. This drives the real listing → render path,
// which also covers the `flags` copy in trainingLoadDatasets that has no other
// test — an omitted key there would leave the picker permanently empty with no
// error anywhere.
(async () => {
  const rendered = [];
  const holderFor = (attr) => ({
    getAttribute: () => attr,
    querySelector: () => ({
      set innerHTML(html) {
        rendered.push([attr, html]);
      },
    }),
  });
  const form = {
    querySelector: (sel) => (sel.includes("dataset_id") ? { value: "ds/one" } : null),
    querySelectorAll: (sel) =>
      sel.includes("data-flags-field") ? [holderFor("dataset.exclude_flags")] : [],
  };
  context.document.getElementById = (id) => (id === "training-start-form" ? form : null);
  context.fetch = async (url) =>
    url.endsWith("/sources")
      ? { json: async () => [{ path: "/src" }] }
      : {
          json: async () => [
            {
              name: "ds/one",
              total_episodes: 1,
              total_frames: 1,
              cameras: ["top"],
              flags: ["blurry", "fumble"],
            },
          ],
        };

  await context.trainingLoadDatasets();
  context.trainingRefreshDatasetPickers();

  assert.strictEqual(rendered.length, 1, "the flag picker must have been rendered once");
  const html = rendered[0][1];
  const checkboxes = html.match(/<input[^>]*type="checkbox"[^>]*>/g) || [];
  assert.strictEqual(checkboxes.length, 2, "one checkbox per flag the dataset declares");
  for (const b of checkboxes) {
    assert.ok(!b.includes("checked"), `default state must be unticked, got: ${b}`);
  }
  assert.ok(
    html.includes('value="blurry"') && html.includes('value="fumble"'),
    "flag names must reach the control",
  );

  // A dataset with no flags column must say so, and must not render a control
  // that looks like it has nothing wrong with it.
  rendered.length = 0;
  context.fetch = async (url) =>
    url.endsWith("/sources")
      ? { json: async () => [{ path: "/src" }] }
      : {
          json: async () => [
            { name: "ds/one", total_episodes: 1, total_frames: 1, cameras: ["top"] },
          ],
        };
  await context.trainingLoadDatasets();
  context.trainingRefreshDatasetPickers();
  assert.strictEqual(rendered.length, 1);
  assert.ok(
    rendered[0][1].includes("declares no flags"),
    `a dataset without flags must explain the empty picker, got: ${rendered[0][1]}`,
  );
  assert.ok(
    !rendered[0][1].includes("<input"),
    "a dataset without flags must render no checkboxes at all",
  );

  console.log("flag_picker.test.js: all assertions passed");
})();
