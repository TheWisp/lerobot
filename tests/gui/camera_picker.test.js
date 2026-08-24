// Unit tests for the camera picker's read-back rule in training.js
// (run from pytest via test_camera_picker_js.py, or directly with
// `node tests/gui/camera_picker.test.js`).
//
// The rule that matters: every camera ticked must submit NOTHING, not the full
// list. Both trainers read an absent value as "use every camera", so sending
// the list instead would pin a run to the camera set the dataset happened to
// have on the day the form was opened — and would make every recipe recorded
// before this field existed replay differently from how it ran.
//
// The mirror-image hazard is treating an empty selection as absent too, which
// would silently train on everything after the user deliberately unticked it.
// That case must be distinguishable, which is why it returns [] and not
// undefined.

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

// A form whose only queryable content is the given checkboxes.
function formWith(boxes) {
  return {
    querySelectorAll: () => boxes,
    querySelector: () => boxes[0] || null,
  };
}
const box = (value, checked) => ({ value, checked, type: "checkbox" });
// Arrays built inside the vm have that realm's Array prototype, so
// deepStrictEqual would reject them against a host-realm literal. Copy across
// the boundary before comparing; undefined must survive as undefined.
const selection = (v) => (Array.isArray(v) ? Array.from(v) : v);
const FIELD = { type: "cameras", key: "dataset.cameras" };
const emptyFd = { get: () => null };

// ── The args-dict key ──────────────────────────────────────────────────────
{
  const draccus = { arg_key_prefix: "policy." };
  assert.strictEqual(
    trainingFormKey(draccus, { name: "cameras", arg_key: "dataset.cameras" }),
    "dataset.cameras",
    "arg_key must override the entry's prefix",
  );
  assert.strictEqual(
    trainingFormKey(draccus, { name: "chunk_size" }),
    "policy.chunk_size",
    "a field without arg_key still takes the prefix",
  );
  assert.strictEqual(
    trainingFormKey({ arg_key_prefix: "" }, { name: "cameras" }),
    "cameras",
    "HVLA's empty prefix leaves the bare name",
  );
}

// ── Reading the selection back ─────────────────────────────────────────────
{
  const all = formWith([box("top_l", true), box("top_r", true)]);
  assert.strictEqual(
    formValue(emptyFd, all, FIELD),
    undefined,
    "every camera ticked must submit nothing",
  );
}

{
  const some = formWith([box("top_l", true), box("top_r", false)]);
  assert.deepStrictEqual(
    selection(formValue(emptyFd, some, FIELD)),
    ["top_l"],
    "a subset must submit exactly that subset",
  );
}

{
  const none = formWith([box("top_l", false), box("top_r", false)]);
  assert.deepStrictEqual(
    selection(formValue(emptyFd, none, FIELD)),
    [],
    "an empty selection must be distinguishable from an absent one",
  );
}

{
  // A dataset that declares no cameras renders no checkboxes at all. That is
  // absence, not an empty selection — there was nothing to untick.
  assert.strictEqual(
    formValue(emptyFd, formWith([]), FIELD),
    undefined,
    "no checkboxes at all must read as absent",
  );
}

{
  // Order follows the rendered boxes, which follow the dataset's feature order.
  const boxes = formWith([box("a", true), box("b", false), box("c", true)]);
  assert.deepStrictEqual(selection(formValue(emptyFd, boxes, FIELD)), ["a", "c"]);
}

// ── Rendering ──────────────────────────────────────────────────────────────
{
  const html = fieldHtml({
    type: "cameras",
    key: "dataset.cameras",
    label: "Cameras to train on",
    description: "pick some",
  });
  assert.ok(
    html.includes('data-cameras-field="dataset.cameras"'),
    "the holder must carry the form key so the refresh can find it",
  );
  assert.ok(html.includes("training-cameras-box"), "the box the choices land in must exist");
  assert.ok(
    html.includes("Select a dataset to see its cameras"),
    "with no dataset chosen the picker must say why it is empty",
  );
  assert.ok(
    !html.startsWith("\n      <label"),
    "must not be a <label>: it wraps several checkboxes and a label may own only one",
  );
}

// ── The default state the user is shown ────────────────────────────────────
//
// "All ticked" and "submits nothing" are two halves of one contract, and only
// the second half is checked above. If the rendered checkboxes lost their
// `checked` attribute, every fresh form would come up with nothing selected and
// submitting would be refused — visible rather than silent, but still wrong.
// This drives the real listing → render path, which also covers the `cameras`
// copy in trainingLoadDatasets that has no other test.
(async () => {
  const boxes = [];
  const holder = {
    getAttribute: () => "dataset.cameras",
    querySelector: () => ({
      set innerHTML(html) {
        boxes.push(html);
      },
    }),
  };
  const form = {
    querySelector: (sel) => (sel.includes("dataset_id") ? { value: "ds/one" } : null),
    querySelectorAll: () => [holder],
  };
  context.document.getElementById = (id) => (id === "training-start-form" ? form : null);
  context.fetch = async (url) =>
    url.endsWith("/sources")
      ? { json: async () => [{ path: "/src" }] }
      : {
          json: async () => [
            { name: "ds/one", total_episodes: 1, total_frames: 1, cameras: ["top_l", "top_r"] },
          ],
        };

  await context.trainingLoadDatasets();
  context.trainingRefreshCameraPickers();

  assert.strictEqual(boxes.length, 1, "the picker must have been rendered once");
  const html = boxes[0];
  const checkboxes = html.match(/<input[^>]*type="checkbox"[^>]*>/g) || [];
  assert.strictEqual(checkboxes.length, 2, "one checkbox per camera the dataset declares");
  for (const box of checkboxes) {
    assert.ok(box.includes("checked"), `default state must be ticked, got: ${box}`);
  }
  assert.ok(html.includes('value="top_l"') && html.includes('value="top_r"'), "camera names must reach the control");

  console.log("camera_picker.test.js: all assertions passed");
})();
