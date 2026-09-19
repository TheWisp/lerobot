// What a segmentation step's prompt rows start from when the panel leaves
// saved-effects mode (run from pytest via test_mask_seed_js.py, or directly
// with `node tests/gui/mask_seed.test.js`).
//
// The regression this locks: picking SAM3 on a dataset that already had saved
// masks blanked every prompt box. Reported as "loading SAM3 actually clears
// all the current masks — it's a hassle if I had like 5 specifically named
// ones". Saved-effects mode borrows the rows to show the stored vocabulary,
// and leaving it restored whatever the operator had before — which on a
// freshly opened dataset is one empty row, so five typed names became none.
//
// It was introduced fixing the opposite fault: the rows used to be left
// holding the saved vocabulary when a step was picked, so typing into one
// appended and the worker was asked for "white traywhite tray". Seeding and
// not-clobbering are therefore both pinned here, because a fix for either one
// alone is how this got broken.
const assert = require("assert");
const { seedForStep } = require("../../src/lerobot/gui/static/mask_seed.js");

const BLANK = [{ name: "", sign: "+", treatment: { key: "none", params: {} } }];
const RECIPE = {
  labels: ["white tray", "orange ball", "robot"],
  treatments: { "white tray": { key: "tint", params: { color: [255, 0, 0] } } },
  background: { key: "blur", params: {} },
};

// 1. The reported case: nothing of the operator's own, a dataset that names
//    its objects — the step opens with them rather than with a blank row.
{
  const seed = seedForStep(BLANK, RECIPE);
  assert.deepStrictEqual(seed.objects.map((o) => o.name), RECIPE.labels, "saved vocabulary not seeded");
  assert.strictEqual(seed.source, "saved");
}

// 2. A saved treatment travels with its object, so re-running a segmenter does
//    not silently reset how each region renders.
{
  const seed = seedForStep(BLANK, RECIPE);
  assert.strictEqual(seed.objects[0].treatment.key, "tint");
  assert.strictEqual(seed.objects[1].treatment.key, "none", "unnamed treatment should default to none");
}

// 3. The data-loss trap: a save writes the panel's background treatment, so
//    opening at 'none' over a dataset saved with 'blur' would downgrade it.
{
  assert.strictEqual(seedForStep(BLANK, RECIPE).background.key, "blur", "saved background not carried");
}

// 4. The fault the seeding must not reintroduce: typed prompts are never
//    overwritten by the saved vocabulary.
{
  const typed = [{ name: "wooden block", sign: "+", treatment: { key: "none", params: {} } }];
  const seed = seedForStep(typed, RECIPE);
  assert.deepStrictEqual(seed.objects.map((o) => o.name), ["wooden block"], "operator's prompts were clobbered");
  assert.strictEqual(seed.source, "operator");
}

// 5. A dataset with no saved masks still opens with something to type into.
{
  for (const recipe of [null, undefined, { labels: [] }]) {
    const seed = seedForStep(BLANK, recipe);
    assert.strictEqual(seed.objects.length, 1, "no row to type into");
    assert.strictEqual(seed.objects[0].name, "");
    assert.strictEqual(seed.source, "blank");
  }
}

// 6. The seed is a copy. Editing a row must not reach back into the saved
//    recipe, which the rest of the panel reads to decide what is committed.
{
  const recipe = JSON.parse(JSON.stringify(RECIPE));
  const seed = seedForStep(BLANK, recipe);
  seed.objects[0].name = "edited";
  seed.objects[0].treatment.key = "solid";
  seed.background.key = "random";
  assert.strictEqual(recipe.labels[0], "white tray", "editing a row mutated the saved recipe");
  assert.strictEqual(recipe.treatments["white tray"].key, "tint");
  assert.strictEqual(recipe.background.key, "blur");
}

// 7. Whitespace is not a prompt: a row holding only spaces is not the
//    operator's work, and must not suppress the seeding.
{
  const seed = seedForStep([{ name: "   ", sign: "+", treatment: { key: "none", params: {} } }], RECIPE);
  assert.strictEqual(seed.source, "saved");
}

console.log("mask_seed.test.js: all assertions passed");
