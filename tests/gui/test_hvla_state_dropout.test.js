// Execute the unchanged production renderer and checkbox reader with the live schema.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const field = JSON.parse(fs.readFileSync(0, "utf8"));
field.key = field.name;
const noop = () => {};
const context = vm.createContext({
  console, window: {}, setTimeout: noop, setInterval: noop, fetch: noop,
  document: { addEventListener: noop, getElementById: () => null, querySelector: () => null, querySelectorAll: () => [] },
  localStorage: { getItem: () => null, setItem: noop }, CSS: { escape: String },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../../src/lerobot/gui/static/training.js"), "utf8"), context);
const html = context.fieldHtml(field);
assert.match(html, /type="checkbox"/);
assert.match(html, /name="state_dropout"/);
assert.doesNotMatch(html, /\schecked[\s/>]/);
for (const checked of [true, false, true, false]) {
  const form = { querySelector: () => ({ checked }) };
  assert.equal(context.formValue({ get: () => null }, form, field), checked);
}
console.log("HVLA checkbox: default off, render and repeated toggle read-back passed");
