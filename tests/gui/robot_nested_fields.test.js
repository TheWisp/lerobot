const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "../../src/lerobot/gui/static/robot.js"),
  "utf8",
);
const context = vm.createContext({ console });
vm.runInContext(source, context);

const profileFields = {};
context._setNestedField(profileFields, "left_arm_config.port", "can1");
context._setNestedField(profileFields, "left_arm_config.side", "left");
context._setNestedField(profileFields, "right_arm_config.port", "can0");

assert.deepStrictEqual(JSON.parse(JSON.stringify(profileFields)), {
  left_arm_config: { port: "can1", side: "left" },
  right_arm_config: { port: "can0" },
});
assert.strictEqual(context._getNestedField(profileFields, "left_arm_config.port"), "can1");
assert.strictEqual(context._getNestedField(profileFields, "missing.port"), undefined);

context._deleteNestedField(profileFields, "right_arm_config.port");
assert.strictEqual(profileFields.right_arm_config, undefined, "empty nested containers are pruned");

console.log("robot_nested_fields.test.js: all assertions passed");
