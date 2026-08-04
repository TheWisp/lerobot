const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "../../src/lerobot/gui/static/robot.js"),
  "utf8",
);
const grid = { innerHTML: "" };
let copiedCommand = null;
let selectedText = null;
const context = vm.createContext({
  console,
  navigator: {
    clipboard: {
      writeText(value) {
        copiedCommand = value;
        return {
          then(callback) {
            callback();
            return { catch() {} };
          },
        };
      },
    },
  },
  window: { isSecureContext: true },
  setTimeout(callback) {
    callback();
  },
  document: {
    getElementById(id) {
      return id === "camera-preview-grid" ? grid : null;
    },
    createElement() {
      return {
        value: "",
        style: {},
        setAttribute() {},
        select() {
          selectedText = this.value;
        },
      };
    },
    body: {
      appendChild() {},
      removeChild() {},
    },
    execCommand(command) {
      if (command === "copy") copiedCommand = selectedText;
      return command === "copy";
    },
  },
  esc(value) {
    return String(value);
  },
});
vm.runInContext(source, context);
context.esc = (value) => String(value);

vm.runInContext(
  `
  currentProfile = { data: { cameras: {} } };
  detectedCameras = [
    {
      name: "Arducam A @ /dev/video0",
      type: "OpenCV",
      id: "/dev/video0",
      error: "Permission denied for /dev/video0",
      error_code: "permission_denied",
      error_summary: "Camera access denied",
      error_action: "Run this on the GUI server, then sign out and back in and restart LeRobot GUI:",
      error_command: "sudo usermod -aG video gui-user"
    },
    {
      name: "Arducam B @ /dev/video2",
      type: "OpenCV",
      id: "/dev/video2",
      error: "Permission denied for /dev/video2",
      error_code: "permission_denied",
      error_summary: "Camera access denied",
      error_action: "Run this on the GUI server, then sign out and back in and restart LeRobot GUI:",
      error_command: "sudo usermod -aG video gui-user"
    }
  ];
  renderCameraPreview();
  `,
  context,
);

assert.strictEqual((grid.innerHTML.match(/camera-preview-status/g) || []).length, 2);
assert.strictEqual((grid.innerHTML.match(/role="status"/g) || []).length, 2);
assert.strictEqual((grid.innerHTML.match(/<img/g) || []).length, 0);
assert.strictEqual((grid.innerHTML.match(/<select/g) || []).length, 0);
assert.strictEqual((grid.innerHTML.match(/Camera access denied/g) || []).length, 4);
assert.match(grid.innerHTML, /title="Permission denied for \/dev\/video0"/);
assert.match(grid.innerHTML, /title="Permission denied for \/dev\/video2"/);
assert.strictEqual((grid.innerHTML.match(/camera-remediation/g) || []).length, 1);
assert.strictEqual((grid.innerHTML.match(/sudo usermod -aG video gui-user/g) || []).length, 2);
assert.match(grid.innerHTML, /<code>sudo usermod -aG video gui-user<\/code>/);
assert.strictEqual((grid.innerHTML.match(/copyCameraCommand\(this\)/g) || []).length, 1);
assert.strictEqual((grid.innerHTML.match(/>Copy<\/button>/g) || []).length, 1);
assert.match(grid.innerHTML, /sign out and back in and restart LeRobot GUI/);

const copyButton = {
  dataset: { command: "sudo usermod -aG video gui-user" },
  textContent: "Copy",
};
context.copyCameraCommand(copyButton);
assert.strictEqual(copiedCommand, "sudo usermod -aG video gui-user");

copiedCommand = null;
context.navigator.clipboard = null;
context.window.isSecureContext = false;
context.copyCameraCommand(copyButton);
assert.strictEqual(copiedCommand, "sudo usermod -aG video gui-user");

vm.runInContext(
  `
  detectedCameras = [
    {
      name: "Working camera",
      type: "OpenCV",
      id: "/dev/video0",
      preview_index: 0
    }
  ];
  renderCameraPreview();
  `,
  context,
);
assert.strictEqual((grid.innerHTML.match(/<img/g) || []).length, 1);
assert.doesNotMatch(grid.innerHTML, /src=""/);
assert.match(grid.innerHTML, /onload="handleCameraPreviewLoaded\(0\)"/);
assert.match(grid.innerHTML, /onerror="handleCameraPreviewError\(0\)"/);

console.log("robot_camera_errors.test.js: all assertions passed");
