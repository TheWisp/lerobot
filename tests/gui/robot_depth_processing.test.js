// The Robots tab's camera card offers the driver's own depth processing as
// checkboxes once the camera's depth is on: the RealSense's four filters,
// checked by default, and the ZED's fill mode, unchecked by default. A depth
// camera with depth off, or a UVC camera, shows none. Ticking one writes the
// config flag the driver reads, and the card shows it back.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "../../src/lerobot/gui/static/robot.js"),
  "utf8",
);
const context = vm.createContext({
  console,
  window: {},
  document: {
    getElementById() {
      return null;
    },
    createElement() {
      return { style: {}, setAttribute() {} };
    },
    body: { appendChild() {}, removeChild() {} },
  },
  setTimeout(callback) {
    callback();
  },
  esc(value) {
    return String(value);
  },
});
vm.runInContext(source, context);
context.esc = (value) => String(value); // the page's own esc needs a real DOM
const get = (expr) => JSON.parse(JSON.stringify(vm.runInContext(expr, context) ?? null));
vm.runInContext(
  `
  _rerender = function () {};
  _updateDirtyState = function () {};
  showToast = function () {};
  renderEditor = function () {};
  detectedCameras = [];
  currentProfile = { data: { cameras: {
    front: { type: "intelrealsense", serial_number_or_name: "0042", width: 640, height: 480, fps: 30, use_depth: true },
    top: { type: "zed", serial_number: 11975257, width: 1280, height: 720, fps: 30, use_depth: true, stereo_split: true },
    side: { type: "intelrealsense", serial_number_or_name: "0043", width: 640, height: 480, fps: 30 },
    wrist: { type: "opencv", index_or_path: "/dev/video2", width: 960, height: 600, fps: 30, fourcc: "YUYV" },
  } } };
  `,
  context,
);

const cardsOf = (html) => html.split('class="camera-assigned"');
const cardFor = (html, header) => {
  const card = cardsOf(html).find((c) => c.includes(header));
  assert.ok(card, `a card for ${header}`);
  return card;
};
const REALSENSE_FLAGS = ["enable_decimation", "enable_spatial", "enable_temporal", "enable_hole_filling"];

let html = get("renderCamerasSection()");
const front = cardFor(html, "intelrealsense - 0042");
for (const flag of REALSENSE_FLAGS) {
  assert.match(front, new RegExp(`<input type="checkbox" checked onchange="updateCameraConfig\\('front', '${flag}'`), `${flag} on by default`);
}
assert.doesNotMatch(front, /'fill_holes'/);
const top = cardFor(html, "zed - 11975257");
assert.match(top, /<input type="checkbox" onchange="updateCameraConfig\('top', 'fill_holes'/, "the ZED's fill mode is off by default");
for (const flag of REALSENSE_FLAGS) assert.doesNotMatch(top, new RegExp(flag));
assert.doesNotMatch(cardFor(html, "intelrealsense - 0043"), /Depth processing/, "no row while depth is off");
assert.doesNotMatch(cardFor(html, "opencv - /dev/video2"), /Depth processing/, "a UVC camera computes no depth");

// Ticking writes the flag the driver reads, and the card shows it back.
get(`updateCameraConfig("front", "enable_hole_filling", false)`);
get(`updateCameraConfig("top", "fill_holes", true)`);
assert.strictEqual(get("currentProfile.data.cameras.front.enable_hole_filling"), false);
assert.strictEqual(get("currentProfile.data.cameras.front.enable_spatial"), null, "untouched flags stay at the driver's default");
assert.strictEqual(get("currentProfile.data.cameras.top.fill_holes"), true);
html = get("renderCamerasSection()");
assert.match(cardFor(html, "intelrealsense - 0042"), /<input type="checkbox" onchange="updateCameraConfig\('front', 'enable_hole_filling'/);
assert.match(cardFor(html, "zed - 11975257"), /<input type="checkbox" checked onchange="updateCameraConfig\('top', 'fill_holes'/);

// The row follows the depth switch.
get(`updateCameraConfig("side", "use_depth", "true")`);
html = get("renderCamerasSection()");
assert.match(cardFor(html, "intelrealsense - 0043"), /'enable_hole_filling'/);
get(`updateCameraConfig("side", "use_depth", "false")`);
html = get("renderCamerasSection()");
assert.doesNotMatch(cardFor(html, "intelrealsense - 0043"), /'enable_hole_filling'/);
console.log("robot_depth_processing.test.js ok");
