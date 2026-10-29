// The Robots tab's profile form with a ZED camera: assigning a detected ZED
// writes a `zed` camera config keyed by serial, the card offers depth and a
// depth mode and no UVC format, and the stereo toggle leaves a ZED's width
// alone while it still halves a UVC camera's.
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
// Script-level `let` state is not a property of the context; read it in-context,
// and through JSON so the realm's own Object prototype does not fail deep equality.
const get = (expr) => JSON.parse(JSON.stringify(vm.runInContext(expr, context) ?? null));
vm.runInContext(
  `
  _rerender = function () {};
  _updateDirtyState = function () {};
  showToast = function () {};
  renderEditor = function () {};
  currentProfile = { data: { cameras: {
    top: { type: "opencv", index_or_path: "/dev/v4l/by-id/usb-Technologies__Inc._ZED-M-video-index0",
           width: 1280, height: 720, fps: 30, fourcc: "YUYV", stereo_split: true },
    wrist: { type: "opencv", index_or_path: "/dev/video2", width: 960, height: 600, fps: 30, fourcc: "YUYV" },
  } } };
  detectedCameras = [{ name: "Stereolabs ZED-M", type: "ZED", id: "11975257",
                       default_stream_profile: { width: 1280, height: 720, fps: 30 } }];
  assignCameraRole(0, "top");
  `,
  context,
);

assert.deepStrictEqual(get("currentProfile.data.cameras.top"), {
  type: "zed",
  serial_number: 11975257,
  width: 1280,
  height: 720,
  fps: 30,
  stereo_split: true,
});

// The card: depth and depth mode offered, no UVC format; the wrist card keeps its format.
const html = get("renderCamerasSection()");
const cards = html.split('class="camera-assigned"');
const topCard = cards.find((c) => c.includes("zed - 11975257"));
const wristCard = cards.find((c) => c.includes("opencv - /dev/video2"));
assert.ok(topCard, "the ZED card names its type and serial");
assert.match(topCard, /'use_depth'/);
assert.match(topCard, /'depth_mode'/);
assert.match(topCard, /value="neural_light" selected/);
assert.doesNotMatch(topCard, /'fourcc'/);
assert.match(wristCard, /'fourcc'/);
assert.doesNotMatch(wristCard, /'depth_mode'/);

// Stereo toggle: a ZED is sized per eye either way; a UVC camera's width is one eye only when split.
get(`updateCameraConfig("top", "stereo_split", false)`);
assert.strictEqual(get("currentProfile.data.cameras.top.width"), 1280);
assert.strictEqual(get("currentProfile.data.cameras.top.stereo_split"), null);
get(`updateCameraConfig("wrist", "stereo_split", true)`);
assert.strictEqual(get("currentProfile.data.cameras.wrist.width"), 480);

get(`updateCameraConfig("top", "depth_mode", "neural")`);
get(`updateCameraConfig("top", "use_depth", "true")`);
assert.strictEqual(get("currentProfile.data.cameras.top.depth_mode"), "neural");
assert.strictEqual(get("currentProfile.data.cameras.top.use_depth"), true);

// Reassigning the same unit to another role finds it by serial and moves it.
get(`assignCameraRole(0, "overhead")`);
assert.strictEqual(get("currentProfile.data.cameras.top"), null);
assert.strictEqual(get("currentProfile.data.cameras.overhead.serial_number"), 11975257);

// A role sized for the UVC path's whole side-by-side frame is resized to one
// eye from the detected profile: 2560x720 is not a size the SDK opens at.
vm.runInContext(
  `currentProfile.data.cameras.side = { type: "opencv", index_or_path: "/dev/video4",
     width: 2560, height: 720, fps: 15, fourcc: "YUYV" };
   assignCameraRole(0, "side");`,
  context,
);
assert.deepStrictEqual(get("currentProfile.data.cameras.side"), {
  type: "zed",
  serial_number: 11975257,
  width: 1280,
  height: 720,
  fps: 30,
  stereo_split: true,
});
console.log("robot_zed_camera.test.js ok");
