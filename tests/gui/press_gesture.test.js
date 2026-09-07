// The press gesture, decided without a browser.
//
// `deferPress` is the rule that says what a press MEANS: the first movement
// past the slop is a drag, a release before it is a click, another button
// abandons it, and losing focus drops it. Every one of those was a defect
// once, and until now all of them were reachable only through Playwright --
// which is slow, and which tests the wiring rather than the rule.
//
// The module reaches for `document` and `window`, so this supplies the two
// listener registries it needs and nothing else.
const assert = require("assert");
const path = require("path");

function freshModule() {
  // A listener registry per case: the module keeps one pending press in module
  // scope, so cases must not share an instance.
  const listeners = { document: {}, window: {} };
  const target = (bag) => ({
    addEventListener(type, fn) { (bag[type] = bag[type] || []).push(fn); },
    removeEventListener(type, fn) {
      bag[type] = (bag[type] || []).filter((f) => f !== fn);
    },
  });
  global.document = target(listeners.document);
  global.window = target(listeners.window);
  const p = path.join(__dirname, "../../src/lerobot/gui/static/timeline_lanes.js");
  delete require.cache[require.resolve(p)];
  const TL = require(p);
  const fire = (where, type, ev) => (listeners[where][type] || []).slice().forEach((f) => f(ev));
  return { TL, fire, listeners };
}

const press = (x = 100, y = 50) => ({ clientX: x, clientY: y, button: 0 });

// ── A release without travel is a click ────────────────────────────────────
{
  const { TL, fire } = freshModule();
  const seen = [];
  TL.deferPress(press(), {
    onDrag: () => seen.push("drag"),
    onClick: () => seen.push("click"),
  });
  assert.strictEqual(TL.pressIsPending(), true, "the press should be held undecided");
  assert.deepStrictEqual(seen, [], "nothing may run before the gesture declares itself");
  fire("document", "mouseup", { clientX: 100, clientY: 50, button: 0 });
  assert.deepStrictEqual(seen, ["click"]);
  assert.strictEqual(TL.pressIsPending(), false);
}

// A wobble inside the slop is still a click -- the hand is not steady, and a
// press that has not travelled has not selected anything different.
const SLOP = freshModule().TL.DRAG_SLOP;
for (const [dx, dy] of [[0, 0], [SLOP, 0], [0, SLOP], [-SLOP, SLOP]]) {
  const { TL, fire } = freshModule();
  const seen = [];
  TL.deferPress(press(), { onDrag: () => seen.push("drag"), onClick: () => seen.push("click") });
  fire("document", "mousemove", { clientX: 100 + dx, clientY: 50 + dy });
  fire("document", "mouseup", { clientX: 100 + dx, clientY: 50 + dy, button: 0 });
  assert.deepStrictEqual(seen, ["click"], `travel (${dx},${dy}) should still be a click`);
}

// ── Travel past the slop is a drag, and it happens on the move ─────────────
{
  const { TL, fire } = freshModule();
  const seen = [];
  TL.deferPress(press(), {
    onDrag: (ev) => seen.push(["drag", ev.clientX]),
    onClick: () => seen.push("click"),
  });
  fire("document", "mousemove", { clientX: 100 + TL.DRAG_SLOP + 1, clientY: 50 });
  assert.deepStrictEqual(seen, [["drag", 100 + TL.DRAG_SLOP + 1]],
    "the drag must fire on the move that resolved it, and carry that event");
  // The later release must not also fire a click.
  fire("document", "mouseup", { clientX: 200, clientY: 50, button: 0 });
  assert.strictEqual(seen.length, 1, "exactly one outcome may run");
}

// ── Exactly one outcome, however many events arrive ────────────────────────
{
  const { TL, fire } = freshModule();
  let drags = 0, clicks = 0;
  TL.deferPress(press(), { onDrag: () => drags++, onClick: () => clicks++ });
  for (let i = 0; i < 5; i++) fire("document", "mousemove", { clientX: 300, clientY: 50 });
  for (let i = 0; i < 5; i++) fire("document", "mouseup", { clientX: 300, clientY: 50, button: 0 });
  assert.strictEqual(drags, 1, "the drag ran more than once");
  assert.strictEqual(clicks, 0, "a click ran after the drag had already won");
}

// ── Only the button that made the press may end it ─────────────────────────
{
  const { TL, fire } = freshModule();
  const seen = [];
  TL.deferPress(press(), { onDrag: () => seen.push("drag"), onClick: () => seen.push("click") });
  fire("document", "mouseup", { clientX: 100, clientY: 50, button: 2 });
  assert.deepStrictEqual(seen, [], "a right-button release resolved the left button's press");
}

// ── Another button going down abandons it ──────────────────────────────────
// A context menu between press and release makes the eventual release a
// different act from the one that was started.
{
  const { TL, fire } = freshModule();
  const seen = [];
  TL.deferPress(press(), { onDrag: () => seen.push("drag"), onClick: () => seen.push("click") });
  fire("document", "mousedown", { clientX: 100, clientY: 50, button: 2 });
  assert.strictEqual(TL.pressIsPending(), false, "the press should have been abandoned");
  fire("document", "mouseup", { clientX: 100, clientY: 50, button: 0 });
  assert.deepStrictEqual(seen, [], "the abandoned press fired on the later release");
}

// ── A press whose release never arrives is dropped, not left armed ─────────
// Pointer leaves the window, tab loses focus: the next unrelated click must
// not commit an edit the operator walked away from.
{
  const { TL, fire } = freshModule();
  const seen = [];
  TL.deferPress(press(), { onDrag: () => seen.push("drag"), onClick: () => seen.push("click") });
  fire("window", "blur", {});
  assert.strictEqual(TL.pressIsPending(), false);
  fire("document", "mouseup", { clientX: 100, clientY: 50, button: 0 });
  assert.deepStrictEqual(seen, []);
}

// ── A new press replaces an unresolved one ─────────────────────────────────
{
  const { TL, fire } = freshModule();
  const seen = [];
  TL.deferPress(press(100, 50), { onDrag: () => seen.push("drag1"), onClick: () => seen.push("click1") });
  TL.deferPress(press(400, 50), { onDrag: () => seen.push("drag2"), onClick: () => seen.push("click2") });
  fire("document", "mouseup", { clientX: 400, clientY: 50, button: 0 });
  assert.deepStrictEqual(seen, ["click2"], "the stale press outlived the one that replaced it");
}

// ── Listeners are released with the gesture ────────────────────────────────
// The pre-deferral code left a document listener per press; over a session
// that is an unbounded pile, each holding a row closure.
{
  const { TL, fire, listeners } = freshModule();
  const before = (listeners.document.mousemove || []).length;
  for (let i = 0; i < 20; i++) {
    TL.deferPress(press(), { onDrag() {}, onClick() {} });
    fire("document", "mouseup", { clientX: 100, clientY: 50, button: 0 });
  }
  assert.strictEqual((listeners.document.mousemove || []).length, before,
    "each press left its mousemove listener behind");
  assert.strictEqual((listeners.document.mouseup || []).length,
    (listeners.document.mouseup || []).filter(Boolean).length);
}

console.log("press_gesture.test.js: all assertions passed");
