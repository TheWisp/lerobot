// Regression: the data-overlay pull gate. Root cause it locks (found by bisection,
// 2026-08-03): the page re-pulled + PNG-decoded overlays on every frame change
// (~52/s) while the worker produced ~8/s — the wasted client decodes cost ~4 fps of
// measured worker throughput. Contract: pulls follow the worker's overlay SEQ, ticks
// only pull as a rate-limited fallback when SSE is down.
'use strict';
const assert = require('assert');
const mod = require('../../src/lerobot/gui/static/overlay_pull_gate.js');
const { create } = mod;

// New seq pulls; identical seq does not; the next new one does.
let g = create();
assert.strictEqual(g.onSse('front', 1), true, 'first seq pulls');
assert.strictEqual(g.onSse('front', 1), false, 'repeat seq must not pull');
assert.strictEqual(g.onSse('front', 2), true, 'advanced seq pulls');

// Worker respawn: seqs restart BELOW the last seen value — must still pull
// (inequality, not greater-than, or a respawned worker would freeze the overlay).
assert.strictEqual(g.onSse('front', 50), true);
assert.strictEqual(g.onSse('front', 1), true, 'seq regression (respawn) must pull');

// Cameras are independent.
assert.strictEqual(g.onSse('top', 1), true, 'other camera has its own seq history');

// A missing seq (older server) fails open: always pull.
assert.strictEqual(g.onSse('front', undefined), true);
assert.strictEqual(g.onSse('front', undefined), true, 'seq-less events keep pulling');

// THE regression: ticks (frame changes / status polls) never pull while SSE is up.
g = create();
for (let i = 0; i < 100; i++) {
    assert.strictEqual(g.onTick('front', true, i * 33), false, 'tick with SSE connected must not pull');
}

// SSE down: ticks pull, but rate-limited per camera (default 400 ms).
g = create(400);
assert.strictEqual(g.onTick('front', false, 0), true, 'fallback first tick pulls');
assert.strictEqual(g.onTick('front', false, 100), false, 'within the window: no pull');
assert.strictEqual(g.onTick('front', false, 399), false);
assert.strictEqual(g.onTick('front', false, 401), true, 'after the window: pulls again');
assert.strictEqual(g.onTick('top', false, 401), true, 'cameras rate-limit independently');

// reset() forgets history: the same seq pulls again (config/dataset change re-renders).
g = create();
g.onSse('front', 7);
g.reset();
assert.strictEqual(g.onSse('front', 7), true, 'reset must allow a same-seq refresh');

// ---- createLoader: completion-gated latest-wins image loading ----
// Contract it pins (root cause: reassigning img.src mid-download aborts the fetch,
// so unthrottled reassignment under remote bandwidth means NO overlay ever finishes
// loading — the frozen-tile-with-healthy-badge report): one in-flight load per
// camera; newer requests while busy replace the pending slot; completion (load or
// error path both call done) starts the newest pending.
const { createLoader } = mod;
let l = createLoader();
assert.strictEqual(l.request('front', 'u1'), 'u1', 'idle camera: assign immediately');
assert.strictEqual(l.request('front', 'u2'), null, 'busy camera: hold');
assert.strictEqual(l.request('front', 'u3'), null, 'still busy: latest replaces pending');
assert.strictEqual(l.request('top', 'v1'), 'v1', 'cameras are independent');
assert.strictEqual(l.done('front'), 'u3', 'completion starts the NEWEST pending (u2 dropped)');
assert.strictEqual(l.done('front'), null, 'no pending: camera goes idle');
assert.strictEqual(l.request('front', 'u4'), 'u4', 'idle again: assign immediately');
assert.strictEqual(l.done('top'), null, 'top had no pending');

// reset() clears in-flight bookkeeping (config change / overlay teardown).
l = createLoader();
l.request('front', 'u1');
l.request('front', 'u2');
l.reset();
assert.strictEqual(l.request('front', 'u3'), 'u3', 'after reset the camera is idle');
assert.strictEqual(l.done('front'), null, 'reset dropped the stale pending');

// THE regression (found on real teleop, front camera only + a Blur background): the run
// tile loop resets the loader for every camera that is currently OFF, once per 50 ms
// tick. An unscoped reset there wiped the ON camera's in-flight flag ~60x/s, so the gate
// never engaged, every tick reassigned src and aborted the download, and with a treated
// (opaque, ~370 KB) overlay NO load ever completed: 404 requests, 0 responses, a tile
// frozen while the worker ran at 12 infer/s. reset(cam) must touch only that camera.
l = createLoader();
l.request('front', 'u1');            // front is loading
l.reset('top');                      // an off camera being swept must not disturb it
assert.strictEqual(l.request('front', 'u2'), null, 'front must still be in flight');
l.reset('front');
assert.strictEqual(l.request('front', 'u3'), 'u3', 'its own reset does clear it');

console.log('overlay_pull_gate: all assertions passed');
