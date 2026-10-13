// A stale editor can save one demonstration's marks into a different loaded demo.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const nodes = new Map();
const context = vm.createContext({
    console, assert, setImmediate,
    document: {getElementById: id => nodes.get(id) || null, querySelector: () => null},
    clearInterval: () => {},
});
const source = process.argv[2] || path.join(__dirname, '../../src/lerobot/gui/static/showservo.js');
vm.runInContext(fs.readFileSync(source, 'utf8'), context);
for (const id of ['ap-sub-demo', 'de-time', 'de-slider', 'de-frame', 'de-landing', 'de-play']) {
    nodes.set(id, {style: {}, removeAttribute() {}});
}

vm.runInContext(`(async () => {
    deStatus = demoStatus = pgState = deRenderList = deDrawStrip = deDrawOverlay = deLandingShow =
        deObjRefresh = deCheck = deRefreshPath = () => {};
    deSeek = i => { de.i = i; };
    deVisible = () => true;
    const curve = name => ({name, n: 8, t: [0,1,2,3,4,5,6,7], has_frames: true,
        keypoints: [{t: 1, kind: 'key', frame: name, move: 'line', grip: 'first'}]});
    const response = c => ({ok: true, json: async () => c});
    let current = curve('first');
    fetch = async () => response(current);
    pgPost = async (_url, {name}) => { current = curve(name); return current; };

    await deLoad(true);
    de.kps[0].t = 7; de.dirty = true; de.i = 6;
    de.playing = true; de.timer = 123; de.framePending = 7;
    await demoLoad('second');
    assert.equal(de.curve.name, 'second');
    assert.equal(de.kps[0].frame, 'second');
    assert.equal(de.kps[0].t, 1);
    assert.equal(de.dirty, false);
    assert.equal(de.playing, false);
    assert.equal(de.timer, null);
    assert.equal(de.framePending, null);
    assert.equal(de.i, 0);

    // Background polling preserves unsaved edits to the same demonstration.
    de.kps[0].t = 7; de.dirty = true;
    let reads = 0;
    fetch = async () => { reads++; return response(current); };
    deSync({demo: current, arm_connected: false});
    await new Promise(setImmediate);
    assert.equal(de.kps[0].t, 7);
    assert.equal(reads, 0);

    // A demo selected in another tab must replace dirty state from the old one.
    current = curve('third');
    deSync({demo: current, arm_connected: false});
    await new Promise(setImmediate);
    assert.equal(de.curve.name, 'third');
    assert.equal(de.kps[0].frame, 'third');
    assert.equal(de.dirty, false);

    // Explicit Load also replaces an unsaved edit when reloading the same name.
    de.kps[0].t = 7; de.dirty = true;
    await demoLoad('third');
    assert.equal(de.kps[0].t, 1);
    assert.equal(de.dirty, false);

    // An older curve response arriving last cannot put the previous demo back.
    let release;
    fetch = () => new Promise(resolve => { release = resolve; });
    const old = deLoad(true);
    fetch = async () => response(curve('newest'));
    await deLoad(true);
    release(response(curve('stale')));
    await old;
    assert.equal(de.curve.name, 'newest');
    assert.equal(de.kps[0].frame, 'newest');
})()`, context).catch(e => { console.error(e); process.exitCode = 1; });
