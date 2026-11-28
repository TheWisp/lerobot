// Exercise the shipped editor at its DOM, player and HTTP boundaries. Media
// decoding and durable edits are covered by test_flash_editor.py.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const source = fs.readFileSync(process.env.FLASH_EDITOR_SOURCE ||
    path.join(__dirname, '../../src/lerobot/gui/static/flash_editor.js'), 'utf8');

class Element {
    constructor() {
        Object.assign(this, {textContent:'', value:'', checked:false, disabled:false,
            hidden:false, children:[], dataset:{}, style:{}, attributes:{}, width:0, height:0});
        this.classList = {toggle() {}};
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    get firstChild() { return this.children[0]; }
    setAttribute(key, value) { this.attributes[key] = value; }
    closest() { return this; }
    querySelectorAll() { return []; }
    getBoundingClientRect() { return {width:0, height:0, left:0, top:0}; }
    getContext() { return {clearRect() {}}; }
}

function editor() {
    const elements = new Map(), players = [], requests = [];
    const element = id => {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
    };
    const context = vm.createContext({
        console, URLSearchParams, setTimeout:()=>0, clearTimeout() {},
        window:{addEventListener() {}}, Option:Element,
        document:{getElementById:element, createElement:()=>new Element(), addEventListener() {}},
        ChunkPlayer:{create(options) {
            const player = {options, closed:false, open() {}, close() {this.closed=true;},
                setRange(start,end) {this.range=[start,end];}, seek(frame) {this.seekFrame=frame;}, pause() {}};
            players.push(player);
            return player;
        }},
        fetch:async(url, options) => {
            requests.push({url, body:options?.body && JSON.parse(options.body)});
            return {ok:true, json:async()=>({})};
        },
    });
    vm.runInContext(source, context);
    const state = vm.runInContext('flashEditor', context);
    return {context, state, players, requests, el:id=>element('flash-editor-'+id)};
}

function recording(h, frames=[2,7]) {
    Object.assign(h.state, {mode:'demo', dataset:{id:'fixture'}, source:{fps:10, length:10,
        cameras:{scene:{dtype:'video'}}}, document:{camera:'scene', reference_frame:0,
        objects:[], keyframes:frames.map(frame=>({frame, reference:'world'}))}});
}

test('one keyframe and deletion down to one keep the full recording playable', () => {
    const h = editor();
    recording(h, [2]);
    h.context.openFlashDemoPlayer();
    assert.deepEqual(h.players[0].range, [0,10]);
    assert.equal(h.el('full-recording').disabled, true);
    assert.equal(h.el('full-recording').checked, true);

    // A real span still restricts playback when the user chooses it.
    h.state.document.keyframes.push({frame:7, reference:'world'});
    h.el('full-recording').checked=false;
    h.context.renderFlashEditor();
    assert.deepEqual(h.players[0].range, [2,8]);
    assert.equal(h.el('full-recording').disabled, false);
    h.el('keys').children[1].children[1].onclick();
    assert.deepEqual(h.players[0].range, [0,10]);
    assert.equal(h.state.document.keyframes.length, 1);
    assert.equal(h.el('full-recording').checked, true);
});

test('replacing a buffering player restores the status and ignores its old callbacks', () => {
    const h = editor();
    recording(h);
    h.context.editorStatus('Choose objects.');
    h.context.openFlashDemoPlayer();
    const old = h.players[0];
    old.options.onHold(true);
    assert.equal(h.el('status').textContent, 'Buffering recording…');
    h.context.openFlashDemoPlayer();
    const current = h.players[1];
    assert.equal(old.closed, true);
    current.options.onHold(true);
    old.options.onHold(false);
    assert.equal(h.el('status').textContent, 'Buffering recording…');
    current.options.onPaint(8, 0, null);
    current.options.onHold(false);
    assert.equal(h.el('status').textContent, 'Choose objects.');
    assert.equal(h.el('time').textContent, 'Frame 8 · 10 frames · 0.80 s');
    old.options.onPaint(2, 0, null);
    assert.equal(h.state.frame, 8);
    // Completing a load must not overwrite a newer user-facing error.
    current.options.onHold(true);
    h.context.editorStatus('Cannot save this edit.');
    current.options.onHold(false);
    assert.equal(h.el('status').textContent, 'Cannot save this edit.');
});

test('range keeps its starting reference and explains conflicts without changing keys', () => {
    const h = editor();
    recording(h, [5]);
    h.state.document.objects=[{id:'object_1', name:'green gamepad', point:null}];
    h.state.document.keyframes[0].reference='object_1';
    h.state.frame=3;
    h.context.startFlashDemoRange();
    assert.match(h.el('range-hint').textContent, /Start: frame 3 · Relative to World/);
    h.state.reference='object_1'; // Selecting another button doesn't rewrite a marked start.
    h.state.frame=6;
    h.context.finishFlashDemoRange();
    assert.match(h.el('status').textContent, /Frame 5 uses another reference: green gamepad/);
    assert.match(h.el('status').textContent, /This range uses World/);
    assert.equal(h.state.document.keyframes.length, 1);
    assert.equal(h.state.dirty, false);
    h.state.frame=3;
    h.context.startFlashDemoRange();
    assert.match(h.el('range-hint').textContent, /Relative to green gamepad/);
    h.state.frame=6;
    h.context.finishFlashDemoRange();
    assert.deepEqual(Array.from(h.state.document.keyframes, k=>[k.frame,k.reference]),
        [3,4,5,6].map(frame=>[frame,'object_1']));
    assert.equal(h.state.dirty, true);
});

for (const [name, active, label, endpoint, pid] of [
    ['recording', {recording:{pid:11}}, 'Finish recording', 'stop', 11],
    ['robot', {execution:{running:true, mode:'robot', pid:12}}, 'Stop robot', 'execution/stop', 12],
    ['simulation', {execution:{running:true, mode:'simulation', pid:13}}, 'Stop test run', 'execution/stop', 13],
    ['starting run with no mode yet', {execution:{running:true, state:'starting', pid:17}}, 'Stop run', 'execution/stop', 17],
    ['finished path', {execution:{running:true, mode:'robot', state:'complete', pid:14}}, 'End run', 'execution/stop', 14],
    ['reset cleanup', {execution:{running:true, mode:'robot', operation:'rest', state:'complete', pid:15}}, 'Stop robot', 'execution/stop', 15],
    ['rebuild', {rebuildPid:16}, 'Cancel tracking', 'cancel-tracking', 16],
    ['camera', {previewRunning:true}, 'Stop camera', 'preview/stop', undefined],
]) {
    test(`Stop describes and cancels the active ${name}`, async () => {
        const h = editor();
        Object.assign(h.state, active);
        // Polling/rendering is tested separately against the real server. Hold
        // it here so the request produced by the click is directly observable.
        h.context.pollFlashDemo=async()=>{};
        h.context.renderFlashEditor();
        assert.equal(h.el('stop').hidden, false);
        assert.equal(h.el('stop').disabled, false);
        assert.equal(h.el('stop').textContent, label);
        await h.context.stopFlashDemo();
        assert.equal(h.requests.length, 1);
        assert.equal(h.requests[0].url, '/api/run/flash/editor/'+endpoint);
        assert.equal(h.requests[0].body?.pid, pid);
    });
}

test('browsing an idle demo or a saved run has no Stop operation', () => {
    const h = editor();
    for (const mode of ['live','demo','rollout']) {
        h.state.mode=mode;
        h.state.execution={running:false, state:'failed', mode:'robot'};
        h.context.renderFlashEditor();
        assert.equal(h.el('stop').hidden, true);
        assert.equal(h.el('stop').disabled, true);
    }
});
