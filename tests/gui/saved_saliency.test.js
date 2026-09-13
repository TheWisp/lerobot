const assert = require('node:assert/strict');
const { indexAt, create } = require('../../src/lerobot/gui/static/saved_saliency.js');
assert.deepEqual([-1, 0, 49, 50, 101].map(x => indexAt([0, 50, 100], x)), [-1, 0, 0, 1, 2]);
const settle = () => new Promise(r => setImmediate(r));
async function main() {
    let c = { dataset: 'a', episode: 0, frame: 0, cameras: ['camera'], selected: new Set(['camera']) };
    const pending = [], loads = [], notices = [];
    const tile = { style: {}, dataset: {}, src: '', removeAttribute() { this.src = ''; } };
    const controller = create({ context: () => c, imageFor: () => tile,
        report: (...x) => notices.push(x),
        fetcher: url => new Promise(resolve => pending.push({ url, resolve })),
        makeImage: () => { const image = {}; loads.push(image); return image; },
    });
    const respond = (n, meta) => pending[n].resolve({ ok: true, json: async () => meta });
    const meta = { available: true, revision: 'v1', frames: [0, 50, 100], cameras: { camera: {} } };
    controller.configure({}); respond(0, meta); await settle();
    assert.equal(loads.length, 2);
    loads[0].onload(); loads[1].onload();
    assert.equal(tile.dataset.heatmapFrame, '0');
    c.frame = 49; controller.tick(); assert.equal(tile.dataset.heatmapFrame, '0');
    c.frame = 50; controller.tick(); assert.equal(tile.dataset.heatmapFrame, '50');
    c.frame = 10; controller.tick(); assert.equal(tile.dataset.heatmapFrame, '0');
    c.selected.clear(); controller.tick(); assert.equal(tile.style.display, 'none');
    c.selected.add('camera');
    c.dataset = 'b'; controller.tick(); assert.equal(tile.style.display, 'none');
    // A late A image must not paint B. B metadata arriving after switching to C is also stale.
    loads[0].onload(); assert.equal(tile.style.display, 'none');
    c.dataset = 'c'; controller.tick(); respond(1, meta); await settle();
    assert.equal(tile.style.display, 'none');
    respond(2, { available: false, state: 'missing', message: '暂无热图' }); await settle();
    assert.equal(notices.at(-1)[0], '暂无热图');
    controller.configure({ style: 'inferno' });
    controller.stop(); respond(3, meta); await settle();
    assert.equal(tile.style.display, 'none');
    assert.equal(controller.isEnabled(), false);
    console.log('saved saliency: boundary, seek, camera, stale response, missing, off checks passed');
}
main().catch(e => { console.error(e); process.exitCode = 1; });
