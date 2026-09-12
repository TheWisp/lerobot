// A decoder with a small output pool must finish a chunk larger than its pool.
// Run: node tests/gui/chunk_frame_lifetime.test.js [optional player source]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = process.argv[2] || path.join(__dirname, '../../src/lerobot/gui/static/chunk_player.js');

async function exercise(copyFails = false) {
  let now = 0, raf = [], nativeLive = 0, nativePeak = 0, bitmapLive = 0;
  const painted = [];
  class Bitmap {
    constructor() { this.width = 320; this.height = 200; this.closed = false; bitmapLive++; }
    close() { if (!this.closed) { this.closed = true; bitmapLive--; } }
  }
  class Decoder {
    constructor(init) { this.init = init; this.state = 'unconfigured'; this.queue = []; this.pending = null; this.scheduled = false; }
    configure() { this.state = 'configured'; }
    decode(chunk) { this.queue.push(chunk); this.schedule(); }
    schedule() {
      if (this.scheduled) return;
      this.scheduled = true;
      queueMicrotask(() => {
        this.scheduled = false;
        while (this.state === 'configured' && this.queue.length && nativeLive < 15) {
          const chunk = this.queue.shift();
          nativeLive++; nativePeak = Math.max(nativePeak, nativeLive);
          let closed = false;
          this.init.output({
            timestamp: chunk.timestamp, displayWidth: 320, displayHeight: 200,
            close: () => { if (!closed) { closed = true; nativeLive--; this.schedule(); } },
          });
        }
        if (!this.queue.length && this.pending) { this.pending(); this.pending = null; }
      });
    }
    flush() { return new Promise(resolve => { this.pending = resolve; this.schedule(); }); }
    close() { this.state = 'closed'; this.queue.length = 0; }
  }
  const header = Buffer.from(JSON.stringify({first_frame: 0, frames: 60, chunk_frames: 60, parts: [{
    kind: 'video', camera: 'cam', offset: 0, length: 60, width: 320, height: 200,
    stored_width: 320, stored_height: 200, frame_sizes: Array(60).fill(1),
  }]}));
  const body = Buffer.alloc(4 + header.length + 60);
  body.writeUInt32LE(header.length); header.copy(body, 4);
  const canvas = {width: 320, height: 200, getContext: () => ({drawImage: image => {
    assert.ok(image instanceof Bitmap, 'paint cached independent pictures');
    assert.equal(image.closed, false);
  }})};
  const ctx = {
    window: {}, Date, performance: {now: () => now},
    requestAnimationFrame: callback => raf.push(callback),
    AbortController, TextDecoder, Uint8Array, DataView, Map, Set, WeakMap,
    setTimeout, clearTimeout, setInterval, clearInterval,
    VideoDecoder: Decoder, EncodedVideoChunk: class {constructor(init) {Object.assign(this, init);}},
    OffscreenCanvas: class {
      constructor(w, h) {this.width = w; this.height = h;}
      getContext() {return {drawImage() {if (copyFails) throw new Error('snapshot failed');}};}
      transferToImageBitmap() {return new Bitmap();}
    },
    fetch: async () => ({ok: true, headers: {get: () => null}, arrayBuffer: async () => body.buffer.slice(body.byteOffset, body.byteOffset + body.length)}),
  };
  vm.createContext(ctx); vm.runInContext(fs.readFileSync(source, 'utf8'), ctx);
  const player = ctx.window.ChunkPlayer.create({datasetId: 'test', fps: 30, length: 60, cameras: ['cam'], tiles: () => canvas, onPaint: frame => painted.push(frame)});
  try {
    player.open(0);
    for (let i = 0; i < 10; i++) await new Promise(setImmediate);
    assert.equal(nativeLive, 0, 'decoder output frames are released, including copy failures');
    if (copyFails) {
      assert.equal(bitmapLive, 0);
      assert.ok(player.metrics.errors.length > 0);
    } else {
      assert.equal(player.metrics.decode.cam.frames, 60, 'all 60 frames decode with a 15-frame output pool');
      assert.equal(bitmapLive, 60, 'cached pictures remain available for scrubbing');
      for (const frame of [45, 5, 59]) {
        player.seek(frame); now += 34;
        const callbacks = raf; raf = []; callbacks.forEach(f => f());
        assert.equal(painted.at(-1), frame);
      }
      assert.equal(player.metrics.errors.length, 0);
    }
    assert.ok(nativePeak <= 15);
  } finally { player.close(); }
  assert.equal(bitmapLive, 0, 'closing the player releases all snapshots');
  assert.equal(nativeLive, 0);
}
(async () => {
  await exercise();
  await exercise(true);
  console.log('chunk_frame_lifetime: ok');
})().catch(error => { console.error(error); process.exitCode = 1; });
