/*
  Video playback of a stored episode in the Data tab, a few seconds at a time
  (gui/docs/dataset_playback.md).

  The player owns one frame counter, the buffer of decoded chunks, one
  VideoDecoder per camera, and the paint of every camera's frame j onto the
  canvas the host hands it per camera, with that frame's mask rows handed back
  to the host to draw. The host owns everything around it: the tiles, the
  readouts, the timeline, the mask layer's palette and label visibility, the
  controls. The contract:

    const player = ChunkPlayer.create({
      datasetId, fps, length, cameras,        // what the tab already knows about the episode
      tiles: (cameraKey) => canvas,           // the canvas a camera's frames paint on, or null
      drawMasks: (cameraKey, rows, size, labels, declared) => {},  // this frame's mask rows at `size`; `declared` is the camera's own [h, w]
      entryEnabled: (entry) => true,   // the mask layer's rule for a stored row's enabled flag
      onPaint: (frame, episode) => {},        // frame j of every camera is on screen
      onHold: (waiting) => {},                // the frame counter is (not) waiting for a chunk
      onLog: (line) => {},
    });
    player.open(episode);

  Nothing adapts: one profile, one chunk length, learned from the server's
  header and checked against what the page assumed. Playback wraps within the
  range the host sets (the episode by default) and never leaves the episode on
  its own. If any camera lacks frame j, no camera advances.

  player.metrics is the record a test reads: chunks, painted, stalls, seeks,
  wraps, decode, errors, firstPicture.
*/
(function () {
  'use strict';

  const CHUNK_SECONDS = 2;     // must agree with the server's CHUNK_SECONDS; the first header is checked against it
  const MAX_INFLIGHT = 2;      // a slow link is shared by every request in flight
  const BUFFER_TARGET_S = 4;
  // A chunk the page holds but cannot show -- its frames never arrive from the
  // decoder, its masks never decode -- is dropped and asked for again after
  // this long, a few times, and then reported as an error. Without it the
  // player held such a chunk forever, and the plan, seeing it held, never
  // asked again: a silent stop with nothing in any log.
  const HOLD_RETRY_MS = 4000;
  const MAX_CHUNK_RETRIES = 3;
  // How long a chunk's decode may take before the page stops waiting for it
  // and treats the chunk as one that will never be ready. Generous next to a
  // real decode, which is milliseconds, because a loaded machine is not a
  // broken one.
  const DECODE_BUDGET_MS = 4000;
  // A decoder that has taken its input and gone quiet -- no output, no error,
  // no flush -- is not going to finish on its own. The page holds the chunk's
  // bytes, so it costs one decode to find out, and a decoder torn down gives
  // back whatever it was holding. Bounded, because a chunk that will not
  // decode must still reach the give-up rather than loop here.
  const DECODE_QUIET_MS = 2500;
  const PAINT_LOG = 2000;      // painted frames kept for the metrics; ~a minute at 30 fps
  const MAX_REDECODES = 2;
  const PROFILE = 'low';

  // ---------------------------------------------------------------- decisions
  //
  // The player's decisions as pure functions of what it holds, so they can be
  // driven directly (tests/gui/chunk_player.test.js). Their invariants hold at
  // runtime too: a violation is a programming error, raised and recorded, but
  // playback is not the place to die, so the caller keeps its last choice.

  class RuleViolation extends Error {}
  function ruleAssert(cond, msg) { if (!cond) throw new RuleViolation(msg); }

  /** The start frame of the chunk that holds `frame`. */
  function chunkFor(frame, chunkFrames) {
    ruleAssert(chunkFrames > 0, `chunk of ${chunkFrames} frames`);
    return Math.floor(frame / chunkFrames) * chunkFrames;
  }

  /**
   * Frames of ready media from the clock forward, stopping at the first frame
   * not decoded on every camera, wrapping once within the range.
   * `held`: {start, end, readyTo} per chunk.
   */
  function coverageFrom(s) {
    const { held, rangeStart, rangeEnd } = s;
    ruleAssert(rangeEnd > rangeStart, `empty range ${rangeStart}..${rangeEnd}`);
    const clock = s.clock >= rangeStart && s.clock < rangeEnd ? s.clock : rangeStart;
    const at = (f) => held.find((h) => f >= h.start && f < h.end) || null;
    let f = clock, n = 0, wrapped = false;
    for (let guard = 0; guard <= 2 * held.length + 2; guard++) {
      if (f >= rangeEnd) { if (wrapped) break; f = rangeStart; wrapped = true; }
      if (wrapped && f >= clock) break;
      const h = at(f);
      if (!h || f >= h.readyTo) break;
      const stop = wrapped ? clock : rangeEnd;
      const end = Math.min(h.end, h.readyTo, stop);
      n += end - f;
      if (end < h.end || end < h.readyTo) break;
      f = end;
    }
    ruleAssert(n >= 0 && n <= rangeEnd - rangeStart, `covered ${n} of ${rangeEnd - rangeStart} frames`);
    return n;
  }

  /**
   * Which chunks to ask for next, as start frames in the order they are
   * needed: walk forward from the clock over what is held or in flight,
   * wrapping once, and ask at each gap until `targetFrames` are covered or
   * `maxInflight` requests stand. One request per chunk, every camera in it,
   * nothing per frame.
   *
   * Post: every start is on the grid, none repeats one held or in flight,
   * each starts at or before the frame it serves, at most the free slots.
   */
  function fetchPlan(s) {
    const { held, inflight, rangeStart, rangeEnd, chunkFrames, targetFrames, maxInflight } = s;
    ruleAssert(rangeEnd > rangeStart, `empty range ${rangeStart}..${rangeEnd}`);
    ruleAssert(chunkFrames > 0, `chunk of ${chunkFrames} frames`);
    const clock = s.clock >= rangeStart && s.clock < rangeEnd ? s.clock : rangeStart;
    const spans = held.concat(inflight);
    const at = (f) => spans.find((h) => f >= h.start && f < h.end) || null;
    const out = [];
    let f = clock, covered = 0, wrapped = false;
    let room = maxInflight - inflight.length;
    for (let guard = 0; guard < 64; guard++) {
      if (f >= rangeEnd) { if (wrapped) break; f = rangeStart; wrapped = true; }
      if (wrapped && f >= clock) break;
      const h = at(f);
      if (h) { covered += Math.min(h.end, rangeEnd) - f; f = h.end; continue; }
      if (covered >= targetFrames || room <= 0) break;
      const start = chunkFor(f, chunkFrames);
      ruleAssert(start <= f, `chunk ${start} starts after the frame ${f} it must serve`);
      out.push(start);
      spans.push({ start, end: start + chunkFrames });
      covered += Math.min(start + chunkFrames, rangeEnd) - f;
      f = start + chunkFrames;
      room--;
    }
    ruleAssert(out.length <= Math.max(0, maxInflight - inflight.length), `planned ${out.length} fetches`);
    ruleAssert(new Set(out).size === out.length, `planned the same chunk twice: ${out.join(', ')}`);
    return out;
  }

  /**
   * Which in-flight fetches a seek to `frame` gives up on: those neither
   * holding the frame nor within `keepAhead` frames ahead of it around the
   * wrap -- the same reach the plan asks for, so a seek that lands where the
   * player already is keeps the transfers it would ask for again. A frame
   * outside the range (nothing wanted) aborts everything.
   */
  function inflightToAbort(s) {
    const { inflight, frame, rangeStart, rangeEnd, keepAhead } = s;
    ruleAssert(rangeEnd > rangeStart, `empty range ${rangeStart}..${rangeEnd}`);
    if (frame < rangeStart || frame >= rangeEnd) return inflight.map((h) => h.start);
    const span = Math.max(1, rangeEnd - rangeStart);
    const fwd = (a, b) => (((b - a) % span) + span) % span;
    const out = [];
    for (const h of inflight) {
      if (h.start <= frame && frame < h.end) continue;
      if (fwd(frame, h.start) <= keepAhead) continue;
      out.push(h.start);
    }
    return out;
  }
  /**
   * Which held chunks to release: those neither just behind the clock nor
   * within reach ahead of it, measured around the wrap. Never the chunk the
   * clock is inside.
   */
  function chunksToDrop(s) {
    const { held, clock, rangeStart, rangeEnd, keepBehind, keepAhead } = s;
    ruleAssert(rangeEnd > rangeStart, `empty range ${rangeStart}..${rangeEnd}`);
    const span = Math.max(1, rangeEnd - rangeStart);
    const fwd = (a, b) => (((b - a) % span) + span) % span;
    const out = [];
    for (const h of held) {
      if (h.start <= clock && clock < h.end) continue;
      if (fwd(h.end, clock) <= keepBehind || fwd(clock, h.start) <= keepAhead) continue;
      out.push(h.start);
    }
    ruleAssert(!out.some((st) => held.some((h) => h.start === st && h.start <= clock && clock < h.end)),
      'about to drop the chunk the clock is inside');
    return out;
  }


  // ---------------------------------------------------------------- the player
  function create(opts) {
    const T0 = performance.now();
    const now = () => performance.now() - T0;
    const { datasetId, fps, length, cameras } = opts;
    const tiles = opts.tiles || (() => null);
    const drawMasks = opts.drawMasks || (() => {});
    const recipeFor = opts.recipeFor || (() => null);
    const decodeMask = opts.decodeMask || null;
    const entryEnabled = opts.entryEnabled || null;
    const offscreens = new Map();   // camera -> canvas at the encoded size, where treatments are applied
    const textures = new Map();     // `${episode}:${fingerprint}:${w}x${h}` -> the episode's noise
    const onPaint = opts.onPaint || (() => {});
    const onHold = opts.onHold || (() => {});
    const onGaveUp = opts.onGaveUp || (() => {});
    // The deadlines are configuration with the constants above as defaults.
    // The page passes none of them; tests drive the same logic without
    // sleeping the production values, which is minutes of CI across the
    // suites that exercise the give-up.
    const holdRetryMs = opts.holdRetryMs || HOLD_RETRY_MS;
    const decodeBudgetMs = opts.decodeBudgetMs || DECODE_BUDGET_MS;
    const decodeQuietMs = opts.decodeQuietMs || DECODE_QUIET_MS;
    const onLog = opts.onLog || (() => {});
    const M = { chunks: [], painted: [], stalls: [], seeks: [], wraps: [], decode: {}, errors: [], retries: [], events: [], firstPicture: null, peakFrames: 0, ticks: 0 };
    function log(msg) { onLog(`${(now() / 1000).toFixed(3)}s ${msg}`); }
    /** What the page holds, in one line: sent with every chunk request, so the
     *  server's chunk line is a record of the page as well as of the chunk. */
    function playerState() {
      const list = (m) => { const k = [...m.keys()].sort((a, b) => a - b); return k.length ? k.join(',') : '-'; };
      const dd = [...dead].sort((a, b) => a - b);
      return `cur=${Math.floor(cur)} held=${list(chunks)} inflight=${list(inflight)} dead=${dd.length ? dd.join(',') : '-'}`
        + ` frames=${liveFrames}/${M.peakFrames} painted=${M.painted.length} holds=${M.stalls.length} errors=${M.errors.length}`;
    }
    /** An event the operator would want in the server log -- a chunk given up
     *  on, a decoder error, a rule violation -- posted where the chunk log is,
     *  because a page that has stopped asking for chunks tells the log nothing. */
    function report(kind, detail) {
      M.events.push({ kind, detail, at: Date.now() });
      log(`${kind}: ${detail}`);
      if (episode === null) return;
      try {
        fetch(`${base()}/player-event`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, keepalive: true,
          body: JSON.stringify({ kind, detail: `${detail} [${playerState()}]` }),
        }).catch(() => {});
      } catch (e) { /* no fetch in this context: the console still has it */ }
    }
    function fail(msg) { M.errors.push(msg); report('error', msg); }
    if (typeof VideoDecoder === 'undefined') { M.errors.push('WebCodecs unavailable (insecure origin?)'); log('WebCodecs unavailable: serve over https or open on localhost'); }
    const retriesFor = new Map();   // chunk start -> times it was dropped as never ready, this episode
    // Chunk starts the player has spent its budget on. Asking again is what it
    // did on the rig -- the same chunk every four seconds for an hour, each
    // time reporting that it had already given up -- so a spent budget has to
    // mean something: the plan stops asking and playback steps over the gap.
    const dead = new Set();
    // Decoded frames the page is holding. A `VideoFrame` is a decoder output
    // buffer, not a picture in memory, and a decoder stalls when its client
    // does not give them back -- a hardware decoder's pool is small. The rig
    // stopped after fifteen frames of a sixty-frame chunk, which is the shape
    // of a pool running out, so the count rides on every chunk request.
    let liveFrames = 0;
    function openDecoders() {
      let n = 0;
      for (const rec of chunks.values()) {
        for (const d of Object.values(rec.decoders || {})) if (d.state === 'configured') n++;
      }
      return n;
    }

    const n = Math.max(1, Math.round(CHUNK_SECONDS * fps));   // frames per chunk, checked against every header
    let episode = null;
    let cur = 0, playing = false, rate = 1, lastTick = null, lastPainted = -1;
    let rangeStart = 0, rangeEnd = length;
    let holdSince = null, seekPending = null, closed = false;
    const chunks = new Map();    // start -> chunk record
    const inflight = new Map();  // start -> {abort}
    const ctxs = new WeakMap();
    const base = () => `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episode}`;

    function guarded(what, fn, fallback) {
      try { return fn(); }
      catch (e) {
        if (e instanceof RuleViolation) { fail(`rule (${what}): ${e.message}`); return fallback; }
        throw e;
      }
    }

    function parseChunk(buf) {
      const dv = new DataView(buf);
      const hl = dv.getUint32(0, true);
      const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
      const parts = {};
      for (const p of header.parts) parts[`${p.kind}:${p.camera}`] = { ...p, bytes: new Uint8Array(buf, 4 + hl + p.offset, p.length) };
      return { header, parts };
    }

    async function fetchChunk(start) {
      if (inflight.has(start) || chunks.has(start)) return;
      const abort = new AbortController();
      const mine = { abort };
      inflight.set(start, mine);
      const done = () => { if (inflight.get(start) === mine) inflight.delete(start); };
      const t = performance.now();
      const ep = episode;
      const url = `${base()}/chunk?start=${start}&profile=${PROFILE}`;
      let resp, buf;
      try {
        resp = await fetch(url, { signal: abort.signal, cache: 'no-store', headers: { 'X-Player': playerState() } });
        if (!resp.ok) { fail(`chunk ${start}: HTTP ${resp.status}`); done(); return; }
        buf = await resp.arrayBuffer();
      } catch (e) {
        done();
        if (e.name !== 'AbortError') fail(`chunk ${start}: ${e.message}`);
        return;
      }
      if (ep !== episode || closed) { done(); return; }
      const { header, parts } = parseChunk(buf);
      if (header.chunk_frames !== n) fail(`rule (grid): server chunks are ${header.chunk_frames} frames, page assumed ${n}`);
      const rec = { start: header.first_frame, frames: header.frames, header, parts, decoded: {}, decoders: {}, done: {}, masks: {}, maskSize: {}, maskLabels: {}, masksPending: 0, closed: false, arrivedAt: performance.now(), decodeDoneAt: null,
        lastOutputAt: performance.now(), redecodes: 0 };
      M.chunks.push({ start: rec.start, frames: rec.frames, bytes: buf.byteLength, ms: +(performance.now() - t).toFixed(1), cache: resp.headers.get('x-chunk-cache'), serverMs: +(resp.headers.get('x-chunk-ms') || 0), url, at: Date.now() });
      log(`chunk ${rec.start}+${rec.frames} ${resp.headers.get('x-chunk-cache')} ${(buf.byteLength / 1000).toFixed(0)} kB in ${(performance.now() - t).toFixed(0)} ms`);
      chunks.set(rec.start, rec);
      done();
      decodeChunk(rec);
    }

    /** Decodes every camera of one chunk and marks when they settled, which is
     *  when the chunk has had its chance to be paintable. */
    function decodeChunk(rec) {
      const flushes = [];
      for (const key of cameras) {
        const part = rec.parts[`video:${key}`];
        if (!part) { fail(`chunk ${rec.start}: no video for ${key}`); continue; }
        const frames = new Array(rec.frames).fill(null);
        rec.decoded[key] = frames;
        const stat = M.decode[key] || (M.decode[key] = { frames: 0, ms: 0 });
        const tDec = performance.now();
        // Cached pictures must not keep the decoder's output surfaces alive.
        // A canvas snapshot owns separate pixels;
        // cloning the VideoFrame would still retain its decoder surface.
        const surface = new OffscreenCanvas(part.width, part.height);
        const copy = surface.getContext('2d');
        const dec = new VideoDecoder({
          output: (frame) => {
            try {
              const k = Math.round(frame.timestamp * fps / 1e6);
              if (rec.closed || k < 0 || k >= rec.frames || frames[k]) return;
              if (surface.width !== frame.displayWidth) surface.width = frame.displayWidth;
              if (surface.height !== frame.displayHeight) surface.height = frame.displayHeight;
              copy.drawImage(frame, 0, 0);
              frames[k] = surface.transferToImageBitmap();
              rec.lastOutputAt = performance.now();
              stat.frames++; liveFrames++;
              if (liveFrames > M.peakFrames) M.peakFrames = liveFrames;
            } catch (e) {
              if (!rec.closed) fail(`chunk ${rec.start} ${key} snapshot: ${e.message}`);
            } finally {
              frame.close();
            }
          },
          error: (e) => { if (!rec.closed) fail(`chunk ${rec.start} ${key} decode: ${e.message}`); },
        });
        rec.decoders[key] = dec;
        dec.configure({ codec: part.codec_string || 'avc1.4d401f', optimizeForLatency: true });
        let off = 0;
        try {
          part.frame_sizes.forEach((size, i) => {
            const data = part.bytes.subarray(off, off + size); off += size;
            dec.decode(new EncodedVideoChunk({ type: i === 0 ? 'key' : 'delta', timestamp: Math.round(i * 1e6 / fps), data }));
          });
        } catch (e) {
          // An unusable stream (no keyframe after configure, say) throws here.
          // Close this camera's decoder now rather than leave it open until the
          // chunk is given up on, and let the other cameras decode.
          if (!rec.closed) fail(`chunk ${rec.start} ${key} decode: ${e.message}`);
          if (dec.state !== 'closed') dec.close();
          continue;
        }
        flushes.push(dec.flush()
          .then(() => { rec.done[key] = true; stat.ms += performance.now() - tDec; if (dec.state !== 'closed') dec.close(); })
          .catch((e) => { if (!rec.closed) fail(`chunk ${rec.start} ${key} flush: ${e.message}`); }));
      }
      for (const key of cameras) {
        const part = rec.parts[`masks:${key}`];
        if (!part) { rec.masks[key] = null; continue; }
        rec.maskSize[key] = part.size || null;
        rec.maskLabels[key] = part.labels || [];
        rec.masksPending++;
        const bytes = part.encoding === 'gzip'
          ? new Response(new Blob([part.bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
          : Promise.resolve(part.bytes.buffer.slice(part.bytes.byteOffset, part.bytes.byteOffset + part.bytes.byteLength));
        bytes.then((b) => { rec.masks[key] = JSON.parse(new TextDecoder().decode(b)); rec.masksPending--; })
          .catch((e) => { if (!rec.closed) fail(`chunk ${rec.start} ${key} masks: ${e.message}`); rec.masks[key] = null; rec.masksPending--; });
      }
      Promise.all(flushes).then(() => { rec.decodeDoneAt = performance.now(); });
    }

    function chunkAt(frame) { for (const rec of chunks.values()) if (frame >= rec.start && frame < rec.start + rec.frames) return rec; return null; }
    function frameReady(rec, j) { return rec.masksPending === 0 && cameras.every((k) => rec.decoded[k] && rec.decoded[k][j - rec.start]); }
    function closeRec(rec) {
      if (rec.closed) return;   // idempotent: the frame count must not be decremented twice
      rec.closed = true;
      for (const k of cameras) (rec.decoded[k] || []).forEach((fr) => { if (fr) { fr.close(); liveFrames--; } });
      // The decoders too: one still decoding a dropped chunk is work and a
      // decoder instance spent on frames nobody will paint.
      for (const d of Object.values(rec.decoders || {})) { if (d.state !== 'closed') { try { d.close(); } catch (e) { /* already gone */ } } }
    }

    function heldSpans() {
      const out = [];
      for (const rec of chunks.values()) {
        let readyTo = rec.start;
        if (rec.masksPending === 0) {
          while (readyTo < rec.start + rec.frames && cameras.every((k) => rec.decoded[k] && rec.decoded[k][readyTo - rec.start])) readyTo++;
        }
        out.push({ start: rec.start, end: rec.start + rec.frames, readyTo });
      }
      return out;
    }

    function abortInflightExcept(frame) {
      for (const [start, f] of inflight) {
        if (frame < start || frame >= start + n) { f.abort.abort(); inflight.delete(start); }
      }
    }

    function ensureAhead() {
      if (episode === null || closed) return;
      const clock = Math.floor(cur);
      // A dead chunk counts as covered: the plan must not ask for it again,
      // and the frames it would have carried are gone until a scrub back in.
      const held = [...chunks.values()].map((r) => ({ start: r.start, end: r.start + r.frames }))
        .concat([...dead].map((st) => ({ start: st, end: st + n })));
      const standing = [...inflight.keys()].map((st) => ({ start: st, end: st + n }));
      const targetFrames = BUFFER_TARGET_S * rate * fps;
      const plan = guarded('plan', () => fetchPlan({ held, inflight: standing, clock, rangeStart, rangeEnd, chunkFrames: n, targetFrames, maxInflight: MAX_INFLIGHT }), []);
      for (const start of plan) fetchChunk(start);
      for (const start of guarded('evict', () => chunksToDrop({ held, clock, rangeStart, rangeEnd, keepBehind: n, keepAhead: targetFrames + 2 * fps }), [])) {
        const rec = chunks.get(start);
        if (rec) { closeRec(rec); chunks.delete(start); }
      }
    }

    function bufferedFrames() { return guarded('coverage', () => coverageFrom({ held: heldSpans(), clock: Math.floor(cur), rangeStart, rangeEnd }), 0); }

    // ---------------------------------------------------------------- paint
    function paint(j) {
      const rec = chunkAt(j);
      const i = j - rec.start;
      for (const key of cameras) {
        const frame = rec.decoded[key][i];
        const canvas = tiles(key);
        if (!canvas) continue;
        let ctx = ctxs.get(canvas);
        if (!ctx) { ctx = canvas.getContext('2d'); ctxs.set(canvas, ctx); }
        // The canvas is the camera's declared resolution, as the JPEG path's <img>
        // is, so the tile has the same geometry at either profile; the decoded
        // frame is scaled up into it. Treatments are applied at the encoded size,
        // where the frame and the mask rows agree pixel for pixel.
        const vpart = rec.parts[`video:${key}`];
        const dh = (vpart && vpart.stored_height) || frame.height, dw = (vpart && vpart.stored_width) || frame.width;
        if (canvas.width !== dw || canvas.height !== dh) { canvas.width = dw; canvas.height = dh; }
        const rows = rec.masks[key] ? rec.masks[key][i] : null;
        const size = rec.maskSize[key] || null;
        const labels = rec.maskLabels[key] || [];
        const recipe = recipeFor(key);
        let source = frame;
        if (rows && size && recipe && decodeMask && entryEnabled && window.MaskComposite && size[0] === frame.height && size[1] === frame.width) {
          // The recipe's composite, by the same definition the library uses on
          // the JPEG path (mask_composite.js), at the encoded size where the
          // frame and the rows agree pixel for pixel; then scaled into the tile.
          const ew = frame.width, eh = frame.height;
          const masks = {};
          for (const entry of rows) {
            if (!entryEnabled(entry)) continue;   // muted: not part of the composite, as on the JPEG path
            const name = labels[entry[0]];
            if (name == null) continue;
            const m = decodeMask(entry[1], eh, ew);
            if (masks[name]) { const acc = masks[name]; for (let q = 0; q < acc.length; q++) if (m[q]) acc[q] = 1; } else masks[name] = m;
          }
          const fp = recipe.fingerprint || '';
          const texture = (region) => {
            const k = `${episode}:${fp}:${region}:${ew}x${eh}`;
            if (!textures.has(k)) textures.set(k, window.MaskComposite.noiseTexture(ew, eh, `${episode}:${fp}:${region}`));
            return textures.get(k);
          };
          const tex = { __bg__: texture('__bg__') };
          for (const name of Object.keys(masks)) tex[name] = texture(name);
          const scale = ew / dw;
          let off = offscreens.get(key);
          if (!off) { off = document.createElement('canvas'); offscreens.set(key, off); }
          if (off.width !== ew || off.height !== eh) { off.width = ew; off.height = eh; }
          const octx = off.getContext('2d', { willReadFrequently: true });
          octx.drawImage(frame, 0, 0, ew, eh);
          const img = octx.getImageData(0, 0, ew, eh);
          window.MaskComposite.compositeFrame({
            rgba: img.data, w: ew, h: eh, masks, treatments: recipe.treatments || {}, background: recipe.background,
            textures: tex, feather: Math.max(1, Math.round(5 * scale)), scale,
          });
          octx.putImageData(img, 0, 0);
          source = off;
        }
        ctx.drawImage(source, 0, 0, dw, dh);
        drawMasks(key, rows, size, labels, [dh, dw]);
      }
      // One entry per painted frame is thirty a second: an hour of review is a
      // hundred thousand objects, and `playerState` reads the length on every
      // chunk request. Keep a window long enough for anything asking "did it
      // paint frame N" and let the rest go.
      M.painted.push({ frame: j, episode, t: +now().toFixed(1), buffered: bufferedFrames() });
      if (M.painted.length > PAINT_LOG) M.painted.splice(0, M.painted.length - PAINT_LOG);
      if (M.firstPicture == null) { M.firstPicture = +now().toFixed(1); log(`first picture, frame ${j}, at ${M.firstPicture} ms`); }
      if (seekPending && j === seekPending.frame) { const ms = +(performance.now() - seekPending.t).toFixed(1); M.seeks.push({ frame: j, ms }); log(`seek to ${j}: painted after ${ms} ms`); seekPending = null; }
      onPaint(j, episode);
    }

    function clampIntoRange() { if (cur < rangeStart || cur >= rangeEnd) { cur = rangeStart; lastPainted = -1; } }

    // `fromFrame` says who called. Only the animation-frame chain re-arms
    // itself: when the timer below also re-armed it, every beat added another
    // chain and they multiplied -- 3,700 ticks a second after four seconds,
    // climbing. The work per tick is idempotent, so being called from both is
    // harmless; spawning a caller per call is not.
    function tick(fromFrame) {
      if (closed) return;
      if (fromFrame) requestAnimationFrame(() => tick(true));
      M.ticks++;
      if (episode === null) return;
      if (playing) clampIntoRange();
      const t = performance.now();
      const j = Math.floor(cur);
      const rec = chunkAt(j);
      if (!(rec && frameReady(rec, j))) {
        // A chunk given up on never arrives: while playing, step to the next
        // one rather than hold at a frame that is not coming. Only while
        // playing -- an operator who scrubbed to this frame asked for this
        // frame, and moving them somewhere else in the episode is not a
        // recovery, it is losing their place. Paused, the tile holds and the
        // toast says which frames are gone. The whole range dead is a stop,
        // not a spin.
        if (playing && dead.has(chunkFor(j, n))) {
          let probe = chunkFor(j, n), steps = 0;
          const all = Math.ceil((rangeEnd - rangeStart) / n);
          do {
            probe += n;
            if (probe >= rangeEnd) probe = rangeStart;
            steps++;
          } while (dead.has(probe) && steps <= all);
          // Every chunk given up on: hold, rather than race the counter through
          // an episode with nothing to paint.
          if (steps > all) { if (holdSince == null) { holdSince = t; onHold(true); } lastTick = t; return; }
          cur = probe; lastPainted = -1; lastTick = t; ensureAhead(); return;
        }
        if (holdSince == null) { holdSince = t; onHold(true); }
        if (rec && decodeQuiet(rec, t)) redecode(rec, t);
        else if (rec && giveUpDue(rec, t)) giveUpOn(rec, t);
        lastTick = t; ensureAhead(); return;
      }
      if (holdSince != null) { const ms = t - holdSince; if (ms > 50) { M.stalls.push({ frame: j, ms: +ms.toFixed(0) }); log(`held ${ms.toFixed(0)} ms at frame ${j}`); } holdSince = null; onHold(false); }
      if (lastPainted !== j) { paint(j); lastPainted = j; }
      if (playing) {
        const dt = lastTick == null ? 0 : (t - lastTick) / 1000;
        const next = cur + dt * fps * rate;
        // Never step past a frame that is not decoded on every camera: hold at its boundary.
        const nj = Math.min(Math.floor(next), rangeEnd - 1);
        if (nj !== j) { const r2 = chunkAt(nj); if (!(r2 && frameReady(r2, nj))) { cur = nj; lastTick = t; ensureAhead(); return; } }
        cur = Math.min(next, rangeEnd - 1);
        // The last frame of the range has been painted: wrap to its first, as the Data tab does.
        if (cur >= rangeEnd - 1 && lastPainted === rangeEnd - 1) { cur = rangeStart; M.wraps.push({ t: +now().toFixed(1) }); log(`wrap to frame ${rangeStart}`); }
      }
      lastTick = t;
      ensureAhead();
    }

    /** A decoder that went quiet mid-chunk: throw it away and decode the
     *  bytes again. The page still holds them, so this asks nothing of the
     *  server and gives the browser back whatever the old decoder had. */
    function redecode(rec, t) {
      rec.redecodes++;
      for (const k of cameras) {
        const d = rec.decoders[k];
        if (d && d.state !== 'closed') { try { d.close(); } catch (e) { /* already gone */ } }
        for (const fr of rec.decoded[k] || []) if (fr) { fr.close(); liveFrames--; }
        rec.decoded[k] = null;
        delete rec.done[k];
      }
      rec.lastOutputAt = t;
      rec.decodeDoneAt = null;
      report('decoder-quiet',
        `chunk ${rec.start} decoded ${readiness(rec)} and went quiet; decoding it again `
        + `(${rec.redecodes} of ${MAX_REDECODES})`);
      decodeChunk(rec);
    }

    /** Whether a chunk's decode has stopped without finishing or failing. */
    function decodeQuiet(rec, t) {
      return rec.decodeDoneAt == null
        && rec.redecodes < MAX_REDECODES
        && t - rec.lastOutputAt > decodeQuietMs
        && cameras.some((k) => rec.decoders[k] && !rec.done[k]);
    }

    /** Why a chunk is not showing: how many frames each camera has handed
     *  over, whether its flush settled, and whether its mask rows decoded. A
     *  chunk that is held and never ready says nothing about which of those
     *  stalled, which is what left a 19-second hold on CI unexplained. */
    function readiness(rec) {
      const per = cameras.map((k) => {
        const got = (rec.decoded[k] || []).reduce((n, f) => n + (f ? 1 : 0), 0);  // null while re-decoding
        const d = rec.decoders[k];
        const q = d ? `q=${d.decodeQueueSize} ${d.state}` : 'no decoder';
        return `${k.split('.').pop()}=${got}/${rec.frames}${rec.done[k] ? '' : `(${q})`}`;
      }).join(' ');
      return `${per} masks=${rec.masksPending ? `${rec.masksPending} pending` : 'ready'}`
        + ` decoders=${openDecoders()} frames=${liveFrames}`;
    }

    /** Whether a held chunk has had its chance: its decode settled and the
     *  frame still is not there, or its decode never settled at all. Measuring
     *  from arrival alone gave up on chunks that were still decoding, which a
     *  loaded machine makes common and the skip below makes permanent. */
    function giveUpDue(rec, t) {
      // A decode still handing frames over is slow, not stuck. Dropping it
      // throws the work away and starts it again from the chunk's keyframe,
      // which on a machine slow enough to need the time never converges: CI
      // gave up on a chunk that was two frames of twenty from done, three
      // times over, and the page never showed the frame it was asked for.
      if (t - rec.lastOutputAt <= decodeQuietMs) return false;
      return rec.decodeDoneAt != null
        ? t - rec.decodeDoneAt > holdRetryMs
        : t - rec.arrivedAt > decodeBudgetMs + holdRetryMs;
    }

    /** A held chunk that never became ready: drop it and ask again, a bounded
     *  number of times, saying so where the chunk log is. */
    function giveUpOn(rec, t) {
      const held = Math.round(t - rec.arrivedAt);
      // Before the teardown: closing the chunk closes its decoders, so a
      // reading taken after it says `closed` about every stall.
      const why = readiness(rec);
      const times = (retriesFor.get(rec.start) || 0) + 1;
      retriesFor.set(rec.start, times);
      closeRec(rec); chunks.delete(rec.start);
      if (times > MAX_CHUNK_RETRIES) {
        dead.add(rec.start);
        fail(`chunk ${rec.start} never ready after ${MAX_CHUNK_RETRIES} tries; held ${held} ms without a frame; `
          + `giving up on frames ${rec.start}..${rec.start + rec.frames - 1} [${why}]`);
        onGaveUp(rec.start, rec.frames);
        return;
      }
      M.retries.push({ start: rec.start, ms: held, times, at: Date.now() });
      report('never-ready',
        `chunk ${rec.start} held ${held} ms without a frame [${why}]; `
        + `dropped, asking again (${times} of ${MAX_CHUNK_RETRIES})`);
    }

    // ---------------------------------------------------------------- controls
    function seek(frame) {
      frame = Math.max(0, Math.min(length - 1, Math.round(frame)));
      // Scrubbing into frames the player gave up on is the operator asking for
      // them again -- the only way back to a skipped gap, and the one they
      // reach for without being told about a control.
      const into = chunkFor(frame, n);
      if (dead.delete(into)) { retriesFor.delete(into); log(`seek into given-up chunk ${into}: asking again`); }
      seekPending = { frame, t: performance.now() };
      cur = frame; lastPainted = -1;
      // Keep the transfers the plan from here would ask for again; abort the rest.
      const standing = [...inflight.keys()].map((st) => ({ start: st, end: st + n }));
      const drop = guarded('abort', () => inflightToAbort({ inflight: standing, frame, rangeStart, rangeEnd, keepAhead: BUFFER_TARGET_S * rate * fps }), standing.map((h) => h.start));
      for (const start of drop) { const f = inflight.get(start); if (f) { f.abort.abort(); inflight.delete(start); } }
      ensureAhead();
    }
    function setPlaying(p) { playing = p; lastTick = null; if (p) clampIntoRange(); log(p ? 'play' : 'pause'); }
    function closeAll() {
      abortInflightExcept(-1);
      for (const rec of chunks.values()) closeRec(rec);
      chunks.clear();
      cur = 0; lastPainted = -1; holdSince = null; seekPending = null; lastTick = null;
    }
    function open(ep) {
      closeAll();
      retriesFor.clear();
      dead.clear();
      // The noise is keyed by episode and recipe, so nothing here is ever read
      // again once either moves -- and a texture is a picture's worth of
      // memory per region. Kept, it is a leak that grows with every episode an
      // operator opens.
      textures.clear();
      episode = ep;
      rangeStart = 0; rangeEnd = length;
      log(`episode ${ep}: ${length} frames at ${fps} fps, ${cameras.length} cameras, ${n}-frame chunks`);
      ensureAhead();
    }

    // The transport cannot hang on the compositor. `requestAnimationFrame` is
    // throttled by whatever else is on the page -- CI measured 45 ticks in 30
    // seconds, one and a half a second, while the robot tile's WebGL stalled
    // the compositor on ReadPixels -- and with it went the readiness checks and
    // the fetch plan, so the picture stopped for a reason that had nothing to
    // do with the picture. rAF still drives painting when it runs; a timer
    // keeps the clock honest when it does not. The tick is time-based, so
    // running it from both costs nothing.
    requestAnimationFrame(() => tick(true));
    const beat = setInterval(() => tick(false), Math.max(8, Math.round(1000 / 30)));

    return {
      metrics: M,
      open,
      seek,
      play: () => setPlaying(true),
      pause: () => setPlaying(false),
      playing: () => playing,
      rate: (r) => { if (r != null && r > 0) { rate = r; } return rate; },
      setRange: (s, e) => { rangeStart = Math.max(0, s | 0); rangeEnd = Math.min(length, e | 0) || length; },
      frame: () => Math.floor(cur),
      episode: () => episode,
      ready: () => M.firstPicture != null,
      // The host's mask layer changed what is visible: the frame on screen is repainted from what is held.
      repaintMasks: () => { lastPainted = -1; },
      // The mask layer's version moved (a save, a run or range edit, a treatment
      // change): every held chunk may carry stale rows, and the server's cache was
      // dropped by the same write. Drop the buffer and ask again from the counter.
      masksChanged: () => {
        abortInflightExcept(-1);
        for (const rec of chunks.values()) closeRec(rec);
        chunks.clear();
        // The server rebuilds every chunk of this dataset on a mask write, so
        // a chunk that could not be shown before is worth asking for again.
        dead.clear(); retriesFor.clear();
        textures.clear();   // keyed by the recipe that just changed
        lastPainted = -1; ensureAhead();
      },
      // Give up on the chunk under the frame counter and ask for it again.
      // Ask for everything again from the frame counter, including what the
      // budget gave up on. No control calls this yet -- a scrub into a
      // given-up range is what an operator reaches for -- but the transport
      // tests drive it, and it is the one place that means "start over here".
      retry: () => {
        const st = chunkFor(Math.floor(cur), n);
        const rec = chunks.get(st); if (rec) { closeRec(rec); chunks.delete(st); }
        const inf = inflight.get(st); if (inf) { inf.abort.abort(); inflight.delete(st); }
        dead.clear(); retriesFor.clear();
        lastPainted = -1; ensureAhead();
      },
      // Per held chunk, what it is still missing. `liveFrames` says ten
      // frames are absent; this says which chunk and which camera, without
      // waiting for a give-up to log it -- the case that stalls without ever
      // reaching one is exactly the case that has no line anywhere.
      detail: () => [...chunks.values()].map((rec) => ({ start: rec.start, why: readiness(rec) })),
      state: () => ({ episode, cur: Math.floor(cur), playing, ticks: M.ticks, lastPainted, held: holdSince != null, chunks: [...chunks.keys()], inflight: [...inflight.keys()], dead: [...dead], liveFrames, peakFrames: M.peakFrames, range: [rangeStart, rangeEnd], buffered: bufferedFrames() }),
      close: () => { closed = true; clearInterval(beat); closeAll(); episode = null; },
    };
  }

  window.ChunkPlayer = { create, chunkFor, fetchPlan, chunksToDrop, coverageFrom, inflightToAbort, RuleViolation, CHUNK_SECONDS, MAX_INFLIGHT, BUFFER_TARGET_S, HOLD_RETRY_MS, MAX_CHUNK_RETRIES, DECODE_BUDGET_MS };
})();
