/*
  Windowed playback of a stored episode under one clock: the player the Data
  tab and the standalone measurement page share (docs/dataset_playback.md).

  The player owns the clock, the buffer of decoded windows, the adaptation
  rule and the paint of every camera's frame onto the canvas the host hands it
  per camera. The host owns everything around that: the tiles, the overview,
  the readout, the mask layer, the controls. The contract:

    const player = WindowPlayer.create({
      datasetId,                       // the dataset id in the API
      tiles: (cameraKey) => canvas,    // the canvas a camera's frames paint on, or null
      onOpen: (bundle, episode, prefetched) => {},   // an episode's bundle arrived
      onPaint: (frame, rec, upgrade) => {},  // frame j painted; rec is its window, upgrade = same frame, better rung
      onHold: (waiting) => {},         // the clock is (not) waiting for a window
      onLog: (line) => {},
      masks: () => ({ mode: 'runs' | 'none' | 'composited', mv: '' }),  // per fetch
      enc: { codec, rc, q, preset },
      rung: 'auto' | '160' | ... ,
      probePath: '/',                  // a HEAD of it measures the round trip
    });
    await player.open(episode);

  Every window is fetched by the rule in the design document: length from the
  buffer and the predicted arrival, rung from the measured link rate, one
  step per window, a rescue on a hold, an upgrade of held windows on an idle
  link, the rung and rate remembered per server origin. Playback wraps within
  the range the host sets (the episode by default) and never leaves the
  episode on its own; the next episode's bundle and window 0 are fetched
  ahead for a manual switch.

  player.metrics is the record the harness reads (bundle, windows, painted,
  stalls, seeks, decode, marks, errors, episodes, prefetch, wraps, upgrades,
  memory (the rule's latest), recalled (what this load started from), rttMs,
  rungChanges).
*/
(function () {
  'use strict';

  const MAX_INFLIGHT = 2;      // a slow link is shared by every request in flight; two keeps one arriving while one is asked
  const BUFFER_TARGET_S = 4;   // seconds of media to hold ahead of the clock at 1x; scaled by the rate
  const MARGIN = 1.2;          // a rung must fit the measured rate with this much room before it is chosen
  const UP_VOTES = 2;          // consecutive windows of evidence before a step up; a step down takes one

  // ---------------------------------------------------------------- the rule
  //
  // The ladder's two decisions -- which rung, and how long a window -- as pure
  // functions of what the page has measured. They are separated from the player
  // so they can be driven directly: for a week they could only be exercised by
  // running a browser against an emulated link, and two flaws lived through that
  // (a step judged on the round trip rather than the transfer, and a single
  // window's evidence moving the rung) until an operator saw the picture change
  // sharpness twice a second.
  //
  // Their invariants hold at runtime, not only in tests. A violation is a
  // programming error, so it is raised, recorded and logged -- but playback is
  // not the place to die, so the caller keeps its current choice and plays on.

  class RuleViolation extends Error {}

  function ruleAssert(cond, msg) {
    if (!cond) throw new RuleViolation(msg);
  }

  /**
   * Which rung the next window should ask for, given the one that just arrived.
   *
   * `s` carries only measurements: the ladder in ascending cost (`rungs`) with
   * its nominal `kbps`, the current `rung`, the seconds `buffered` ahead of the
   * clock, the playback `rate`, the averaged `linkMbps` (null while no window
   * was large enough to measure one), `cost(rung)` in bytes per second of media,
   * whether the last window's transfer was `slow`, and the `upVotes` standing.
   *
   * Post: the result names a rung on the ladder, at most one step from the
   * current one, and a step up carries evidence for it.
   */
  function nextRung(s) {
    const { rungs, kbps, rung, buffered, rate, linkMbps, cost, slow, upVotes } = s;
    ruleAssert(Array.isArray(rungs) && rungs.length > 0, 'the ladder is empty');
    // The ladder is ordered by QUALITY, which is the order the server lists it
    // in, and not by cost: `full` is the archive's own samples, and an archive
    // encoded well can cost less per second than a rung transcoded in 100 ms.
    // A step is a step in quality; what it costs is measured, never assumed.
    ruleAssert(new Set(rungs).size === rungs.length, `the ladder repeats a rung: ${rungs.join(', ')}`);
    ruleAssert(
      rungs.every((r) => kbps[r] > 0),
      `a rung has no cost: ${rungs.map((r) => `${r}=${kbps[r]}`).join(' ')}`,
    );
    const i = rungs.indexOf(rung);
    ruleAssert(i >= 0, `${rung} is not a rung of ${rungs.join(', ')}`);
    ruleAssert(rate > 0, `playback rate ${rate}`);

    const b = buffered / rate;
    const need = (r) => cost(r) * 8 / 1e6 * rate;   // Mbit/s this rung costs at this rate
    // A step down takes one window: either the buffer is nearly gone and the
    // link could not carry the last window in the time it plays, or the
    // averaged rate no longer carries this rung with margin.
    const thin = b < 0.5 && slow;
    const overRate = linkMbps != null && need(rung) * MARGIN > linkMbps;
    // A step up needs slack in the buffer and evidence the next rung fits.
    // While no window has been large enough to measure a rate, a buffer at
    // three quarters of its target stands in for it: the link outpaces this
    // rung by some unknown amount, which is worth one step.
    const carries = i < rungs.length - 1 && b >= 1.0
      && (linkMbps != null ? need(rungs[i + 1]) * MARGIN <= linkMbps : b >= 0.75 * BUFFER_TARGET_S);

    let out = { rung, upVotes: carries ? upVotes + 1 : 0, reason: 'hold' };
    if (i > 0 && (thin || overRate)) out = { rung: rungs[i - 1], upVotes: 0, reason: thin ? 'down:thin' : 'down:rate' };
    else if (carries && upVotes + 1 >= UP_VOTES) out = { rung: rungs[i + 1], upVotes: 0, reason: 'up' };

    const j = rungs.indexOf(out.rung);
    ruleAssert(j >= 0, `chose ${out.rung}, which is not on the ladder`);
    ruleAssert(Math.abs(j - i) <= 1, `stepped ${rung} -> ${out.rung}, more than one rung`);
    ruleAssert(
      j <= i || (out.reason === 'up' && carries),
      `stepped up to ${out.rung} without evidence (buffered ${b.toFixed(2)}s, link ${linkMbps})`,
    );
    ruleAssert(j >= i || out.reason.startsWith('down'), `stepped down to ${out.rung} for reason ${out.reason}`);
    ruleAssert(j !== i || out.reason === 'hold', `held ${rung} but reported ${out.reason}`);
    return out;
  }

  /**
   * The frames held from the clock forward, following playback around the wrap.
   *
   * `held` is the ready media as `{start, end, readyTo}` spans in frame
   * numbers, `readyTo` being the first frame of the span that is *not* decoded
   * (equal to `end` when the whole span is ready). The walk stops at the first
   * gap, and stops again if it comes back around to the clock.
   *
   * Post: a frame count in `[0, rangeEnd - rangeStart]`.
   */
  function coverageFrom(s) {
    const { held, rangeStart, rangeEnd } = s;
    ruleAssert(rangeEnd > rangeStart, `empty range ${rangeStart}..${rangeEnd}`);
    // The host may set a range the clock is not in yet -- a trim dragged while
    // paused -- and playback enters it at its start. That is a state, not an
    // error, so the walk starts where playback will.
    const clock = s.clock >= rangeStart && s.clock < rangeEnd ? s.clock : rangeStart;
    const at = (f) => held.find((h) => f >= h.start && f < h.end) || null;
    let f = clock;
    let n = 0;
    let wrapped = false;
    for (let guard = 0; guard <= 2 * held.length + 2; guard++) {
      if (f >= rangeEnd) {
        if (wrapped) break;
        f = rangeStart; wrapped = true;
      }
      if (wrapped && f >= clock) break;   // met the clock from behind: the whole range is held
      const h = at(f);
      if (!h || f >= h.readyTo) break;
      // Past the wrap the walk stops at the clock, not at the end of the
      // range: a window that spans the clock would otherwise be counted twice,
      // and a fully buffered episode reported more media than it holds.
      const stop = wrapped ? clock : rangeEnd;
      const end = Math.min(h.end, h.readyTo, stop);
      n += end - f;
      if (end < h.end || end < h.readyTo) break;   // a gap or a partly decoded window ends the walk
      f = end;
    }
    ruleAssert(n >= 0 && n <= rangeEnd - rangeStart, `covered ${n} of ${rangeEnd - rangeStart} frames`);
    return n;
  }

  /**
   * Which windows to ask for next: `{start, len}` in the order they are needed.
   *
   * Walks forward from the clock over what is `held` or already `inflight`
   * (both `{start, end}` in frames), wrapping once, and asks for a window at
   * each gap until the media covered reaches `targetFrames` or `maxInflight`
   * requests stand.
   *
   * A window starts at the gap's cell on the SHORTEST length's grid, and takes
   * the longest length `allowedIndex(coveredFrames)` permits whose own grid
   * that cell also sits on. Starting instead at the chosen length's own cell
   * would put the window's start before media already held, and the fetch
   * would be dropped as a duplicate: the buffer then stops growing and
   * playback holds at the first gap.
   *
   * Post: every window sits on the grid of its own length, starts at or before
   * the frame it must serve, none repeats one already held or in flight, and
   * no more than `maxInflight` stand at once.
   */
  function fetchPlan(s) {
    const { held, inflight, rangeStart, rangeEnd, fps, targetFrames, maxInflight, lengths, allowedIndex } = s;
    ruleAssert(Array.isArray(lengths) && lengths.length > 0, 'no window lengths');
    ruleAssert(rangeEnd > rangeStart, `empty range ${rangeStart}..${rangeEnd}`);
    ruleAssert(fps > 0, `fps ${fps}`);
    // As above: fetch for where playback will be, not where a stale clock is.
    const clock = s.clock >= rangeStart && s.clock < rangeEnd ? s.clock : rangeStart;
    const spans = held.concat(inflight);
    const at = (f) => spans.find((h) => f >= h.start && f < h.end) || null;
    const out = [];
    let f = clock;
    let covered = 0;
    let wrapped = false;
    let room = maxInflight - inflight.length;
    for (let guard = 0; guard < 16; guard++) {
      if (f >= rangeEnd) {
        if (wrapped) break;
        f = rangeStart; wrapped = true;
      }
      if (wrapped && f >= clock) break;
      const h = at(f);
      if (h) { covered += Math.min(h.end, rangeEnd) - f; f = h.end; continue; }
      if (covered >= targetFrames || room <= 0) break;
      const cell = Math.round(lengths[0] * fps);
      ruleAssert(cell > 0, `the shortest length ${lengths[0]} at ${fps} fps is ${cell} frames`);
      const base = Math.floor(f / cell) * cell;
      let len = lengths[0];
      for (let k = Math.min(allowedIndex(covered), lengths.length - 1); k >= 0; k--) {
        if (base % Math.round(lengths[k] * fps) === 0) { len = lengths[k]; break; }
      }
      const n = Math.round(len * fps);
      const start = base;
      ruleAssert(start % n === 0, `window ${start} is not on the ${len}s grid`);
      ruleAssert(start <= f, `window ${start} starts after the frame ${f} it must serve`);
      out.push({ start, len });
      spans.push({ start, end: start + n });
      covered += Math.min(start + n, rangeEnd) - f;
      f = start + n;
      room--;
    }
    ruleAssert(out.length <= Math.max(0, maxInflight - inflight.length), `planned ${out.length} fetches`);
    ruleAssert(
      new Set(out.map((w) => w.start)).size === out.length,
      `planned the same window twice: ${out.map((w) => w.start).join(', ')}`,
    );
    return out;
  }

  /**
   * Which held windows to release: those neither just behind the clock nor
   * within reach ahead of it, measured around the wrap.
   *
   * Post: never the window the clock is inside.
   */
  function windowsToDrop(s) {
    const { held, clock, rangeStart, rangeEnd, keepBehind, keepAhead } = s;
    ruleAssert(rangeEnd > rangeStart, `empty range ${rangeStart}..${rangeEnd}`);
    const span = Math.max(1, rangeEnd - rangeStart);
    const fwd = (a, b) => (((b - a) % span) + span) % span;   // frames from a forward to b, around the wrap
    const out = [];
    for (const h of held) {
      if (h.start <= clock && clock < h.end) continue;
      if (fwd(h.end, clock) <= keepBehind || fwd(clock, h.start) <= keepAhead) continue;
      out.push(h.start);
    }
    ruleAssert(
      !out.some((start) => held.some((h) => h.start === start && h.start <= clock && clock < h.end)),
      'about to drop the window the clock is inside',
    );
    return out;
  }

  /**
   * How long the next window should be: the index into `lengths`.
   *
   * The longest grid length the buffer allows, and among those the longest that
   * can arrive before the buffer runs out -- its bytes at the current rung's
   * cost over the link rate shared with the requests in flight, plus the round
   * trip and the build.
   *
   * Post: an index into `lengths`; 0 whenever the buffer is nearly empty.
   */
  function nextLength(s) {
    const { lengths, buffered, rate, linkMbps, cost, inflight, rttMs, buildFixed, buildPerSec } = s;
    ruleAssert(Array.isArray(lengths) && lengths.length > 0, 'no window lengths');
    ruleAssert(rate > 0, `playback rate ${rate}`);
    ruleAssert(inflight >= 0, `${inflight} requests in flight`);
    const b = buffered / rate;
    let i = b < 0.4 ? 0 : b < 1.2 ? 1 : b < 2.5 ? 2 : lengths.length - 1;
    i = Math.min(i, lengths.length - 1);
    if (cost != null && linkMbps != null) {
      const share = linkMbps * 1e6 / 8 / (inflight + 1);
      const tx = (len) => (len * rate * cost) / share + rttMs / 1000 + (buildFixed + buildPerSec * len) / 1000;
      while (i > 0 && tx(lengths[i]) >= b) i--;
    }
    ruleAssert(i >= 0 && i < lengths.length, `length index ${i} outside 0..${lengths.length - 1}`);
    ruleAssert(b >= 0.4 || i === 0, `chose ${lengths[i]}s with ${b.toFixed(2)}s buffered`);
    return i;
  }

  function create(opts) {
    const T0 = performance.now();
    const now = () => performance.now() - T0;
    const datasetId = opts.datasetId;
    const tiles = opts.tiles || (() => null);
    const onOpen = opts.onOpen || (() => {});
    const onPaint = opts.onPaint || (() => {});
    const onHold = opts.onHold || (() => {});
    const onLog = opts.onLog || (() => {});
    const masksOf = opts.masks || (() => ({ mode: 'runs', mv: '' }));
    const probePath = opts.probePath || location.pathname;
    const M = { bundle: null, windows: [], firstPicture: null, seeks: [], painted: [], stalls: [], decode: {}, marks: [], errors: [], episodes: [], prefetch: [], wraps: [], upgrades: [], rungChanges: [], memory: null, recalled: null, rttMs: null };
    function log(msg) { onLog(`${(now() / 1000).toFixed(3)}s ${msg}`); }
    if (typeof VideoDecoder === 'undefined') {
      M.errors.push('WebCodecs unavailable (insecure origin?)');
      log('WebCodecs unavailable: serve over https or open on localhost');
    }

    // ---------------------------------------------------------------- state
    let episode = null;
    let bundle = null, fps = 30, length = 0, cameras = [], maskKeyOf = {};
    let autoRung = (opts.rung || 'auto') === 'auto';
    // Remembered per server origin: the last measured link rate and the rung
    // chosen for it. The rule still runs from the first window, so a link that
    // has changed since is stepped down at once; the memory only sets the start.
    const MEM_KEY = `window-playback:link:${location.origin}`;
    // The rate is null on a link too fast to measure (every window arrives
    // within a round trip); the rung is still worth keeping.
    function recall() { try { const m = JSON.parse(localStorage.getItem(MEM_KEY)); return m && typeof m.rung === 'string' ? m : null; } catch (e) { return null; } }
    function remember(mbps) { const m = { rung, mbps: mbps == null ? null : +mbps.toFixed(2), t: Date.now() }; M.memory = m; try { localStorage.setItem(MEM_KEY, JSON.stringify(m)); } catch (e) { /* no storage: the next load starts at the bottom */ } }
    const memory = autoRung ? recall() : null;
    let rung = autoRung ? (memory ? memory.rung : '160') : opts.rung || '640';
    let lastLinkMbps = memory && memory.mbps > 0 ? memory.mbps : null;
    let lastCostBps = null;   // bytes per second of media of the last window, at its rung
    const seenCost = {};      // rung -> bytes per second of media of the last window seen at it
    // Predicted cost per second of media at rung r: measured if r has been seen,
    // otherwise the last window's cost scaled by the ladder's nominal bitrates.
    function costBps(r) {
      if (seenCost[r] != null) return seenCost[r];
      if (lastCostBps == null || !bundle) return null;
      const last = Object.keys(seenCost).find((k) => seenCost[k] === lastCostBps) || r;
      return lastCostBps * (bundle.rung_kbps[r] / bundle.rung_kbps[last]);
    }
    const enc = Object.assign({ codec: 'h264', rc: 'cbr', q: '26', preset: '' }, opts.enc || {});
    const encQuery = () => { const m = masksOf(); return `&codec=${enc.codec}&rc=${enc.rc}&q=${enc.q}${enc.preset ? '&preset=' + enc.preset : ''}&masks=${m.mode}${m.mv ? '&mv=' + m.mv : ''}`; };
    // The browser caches window responses for an hour, so the URL carries what
    // makes a window mean something: the body's format version, so an upgrade
    // is not answered from a window of the older layout, and the dataset's
    // generation, so a trim or a delete is not answered from a window built
    // before it. Window 0 of a fresh open goes out before the bundle names
    // either and is the one request without them.
    const versionQuery = () => (bundle && bundle.format_version != null
      ? `&v=${bundle.format_version}&g=${bundle.generation || 0}`
      : '');
    let rate = 1, playing = false;
    let cur = 0;            // the clock: frame index, fractional while playing
    let rangeStart = 0, rangeEnd = 0;   // playback wraps within [rangeStart, rangeEnd)
    let lastTick = null;    // performance.now() of the last clock advance
    let holdSince = null;   // when the clock started waiting for a window
    let rescued = false;    // whether the current hold already abandoned its in-flight window
    let seekPending = null; // {frame, t} until the first paint at/after it
    let lastPainted = -1;
    // Why the next paint is happening. A quality swap repaints the frame that is
    // already on screen; every other reason is a frame the clock moved to. Set
    // through one function so a reason cannot outlive the repaint it belongs to:
    // a stale `true` here suppresses the playhead of a genuine seek.
    let forUpgrade = false;
    function forceRepaint(upgrade) { lastPainted = -1; forUpgrade = !!upgrade; }
    let rttMs = 250;        // warm-connection round trip, from HEAD probes
    let buildPerSec = 90;   // running estimate of the server's build time per second of window (misses)
    let buildFixed = 60;    // and its fixed part
    let lastUrl = null;     // the last window URL asked for
    let closed = false;
    const windows = new Map();   // start frame -> window record
    const inflight = new Map();  // start frame -> {promise, abort, len}
    const upgrades = new Map();  // start frame -> a window at a higher rung, decoding, to replace the held one
    let bufferedS = 0;           // seconds of decoded media ahead of the clock, measured each tick
    let lengths = [0.5, 1, 2, 4];
    const ctxs = new WeakMap();  // tile canvas -> its 2d context; the host may replace its tiles
    const sizedTiles = new WeakSet();  // canvases whose backing store is already set for this episode
    const base = () => `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episode}`;

    // ---------------------------------------------------------------- windows
    function cellFor(frame, len) { const n = Math.round(len * fps); return Math.floor(frame / n) * n; }

    // The next window after `after`: the longest length allowed by the buffer
    // whose grid cell starts there. A window starts with a keyframe, so a long
    // window is far cheaper per second than a short one (measured at 640 wide
    // with AV1: 348 kB/s for half-second windows, 144 kB/s for four-second
    // ones); the only reason for a short one is having nothing to play while it
    // arrives. So: the more media is buffered, the longer the next window.
    function allowedLength(bufferedNow) {
      return guarded('length', () => nextLength({
        lengths, buffered: bufferedNow == null ? bufferedS : bufferedNow, rate,
        linkMbps: lastLinkMbps, cost: costBps(rung),
        inflight: inflight.size, rttMs, buildFixed, buildPerSec,
      }), 0);
    }
    function parseWindow(buf) {
      const dv = new DataView(buf);
      const hl = dv.getUint32(0, true);
      const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
      const body = 4 + hl;
      const parts = {};
      for (const p of header.parts) parts[`${p.kind}:${p.camera}`] = { ...p, bytes: new Uint8Array(buf, body + p.offset, p.length) };
      return { header, parts };
    }

    async function fetchWindow(start, len, upgrade) {
      if (inflight.has(start) || (windows.has(start) && !upgrade)) return (inflight.get(start) || {}).promise;
      const abort = new AbortController();
      const mine = { promise: null, abort, len };
      // Only this fetch's own record: a rescue replaces the record under the
      // same start frame, and the aborted fetch's cleanup must not remove it.
      const done = () => { if (inflight.get(start) === mine) inflight.delete(start); };
      // Windows sit on the grid of their own length, so they repeat across
      // users and the cache hits; a window off the grid is a client bug the
      // server answers with a 400.
      const cells = Math.round(len * fps);
      if (start % cells !== 0) {
        M.errors.push(`rule (grid): window ${start} is not a multiple of ${cells}`);
        log(`RULE VIOLATION: window ${start} is not on the ${len}s grid`);
      }
      const p = (async () => {
        const t = performance.now();
        const ep = episode;
        const url = `${base()}/window?start=${start}&len=${len}&rung=${rung}${encQuery()}${versionQuery()}`;
        lastUrl = url;
        log(`ask ${start}+${Math.round(len * fps)} ${rung}, buffered ${bufferedS.toFixed(1)}s, ${inflight.size} in flight`);
        let resp, buf;
        try {
          resp = await fetch(url, { signal: abort.signal });
          if (!resp.ok) { M.errors.push(`window ${start}: HTTP ${resp.status}`); log(`window ${start} HTTP ${resp.status}`); done(); return null; }
          buf = await resp.arrayBuffer();
        } catch (e) { done(); if (e.name !== 'AbortError') { M.errors.push(`window ${start}: ${e.message}`); log(`window ${start} failed ${e.message}`); } else log(`window ${start} aborted`); return null; }
        if (ep !== episode || closed) { done(); return null; }   // the host moved on
        const ms = performance.now() - t;
        const { header, parts } = parseWindow(buf);
        const serverMs = +(resp.headers.get('x-window-ms') || 0);
        const rec = { start: header.first_frame, frames: header.frames, len, rung: header.rung, header, parts, decoded: {}, done: {}, closed: false };
        // Throughput excludes the server's time and one round trip, so it is the link's rate, not the request's.
        const xfer = Math.max(50, ms - serverMs - rttMs);
        const w = { start: rec.start, frames: rec.frames, len, rung: rec.rung, bytes: buf.byteLength, ms: +ms.toFixed(1), cache: resp.headers.get('x-window-cache'), serverMs, buildMs: header.build_ms, rateMbps: +(buf.byteLength * 8 / xfer / 1000).toFixed(2), upgrade: !!upgrade, url };
        M.windows.push(w);
        lastCostBps = seenCost[rec.rung] = w.bytes / (rec.frames / fps);
        if (w.cache === 'miss' && serverMs > 0) { const per = Math.max(0, (serverMs - buildFixed) / len); buildPerSec = 0.7 * buildPerSec + 0.3 * per; }
        if (autoRung && w.rateMbps != null) chooseRung(w, rec);
        log(`window ${rec.start}+${rec.frames} ${rec.rung} ${header.enc ? header.enc.codec + '/' + header.enc.rc + (header.enc.rc === 'crf' ? header.enc.q : '') + '/' + header.enc.preset : ''} ${header.masks || ''} ${w.cache} ${(w.bytes / 1000).toFixed(0)} kB in ${w.ms} ms (server ${serverMs} ms) ${w.rateMbps} Mbit/s, buffered ${bufferedS.toFixed(1)}s${autoRung ? ', auto ' + rung : ''}`);
        if (upgrade && windows.has(rec.start)) { rec.upgradeOf = windows.get(rec.start).rung; upgrades.set(rec.start, rec); }
        else windows.set(rec.start, rec);
        done();
        decodeWindow(rec);
        return rec;
      })();
      mine.promise = p;
      inflight.set(start, mine);
      return p;
    }

    // Link rate as a bytes-weighted average of the recent windows large enough
    // to measure: a 40 kB window arriving in one round trip says nothing about
    // the rate, and a single sample sent the rung from 160 to 1280 at once.
    const rateSamples = [];
    let upVotes = 0;   // consecutive windows whose evidence carries the next rung up
    if (memory && memory.mbps > 0) rateSamples.push({ bytes: 60000, mbps: memory.mbps });
    function linkRateMbps(w) {
      const xferMs = w.ms - w.serverMs - rttMs;
      if (xferMs >= 150 && w.bytes >= 60000) rateSamples.push({ bytes: w.bytes, mbps: w.rateMbps });
      while (rateSamples.length > 6) rateSamples.shift();
      if (!rateSamples.length) return null;
      const tot = rateSamples.reduce((a, r) => a + r.bytes, 0);
      lastLinkMbps = rateSamples.reduce((a, r) => a + r.mbps * r.bytes, 0) / tot;
      return lastLinkMbps;
    }
    function rungNames() { return bundle.rungs.filter((r) => bundle.rung_kbps[r] != null); }
    // The rule's own inputs, gathered from what the page has measured.
    function ruleState(extra) {
      return Object.assign(
        {
          rungs: rungNames(),
          kbps: bundle.rung_kbps,
          rung,
          buffered: bufferedS,
          rate,
          linkMbps: lastLinkMbps,
          cost: costBps,
          upVotes,
        },
        extra,
      );
    }
    // A rule violation is a programming error, but playback is not the place to
    // die: record it (every test asserts this list is empty), log it, and keep
    // the current choice.
    function guarded(what, fn, fallback) {
      try {
        return fn();
      } catch (e) {
        if (!(e instanceof RuleViolation)) throw e;
        M.errors.push(`rule (${what}): ${e.message}`);
        log(`RULE VIOLATION in ${what}: ${e.message}`);
        return fallback;
      }
    }
    // Whether the next rung up is carried, for the idle-link upgrade: the same
    // evidence the per-window step uses, asked one window ahead of time.
    function nextRungCarried(names, i) {
      if (i >= names.length - 1) return false;
      const out = guarded('upgrade', () => nextRung(ruleState({ slow: false, upVotes: UP_VOTES - 1 })), null);
      return !!out && names.indexOf(out.rung) > i;
    }
    function chooseRung(w, rec) {
      linkRateMbps(w);   // fold this window into the averaged rate first
      // Slow means the LINK could not carry this window in the time it plays.
      // The round trip and the build are fixed overheads that shrink per second
      // of media as windows grow, and counting them made every half-second
      // window over a 250 ms link "slow", which is what set the rung hunting
      // between 160 and 640 in the reviewer's session.
      const xferMs = Math.max(0, w.ms - w.serverMs - rttMs);
      const slow = xferMs > (rec.frames / fps / rate) * 1000;
      const was = rung;
      const out = guarded('rung', () => nextRung(ruleState({ slow })), null);
      if (out) {
        rung = out.rung;
        upVotes = out.upVotes;
        if (rung !== was) noteRungChange(was, rung, out.reason);
      }
      remember(lastLinkMbps);
    }

    // The rule may move the rung down and up as a link varies; what it must not
    // do is hunt. This is not an invariant -- a link that really does swing
    // makes the picture swing with it -- so it is counted and reported rather
    // than asserted, and a session's log says how often it happened.
    function noteRungChange(from, to, reason) {
      M.rungChanges.push({ from, to, reason, t: +now().toFixed(1), window: M.windows.length });
      const recent = M.rungChanges.filter((c) => M.windows.length - c.window <= 12);
      if (recent.length >= 5 && M.windows.length - (M.lastFlapAt || -99) > 12) {
        M.lastFlapAt = M.windows.length;
        log(`rung changed ${recent.length} times in the last 12 windows: ${recent.map((c) => c.to).join(' ')}`);
      }
    }

    // Drop every fetch that cannot serve the target frame; their bytes would
    // otherwise share the link with the window the operator is waiting for.
    function abortInflightExcept(frame) {
      for (const [start, f] of inflight) {
        const n = Math.round(f.len * fps);
        if (frame < start || frame >= start + n) { f.abort.abort(); inflight.delete(start); }
      }
    }

    function decodeWindow(rec) {
      for (const key of cameras) {
        const part = rec.parts[`video:${key}`];
        if (!part) { M.errors.push(`window ${rec.start}: no video for ${key}`); continue; }
        const frames = new Array(rec.frames).fill(null);
        rec.decoded[key] = frames;
        const stat = M.decode[key] || (M.decode[key] = { frames: 0, ms: 0 });
        const tDec = performance.now();
        const dec = new VideoDecoder({
          output: (frame) => {
            const k = Math.round((frame.timestamp - Math.round(rec.start * 1e6 / fps)) * fps / 1e6);
            if (rec.closed || k < 0 || k >= rec.frames || frames[k]) { frame.close(); return; }
            frames[k] = frame; stat.frames++;
          },
          error: (e) => { M.errors.push(`${key} decode: ${e.message}`); log(`${key} decode error ${e.message}`); },
        });
        dec.configure({ codec: part.codec_string || (part.codec === 'av1' ? 'av01.0.08M.08' : 'avc1.4d401f'), optimizeForLatency: true });
        // Frames are keyed by their presentation time, not by decode order: at
        // the full rung the archive's samples start at the keyframe before the
        // window (negative times, decoded and dropped) and may carry more than
        // one keyframe.
        const base = Math.round(rec.start * 1e6 / fps);
        const keyed = new Set(part.key_frames || [0]);
        let off = 0;
        part.frame_sizes.forEach((size, i) => {
          const data = part.bytes.subarray(off, off + size); off += size;
          const ts = part.frame_ts_us ? part.frame_ts_us[i] : Math.round(i * 1e6 / fps);
          dec.decode(new EncodedVideoChunk({ type: keyed.has(i) ? 'key' : 'delta', timestamp: base + ts, data }));
        });
        dec.flush().then(() => { rec.done[key] = true; stat.ms += performance.now() - tDec; dec.close(); }).catch((e) => { M.errors.push(`${key} flush: ${e.message}`); });
      }
      rec.features = null;
      rec.featuresPending = 1;
      const fpart = rec.parts['features:'];
      if (fpart) {
        new Response(new Blob([fpart.bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
          .then((b) => { rec.features = JSON.parse(new TextDecoder().decode(b)); rec.featuresPending = 0; })
          .catch((e) => { M.errors.push(`features: ${e.message}`); rec.featuresPending = 0; });
      } else rec.featuresPending = 0;
      rec.masks = {};
      rec.maskSize = {};
      rec.masksPending = 0;
      for (const key of cameras) {
        const mk = maskKeyOf[key];
        const part = mk && rec.parts[`masks:${mk}`];
        if (!part) { rec.masks[key] = null; continue; }
        rec.maskSize[key] = part.size || null;
        rec.masksPending++;
        const bytes = part.encoding === 'gzip'
          ? new Response(new Blob([part.bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
          : Promise.resolve(part.bytes.buffer.slice(part.bytes.byteOffset, part.bytes.byteOffset + part.bytes.byteLength));
        bytes.then((b) => { rec.masks[key] = JSON.parse(new TextDecoder().decode(b)); rec.masksPending--; })
          .catch((e) => { M.errors.push(`${key} masks: ${e.message}`); rec.masks[key] = null; rec.masksPending--; });
      }
    }

    function windowFor(frame) { for (const rec of windows.values()) if (frame >= rec.start && frame < rec.start + rec.frames) return rec; return null; }
    function frameReady(rec, j) { return rec.masksPending === 0 && rec.featuresPending === 0 && cameras.every((k) => rec.decoded[k] && rec.decoded[k][j - rec.start]); }
    function closeRec(rec) { rec.closed = true; for (const k of cameras) (rec.decoded[k] || []).forEach((fr) => fr && fr.close()); }

    // Seconds of media decoded and ready from the clock forward, stopping at the
    // first frame that is not; the walk wraps with playback.
    // The ready media as spans, for the walks above: a window counts up to its
    // first frame that is not decoded on every camera.
    function heldSpans() {
      const out = [];
      for (const rec of windows.values()) {
        if (rec.masksPending !== 0 || rec.featuresPending !== 0) continue;
        let ready = 0;
        while (ready < rec.frames && cameras.every((k) => rec.decoded[k] && rec.decoded[k][ready])) ready++;
        out.push({ start: rec.start, end: rec.start + rec.frames, readyTo: rec.start + ready });
      }
      return out;
    }

    function measureBuffered() {
      bufferedS = guarded('coverage', () => coverageFrom({
        held: heldSpans(), clock: Math.floor(cur), rangeStart, rangeEnd,
      }), 0) / fps;
      return bufferedS;
    }

    // Keep fetching, never discarding what is held, until the buffer holds the
    // target seconds of media ahead of the clock; at most MAX_INFLIGHT requests.
    // Playback wraps within the range, so past its end the walk continues from
    // its start, and past the episode's end the next episode is fetched ahead
    // for a manual switch.
    function ensureAhead() {
      if (!bundle || closed) return;
      measureBuffered();
      const targetS = BUFFER_TARGET_S * rate;
      const held = [...windows.values()].map((r) => ({ start: r.start, end: r.start + r.frames }));
      const standing = [...inflight.entries()].map(([st, x]) => ({ start: st, end: st + Math.round(x.len * fps) }));
      const plan = guarded('plan', () => fetchPlan({
        held, inflight: standing, clock: Math.floor(cur), rangeStart, rangeEnd, fps,
        targetFrames: targetS * fps, maxInflight: MAX_INFLIGHT, lengths,
        // The length is chosen from the media already covered at that point in
        // the walk, which is what ties a long window to a full buffer.
        allowedIndex: (coveredFrames) => allowedLength(coveredFrames / fps),
      }), []);
      for (const w of plan) fetchWindow(w.start, w.len);
      // The walk reached the end of the range: the next episode is worth having
      // ready for the switch the operator makes.
      const coveredS = plan.reduce((a, w) => a + w.len, 0) + bufferedS;
      if (!nextEp && rangeEnd >= length && episode + 1 < bundle.episodes && (length - cur) / fps <= targetS) {
        prefetchEpisode(episode + 1);
      }
      // Upgrade: the buffer is at its target and the link is idle. That is the
      // one state the per-window choice never sees, since no window arrives.
      // Step the rung up by the same evidence, then re-fetch the first held
      // window at or after the clock that is below the rung, one at a time; it
      // replaces the held one once decoded. On localhost this is what takes a
      // short episode from the lowest rung to the archive's own samples.
      if (autoRung && inflight.size === 0 && upgrades.size === 0 && coveredS >= Math.min(targetS, (rangeEnd - rangeStart) / fps)) {
        const names = rungNames();
        const i = names.indexOf(rung);
        if (nextRungCarried(names, i)) { rung = names[i + 1]; remember(lastLinkMbps); log(`idle link, buffer full: rung ${rung}`); }
        const rank = names.indexOf(rung);
        let g = Math.floor(cur);
        let w2 = false;
        for (let n = 0; n < 64; n++) {
          const rec = windowFor(g);
          if (!rec) break;
          if (names.indexOf(rec.rung) < rank) { fetchWindow(rec.start, rec.len, true); break; }
          g = rec.start + rec.frames;
          if (g >= rangeEnd) { if (w2) break; g = rangeStart; w2 = true; }
          if (w2 && g >= Math.floor(cur)) break;
        }
      }
      // Release what is neither just behind the clock nor within reach ahead.
      for (const start of guarded('evict', () => windowsToDrop({
        held, clock: Math.floor(cur), rangeStart, rangeEnd,
        keepBehind: 2 * fps, keepAhead: (targetS + 2) * fps,
      }), [])) {
        const rec = windows.get(start);
        if (rec) { closeRec(rec); windows.delete(start); }
      }
    }
    // The grid cell of the shortest length that contains f (where a fresh sequence starts).
    function cellStartAt(f) { return cellFor(f, lengths[0]); }

    // ---------------------------------------------------------------- paint
    function paint(j, upgrade) {
      const rec = windowFor(j);
      const i = j - rec.start;
      for (const key of cameras) {
        const frame = rec.decoded[key][i];
        const canvas = tiles(key);
        if (!canvas) continue;
        let ctx = ctxs.get(canvas);
        if (!ctx) { ctx = canvas.getContext('2d'); ctxs.set(canvas, ctx); }
        // A rung is a bitrate, not a tile size: the canvas keeps the camera's
        // stored resolution and the frame is scaled into it, so the picture
        // changes sharpness when the rung moves and never changes size. Sizing
        // the canvas to the decoded frame made the tiles grow and shrink with
        // every step of the ladder, and moved the mask layer out of register.
        if (!sizedTiles.has(canvas)) sizeTile(canvas, frame.displayWidth, frame.displayHeight);
        ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
      }
      M.painted.push({ frame: j, t: +now().toFixed(1), buffered: +bufferedS.toFixed(2), rung: rec.rung, len: rec.len });
      if (M.firstPicture == null) { M.firstPicture = +now().toFixed(1); log(`first picture, frame ${j}, at ${M.firstPicture} ms`); }
      const epRec = M.episodes[M.episodes.length - 1];
      if (epRec && epRec.firstPicture == null) { epRec.firstPicture = +now().toFixed(1); if (epRec.switchAt != null) { epRec.switchMs = +(performance.now() - epRec.switchAt).toFixed(1); log(`episode ${episode}: first picture ${epRec.switchMs} ms after the switch`); } }
      if (seekPending && j >= seekPending.frame) { const ms = +(performance.now() - seekPending.t).toFixed(1); M.seeks.push({ frame: seekPending.frame, ms }); log(`seek to ${seekPending.frame}: painted ${j} after ${ms} ms`); seekPending = null; }
      onPaint(j, rec, !!upgrade);
    }

    function sizeTile(canvas, w, h) {
      if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
      sizedTiles.add(canvas);
    }

    function windowDecoded(rec) { return rec.masksPending === 0 && rec.featuresPending === 0 && cameras.every((k) => rec.done[k]); }
    function swapUpgrades() {
      for (const [start, rec] of upgrades) {
        if (!windowDecoded(rec)) continue;
        const old = windows.get(start);
        windows.set(start, rec);
        upgrades.delete(start);
        if (old) closeRec(old);
        M.upgrades.push({ start, from: rec.upgradeOf, to: rec.rung, t: +now().toFixed(1) });
        log(`window ${start} now ${rec.rung} (was ${rec.upgradeOf})`);
        forceRepaint(true);   // the frame on screen, from the better window
      }
    }

    function tick() {
      if (closed) return;
      requestAnimationFrame(tick);
      if (!bundle) return;
      if (upgrades.size) swapUpgrades();
      // Before the clock is read: an out-of-range clock whose window is not
      // held takes the hold path below and returns, so a clamp placed after it
      // never runs -- which is the deadlock itself.
      if (playing) clampIntoRange();
      const t = performance.now();
      const j = Math.floor(cur);
      const rec = windowFor(j);
      const ready = rec && frameReady(rec, j);
      if (!ready) {
        if (holdSince == null) { holdSince = t; onHold(true); }
        // Held for over a second on a window still in flight: the link has
        // dropped under what that window needs. Give it up and ask for the
        // shortest window at the lowest rung for this position instead.
        if (t - holdSince > 1000 && !rescued) {
          const inf = [...inflight.entries()].find(([st, x]) => j >= st && j < st + Math.round(x.len * fps));
          if (inf) {
            inf[1].abort.abort(); inflight.delete(inf[0]);
            if (autoRung) { rung = rungNames()[0]; remember(lastLinkMbps); }
            rescued = true;
            log(`held ${(t - holdSince).toFixed(0)} ms on window ${inf[0]} in flight: aborted, refetching at ${rung}`);
            fetchWindow(cellStartAt(j), lengths[0]);
          }
        }
        lastTick = t; ensureAhead(); return;
      }
      if (holdSince != null) { const ms = t - holdSince; if (ms > 50) { M.stalls.push({ frame: j, ms: +ms.toFixed(0) }); log(`held ${ms.toFixed(0)} ms at frame ${j}`); } holdSince = null; rescued = false; onHold(false); }
      if (lastPainted !== j) { paint(j, forUpgrade); lastPainted = j; forUpgrade = false; }
      if (playing) {
        const dt = lastTick == null ? 0 : (t - lastTick) / 1000;
        const next = cur + dt * fps * rate;
        // Never step past a frame that is not decoded yet: hold at its boundary instead.
        const nj = Math.min(Math.floor(next), rangeEnd - 1);
        if (nj !== j) { const r2 = windowFor(nj); if (!(r2 && frameReady(r2, nj))) { cur = nj; lastTick = t; ensureAhead(); return; } }
        cur = Math.min(next, rangeEnd - 1);
        // The last frame of the range has been painted: wrap to its first, as the Data tab does.
        if (cur >= rangeEnd - 1 && lastPainted === rangeEnd - 1) { cur = rangeStart; M.wraps.push({ t: +now().toFixed(1) }); log(`wrap to frame ${rangeStart}`); }
      }
      lastTick = t;
      ensureAhead();
    }

    // ---------------------------------------------------------------- controls
    function seek(frame) {
      if (!bundle) { cur = Math.max(0, Math.round(frame)); return; }
      frame = Math.max(0, Math.min(length - 1, Math.round(frame)));
      seekPending = { frame, t: performance.now() };
      cur = frame; forceRepaint(false); bufferedS = 0;
      abortInflightExcept(frame);
      log(`seek requested to ${frame}`);
      ensureAhead();
    }
    // The fetch planner reads a clock that sits outside the range AS rangeStart,
    // while the paint loop uses it raw. Leave the two disagreeing and the picture
    // waits on a window the planner is never going to ask for: playback stops
    // with no stall recorded and no error, because nothing is wrong -- the frame
    // it wants is simply not coming. Any move of the range has to bring the clock
    // with it, which is why this is one function and not a line in `play`.
    function clampIntoRange() {
      if (bundle && (cur < rangeStart || cur >= rangeEnd)) { cur = rangeStart; forceRepaint(false); }
    }
    // Playback wraps within the range, so while playing the clock has to be
    // inside it. Three things move one or the other -- starting playback, the
    // trim handles moving the range under a running clock, and a seek arriving
    // from an edit or a mask-mode change -- and it only takes one of them to
    // leave the clock outside for the player to deadlock, because the fetch
    // planner reads an out-of-range clock as the range start and stops asking
    // for the window the picture is waiting on. Restated here every tick rather
    // than guarded at each of the three doors, so a fourth door cannot reopen it.
    function setPlaying(p) {
      playing = p; lastTick = null;
      if (p) clampIntoRange();
      log(p ? 'play' : 'pause');
    }

    // ---------------------------------------------------------------- episodes
    const baseFor = (n) => `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${n}`;
    function fetchBundle(n) {
      const t = performance.now();
      return fetch(`${baseFor(n)}/bundle`).then(async (r) => {
        if (!r.ok) throw new Error(`bundle HTTP ${r.status}`);
        const b = await r.json();
        b._ms = +(performance.now() - t).toFixed(1); b._bytes = +(r.headers.get('content-length') || 0);
        return b;
      });
    }
    // Window 0 goes out with the bundle; the fps and camera list are not known
    // yet, so it uses the server's defaults for the shortest window at frame 0.
    function fetchWindow0(n, signal) {
      const url = `${baseFor(n)}/window?start=0&len=0.5&rung=${rung}${encQuery()}${versionQuery()}`;
      lastUrl = url;
      // The one request the browser must not answer from its own cache: it
      // goes out before the bundle names the format version and the dataset's
      // generation, so its URL is the same before and after an edit. Without
      // this an operator trims an episode and the first frame they see is the
      // one that was cut, served from the copy the browser kept.
      return fetch(url, { signal, cache: 'no-store' }).then((r) => {
        if (!r.ok) throw new Error(`window 0 HTTP ${r.status}`);
        return r.arrayBuffer();
      });
    }
    // Window 0 is fetched beside the bundle, outside the fetch walk. Its place
    // is held in the in-flight map for two reasons: the walk would otherwise
    // ask for the same window again the moment the bundle lands (measured on
    // the rig as a second 125 kB request for a window already on the wire),
    // and the rescue needs something it can actually abort -- a first window at
    // a remembered rung the link cannot carry is exactly the case the rescue
    // exists for, and a placeholder it cannot cancel leaves the clock waiting
    // for that window anyway.
    function claimWindow0(ctl) {
      inflight.set(0, { promise: null, abort: ctl, len: 0.5 });
      return () => { if (inflight.get(0) && inflight.get(0).promise === null) inflight.delete(0); };
    }
    let nextEp = null;   // {episode, bundle, w0}: the next episode, fetched ahead
    function prefetchEpisode(n) {
      const ctl = new AbortController();
      nextEp = { episode: n, ctl, bundle: fetchBundle(n), w0: fetchWindow0(n, ctl.signal) };
      nextEp.bundle.catch(() => {}); nextEp.w0.catch(() => {});
      M.prefetch.push({ episode: n, t: +now().toFixed(1) });
      log(`prefetch episode ${n}: bundle and window 0`);
    }
    function closeAll() {
      abortInflightExcept(-1);
      for (const rec of [...windows.values(), ...upgrades.values()]) closeRec(rec);
      windows.clear(); upgrades.clear();
      bundle = null; cur = 0; forceRepaint(false); bufferedS = 0; holdSince = null; rescued = false; seekPending = null; lastTick = null;
    }
    // Leave the current episode for n; `switchAt` (performance.now()) times the
    // first paint of the new episode against the moment of the switch.
    function switchEpisode(n, switchAt) {
      const pre = nextEp && nextEp.episode === n ? nextEp : null;
      nextEp = null;
      closeAll();
      return openEpisode(n, pre, switchAt);
    }

    async function openEpisode(n, pre, switchAt) {
      episode = n;
      const epRec = { episode: n, t: +now().toFixed(1), firstPicture: null, switchAt: switchAt == null ? null : switchAt, switchMs: null };
      M.episodes.push(epRec);
      const tB = performance.now();
      const bundleP = pre ? pre.bundle : fetchBundle(n);
      const w0ctl = pre ? pre.ctl : new AbortController();
      const w0 = pre ? pre.w0 : fetchWindow0(n, w0ctl.signal);
      const releaseWindow0 = claimWindow0(w0ctl);
      let b;
      try { b = await bundleP; } catch (e) { releaseWindow0(); M.errors.push(`episode ${n}: ${e.message}`); log(`episode ${n} ${e.message}`); return; }
      if (episode !== n || closed) { releaseWindow0(); return; }   // switched again while the bundle was in flight
      if (M.bundle == null) M.bundle = { ms: b._ms, bytes: b._bytes };
      bundle = b;
      fps = bundle.fps; length = bundle.length; cameras = Object.keys(bundle.cameras); lengths = bundle.window_lengths;
      rangeStart = 0; rangeEnd = length;
      maskKeyOf = {};
      for (const key of cameras) { const short = key.split('.').pop(); for (const mk of Object.keys(bundle.masks)) if (mk.split('.').pop() === short) maskKeyOf[key] = mk; }
      log(`episode ${n} bundle ${b._ms} ms${pre ? ' (prefetched)' : ''}: ${length} frames at ${fps} fps, ${cameras.length} cameras, masks for ${Object.keys(maskKeyOf).length}, ${bundle.episodes} episodes`);
      const names = rungNames();
      if (autoRung && !names.includes(rung)) rung = names[0];
      onOpen(bundle, n, !!pre);
      // The host's tiles exist by now (it built them in onOpen, or they were
      // already there): give each the camera's stored resolution.
      for (const key of cameras) {
        const canvas = tiles(key);
        const cam = bundle.cameras[key] || {};
        if (canvas && cam.width && cam.height) sizeTile(canvas, cam.width, cam.height);
      }
      try {
        const buf = await w0;
        releaseWindow0();
        if (episode !== n || closed) return;
        const { header, parts } = parseWindow(buf);
        const rec = { start: 0, frames: header.frames, len: 0.5, rung: header.rung, header, parts, decoded: {}, done: {}, closed: false };
        M.windows.push({ start: 0, frames: header.frames, len: 0.5, rung: header.rung, bytes: buf.byteLength, ms: +(performance.now() - tB).toFixed(1), cache: pre ? 'prefetched' : 'first', serverMs: 0, buildMs: header.build_ms, rateMbps: null, url: lastUrl });
        windows.set(0, rec); decodeWindow(rec);
        log(`window 0 (with bundle) ${(buf.byteLength / 1000).toFixed(0)} kB at ${(performance.now() - tB).toFixed(0)} ms, rung ${header.rung}`);
      } catch (e) {
        releaseWindow0();
        // The rescue gave this window up: the shortest window at the lowest
        // rung is already on its way in its place.
        if (e.name === 'AbortError') log('window 0 aborted');
        else { M.errors.push('window 0: ' + e.message); log('window 0 failed ' + e.message); }
      }
    }

    // Round trip on a warm connection: a HEAD of the host's page, repeated while
    // playing. The bundle's own time includes the TLS handshake and is no measure of it.
    const probeRtt = async () => { const t = performance.now(); try { await fetch(probePath, { method: 'HEAD', cache: 'no-store' }); const v = performance.now() - t; rttMs = rttMs == null ? v : Math.min(rttMs, v); M.rttMs = +rttMs.toFixed(0); } catch (e) { /* keep the last value */ } };
    const probeTimer = setTimeout(() => { probeRtt(); probeInterval = setInterval(probeRtt, 10000); }, 500);
    let probeInterval = null;
    // What this load started from, kept apart from `memory`, which follows the
    // rule and is overwritten within the first second.
    M.recalled = memory;
    if (memory) { log(`remembered for ${location.origin}: rung ${memory.rung}${memory.mbps > 0 ? ' at ' + memory.mbps + ' Mbit/s' : ''}`); M.memory = memory; }
    requestAnimationFrame(tick);

    return {
      metrics: M,
      open: (n) => openEpisode(n, null, null),
      switchEpisode: (n) => switchEpisode(n, performance.now()),
      seek,
      play: () => setPlaying(true),
      pause: () => setPlaying(false),
      playing: () => playing,
      rate: (r) => { if (r != null) rate = r; return rate; },
      rung: (r) => { if (r === 'auto') autoRung = true; else if (r != null) { autoRung = false; rung = r; } return autoRung ? 'auto' : rung; },
      currentRung: () => rung,
      encoder: (o) => { Object.assign(enc, o); },
      // The host's mask mode changed (composited on or off, or the recipe's
      // version): every held window carries the wrong pixels. Drop them and
      // fetch again from the clock; the bundle stays.
      masksChanged: () => {
        if (!bundle) return;
        abortInflightExcept(-1);
        for (const rec of [...windows.values(), ...upgrades.values()]) closeRec(rec);
        windows.clear(); upgrades.clear();
        forceRepaint(false); bufferedS = 0;
        log('masks changed: windows dropped');
        ensureAhead();
      },
      setRange: (s, e) => {
        rangeStart = Math.max(0, s | 0);
        rangeEnd = Math.min(length || e, e | 0) || length;
      },
      mark: (phase) => { M.marks.push({ phase, t: +now().toFixed(1) }); log(`phase ${phase}`); },
      ready: () => M.firstPicture != null,
      episode: () => episode,
      bundle: () => bundle,
      frame: () => Math.floor(cur),
      windowFor,
      lastWindowUrl: () => lastUrl,
      abortAll: () => abortInflightExcept(-1),
      state: () => ({ episode, cur: Math.floor(cur), playing, rung, windows: [...windows.keys()], inflight: [...inflight.keys()], nextEp: nextEp ? nextEp.episode : null, range: [rangeStart, rangeEnd] }),
      close: () => { closed = true; clearTimeout(probeTimer); if (probeInterval) clearInterval(probeInterval); closeAll(); },
    };
  }

  // The rule is exported so it can be driven directly, without a server, a
  // video or a link: tests/gui/test_window_rule.py enumerates its decisions.
  window.WindowPlayer = {
    create, MAX_INFLIGHT, BUFFER_TARGET_S, MARGIN, UP_VOTES,
    nextRung, nextLength, coverageFrom, fetchPlan, windowsToDrop, RuleViolation,
  };
})();
