// Saved heatmaps share the live policy renderer and existing overlay tiles.
// Generation guards prevent late loads from painting another episode or seek.
(function (root) {
    'use strict';
    function indexAt(frames, frame) {
        let lo = 0, hi = frames.length;
        while (lo < hi) { const mid = (lo + hi) >>> 1; if (frames[mid] <= frame) lo = mid + 1; else hi = mid; }
        return lo - 1;
    }
    function create({ context, imageFor, report, fetcher = root.fetch.bind(root), makeImage = () => new root.Image() }) {
        let enabled = false, key = '', generation = 0, meta = null, busy = false, checked = 0;
        let style = 'blue_yellow', smooth = 1.2;
        const images = new Map(), tiles = new Set();
        function hide() { for (const tile of tiles) { tile.style.display = 'none'; tile.removeAttribute('src'); } }
        function clear() {
            generation++; busy = false; checked = 0; meta = null;
            for (const entry of images.values()) if (!entry.ready) entry.image.src = '';
            images.clear(); hide(); tiles.clear();
        }
        function base(c) { return `/api/overlays/saved/${encodeURIComponent(c.dataset)}/episode/${c.episode}`; }
        function activeContext() {
            const c = context();
            return enabled && c.dataset && c.episode !== null && c.episode !== undefined ? c : null;
        }
        function metadata(c) {
            if (busy || Date.now() - checked < 5000) return;
            checked = Date.now(); busy = true;
            const token = generation;
            if (!meta) report('读取热图…', 'loading');
            fetcher(base(c), { cache: 'no-store' }).then(r => {
                if (!r.ok) throw new Error('metadata');
                return r.json();
            }).then(data => {
                if (!enabled || token !== generation) return;
                if (meta && meta.revision !== data.revision) { images.clear(); hide(); }
                meta = data;
                report(data.available ? '已保存热图' : (data.message || '暂无热图'), data.available ? 'ok' : 'idle');
                tick();
            }).catch(() => {
                if (enabled && token === generation) { meta = null; hide(); report('热图读取失败', 'error'); }
            }).finally(() => { if (token === generation) busy = false; });
        }
        function request(c, camera, index) {
            const source = meta.frames[index];
            const id = `${meta.revision}:${camera}:${source}:${style}:${smooth}`;
            let entry = images.get(id);
            if (entry && (!entry.failed || Date.now() - entry.failed < 5000)) return entry;
            const image = makeImage();
            entry = { image, ready: false, failed: 0 };
            images.set(id, entry);
            const token = generation;
            image.onload = () => { if (enabled && token === generation) { entry.ready = true; tick(); } };
            image.onerror = () => { if (enabled && token === generation) { entry.failed = Date.now(); report('部分热图读取失败', 'error'); } };
            image.src = `${base(c)}/frame/${source}?camera=${encodeURIComponent(camera)}&style=${encodeURIComponent(style)}&smooth=${smooth}&revision=${meta.revision}`;
            while (images.size > 32) {
                const first = images.keys().next().value, old = images.get(first);
                if (!old.ready) old.image.src = '';
                images.delete(first);
            }
            return entry;
        }
        function tick() {
            const c = activeContext();
            if (!c) { hide(); return; }
            const nextKey = `${c.dataset}\n${c.episode}`;
            if (key !== nextKey) { clear(); key = nextKey; }
            metadata(c);
            if (!meta?.available) { hide(); return; }
            const index = indexAt(meta.frames, c.frame);
            for (const camera of c.cameras || []) {
                const tile = imageFor(camera);
                if (!tile) continue;
                tiles.add(tile);
                if (index < 0 || !meta.cameras[camera] || !c.selected?.has(camera)) { tile.style.display = 'none'; tile.removeAttribute('src'); continue; }
                const entry = request(c, camera, index);
                if (entry.ready) {
                    if (tile.src !== entry.image.src) tile.src = entry.image.src;
                    tile.style.display = 'block';
                    tile.dataset.heatmapFrame = String(meta.frames[index]);
                } else { tile.style.display = 'none'; }
                // At most one prediction ahead; this avoids a network wait at each chunk boundary.
                if (index + 1 < meta.frames.length) request(c, camera, index + 1);
            }
        }
        function configure(options) {
            const nextStyle = options.style || 'blue_yellow', nextSmooth = options.smooth ?? 1.2;
            if (!enabled || style !== nextStyle || smooth !== nextSmooth) clear();
            enabled = true; style = nextStyle; smooth = nextSmooth; tick();
        }
        return { configure, tick, reset: clear, stop() { enabled = false; key = ''; clear(); }, isEnabled: () => enabled };
    }
    const api = { create, indexAt };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.SavedSaliency = api;
})(typeof window !== 'undefined' ? window : globalThis);
