// Feature Editing — per-frame view + edit (Phase B).
//
// Owns the right-hand Inspector panel, the per-feature timeline rows under
// the existing scrubber, the vertical-slice selection model, and the
// click-to-seek-and-select gesture. State / DOM lookups talk to the existing
// app.js globals (datasets, episodes, currentDataset, currentEpisode,
// currentFrame, pendingEdits, fps, totalFrames, trimStart, trimEnd, etc.).

(function () {
    "use strict";

    // ── Per-dataset / per-episode caches ─────────────────────────────────
    const seriesCache = new Map(); // key = `${datasetId}:${episodeIdx}` → {length, series}
    const featureRowState = new Map(); // featureName → {pinned, expanded}

    // Selection: {episodeIndex, frameFrom, frameTo, originRow}
    let selection = null;
    let showPendingEdits = false;

    // Dragging state for selection on a feature row.
    let dragState = null; // {anchorFrame, originRow}

    // ── Public API exposed on window for app.js wiring ───────────────────

    window.FeatureEditing = {
        onDatasetOpened,
        onDatasetClosed,
        onEpisodeSelected,
        onPlayheadChanged,
        onPendingEditsChanged,
        clearSelection,
    };

    // ── Hooks called from app.js ─────────────────────────────────────────

    function onDatasetOpened(datasetId) {
        renderInspectorEmpty(datasetId);
    }

    function onDatasetClosed(datasetId) {
        // Drop cached series for this dataset.
        for (const key of Array.from(seriesCache.keys())) {
            if (key.startsWith(`${datasetId}:`)) seriesCache.delete(key);
        }
        if (selection && selection.datasetId === datasetId) selection = null;
        renderInspectorEmpty(null);
        renderFeatureRows();
    }

    function onEpisodeSelected(datasetId, episodeIdx) {
        // Reset selection on episode switch — selection is per-episode.
        selection = null;
        renderInspector();
        loadFeatureSeries(datasetId, episodeIdx).then(() => {
            renderFeatureRows();
        }).catch((err) => console.warn("feature-series load failed", err));
    }

    function onPlayheadChanged() {
        // Update the per-frame value display in Inspector when no selection.
        if (!selection) renderInspectorFrameValues();
    }

    function onPendingEditsChanged() {
        renderFeatureRows(); // pending overlay changes
        renderInspector(); // pending indicator on cards
    }

    function clearSelection() {
        if (!selection) return;
        selection = null;
        renderInspector();
        renderFeatureRows();
    }

    // ── Series fetch ─────────────────────────────────────────────────────

    async function loadFeatureSeries(datasetId, episodeIdx) {
        const key = `${datasetId}:${episodeIdx}`;
        if (seriesCache.has(key)) return seriesCache.get(key);
        const url = `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIdx}/feature-series`;
        const res = await fetch(url);
        if (!res.ok) {
            console.warn(`feature-series failed: ${res.status}`);
            return null;
        }
        const data = await res.json();
        seriesCache.set(key, data);
        return data;
    }

    // ── Editability classification ──────────────────────────────────────

    const DEFAULT_FEATURES = new Set([
        "timestamp", "frame_index", "episode_index", "index", "task_index",
    ]);
    const READONLY_DTYPES = new Set(["image", "video"]);

    function isEditable(name, ft) {
        if (!ft) return false;
        if (DEFAULT_FEATURES.has(name)) return false;
        if (READONLY_DTYPES.has(ft.dtype)) return false;
        if (name === "action") return false;
        if (name.startsWith("observation.")) return false;
        return true;
    }

    function isHiddenByDefault(name, ft) {
        // Hidden by default per the design: image/video, defaults, action / observation.*
        if (DEFAULT_FEATURES.has(name)) return true;
        if (READONLY_DTYPES.has(ft.dtype)) return true;
        if (name === "action") return true;
        if (name.startsWith("observation.")) return true;
        return false;
    }

    function getEpisodeStartFrame(datasetId, episodeIdx) {
        // Episode is per-episode; we map back to local frame indices for staging.
        // (Selection's frameFrom / frameTo are LOCAL to the episode.)
        return 0;
    }

    function getActiveTrim(datasetId, episodeIdx, episodeLength) {
        // Mirrors app.js trimStart / trimEnd — those are populated for the
        // currently-playing episode. If we're inspecting another episode, use
        // its full range as the envelope.
        if (window.currentDataset === datasetId && window.currentEpisode === episodeIdx) {
            const ts = (typeof window.trimStart === "number") ? window.trimStart : 0;
            const te = (typeof window.trimEnd === "number") ? window.trimEnd : episodeLength;
            return [Math.max(0, Math.min(ts, episodeLength)), Math.max(ts, Math.min(te, episodeLength))];
        }
        return [0, episodeLength];
    }

    // ── Inspector rendering ──────────────────────────────────────────────

    function renderInspectorEmpty(datasetId) {
        const body = document.getElementById("inspector-body");
        if (!body) return;
        if (!datasetId || !window.datasets || !window.datasets[datasetId]) {
            body.innerHTML = '<div class="empty-state">Open a dataset to inspect its features</div>';
            return;
        }
        const ds = window.datasets[datasetId];
        body.innerHTML = `
            <div class="inspector-summary">
                <div><span class="summary-key">repo_id:</span> <span class="summary-value">${escapeHtml(ds.repo_id || ds.id)}</span></div>
                <div><span class="summary-key">episodes:</span> <span class="summary-value">${ds.total_episodes ?? "?"}</span></div>
                <div><span class="summary-key">frames:</span> <span class="summary-value">${ds.total_frames ?? "?"}</span></div>
                <div><span class="summary-key">fps:</span> <span class="summary-value">${ds.fps ?? "?"}</span></div>
                <div><span class="summary-key">robot:</span> <span class="summary-value">${escapeHtml(ds.robot_type || "—")}</span></div>
                <div style="margin-top:10px; color:#888; font-style:italic;">Click or drag inside the timeline area to edit feature values.</div>
            </div>
        `;
    }

    function renderInspector() {
        if (!selection) {
            if (window.currentDataset) {
                renderInspectorEmpty(window.currentDataset);
                renderInspectorFrameValues();
            }
            return;
        }

        const datasetId = window.currentDataset;
        const ds = window.datasets && window.datasets[datasetId];
        if (!ds) return;

        const body = document.getElementById("inspector-body");
        if (!body) return;

        const featuresSchema = ds.features_schema || {};
        const epLen = (window.totalFrames > 0) ? window.totalFrames : 0;
        const from = selection.frameFrom;
        const to = selection.frameTo;
        const k = to - from;
        const m = to - 1;

        // Compute global indices for staging (frame index ↔ global index mapping).
        // We rely on the backend's _get_episode_start_index by sending local
        // frame_from / frame_to; the backend computes the global offset.

        const cards = [];
        for (const [name, ft] of Object.entries(featuresSchema)) {
            cards.push(renderFeatureCard(name, ft, from, to, datasetId, selection.episodeIndex, selection.originRow));
        }

        body.innerHTML = `
            <div class="inspector-selection-header">
                <div class="sel-title">frames ${from}…${m} (${k} frame${k === 1 ? "" : "s"})</div>
                <div class="sel-meta">ep ${selection.episodeIndex} · drag-select on any row</div>
            </div>
            ${cards.join("")}
        `;

        // Wire edit widgets (auto-staging on change).
        wireWidgets(body);
    }

    function renderInspectorFrameValues() {
        // Show per-frame values in the Inspector when there's no selection.
        if (selection) return;
        const datasetId = window.currentDataset;
        const ds = window.datasets && window.datasets[datasetId];
        const epIdx = window.currentEpisode;
        const frame = window.currentFrame;
        const body = document.getElementById("inspector-body");
        if (!ds || epIdx == null || !body) return;

        const url = `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${epIdx}/frame/${frame}/features`;
        fetch(url).then(r => r.ok ? r.json() : null).then(data => {
            if (!data || selection) return; // selection raced in; abort
            const featuresSchema = ds.features_schema || {};
            const cards = [];
            cards.push(`
                <div class="inspector-summary" style="margin-bottom:12px;">
                    <div><span class="summary-key">frame:</span> <span class="summary-value">${frame} / ${window.totalFrames}</span> · <span class="summary-key">ep:</span> <span class="summary-value">${epIdx}</span></div>
                </div>
            `);
            for (const [name, ft] of Object.entries(featuresSchema)) {
                if (isHiddenByDefault(name, ft) && !featureRowState.get(name)?.pinned) continue;
                const val = data.values[name];
                cards.push(renderReadonlyValueCard(name, ft, val));
            }
            body.innerHTML = cards.join("");
        }).catch(err => console.warn("frame features fetch failed", err));
    }

    function renderReadonlyValueCard(name, ft, value) {
        const dtype = ft.dtype || "?";
        const shape = (ft.shape || []).join("×") || "1";
        return `
            <div class="feature-card readonly">
                <div class="card-header">
                    <span class="card-name">${escapeHtml(name)}</span>
                    <span class="card-dtype">${escapeHtml(dtype)}[${shape}]</span>
                </div>
                <div class="card-summary">${formatValueShort(value)}</div>
            </div>
        `;
    }

    function renderFeatureCard(name, ft, frameFrom, frameTo, datasetId, episodeIndex, originRow) {
        const editable = isEditable(name, ft);
        const focused = (originRow === name);
        const dtype = ft.dtype || "?";
        const shape = (ft.shape || []).join("×") || "1";
        const pendingEdit = findPendingFeatureEdit(datasetId, episodeIndex, name, frameFrom, frameTo);

        const headerExtras = pendingEdit
            ? `<span class="card-pending">● pending</span>`
            : "";

        let widget = "";
        if (!editable) {
            widget = `<span class="card-readonly-tag">read-only in V1</span>`;
        } else {
            widget = renderWidgetForType(name, ft, frameFrom, frameTo, datasetId, episodeIndex);
        }

        return `
            <div class="feature-card ${focused ? "focused" : ""} ${editable ? "" : "readonly"}" data-feature="${escapeHtml(name)}">
                <div class="card-header">
                    <span class="card-name">${escapeHtml(name)}${headerExtras}</span>
                    <span class="card-dtype">${escapeHtml(dtype)}[${shape}]</span>
                </div>
                <div class="card-summary">${cardSummary(name, ft, datasetId, episodeIndex, frameFrom, frameTo)}</div>
                <div class="card-widget">${widget}</div>
            </div>
        `;
    }

    function cardSummary(name, ft, datasetId, episodeIndex, frameFrom, frameTo) {
        const key = `${datasetId}:${episodeIndex}`;
        const cached = seriesCache.get(key);
        if (!cached || !cached.series || !cached.series[name]) return "&nbsp;";
        const slice = cached.series[name].slice(frameFrom, frameTo);
        if (!slice.length) return "&nbsp;";
        if (typeof slice[0] === "number") {
            const min = Math.min(...slice.filter(v => typeof v === "number"));
            const max = Math.max(...slice.filter(v => typeof v === "number"));
            return `range: ${formatNumber(min)} … ${formatNumber(max)}`;
        }
        if (typeof slice[0] === "boolean") {
            const t = slice.filter(v => v === true).length;
            const f = slice.length - t;
            return `${t} true · ${f} false`;
        }
        if (typeof slice[0] === "string") {
            const unique = new Set(slice);
            if (unique.size === 1) return `uniform: "${escapeHtml(slice[0])}"`;
            return `${unique.size} unique values`;
        }
        return "&nbsp;";
    }

    function renderWidgetForType(name, ft, frameFrom, frameTo, datasetId, episodeIndex) {
        const dtype = ft.dtype || "";
        const shape = ft.shape || [];
        const isScalar = (shape.length === 0) || (shape.length === 1 && shape[0] === 1);

        if (dtype === "bool" && isScalar) {
            return `<input type="checkbox" data-widget="bool" data-feature="${escapeHtml(name)}">`;
        }
        if (dtype === "string") {
            return `<input type="text" data-widget="string" data-feature="${escapeHtml(name)}" placeholder="(value for range)">`;
        }
        if (isScalar && (dtype.startsWith("int") || dtype.startsWith("float"))) {
            // slider + number input. Slider range derived from the loaded series, if available.
            const key = `${datasetId}:${episodeIndex}`;
            const cached = seriesCache.get(key);
            let lo = -1, hi = 1;
            if (cached && cached.series && cached.series[name]) {
                const all = cached.series[name].filter(v => typeof v === "number");
                if (all.length) {
                    lo = Math.min(...all);
                    hi = Math.max(...all);
                    if (lo === hi) { lo -= 1; hi += 1; }
                }
            }
            const step = (dtype.startsWith("int")) ? "1" : "any";
            return `
                <input type="range" data-widget="scalar-slider" data-feature="${escapeHtml(name)}" min="${lo}" max="${hi}" step="${step}">
                <input type="number" data-widget="scalar-number" data-feature="${escapeHtml(name)}" step="${step}">
            `;
        }
        if (shape.length === 1 && shape[0] > 0 && shape[0] <= 8) {
            // Small numeric vector → row of inputs.
            const inputs = [];
            for (let i = 0; i < shape[0]; i++) {
                inputs.push(`<input type="number" data-widget="vector-cell" data-feature="${escapeHtml(name)}" data-cell="${i}" step="any">`);
            }
            return `<div class="vector-row">${inputs.join("")}</div>`;
        }
        // Large vector / matrix → JSON textarea.
        return `<textarea data-widget="json" data-feature="${escapeHtml(name)}" placeholder="JSON value (matches dtype/shape)"></textarea>`;
    }

    // ── Edit-widget wiring (auto-staging on change) ─────────────────────

    function wireWidgets(root) {
        const cards = root.querySelectorAll(".feature-card[data-feature]");
        cards.forEach(card => {
            const featureName = card.getAttribute("data-feature");
            const widgets = card.querySelectorAll("[data-widget]");

            // Slider <-> number sync for scalar-slider/scalar-number pair.
            const slider = card.querySelector('[data-widget="scalar-slider"]');
            const numInput = card.querySelector('[data-widget="scalar-number"]');
            if (slider && numInput) {
                slider.addEventListener("input", () => {
                    numInput.value = slider.value;
                });
                slider.addEventListener("change", () => {
                    stageFeatureEdit(featureName, parseFloat(slider.value));
                });
                numInput.addEventListener("change", () => {
                    if (numInput.value === "") return;
                    slider.value = numInput.value;
                    stageFeatureEdit(featureName, parseFloat(numInput.value));
                });
            }

            widgets.forEach(w => {
                const kind = w.getAttribute("data-widget");
                if (kind === "bool") {
                    w.addEventListener("change", () => stageFeatureEdit(featureName, w.checked));
                } else if (kind === "string") {
                    w.addEventListener("change", () => stageFeatureEdit(featureName, w.value));
                } else if (kind === "vector-cell") {
                    // Stage when *any* cell changes — collect all cells into the vector.
                    w.addEventListener("change", () => {
                        const cells = card.querySelectorAll('[data-widget="vector-cell"]');
                        const vec = [];
                        cells.forEach(c => {
                            const v = c.value === "" ? 0 : parseFloat(c.value);
                            vec.push(v);
                        });
                        stageFeatureEdit(featureName, vec);
                    });
                } else if (kind === "json") {
                    w.addEventListener("change", () => {
                        try {
                            const parsed = JSON.parse(w.value);
                            stageFeatureEdit(featureName, parsed);
                        } catch (e) {
                            window.setStatus && window.setStatus("Invalid JSON: " + e.message);
                        }
                    });
                }
                // scalar-slider / scalar-number handled above via slider/numInput pair.
            });
        });
    }

    async function stageFeatureEdit(featureName, value) {
        if (!selection) return;
        const datasetId = window.currentDataset;
        const body = {
            dataset_id: datasetId,
            episode_index: selection.episodeIndex,
            feature: featureName,
            frame_from: selection.frameFrom,
            frame_to: selection.frameTo,
            value: value,
        };
        try {
            const res = await fetch("/api/edits/feature-set", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.text();
                window.setStatus && window.setStatus(`Edit rejected: ${err}`);
                return;
            }
            // Refresh pending edits list (app.js owns the global state).
            if (typeof window.refreshPendingEdits === "function") {
                await window.refreshPendingEdits();
            }
            // Re-render Inspector to show the ● pending indicator.
            renderInspector();
            renderFeatureRows();
        } catch (e) {
            console.error("stageFeatureEdit failed", e);
        }
    }

    function findPendingFeatureEdit(datasetId, episodeIndex, feature, frameFrom, frameTo) {
        const edits = (window.pendingEdits || []).filter(e =>
            e.dataset_id === datasetId &&
            e.episode_index === episodeIndex &&
            e.edit_type === "feature_set" &&
            e.params &&
            e.params.feature === feature &&
            e.params.frame_from === frameFrom &&
            e.params.frame_to === frameTo
        );
        return edits[0] || null;
    }

    // ── Feature-row rendering ───────────────────────────────────────────

    function renderFeatureRows() {
        const container = document.getElementById("feature-rows");
        if (!container) return;
        const datasetId = window.currentDataset;
        const epIdx = window.currentEpisode;
        if (!datasetId || epIdx == null || epIdx === undefined) {
            container.innerHTML = "";
            return;
        }
        const ds = window.datasets && window.datasets[datasetId];
        if (!ds || !ds.features_schema) {
            container.innerHTML = "";
            return;
        }
        const key = `${datasetId}:${epIdx}`;
        const cached = seriesCache.get(key);
        if (!cached) {
            container.innerHTML = '<div class="feature-rows-empty">Loading feature series…</div>';
            return;
        }

        const visibleFeatures = [];
        for (const [name, ft] of Object.entries(ds.features_schema)) {
            const state = featureRowState.get(name) || {};
            const hidden = isHiddenByDefault(name, ft) && !state.pinned;
            if (!hidden) visibleFeatures.push([name, ft]);
        }
        if (!visibleFeatures.length) {
            container.innerHTML = '<div class="feature-rows-empty">No editable features visible. (action / observation.* hidden by default.)</div>';
            return;
        }

        const rows = visibleFeatures.map(([name, ft]) => renderFeatureRow(name, ft, cached));
        container.innerHTML = rows.join("");

        // Wire mouse handlers on each row's track.
        container.querySelectorAll(".row-track").forEach(track => {
            wireFeatureRowTrack(track);
        });
    }

    function renderFeatureRow(name, ft, cached) {
        const dtype = ft.dtype || "?";
        const shape = (ft.shape || []).join("×") || "1";
        const editable = isEditable(name, ft);
        const series = cached.series[name] || [];
        const length = cached.length;
        const trackContent = renderTrackSvg(name, ft, series, length);
        const [trimFrom, trimTo] = getActiveTrim(window.currentDataset, window.currentEpisode, length);

        const dimLeftPct = (trimFrom / length) * 100;
        const dimRightPct = (1 - trimTo / length) * 100;

        const overlays = [];
        // Trim envelope dim overlays (left + right).
        if (dimLeftPct > 0) {
            overlays.push(`<div class="row-trim-dim" style="left:0; width:${dimLeftPct}%;"></div>`);
        }
        if (dimRightPct > 0) {
            overlays.push(`<div class="row-trim-dim" style="right:0; width:${dimRightPct}%;"></div>`);
        }
        // Selection band — vertical slice; rendered on every row at the same x range.
        if (selection && selection.episodeIndex === window.currentEpisode) {
            const left = (selection.frameFrom / length) * 100;
            const width = ((selection.frameTo - selection.frameFrom) / length) * 100;
            overlays.push(`<div class="row-selection" style="left:${left}%; width:${width}%;"></div>`);
        }
        // "Show pending edits" overlay — paint each pending feature_set edit for this feature.
        if (showPendingEdits && editable) {
            const pendingForFeature = (window.pendingEdits || []).filter(e =>
                e.dataset_id === window.currentDataset &&
                e.episode_index === window.currentEpisode &&
                e.edit_type === "feature_set" &&
                e.params && e.params.feature === name
            );
            for (const e of pendingForFeature) {
                const left = (e.params.frame_from / length) * 100;
                const width = ((e.params.frame_to - e.params.frame_from) / length) * 100;
                overlays.push(`<div class="row-pending-overlay" style="left:${left}%; width:${width}%;"></div>`);
            }
        }

        const readonlyTag = editable ? "" : `<div class="row-readonly">read-only</div>`;
        const rowClass = editable ? "feature-row" : "feature-row readonly";

        return `
            <div class="${rowClass}" data-feature="${escapeHtml(name)}">
                <div class="row-label">
                    <div class="row-name">${escapeHtml(name)}</div>
                    <div class="row-dtype">${escapeHtml(dtype)}[${shape}]</div>
                    ${readonlyTag}
                </div>
                <div class="row-track" data-feature="${escapeHtml(name)}" data-length="${length}">
                    ${trackContent}
                    ${overlays.join("")}
                </div>
            </div>
        `;
    }

    function renderTrackSvg(name, ft, series, length) {
        if (!series || !series.length) return "";
        const dtype = ft.dtype || "";
        const shape = ft.shape || [];

        if (dtype === "bool" && (shape.length === 0 || (shape.length === 1 && shape[0] === 1))) {
            // band: green where true, light-grey where false.
            const segs = [];
            for (let i = 0; i < series.length; i++) {
                const x = (i / length) * 100;
                const w = (1 / length) * 100 + 0.05; // tiny overdraw to avoid gaps
                if (series[i] === true) {
                    segs.push(`<rect x="${x}%" y="20%" width="${w}%" height="60%" fill="#27ae60"/>`);
                }
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${segs.join("")}</svg>`;
        }

        if (dtype === "string") {
            // colored stripe — each unique string gets a color; render run-length segments.
            const colors = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085"];
            const colorMap = new Map();
            const segs = [];
            let i = 0;
            while (i < series.length) {
                const v = series[i];
                let j = i;
                while (j < series.length && series[j] === v) j++;
                if (!colorMap.has(v)) colorMap.set(v, colors[colorMap.size % colors.length]);
                const color = colorMap.get(v);
                const x = (i / length) * 100;
                const w = ((j - i) / length) * 100;
                segs.push(`<rect x="${x}%" y="10%" width="${w}%" height="80%" fill="${color}" opacity="0.7"/>`);
                if (j - i > 4) { // only label long-enough segments
                    segs.push(`<text x="${x + w / 2}%" y="55%" font-size="9" fill="#fff" text-anchor="middle" dominant-baseline="middle" pointer-events="none">${escapeHtml(String(v).slice(0, 10))}</text>`);
                }
                i = j;
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${segs.join("")}</svg>`;
        }

        // Numeric: scalar → line; vector → mini multi-line (up to 8); large → single line of norms.
        const scalarSeries = (typeof series[0] === "number") ? series : null;
        if (scalarSeries) {
            return numericLineSvg(scalarSeries, length);
        }
        if (Array.isArray(series[0]) && series[0].length <= 8) {
            // Mini multi-line: overlay up to 8 series.
            const dims = series[0].length;
            const colors = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085", "#e74c3c", "#7f8c8d"];
            const lines = [];
            for (let d = 0; d < dims; d++) {
                const dim = series.map(row => row[d]);
                lines.push(numericLinePath(dim, length, colors[d % colors.length]));
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${lines.join("")}</svg>`;
        }
        if (Array.isArray(series[0])) {
            // Large vector: just render the norm of each vector.
            const norms = series.map(row => {
                let s = 0;
                for (const x of row) s += (typeof x === "number") ? x * x : 0;
                return Math.sqrt(s);
            });
            return numericLineSvg(norms, length);
        }
        return "";
    }

    function numericLineSvg(values, length) {
        return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${numericLinePath(values, length, "#5b8def")}</svg>`;
    }

    function numericLinePath(values, length, color) {
        const finite = values.filter(v => typeof v === "number" && isFinite(v));
        if (!finite.length) return "";
        let lo = Math.min(...finite);
        let hi = Math.max(...finite);
        if (lo === hi) { lo -= 1; hi += 1; }
        const points = [];
        for (let i = 0; i < values.length; i++) {
            const v = (typeof values[i] === "number" && isFinite(values[i])) ? values[i] : (lo + hi) / 2;
            const x = (i / Math.max(1, length - 1)) * 100;
            const y = 100 - ((v - lo) / (hi - lo)) * 80 - 10; // 10% pad top/bottom
            points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
        }
        return `<polyline points="${points.join(" ")}" fill="none" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
    }

    // ── Mouse handlers (selection + click-to-seek) ──────────────────────

    function wireFeatureRowTrack(track) {
        const length = parseInt(track.getAttribute("data-length"), 10);
        const featureName = track.getAttribute("data-feature");
        if (!length) return;

        track.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            const datasetId = window.currentDataset;
            const epIdx = window.currentEpisode;
            const ds = window.datasets && window.datasets[datasetId];
            if (!ds) return;
            const ft = (ds.features_schema || {})[featureName];

            const trimRange = getActiveTrim(datasetId, epIdx, length);
            const frame = pixelToFrame(e, track, length);
            // Clamp to trim envelope. Click outside trim is a no-op per the design.
            if (frame < trimRange[0] || frame >= trimRange[1]) return;

            // Always seek the playhead.
            if (typeof window.loadAllFrames === "function") {
                window.loadAllFrames(frame);
            }

            // Set a single-frame selection (will extend on drag).
            selection = {
                datasetId,
                episodeIndex: epIdx,
                frameFrom: frame,
                frameTo: frame + 1,
                originRow: featureName,
            };
            dragState = { anchorFrame: frame, originRow: featureName };
            renderInspector();
            renderFeatureRows();

            e.preventDefault();
        });
    }

    document.addEventListener("mousemove", (e) => {
        if (!dragState) return;
        const track = document.querySelector(`.row-track[data-feature="${cssEscape(dragState.originRow)}"]`);
        if (!track) return;
        const length = parseInt(track.getAttribute("data-length"), 10);
        const trimRange = getActiveTrim(window.currentDataset, window.currentEpisode, length);
        let frame = pixelToFrame(e, track, length);
        frame = Math.max(trimRange[0], Math.min(trimRange[1] - 1, frame));
        if (selection) {
            selection.frameFrom = Math.min(dragState.anchorFrame, frame);
            selection.frameTo = Math.max(dragState.anchorFrame, frame) + 1;
            // Track playhead at drag-end.
            if (typeof window.loadAllFrames === "function") {
                window.loadAllFrames(frame);
            }
            renderFeatureRows();
        }
    });

    document.addEventListener("mouseup", () => {
        if (dragState) {
            dragState = null;
            renderInspector();
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            clearSelection();
        }
    });

    function pixelToFrame(e, track, length) {
        const rect = track.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const t = Math.max(0, Math.min(1, x / rect.width));
        return Math.max(0, Math.min(length - 1, Math.floor(t * length)));
    }

    // ── "Show pending edits" toggle ─────────────────────────────────────

    window.onShowPendingEditsToggle = function () {
        const cb = document.getElementById("show-pending-edits-toggle");
        showPendingEdits = !!(cb && cb.checked);
        renderFeatureRows();
    };

    // ── Resize handles ──────────────────────────────────────────────────

    function setupVerticalResize() {
        const handle = document.getElementById("inspector-resize");
        const inspector = document.getElementById("inspector");
        if (!handle || !inspector) return;
        let dragging = false;
        let startX = 0, startW = 0;

        const stored = parseInt(localStorage.getItem("featureEditing.inspectorWidth") || "", 10);
        if (stored && stored >= 220 && stored <= 600) inspector.style.width = `${stored}px`;

        handle.addEventListener("mousedown", (e) => {
            dragging = true;
            startX = e.clientX;
            startW = inspector.getBoundingClientRect().width;
            handle.classList.add("dragging");
            e.preventDefault();
        });
        document.addEventListener("mousemove", (e) => {
            if (!dragging) return;
            const dx = e.clientX - startX;
            const next = Math.max(220, Math.min(600, startW - dx));
            inspector.style.width = `${next}px`;
        });
        document.addEventListener("mouseup", () => {
            if (!dragging) return;
            dragging = false;
            handle.classList.remove("dragging");
            const px = parseInt(inspector.style.width, 10);
            if (px) localStorage.setItem("featureEditing.inspectorWidth", String(px));
        });
    }

    function setupHorizontalResize() {
        const handle = document.getElementById("cameras-timeline-resize");
        const grid = document.getElementById("camera-grid");
        if (!handle || !grid) return;
        let dragging = false;
        let startY = 0, startH = 0;

        const stored = parseInt(localStorage.getItem("featureEditing.cameraGridHeight") || "", 10);
        if (stored && stored >= 120) grid.style.flex = `0 0 ${stored}px`;

        handle.addEventListener("mousedown", (e) => {
            dragging = true;
            startY = e.clientY;
            startH = grid.getBoundingClientRect().height;
            handle.classList.add("dragging");
            e.preventDefault();
        });
        document.addEventListener("mousemove", (e) => {
            if (!dragging) return;
            const dy = e.clientY - startY;
            const next = Math.max(120, startH + dy);
            grid.style.flex = `0 0 ${next}px`;
        });
        document.addEventListener("mouseup", () => {
            if (!dragging) return;
            dragging = false;
            handle.classList.remove("dragging");
            const m = grid.style.flex.match(/(\d+)px/);
            if (m) localStorage.setItem("featureEditing.cameraGridHeight", m[1]);
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        setupVerticalResize();
        setupHorizontalResize();
    });

    // ── Helpers ─────────────────────────────────────────────────────────

    function escapeHtml(s) {
        return String(s ?? "").replace(/[&<>"']/g, ch => (
            { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
        ));
    }
    function cssEscape(s) {
        return String(s).replace(/(["\\])/g, "\\$1");
    }
    function formatNumber(n) {
        if (typeof n !== "number") return String(n);
        if (Math.abs(n) >= 1000) return n.toFixed(0);
        if (Math.abs(n) >= 1) return n.toFixed(3);
        return n.toFixed(4);
    }
    function formatValueShort(v) {
        if (v == null) return "—";
        if (Array.isArray(v)) {
            if (v.length <= 6) return "[" + v.map(formatNumber).join(", ") + "]";
            return `[${formatNumber(v[0])}, … (${v.length} dims)]`;
        }
        if (typeof v === "number") return formatNumber(v);
        if (typeof v === "boolean") return v ? "✓ true" : "✗ false";
        return String(v);
    }
})();
