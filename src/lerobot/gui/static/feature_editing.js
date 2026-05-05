// Feature Editing — per-frame view + edit (Phase B).
//
// Owns the right-hand Inspector panel, the per-feature timeline rows under
// the existing scrubber, the vertical-slice selection model, and the
// click-to-seek-and-select gesture. State / DOM lookups talk to the existing
// app.js globals (datasets, episodes, currentDataset, currentEpisode,
// currentFrame, pendingEdits, fps, totalFrames, trimStart, trimEnd, etc.).

(function () {
    "use strict";

    // Lightweight log prefix so DevTools console is searchable / filterable.
    // Toggle verbose mode in the browser via:
    //     window.FeatureEditing.verbose = true
    // before opening a dataset.
    const LOG = "[feature-editing]";
    const _log = (...a) => console.info(LOG, ...a);
    const _warn = (...a) => console.warn(LOG, ...a);
    const _err = (...a) => console.error(LOG, ...a);
    _log("module loaded");

    // ── Per-dataset / per-episode caches ─────────────────────────────────
    const seriesCache = new Map(); // key = `${datasetId}:${episodeIdx}` → {length, series}
    const featureRowState = new Map(); // featureName → {pinned, expanded}

    // Selection: {episodeIndex, frameFrom, frameTo, originRow}
    let selection = null;
    let showPendingEdits = false;

    // Display↔storage name mapping for synthetic features. Backend stores
    // pending edits keyed by the storage name, but the row is rendered with
    // the display name — pending overlays + live value merging must bridge
    // both sides. Currently only the LeRobot 3.0 subtask format goes through
    // this synthesis (see SUBTASK_DISPLAY_FEATURE in api/datasets.py).
    const DISPLAY_TO_STORAGE = { subtask: "subtask_index" };

    function rowMatchesPendingFeature(rowName, pendingFeature) {
        if (rowName === pendingFeature) return true;
        return DISPLAY_TO_STORAGE[rowName] === pendingFeature;
    }

    function pendingFeatureEditsFor(rowName) {
        return (window.pendingEdits || []).filter(e =>
            e.dataset_id === window.currentDataset &&
            e.episode_index === window.currentEpisode &&
            e.edit_type === "feature_set" &&
            e.params && rowMatchesPendingFeature(rowName, e.params.feature)
        );
    }

    function applyPendingEditsToSeries(rowName, series) {
        const edits = pendingFeatureEditsFor(rowName);
        if (!edits.length) return series;
        const merged = series.slice();
        for (const e of edits) {
            const from = Math.max(0, e.params.frame_from);
            const to = Math.min(merged.length, e.params.frame_to);
            for (let i = from; i < to; i++) merged[i] = e.params.value;
        }
        return merged;
    }

    function getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo) {
        const key = `${datasetId}:${episodeIndex}`;
        const cached = seriesCache.get(key);
        if (!cached || !cached.series || !cached.series[name]) return null;
        const merged = applyPendingEditsToSeries(name, cached.series[name]);
        return merged.slice(frameFrom, frameTo);
    }

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
        const ds = window.datasets && window.datasets[datasetId];
        const fs = (ds && ds.features_schema) || {};
        const featureNames = Object.keys(fs);
        _log("onDatasetOpened", datasetId, "features:", featureNames.length, "schema present:", !!ds);
        if (!ds) {
            _err("onDatasetOpened: dataset not in window.datasets — wiring issue?");
        } else if (!ds.features_schema) {
            _warn("onDatasetOpened: dataset.features_schema is undefined — likely a stale frontend. Reload (Ctrl/Cmd-Shift-R).");
        } else if (featureNames.length === 0) {
            _warn("onDatasetOpened: features_schema is empty — dataset has no declared features.");
        }
        renderInspectorEmpty(datasetId);
    }

    function onDatasetClosed(datasetId) {
        _log("onDatasetClosed", datasetId);
        // Drop cached series for this dataset.
        for (const key of Array.from(seriesCache.keys())) {
            if (key.startsWith(`${datasetId}:`)) seriesCache.delete(key);
        }
        if (selection && selection.datasetId === datasetId) selection = null;
        renderInspectorEmpty(null);
        renderFeatureRows();
    }

    function onEpisodeSelected(datasetId, episodeIdx) {
        _log("onEpisodeSelected", datasetId, "ep=", episodeIdx);
        // Reset selection on episode switch — selection is per-episode.
        selection = null;
        renderInspector();
        loadFeatureSeries(datasetId, episodeIdx).then((data) => {
            _log("feature-series loaded for ep", episodeIdx, "→", data ? "OK, " + Object.keys(data.series || {}).length + " series" : "NULL");
            renderFeatureRows();
        }).catch((err) => _err("feature-series load failed", err));
    }

    function onPlayheadChanged() {
        // Update the per-frame value display in Inspector when no selection.
        if (!selection) renderInspectorFrameValues();
    }

    // Tracks the previous pending-count so we can detect a Save (>0 → 0).
    let _lastPendingCount = 0;

    function onPendingEditsChanged() {
        const pending = (window.pendingEdits || []);
        const prev = _lastPendingCount;
        const curr = pending.length;
        _lastPendingCount = curr;

        // Pending dropping to 0 means Save or Discard just fired. The series
        // cache holds pre-edit values that are now stale relative to disk —
        // drop it so the next render fetches fresh data.
        if (prev > 0 && curr === 0) {
            const datasetId = window.currentDataset;
            if (datasetId) {
                _log("onPendingEditsChanged: pending dropped to 0; invalidating seriesCache for", datasetId);
                for (const key of Array.from(seriesCache.keys())) {
                    if (key.startsWith(`${datasetId}:`)) seriesCache.delete(key);
                }
                // Re-fetch for the current episode so the row plots redraw fresh.
                const epIdx = window.currentEpisode;
                if (epIdx != null) {
                    loadFeatureSeries(datasetId, epIdx).then(() => renderFeatureRows()).catch(
                        (err) => _err("post-save series reload failed", err)
                    );
                    return; // renderFeatureRows is called inside the .then()
                }
            }
        }

        renderFeatureRows(); // pending overlay changes
        renderInspector(); // pending indicator on cards
    }

    function clearSelection() {
        if (!selection) return;
        _log("clearSelection");
        selection = null;
        renderInspector();
        renderFeatureRows();
    }

    // ── Series fetch ─────────────────────────────────────────────────────

    async function loadFeatureSeries(datasetId, episodeIdx) {
        const key = `${datasetId}:${episodeIdx}`;
        if (seriesCache.has(key)) {
            _log("loadFeatureSeries cache hit", key);
            return seriesCache.get(key);
        }
        // Only fetch features the user can actually see (visible by default + pinned),
        // not every column in the dataset. Avoids pulling 14-dim observation.state etc.
        // when the user only cares about reward / success / subtask.
        const ds = window.datasets && window.datasets[datasetId];
        const visible = (ds && ds.features_schema)
            ? Object.entries(ds.features_schema)
                .filter(([name, ft]) => {
                    const state = featureRowState.get(name) || {};
                    if (state.pinned) return true;
                    if (isHiddenByDefault(name, ft)) return false;
                    return true;
                })
                .map(([name, _]) => name)
            : [];
        let url = `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIdx}/feature-series`;
        if (visible.length > 0) {
            url += `?features=${visible.map(encodeURIComponent).join(",")}`;
        }
        _log("loadFeatureSeries", url, "visible features:", visible);
        const res = await fetch(url);
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            _err(`feature-series ${res.status}: ${text}`);
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
        const from = selection.frameFrom;
        const to = selection.frameTo;
        const k = to - from;
        const m = to - 1;

        const cards = [];
        for (const [name, ft] of Object.entries(featuresSchema)) {
            if (!isEditable(name, ft)) continue;
            cards.push(renderFeatureCard(name, ft, from, to, datasetId, selection.episodeIndex, selection.originRow));
        }
        if (cards.length === 0) {
            cards.push(
                '<div class="empty-state">' +
                'No editable features in the selection. action / observation.* / images / DEFAULT_FEATURES are read-only in V1.' +
                '</div>'
            );
        }

        const titleText = (k === 1)
            ? `frame ${from}`
            : `frames ${from}…${m} (${k} frames)`;

        body.innerHTML = `
            <div class="inspector-selection-header">
                <div class="sel-title">${titleText}</div>
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
        const isBroadcast = !!ft.is_per_episode;
        // Broadcast features always edit the whole episode; show the effective
        // range so the user sees what the staging endpoint will write.
        let effFrom = frameFrom, effTo = frameTo;
        if (editable && isBroadcast) {
            effFrom = 0;
            effTo = window.totalFrames || frameTo;
        }
        const pendingEdit = findPendingFeatureEdit(datasetId, episodeIndex, name, effFrom, effTo);

        const headerExtras = pendingEdit
            ? `<span class="card-pending">● pending</span>`
            : "";
        const broadcastNote = (editable && isBroadcast)
            ? `<div class="card-broadcast-note">per-episode — edit applies to the full episode (frames 0…${(window.totalFrames || 0) - 1})</div>`
            : "";

        let widget = "";
        if (!editable) {
            widget = `<span class="card-readonly-tag">read-only in V1</span>`;
        } else {
            widget = renderWidgetForType(name, ft, effFrom, effTo, datasetId, episodeIndex);
        }

        // Range shown next to the feature name. Declared bounds (info.json
        // ``min`` / ``max``) take precedence — they're authoritative and
        // enforced. Otherwise fall back to the dataset-wide observed extrema
        // from meta/stats.json. The two get distinct labels so the user knows
        // which one they're looking at.
        let observedRange = "";
        if (ft.declared_min != null && ft.declared_max != null) {
            observedRange =
                `<span class="card-observed-range" title="declared bounds (info.json)">` +
                `[${formatNumber(ft.declared_min)} … ${formatNumber(ft.declared_max)}]</span>`;
        } else if (ft.observed_min != null && ft.observed_max != null) {
            observedRange =
                `<span class="card-observed-range" title="observed across the dataset (meta/stats.json)">` +
                `[${formatNumber(ft.observed_min)} … ${formatNumber(ft.observed_max)}]</span>`;
        }

        return `
            <div class="feature-card ${focused ? "focused" : ""} ${editable ? "" : "readonly"}" data-feature="${escapeHtml(name)}">
                <div class="card-header">
                    <span class="card-name">${escapeHtml(name)}${observedRange}${headerExtras}</span>
                    <span class="card-dtype">${escapeHtml(dtype)}[${shape}]</span>
                </div>
                <div class="card-summary">${cardSummary(name, ft, datasetId, episodeIndex, effFrom, effTo)}</div>
                ${broadcastNote}
                <div class="card-widget">${widget}</div>
            </div>
        `;
    }

    function cardSummary(name, ft, datasetId, episodeIndex, frameFrom, frameTo) {
        const slice = getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo);
        if (slice === null || !slice.length) return "&nbsp;";
        // Single-frame selection: just show the value (no range/uniform framing).
        if (slice.length === 1) {
            const v = slice[0];
            if (typeof v === "number") return `value: ${formatNumber(v)}`;
            if (typeof v === "boolean") return `value: ${v ? "✓ true" : "✗ false"}`;
            if (typeof v === "string") return `value: "${escapeHtml(v)}"`;
            return `value: ${escapeHtml(String(v))}`;
        }
        if (typeof slice[0] === "number") {
            const nums = slice.filter(v => typeof v === "number");
            const min = Math.min(...nums);
            const max = Math.max(...nums);
            // Avoid the misleading "range: X … X" when every frame in the
            // selection has the same value — call it uniform explicitly so the
            // user doesn't read it as the feature's schema bounds.
            if (min === max) return `uniform: ${formatNumber(min)} (${slice.length} frames)`;
            return `selection min … max: ${formatNumber(min)} … ${formatNumber(max)}`;
        }
        if (typeof slice[0] === "boolean") {
            const t = slice.filter(v => v === true).length;
            const f = slice.length - t;
            return `${t} true · ${f} false`;
        }
        if (typeof slice[0] === "string") {
            const unique = new Set(slice);
            if (unique.size === 1) return `uniform: "${escapeHtml(slice[0])}" (${slice.length} frames)`;
            return `${unique.size} unique values`;
        }
        return "&nbsp;";
    }

    function renderWidgetForType(name, ft, frameFrom, frameTo, datasetId, episodeIndex) {
        const dtype = ft.dtype || "";
        const shape = ft.shape || [];
        const isScalar = (shape.length === 0) || (shape.length === 1 && shape[0] === 1);

        if (dtype === "bool" && isScalar) {
            // Initial state mirrors the merged slice (disk + pending edits):
            // all-true → checked, all-false → unchecked, mixed → indeterminate.
            // Without this, an all-true range renders as an unchecked box and
            // the user's click stages true (a no-op), then the roundtrip
            // re-renders unchecked → looks like the click did nothing.
            const slice = getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo);
            let checkedAttr = "";
            let dataInitial = "false";
            if (slice && slice.length) {
                const t = slice.filter(v => v === true).length;
                const f = slice.length - t;
                if (t > 0 && f === 0) { checkedAttr = " checked"; dataInitial = "true"; }
                else if (t > 0 && f > 0) { dataInitial = "mixed"; }
            }
            return (
                `<input type="checkbox" data-widget="bool" data-feature="${escapeHtml(name)}"` +
                ` data-initial="${dataInitial}"${checkedAttr}>`
            );
        }
        if (dtype === "string") {
            return `<input type="text" data-widget="string" data-feature="${escapeHtml(name)}" placeholder="(value for range)">`;
        }
        // Categorical integer feature (int + names). The on-disk value is the
        // index ``[0, len(names))``; the user picks by label. Detected before
        // the generic scalar path so the slider doesn't take over for these.
        if (isScalar && dtype.startsWith("int") && Array.isArray(ft.names) && ft.names.length > 0) {
            const slice = getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo);
            let initialIdx = null;
            if (slice && slice.length) {
                const nums = slice.filter(v => typeof v === "number");
                if (nums.length) {
                    const min = Math.min(...nums);
                    const max = Math.max(...nums);
                    if (min === max) initialIdx = Math.round(min);
                }
            }
            const options = ft.names.map((label, idx) => {
                const sel = (idx === initialIdx) ? " selected" : "";
                return `<option value="${idx}"${sel}>${escapeHtml(label)}</option>`;
            });
            // Leading blank option used when the selection is mixed — so the
            // dropdown doesn't lie about the current value.
            const placeholder =
                initialIdx == null
                    ? `<option value="" selected disabled>(mixed)</option>`
                    : "";
            return (
                `<select data-widget="categorical" data-feature="${escapeHtml(name)}">` +
                placeholder +
                options.join("") +
                `</select>`
            );
        }
        if (isScalar && (dtype.startsWith("int") || dtype.startsWith("float"))) {
            // Slider lo/hi precedence:
            //   1. Declared bounds from info.json (enforced by the backend)
            //   2. Dataset-wide observed extrema from meta/stats.json
            //   3. Current episode's loaded series (fallback for older datasets)
            // Declared bounds win because they're authoritative — a 1-5 quality
            // rating shouldn't let the slider scroll outside [1, 5] just because
            // the observed values happen to span the same range.
            let lo = -1, hi = 1;
            if (ft.declared_min != null && ft.declared_max != null) {
                lo = ft.declared_min;
                hi = ft.declared_max;
            } else if (ft.observed_min != null && ft.observed_max != null) {
                lo = ft.observed_min;
                hi = ft.observed_max;
            } else {
                const key = `${datasetId}:${episodeIndex}`;
                const cached = seriesCache.get(key);
                if (cached && cached.series && cached.series[name]) {
                    const all = cached.series[name].filter(v => typeof v === "number");
                    if (all.length) {
                        lo = Math.min(...all);
                        hi = Math.max(...all);
                    }
                }
            }
            if (lo === hi) { lo -= 1; hi += 1; }
            // Initial value mirrors the merged slice: a single value when the
            // selection is uniform, blank when mixed. Without this, the number
            // box renders empty (looking like an unfilled color-picker swatch),
            // and the slider sits at its midpoint regardless of actual values.
            const slice = getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo);
            let initialValueAttr = "";
            let initialSliderAttr = "";
            if (slice && slice.length) {
                const nums = slice.filter(v => typeof v === "number");
                if (nums.length) {
                    const min = Math.min(...nums);
                    const max = Math.max(...nums);
                    if (min === max) {
                        const formatted = (dtype.startsWith("int")) ? String(Math.round(min)) : String(min);
                        initialValueAttr = ` value="${formatted}"`;
                        initialSliderAttr = ` value="${formatted}"`;
                    }
                    // Mixed: leave both blank so the user sees no spurious value.
                }
            }
            const step = (dtype.startsWith("int")) ? "1" : "any";
            return `
                <input type="range" data-widget="scalar-slider" data-feature="${escapeHtml(name)}" min="${lo}" max="${hi}" step="${step}"${initialSliderAttr}>
                <input type="number" data-widget="scalar-number" data-feature="${escapeHtml(name)}" step="${step}"${initialValueAttr} placeholder="(value)">
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

    // 300 ms debounce — text-style inputs stage on idle, not every keystroke.
    function _debounce(fn, ms) {
        let t = null;
        return (...args) => {
            if (t) clearTimeout(t);
            t = setTimeout(() => { t = null; fn(...args); }, ms);
        };
    }

    function wireWidgets(root) {
        const cards = root.querySelectorAll(".feature-card[data-feature]");
        cards.forEach(card => {
            const featureName = card.getAttribute("data-feature");
            const widgets = card.querySelectorAll("[data-widget]");

            // Indeterminate state for bool[1] checkboxes when the range has
            // mixed values. The data-initial attribute is set in renderWidgetForType
            // from the merged slice — no need to re-derive from card text.
            const boolBox = card.querySelector('[data-widget="bool"]');
            if (boolBox && boolBox.getAttribute("data-initial") === "mixed") {
                boolBox.indeterminate = true;
            }

            // Slider <-> number sync for scalar-slider/scalar-number pair.
            const slider = card.querySelector('[data-widget="scalar-slider"]');
            const numInput = card.querySelector('[data-widget="scalar-number"]');
            if (slider && numInput) {
                slider.addEventListener("input", () => {
                    numInput.value = slider.value;
                });
                // Slider commits on `change` (released) — discrete, not flooding.
                slider.addEventListener("change", () => {
                    stageFeatureEdit(featureName, parseFloat(slider.value));
                });
                // Number input: stage on blur OR after 300ms idle to avoid hammering
                // the staging endpoint on every keystroke.
                const stageNum = () => {
                    if (numInput.value === "") return;
                    slider.value = numInput.value;
                    stageFeatureEdit(featureName, parseFloat(numInput.value));
                };
                numInput.addEventListener("blur", stageNum);
                numInput.addEventListener("input", _debounce(stageNum, 300));
            }

            widgets.forEach(w => {
                const kind = w.getAttribute("data-widget");
                if (kind === "bool") {
                    w.addEventListener("change", () => {
                        // User clicked → no longer indeterminate.
                        w.indeterminate = false;
                        stageFeatureEdit(featureName, w.checked);
                    });
                } else if (kind === "categorical") {
                    // Dropdown over an int+names feature. The select's value
                    // is the index string; stage as int so it lands as
                    // categorical-valid via the backend's bounds check.
                    w.addEventListener("change", () => {
                        if (w.value === "") return;  // (mixed) placeholder
                        stageFeatureEdit(featureName, parseInt(w.value, 10));
                    });
                } else if (kind === "string") {
                    // Stage on blur, Enter, or after 600ms idle while typing.
                    // Repeated stages on the same range collapse via the
                    // _lastStagedKey path in stageFeatureEdit — typing "appr",
                    // pausing, then typing "oach" produces one staged edit
                    // with value "approach", not two overlapping ones.
                    const stageText = () => {
                        if (w.value === "") return;
                        stageFeatureEdit(featureName, w.value);
                    };
                    w.addEventListener("blur", stageText);
                    w.addEventListener("input", _debounce(stageText, 600));
                    w.addEventListener("keydown", (e) => {
                        if (e.key === "Enter") {
                            e.preventDefault();
                            w.blur(); // triggers stageText via the blur handler
                        }
                    });
                } else if (kind === "vector-cell") {
                    // Stage when *any* cell changes — collect all cells into the vector.
                    const stageVec = () => {
                        const cells = card.querySelectorAll('[data-widget="vector-cell"]');
                        const vec = [];
                        cells.forEach(c => {
                            const v = c.value === "" ? 0 : parseFloat(c.value);
                            vec.push(v);
                        });
                        stageFeatureEdit(featureName, vec);
                    };
                    w.addEventListener("blur", stageVec);
                    w.addEventListener("input", _debounce(stageVec, 300));
                } else if (kind === "json") {
                    const stageJson = () => {
                        if (w.value === "") return;
                        try {
                            const parsed = JSON.parse(w.value);
                            stageFeatureEdit(featureName, parsed);
                        } catch (e) {
                            window.setStatus && window.setStatus("Invalid JSON: " + e.message);
                        }
                    };
                    w.addEventListener("blur", stageJson);
                    w.addEventListener("input", _debounce(stageJson, 600));
                }
                // scalar-slider / scalar-number handled above via slider/numInput pair.
            });
        });
    }

    // Identity of the last successfully-staged edit. When the user keeps
    // editing the SAME (dataset, feature, episode, range), we treat it as
    // "still updating the same edit" and auto-confirm overlap silently —
    // the backend's fully-contained-removal collapses to one staged edit.
    // Cleared when the selection changes (different range = different edit).
    let _lastStagedKey = null;
    function _stageKey(datasetId, feature, episodeIndex, frameFrom, frameTo) {
        return `${datasetId}|${feature}|${episodeIndex}|${frameFrom}|${frameTo}`;
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
        const stageKey = _stageKey(
            datasetId, featureName, selection.episodeIndex,
            selection.frameFrom, selection.frameTo
        );
        // Same key as last successful stage? Skip the 409-dialog round-trip
        // and confirm overlap upfront — the backend collapses the prior
        // edit into the new one (full containment).
        if (_lastStagedKey === stageKey) {
            body.confirm_overlap = true;
        }
        try {
            let res = await fetch("/api/edits/feature-set", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            // Overlap with a *different* edit (e.g. a separately-staged range)
            // — prompt the user. Same-key collisions are pre-confirmed above.
            if (res.status === 409) {
                const payload = await res.json();
                const detail = payload && payload.detail;
                if (detail && detail.code === "overlapping_edit") {
                    const ranges = (detail.overlapping || [])
                        .map(o => `[${o.frame_from}…${o.frame_to - 1}]`)
                        .join(", ");
                    const ok = window.confirm(
                        `You already have ${detail.overlapping.length} staged edit(s) on ` +
                        `${detail.feature} (episode ${detail.episode_index}) overlapping ` +
                        `frames ${detail.new_range[0]}…${detail.new_range[1] - 1}: ${ranges}.\n\n` +
                        `Continue? Prior edits will be clipped (last-write-wins).`
                    );
                    if (!ok) return;
                    res = await fetch("/api/edits/feature-set", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ ...body, confirm_overlap: true }),
                    });
                }
            }
            if (!res.ok) {
                const err = await res.text();
                window.setStatus && window.setStatus(`Edit rejected: ${err}`);
                return;
            }
            // Backend may have coerced the range (per-episode broadcast).
            // Update local selection so the visualization matches what was staged.
            let coercedFrom = selection.frameFrom, coercedTo = selection.frameTo;
            try {
                const responseBody = await res.json();
                if (responseBody && Array.isArray(responseBody.coerced_range)) {
                    coercedFrom = responseBody.coerced_range[0];
                    coercedTo = responseBody.coerced_range[1];
                    selection.frameFrom = coercedFrom;
                    selection.frameTo = coercedTo;
                }
            } catch (_) { /* response body wasn't JSON — ignore */ }
            // Record the identity of this successful stage so subsequent
            // typing in the SAME widget+selection auto-collapses into one
            // edit instead of stacking 409 dialogs. Use the (possibly
            // coerced) range that actually landed in PendingEdit.
            _lastStagedKey = _stageKey(
                datasetId, featureName, selection.episodeIndex,
                coercedFrom, coercedTo
            );
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
        if (!container) {
            _err("renderFeatureRows: #feature-rows container missing from DOM");
            return;
        }
        const datasetId = window.currentDataset;
        const epIdx = window.currentEpisode;
        if (!datasetId || epIdx == null || epIdx === undefined) {
            _log("renderFeatureRows: no current dataset/episode (datasetId=", datasetId, "ep=", epIdx, ")");
            container.innerHTML = "";
            return;
        }
        const ds = window.datasets && window.datasets[datasetId];
        if (!ds || !ds.features_schema) {
            _warn("renderFeatureRows: ds or features_schema missing for", datasetId, "→ rendering nothing. Hard-reload to refresh schema.");
            container.innerHTML =
                '<div class="feature-rows-empty">' +
                'Schema unavailable for this dataset. ' +
                'Hard-reload (Ctrl/Cmd-Shift-R) to fetch the latest schema, then reopen the dataset.' +
                "</div>";
            return;
        }
        const key = `${datasetId}:${epIdx}`;
        const cached = seriesCache.get(key);
        if (!cached) {
            _log("renderFeatureRows: series not cached yet for", key, "(loading…)");
            container.innerHTML = '<div class="feature-rows-empty">Loading feature series…</div>';
            return;
        }

        const visibleFeatures = [];
        const hiddenNames = [];
        for (const [name, ft] of Object.entries(ds.features_schema)) {
            const state = featureRowState.get(name) || {};
            const hidden = isHiddenByDefault(name, ft) && !state.pinned;
            if (!hidden) visibleFeatures.push([name, ft]);
            else hiddenNames.push(name);
        }
        _log("renderFeatureRows: visible=", visibleFeatures.map(p => p[0]), "hidden=", hiddenNames);
        if (!visibleFeatures.length) {
            container.innerHTML =
                '<div class="feature-rows-empty">' +
                'No visible features. action / observation.* / images / DEFAULT_FEATURES are ' +
                'hidden by default — pin a feature to show it.' +
                "</div>";
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
        const rawSeries = cached.series[name] || [];
        const length = cached.length;
        // Live-merge pending feature_set edits so the row reflects in-progress
        // values immediately — without this, a typed-but-not-saved subtask
        // change is invisible until the user clicks Save.
        const series = applyPendingEditsToSeries(name, rawSeries);
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
            const pendingForFeature = pendingFeatureEditsFor(name);
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
            // Colored stripe — each unique string gets a color; render run-length segments.
            // The colored rectangles go in a stretched SVG (preserveAspectRatio="none")
            // so they fill the row exactly. Text labels go in HTML overlays — putting
            // them in the stretched SVG would non-uniformly scale the glyphs (the cause
            // of the "white stretched artifact" before the rewrite).
            const colors = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085"];
            const colorMap = new Map();
            const rects = [];
            const labels = [];
            let i = 0;
            while (i < series.length) {
                const v = series[i];
                let j = i;
                while (j < series.length && series[j] === v) j++;
                if (!colorMap.has(v)) colorMap.set(v, colors[colorMap.size % colors.length]);
                const color = colorMap.get(v);
                const x = (i / length) * 100;
                const w = ((j - i) / length) * 100;
                rects.push(`<rect x="${x}%" y="10%" width="${w}%" height="80%" fill="${color}" opacity="0.7"/>`);
                if (j - i > 4) {
                    labels.push(
                        `<div class="row-string-label" ` +
                        `style="left:${x}%; width:${w}%;">` +
                        `${escapeHtml(String(v).slice(0, 24))}</div>`
                    );
                }
                i = j;
            }
            return (
                `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${rects.join("")}</svg>` +
                labels.join("")
            );
        }

        // Categorical (int + names): render as a colored band with the label
        // for each segment, similar to strings but indexed via ft.names.
        const isScalar = (shape.length === 0 || (shape.length === 1 && shape[0] === 1));
        if (
            isScalar
            && dtype.startsWith("int")
            && Array.isArray(ft.names)
            && ft.names.length > 0
            && typeof series[0] === "number"
        ) {
            const colors = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085"];
            const rects = [];
            const labels = [];
            let i = 0;
            while (i < series.length) {
                const v = series[i];
                let j = i;
                while (j < series.length && series[j] === v) j++;
                const idx = (typeof v === "number") ? Math.round(v) : -1;
                const label = (idx >= 0 && idx < ft.names.length) ? ft.names[idx] : `?(${v})`;
                const color = colors[((idx >= 0) ? idx : 0) % colors.length];
                const x = (i / length) * 100;
                const w = ((j - i) / length) * 100;
                rects.push(`<rect x="${x}%" y="10%" width="${w}%" height="80%" fill="${color}" opacity="0.7"/>`);
                if (j - i > 4) {
                    labels.push(
                        `<div class="row-string-label" style="left:${x}%; width:${w}%;">` +
                        `${escapeHtml(String(label).slice(0, 24))}</div>`
                    );
                }
                i = j;
            }
            return (
                `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${rects.join("")}</svg>` +
                labels.join("")
            );
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
