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

    // Banner dismissal — in-memory only (per browser session). Cleared on
    // page reload so the user is re-asked about adding default features.
    const bannerDismissed = new Set();

    // Names of MUST-have default features the banner offers to add. Mirrors
    // the backend's _DEFAULT_FEATURE_SPECS keys; if the two diverge, the
    // banner lists names the backend won't actually add.
    const DEFAULT_FEATURE_NAMES = ["reward", "success"];

    // Selection: {episodeIndex, frameFrom, frameTo, originRow}
    let selection = null;
    // label -> {key}, edited but not committed. Config commits IN PLACE, so
    // these never reach the timeline's pending queue.
    let _stagedTreatments = null;
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

    // Bit maths without bitwise operators. JavaScript's &, | and ~ coerce to
    // *32-bit* integers, so `value & Math.pow(2, 40)` is 0 and every flag past
    // bit 30 would be silently invisible and untickable -- while the stored
    // contract allows 63. Division and modulo stay exact to 2^53, which is also
    // where JSON stops carrying integers faithfully, so this is as far as the
    // browser can go regardless.
    const MAX_JS_BIT = 52;  // Number.MAX_SAFE_INTEGER is 2^53 - 1

    function bitIsSet(value, bit) {
        if (bit > MAX_JS_BIT) return false;
        return Math.floor(Math.round(value) / Math.pow(2, bit)) % 2 === 1;
    }

    function bitsOfMask(mask) {
        const bits = [];
        for (let b = 0; b <= MAX_JS_BIT; b++) {
            if (Math.pow(2, b) > mask) break;
            if (bitIsSet(mask, b)) bits.push(b);
        }
        return bits;
    }

    function withBits(value, setMask, clearMask) {
        let v = Math.round(value);
        for (const b of bitsOfMask(setMask)) if (!bitIsSet(v, b)) v += Math.pow(2, b);
        for (const b of bitsOfMask(clearMask)) if (bitIsSet(v, b)) v -= Math.pow(2, b);
        return v;
    }

    function pendingFeatureEditsFor(rowName, editType = "feature_set") {
        return (window.pendingEdits || []).filter(e =>
            e.dataset_id === window.currentDataset &&
            e.episode_index === window.currentEpisode &&
            e.edit_type === editType &&
            e.params && rowMatchesPendingFeature(rowName, e.params.feature)
        );
    }

    /**
     * Merge staged `mask_range` edits into a lane's two bitsets.
     *
     * Without this a click springs back: the lane is drawn from the stored
     * column, which does not carry the edit until Save — so the operator would
     * click a segment, see nothing change, and click again.
     *
     * Returns `[enabled, disabled]`, both fresh arrays.
     */
    function applyPendingMaskEdits(rowName, labels, enabled, disabled, length) {
        const edits = (window.pendingEdits || []).filter(
            (e) => e.edit_type === "mask_range"
                && e.params?.camera === maskCameraOf(rowName)
                && e.episode_index === window.currentEpisode
        );
        if (!edits.length) return [enabled, disabled];
        const en = enabled.slice();
        const dis = disabled.slice ? disabled.slice() : [];
        while (dis.length < length) dis.push(0);
        for (const e of edits) {
            const bit = labels.indexOf(e.params.label);
            if (bit < 0) continue;
            const from = Math.max(0, e.params.from_frame);
            const to = Math.min(length, e.params.to_frame);
            // Through the file's own bit helpers, not `1 << bit`. JavaScript's
            // bitwise operators are 32-bit signed: at bit 31 the shift goes
            // negative and at bit 32 it wraps to 1, so editing the 33rd label
            // would silently flip the FIRST label's lane. The vocabulary grows
            // and never shrinks, and the design sizes the panel for 40 labels,
            // so 32 is reachable. `bitIsSet`/`withBits` do the same arithmetic
            // in floats, which is why they exist -- and why `isAbsent` in
            // apply_run_filter.js reads the same bitsets that way, with tests
            // naming bit 31 and bit 40.
            const B = Math.pow(2, bit);
            for (let i = from; i < to; i++) {
                const carried = bitIsSet(en[i] || 0, bit) || bitIsSet(dis[i] || 0, bit);
                // Absent frames are skipped, exactly as the server skips them:
                // there is no mask to mute or delete.
                if (!carried) continue;
                if (e.params.action === "delete") {
                    en[i] = withBits(en[i] || 0, 0, B);
                    dis[i] = withBits(dis[i] || 0, 0, B);
                } else if (e.params.action === "disable") {
                    en[i] = withBits(en[i] || 0, 0, B);
                    dis[i] = withBits(dis[i] || 0, B, 0);
                } else {
                    en[i] = withBits(en[i] || 0, B, 0);
                    dis[i] = withBits(dis[i] || 0, 0, B);
                }
            }
        }
        return [en, dis];
    }

    function applyPendingEditsToSeries(rowName, series) {
        const valueEdits = pendingFeatureEditsFor(rowName);
        // Flag edits set and clear bits rather than replacing the cell, so
        // they merge differently. Without this a ticked flag springs back to
        // unticked: the box is drawn from the stored column, which does not
        // carry the edit until Save.
        const bitEdits = pendingFeatureEditsFor(rowName, "feature_bits");
        if (!valueEdits.length && !bitEdits.length) return series;
        const merged = series.slice();
        for (const e of valueEdits) {
            const from = Math.max(0, e.params.frame_from);
            const to = Math.min(merged.length, e.params.frame_to);
            for (let i = from; i < to; i++) merged[i] = e.params.value;
        }
        for (const e of bitEdits) {
            const from = Math.max(0, e.params.frame_from);
            const to = Math.min(merged.length, e.params.frame_to);
            const set = Number(e.params.set_bits) || 0;
            const clear = Number(e.params.clear_bits) || 0;
            for (let i = from; i < to; i++) merged[i] = withBits(merged[i], set, clear);
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

    // Pure decision and formatting functions, exposed for unit tests. Everything
    // here is a function of its arguments alone — no DOM, no module state — so
    // feature_editing.test.js can cover the render rules under node instead of
    // only through a browser.
    const _internals = {
        renderDatasetSection,
        // Exposed so a test can prove the filler hands the job runner somewhere
        // to report progress. It passed no button and no callback, so a run that
        // can last hours reported nothing at all.
        runFillGaps,
        bitIsSet,
        withBits,
        isInternalFeature,
        isBinaryFeature,
        isRecordedFeature,
        isEditable,
        isDeletable,
        isHiddenByDefault,
        summarizeSlice,
        readOnlyValueHtml,
        renderTrackSvg,
    };

    /**
     * Per-camera mask coverage for the current episode, as the client holds it.
     *
     * What an apply run needs to obey the write rule without asking the server:
     * {camera: {labels, enabled, disabled}}, the two per-frame bitsets beside
     * the vocabulary they index. Returns {} when no series is loaded, which the
     * caller must treat as "cannot filter" rather than "nothing is covered" --
     * staging against unknown coverage is how a disabled mask gets refilled.
     */
    function maskCoverage(datasetId, episodeIndex) {
        const cached = seriesCache.get(`${datasetId}:${episodeIndex}`);
        const ds = window.datasets?.[datasetId];
        if (!cached || !ds) return {};
        const out = {};
        for (const [name, ft] of Object.entries(ds.features_schema || {})) {
            if (!Array.isArray(ft?.mask_labels) || !ft.mask_labels.length) continue;
            const camera = maskCameraOf(name);
            if (!camera) continue;
            out[camera] = {
                labels: ft.mask_labels,
                enabled: cached.series[name] || [],
                disabled: cached.series[`${name}__disabled`] || [],
            };
        }
        return out;
    }

    window.FeatureEditing = {
        maskCoverage,
        // The dataset tier's empty state depends on what the Overlays panel is
        // looking for, and that panel is where the typing happens. Without a way
        // to say "my objects changed", the offer to segment appeared only after
        // some unrelated action happened to re-render the Inspector.
        onLiveObjectsChanged: () => renderInspector(),
        maskSegments,
        renderFeatureRows,
        applyPendingMaskEdits,
        maskCameraOf,
        maskSegmentAt,
        stageMaskSegmentEdit,
        _internals,
        onDatasetOpened,
        onDatasetClosed,
        onEpisodeSelected,
        onPlayheadChanged,
        onPendingEditsChanged,
        clearSelection,
        refreshAfterSchemaAdd,
        refreshFromServer,
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
        maybeShowDefaultsBanner(datasetId);
    }

    function onDatasetClosed(datasetId) {
        _log("onDatasetClosed", datasetId);
        // Drop cached series for this dataset.
        for (const key of Array.from(seriesCache.keys())) {
            if (key.startsWith(`${datasetId}:`)) seriesCache.delete(key);
        }
        if (selection && selection.datasetId === datasetId) selection = null;
        bannerDismissed.delete(datasetId);
        hideDefaultsBanner();
        renderInspectorEmpty(null);
        renderFeatureRows();
    }

    // ── Default-features banner (T12) ──────────────────────────────────

    function _missingDefaults(datasetId) {
        const ds = window.datasets && window.datasets[datasetId];
        const fs = (ds && ds.features_schema) || {};
        return DEFAULT_FEATURE_NAMES.filter(n => !fs[n]);
    }

    // Known alternate column names that the backend will rename rather
    // than duplicate. Mirrors the `rename_from` lists in
    // _DEFAULT_FEATURE_SPECS on the backend; frontend only uses these
    // to show the user what will happen before they click Add.
    const DEFAULT_RENAME_FROM = {
        reward: ["next.reward"],
        success: [],
    };

    function _plannedRenames(datasetId) {
        const ds = window.datasets && window.datasets[datasetId];
        const fs = (ds && ds.features_schema) || {};
        const out = [];
        for (const target of DEFAULT_FEATURE_NAMES) {
            if (fs[target]) continue;
            for (const alt of (DEFAULT_RENAME_FROM[target] || [])) {
                if (fs[alt]) {
                    out.push({ from: alt, to: target });
                    break;
                }
            }
        }
        return out;
    }

    // Lossy migrations: per-frame bool next.success → per-episode int8
    // success (frame-level success timing is unrecoverable). The rename
    // machinery doesn't handle dtype changes, so the backend runs a
    // bespoke migration; we surface it here so the user sees the cost
    // before clicking Add.
    function _plannedLossyMigrations(datasetId) {
        const ds = window.datasets && window.datasets[datasetId];
        const fs = (ds && ds.features_schema) || {};
        const out = [];
        if (!fs.success && fs["next.success"]?.dtype === "bool") {
            out.push({
                from: "next.success",
                to: "success",
                reason: "per-frame bool → per-episode tri-state; frame-level success timing will be lost",
            });
        }
        return out;
    }

    function maybeShowDefaultsBanner(datasetId) {
        const banner = document.getElementById("default-features-banner");
        if (!banner) return;
        const missing = _missingDefaults(datasetId);
        if (missing.length === 0 || bannerDismissed.has(datasetId)) {
            banner.hidden = true;
            return;
        }
        document.getElementById("banner-missing-list").textContent = missing.join(", ");
        // Show rename plan + lossy-migration plan when applicable. Lossy
        // migrations are surfaced separately because they're irreversible —
        // the user needs to know they'll lose data before clicking Add.
        const renames = _plannedRenames(datasetId);
        const lossy = _plannedLossyMigrations(datasetId);
        const noteEl = document.getElementById("banner-rename-note");
        if (noteEl) {
            const parts = [];
            if (renames.length) {
                const items = renames.map(r => `<code>${r.from}</code> → <code>${r.to}</code>`).join(", ");
                parts.push(`Will preserve existing data: ${items}.`);
            }
            if (lossy.length) {
                const items = lossy.map(m =>
                    `<code>${m.from}</code> → <code>${m.to}</code> (${escapeHtml(m.reason)})`
                ).join("; ");
                parts.push(`<strong>Lossy migration:</strong> ${items}.`);
            }
            if (parts.length) {
                noteEl.innerHTML = parts.join("<br>");
                noteEl.hidden = false;
            } else {
                noteEl.hidden = true;
            }
        }
        banner.hidden = false;
        banner.dataset.datasetId = datasetId;
        const addBtn = document.getElementById("banner-add-btn");
        const dismissBtn = document.getElementById("banner-dismiss-btn");
        addBtn.onclick = () => addDefaultsFor(datasetId);
        dismissBtn.onclick = () => {
            bannerDismissed.add(datasetId);
            banner.hidden = true;
        };
    }

    function hideDefaultsBanner() {
        const banner = document.getElementById("default-features-banner");
        if (banner) banner.hidden = true;
    }

    async function addDefaultsFor(datasetId) {
        // Hard confirm before any lossy migration. The backend's
        // _migrate_next_success_inplace permanently drops next.success
        // and replaces it with a per-episode tri-state — frame-level
        // timing is unrecoverable. The user MUST acknowledge before we
        // POST. Plain renames (next.reward → reward) and adds-from-fill
        // are reversible enough that we don't gate on them.
        const lossy = _plannedLossyMigrations(datasetId);
        if (lossy.length) {
            const lines = lossy.map(m =>
                `  • ${m.from} → ${m.to}\n    ${m.reason}`
            ).join("\n");
            const ok = window.confirm(
                "About to perform a LOSSY migration on this dataset:\n\n" +
                lines + "\n\n" +
                "This rewrites parquet shards in place and CANNOT be undone via Discard.\n\n" +
                "Continue?"
            );
            if (!ok) return;
        }
        const addBtn = document.getElementById("banner-add-btn");
        const dismissBtn = document.getElementById("banner-dismiss-btn");
        const originalText = addBtn.textContent;
        addBtn.disabled = true;
        addBtn.textContent = "Adding…";
        if (dismissBtn) dismissBtn.disabled = true;
        try {
            const r = await fetch(
                `/api/datasets/${encodeURIComponent(datasetId)}/features/defaults`,
                { method: "POST" }
            );
            if (!r.ok) {
                const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
                window.setStatus && window.setStatus(`Add defaults failed: ${detail}`);
                return;
            }
            const payload = await r.json();
            // Update window.datasets from the POST response so the row column
            // re-renders against the new schema without a separate GET.
            if (payload && payload.info) {
                window.datasets[datasetId] = payload.info;
            }
            hideDefaultsBanner();
            renderInspector();
            renderFeatureRows();
            window.setStatus && window.setStatus(
                `Added: ${payload.added && payload.added.length ? payload.added.join(", ") : "(nothing — already present)"}`
            );
        } catch (err) {
            _err("addDefaultsFor failed", err);
            window.setStatus && window.setStatus("Add defaults failed: " + err.message);
        } finally {
            addBtn.disabled = false;
            addBtn.textContent = originalText;
            if (dismissBtn) dismissBtn.disabled = false;
        }
    }

    // Public: called by add_feature_dialog.js after a successful POST so the
    // schema-bound caches can refresh and rows re-render.
    // Public: called after a write this panel did not make — a mask save, an
    // effects apply — so the row shows what is on disk. The panel caches both
    // the schema (which gains `masks.*` on a first adopt, and
    // carries the treatment each lane displays) and a per-episode series
    // cache, and nothing else invalidates either. Best-effort: the write has
    // already landed, so a failure here is a stale row, not a failed save.
    async function refreshFromServer(datasetId) {
        try {
            const body = datasetId.startsWith("/") ? { local_path: datasetId } : { repo_id: datasetId };
            const res = await fetch("/api/datasets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            refreshAfterSchemaAdd(datasetId, res.ok ? await res.json() : null);
        } catch (err) {
            _err("refreshFromServer failed", err);
        }
    }

    function refreshAfterSchemaAdd(datasetId, info) {
        if (info) {
            window.datasets[datasetId] = info;
        }
        // Drop cached series — new feature columns aren't in the cached
        // payload yet. Next renderFeatureRows triggers a fresh load.
        for (const key of Array.from(seriesCache.keys())) {
            if (key.startsWith(`${datasetId}:`)) seriesCache.delete(key);
        }
        maybeShowDefaultsBanner(datasetId);
        // Trigger a feature-series reload for the current episode if any.
        if (window.currentEpisode != null) {
            loadFeatureSeries(datasetId, window.currentEpisode).then(() => {
                renderFeatureRows();
                renderInspector();
            }).catch(e => _err("refreshAfterSchemaAdd: feature-series reload failed", e));
        } else {
            renderFeatureRows();
            renderInspector();
        }
    }

    function onEpisodeSelected(datasetId, episodeIdx) {
        _log("onEpisodeSelected", datasetId, "ep=", episodeIdx);
        // Reset selection on episode switch — selection is per-episode.
        selection = null;
        renderInspector();
        loadFeatureSeries(datasetId, episodeIdx).then((data) => {
            _log("feature-series loaded for ep", episodeIdx, "→", data ? "OK, " + Object.keys(data.series || {}).length + " series" : "NULL");
            renderFeatureRows();
            // The Inspector render above ran before the series existed. Cards that
            // display recorded values rather than edit widgets — the read-only
            // `task` instruction — come up empty until they see the data, and
            // nothing else re-renders them. Matches the schema-add path, which
            // has always refreshed both.
            renderInspector();
        }).catch((err) => _err("feature-series load failed", err));
    }

    function onPlayheadChanged() {
        // No-op. The per-frame section tracks the timeline marker at render
        // time only — re-rendering on every scrub would tear down DOM
        // listeners and steal focus from any per-episode widget the user
        // is editing. The user gets fresh per-frame values by clicking on
        // the timeline (which both moves the marker AND creates a
        // single-frame selection), or by drag-selecting a range.
    }

    // Tracks the previous pending-count so we can detect a Save (>0 → 0).
    let _lastPendingCount = 0;

    function onPendingEditsChanged() {
        const pending = (window.pendingEdits || []);
        const prev = _lastPendingCount;
        const curr = pending.length;
        _lastPendingCount = curr;

        // Pending dropping to 0 means the staged edits are gone: Save, Discard,
        // or a flag edit that collapsed against its own opposite. In the first
        // two cases the series cache holds pre-edit values now stale relative
        // to disk, so it is dropped and refetched; in the third the refetch is
        // redundant but harmless, and telling them apart here would mean
        // teaching this hook what kind of edit disappeared.
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
                    // Both, not just the rows: the Inspector card is drawn from
                    // the same series, and leaving it un-rendered showed a card
                    // with no values at all until the next interaction.
                    loadFeatureSeries(datasetId, epIdx).then(() => {
                        renderFeatureRows();
                        renderInspector();
                    }).catch(
                        (err) => _err("post-save series reload failed", err)
                    );
                    return; // both renders happen inside the .then()
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

    // ── Feature classification ──────────────────────────────────────────
    //
    // One source of truth for "is this feature special?" Each predicate
    // answers a single question; higher-level predicates (isEditable,
    // isDeletable, isHiddenByDefault) compose them. Mirrors the backend's
    // checks in gui/api/edits.py and datasets/dataset_tools.py — keep
    // these in sync.
    //
    // Categories:
    //   internal     — auto-managed bookkeeping columns (timestamp etc.)
    //   binary       — stored as separate files (image, video)
    //   recorded     — sensor / control data the rest of the pipeline
    //                  depends on (action, observation.*)
    //   bannerManaged — has a dedicated banner / migration flow
    //                  (reward, success); generic dialog/delete refused

    const DEFAULT_FEATURES = new Set([
        "timestamp", "frame_index", "episode_index", "index", "task_index",
    ]);
    const READONLY_DTYPES = new Set(["image", "video"]);
    // DEFAULT_FEATURE_NAMES (reward / success) is declared near the top of
    // this module — the banner handlers reference it before this section
    // executes, so its declaration has to come first. Predicates below use
    // that single source of truth.

    function isInternalFeature(name) {
        return DEFAULT_FEATURES.has(name);
    }
    function isBinaryFeature(ft) {
        return !!ft && READONLY_DTYPES.has(ft.dtype);
    }
    function isRecordedFeature(name) {
        // "task" is the decoded language instruction the backend synthesizes in
        // place of task_index. Read-only here, but not immutable: upstream
        // changes it through modify_tasks, which reindexes meta/tasks.parquet
        // and rewrites total_tasks alongside every row. This pipeline stages
        // per-frame values over a range, so routing it here would leave the
        // lookup table and info.json disagreeing. GUI editing is tracked in
        // issue #125. Subtasks are deliberately not in this list — they have
        // an edit path.
        return name === "action" || name === "task" || name.startsWith("observation.");
    }
    function isBannerManaged(name) {
        return DEFAULT_FEATURE_NAMES.includes(name);
    }

    // A mask column is segmenter output, not something to type into: its cell
    // is an RLE string whose meaning is positional against mask_labels, and
    // the treatments every consumer reads live in its spec. Neither belongs to
    // the generic column controls. There is no mask-aware removal yet -- see
    // gui/docs/saved_masks.md, which designs it -- so a mask column cannot be
    // dropped at all today.
    //
    // This used to hold by accident: the columns were under `observation.`,
    // which isRecordedFeature already excludes. Moving them out of that
    // namespace removed the accident, so the rule is stated -- keyed on the
    // encoding, because the name has moved once already.
    function isMaskFeature(ft) {
        return !!(ft && ft.mask_encoding);
    }

    function isEditable(name, ft) {
        if (!ft) return false;
        if (isInternalFeature(name)) return false;
        if (isBinaryFeature(ft)) return false;
        if (isRecordedFeature(name)) return false;
        if (isMaskFeature(ft)) return false;
        return true;
    }

    function isDeletable(name, ft) {
        // Same exclusions as isEditable (you can't drop what you can't
        // edit — recorded data and internal bookkeeping are off-limits)
        // PLUS banner-managed defaults (reward / success have a separate
        // flow). Equivalent: anything you'd add via the generic dialog
        // is also deletable via the per-row ✕ / Inspector ✕.
        if (!ft) return false;
        if (!isEditable(name, ft)) return false;
        if (isBannerManaged(name)) return false;
        return true;
    }

    function isHiddenByDefault(name, ft) {
        // Hidden by default: only internal bookkeeping and binary blobs
        // (image/video have their own grid). Recorded data (action /
        // observation.*) is SHOWN by default as timeline overlays —
        // the user wants to see what was recorded, even though it's
        // read-only in V1. Pin/unpin still works to override per-feature.
        return isInternalFeature(name) || isBinaryFeature(ft);
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

    //: The effects a label can carry. Fetched once; falls back to the four
    //: the server has always offered if that request has not landed yet.
    let TREATMENT_KEYS = ["none", "tint", "blur", "random"];
    let _treatmentKeysAsked = false;

    /** Ask the server what effects exist, once, and only when one is drawn.
     *
     *  Lazy rather than at module load: this file is loaded outside a browser
     *  by its own unit tests, where `fetch` does not exist, and a request at
     *  import time took the whole module down with a ReferenceError. Nothing
     *  needs the answer until a dataset with masks is on screen anyway. */
    function ensureTreatmentKeys() {
        if (_treatmentKeysAsked || typeof fetch !== "function") return;
        _treatmentKeysAsked = true;
        fetch("/api/process/treatments")
            .then((r) => r.json())
            .then((d) => {
                const keys = (d.treatments || []).map((x) => x.key).filter(Boolean);
                if (keys.length && keys.join() !== TREATMENT_KEYS.join()) {
                    TREATMENT_KEYS = keys;
                    renderInspector();
                }
            })
            .catch(() => {});
    }

    /** Every mask column's shared vocabulary, or null when there are none. */
    function maskVocabulary(ds) {
        const cols = Object.entries(ds.features_schema || {})
            .filter(([k, ft]) => k.startsWith("masks.") && Array.isArray(ft.mask_labels));
        if (!cols.length) return null;
        const [, first] = cols[0];
        return {
            labels: first.mask_labels || [],
            treatments: first.mask_treatments || {},
            background: first.mask_background || { key: "none", params: {} },
            // What the segmenter is asked for, when it differs from the stored
            // name. Absent for every dataset that never sharpened a prompt.
            prompts: first.mask_prompts || {},
            cameras: cols.length,
        };
    }

    /**
     * The dataset-scoped section: the mask vocabulary and its treatments.
     *
     * Keyed by NAME, not by column. The vocabulary is shared by every camera —
     * the same object seen from three of them is one label — so a section per
     * column would ask the same question three times.
     *
     * Presentation is one row per label with a flat, mutually exclusive control:
     * reading and setting one label's treatment is a glance and a click, and the
     * tint button carries the colour, which a menu cannot show at all. Changing
     * one treatment across many labels is still N edits.
     */
    /** The dataset's own facts. Shown here because this is the DATASET tier, and
     *  because the summary they used to live in is an empty state: it is
     *  replaced the moment an episode is selected, so what dataset you are
     *  looking at stopped being on screen exactly when you started working. */
    function datasetFactsCard(ds) {
        const cams = (ds.camera_keys || []).length;
        const fact = (k, v) => `<div class="ds-fact"><span class="ds-fact-key">${k}</span>` +
            `<span class="ds-fact-val">${escapeHtml(String(v))}</span></div>`;
        return (
            `<div class="inspector-card ds-facts">` +
            fact("repo", ds.repo_id || ds.id || "—") +
            fact("episodes", ds.total_episodes ?? "?") +
            fact("frames", (ds.total_frames ?? 0).toLocaleString?.() ?? (ds.total_frames ?? "?")) +
            fact("fps", ds.fps ?? "?") +
            fact("cameras", cams || "—") +
            fact("robot", ds.robot_type || "—") +
            `</div>`
        );
    }

    function renderDatasetSection(datasetId, ds) {
        if (!ds) return "";
        const vocab = maskVocabulary(ds);
        const header = (meta) =>
            `<div class="inspector-section-header">` +
            `<div class="sel-title">Dataset</div>` +
            `<div class="sel-meta">${meta}</div></div>`;
        // The tier exists for any open dataset, not only a masked one: it is the
        // home for dataset scope, and a dataset with no masks still has facts.
        //
        // The fill is still offered here, because a dataset with no mask column
        // is exactly the one that needs the first pass. Withholding it made the
        // whole feature unreachable on anything new: the overlay panel has no
        // write by design, apply-while-playing is refused until a column exists,
        // and this button -- the only remaining way in -- was hidden precisely
        // when it was needed. The endpoint already carries the adopt handshake.
        if (!vocab) {
            const named = ((window.Overlays?.dataQuery?.() || {}).objects || [])
                .map((o) => String(o.name || "").trim()).filter(Boolean);
            return header("no masks") + datasetFactsCard(ds) +
                `<div class="inspector-card ds-treatments">` +
                (named.length
                    ? `<div class="ds-treat-hint">No masks stored yet. A first pass will add the ` +
                      `column and fill it with what the panel is looking for.</div>` +
                      `<button class="btn-small secondary ds-fill-gaps" type="button">` +
                      `Segment across all episodes…</button>`
                    : `<div class="ds-treat-hint">No masks stored yet. Name an object in the ` +
                      `Overlays panel to segment for it.</div>`) +
                `</div>`;
        }
        ensureTreatmentKeys();
        const staged = _stagedTreatments || {};
        // A flat row of mutually exclusive buttons, not a dropdown: the choice is
        // exclusive and there are four of them, so hiding three behind a menu buys
        // nothing -- and `tint` carries a COLOUR, which a <select> cannot show or
        // pick. This is the control that used to live in the overlay panel; it was
        // deleted with that panel for a SCOPE reason (a treatment is dataset-wide
        // and the panel had no scope), not because a dropdown was better. Here the
        // scope is right, so the control comes back.
        const row = (name, tr) => (
            `<div class="ds-treat-row">` +
            `<span class="ds-treat-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>` +
            treatWidget(tr, escapeHtml(name)) +
            `</div>`
        );
        const rows = vocab.labels
            .map((n) => row(n, staged[n] || vocab.treatments[n] || { key: "none", params: {} }))
            .join("");
        const bgTr = staged.__background__ || vocab.background || { key: "none", params: {} };
        const dirty = Object.keys(staged).length > 0;
        return (
            header(`masks · ${vocab.cameras} camera${vocab.cameras === 1 ? "" : "s"}` +
                   ` · applies to every camera`) +
            datasetFactsCard(ds) +
            `<div class="inspector-card ds-treatments">` +
            rows +
            row("background", bgTr).replace('data-label="background"', 'data-label="__background__"') +
            `<div class="ds-treat-actions"${dirty ? "" : ' style="display:none"'}>` +
            `<button class="btn-small ds-treat-save" type="button">Save</button>` +
            `<button class="btn-small secondary ds-treat-cancel" type="button">Cancel</button>` +
            `</div>` +
            `<button class="btn-small secondary ds-fill-gaps" type="button">Fill gaps across all episodes…</button>` +
            `</div>`
        );
    }


    // ── the treatment control (rendering lives in treatment_control.js) ─────
    //
    // Shared with the Run tab's live panel, which owns the same control with
    // different write semantics: here a change is STAGED and then saved, there
    // it is pushed to the worker and nothing is written.

    const TC = () => window.TreatmentControl;
    const _tintPop = { pop: null, get() { return (this.pop = this.pop || TC().makePopover()); } };

    function treatWidget(tr, label) {
        return TC().widget(tr, TREATMENT_KEYS, `data-label="${label}"`);
    }

    /** The treatment a label currently shows: staged if it has one, else stored. */
    function shownTreatment(label) {
        const ds = window.datasets?.[window.currentDataset];
        const vocab = ds && maskVocabulary(ds);
        const staged = _stagedTreatments || {};
        if (staged[label]) return staged[label];
        if (label === "__background__") return (vocab && vocab.background) || { key: "none", params: {} };
        return (vocab && vocab.treatments[label]) || { key: "none", params: {} };
    }

    function stageTreatment(label, key) {
        const cur = shownTreatment(label);
        const params = Object.assign({}, cur.params);
        if (key === "tint" && !params.color) params.color = TC().TINT_PRESETS[2];
        _stagedTreatments = _stagedTreatments || {};
        _stagedTreatments[label] = { key, params: (key === "tint" || key === "blur") ? params : {} };
        renderInspector();
    }

    // Repaints the chip in place rather than re-rendering: a re-render destroys
    // the open native colour picker mid-drag, which is how custom colours used
    // to get dropped.
    function stageTintColor(label, rgb) {
        const cur = shownTreatment(label);
        _stagedTreatments = _stagedTreatments || {};
        _stagedTreatments[label] = { key: "tint", params: Object.assign({}, cur.params, { color: rgb }) };
        const chip = document.querySelector(`.ds-treat[data-label="${CSS.escape(label)}"] .ds-tint-chip`);
        if (chip) chip.style.background = TC().rgbCss(rgb);
        const actions = document.querySelector(".ds-treat-actions");
        if (actions) actions.style.display = "";
    }

    /**
     * The dataset section's dropdowns and its in-place Save / Cancel.
     *
     * Committing here rather than on the timeline's bottom bar is the rule the
     * design draws by SCOPE: frame data goes to the bottom bar, dataset config
     * commits next to itself. Routing a dataset-wide write through a bar
     * labelled for the timeline is what made the previous panel ambiguous.
     *
     * A treatment is metadata every consumer reads, training included, so the
     * write still goes through the edits pipeline -- staged and applied in one
     * step, which is what "in place" means here: no pending entry survives the
     * click.
     */
    function wireDatasetTreatments(body) {
        body.querySelectorAll(".ds-treat-btn").forEach((btn) => {
            btn.addEventListener("click", (e) => {
                e.stopPropagation();
                const label = btn.closest(".ds-treat").getAttribute("data-label");
                stageTreatment(label, btn.dataset.key);
                if (btn.dataset.key === "tint") {
                    // Staging re-rendered the row, so anchor to the fresh button.
                    const fresh = document.querySelector(`.ds-treat[data-label="${CSS.escape(label)}"] .ds-treat-btn[data-key="tint"]`);
                    if (fresh) {
                        _tintPop.get().open(fresh, (shownTreatment(label).params || {}).color,
                            (rgb) => stageTintColor(label, rgb));
                    }
                }
            });
        });
        const cancel = body.querySelector(".ds-treat-cancel");
        if (cancel) {
            cancel.addEventListener("click", () => {
                _stagedTreatments = null;
                renderInspector();
            });
        }
        const save = body.querySelector(".ds-treat-save");
        if (save) save.addEventListener("click", () => commitDatasetTreatments());
        const fill = body.querySelector(".ds-fill-gaps");
        if (fill) fill.addEventListener("click", () => openFillGaps());
    }

    /**
     * The whole-dataset fill: run a segmentation over every episode, writing
     * only where a label is missing.
     *
     * The label set is PICKED here, not inherited from the vocabulary. The
     * vocabulary is the accumulated union of everything ever segmented
     * anywhere, so it answers "what has been seen", never "what should be
     * looked for everywhere" -- an episode with a `blue towel` that appears
     * nowhere else would otherwise send the job hunting for one across every
     * episode, for hours, returning false positives where it half-matches.
     *
     * The per-label episode count is what makes that choice obvious rather
     * than a memory test.
     */
    async function openFillGaps() {
        const datasetId = window.currentDataset;
        const ds = window.datasets?.[datasetId];
        const vocab = ds && maskVocabulary(ds);
        // With nothing stored, the panel's named objects ARE the label set: the
        // vocabulary cannot supply a menu it does not have yet, and the first
        // pass is what creates it.
        const live = ((window.Overlays?.dataQuery?.() || {}).objects || [])
            .map((o) => String(o.name || "").trim()).filter(Boolean);
        const seeding = !vocab;
        if (!vocab && !live.length) return;
        let cov = { labels: [], total_episodes: 0 };
        try {
            const r = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/masks/label-coverage`);
            if (r.ok) cov = await r.json();
        } catch (err) {
            _err("label coverage failed", err);
        }
        const seen = Object.fromEntries((cov.labels || []).map((x) => [x.name, x]));
        const total = cov.total_episodes || 0;
        // Ticked by default only where the label is already widespread. A label
        // seen in one episode is local to it; ticking it by default is the
        // mistake this dialog exists to prevent.
        const widespread = (n) => total > 0 && (seen[n]?.episodes || 0) > Math.max(1, total * 0.5);
        const prompts = (vocab && vocab.prompts) || {};
        // Seeded rows are ticked: the operator just typed them, which is the
        // intent this dialog otherwise has to infer from coverage.
        const labelList = vocab ? vocab.labels : live;
        const rows = labelList.map((n) => {
            const eps = seen[n]?.episodes || 0;
            return (
                `<label class="fg-row"><input type="checkbox" data-label="${escapeHtml(n)}"` +
                `${seeding || widespread(n) ? " checked" : ""}> ` +
                `<span class="fg-name">${escapeHtml(n)}</span>` +
                `<span class="fg-prompt">${escapeHtml(prompts[n] || n)}</span>` +
                `<span class="fg-seen">seen in ${eps}/${total} ep</span></label>`
            );
        }).join("");

        if (document.querySelector(".fg-backdrop")) return;   // already open
        const back = document.createElement("div");
        back.className = "fg-backdrop";
        // This is the only dataset-wide way to add masks, so it is also the
        // confirmation for one: it has to say what it will run over, with what,
        // and what it will not touch, before OK is available.
        const q = window.Overlays?.dataQuery?.() || {};
        const dsCams = (q.cameras && q.cameras.length) ? q.cameras : (ds.camera_keys || []);
        const camNames = dsCams.map((k) => k.split(".").pop()).join(", ");
        back.innerHTML =
            `<div class="fg-modal"><h3>Fill gaps across ${total} episodes</h3>` +
            `<div class="fg-rows">${rows}</div>` +
            `<div class="fg-summary">` +
            `<div><b>Runs over</b> ${total} episode${total === 1 ? "" : "s"} of ` +
            `<b>${escapeHtml(datasetId)}</b>, cameras: ${escapeHtml(camNames || "all")}</div>` +
            `<div><b>Fills</b> <span class="fg-picked-count">0</span> label(s), only where that label is ` +
            `<b>absent</b> — detected and disabled masks are left untouched</div>` +
            `<div><b>Leaves alone</b> the stored effects and the video: treatments stay a recipe ` +
            `you can change afterwards, and nothing is re-encoded</div>` +
            `<div class="fg-est"></div>` +
            `</div>` +
            `<div class="fg-actions">` +
            `<button class="btn-small secondary fg-cancel" type="button">Cancel</button>` +
            `<button class="btn-small fg-run" type="button">OK</button>` +
            `</div></div>`;
        document.body.appendChild(back);
        // Escape is watched on the document: the backdrop never takes focus, so
        // a keydown bound to it is never delivered. Removed with the dialog, or
        // it keeps firing at whatever is on screen next.
        const onKey = (e) => { if (e.key === "Escape") close(); };
        const close = () => { document.removeEventListener("keydown", onKey); back.remove(); };
        document.addEventListener("keydown", onKey);
        const okBtn = back.querySelector(".fg-run");
        const picked = () => [...back.querySelectorAll(".fg-rows input:checked")]
            .map((c) => c.getAttribute("data-label"));
        // The count and the estimate follow the ticks, so the dialog always
        // describes the run OK would start rather than the one it opened with.
        const sync = () => {
            const n = picked().length;
            back.querySelector(".fg-picked-count").textContent = String(n);
            okBtn.disabled = !n;
            okBtn.title = n ? "" : "Tick at least one label";
            const est = back.querySelector(".fg-est");
            const perFrame = q.computeMs;
            const frames = (window.episodes?.[datasetId] || [])
                .reduce((a, e) => a + (e.length || 0), 0);
            // With no measurement the line used to render empty, so the dialog
            // simply had no estimate and no reason for not having one -- which
            // reads as a missing feature rather than a missing measurement.
            est.textContent = (perFrame && frames && dsCams.length)
                ? `Roughly ${_fmtDur(perFrame * frames * dsCams.length / 1000)}, ` +
                  `from the live preview's measured ${perFrame.toFixed(0)} ms/frame/camera (excludes model load)`
                : "No time estimate yet — it comes from the live preview's measured rate. "
                  + "Turn a segmenter on and let it run a few frames to get one.";
        };
        back.querySelectorAll(".fg-rows input").forEach((c) => c.addEventListener("change", sync));
        sync();
        back.addEventListener("click", (e) => { if (e.target === back) close(); });
        back.querySelector(".fg-cancel").addEventListener("click", close);
        okBtn.addEventListener("click", async () => {
            const labels = picked();
            if (!labels.length) return;   // OK is disabled, but a stray Enter must not run
            close();
            await runFillGaps(datasetId, labels, total);
        });
    }

    const _fmtDur = (s) => (s < 90 ? `~${Math.max(1, Math.round(s))}s` : `~${Math.round(s / 60)} min`);

    async function runFillGaps(datasetId, labels, total) {
        const eps = (window.episodes?.[datasetId] || []).map((e) => e.episode_index);
        // Through the shared job runner, not a bare fetch. It carries the 409
        // consent handshake, the progress polling, the report when a pass finds
        // nothing on a camera, and the cache invalidation afterwards -- all of
        // which this path went without while it posted for itself.
        const run = window.OverlayStream?.runMaskJob;
        if (!run) {
            window.setStatus && window.setStatus("The overlay module is not ready");
            return;
        }
        window.setStatus && window.setStatus(
            `Filling ${labels.length} label${labels.length === 1 ? "" : "s"} across ${total} episodes…`
        );
        try {
            await run(null, eps, {
                // Confirmed in the dialog that just closed: it named the
                // episodes, the cameras and the labels before OK was available.
                confirmed: true,
                overwriteOk: true,
                // Treatment is not an input -- it comes from the dataset's own
                // recipe, and the writer prefers what is stored.
                objects: labels.map((n) => ({ name: n, sign: "+", treatment: { key: "none" } })),
                // This panel has no button to write into, and a dataset-wide
                // fill is the longest-running thing the GUI starts, so it needs
                // the progress more than the panel's own save does.
                onProgress: (msg) => window.setStatus && window.setStatus(msg),
            });
        } catch (err) {
            _err("fill gaps failed", err);
        }
    }

    async function commitDatasetTreatments() {
        const datasetId = window.currentDataset;
        const ds = window.datasets?.[datasetId];
        const vocab = ds && maskVocabulary(ds);
        if (!vocab || !_stagedTreatments) return;
        const staged = _stagedTreatments;
        // The whole map, not just what changed: the endpoint records the
        // intended end state for every label, and sending a subset would read
        // as "the others have no treatment".
        const treatments = {};
        for (const name of vocab.labels) {
            treatments[name] = staged[name] || vocab.treatments[name] || { key: "none", params: {} };
        }
        const background = staged.__background__ || vocab.background || { key: "none", params: {} };
        try {
            const res = await fetch("/api/edits/mask-treatments", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dataset_id: datasetId, treatments, background }),
            });
            if (!res.ok) {
                const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
                window.setStatus && window.setStatus(`Treatment edit failed: ${detail}`);
                return;
            }
            // Applied immediately: this is a config edit, and the design says
            // config commits in place rather than waiting on the bottom bar.
            await fetch(`/api/edits/apply?dataset_id=${encodeURIComponent(datasetId)}`, { method: "POST" });
            _stagedTreatments = null;
            if (typeof window.refreshPendingEdits === "function") await window.refreshPendingEdits();
            // The recipe changed, so every composited tile is now describing the
            // previous one.
            window.MaskOverlay?.invalidate?.(datasetId);
            await refreshFromServer(datasetId);
            if (typeof window.loadAllFrames === "function") window.loadAllFrames(window.currentFrame || 0);
            window.setStatus && window.setStatus("Treatments saved");
        } catch (err) {
            _err("treatment commit failed", err);
        }
    }

    function renderInspectorEmpty(datasetId) {
        const body = document.getElementById("inspector-body");
        if (!body) return;
        if (!datasetId || !window.datasets || !window.datasets[datasetId]) {
            body.innerHTML = '<div class="empty-state">Open a dataset to inspect its features</div>';
            return;
        }
        const ds = window.datasets[datasetId];
        // The same section, not a second rendering of the same facts: two places
        // to edit when a fact is added, and two ways for the panel to look.
        body.innerHTML = renderDatasetSection(datasetId, ds) +
            `<div class="inspector-summary">` +
            `<div style="color:#888; font-style:italic;">Click or drag inside the timeline area to edit feature values.</div>` +
            `</div>`;
    }

    function renderInspector() {
        const datasetId = window.currentDataset;
        const ds = window.datasets && window.datasets[datasetId];
        if (!ds) {
            // No dataset open → fallback to the schema-only empty state.
            if (datasetId) renderInspectorEmpty(datasetId);
            return;
        }
        const body = document.getElementById("inspector-body");
        if (!body) return;

        // We always render both sections. Per-episode is always editable
        // (it doesn't depend on frame selection). Per-frame is editable
        // only when the user has a selection — otherwise the cards show
        // the current playhead frame's values with widgets disabled, so
        // the schema is always visible (consistent with how the user
        // expects the inspector to behave when scrubbing).
        const epIdx = (selection && selection.episodeIndex != null)
            ? selection.episodeIndex
            : window.currentEpisode;
        if (epIdx == null) {
            // Dataset open but no episode selected yet.
            renderInspectorEmpty(datasetId);
            return;
        }

        const featuresSchema = ds.features_schema || {};
        const epLen = (window.totalFrames > 0) ? window.totalFrames : 0;

        // Per-frame "effective range": the selection if there is one,
        // otherwise the playhead frame as a [N, N+1) single-frame view.
        const hasSelection = selection != null && selection.episodeIndex === epIdx;
        const playhead = window.currentFrame ?? 0;
        const fFrom = hasSelection ? selection.frameFrom : playhead;
        const fTo = hasSelection ? selection.frameTo : playhead + 1;
        const k = fTo - fFrom;
        const m = fTo - 1;
        const originRow = hasSelection ? selection.originRow : null;

        const perFrameCards = [];
        const perEpisodeCards = [];
        for (const [name, ft] of Object.entries(featuresSchema)) {
            const editable = isEditable(name, ft);
            // Read-only features are otherwise timeline-only, but per-episode ones
            // are hidden from the timeline — one constant band across every frame
            // wastes a row — so a feature that is both would appear nowhere at all.
            // `task` is the only one today; the gate is on the combination, not on it.
            if (!editable && !ft.is_per_episode) continue;
            if (ft.is_per_episode) {
                // Covers the whole episode by definition.
                perEpisodeCards.push(
                    renderFeatureCard(name, ft, 0, epLen, datasetId, epIdx, originRow,
                        { editable })
                );
            } else {
                // Editable only when the user has actively selected a range.
                perFrameCards.push(
                    renderFeatureCard(name, ft, fFrom, fTo, datasetId, epIdx, originRow,
                        { editable: hasSelection })
                );
            }
        }

        const frameTitle = hasSelection
            ? (k === 1 ? `frame ${fFrom}` : `frames ${fFrom}…${m} (${k} frames)`)
            : `frame ${playhead}`;
        const frameMeta = hasSelection
            ? "Frame-specific features · drag-select on any row to change range"
            : "Frame-specific features · drag-select on any row to edit";

        const sections = [];
        // Dataset first: it is the broadest scope and the one always present.
        // The section itself comes from the branch below; this branch adds the
        // mask recipe into it.
        const dsSection = renderDatasetSection(datasetId, ds);
        if (dsSection) sections.push(dsSection);
        // Per-episode goes ABOVE per-frame: episode is broader context.

        if (perEpisodeCards.length) {
            sections.push(
                `<div class="inspector-section-header">` +
                `<div class="sel-title">Episode ${epIdx}` +
                (epLen > 0 ? ` (${epLen} frames)` : "") + `</div>` +
                `<div class="sel-meta">Episode-only features</div>` +
                `</div>` +
                perEpisodeCards.join("")
            );
        }
        if (perFrameCards.length) {
            sections.push(
                `<div class="inspector-section-header">` +
                `<div class="sel-title">${frameTitle}</div>` +
                `<div class="sel-meta">${frameMeta}</div>` +
                `</div>` +
                perFrameCards.join("")
            );
        }
        if (!sections.length) {
            sections.push(
                '<div class="empty-state">' +
                'No editable features. action / observation.* / images / DEFAULT_FEATURES are read-only in V1.' +
                '</div>'
            );
        }

        body.innerHTML = sections.join("");

        // Wire edit widgets (auto-staging on change). Disabled widgets are
        // skipped naturally — they have no listeners that could fire.
        wireWidgets(body);
        wireDatasetTreatments(body);

        // Inspector-card delete buttons share the same handler / confirm
        // flow as the timeline-row delete; the only difference is the
        // mount point. Per-episode features are hidden from the timeline
        // rows unless pinned, so this is the primary delete path for them.
        body.querySelectorAll(".card-delete-btn").forEach(btn => {
            btn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                deleteFeature(btn.getAttribute("data-feature"));
            });
        });
    }

    function renderFeatureCard(name, ft, frameFrom, frameTo, datasetId, episodeIndex, originRow, opts) {
        // Schema-level read-only check (action / observation.* / images / DEFAULT_FEATURES).
        const schemaEditable = isEditable(name, ft);
        // Caller can downgrade to read-only (used for per-frame cards when no
        // selection — they show current playhead values but the user must
        // drag-select to actually edit).
        const callerEditable = (opts && opts.editable === false) ? false : true;
        const editable = schemaEditable && callerEditable;
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

        let widget = "";
        if (!schemaEditable) {
            // Read-only features (action / observation.* / images / DEFAULT_FEATURES)
            // still need to show their actual values so the user can inspect
            // recorded data. Earlier this rendered just a "read-only" placeholder
            // tag — that left high-DOF vectors completely unreadable.
            widget = renderReadOnlyView(name, ft, effFrom, effTo, datasetId, episodeIndex);
        } else if (!callerEditable) {
            // Per-frame card without a selection: render the widget but
            // disabled, so the user sees current values without being able
            // to edit until they drag-select.
            widget = renderWidgetForType(name, ft, effFrom, effTo, datasetId, episodeIndex, { disabled: true });
        } else {
            widget = renderWidgetForType(name, ft, effFrom, effTo, datasetId, episodeIndex);
        }

        // Range shown next to the feature name. Declared bounds (info.json
        // ``min`` / ``max``) take precedence — they're authoritative and
        // enforced. Otherwise fall back to the dataset-wide observed extrema
        // from meta/stats.json. The two get distinct flags so the user knows
        // which one they're looking at.
        let observedRange = "";
        // A bitset has no meaningful extrema: the smallest and largest bit
        // patterns present say nothing about a range, and rendering them as
        // one invites reading flag 2 as "twice flag 1".
        const isBitset = Array.isArray(ft.flags) && ft.flags.length > 0;
        if (isBitset) {
            observedRange =
                `<span class="card-observed-range" title="flags declared in info.json">` +
                `${ft.flags.length} flag${ft.flags.length === 1 ? "" : "s"}</span>`;
        } else if (ft.declared_min != null && ft.declared_max != null) {
            observedRange =
                `<span class="card-observed-range" title="declared bounds (info.json)">` +
                `[${formatNumber(ft.declared_min)} … ${formatNumber(ft.declared_max)}]</span>`;
        } else if (ft.observed_min != null && ft.observed_max != null) {
            observedRange =
                `<span class="card-observed-range" title="observed across the dataset (meta/stats.json)">` +
                `[${formatNumber(ft.observed_min)} … ${formatNumber(ft.observed_max)}]</span>`;
        }

        // Inspector-card delete (✕). Same eligibility as the timeline-row
        // ✕ — uses the shared isDeletable predicate. This is the only
        // delete affordance for per-episode features (which are hidden
        // from the timeline rows by default unless pinned), so it's the
        // primary path for those.
        //
        // Grouped with the dtype flag inside .card-header-right so they
        // sit next to each other on the right edge instead of overlapping
        // (the original `position: absolute` placement collided with the
        // dtype text — user-reported).
        const cardDeleteBtn = isDeletable(name, ft)
            ? `<button class="card-delete-btn" data-feature="${escapeHtml(name)}" type="button" title="Delete this feature column">✕</button>`
            : "";
        return `
            <div class="feature-card ${focused ? "focused" : ""} ${editable ? "" : "readonly"}" data-feature="${escapeHtml(name)}">
                <div class="card-header">
                    <span class="card-name">${escapeHtml(name)}${observedRange}${headerExtras}</span>
                    <span class="card-header-right">
                        <span class="card-dtype">${escapeHtml(dtype)}[${shape}]</span>
                        ${cardDeleteBtn}
                    </span>
                </div>
                ${schemaEditable
                    ? `<div class="card-summary">${cardSummary(name, ft, datasetId, episodeIndex, effFrom, effTo)}</div>`
                    : ""}
                <div class="card-widget">${widget}</div>
            </div>
        `;
    }

    // Previews what an edit over [from, to) would overwrite. Editable cards only —
    // with no edit to preview it just restates the value shown beneath it.
    function cardSummary(name, ft, datasetId, episodeIndex, frameFrom, frameTo) {
        return summarizeSlice(getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo), ft);
    }

    // Split from the lookup above so the formatting is a pure function of the
    // values and can be unit-tested without a browser.
    function summarizeSlice(slice, ft) {
        if (slice === null || !slice.length) return "&nbsp;";
        // A bitset's value is a bit pattern, not a magnitude: "min 0 max 2"
        // reads as a range when 2 is simply the second flag. Count the frames
        // carrying each flag instead, which is the question being asked.
        if (ft && Array.isArray(ft.flags) && ft.flags.length) {
            const nums = slice.filter(v => typeof v === "number").map(v => Math.round(v));
            const carried = ft.flags
                .map((flag, bit) => [flag, nums.filter(v => bitIsSet(v, bit)).length])
                .filter(([, n]) => n > 0)
                .map(([flag, n]) => `${escapeHtml(flag)} ${n}/${nums.length}`);
            return carried.length ? carried.join(", ") : `no flags set (${nums.length} frames)`;
        }
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

    // Read-only display for features that aren't editable in V1 (action /
    // observation.* / images / DEFAULT_FEATURES). Renders the current
    // frame's value in a typed format so the user can still inspect recorded
    // data — the schema row already shows the row flag "read-only".
    function renderReadOnlyView(name, ft, frameFrom, frameTo, datasetId, episodeIndex) {
        const slice = getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo);
        return readOnlyValueHtml(ft, slice, frameFrom);
    }

    // Split from the lookup above so the formatting is a pure function of the
    // values and can be unit-tested without a browser.
    function readOnlyValueHtml(ft, slice, frameFrom) {
        const dtype = ft.dtype || "";
        if (dtype === "image" || dtype === "video") {
            return `<span class="card-readonly-tag">${escapeHtml(dtype)} (rendered in viewer)</span>`;
        }
        if (slice == null || !slice.length) {
            return `<span class="card-readonly-tag">no data in selection</span>`;
        }
        // Pick a representative frame: first frame of the selection.
        const sample = slice[0];
        const isVector = Array.isArray(sample);
        // Multi-frame range: show the sample plus a hint that it's a snapshot.
        // Not when every frame carries the same value — the displayed value is
        // then the whole answer, and pointing at one frame of 385 suggests the
        // rest might differ. That reads as a contradiction on an episode-wide
        // feature like the decoded `task` instruction, which cannot differ.
        const isUniform = !isVector && slice.every(v => v === sample);
        const rangeHint = (slice.length > 1 && !isUniform)
            ? `<span class="readonly-range-hint">(frame ${frameFrom} of ${slice.length})</span>`
            : "";
        if (isVector) {
            // shape [N] — render every component as label/value pairs. Long
            // vectors get a scrollable container; component names from
            // ft.names take precedence over numeric indices when present.
            const names = Array.isArray(ft.names) ? ft.names : null;
            const cells = sample.map((v, i) => {
                const label = names?.[i] != null ? names[i] : `[${i}]`;
                const valStr = (typeof v === "number") ? formatNumber(v) : escapeHtml(String(v));
                return `<div class="readonly-cell"><span class="readonly-label">${escapeHtml(String(label))}</span><span class="readonly-value">${valStr}</span></div>`;
            }).join("");
            return `<div class="readonly-vector">${cells}</div>${rangeHint}`;
        }
        // Scalar.
        let valStr;
        if (Array.isArray(ft.flags) && ft.flags.length && typeof sample === "number") {
            // The stored integer is a bit pattern; "3.000" says nothing. Name
            // the flags it sets, as the editable widget does.
            const on = ft.flags.filter((_, b) => bitIsSet(sample, b));
            valStr = on.length ? on.map(escapeHtml).join(", ") : "(no flags)";
        } else if (typeof sample === "number") valStr = formatNumber(sample);
        else if (typeof sample === "boolean") valStr = sample ? "✓ true" : "✗ false";
        else if (typeof sample === "string") valStr = `"${escapeHtml(sample)}"`;
        else valStr = escapeHtml(String(sample));
        return `<div class="readonly-scalar">${valStr}</div>${rangeHint}`;
    }

    function renderWidgetForType(name, ft, frameFrom, frameTo, datasetId, episodeIndex, opts) {
        const dtype = ft.dtype || "";
        const shape = ft.shape || [];
        const isScalar = (shape.length === 0) || (shape.length === 1 && shape[0] === 1);
        const disabledAttr = (opts && opts.disabled) ? " disabled" : "";

        // Tri-state success widget: int8 per-episode named "success".
        // -1 = failure, 0 = unmarked, +1 = success. Renders as a three-button
        // segment control. The per-episode coercion already widened the
        // [from,to) range to the full episode in the calling card.
        if (name === "success" && dtype === "int8" && ft.is_per_episode && isScalar) {
            return renderSuccessSegment(name, frameFrom, frameTo, datasetId, episodeIndex, opts);
        }

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
                ` data-initial="${dataInitial}"${checkedAttr}${disabledAttr}>`
            );
        }
        if (dtype === "string") {
            return `<input type="text" data-widget="string" data-feature="${escapeHtml(name)}" placeholder="(value for range)"${disabledAttr}>`;
        }
        // Bitset feature (int + flags). Each bit is an independent boolean, so
        // this is a checkbox per flag rather than one control for the cell.
        // A box is indeterminate when only some frames in the selection carry
        // its flag -- a two-state box would have to lie about one of them.
        if (isScalar && Array.isArray(ft.flags) && ft.flags.length > 0) {
            const slice = getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo);
            const values = (slice || []).filter(v => typeof v === "number").map(v => Math.round(v));
            const boxes = ft.flags.map((flag, bit) => {
                const carrying = values.filter(v => bitIsSet(v, bit)).length;
                const state = !values.length ? "none"
                    : carrying === values.length ? "all"
                    : carrying === 0 ? "none" : "mixed";
                const checked = state === "all" ? " checked" : "";
                // The rename control sits outside the <label> deliberately: a
                // click anywhere inside a label toggles its checkbox, so a
                // rename affordance in there would flag the frames on its way
                // to opening the prompt.
                const rename = disabledAttr
                    ? ""
                    : `<button type="button" class="flag-rename" title="Rename this flag"` +
                      ` data-feature="${escapeHtml(name)}" data-bit="${bit}"` +
                      ` data-flag="${escapeHtml(flag)}">✎</button>`;
                return (
                    `<div class="flag-row">` +
                    `<label class="flag-box" title="bit ${bit}">` +
                    `<input type="checkbox" data-widget="flag" data-feature="${escapeHtml(name)}"` +
                    ` data-flag="${escapeHtml(flag)}" data-state="${state}"${checked}${disabledAttr}>` +
                    `<span>${escapeHtml(flag)}</span></label>${rename}</div>`
                );
            });
            // Growing the vocabulary lives beside the flags rather than in the
            // Add-column dialog: by the time you find you need another flag,
            // the column already exists. Appending takes the next bit and
            // rewrites no data, so it is safe to offer inline.
            const add = disabledAttr
                ? ""
                : `<button type="button" class="flag-add" data-feature="${escapeHtml(name)}">+ flag</button>`;
            return `<div class="flag-boxes">${boxes.join("")}${add}</div>`;
        }

        // Categorical integer feature (int + names). The on-disk value is the
        // index ``[0, len(names))``; the user picks by flag. Detected before
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
                `<select data-widget="categorical" data-feature="${escapeHtml(name)}"${disabledAttr}>` +
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
                <input type="range" data-widget="scalar-slider" data-feature="${escapeHtml(name)}" min="${lo}" max="${hi}" step="${step}"${initialSliderAttr}${disabledAttr}>
                <input type="number" data-widget="scalar-number" data-feature="${escapeHtml(name)}" step="${step}"${initialValueAttr} placeholder="(value)"${disabledAttr}>
            `;
        }
        if (shape.length === 1 && shape[0] > 0 && shape[0] <= 8) {
            // Small numeric vector → row of inputs.
            const inputs = [];
            for (let i = 0; i < shape[0]; i++) {
                inputs.push(`<input type="number" data-widget="vector-cell" data-feature="${escapeHtml(name)}" data-cell="${i}" step="any"${disabledAttr}>`);
            }
            return `<div class="vector-row">${inputs.join("")}</div>`;
        }
        // Large vector / matrix → JSON textarea.
        return `<textarea data-widget="json" data-feature="${escapeHtml(name)}" placeholder="JSON value (matches dtype/shape)"${disabledAttr}></textarea>`;
    }

    function renderSuccessSegment(name, frameFrom, frameTo, datasetId, episodeIndex, opts) {
        // Three-button segment control: -1 / 0 / +1. Determines "active"
        // from the merged slice (uniform value → that button is active;
        // mixed value → none active). Disabled state matches other widgets.
        const disabledAttr = (opts && opts.disabled) ? ' disabled class="disabled"' : "";
        const slice = getMergedSlice(name, datasetId, episodeIndex, frameFrom, frameTo);
        let uniformValue = null;
        if (slice && slice.length) {
            const first = slice[0];
            if (slice.every(v => v === first)) uniformValue = first;
        }
        const states = [
            { value: -1, label: "✗ Failure", cls: "failure" },
            { value: 0,  label: "— Unmarked", cls: "unmarked" },
            { value: 1,  label: "✓ Success", cls: "success" },
        ];
        const buttons = states.map(s => {
            const isActive = (uniformValue === s.value);
            const activeCls = isActive ? " active" : "";
            return (
                `<button type="button" data-widget="success-segment" data-feature="${escapeHtml(name)}"` +
                ` data-value="${s.value}" class="${s.cls}${activeCls}"${disabledAttr}>` +
                `${escapeHtml(s.label)}</button>`
            );
        });
        return `<div class="success-segment">${buttons.join("")}</div>`;
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

            card.querySelectorAll(".flag-rename").forEach(btn => {
                btn.addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    ev.preventDefault();
                    renameFlag(
                        btn.getAttribute("data-feature"),
                        parseInt(btn.getAttribute("data-bit"), 10),
                        btn.getAttribute("data-flag"),
                        btn.closest(".flag-row")
                    );
                });
            });

            card.querySelectorAll(".flag-add").forEach(btn => {
                btn.addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    appendFlag(btn.getAttribute("data-feature"), btn);
                });
            });

            // Same idea per flag: "some frames carry this" is neither on nor off.
            card.querySelectorAll('[data-widget="flag"]').forEach(box => {
                if (box.getAttribute("data-state") === "mixed") box.indeterminate = true;
            });

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
                if (kind === "flag") {
                    w.addEventListener("change", () => {
                        // A click on an indeterminate box resolves it: the
                        // browser reports checked, and the operator means the
                        // whole selection, which is exactly what we stage.
                        w.indeterminate = false;
                        stageFlagEdit(featureName, w.getAttribute("data-flag"), w.checked);
                    });
                } else if (kind === "bool") {
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
                } else if (kind === "success-segment") {
                    // Tri-state success: stage as int8 (-1 / 0 / +1).
                    w.addEventListener("click", () => {
                        if (w.disabled) return;
                        const v = parseInt(w.getAttribute("data-value"), 10);
                        stageFeatureEdit(featureName, v);
                    });
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

    // Resolve the effective range for a feature edit:
    //   1. an explicit drag-select wins (whatever the user picked)
    //   2. otherwise, per-episode features synthesize whole-episode range — the
    //      backend would coerce to that anyway, so a bare click on a per-episode
    //      widget (e.g. ✓ Success) without a prior selection should still stage
    //   3. per-frame features without a selection: return null (caller no-ops
    //      with a status message — silently dropping the click is the bug
    //      previously seen on per-episode widgets)
    function _resolvedRangeFor(featureName) {
        if (selection) return selection;
        const datasetId = window.currentDataset;
        const epIdx = window.currentEpisode;
        if (datasetId == null || epIdx == null) return null;
        const ft = window.datasets?.[datasetId]?.features_schema?.[featureName];
        if (!ft?.is_per_episode) return null;
        const ep = window.episodes?.[datasetId]?.find(e => e.episode_index === epIdx);
        if (!ep || !ep.length) return null;
        return {
            datasetId,
            episodeIndex: epIdx,
            frameFrom: 0,
            frameTo: ep.length,
            originRow: featureName,
        };
    }

    // Naming a flag is an in-place edit on the row it belongs to, not a
    // browser prompt. Two reasons beyond taste: a prompt cannot show which flag
    // it is editing beside the ones it is not, and it has nowhere to put a
    // rejection -- the server refuses a duplicate name, and with a prompt that
    // refusal arrives after the dialog is gone, so the edit looks accepted.
    // The editor stays open on an error with the message beside the input, so a
    // duplicate is a correctable state rather than a lost one.
    //
    // Unlike the note editor this does not commit on blur. A note is free text
    // where losing keystrokes is the only failure mode; a flag name is
    // validated, so committing on the way out would fire a request whose answer
    // the user is no longer looking at. Enter and Save commit, Escape and
    // Cancel discard, and clicking away leaves the editor open and untouched.
    function flagEditorHtml(value) {
        return (
            `<input type="text" class="flag-editor-input" value="${escapeHtml(value)}"` +
            ` spellcheck="false" placeholder="flag name">` +
            `<button type="button" class="flag-editor-btn flag-editor-save">Save</button>` +
            `<button type="button" class="flag-editor-btn flag-editor-cancel">Cancel</button>` +
            `<span class="flag-editor-error" role="alert"></span>`
        );
    }

    /** Open the name editor on `row`, committing through `submit(name)`.
     *
     * Preconditions:
     *   `row` is a `.flag-row`, either an existing flag's or a blank one added
     *   for a new flag. `submit` resolves to null on success, or to a message
     *   to show beside the input.
     *
     * Postconditions:
     *   On success the panel is re-rendered and the editor is gone with it. On
     *   failure the editor is still open, still holding what was typed, with
     *   the message visible. `discard` runs only if the user cancels.
     */
    function openFlagEditor(row, current, submit, discard) {
        const previous = row.innerHTML;
        row.classList.add("flag-row-editing");
        row.innerHTML = flagEditorHtml(current);
        const input = row.querySelector(".flag-editor-input");
        const errorEl = row.querySelector(".flag-editor-error");
        const save = row.querySelector(".flag-editor-save");
        const cancel = row.querySelector(".flag-editor-cancel");
        input.focus();
        input.select();

        const restore = () => {
            row.classList.remove("flag-row-editing");
            if (discard) discard();
            else row.innerHTML = previous;
        };
        const setBusy = (busy) => {
            for (const el of [input, save, cancel]) el.disabled = busy;
        };
        const commit = async () => {
            const name = input.value.trim();
            if (!name) {
                // Refused here rather than at the server: it is the one
                // rejection statable without a round trip, and the add case has
                // no prior value to restore if the request were sent and failed.
                errorEl.textContent = "A flag needs a name.";
                input.focus();
                return;
            }
            if (name === current) {
                restore();
                return;
            }
            errorEl.textContent = "";
            setBusy(true);
            const problem = await submit(name);
            if (problem === null) return;  // the panel re-rendered under us
            setBusy(false);
            errorEl.textContent = problem;
            input.focus();
            input.select();
        };

        save.addEventListener("click", commit);
        cancel.addEventListener("click", restore);
        input.addEventListener("keydown", (ev) => {
            ev.stopPropagation();  // the timeline owns arrows and Delete
            if (ev.key === "Enter") {
                ev.preventDefault();
                commit();
            } else if (ev.key === "Escape") {
                ev.preventDefault();
                restore();
            }
        });
    }

    /** POST/PATCH a vocabulary edit. Returns null on success, else a message. */
    async function _submitFlagName(url, method, flag, okStatus) {
        const datasetId = window.currentDataset;
        if (!datasetId) return "No dataset is open.";
        try {
            const res = await fetch(url, {
                method,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ flag }),
            });
            if (!res.ok) {
                return (await res.json().catch(() => ({}))).detail || res.statusText;
            }
            const payload = await res.json();
            // Only info.json changed, so the frames need no reload -- but the
            // schema the panel renders from does.
            if (payload && payload.info) window.datasets[datasetId] = payload.info;
            renderFeatureRows();
            renderInspector();
            window.setStatus && window.setStatus(okStatus);
            return null;
        } catch (err) {
            _err("flag vocabulary edit failed", err);
            return err.message;
        }
    }

    function renameFlag(featureName, bit, current, row) {
        const datasetId = window.currentDataset;
        if (!datasetId || !row) return;
        openFlagEditor(
            row,
            current,
            (flag) => _submitFlagName(
                `/api/datasets/${encodeURIComponent(datasetId)}/features/` +
                `${encodeURIComponent(featureName)}/flags/${bit}`,
                "PATCH",
                flag,
                `Renamed flag: ${current} -> ${flag}`
            ),
            null
        );
    }

    function appendFlag(featureName, addBtn) {
        const datasetId = window.currentDataset;
        if (!datasetId || !addBtn) return;
        // A blank row where the flag will be, rather than a dialog over the
        // panel: the new name is read in the column of the names it must be
        // unique against.
        const row = document.createElement("div");
        row.className = "flag-row flag-row-new";
        addBtn.parentElement.insertBefore(row, addBtn);
        addBtn.disabled = true;
        openFlagEditor(
            row,
            "",
            (flag) => _submitFlagName(
                `/api/datasets/${encodeURIComponent(datasetId)}/features/` +
                `${encodeURIComponent(featureName)}/flags`,
                "POST",
                flag,
                `Added flag: ${flag}`
            ),
            () => {
                row.remove();
                addBtn.disabled = false;
            }
        );
    }

    /**
     * The segment a pointer is over, clipped to the selection. Null when the
     * pointer is on an absent stretch, outside any selection, or on a lane
     * whose label the click cannot act on.
     *
     * Clipping to the selection is what makes the scope positional: the click
     * acts on what you selected AND what you pointed at, never on the whole
     * run that happens to extend past the selection's edge.
     */
    function maskSegmentAt(featureName, ft, laneIndex, frame) {
        const sel = selection;
        if (!sel || sel.originRow !== featureName) return null;
        // THE POINTER must be inside the selection, not merely the segment.
        // Clipping a segment to the selection is not the same test: a segment
        // running from 0 to 40 still overlaps a selection of 0..10 when the
        // pointer is at frame 30, so a click far outside the range was toggling
        // the range. That is the reported "my click outside the range toggled
        // it", and it is why the edits landed on frames nobody clicked.
        if (frame < sel.frameFrom || frame >= sel.frameTo) return null;
        // And a toggle needs a DRAGGED range: clicking the row is how you seek,
        // which leaves a one-frame selection behind, and one frame is not a
        // change anyone can see.
        if (sel.frameTo - sel.frameFrom < 2) return null;
        const cached = seriesCache.get(`${sel.datasetId}:${sel.episodeIndex}`);
        if (!cached) return null;
        // The MERGED view, not the stored one: hit-testing the stored series
        // would make a second click re-stage the first action rather than
        // toggle it back, because the segment would still read as detected.
        const [enabled, muted] = applyPendingMaskEdits(
            featureName,
            ft.mask_labels || [],
            cached.series[featureName] || [],
            cached.series[`${featureName}__disabled`] || [],
            cached.length,
        );
        const seg = maskSegments(enabled, muted, laneIndex, cached.length)
            .find((s) => frame >= s.from && frame < s.to);
        if (!seg || seg.state === "absent") return null;
        const from = Math.max(seg.from, sel.frameFrom);
        const to = Math.min(seg.to, sel.frameTo);
        if (from >= to) return null;
        return { ...seg, from, to, label: (ft.mask_labels || [])[laneIndex] };
    }

    /**
     * Stage one segment edit. `action` is "toggle" or "delete".
     *
     * A toggle's direction comes from the segment's own state rather than from
     * a control, which is why there is no tri-state to resolve: the thing
     * clicked is all one state by construction.
     */
    async function stageMaskSegmentEdit(featureName, seg, action) {
        const camera = maskCameraOf(featureName);
        if (!camera || !seg) return;
        const verb = action === "delete" ? "delete" : (seg.state === "detected" ? "disable" : "enable");
        try {
            const res = await fetch("/api/edits/mask-range", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    dataset_id: selection.datasetId,
                    episode_index: selection.episodeIndex,
                    camera,
                    label: seg.label,
                    from_frame: seg.from,
                    to_frame: seg.to,
                    action: verb,
                }),
            });
            if (!res.ok) {
                const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
                window.setStatus && window.setStatus(`Mask edit failed: ${detail}`);
                return;
            }
            const n = seg.to - seg.from;
            window.setStatus && window.setStatus(
                `${seg.label}: ${verb}d ${n} frame${n === 1 ? "" : "s"} — staged`
            );
            if (typeof window.refreshPendingEdits === "function") await window.refreshPendingEdits();
        } catch (err) {
            _err("mask segment edit failed", err);
        }
    }

    /** The camera a mask column describes — the inverse of `mask_feature_of`. */
    function maskCameraOf(featureName) {
        const p = "masks.";
        if (!featureName.startsWith(p)) return null;
        return `observation.images.${featureName.slice(p.length)}`;
    }

    async function stageFlagEdit(featureName, flag, ticked) {
        const sel = _resolvedRangeFor(featureName);
        if (!sel) {
            window.setStatus && window.setStatus("Drag-select a frame range first");
            return;
        }
        const body = {
            dataset_id: sel.datasetId,
            episode_index: sel.episodeIndex,
            feature: featureName,
            frame_from: sel.frameFrom,
            frame_to: sel.frameTo,
            set_flags: ticked ? [flag] : [],
            clear_flags: ticked ? [] : [flag],
        };
        // No overlap pre-confirmation and no 409 retry, unlike a value edit:
        // ticking a flag never contests another flag, and re-specifying the
        // same one supersedes it rather than conflicting.
        try {
            let res = await fetch("/api/edits/feature-bits", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (res.status === 409) {
                const detail = (await res.json()).detail || {};
                if (detail.code === "large_edit_confirmation_required") {
                    const ok = window.confirm(
                        `This edit touches ${detail.frames} frames.\n\n` +
                        `Continue? Saves over ~10,000 frames can take a while.`
                    );
                    if (!ok) return;
                    res = await fetch("/api/edits/feature-bits", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ ...body, confirm_large: true }),
                    });
                }
            }
            if (!res.ok) {
                const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
                window.setStatus && window.setStatus(`Flag edit failed: ${detail}`);
                return;
            }
            const payload = await res.json();
            window.setStatus && window.setStatus(
                payload.pending ? `${flag}: staged` : `${flag}: nothing to change`
            );
            if (typeof window.refreshPendingEdits === "function") {
                await window.refreshPendingEdits();
            }
            // No render here: refreshPendingEdits -> onPendingEditsChanged
            // already redraws from the merged view, and when the pending count
            // reaches zero it does so only after refetching the series. A
            // second synchronous render would race that refetch and draw a card
            // with no values.
        } catch (err) {
            _err("stageFlagEdit failed", err);
            window.setStatus && window.setStatus(`Flag edit failed: ${err.message}`);
        }
    }

    async function stageFeatureEdit(featureName, value) {
        const sel = _resolvedRangeFor(featureName);
        if (!sel) {
            window.setStatus && window.setStatus("Drag-select a frame range first");
            return;
        }
        const datasetId = sel.datasetId;
        const body = {
            dataset_id: datasetId,
            episode_index: sel.episodeIndex,
            feature: featureName,
            frame_from: sel.frameFrom,
            frame_to: sel.frameTo,
            value: value,
        };
        const stageKey = _stageKey(
            datasetId, featureName, sel.episodeIndex,
            sel.frameFrom, sel.frameTo
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
            // 409 paths: overlapping edit, or large-edit (>10k frames). Both
            // surface a confirmation dialog and re-POST with the right ack
            // flag. Same-key overlap collisions were pre-confirmed above so
            // they shouldn't reach this branch in normal typing flow.
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
                } else if (detail && detail.code === "large_edit_confirmation_required") {
                    const ok = window.confirm(
                        `This edit touches ${detail.frames} frames.\n\n` +
                        `Continue? Saves over ~10,000 frames can take a while.`
                    );
                    if (!ok) return;
                    res = await fetch("/api/edits/feature-set", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ ...body, confirm_large: true }),
                    });
                }
            }
            if (!res.ok) {
                const err = await res.text();
                window.setStatus && window.setStatus(`Edit rejected: ${err}`);
                return;
            }
            // The backend coerces the range for per-episode features (the
            // staged edit covers [0, episode_length)). We use the coerced
            // range for the same-stage-key collapse logic, but DON'T mutate
            // the user's selection: the user dragged a sub-range for a
            // reason and snapping it to the whole episode every time they
            // tweak success/control_mode is jarring.
            let coercedFrom = sel.frameFrom, coercedTo = sel.frameTo;
            try {
                const responseBody = await res.json();
                if (responseBody && Array.isArray(responseBody.coerced_range)) {
                    coercedFrom = responseBody.coerced_range[0];
                    coercedTo = responseBody.coerced_range[1];
                }
            } catch (_) { /* response body wasn't JSON — ignore */ }
            // Record the identity of this successful stage so subsequent
            // typing in the SAME widget+selection auto-collapses into one
            // edit instead of stacking 409 dialogs. Keyed on the coerced
            // range so two stages on the same per-episode feature collapse
            // even when the user's selection drifted between them.
            _lastStagedKey = _stageKey(
                datasetId, featureName, sel.episodeIndex,
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
            const hiddenDefault = isHiddenByDefault(name, ft);
            // Per-episode features are uniform across the episode — a
            // full-width solid-color row is wasted real estate. Surface
            // them only in the Inspector under their own section. The
            // user can still pin them to the timeline if they want.
            const hiddenPerEpisode = ft.is_per_episode && !state.pinned;
            const hidden = (hiddenDefault || hiddenPerEpisode) && !state.pinned;
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
        // "+ Add feature" affordance for non-default custom features. The
        // banner handles reward / success; this dialog covers the rest.
        rows.push(
            '<div class="feature-row feature-row-addbtn">' +
            '<button class="feature-add-btn" type="button" id="feature-row-add-feature-btn">+ Add feature</button>' +
            '</div>'
        );
        container.innerHTML = rows.join("");

        const addBtn = document.getElementById("feature-row-add-feature-btn");
        if (addBtn) {
            addBtn.onclick = () => {
                if (window.AddFeatureDialog && window.AddFeatureDialog.open) {
                    window.AddFeatureDialog.open();
                } else {
                    _warn("AddFeatureDialog not available — script load issue?");
                }
            };
        }

        // Wire mouse handlers on each row's track.
        container.querySelectorAll(".row-track").forEach(track => {
            wireFeatureRowTrack(track);
            wireMaskSegments(track);
        });
        // Wire per-row delete buttons.
        container.querySelectorAll(".row-delete-btn").forEach(btn => {
            btn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                deleteFeature(btn.getAttribute("data-feature"));
            });
        });
    }

    /**
     * Click and hover on a mask row's segments.
     *
     * Bound on the track rather than on each rect so it survives a re-render,
     * and it must not fall through to the track's own drag handler — the same
     * reason the timeline's seek handler returns early for a trim handle.
     */
    // The claimed gesture lives OUTSIDE any row's closure, and the release is
    // heard on the document, because staging re-renders the row and replaces
    // the track node. A mouseup listener on the track would be attached to the
    // node that no longer exists, and per-node state would go with it -- which
    // is what made the toggle work only when the re-render happened to land
    // outside the press.
    let _maskClaim = null;
    let _maskReleaseBound = false;

    function bindMaskRelease() {
        if (_maskReleaseBound) return;
        _maskReleaseBound = true;
        document.addEventListener("mouseup", (ev) => {
            const c = _maskClaim;
            _maskClaim = null;
            if (!c) return;
            // A press that travelled is a drag, not a click; without this a
            // wobble while pressing would toggle.
            if (Math.abs(ev.clientX - c.x) > 4 || Math.abs(ev.clientY - c.y) > 4) return;
            stageMaskSegmentEdit(c.feature, c.seg, "toggle");
        }, true);
    }

    const laneIndexOf = (ft, label) => (ft.mask_labels || []).indexOf(label);

    function wireMaskSegments(track) {
        bindMaskRelease();
        const featureName = track.getAttribute("data-feature");
        const ft = window.datasets?.[window.currentDataset]?.features_schema?.[featureName];
        if (!Array.isArray(ft?.mask_labels) || !ft.mask_labels.length) return;
        const length = Number(track.getAttribute("data-length")) || 0;

        const hit = (ev) => {
            const rect = track.getBoundingClientRect();
            if (!rect.width || !rect.height || !length) return null;
            const frame = Math.min(length - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * length)));
            // Lanes occupy 10%..90% of the row; outside that is padding.
            const yPct = ((ev.clientY - rect.top) / rect.height) * 100;
            const n = ft.mask_labels.length;
            const laneH = 80 / n;
            const lane = Math.floor((yPct - 10) / laneH);
            if (yPct < 10 || lane < 0 || lane >= n) return null;
            const seg = maskSegmentAt(featureName, ft, lane, frame);
            return seg ? { seg, frame } : null;
        };

        // The x lives where the cursor is, on the segment under it -- a row
        // with three segments offers three deletions, not one for the label.
        // The x is PINNED to the segment it deletes -- centred on the part of it
        // inside the selection -- rather than following the cursor. A control
        // that moves with the pointer is a target you cannot aim at, and it
        // says nothing about which of several segments it would act on.
        let killer = null;
        let killerFor = null;  // the segment the current button belongs to
        const clearKiller = () => {
            if (killer) killer.remove();
            killer = null;
            killerFor = null;
        };
        // How close to a segment's trailing edge the pointer must come before
        // the delete affordance appears at all.
        const KILL_ZONE_PX = 28;

        track.addEventListener("mousemove", (ev) => {
            const h = hit(ev);
            if (!h) { clearKiller(); return; }
            // Deleting is a DELIBERATE reach for the segment's trailing edge,
            // not something the whole bar offers. A button covering the middle
            // of a segment sits exactly where a click means "toggle", so an
            // ordinary click lands on delete -- and once shown it follows you
            // across the track eating clicks meant to re-select.
            const rect = track.getBoundingClientRect();
            const edgeX = rect.x + rect.width * (h.seg.to / length);
            if (edgeX - ev.clientX > KILL_ZONE_PX || ev.clientX > edgeX) { clearKiller(); return; }
            const key = `${h.seg.label}:${h.seg.from}:${h.seg.to}`;
            if (killerFor === key) return;  // already placed on this segment
            clearKiller();
            const seg = h.seg;
            killer = document.createElement("button");
            killer.className = "mask-seg-kill";
            killer.textContent = "×";
            killer.title = `Delete "${seg.label}" over frames ${seg.from}–${seg.to - 1}`;
            killer.setAttribute("data-label", seg.label);
            killer.setAttribute("data-from", String(seg.from));
            killer.setAttribute("data-to", String(seg.to));
            killer.addEventListener("mousedown", (e) => { e.stopPropagation(); e.preventDefault(); });
            killer.addEventListener("click", (e) => {
                e.stopPropagation();
                stageMaskSegmentEdit(featureName, seg, "delete");
                clearKiller();
            });
            const n = ft.mask_labels.length;
            const laneH = 80 / n;
            // The RIGHT EDGE of the segment, not its centre: the centre is
            // exactly where you click to toggle, so a button there occludes the
            // gesture it sits on -- the click lands on delete instead.
            killer.style.left = `${(seg.to / length) * 100}%`;
            killer.style.top = `${10 + laneIndexOf(ft, seg.label) * laneH + laneH * 0.4}%`;
            // Pulled fully inside the segment so it cannot read as belonging to
            // whatever sits to its right.
            killer.style.transform = "translate(-100%, -50%)";
            track.appendChild(killer);
            killerFor = key;
        });
        track.addEventListener("mouseleave", clearKiller);

        // The row's own mousedown seeks the playhead AND replaces the selection
        // with a single frame. It is registered first and fires first, so
        // stopping propagation on `click` is far too late -- the selection the
        // toggle needs is already gone, and the edit silently covered one
        // frame. Claim the gesture in the CAPTURE phase instead, which runs
        // before any bubble-phase listener on the same element.
        // The gesture is decided at MOUSEDOWN and performed at MOUSEUP, and
        // never via `click`.
        //
        // Two things forced this. The row's own mousedown seeks and replaces
        // the selection, so a toggle that reads the selection later reads the
        // one that mousedown just made -- a single frame. And staging
        // re-renders the row, which replaces the node between mousedown and
        // mouseup, so the browser has no common target to fire `click` on and
        // the toggle simply did not happen. That is the same defect from both
        // ends: sometimes it edited one frame, sometimes it did nothing.
        //
        // Claiming here means the toggle runs only when a usable selection
        // ALREADY existed. The click that creates a selection can never also
        // act on it.
        track.addEventListener("mousedown", (ev) => {
            if (ev.button !== 0) return;
            const h = hit(ev);
            if (!h) return;  // no selection here yet: let the row select
            _maskClaim = { feature: featureName, seg: h.seg, x: ev.clientX, y: ev.clientY };
            ev.stopPropagation();
            ev.preventDefault();
        }, true);
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
        // The muted companion travels beside the enabled series; a lane
        // needs both to tell disabled from absent.
        const rawMuted = cached.series[`${name}__disabled`] || [];
        // Staged segment edits show immediately, or a click looks like it did
        // nothing until Save.
        const [maskEnabled, mutedSeries] = Array.isArray(ft.mask_labels) && ft.mask_labels.length
            ? applyPendingMaskEdits(name, ft.mask_labels, series, rawMuted, length)
            : [series, rawMuted];
        const trackContent = renderTrackSvg(name, ft, maskEnabled, length, mutedSeries);
        // Lanes are unreadable without saying which is which, and the names sit
        // in HTML rather than the stretched SVG so the glyphs are not scaled.
        const isFlagsRow = Array.isArray(ft.flags) && ft.flags.length > 0;
        const isMasksRow = Array.isArray(ft.mask_labels) && ft.mask_labels.length > 0;
        const flagLegend = !isFlagsRow ? "" : ft.flags.map((flag, bit) => {
            const laneH = 80 / ft.flags.length;
            const y = 10 + bit * laneH;
            return `<span class="row-flag-name" style="top: ${y.toFixed(2)}%; ` +
                   `height: ${(laneH * 0.8).toFixed(2)}%">` +
                   `<i style="background: ${flagColor(bit)}"></i>${escapeHtml(flag)}</span>`;
        }).join("");
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
        // Mask rows are deliberately NOT `editable` -- their values cannot be
        // typed -- so the branch above skips them, and its params are the wrong
        // shape anyway: a mask edit names a label and a span, not a value. Draw
        // them per lane, or a staged segment edit is invisible in the one view
        // that exists to show what is staged.
        if (showPendingEdits && isMasksRow) {
            const camera = maskCameraOf(name);
            const laneH = 80 / ft.mask_labels.length;
            for (const e of (window.pendingEdits || [])) {
                if (e.edit_type !== "mask_range") continue;
                if (e.params?.camera !== camera || e.episode_index !== window.currentEpisode) continue;
                const bit = ft.mask_labels.indexOf(e.params.label);
                if (bit < 0) continue;
                const left = (e.params.from_frame / length) * 100;
                const width = ((e.params.to_frame - e.params.from_frame) / length) * 100;
                overlays.push(
                    `<div class="row-pending-overlay mask-pending" ` +
                    `title="${escapeHtml(e.params.action)} ${escapeHtml(e.params.label)}: ` +
                    `frames ${e.params.from_frame}–${e.params.to_frame - 1}" ` +
                    `style="left:${left}%; width:${width}%; ` +
                    `top:${10 + bit * laneH}%; height:${laneH * 0.8}%;"></div>`
                );
            }
        }

        // Read-only state is conveyed by the row class (CSS dims the
        // flag background and adds a left border) and by the Inspector
        // card on click. The previous design rendered an explicit
        // "read-only" line in the row flag, which got clipped by the
        // 36px row height plus the row-label's overflow:hidden.
        const rowClass = editable ? "feature-row" : "feature-row readonly";

        // Per-row delete (✕). isDeletable handles all the exclusions —
        // recorded data (action/observation.*), internal bookkeeping
        // (timestamp/etc.), banner-managed defaults (reward/success),
        // and binary blobs (image/video).
        const deleteBtn = isDeletable(name, ft)
            ? `<button class="row-delete-btn" data-feature="${escapeHtml(name)}" type="button" title="Delete this feature column">✕</button>`
            : "";

        return `
            <div class="${rowClass}${isFlagsRow ? " flags-row" : ""}${isMasksRow ? " masks-row" : ""}" data-feature="${escapeHtml(name)}"
                 style="${isFlagsRow ? `--flag-count: ${ft.flags.length};` : ""}${isMasksRow ? `--mask-count: ${ft.mask_labels.length};` : ""}">
                <div class="row-label">
                    <div class="row-name">${escapeHtml(name)}</div>
                    <div class="row-dtype">${escapeHtml(dtype)}[${shape}]</div>
                    ${deleteBtn}
                </div>
                <div class="row-track" data-feature="${escapeHtml(name)}" data-length="${length}">
                    ${trackContent}
                    ${flagLegend}
                    ${overlays.join("")}
                </div>
            </div>
        `;
    }

    async function deleteFeature(featureName) {
        const datasetId = window.currentDataset;
        if (!datasetId) return;
        const ok = window.confirm(
            `Permanently delete feature column "${featureName}"?\n\n` +
            `This rewrites the dataset's parquet shards in place. Cannot be undone.`
        );
        if (!ok) return;
        try {
            const r = await fetch(
                `/api/datasets/${encodeURIComponent(datasetId)}/features/${encodeURIComponent(featureName)}`,
                { method: "DELETE" }
            );
            if (!r.ok) {
                const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
                window.setStatus && window.setStatus(`Delete failed: ${detail}`);
                return;
            }
            const payload = await r.json();
            if (payload && payload.info) {
                window.datasets[datasetId] = payload.info;
            }
            // Drop cached series — schema changed.
            for (const key of Array.from(seriesCache.keys())) {
                if (key.startsWith(`${datasetId}:`)) seriesCache.delete(key);
            }
            // Re-show the banner if reward/success were missing again.
            maybeShowDefaultsBanner(datasetId);
            // Re-render rows + inspector.
            if (window.currentEpisode != null) {
                loadFeatureSeries(datasetId, window.currentEpisode).then(() => {
                    renderFeatureRows();
                    renderInspector();
                });
            } else {
                renderFeatureRows();
                renderInspector();
            }
            window.setStatus && window.setStatus(`Deleted feature: ${featureName}`);
        } catch (err) {
            _err("deleteFeature failed", err);
            window.setStatus && window.setStatus(`Delete failed: ${err.message}`);
        }
    }

    // Distinct hues per flag. Fixed rather than hashed so a flag keeps its
    // colour across renders and between the row and the legend.
    const FLAG_COLORS = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6",
                         "#16a085", "#e15f9d", "#7f8c8d"];
    function flagColor(bit) { return FLAG_COLORS[bit % FLAG_COLORS.length]; }

    // Mask lanes take their colour from the overlay's palette rather than
    // FLAG_COLORS, so a lane and the boundary drawn on the frame for the same
    // object are the same colour. Both are keyed by position in mask_labels;
    // two palettes agreed by eye for the first three entries and diverged at
    // the fourth (mustard against purple), which is exactly where an operator
    // with four objects would start matching the wrong lane to the wrong
    // outline. Falls back while masks.js has not loaded.
    /**
     * Contiguous runs of one state for label `bit`, over `[0, len)`.
     *
     * The unit every mask edit acts on. A segment is a maximal run where the
     * label is in ONE state, so a click never has to resolve a mixed range and
     * the direction of a toggle is decided by what was clicked. Absent runs
     * are returned too — the caller skips them for drawing, and hit-testing
     * needs to know a click landed on nothing rather than on the lane below.
     */
    function maskSegments(enabled, disabled, bit, len) {
        const stateAt = (i) => {
            if ((enabled[i] >> bit) & 1) return "detected";
            if (((disabled[i] || 0) >> bit) & 1) return "disabled";
            return "absent";
        };
        const out = [];
        let i = 0;
        while (i < len) {
            const s = stateAt(i);
            let j = i;
            while (j < len && stateAt(j) === s) j++;
            out.push({ from: i, to: j, state: s });
            i = j;
        }
        return out;
    }

    function maskLaneColor(bit) {
        const p = window.MaskOverlay && window.MaskOverlay.PALETTE;
        if (!p || !p.length) return flagColor(bit);
        const [r, g, b] = p[bit % p.length];
        return `rgb(${r}, ${g}, ${b})`;
    }

    function renderTrackSvg(name, ft, series, length, mutedSeries) {
        if (!series || !series.length) return "";
        const dtype = ft.dtype || "";
        const shape = ft.shape || [];

        // A bitset is not a magnitude: plotting the stored integer as a line
        // puts 3 above 2 and invites reading one flag as more than another.
        // Draw a lane per flag instead, filled where that flag is set --
        // which is also what makes the row usable for picking a range to edit.
        if (Array.isArray(ft.flags) && ft.flags.length && typeof series[0] === "number") {
            const count = ft.flags.length;
            const laneH = 80 / count;  // share the row, leaving 10% margins
            const segs = [];
            for (let bit = 0; bit < count; bit++) {
                const y = 10 + bit * laneH;
                const h = laneH * 0.8;
                // A faint rail for every declared flag, drawn whether or not it
                // ever fires. Without it two filled bands are indistinguishable
                // from two-of-five, and a flag no frame carries would vanish
                // from the row entirely rather than reading as "none here".
                segs.push(
                    `<rect x="0%" y="${y.toFixed(2)}%" width="100%" height="${h.toFixed(2)}%" ` +
                    `fill="${flagColor(bit)}" opacity="0.13"/>`
                );
                for (let i = 0; i < series.length; i++) {
                    const v = typeof series[i] === "number" ? series[i] : 0;
                    if (!bitIsSet(v, bit)) continue;
                    const x = (i / length) * 100;
                    const w = (1 / length) * 100 + 0.05;  // overdraw to avoid seams
                    segs.push(
                        `<rect x="${x}%" y="${y.toFixed(2)}%" width="${w}%" ` +
                        `height="${h.toFixed(2)}%" fill="${flagColor(bit)}"/>`
                    );
                }
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${segs.join("")}</svg>`;
        }

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

        // Stored masks: one thin lane per object, drawn in three states. The
        // value is the server's per-frame ENABLED bitset (bit i =
        // mask_labels[i]); the companion series carries the muted ones. Absent
        // is neither bit — see `_mask_disabled_bits` for why two series rather
        // than two bits per label.
        if (Array.isArray(ft.mask_labels) && ft.mask_labels.length && typeof series[0] === "number") {
            const names = ft.mask_labels;
            const n = names.length;
            const laneH = 80 / n;
            const rects = [];
            const laneNames = [];
            const muted = mutedSeries || [];
            for (let b = 0; b < n; b++) {
                const y = 10 + b * laneH;
                const color = maskLaneColor(b);
                laneNames.push(
                    `<div class="row-flag-name row-mask-name" style="top:${y}%; height:${laneH * 0.8}%;">` +
                    `<i style="background:${color}"></i>${escapeHtml(names[b])}</div>`
                );
                // The faint rail is the lane even when the object is never
                // found — an object SAM never saw has to read as an empty
                // lane, not as a missing one.
                rects.push(
                    `<rect x="0%" y="${y}%" width="100%" height="${laneH * 0.8}%" ` +
                    `fill="${color}" opacity="0.10"/>`
                );
                for (const seg of maskSegments(series, muted, b, series.length)) {
                    if (seg.state === "absent") continue;
                    const x = (seg.from / length) * 100;
                    const w = ((seg.to - seg.from) / length) * 100 + 0.05;
                    // FILLED means it reaches training; HOLLOW means stored but
                    // withheld. An outline, not a dimmer fill or a hatch: a lane
                    // is a few pixels tall, and at that size a texture or an
                    // opacity step is not a difference anyone can see -- which
                    // matters because the bar is also the control.
                    const detected = seg.state === "detected";
                    rects.push(
                        // `data-label` is the label NAME, matching the delete
                        // button this segment offers and every other data-label
                        // in this file. It carried the lane INDEX until the two
                        // were found to disagree, so anything reading one and
                        // writing the other silently addressed the wrong lane.
                        `<rect class="mask-seg" data-feature="${escapeHtml(name)}" ` +
                        `data-label="${escapeHtml(names[b])}" data-lane="${b}" ` +
                        `data-from="${seg.from}" data-to="${seg.to}" data-state="${seg.state}" ` +
                        `x="${x}%" y="${y}%" width="${w}%" height="${laneH * 0.8}%" ` +
                        `fill="${detected ? color : "none"}" opacity="${detected ? 0.85 : 1}" ` +
                        `stroke="${detected ? "none" : color}" stroke-width="${detected ? 0 : 1.5}" ` +
                        `vector-effect="non-scaling-stroke"/>`
                    );
                }
            }
            return (
                `<svg class="mask-lanes" preserveAspectRatio="none" viewBox="0 0 100 100">` +
                `${rects.join("")}</svg>` +
                laneNames.join("")
            );
        }

        if (dtype === "string") {
            // Colored stripe — each unique string gets a color; render run-length segments.
            // The colored rectangles go in a stretched SVG (preserveAspectRatio="none")
            // so they fill the row exactly. Text flags go in HTML overlays — putting
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

        // Categorical (int + names): render as a colored band with the flag
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

        // Numeric: scalar → line; vector → mini multi-line (up to MULTI_LINE_CAP);
        // very-large vectors fall back to L2-norm-per-frame.
        //
        // The cap was 8 originally — dropping a 14-DOF ALOHA action to a single
        // L2-norm line, surprising users (the row flag correctly says
        // float32[14] but the visualization shows one curve, looking like a
        // bug). 32 covers typical robot DOF (so-100 leader+follower=12, ALOHA
        // bimanual=14, humanoids ≤ 30) and keeps the SVG cheap.
        const MULTI_LINE_CAP = 32;
        const scalarSeries = (typeof series[0] === "number") ? series : null;
        if (scalarSeries) {
            return numericLineSvg(scalarSeries, length);
        }
        if (Array.isArray(series[0]) && series[0].length <= MULTI_LINE_CAP) {
            const dims = series[0].length;
            // 16-color palette; recycles on shape > 16. Palette tuned to be
            // distinguishable on a dark background and not collide with
            // common UI accent colors.
            const colors = [
                "#5b8def", "#d97757", "#4caf50", "#b58900",
                "#9b59b6", "#16a085", "#e74c3c", "#7f8c8d",
                "#3498db", "#e67e22", "#27ae60", "#f1c40f",
                "#8e44ad", "#1abc9c", "#c0392b", "#95a5a6",
            ];
            const lines = [];
            for (let d = 0; d < dims; d++) {
                const dim = series.map(row => row[d]);
                lines.push(numericLinePath(dim, length, colors[d % colors.length]));
            }
            return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${lines.join("")}</svg>`;
        }
        if (Array.isArray(series[0])) {
            // Very large vector (> MULTI_LINE_CAP dims). A single L2-norm
            // line is dominated by whichever dims swing widest — on a 48-dim
            // bimanual state (16 pos + 16 vel + 16 torque) it looked like
            // "only the gripper is plotted". Draw one L2-norm line per
            // name-suffix group (.pos / .vel / .torque / ...) when the names
            // group cleanly, so each channel family stays visible.
            const names = ft.names || [];
            const groups = new Map();
            for (let d = 0; d < series[0].length; d++) {
                const n = names[d] || "";
                const dot = n.lastIndexOf(".");
                const suffix = dot >= 0 ? n.slice(dot + 1) : "";
                if (!groups.has(suffix)) groups.set(suffix, []);
                groups.get(suffix).push(d);
            }
            if (groups.size > 1 && groups.size <= 8) {
                const palette = ["#5b8def", "#d97757", "#4caf50", "#b58900", "#9b59b6", "#16a085", "#e74c3c", "#7f8c8d"];
                const lines = [];
                let gi = 0;
                for (const idxs of groups.values()) {
                    const norms = series.map(row => {
                        let s = 0;
                        for (const d of idxs) s += (typeof row[d] === "number" ? row[d] * row[d] : 0);
                        return Math.sqrt(s);
                    });
                    lines.push(numericLinePath(norms, length, palette[gi % palette.length]));
                    gi++;
                }
                return `<svg preserveAspectRatio="none" viewBox="0 0 100 100">${lines.join("")}</svg>`;
            }
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

    function setupRunOverlaysResize() {
        const handle = document.getElementById("run-overlays-resize");
        const panel = document.getElementById("overlays-panel-run");
        if (!handle || !panel) return;
        let dragging = false;
        let startX = 0, startW = 0;

        const stored = parseInt(localStorage.getItem("run.overlaysPanelWidth") || "", 10);
        if (stored && stored >= 220 && stored <= 600) {
            panel.style.setProperty("--run-overlays-width", `${stored}px`);
        }

        handle.addEventListener("mousedown", (e) => {
            dragging = true;
            startX = e.clientX;
            startW = panel.getBoundingClientRect().width;
            handle.classList.add("dragging");
            e.preventDefault();
        });
        document.addEventListener("mousemove", (e) => {
            if (!dragging) return;
            const dx = e.clientX - startX;
            const next = Math.max(220, Math.min(600, startW - dx));
            panel.style.setProperty("--run-overlays-width", `${next}px`);
        });
        document.addEventListener("mouseup", () => {
            if (!dragging) return;
            dragging = false;
            handle.classList.remove("dragging");
            const px = parseInt(panel.style.getPropertyValue("--run-overlays-width"), 10);
            if (px) localStorage.setItem("run.overlaysPanelWidth", String(px));
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
        setupRunOverlaysResize();
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
