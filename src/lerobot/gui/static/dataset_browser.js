// Dataset browser: search, favorites, sort.
//
// Split out of app.js, which is several thousand lines and was the reason this
// file exists -- a reviewer asked that new surfaces stop landing in the
// monolith. Nothing here touches the DOM tree directly; app.js's renderers call
// in for the row decisions and read the state by bare name.
//
// This file MUST load before app.js. Top-level `let`/`const` in a classic
// script live in the shared global lexical environment, so app.js resolves
// `datasetBrowserState` and friends by bare name -- but only once this script
// has run. They are deliberately not on `window`, matching how app.js already
// treats its own script-scope state.
//
// The reverse direction (this file calling `renderSources`, `scanSource`,
// `sources`, `sourceDatasets`) resolves at call time, after both scripts load.

// In-flight source scans, so rapid typing issues at most one request per source.
const sourceScansInFlight = new Map();

const DATASET_BROWSER_STORAGE_KEY = 'lerobot.gui.datasetBrowser.v2';
const DATASET_SORTS = new Set(['last-opened', 'name-asc', 'name-desc', 'episodes-desc', 'episodes-asc']);
let datasetBrowserState = loadDatasetBrowserState();
let datasetFavoritesOnly = false;

function loadDatasetBrowserState() {
    const fallback = { favorites: [], lastOpenedAt: {}, sort: 'last-opened' };
    try {
        const stored = JSON.parse(localStorage.getItem(DATASET_BROWSER_STORAGE_KEY) || 'null');
        if (!stored || typeof stored !== 'object') return fallback;
        const favorites = Array.isArray(stored.favorites)
            ? stored.favorites.filter(root => typeof root === 'string')
            : [];
        const lastOpenedAt = {};
        if (stored.lastOpenedAt && typeof stored.lastOpenedAt === 'object') {
            for (const [root, timestamp] of Object.entries(stored.lastOpenedAt)) {
                const parsedTimestamp = Number(timestamp);
                if (typeof root === 'string' && Number.isFinite(parsedTimestamp)) {
                    lastOpenedAt[root] = parsedTimestamp;
                }
            }
        }
        return {
            favorites,
            lastOpenedAt,
            sort: DATASET_SORTS.has(stored.sort) ? stored.sort : fallback.sort,
        };
    } catch (_) {
        return fallback;
    }
}

function saveDatasetBrowserState() {
    try {
        localStorage.setItem(DATASET_BROWSER_STORAGE_KEY, JSON.stringify(datasetBrowserState));
    } catch (_) {
        // Browsing still works when storage is unavailable (private mode, policy, quota).
    }
}

function datasetIsFavorite(root) {
    return datasetBrowserState.favorites.includes(root);
}

function toggleDatasetFavorite(root, event) {
    if (event) event.stopPropagation();
    const favorites = new Set(datasetBrowserState.favorites);
    if (favorites.has(root)) favorites.delete(root);
    else favorites.add(root);
    datasetBrowserState.favorites = [...favorites].sort();
    saveDatasetBrowserState();
    renderSources();
}

function toggleDatasetFavoritesOnly() {
    datasetFavoritesOnly = !datasetFavoritesOnly;
    refreshDatasetBrowser();
}

function setDatasetSort(sort) {
    if (!DATASET_SORTS.has(sort)) return;
    datasetBrowserState.sort = sort;
    saveDatasetBrowserState();
    renderSources();
}

function clearDatasetSearchOnEscape(event) {
    if (event.key !== 'Escape') return;
    const input = event.currentTarget;
    if (!input.value) return;
    input.value = '';
    refreshDatasetBrowser();
}

function rememberDatasetOpened(root) {
    if (!root) return;
    datasetBrowserState.lastOpenedAt[root] = Date.now();
    saveDatasetBrowserState();
}

function datasetLastOpenedTitle(root) {
    const timestamp = datasetBrowserState.lastOpenedAt[root];
    return Number.isFinite(timestamp) ? `\nLast opened: ${new Date(timestamp).toLocaleString()}` : '';
}

function datasetSearchTokens() {
    const input = document.getElementById('dataset-search');
    return String(input?.value || '').trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
}

function datasetBrowserFiltersActive() {
    return datasetFavoritesOnly || datasetSearchTokens().length > 0;
}

function refreshDatasetBrowser() {
    renderSources();
    if (datasetBrowserFiltersActive()) {
        void scanUnloadedSourcesForDatasetBrowser();
    }
}

async function scanUnloadedSourcesForDatasetBrowser() {
    const unloaded = sources.filter(
        source => !Object.prototype.hasOwnProperty.call(sourceDatasets, source.path),
    );
    await Promise.all(unloaded.map(source => scanSource(source.path)));
}

function datasetRowMatches(row, tokens) {
    if (datasetFavoritesOnly && !datasetIsFavorite(row.root)) return false;
    if (tokens.length === 0) return true;
    // Name only, not the root path. Every dataset under a source shares that
    // path's prefix, so including it made 'lerobot' or 'cache' match all 153
    // of them -- a query that should narrow instead returned everything. The
    // name already carries the namespace, which is the part worth searching.
    const haystack = String(row.name || '').toLocaleLowerCase();
    return tokens.every(token => haystack.includes(token));
}

function compareDatasetRows(a, b) {
    const byName = () => String(a.name || '').localeCompare(String(b.name || ''));
    switch (datasetBrowserState.sort) {
        case 'name-desc':
            return -byName();
        case 'episodes-desc':
            return (Number(b.total_episodes) || 0) - (Number(a.total_episodes) || 0) || byName();
        case 'episodes-asc':
            return (Number(a.total_episodes) || 0) - (Number(b.total_episodes) || 0) || byName();
        case 'last-opened':
            return (datasetBrowserState.lastOpenedAt[b.root] || 0)
                - (datasetBrowserState.lastOpenedAt[a.root] || 0) || byName();
        case 'name-asc':
        default:
            return byName();
    }
}
