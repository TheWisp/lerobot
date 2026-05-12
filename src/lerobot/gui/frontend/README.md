# lerobot GUI — frontend (Svelte 5 + Vite)

This directory holds the **Svelte 5 + Vite** rewrite of the legacy
`gui/static/*.js` files. Migration is happening **island by island**:
each Svelte component mounts into a specific DOM element of the existing
`index.html`, so the legacy code and the new reactive code coexist on the
same page during the migration.

## What's in here today

```
frontend/
  src/
    islands/              # entry points — one per mounted island
      bug-report.ts       # mounts <BugReportModal/> at #bug-report-mount
      run-sidebar.ts      # mounts <RunSidebar/> at #run-form (when ?reactive=1)
    components/
      BugReportModal.svelte
      RunSidebar.svelte
      TeleopForm.svelte
      ReplayForm.svelte
    stores/
      profiles.svelte.ts  # robotProfiles + teleopProfiles, ensureLoaded()
      datasets.svelte.ts  # opened datasets, ensureLoaded()
      run.svelte.ts       # Run-tab form state + derived validation
    lib/
      api.ts              # getJson / postJson wrappers
      toast.ts            # bridge to legacy window.showToast
      types.ts            # narrow API types used by the islands
  vite.config.ts          # outputs to ../static/dist/
  package.json
  tsconfig.json
  svelte.config.js
```

The build emits to `../static/dist/`. FastAPI's `StaticFiles` mount at
`/static` already serves the `dist/` subdirectory; no server changes were
needed.

## Why Svelte 5 + Vite

The full research that led to this choice is in
`src/lerobot/gui/TODO.md` (the [Critical] "Reactive UI state management"
entry plus the [High] "Extract frontend to separate files" entry). The
short version:

- **Svelte 5 runes** (`$state` / `$derived` / `$effect`) directly express
  "UI state derived from data" — which is the gap the legacy frontend has.
  `_toggleHvlaRecordFields()` in `run.js` is called from 4 sites; in Svelte
  the equivalent reactivity is one `$derived` block, called from zero.
- **Smallest practical bundle** (~5 KB gz for the bug-report island, ~14 KB
  gz for the run-sidebar island, with a shared ~12 KB gz runtime). Total
  payload is well under the legacy `app.js` itself.
- **Compiles to plain JS, no runtime VDOM** — every island ships
  independently, can coexist with the legacy code, and can be removed
  without rewriting the rest.
- **Vite** for the build because it's the de-facto Svelte tooling and
  gives us TypeScript + HMR + tree-shaking with zero config.

## Day-to-day

```bash
# From this directory:
npm install            # First time, populates node_modules/
npm run build          # Production build → ../static/dist/
npm run check          # svelte-check (typecheck + a11y warnings)
npm run dev            # Vite dev server on :5173 with HMR (proxies /api → :8000)
```

The legacy GUI server (`python -m lerobot.gui`) serves the production
build directly from `static/dist/`. You don't need `npm run dev` running
unless you want hot reload while editing Svelte files.

**To opt into the reactive Run sidebar** (Phase 1 of the Run-tab port),
open the GUI with `?reactive=1`:

```
http://localhost:8000/?reactive=1
```

The legacy `renderRunForm()` short-circuits when the flag is set, and the
Svelte sidebar mounts in place. Drop `?reactive=1` to fall back to the
legacy form — useful while Policy is being ported.

## Migration playbook

Each island follows the same five steps. They're listed here so the next
contributor doesn't have to re-derive them.

1. **Pick a target.** Smaller is better for the first port (we did
   `BugReportModal` first — 1 file, 1 component, full replacement). Look
   for self-contained components with limited cross-tab state.
2. **Audit the legacy code** for cross-cutting `document.getElementById`
   reads + mutations. Group them by "what data is this reading?" — that
   becomes a `$state` / `$derived` in the Svelte version.
3. **Build a `.svelte.ts` store** if data is shared across islands or
   tabs (see `stores/profiles.svelte.ts`). Stores must expose
   `ensureLoaded()` and NOT consult any UI flags — that's the
   "decouple data loaders from UI state" win.
4. **Write the component.** Use `bind:value` for two-way input binding,
   `$derived` for any "show/disable/highlight based on data" rule,
   `$effect` only for side effects (DOM, network) that can't be
   expressed as derived state.
5. **Mount as an island.** Add an entry under `src/islands/`, register
   it in `vite.config.ts`'s `rollupOptions.input`, add the `<script
type="module">` + `<link rel="stylesheet">` in `index.html`, and put
   the mount-point `<div>` where the island should appear.
6. **Bridge to legacy if needed.** If older code calls into the
   island (e.g. legacy `onclick="openBugReportDialog()"`), expose
   the function on `window` from the island entry. If the island
   needs to call into legacy, defer to a global helper like
   `window.showToast` via `lib/toast.ts`.

## Phase 1 (this PR)

- ✅ Build pipeline (Vite + Svelte 5 + TS)
- ✅ Shared stores (profiles, datasets, run form)
- ✅ `BugReportModal` — full replacement of `static/bug_report.js` and
  the modal markup in `index.html`
- ✅ `RunSidebar` — workflow selector + `TeleopForm` + `ReplayForm`,
  gated by `?reactive=1` so Policy users keep the legacy form

## Phase 2 candidates (next, in priority order)

1. **Policy workflow form** — the worst remaining `_toggleHvla*` /
   `_onRltSelectChange` chain. Replaces the placeholder in
   `RunSidebar.svelte` when ported.
2. **Wire `launchRun()` to read from the reactive store** so Launch
   actually uses the new form's values. Today the legacy
   `launchRun()` still reads from `document.getElementById(...)`.
3. **Inspector panel** (`feature_editing.js` ~1700 lines). Lots of
   schema-driven rendering — a natural fit for `{#each}` + a
   per-feature widget component.
4. **Default-features banner** in the Data tab — small, ~50 LOC,
   another easy reactive win.
5. **Remove the `?reactive=1` flag** once Phase 2 #1 ships — the
   Svelte sidebar becomes the only sidebar.

## Phase 3 (post-migration)

- Delete legacy `static/*.js` files as their consumers move to
  islands. The shared CSS (`static/style.css`) stays — Svelte
  components scope their own styles and inherit the global theme.
- Once `app.js`'s `switchTab` is the last cross-cutting legacy
  global, mount a top-level `App.svelte` and retire the legacy
  `index.html` shell.

## Things not done yet

- **TypeScript strict everywhere.** The store files use small ad-hoc
  types; once the FastAPI Pydantic models stabilize, run
  `openapi-typescript` against the OpenAPI doc and replace
  `lib/types.ts` with generated types.
- **`svelte-check` in CI.** Add a `pre-commit` hook that runs
  `npm run check` so type errors block commits, same shape as
  the Python `mypy` hook.
- **Vendored Svelte runtime.** Today the build pulls Svelte from
  npm; for fully offline / air-gapped installs the playbook should
  document a vendored fallback (similar to the TODO entry for
  `html2canvas`).
- **HMR with the FastAPI dev server.** `npm run dev` proxies to
  FastAPI but doesn't tell FastAPI to serve the Svelte HTML shell.
  Easiest path: a small dev-only FastAPI middleware that proxies
  `/` → `localhost:5173` when an env var is set.
