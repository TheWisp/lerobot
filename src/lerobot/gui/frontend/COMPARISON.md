# Svelte 5 vs Preact + Signals — Apple-to-Apple A/B

Both versions of the Run sidebar (`RunSidebar` + `TeleopForm` + `ReplayForm`)
are mounted from the same `index.html`. Pick via:

```
http://localhost:8000/?reactive=1&framework=svelte
http://localhost:8000/?reactive=1&framework=preact
```

A "rendered by:" badge appears at the top of each sidebar so the framework
is unambiguous at a glance.

## What's identical

- Stores (profile / dataset / run-form) have the same external API:
  `ensureLoaded()`, `reload()`, plus a reactive primitive holding state.
- Components implement the same workflow selector + Teleop form + Replay
  form + Policy placeholder + validation banner.
- Same shared `lib/api.ts`, `lib/toast.ts`, `lib/types.ts`.
- Same Vite build pipeline, output to `static/dist/`.
- CDP-driven smoke test produces equivalent state transitions.

## Measurements (2026-05-14, run-sidebar A/B)

### Lines of code (equivalent units)

| Layer                              | Svelte                              | Preact                  | Δ       |
| ---------------------------------- | ----------------------------------- | ----------------------- | ------- |
| Stores (profiles + datasets + run) | 222                                 | 219                     | -3      |
| Components (3 of them)             | 481 (in `.svelte` w/ scoped styles) | 345 TSX + 203 CSS = 548 | +14%    |
| Island entry                       | 56                                  | 47                      | -9      |
| **Total**                          | **759**                             | **814**                 | **+7%** |

Preact's overhead is split: JSX is slightly more verbose for trivial markup
(every element needs a closing tag, every event needs an explicit binding
expression), and CSS lives in a separate file rather than co-located in
`<style scoped>`. The store layer is a wash; the entry point is slightly
smaller for Preact because `render(<X/>, target)` is one call vs Svelte's
`mount(X, { target })` + boilerplate.

### Bundle sizes (gzipped)

| Bundle                                   | Svelte      | Preact      |
| ---------------------------------------- | ----------- | ----------- |
| Island entry (`run-sidebar(-preact).js`) | 5.5 KB      | 10.2 KB     |
| Shared chunk (Svelte runtime / api util) | 11.7 KB     | 0.3 KB      |
| **First-island total**                   | **17.2 KB** | **10.5 KB** |
| Each additional island (amortized)       | +5 KB       | +9–10 KB    |

Reading: Preact's "all included" first island is **smaller** than Svelte's
runtime + first island. As you add more islands, Svelte amortizes its
runtime across them; Preact's per-island cost is closer to constant.

For this GUI's foreseeable surface (~5 components on the migration path),
both end up in the 25–35 KB gz total range. Not a decision driver.

### Build pipeline

|                        | Svelte                                    | Preact                    |
| ---------------------- | ----------------------------------------- | ------------------------- |
| Number of Vite plugins | 1 (`@sveltejs/vite-plugin-svelte`)        | 1 (`@preact/preset-vite`) |
| Build time (cold)      | ~480 ms                                   | ~480 ms                   |
| `npm install` size     | + 33 packages                             | + 33 packages             |
| TS support             | via `<script lang="ts">` + `svelte-check` | native, `tsc`/Vite        |

Both coexist in the same `vite.config.ts` with no conflict — the Svelte
plugin owns `*.svelte`, the Preact plugin owns `*.tsx`.

## Subjective experience writing the port

### Where Svelte felt better

- **Template syntax is HTML-native.** `{#if}` and `bind:value` are tighter
  than JSX equivalents. For trivial forms this is a real difference.
- **Scoped styles in one file** — no CSS file naming, no class-name
  collisions. The `.svelte` single-file format is genuinely productive.
- **Animations and transitions are built in.** The `reveal-fade` animation
  in TeleopForm is `class:reveal` + a `@keyframes` rule. In Preact you do
  the same thing manually with CSS, but you lose `transition:slide` /
  `transition:fade` directive shorthand.

### Where Preact felt better

- **Errors are plain JS stacks.** When I made a typo in the TeleopForm
  port, the stack trace pointed at the exact TSX line. Svelte's
  compiler-generated output has helped trace errors back to source most
  of the time, but the indirection is real and shows up at the worst
  moments (asynchronous code, dynamic component composition).
- **Stores are plain TS files.** No `.svelte.ts` extension, no rune
  semantics that only work in Svelte-compiled files. The Preact stores
  are testable in vanilla Node without setup.
- **Bridge to legacy code is trivial.** From a vanilla `<script>` block
  the legacy GUI can `import { form } from './stores-preact/run'` and
  read/write `form.value` directly. Svelte's `$state` doesn't expose a
  similar synchronous reader.
- **JSX is just JavaScript.** Wanted to render N components by a list of
  config objects? `cfgs.map(c => <X {...c}/>)`. The Svelte equivalent
  requires either `{#each}` or `svelte:component` + careful prop typing.
- **AI/SO corpus is dense.** The Preact components I wrote here used 0
  external lookups (just standard React patterns). The Svelte 5 runes
  port from the previous branch needed several runes-specific lookups —
  runes shipped Oct 2024 and the corpus is thin.

### Where they were equivalent

- Reactivity model. `$state` ↔ `signal()`, `$derived` ↔ `computed()`,
  `$effect` ↔ `useEffect` (in component) or `effect()` (outside).
- Time to write the port (~3 hours for the components + stores).
- Visual output is pixel-equivalent (by design, see screenshots).
- Performance for this app's scale — both unmeasurably fast.

## Honest verdict against the user's three values

> _"Simplicity when landing correct code (scalability + maintenance),
> flexibility for corner cases, separation of concern."_

1. **Simplicity / maintenance** — close tie. Svelte has the tighter
   template syntax (less to type per feature). Preact has the bigger
   ecosystem and dense AI/SO help, which matters more for a solo project.
   **Preact wins on long-term maintenance**; Svelte wins on per-keystroke
   ergonomics.

2. **Flexibility for corner cases** — **Preact wins clearly.** Signals
   are plain JS objects. Components are plain functions. Dynamic
   composition, programmatic mount, render-to-string, sharing state with
   non-component code (Workers, tests, vanilla scripts) all work
   trivially.

3. **Separation of concern** — **Preact wins.** State (signal),
   components (function), derived (computed) are three independent
   primitives with no framework coupling. A signal works the same whether
   accessed from a TSX component, a unit test, or a legacy `<script>`.
   Svelte's `.svelte` SFC format and `.svelte.ts` store extension couple
   user code to the framework runtime.

## Decision

**Pick Preact + signals** for everything past Phase 1.

The Svelte version is _defensible_ — bundle is smaller for one island,
templates are tighter — but on the values the user prioritized, Preact
wins on two of three and ties on the third. The flexibility and
separation-of-concern wins compound as the migration grows.

**Switching cost from this branch:**

- Stores: already in place (`stores-preact/`). Done.
- 3 components: already in place. Done.
- Bug-report modal (currently Svelte): not yet ported to Preact. ~1 hour.
- Cleanup: delete `components/`, `stores/`, `islands/run-sidebar.ts`,
  Svelte plugin in `vite.config.ts`, Svelte deps in `package.json`,
  Svelte typos exclusions if any. ~30 min.

**Total to collapse to Preact-only**: ~2 hours. Lower if we keep the
Svelte version on `proto/frontend-svelte` for reference and just stop
shipping it.

## Screenshots (2026-05-14)

`/tmp/ab-shots/`, uploaded to catbox.moe:

| State                    | Svelte                              | Preact                              |
| ------------------------ | ----------------------------------- | ----------------------------------- |
| Teleop empty             | https://files.catbox.moe/iqrf77.png | https://files.catbox.moe/57sg6j.png |
| Replay                   | https://files.catbox.moe/seov1r.png | https://files.catbox.moe/0a5jm3.png |
| Teleop + record revealed | https://files.catbox.moe/z4r7zn.png | https://files.catbox.moe/vuq7yx.png |

Visual output is intentionally identical apart from the badge.
