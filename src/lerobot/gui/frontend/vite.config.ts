import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import preact from "@preact/preset-vite";
import { resolve } from "node:path";

// Build outputs into ../static/dist/ so FastAPI's existing
// StaticFiles mount at /static picks them up automatically.
// We emit one bundle per island so each can be loaded
// independently from the legacy index.html.
//
// This config builds both the Svelte islands and a parallel Preact
// version of the Run sidebar so the two can be A/B'd in the same
// running GUI via ?framework=svelte | preact. See README.md for the
// comparison protocol.
export default defineConfig({
  // The Preact plugin transforms .tsx with Babel + aliases react -> preact.
  // It only touches .tsx files; Svelte's plugin owns .svelte files. No
  // overlap, no global JSX runtime change.
  plugins: [svelte(), preact({ include: /\.tsx$/ })],
  build: {
    outDir: resolve(__dirname, "../static/dist"),
    emptyOutDir: true,
    manifest: true,
    target: "es2022",
    sourcemap: true,
    rollupOptions: {
      input: {
        "bug-report": resolve(__dirname, "src/islands/bug-report.ts"),
        "run-sidebar": resolve(__dirname, "src/islands/run-sidebar.ts"),
        "run-sidebar-preact": resolve(__dirname, "src/islands/run-sidebar-preact.tsx"),
      },
      output: {
        // Stable names so legacy index.html can <script src="/static/dist/...">
        // and <link href="/static/dist/..."> without parsing the manifest.
        // Hashed shared chunks stay cacheable; the per-island entries (JS
        // and their CSS sidecars) keep predictable names.
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: (info) => {
          // CSS named after the entry it belongs to → "bug-report.css".
          if (info.name?.endsWith(".css")) return "[name][extname]";
          return "assets/[name]-[hash][extname]";
        },
      },
    },
  },
  server: {
    // `npm run dev` proxies API calls to the running FastAPI server so
    // hot-reload works without CORS hacks.
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/static": "http://127.0.0.1:8000",
    },
  },
});
