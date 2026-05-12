import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "node:path";

// Build outputs into ../static/dist/ so FastAPI's existing
// StaticFiles mount at /static picks them up automatically.
// We emit one bundle per island so each can be loaded
// independently from the legacy index.html.
export default defineConfig({
  plugins: [svelte()],
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
