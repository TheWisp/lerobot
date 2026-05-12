import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

export default {
  preprocess: vitePreprocess(),
  // Tell svelte-check we're using runes-mode (Svelte 5 reactivity).
  compilerOptions: {
    runes: true,
  },
};
