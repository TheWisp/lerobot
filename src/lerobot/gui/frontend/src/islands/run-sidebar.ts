// Island entry for the reactive Run sidebar.
//
// Gated by ?reactive=1 (default off) so the legacy sidebar still works for
// users until Policy is ported in Phase 2. When enabled:
//   - the legacy renderRunForm() call is short-circuited
//   - the #run-form div is taken over by <RunSidebar/>
//   - window.getReactiveRunConfig() returns the current form snapshot for
//     the existing launchRun() to consume
//
// To opt in: open the GUI as http://.../?reactive=1

import { mount } from "svelte";
import RunSidebar from "../components/RunSidebar.svelte";
import { snapshot } from "../stores/run.svelte";

declare global {
  interface Window {
    getReactiveRunConfig?: () => unknown;
    lerobotReactiveRun?: boolean;
  }
}

function init() {
  const params = new URLSearchParams(window.location.search);
  const enabled = params.get("reactive") === "1";
  window.lerobotReactiveRun = enabled;
  if (!enabled) return;

  const target = document.getElementById("run-form");
  if (!target) {
    // eslint-disable-next-line no-console
    console.warn("[run-sidebar] #run-form not found — legacy sidebar will load");
    return;
  }
  // Hide the legacy workflow-selector buttons (they live outside #run-form
  // in the sidebar). The Svelte component renders its own. We hide rather
  // than remove so toggling ?reactive=0 still works without a full reload.
  document.querySelectorAll(".run-workflow-selector").forEach((el) => {
    (el as HTMLElement).style.display = "none";
  });
  // Empty the legacy sidebar contents and let Svelte take over the node.
  target.innerHTML = "";
  mount(RunSidebar, { target });
  window.getReactiveRunConfig = () => snapshot();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}
