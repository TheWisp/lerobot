// Island entry for the Preact + signals version of the Run sidebar.
//
// Mounts only when ?reactive=1 AND ?framework=preact (default is svelte).
// Sets window.lerobotReactiveRun=true so the legacy renderRunForm()
// short-circuits the same way it does for the Svelte version — both
// frameworks share the same legacy-suppression contract.

import { render } from "preact";
import { RunSidebar } from "../components-preact/RunSidebar";
import { snapshot } from "../stores-preact/run";

declare global {
  interface Window {
    getReactiveRunConfig?: () => unknown;
    lerobotReactiveRun?: boolean;
  }
}

function init() {
  const params = new URLSearchParams(window.location.search);
  const reactive = params.get("reactive") === "1";
  const framework = params.get("framework") || "svelte";
  if (!reactive || framework !== "preact") return;

  window.lerobotReactiveRun = true;

  const target = document.getElementById("run-form");
  if (!target) {
    // eslint-disable-next-line no-console
    console.warn("[run-sidebar-preact] #run-form not found — legacy will load");
    return;
  }
  // Same legacy-cleanup as the Svelte entry: hide the duplicate workflow
  // selector buttons that live outside #run-form, then take over the node.
  document.querySelectorAll(".run-workflow-selector").forEach((el) => {
    (el as HTMLElement).style.display = "none";
  });
  target.innerHTML = "";
  render(<RunSidebar />, target);
  window.getReactiveRunConfig = () => snapshot();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}
