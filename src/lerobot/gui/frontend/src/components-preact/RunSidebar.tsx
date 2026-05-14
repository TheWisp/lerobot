// Preact + signals port of components/RunSidebar.svelte.
//
// Drop-in equivalent: same workflow selector, same TeleopForm /
// ReplayForm / Policy-placeholder branching, same validation banner.
// Adds a small "rendered by" badge so the A/B comparison is
// unambiguous when both are mounted in the same GUI session.

import { useEffect } from "preact/hooks";
import type { JSX } from "preact";
import {
  ensureRobotProfilesLoaded,
  ensureTeleopProfilesLoaded,
} from "../stores-preact/profiles";
import { ensureDatasetsLoaded } from "../stores-preact/datasets";
import { form, launchError, setWorkflow } from "../stores-preact/run";
import { TeleopForm } from "./TeleopForm";
import { ReplayForm } from "./ReplayForm";
import type { RunWorkflow } from "../lib/types";

import "./run-sidebar.css";

export function RunSidebar(): JSX.Element {
  // The Critical-TODO win at the data-loader level: kick off loads from
  // the sidebar mount, without any UI flag gating. The stores dedupe
  // across components so concurrent mounts share a single fetch.
  useEffect(() => {
    ensureRobotProfilesLoaded().catch(() => {});
    ensureTeleopProfilesLoaded().catch(() => {});
    ensureDatasetsLoaded().catch(() => {});
  }, []);

  const workflow = form.value.workflow;
  const err = launchError.value;

  const wfClass = (w: RunWorkflow) =>
    "workflow-btn" + (workflow === w ? " active" : "");

  return (
    <div class="run-sidebar-reactive-preact">
      <div class="framework-badge">rendered by: Preact + signals</div>

      <div class="workflow-selector">
        <button class={wfClass("teleop")} onClick={() => setWorkflow("teleop")}>
          Teleop
        </button>
        <button class={wfClass("replay")} onClick={() => setWorkflow("replay")}>
          Replay
        </button>
        <button class={wfClass("policy")} onClick={() => setWorkflow("policy")}>
          Policy
        </button>
      </div>

      <div class="form-body">
        {workflow === "teleop" && <TeleopForm />}
        {workflow === "replay" && <ReplayForm />}
        {workflow === "policy" && (
          <div class="policy-placeholder">
            <strong>Policy view — not yet ported.</strong>
            <p>
              The Policy workflow has HVLA, RLT, intervention-recording, and
              checkpoint discovery — the heaviest form in the GUI and the
              natural Phase 2 of the migration. Switch back to the legacy
              sidebar to use it:
            </p>
            <a class="legacy-link" href="?reactive=0">
              Switch to legacy sidebar
            </a>
          </div>
        )}
      </div>

      <div class={"validation-banner" + (err ? "" : " ok")}>
        {err ? (
          <>
            <span class="validation-icon">!</span> {err}
          </>
        ) : (
          <>
            <span class="validation-icon ok">OK</span> Ready to launch
          </>
        )}
      </div>
    </div>
  );
}
