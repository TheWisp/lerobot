// Run sidebar form state.
//
// The legacy frontend lets renderRunForm() build raw HTML strings and then
// reads form values back via document.getElementById(...).value at launch
// time. Field visibility is patched with _toggleHvlaRecordFields() called
// from 4 different code paths.
//
// This store owns the form state directly:
//   - workflow / teleop fields live in $state
//   - "show record fields" is a $derived from form state, not a manual flag
//   - launch readiness is a $derived from required-field validation
//
// Legacy launchRun() reads via window.getReactiveRunConfig() — exposed by
// the island entry point.

import type { RunWorkflow } from "../lib/types";

export interface TeleopFormState {
  robot: string;
  teleop: string;
  fps: number;
  recordDataset: string; // "" means do not record
  newDatasetName: string;
  task: string;
  numEpisodes: number;
  episodeTime: number;
  resetTime: number;
}

export interface ReplayFormState {
  robot: string;
  episode: string; // "datasetId:episodeIdx" identifier
}

interface RunFormState {
  workflow: RunWorkflow;
  teleop: TeleopFormState;
  replay: ReplayFormState;
}

export const form: RunFormState = $state({
  workflow: "teleop",
  teleop: {
    robot: "",
    teleop: "",
    fps: 60,
    recordDataset: "",
    newDatasetName: "",
    task: "",
    numEpisodes: 50,
    episodeTime: 60,
    resetTime: 60,
  },
  replay: {
    robot: "",
    episode: "",
  },
});

// The reactive win, in one line: legacy had _toggleHvlaRecordFields()
// disabled-state hand-rolled across N call sites; here the visibility is
// just data the template reads.
export function teleopShowRecordFields(): boolean {
  return form.teleop.recordDataset !== "";
}

export function teleopShowNewDatasetName(): boolean {
  return form.teleop.recordDataset === "__new__";
}

// Derived "is the form valid for launching?". The legacy frontend has a
// separate _validateLaunch() that builds a map of required-field rules per
// workflow; we keep the same shape but make it pure so a $derived in the
// component picks it up automatically.
export function validateForLaunch(): string | null {
  if (form.workflow === "teleop") {
    if (!form.teleop.robot) return "Pick a robot profile";
    if (!form.teleop.teleop) return "Pick a teleop profile";
    if (teleopShowRecordFields()) {
      if (teleopShowNewDatasetName() && !form.teleop.newDatasetName.trim()) {
        return "Name the new dataset";
      }
      if (!form.teleop.task.trim()) return "Set a task description";
    }
    return null;
  }
  if (form.workflow === "replay") {
    if (!form.replay.robot) return "Pick a robot profile";
    if (!form.replay.episode) return "Pick an episode";
    return null;
  }
  // Policy is not yet ported; the new sidebar shows a "use legacy" pane.
  return "Policy workflow uses the legacy form (toggle ?reactive=0).";
}

// Serializable snapshot for the legacy launchRun() to consume.
export function snapshot(): RunFormState {
  return JSON.parse(JSON.stringify(form));
}
