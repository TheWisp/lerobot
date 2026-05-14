// Preact + signals version of stores/run.svelte.ts.
//
// Same external API. Form state is one signal containing a plain object;
// derived helpers (validateForLaunch, etc.) are plain functions that read
// from `form.value` — components wrap them in `computed()` if they want
// reactive derived state in templates.

import { signal, computed } from "@preact/signals";
import type { RunWorkflow } from "../lib/types";

export interface TeleopFormState {
  robot: string;
  teleop: string;
  fps: number;
  recordDataset: string;
  newDatasetName: string;
  task: string;
  numEpisodes: number;
  episodeTime: number;
  resetTime: number;
}

export interface ReplayFormState {
  robot: string;
  episode: string;
}

export interface RunFormState {
  workflow: RunWorkflow;
  teleop: TeleopFormState;
  replay: ReplayFormState;
}

export const form = signal<RunFormState>({
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
  replay: { robot: "", episode: "" },
});

// `computed` is the signals equivalent of Svelte's $derived — pure
// function of the current signal state, lazily re-evaluates when a
// dependency changes.
export const teleopShowRecordFields = computed(
  () => form.value.teleop.recordDataset !== "",
);

export const teleopShowNewDatasetName = computed(
  () => form.value.teleop.recordDataset === "__new__",
);

export const launchError = computed<string | null>(() => {
  const f = form.value;
  if (f.workflow === "teleop") {
    if (!f.teleop.robot) return "Pick a robot profile";
    if (!f.teleop.teleop) return "Pick a teleop profile";
    if (teleopShowRecordFields.value) {
      if (teleopShowNewDatasetName.value && !f.teleop.newDatasetName.trim()) {
        return "Name the new dataset";
      }
      if (!f.teleop.task.trim()) return "Set a task description";
    }
    return null;
  }
  if (f.workflow === "replay") {
    if (!f.replay.robot) return "Pick a robot profile";
    if (!f.replay.episode) return "Pick an episode";
    return null;
  }
  return "Policy workflow uses the legacy form (toggle ?reactive=0).";
});

// Helper for legacy launchRun() to read the current form state. Same
// shape as the Svelte snapshot() export.
export function snapshot(): RunFormState {
  return JSON.parse(JSON.stringify(form.value));
}

// Helper used by components for field updates. Pattern: form.value = {
// ...form.value, teleop: { ...form.value.teleop, fps: 30 } }. This
// helper makes the call sites shorter and keeps the immutable-update
// rule visible in one place.
export function updateTeleop(patch: Partial<TeleopFormState>): void {
  form.value = { ...form.value, teleop: { ...form.value.teleop, ...patch } };
}
export function updateReplay(patch: Partial<ReplayFormState>): void {
  form.value = { ...form.value, replay: { ...form.value.replay, ...patch } };
}
export function setWorkflow(w: RunWorkflow): void {
  form.value = { ...form.value, workflow: w };
}
