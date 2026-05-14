// Preact + @preact/signals version of stores/profiles.svelte.ts.
//
// Same external API (ensureLoaded + reload) so components can be swapped
// at the call site. The internal change: signals are plain JS objects
// (`{ value: T }`) that live in normal .ts files — no .svelte.ts
// compiler magic, no special runtime to evaluate them in a unit test.
//
// The "decouple data loaders from UI state" win is identical to the
// Svelte version: ensureLoaded is idempotent, deduplicates in-flight
// promises, and never consults a UI flag.

import { signal } from "@preact/signals";
import { getJson } from "../lib/api";
import type { RobotProfile, TeleopProfile } from "../lib/types";

interface ProfileState<T> {
  value: T[];
  loaded: boolean;
  error: string | null;
}

function makeStore<T>(endpoint: string) {
  // Single signal holding the entire state object. Component reads
  // s.value to subscribe; mutation goes through the wrapper functions.
  const s = signal<ProfileState<T>>({ value: [], loaded: false, error: null });
  let inflight: Promise<T[]> | null = null;

  async function ensureLoaded(): Promise<T[]> {
    if (s.value.loaded) return s.value.value;
    if (inflight) return inflight;
    inflight = (async () => {
      try {
        const data = await getJson<T[]>(endpoint);
        s.value = { value: data, loaded: true, error: null };
        return data;
      } catch (e) {
        s.value = {
          ...s.value,
          error: e instanceof Error ? e.message : String(e),
        };
        throw e;
      } finally {
        inflight = null;
      }
    })();
    return inflight;
  }

  function reload(): Promise<T[]> {
    s.value = { ...s.value, loaded: false };
    return ensureLoaded();
  }

  return { state: s, ensureLoaded, reload };
}

const robotStore = makeStore<RobotProfile>("/api/robot/profiles");
const teleopStore = makeStore<TeleopProfile>("/api/robot/teleops");

export const robotProfiles = robotStore.state;
export const teleopProfiles = teleopStore.state;
export const ensureRobotProfilesLoaded = robotStore.ensureLoaded;
export const ensureTeleopProfilesLoaded = teleopStore.ensureLoaded;
export const reloadRobotProfiles = robotStore.reload;
export const reloadTeleopProfiles = teleopStore.reload;
