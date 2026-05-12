// Shared store for robot + teleop profiles.
//
// The legacy frontend has a documented pain point (TODO.md, "Decouple data
// loaders from UI state"): three loaders gate fetches on UI `expanded` /
// `tabInitialized` flags, so cross-tab consumers get stuck on "Loading...".
// This store fixes that by:
//
//   - exposing idempotent ensureLoaded() that does NOT consult any UI state
//   - caching results in $state-backed Svelte 5 runes so consumers
//     auto-rerender when the data lands
//   - returning a single Promise during in-flight loads so the N callers in
//     N tabs that all race to render get exactly one HTTP request
//
// Anyone — Run tab, Robot tab, a modal, a future React island — can call
// ensureRobotProfilesLoaded() and bind to robotProfiles.value. No UI flags
// in this layer.

import { getJson } from "../lib/api";
import type { RobotProfile, TeleopProfile } from "../lib/types";

interface ProfileState<T> {
  value: T[];
  loaded: boolean;
  error: string | null;
}

// Svelte 5 runes can be declared at module scope when wrapped in a function
// or class. We use plain $state() bindings inside small wrapper functions
// so the runes compiler recognizes them.

function makeStore<T>(endpoint: string) {
  const s: ProfileState<T> = $state({ value: [], loaded: false, error: null });
  let inflight: Promise<T[]> | null = null;

  async function ensureLoaded(): Promise<T[]> {
    if (s.loaded) return s.value;
    if (inflight) return inflight;
    inflight = (async () => {
      try {
        const data = await getJson<T[]>(endpoint);
        s.value = data;
        s.loaded = true;
        s.error = null;
        return data;
      } catch (e) {
        s.error = e instanceof Error ? e.message : String(e);
        throw e;
      } finally {
        inflight = null;
      }
    })();
    return inflight;
  }

  function reload(): Promise<T[]> {
    s.loaded = false;
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
