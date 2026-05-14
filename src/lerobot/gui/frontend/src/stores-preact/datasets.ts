// Preact + signals version of stores/datasets.svelte.ts.

import { signal } from "@preact/signals";
import { getJson } from "../lib/api";
import type { DatasetSummary } from "../lib/types";

interface OpenedDataset {
  id: string;
  repo_id: string;
}

interface OpenedDatasetsResponse {
  datasets: OpenedDataset[];
}

interface DatasetsState {
  value: DatasetSummary[];
  loaded: boolean;
  error: string | null;
}

export const datasets = signal<DatasetsState>({
  value: [],
  loaded: false,
  error: null,
});

let inflight: Promise<DatasetSummary[]> | null = null;

export async function ensureDatasetsLoaded(): Promise<DatasetSummary[]> {
  if (datasets.value.loaded) return datasets.value.value;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const data = await getJson<OpenedDatasetsResponse>("/api/datasets/opened");
      datasets.value = { value: data.datasets, loaded: true, error: null };
      return data.datasets;
    } catch (e) {
      datasets.value = {
        ...datasets.value,
        error: e instanceof Error ? e.message : String(e),
      };
      throw e;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

export function reloadDatasets(): Promise<DatasetSummary[]> {
  datasets.value = { ...datasets.value, loaded: false };
  return ensureDatasetsLoaded();
}
