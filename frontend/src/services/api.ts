import * as types from '../types/api';

const BASE_URL = '/api/v1';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson?.detail) {
        if (typeof errorJson.detail === 'string') {
          errorMessage = errorJson.detail;
        } else if (Array.isArray(errorJson.detail)) {
          errorMessage = errorJson.detail.map((err: any) => `${err.loc?.join('.') || 'Error'}: ${err.msg}`).join('; ');
        }
      }
    } catch {
      // JSON parsing failed, keep default message
    }
    throw new Error(errorMessage);
  }
  return response.json() as Promise<T>;
}

/**
 * Read a file response. The filename comes from `Content-Disposition`, which the API sets and
 * exposes via `Access-Control-Expose-Headers` - without that a cross-origin browser cannot read
 * the header at all and every download would be named "download".
 */
async function downloadResponse(response: Response): Promise<types.ExportResult> {
  if (!response.ok) {
    let message = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === 'string') message = body.detail;
    } catch {
      // the error body was not JSON; keep the status line
    }
    throw new Error(message);
  }
  const disposition = response.headers.get('Content-Disposition') || '';
  const match = disposition.match(/filename="?([^";]+)"?/);
  return { blob: await response.blob(), filename: match ? match[1] : 'spectralearth-export' };
}

export const apiService = {
  // Spectral Transform Engine
  async applyTransform(payload: types.TransformRequest): Promise<types.TransformResponse> {
    const response = await fetch(`${BASE_URL}/transforms/apply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.TransformResponse>(response);
  },

  async listTrainingRepresentations(
    levels: number, wavelet: string, height: number, width: number
  ): Promise<types.TrainingRepresentationCatalogue> {
    const params = new URLSearchParams({
      levels: String(levels), wavelet, height: String(height), width: String(width),
    });
    return handleResponse<types.TrainingRepresentationCatalogue>(
      await fetch(`${BASE_URL}/training/representations?${params}`, { method: 'GET' })
    );
  },

  // Synthetic Field Generator
  async generateSynthetic(payload: types.GenerateRequest): Promise<types.GenerateResponse> {
    const response = await fetch(`${BASE_URL}/synthetic/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.GenerateResponse>(response);
  },

  // Perturbation Engine
  async perturbField(payload: types.PerturbRequest): Promise<types.PerturbResponse> {
    const response = await fetch(`${BASE_URL}/synthetic/perturb`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.PerturbResponse>(response);
  },

  // Boundary Condition Lab
  async analyzeBoundary(payload: types.BoundaryRequest): Promise<types.BoundaryResponse> {
    const response = await fetch(`${BASE_URL}/boundary/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.BoundaryResponse>(response);
  },

  // Meteorological Data Adapter
  async listDatasets(): Promise<types.DatasetMetadata[]> {
    const response = await fetch(`${BASE_URL}/data/datasets`, {
      method: 'GET',
    });
    return handleResponse<types.DatasetMetadata[]>(response);
  },

  async sliceDataset(payload: types.SliceRequest): Promise<types.SliceResponse> {
    const response = await fetch(`${BASE_URL}/data/slice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.SliceResponse>(response);
  },

  // Analysis & Diagnostics Engine
  async computeDiagnostics(payload: types.DiagnosticsRequest): Promise<types.DiagnosticsResponse> {
    const response = await fetch(`${BASE_URL}/analysis/diagnostics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.DiagnosticsResponse>(response);
  },

  async decomposeErrors(payload: types.ErrorDecompositionRequest): Promise<types.ErrorDecompositionResponse> {
    const response = await fetch(`${BASE_URL}/analysis/error-decomposition`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.ErrorDecompositionResponse>(response);
  },

  // Declarative Experiment Engine
  async createExperiment(payload: types.ExperimentRequest): Promise<types.ExperimentResponse> {
    const response = await fetch(`${BASE_URL}/experiments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.ExperimentResponse>(response);
  },

  async getExperiment(id: string): Promise<types.ExperimentDetailResponse> {
    const response = await fetch(`${BASE_URL}/experiments/${id}`, {
      method: 'GET',
    });
    return handleResponse<types.ExperimentDetailResponse>(response);
  },

  async getExperimentLineage(id: string): Promise<types.LineageResponse> {
    const response = await fetch(`${BASE_URL}/experiments/${id}/lineage`, {
      method: 'GET',
    });
    return handleResponse<types.LineageResponse>(response);
  },

  // Automated Hypothesis Engine
  async discoverHypotheses(payload: types.DiscoverRequest): Promise<types.HypothesisResponse[]> {
    const response = await fetch(`${BASE_URL}/hypothesis/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.HypothesisResponse[]>(response);
  },

  async getProposals(patternType?: string, minConfidence?: number): Promise<types.HypothesisResponse[]> {
    const params = new URLSearchParams();
    if (patternType) params.append('pattern_type', patternType);
    if (minConfidence !== undefined) params.append('min_confidence', String(minConfidence));

    const url = `${BASE_URL}/hypothesis/proposals${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url, {
      method: 'GET',
    });
    return handleResponse<types.HypothesisResponse[]>(response);
  },
  // ---------------------------------------------------------------- platform status
  // T3.5.22. Every call below hits an endpoint that already existed and had no consumer.

  async getHealth(): Promise<types.HealthResponse> {
    const response = await fetch(`${BASE_URL}/health`, { method: 'GET' });
    return handleResponse<types.HealthResponse>(response);
  },

  async listBenchmarks(): Promise<types.BenchmarkResponse[]> {
    const response = await fetch(`${BASE_URL}/benchmarks`, { method: 'GET' });
    return handleResponse<types.BenchmarkResponse[]>(response);
  },

  async listDataSources(): Promise<types.DataSourceInfo[]> {
    const response = await fetch(`${BASE_URL}/data/sources`, { method: 'GET' });
    return handleResponse<types.DataSourceInfo[]>(response);
  },

  // ---------------------------------------------------------------- ERA5 over Zarr

  async zarrCatalogue(): Promise<types.ZarrCatalogueResponse> {
    const response = await fetch(`${BASE_URL}/data/zarr/catalogue`, { method: 'GET' });
    return handleResponse<types.ZarrCatalogueResponse>(response);
  },

  async zarrCached(): Promise<types.ZarrCachedResponse> {
    const response = await fetch(`${BASE_URL}/data/zarr/cached`, { method: 'GET' });
    return handleResponse<types.ZarrCachedResponse>(response);
  },

  async zarrInspect(payload: types.ZarrCropRequest): Promise<types.ZarrInspectResponse> {
    const response = await fetch(`${BASE_URL}/data/zarr/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.ZarrInspectResponse>(response);
  },
  // ---------------------------------------------------------------- export (T3.5.23)

  async exportField(payload: types.ExportFieldRequest): Promise<types.ExportResult> {
    return downloadResponse(await fetch(`${BASE_URL}/export/field`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }));
  },

  async exportTable(payload: types.ExportTableRequest): Promise<types.ExportResult> {
    return downloadResponse(await fetch(`${BASE_URL}/export/table`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }));
  },
  // ---------------------------------------------------------------- import (T3.5.24)

  async importInspect(file: File): Promise<types.ImportInspectResponse> {
    const form = new FormData();
    form.append('file', file);
    // No Content-Type header: the browser must set it, because it alone knows the multipart
    // boundary. Setting it by hand produces a body the server cannot parse.
    return handleResponse<types.ImportInspectResponse>(
      await fetch(`${BASE_URL}/import/inspect`, { method: 'POST', body: form }));
  },

  async importField(file: File, variable?: string,
                    selection?: Record<string, number>): Promise<types.ImportFieldResponse> {
    const form = new FormData();
    form.append('file', file);
    if (variable) form.append('variable', variable);
    if (selection && Object.keys(selection).length) {
      form.append('selection', JSON.stringify(selection));
    }
    return handleResponse<types.ImportFieldResponse>(
      await fetch(`${BASE_URL}/import/field`, { method: 'POST', body: form }));
  },

  // ---------------------------------------------------------------- evidence & registries

  async runBenchmarks(rootSeed?: number, names?: string[]): Promise<types.BenchmarkSuiteResponse> {
    const params = new URLSearchParams();
    if (rootSeed !== undefined) params.append('root_seed', String(rootSeed));
    (names || []).forEach(n => params.append('name', n));
    const query = params.toString() ? `?${params.toString()}` : '';
    return handleResponse<types.BenchmarkSuiteResponse>(
      await fetch(`${BASE_URL}/benchmarks/run${query}`, { method: 'POST' }));
  },

  async listTransforms(): Promise<types.RegistryEntry[]> {
    return handleResponse<types.RegistryEntry[]>(
      await fetch(`${BASE_URL}/transforms`, { method: 'GET' }));
  },

  async listActions(): Promise<types.RegistryEntry[]> {
    return handleResponse<types.RegistryEntry[]>(
      await fetch(`${BASE_URL}/actions`, { method: 'GET' }));
  }
};
