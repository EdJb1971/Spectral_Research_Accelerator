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
  }
};
