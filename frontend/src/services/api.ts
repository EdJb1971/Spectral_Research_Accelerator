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

  // ------------------------------------------------ domain-first acquisition (TG10.2)
  async listAcquisitions(): Promise<types.AcquisitionCatalogue> {
    const response = await fetch(`${BASE_URL}/acquisitions`, { method: 'GET' });
    return handleResponse<types.AcquisitionCatalogue>(response);
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

  async zarrProbes(): Promise<types.ZarrProbeLedgerResponse> {
    const response = await fetch(`${BASE_URL}/data/zarr/probes`, { method: 'GET' });
    return handleResponse<types.ZarrProbeLedgerResponse>(response);
  },

  async zarrProbe(payload: types.ZarrProbeRequest): Promise<types.ZarrProbeResponse> {
    const response = await fetch(`${BASE_URL}/data/zarr/probe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<types.ZarrProbeResponse>(response);
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
  },

  // ------------------------------------------------ verified evaluation receipts (T5.6g)
  async listEvaluationReports(): Promise<types.EvaluationReport[]> {
    return handleResponse<types.EvaluationReport[]>(
      await fetch(`${BASE_URL}/evaluation/receipts`, { method: 'GET' }));
  },

  async getEvaluationReport(reportId: string): Promise<types.EvaluationReport> {
    return handleResponse<types.EvaluationReport>(
      await fetch(`${BASE_URL}/evaluation/receipts/${encodeURIComponent(reportId)}`, { method: 'GET' }));
  },

  async importEvaluationReceipt(file: File): Promise<types.EvaluationReport> {
    const form = new FormData();
    form.append('file', file);
    return handleResponse<types.EvaluationReport>(
      await fetch(`${BASE_URL}/evaluation/receipts/import`, { method: 'POST', body: form }));
  },

  // ------------------------------------------------ channel records (TG8.4)
  //
  // Two calls, and the client makes neither choice for the researcher: `inspect` reports what a
  // file is and which declared domains admit it, `read` loads it under the one they pick. Both
  // send multipart, so no Content-Type header is set by hand.

  async inspectChannelRecord(file: File, options: { delimiter?: string; timeColumn?: string }
    = {}): Promise<types.ChannelInspection> {
    const form = new FormData();
    form.append('file', file);
    if (options.delimiter) form.append('delimiter', options.delimiter);
    if (options.timeColumn) form.append('time_column', options.timeColumn);
    return handleResponse<types.ChannelInspection>(
      await fetch(`${BASE_URL}/channels/inspect`, { method: 'POST', body: form }));
  },

  async readChannelRecord(file: File, domain: string, timeColumn: string,
                          options: { delimiter?: string;
                                     supportParentPx?: Record<string, number> } = {}
  ): Promise<types.ChannelRecord> {
    const form = new FormData();
    form.append('file', file);
    form.append('domain', domain);
    form.append('time_column', timeColumn);
    if (options.delimiter) form.append('delimiter', options.delimiter);
    if (options.supportParentPx && Object.keys(options.supportParentPx).length > 0) {
      form.append('support_parent_px', JSON.stringify(options.supportParentPx));
    }
    return handleResponse<types.ChannelRecord>(
      await fetch(`${BASE_URL}/channels/read`, { method: 'POST', body: form }));
  },

  // ------------------------------------------------ analysis workbench (TG11.1)
  // The original File, not the capped ChannelRecord preview, crosses this boundary. The backend
  // re-admits it under the selected domain and derives record facts before calling the engine.
  async getDomainAnalysisCapabilities(): Promise<types.DomainAnalysisCapabilities> {
    return handleResponse<types.DomainAnalysisCapabilities>(
      await fetch(`${BASE_URL}/analysis`, { method: 'GET' }));
  },

  async runDomainAnalysis(selection: types.ChannelRecordSelection,
                          operation: types.DomainAnalysisOperation,
                          configuration: Record<string, any>): Promise<types.DomainAnalysisResponse> {
    const form = new FormData();
    form.append('file', selection.file);
    form.append('operation', operation);
    form.append('domain', selection.record.domain);
    form.append('time_column', selection.timeColumn);
    form.append('configuration', JSON.stringify(configuration));
    if (Object.keys(selection.supportParentPx).length > 0) {
      form.append('support_parent_px', JSON.stringify(selection.supportParentPx));
    }
    return handleResponse<types.DomainAnalysisResponse>(
      await fetch(`${BASE_URL}/analysis/run`, { method: 'POST', body: form }));
  },

  // ------------------------------------------------ preregistration (TG11.2)
  //
  // R18's ordering is the server's to enforce, not this file's to present. Nothing here sends
  // a digest, a sealing time or a p-value: `describePartition` reads geometry and lineage
  // only, `sealFamily` declares a family and is told what it hashes to, and `confirmSeal`
  // sends the record and nothing else, because every setting the confirmatory sweep needs was
  // frozen into the seal and a knob left turnable after sealing is a choice made after the
  // declaration.

  async getPreregistrationCapabilities(): Promise<types.PreregistrationCapabilities> {
    return handleResponse<types.PreregistrationCapabilities>(
      await fetch(`${BASE_URL}/preregistration`, { method: 'GET' }));
  },

  async describePartition(selection: types.ChannelRecordSelection,
                          trainRatio: number,
                          embargoFrames: number): Promise<types.PartitionDescription> {
    const form = new FormData();
    form.append('file', selection.file);
    form.append('domain', selection.record.domain);
    form.append('time_column', selection.timeColumn);
    form.append('train_ratio', String(trainRatio));
    form.append('embargo_frames', String(embargoFrames));
    return handleResponse<types.PartitionDescription>(
      await fetch(`${BASE_URL}/preregistration/partition`, { method: 'POST', body: form }));
  },

  async sealFamily(selection: types.ChannelRecordSelection,
                   studyId: string,
                   generate: types.FamilyDeclaration,
                   confirm: types.FamilyDeclaration,
                   trainRatio: number,
                   embargoFrames: number): Promise<types.SealResponse> {
    const form = new FormData();
    form.append('file', selection.file);
    form.append('domain', selection.record.domain);
    form.append('time_column', selection.timeColumn);
    form.append('study_id', studyId);
    form.append('generate', JSON.stringify(generate));
    form.append('confirm', JSON.stringify(confirm));
    form.append('train_ratio', String(trainRatio));
    form.append('embargo_frames', String(embargoFrames));
    return handleResponse<types.SealResponse>(
      await fetch(`${BASE_URL}/preregistration/seal`, { method: 'POST', body: form }));
  },

  async listSeals(): Promise<types.SealListing> {
    return handleResponse<types.SealListing>(
      await fetch(`${BASE_URL}/preregistration/seals`, { method: 'GET' }));
  },

  async readSeal(sealSha256: string, publishedSha256?: string): Promise<Record<string, any>> {
    const query = publishedSha256
      ? `?published_sha256=${encodeURIComponent(publishedSha256)}` : '';
    return handleResponse<Record<string, any>>(
      await fetch(`${BASE_URL}/preregistration/seals/${encodeURIComponent(sealSha256)}${query}`,
        { method: 'GET' }));
  },

  async confirmSeal(sealSha256: string, file: File,
                    publishedSha256?: string): Promise<types.ConfirmationResponse> {
    const form = new FormData();
    form.append('file', file);
    if (publishedSha256) form.append('published_sha256', publishedSha256);
    return handleResponse<types.ConfirmationResponse>(
      await fetch(`${BASE_URL}/preregistration/seals/${encodeURIComponent(sealSha256)}/confirm`,
        { method: 'POST', body: form }));
  },

  // ------------------------------------------------ the evidence write path (TG11.3)
  //
  // The only writing client in this application. Every method here sends what was observed
  // and reads back a rung the server computed; none of them can send one. `appendPrecedence`
  // sends a record and a lag family and receives a verdict it did not choose.

  async getEvidenceCapabilities(): Promise<types.EvidenceCapabilities> {
    return handleResponse<types.EvidenceCapabilities>(
      await fetch(`${BASE_URL}/evidence`, { method: 'GET' }));
  },

  async openStudy(studyId: string, hypothesisId: string, statement: string,
                  prediction: string): Promise<types.EvidenceState> {
    return handleResponse<types.EvidenceState>(
      await fetch(`${BASE_URL}/evidence/studies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          study_id: studyId, hypothesis_id: hypothesisId,
          statement, prediction
        })
      }));
  },

  async getEvidenceHead(studyId: string): Promise<types.EvidenceState> {
    return handleResponse<types.EvidenceState>(
      await fetch(`${BASE_URL}/evidence/studies/${encodeURIComponent(studyId)}/head`,
        { method: 'GET' }));
  },

  async appendEvidence(studyId: string,
                       append: types.EvidenceAppend): Promise<types.EvidenceState> {
    return handleResponse<types.EvidenceState>(
      await fetch(`${BASE_URL}/evidence/studies/${encodeURIComponent(studyId)}/evidence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(append)
      }));
  },

  async appendPrecedence(studyId: string, expectedHeadSha256: string, label: string,
                         selection: types.ChannelRecordSelection, lags: string,
                         nSurrogates: number): Promise<types.EvidenceState> {
    const form = new FormData();
    form.append('expected_head_sha256', expectedHeadSha256);
    form.append('label', label);
    form.append('file', selection.file);
    form.append('domain', selection.record.domain);
    form.append('time_column', selection.timeColumn);
    form.append('lags', lags);
    form.append('n_surrogates', String(nSurrogates));
    return handleResponse<types.EvidenceState>(
      await fetch(
        `${BASE_URL}/evidence/studies/${encodeURIComponent(studyId)}/evidence/precedence`,
        { method: 'POST', body: form }));
  },

  // ------------------------------------------------ the findings surface (TG9.1)
  //
  // Read-only. None of these can change what may be claimed: a GET does not move a rung
  // (R22). The translation arrives already rendered, because Phase G9's rule is that this
  // client displays strings rather than assembling them.

  async listDomains(): Promise<types.DomainSummary[]> {
    return handleResponse<types.DomainSummary[]>(
      await fetch(`${BASE_URL}/findings/domains`, { method: 'GET' }));
  },

  // TG8.1: the adapter recipe the backend enforces, served rather than only documented, so
  // the contract a reader is held to and the contract the code checks are one tuple.
  async getOnboardingContract(): Promise<types.OnboardingContract> {
    return handleResponse<types.OnboardingContract>(
      await fetch(`${BASE_URL}/findings/onboarding`, { method: 'GET' }));
  },

  async getGlossary(name: string): Promise<types.DomainGlossaryPayload> {
    return handleResponse<types.DomainGlossaryPayload>(
      await fetch(`${BASE_URL}/findings/glossaries/${encodeURIComponent(name)}`,
        { method: 'GET' }));
  },

  async listStudies(): Promise<types.StudySummary[]> {
    return handleResponse<types.StudySummary[]>(
      await fetch(`${BASE_URL}/findings/studies`, { method: 'GET' }));
  },

  async getStudy(studyId: string): Promise<Record<string, unknown>> {
    return handleResponse<Record<string, unknown>>(
      await fetch(`${BASE_URL}/findings/studies/${encodeURIComponent(studyId)}`,
        { method: 'GET' }));
  },

  async getStudyOutputs(studyId: string): Promise<Record<string, unknown>> {
    return handleResponse<Record<string, unknown>>(
      await fetch(`${BASE_URL}/findings/studies/${encodeURIComponent(studyId)}/outputs`,
        { method: 'GET' }));
  },

  async getTranslation(studyId: string, glossary: string): Promise<types.TranslatedFinding> {
    const query = new URLSearchParams({ glossary }).toString();
    return handleResponse<types.TranslatedFinding>(
      await fetch(`${BASE_URL}/findings/studies/${encodeURIComponent(studyId)}/translation?${query}`,
        { method: 'GET' }));
  }
};
