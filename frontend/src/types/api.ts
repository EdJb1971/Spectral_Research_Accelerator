export interface TransformRequest {
  field_data: number[][];
  transform_type: string;
  config?: Record<string, any>;
}

export interface TransformResponse {
  reconstructed_field: number[][];
  coefficients: Record<string, any>;
  metrics: {
    mean_squared_error: number;
    max_absolute_error: number;
  };
}

export interface TrainingRepresentationEntry {
  name: string;
  label: string;
  status: 'accepted' | 'analysis_only';
  batched: boolean;
  autograd: boolean;
  exact_inverse: boolean | null;
  coefficient_ratio: string;
  shift_behavior: string;
  directionality: string;
  boundary: string;
  scientific_role: string;
  limitations: string[];
  verified: string[];
  not_run: string[];
  selected_configuration?: {
    wavelet?: string;
    level1?: string;
    qshift?: string;
    levels: number;
    spatial_shape: number[];
    coefficient_channels_per_input_channel?: number;
    atlas_planes_per_input_channel?: number;
    accumulated_support_by_level?: number[];
    valid_interior_halfwidth_by_level?: number[];
    valid_interior_shape_by_level?: number[][];
    valid_interior_halfwidth_parent_px_by_level?: number[];
    valid_interior_halfwidth_native_px_by_level?: number[];
    native_shape_by_level?: number[][];
    valid_interior_native_shape_by_level?: number[][];
    coarsest_scale_has_valid_interior: boolean;
    implementation_policy?: string;
    display_contract?: string;
  };
}

export interface TrainingRepresentationCatalogue {
  contract: string;
  selected_levels: number;
  selected_wavelet: string;
  selected_spatial_shape: number[];
  representations: TrainingRepresentationEntry[];
}

export interface GenerateRequest {
  type: string;
  height: number;
  width: number;
  params?: Record<string, any>;
}

export interface GenerateResponse {
  field_data: number[][];
  coords: Record<string, number[]>;
  metadata: Record<string, any>;
}

export interface PerturbationItem {
  type: string;
  angle?: number;
  shift_x?: number;
  shift_y?: number;
  noise_type?: string;
  level?: number;
  /** Defect D34: the engine has taken a seed since T3.5.12, but the API never passed one, so
   *  every perturbation requested over HTTP was irreproducible. Omit for an unseeded draw -
   *  which the response then reports as `reproducible: false`. */
  seed?: number | null;
}

export interface PerturbRequest {
  field_data: number[][];
  perturbations: PerturbationItem[];
}

export interface PerturbResponse {
  perturbed_field: number[][];
  metrics: {
    mean_squared_error: number;
    root_mean_squared_error: number;
    peak_signal_to_noise_ratio: number;
    structural_similarity_index: number;
    spectral_energy_shift: number;
  };
  provenance?: Array<Record<string, any>>;
  reproducible?: boolean;
}

export interface BoundaryRequest {
  field_data: number[][];
  treatment: string;
  pad_width: number;
  window_type?: string;
  window_alpha?: number;
  reference_field_data?: number[][];
}

export interface DistanceProfile {
  distance: number;
  mean_gradient: number;
  max_gradient: number;
  mean_absolute_error: number;
  max_absolute_error: number;
}

export interface BoundaryResponse {
  padded_field: number[][];
  distance_profiles: DistanceProfile[];
  spectral_leakage: number;
  has_reference: boolean;
}

export interface DatasetMetadata {
  id: string;
  name: string;
  description: string;
  // The backend has returned these since T3.5.15 and the frontend type omitted all four, so
  // the UI could not have shown them even if it had tried. `is_simulated` is the one fact
  // that changes what every number derived from this dataset means.
  is_simulated?: boolean;
  fallback_reason?: string | null;
  source_kind?: string;
  source_path?: string | null;
  variables: string[];
  variables_metadata?: Record<string, any>;
  pressure_levels: number[] | null;
  time_range: string[];
  spatial_resolution: string;
  bounding_box: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
}

export interface SliceRequest {
  dataset_id: string;
  variable: string;
  time?: string;
  level?: number;
  lat_range?: [number, number];
  lon_range?: [number, number];
}

export interface SliceResponse {
  field_data: number[][];
  coords: Record<string, number[]>;
  metadata: Record<string, any>;
}

export interface DiagnosticsRequest {
  forecast_data: number[][];
  ground_truth_data: number[][];
}

/** A power-law fit. `slope_standard_error` is not optional decoration: a slope quoted without
 *  its uncertainty cannot be compared against Kolmogorov -5/3 or Charney -3, which is the only
 *  reason to measure it. */
export interface SlopeAnalysis {
  slope_beta: number;
  slope_standard_error?: number;
  intercept_ln_c: number;
  r_squared: number;
  regime_interpretation: string;
  convention?: string;
  beta_energy_1d?: number;
  beta_density_2d?: number;
  n_points?: number;
  k_min?: number;
  k_max?: number;
  weighting?: string;
  assumptions?: string[];
}

export interface DiagnosticsResponse {
  spatial_metrics: {
    mean_squared_error: number;
    root_mean_squared_error: number;
    mean_absolute_error: number;
    bias: number;
    structural_similarity_index: number;
    pearson_correlation: number;
  };
  gradient_errors: {
    gradient_magnitude_mae: number;
    gradient_direction_mae_rad: number;
    gradient_direction_mae_deg: number;
  };
  // Physical units and the spectral convention, returned by the backend since T3.5.13 and
  // absent from this type until T3.5.23 - so the UI could not have displayed them. R15: the
  // same field has different exponents under E(k) and S(k), and an unlabelled slope is not an
  // interpretable number.
  grid?: {
    kind: string;
    shape: number[];
    dx: number;
    dy: number;
    length_units: string;
    variable_units?: string | null;
    description: string;
  };
  spectral_diagnostics: {
    k_units?: string;
    power_units?: string;
    convention?: string;
    convention_note?: string;
    warnings?: string[];
    wavenumbers: number[];
    forecast_psd: number[];
    ground_truth_psd: number[];
    spectral_coherence: number[];
    forecast_slope_analysis?: SlopeAnalysis;
    ground_truth_slope_analysis?: SlopeAnalysis;
  };
  wavelet_energy: {
    levels: number;
    forecast_energy: Record<string, any>;
    ground_truth_energy: Record<string, any>;
  };
}

export interface ErrorDecompositionRequest {
  forecast_data?: number[][];
  ground_truth_data?: number[][];
  forecast_series?: number[][][];
  ground_truth_series?: number[][][];
  lead_times?: number[];
  boundary_width?: number;
}

export interface ErrorDecompositionResponse {
  scale_decomposition?: {
    low_scale_rmse: number;
    mid_scale_rmse: number;
    high_scale_rmse: number;
    total_rmse: number;
  } | null;
  boundary_decomposition?: {
    distance: number;
    rmse: number;
    mean_absolute_error: number;
  }[] | null;
  lead_time_decomposition?: {
    lead_time: number;
    rmse: number;
    mean_absolute_error: number;
  }[] | null;
}

export interface PipelineStep {
  name: string;
  action: string;
  args: Record<string, any>;
}

export interface ExperimentRequest {
  name: string;
  description?: string;
  parameter_matrix: Record<string, any[]>;
  pipeline: PipelineStep[];
  metadata?: Record<string, any>;
}

export interface RunResponse {
  id: string;
  experiment_id: string;
  parameters: Record<string, any>;
  status: string;
  results?: Record<string, any>;
  error_message?: string;
  created_at: string;
  completed_at?: string | null;
}

export interface ExperimentResponse {
  id: string;
  name: string;
  description?: string | null;
  status: string;
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface ExperimentDetailResponse extends ExperimentResponse {
  runs: RunResponse[];
}

export interface LineageNodeResponse {
  id: string;
  name: string;
  type: string;
  value: Record<string, any>;
  created_at: string;
}

export interface LineageEdgeResponse {
  id: string;
  source_id: string;
  target_id: string;
  relation: string;
}

export interface LineageResponse {
  nodes: LineageNodeResponse[];
  edges: LineageEdgeResponse[];
}

export interface DiscoverRequest {
  experiment_ids?: string[];
  target_metrics?: string[];
  confidence_threshold?: number;
}

export interface HypothesisResponse {
  id: string;
  experiment_ids: string[];
  pattern_type: string;
  description: string;
  confidence: number;
  metrics_analyzed: string[];
  parameters_analyzed: string[];
  proposed_experiment_config?: Record<string, any> | null;
  created_at: string;
  // Defect D8. `confidence` alone is an effect size, and displaying it alone is what let a
  // 9-run sweep read as nine discoveries. A finding must arrive with its p-value, its
  // multiplicity-corrected q-value, the family size, and the correction's dependence
  // assumption - the UI cannot judge a result it was never sent.
  p_value?: number | null;
  q_value?: number | null;
  n_tests?: number | null;
  statistics?: Record<string, any> | null;
}

// ---------------------------------------------------------------- platform status
// These endpoints existed on the backend for several slices with no consumer, which meant
// the platform could report its own device, executor, schema revision and benchmark results
// and a researcher had no way to see any of it (T3.5.22).

export interface HealthResponse {
  status: string;
  api_version: string;
  database: string;
  database_url_scheme: string;
  datasets_available: number;
  torch_device: string;
  execution: Record<string, any>;
  database_settings: Record<string, any>;
  schema_state: {
    revision?: string | null;
    head?: string;
    pending?: string[] | null;
    up_to_date?: boolean | null;
    auto_migrate?: boolean;
    error?: string | null;
  };
}

export interface BenchmarkResponse {
  name: string;
  kind: string;
  description: string;
  gates: string[];
  is_null: boolean;
  known_answer: Record<string, any>;
  checks: BenchmarkCheckResponse[];
}

export interface DataSourceInfo {
  name: string;
  description: string;
  capabilities: Record<string, any>;
  tags: string[];
  defined_in: string;
}

// ---------------------------------------------------------------- ERA5 over Zarr (T3.5.18)

export interface ZarrStoreChunkFacts {
  megabytes_per_chunk: number | null;
  /** "live inspection", "store metadata" or "not measured" - two observations and an admission. */
  method: string;
  method_means: string;
  measured_on: string;
  shape: number[] | null;
  dims: string[] | null;
  regional_amplification: number | null;
  note: string;
}

export interface ZarrStore {
  uri: string;
  resolution_deg: number;
  cadence_hours: number;
  grid: number[];
  levels: number;
  note: string;
  /** The declared domain this store belongs to (TG10.1). A gridded store that breaks no
   *  inherited assumption is a SOURCE for a domain, never a second domain (R17). */
  domain: string;
  access: string;
  access_means: string;
  /** The store-declared vertical coordinate name; null for no vertical axis. */
  vertical_dim: string | null;
  chunks: ZarrStoreChunkFacts;
  variables_note?: string;
  extra?: {
    acquisition_defaults?: Partial<ZarrCropRequest>;
    [key: string]: unknown;
  };
}

export interface ZarrProbeRecord {
  digest: string;
  uri: string;
  /** "described", or one of the four ways a store declines. Every one is a RESULT. */
  outcome: string;
  outcome_means: string;
  /** "probe run" or "prior recorded inspection" - whether this code produced the figures. */
  evidence: string;
  evidence_means: string;
  probed_on: string;
  dimensions: Record<string, number>;
  variables: string[];
  variable_structure: Record<string, Record<string, unknown>>;
  crop: Record<string, unknown> | null;
  amplification: number | null;
  megabytes_per_chunk: number | null;
  /** Three-valued: null means nobody measured, which is never the same as false. */
  chunk_hostile: boolean | null;
  refusal_detail: string;
  note: string;
}

export interface ZarrProbeRequest {
  uri: string;
  variables?: string[];
  crop?: ZarrCropRequest | null;
  persist?: boolean;
}

export interface ZarrProbeResponse {
  probe: ZarrProbeRecord;
  saved_to: string | null;
  requested: string;
  resolved_uri: string;
}

export interface ZarrProbeLedgerResponse {
  count: number;
  /** Records this code did not produce, published so the number can only fall in the open. */
  transcribed: number;
  probe_dir: string;
  outcomes: Record<string, string>;
  evidence_kinds: Record<string, string>;
  probes: ZarrProbeRecord[];
  note: string;
}

export interface ZarrCatalogueResponse {
  stores: Record<string, ZarrStore>;
  store_domains: string[];
  access_requirements: Record<string, string>;
  network_enabled: boolean;
  network_env_var: string;
  missing_dependencies: string[];
  cache_dir: string;
  r13_minimum_crop: Record<string, number>;
  analysis_transforms: Record<string, {
    description: string;
    params: Record<string, string>;
    capabilities: Record<string, unknown>;
  }>;
  r13_legacy_note: string;
  note: string;
}

export type AcquisitionShape = 'grid_crop' | 'profile_query' | 'channel_table';

export interface AcquisitionDomainLimits {
  declaration: Record<string, any>;
  refuses: Array<Record<string, string>>;
  attribution_caveat: string;
}

export interface AcquisitionOption {
  id: string;
  name: string;
  shape: AcquisitionShape;
  available: boolean;
  access: string;
  access_means: string;
  unavailable_reason?: string | null;
  store?: ZarrStore;
  domain_limits: AcquisitionDomainLimits;
}

export interface AcquisitionDomain {
  name: string;
  description: string;
  licence: string;
  onboarding: Record<string, any>;
  domain_limits: AcquisitionDomainLimits;
  acquisitions: AcquisitionOption[];
}

export interface AcquisitionCatalogue {
  domains: AcquisitionDomain[];
  shapes: Record<AcquisitionShape, string>;
  attribution_caveat: string;
  note: string;
}

export interface ZarrAnalysisRequest {
  transform_family: 'swt' | 'dtcwt';
  wavelet: 'haar' | 'db2' | 'db3';
  boundary_mode: 'periodic' | 'reflect';
  dtcwt_level1: string;
  dtcwt_qshift: string;
}

export interface ZarrCropRequest {
  store: string;
  variables: string[];
  time_start: string;
  time_end: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  levels: number[];
  n_levels_analysis: number;
  analysis: ZarrAnalysisRequest;
}

export interface CropPlanSuggestion {
  feasible: boolean;
  target_shape: number[];
  actual_shape?: number[];
  bounds?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  reason?: string;
  cost?: {
    bytes_wanted: number;
    bytes_fetched_estimate: number;
    megabytes_fetched_estimate: number;
    amplification: number;
    chunk_hostile: boolean;
    byte_basis: string;
  };
}

export interface CropGeometryPlan {
  analysis: Record<string, any>;
  analysis_sha256: string;
  current_shape: number[];
  verdict: 'insufficient' | 'technical_only' | 'recommended';
  meets_absolute_minimum: boolean;
  meets_recommended_minimum: boolean;
  absolute_minimum: { shape: number[]; unrounded_required_side: number; basis: string };
  recommended_minimum: {
    shape: number[]; unrounded_required_side: number;
    valid_parent_side_policy: number; basis: string;
  };
  alignment_cells: number;
  support_source: string;
  levels: Array<{
    level: number;
    support_parent_px: number;
    margin_parent_px: number;
    sampling_factor: number;
    margin_native_px: number;
    valid_parent_shape: number[];
    native_shape: number[];
    valid_native_shape: number[];
  }>;
}

export interface TransformAcquisitionPlan {
  schema: string;
  plan_sha256: string;
  source_observation_sha256: string;
  geometry: CropGeometryPlan;
  suggestions: { absolute: CropPlanSuggestion; recommended: CropPlanSuggestion };
  claim_boundary: string;
}

export interface ZarrInspectResponse {
  spec: Record<string, any>;
  cached: boolean;
  structure: {
    dimensions: Record<string, number>;
    n_data_vars: number;
    variables: Record<string, {
      dims: string[];
      shape: number[];
      dtype: string;
      chunks: number[] | null;
      chunk_bytes?: number;
      chunk_megabytes?: number;
      n_chunks?: number;
    }>;
  };
  assessment: {
    selection: Record<string, number>;
    bytes_wanted: number;
    bytes_fetched_estimate: number;
    megabytes_fetched_estimate: number;
    amplification: number;
    chunk_hostile: boolean;
    threshold: number;
    byte_basis: string;
    warning: string | null;
    advice: string[];
  };
  geometry: CropGeometryPlan;
  acquisition_plan: TransformAcquisitionPlan;
  cli: string;
}

export interface ZarrCachedResponse {
  count: number;
  cache_dir: string;
  crops: Array<{
    content_key: string;
    content_hash: string;
    spec: Record<string, any>;
    shape: Record<string, number>;
    megabytes_transferred: number;
    elapsed_s: number;
    regional_forecast_readiness: {
      applicable: boolean;
      not_applicable_reason: string | null;
      vertical_dim: string;
      structurally_eligible: boolean;
      required_variables: string[];
      resolved_variables: Record<string, string>;
      missing_variables: string[];
      ambiguous_variables: string[];
      required_level_hpa: number;
      level_available: boolean;
      n_frames: number;
      minimum_frames_lower_bound: number;
      content_fingerprinted: boolean;
      split_mode: 'ratios' | 'calendar_boundaries';
      calendar_boundaries: string[] | null;
      expected_cadence_hours: number | null;
      cadence_verified: boolean;
      physical_lead_reporting_available: boolean;
      dataset_prepared: boolean;
      train_only_normalisation_verified: boolean;
      independent_era5_crosscheck: string;
      claim_boundary: string;
    };
  }>;
}

// ---------------------------------------------------------------- export (T3.5.23)

export interface ExportFieldRequest {
  field_data: number[][];
  format: string;
  coords: Record<string, number[]>;
  metadata: Record<string, any>;
  variable: string;
  units: string | null;
  name: string;
}

export interface ExportTableRequest {
  rows: Record<string, any>[];
  format: string;
  columns: string[] | null;
  metadata: Record<string, any>;
  name: string;
}

export interface ExportResult {
  blob: Blob;
  filename: string;
}

// ---------------------------------------------------------------- import (T3.5.24)

export interface ImportInspectResponse {
  filename: string;
  format: string;
  bytes: number;
  content_hash: string;
  variables: Record<string, {
    dims: string[];
    shape: number[];
    dtype?: string;
    units?: string | null;
    spatial_dims?: string[] | null;
    extra_dims: Record<string, number>;
  }>;
  default_variable: string;
  embedded_provenance: Record<string, any>;
  needs_selection: boolean;
  coords: Record<string, number[]>;
}

export interface ImportFieldResponse {
  field_data: number[][];
  coords: Record<string, number[]>;
  units: string | null;
  variable: string;
  provenance: Record<string, any>;
}

// ---------------------------------------------------------------- benchmark runs

export interface BenchmarkCheckResponse {
  stage: string;
  outcome: string;
  detail: string;
  measured?: Record<string, any> | null;
}

export interface BenchmarkSuiteResponse {
  root_seed: number;
  passed: number;
  failed: number;
  not_yet_runnable: number;
  null_failures: string[];
  benchmarks: BenchmarkResponse[];
}

// ---------------------------------------------------------------- registries

export interface RegistryEntry {
  name: string;
  description: string;
  params?: Record<string, any>;
  capabilities?: Record<string, any>;
  tags?: string[];
  defined_in?: string;
  node_type?: string;
}

// ---------------------------------------------------------------- verified forecast evaluation (T5.6g)

export interface EvaluationMetricRow {
  lead_hours: number;
  variable: string;
  unit: string;
  ensemble_mean: { rmse: number; mae: number; bias: number; mse_skill_score_vs_persistence: number | null };
  persistence: { rmse: number; mae: number; bias: number };
  crps: number;
  spread_rms: number;
  spread_skill_ratio: number | null;
  member_errors: Record<string, { rmse: number; mae: number; bias: number }>;
  initialization_count: number;
  gridpoint_count: number;
}

export interface EvaluationReport {
  schema: string;
  report_id: string;
  readiness: {
    receipt_integrity: 'VERIFIED';
    forecast_artifact: 'AUTHENTICATED_AND_VALIDATED';
    truth_source: 'DECLARED_OFFICIAL_ERA5';
    display_eligible: true;
    independent_holdout: boolean;
    scientific_skill: 'NOT_ESTABLISHED';
    reason: string;
  };
  scope: {
    split: string; split_start: string; split_end: string; evaluation_role: string;
    level_hpa: number; bounds: Record<string, any>; variables: string[];
    lead_durations_hours: number[]; initialization_count: number;
    ensemble_members: number[]; grid_shape: number[]; area_weighting: string;
  };
  metrics: EvaluationMetricRow[];
  rank_diagnostics: Array<{
    lead_hours: number; variable: string; area_weighted_frequency: number[];
    fractional_tie_counts: number[]; bin_definition: string; tie_policy: string;
    diagnostic_only: true;
  }>;
  provenance: Record<string, any>;
  claim_boundaries: string[];
}

// ---------------------------------------------------------------- findings (TG9.1/TG9.2)
//
// Phase G9's principle: the client computes and formats no scientific number. Every
// claim-bearing string below arrives already rendered by the backend and is displayed
// verbatim. `AssociationFigures` is the only thing that may carry a confidence, and it
// carries all six of R9's figures or the backend does not send it at all.

export interface DomainSummary {
  name: string;
  domain: string;
  description: string;
  // Null when limits were registered without wording. The listing unions both registries, so a
  // domain that declared what it refuses and never declared how it speaks is still visible.
  glossary_sha256: string | null;
  term_count: number;
  capabilities: Record<string, unknown>;
  defined_in: string;
  declaration: DomainLimits | null;
  onboarding: OnboardingAudit;
}

/**
 * TG8.1. Whether a domain satisfied the whole adapter recipe in one atomic call, or was
 * assembled from separate registrations and never checked as a whole. `complete: false` is a
 * reportable state rather than an error: it is how a half-onboarded domain becomes visible
 * instead of passing for a checked one.
 */
export interface OnboardingAudit {
  name: string;
  complete: boolean;
  registered: Record<string, boolean>;
  missing: string[];
  required: { requirement: string; why: string }[];
  onboarded_by?: string;
  onboarding_sha256?: string;
  geometry?: string | null;
  note?: string;
}

export interface OnboardingContract {
  schema: string;
  required: { requirement: string; why: string }[];
  onboarded: OnboardingAudit[];
  attribution_caveat: string;
}

export interface DomainGlossaryPayload {
  schema: string;
  domain: string;
  phrases: Record<string, string>;
  description: string;
}

export interface StudySummary {
  study_id: string | null;
  file: string;
  readable: boolean;
  refused_because?: string;
  revision?: number;
  bundle_sha256?: string;
  hypothesis?: string;
  rung?: string;
  blocked?: boolean;
  summary_sha256?: string;
}

// ----------------------------------------------------------- recorded review (TG11.5)

export interface ReviewArtifactRow {
  file: string;
  record: {
    schema: string;
    study_id: string;
    bundle_sha256: string;
    bundle_revision: number;
    calls: Array<Record<string, any>>;
    revision: number;
    head_sha256: string;
    record_sha256: string;
  };
  /** Backend-rendered record, including the R23 declaration. */
  rendered: string;
  outcomes: Array<{
    file: string;
    outcome: Record<string, any>;
    /** Backend-rendered outcome, including every retained dissent. */
    rendered: string;
  }>;
  cost_receipts: Array<{
    file: string;
    receipt: {
      schema: string;
      review_record_sha256: string;
      policy_sha256: string;
      call_count: number;
      input_tokens: number;
      output_tokens: number;
      cached_input_tokens: number;
      total_tokens: number;
      batch_names: string[];
      cache_hit_fraction: number;
      receipt_sha256: string;
    };
  }>;
}

export interface ReviewSurface {
  schema: 'review-surface/v1';
  study_id: string;
  bundle_file: string;
  bundle_sha256: string;
  bundle_revision: number;
  declaration: string;
  claim_boundary: string;
  reviews: ReviewArtifactRow[];
  unreadable: Array<{ file: string; refused_because: string }>;
  absence_note: string | null;
}

/**
 * R9's six figures. They travel together or not at all: the API refuses to serve a
 * `confidence` without the other five, so this interface has no optional members.
 */
export interface AssociationFigures {
  support: number;
  confidence: number;
  base_rate: number;
  lift: number;
  lift_interval: [number, number];
  surrogate_corrected_lift: number;
}

/**
 * One structural fact and its domain wording. `rendered` and `licences` are displayed
 * together and never apart: showing a claim without the bound that qualifies it is the
 * failure TG7.4 welded them into one unit to prevent.
 */
export interface TranslationUnit {
  structural_key: string;
  source_sha256: string;
  rendered: string;
  licences: string;
}

export interface TranslatedFinding {
  schema: string;
  study_id: string;
  bundle_sha256: string;
  revision: number;
  summary_sha256: string;
  domain: string;
  glossary_sha256: string;
  claimable: TranslationUnit[];
  not_claimable: TranslationUnit[];
  contradicting: TranslationUnit[];
  alternatives: TranslationUnit[];
  next_observation: TranslationUnit | null;
  figures: AssociationFigures | null;
  /** The assembled figure line. The only string a client may show for association strength. */
  figures_text: string | null;
  commentary: string[];
  rendered_text: string;
  structural_keys: string[];
  translation_sha256: string;
  /** What the selected domain refuses (TG9.3); null when no declaration is registered. */
  domain_limits: DomainLimits | null;
  /** Set when the rung asserts precedence and the selected domain does not admit one. */
  unadmitted_reading: UnadmittedReading | null;
}

// ---------------------------------------------------------------- domain limits (TG9.3)

export interface DomainRefusal {
  /** `violation:<name>` (E15) or `lag_policy:<name>` (R21). */
  basis: string;
  consequence: string;
}

export interface DomainLimits {
  declared: Record<string, unknown>;
  precedence_admissible: boolean;
  minimum_admissible_lag_frames: number | null;
  refuses: DomainRefusal[];
  /**
   * A bundle does not record which domain produced it. These limits describe the selected
   * domain and are not a check on the study, which is why the string is carried rather than
   * written into the view: the caveat and the claim travel together.
   */
  attribution_caveat: string;
}

export interface UnadmittedReading {
  rung: string;
  domain: string;
  lag_policy: string;
  note: string;
  attribution_caveat: string;
}

// ------------------------------------------------------------------ channel records (TG8.4)

/** What a clock *is*, stated without deciding what any domain may do about it. */
export interface ClockFacts {
  strictly_increasing: boolean;
  regular: boolean;
  interval_seconds_min: number | null;
  interval_seconds_max: number | null;
  /** Null for an irregular record. An irregular clock has no cadence, and a fabricated one
   *  would turn every lag in frames into a duration nobody measured. */
  cadence_seconds: number | null;
}

/** One onboarded domain's verdict on a file, with the refusal wording if it has one. */
export interface DomainAdmission {
  name: string;
  admits: boolean;
  refusals: string[];
  violations: string[];
  precedence_admissible: boolean;
}

/**
 * The inspection half of TG8.4's rule: detection may create a required declaration, it may
 * never satisfy one. `required_violations` is what a domain must *already* declare to read this
 * file — never something filled in on the reader's behalf.
 */
export interface ChannelInspection {
  source_name: string;
  content_sha256: string;
  delimiter: string;
  columns: string[];
  n_rows: number;
  candidate_time_columns: string[];
  time_column: string;
  time_units: string;
  readable: boolean;
  refused_because: string | null;
  channel_columns: string[];
  clock: ClockFacts | null;
  required_violations: string[];
  domains: DomainAdmission[];
  aggregate_note: string;
  attribution_caveat: string;
  row_cap: number;
  cell_cap: number;
}

export interface ChannelEntry {
  name: string;
  support_parent_px: number;
  is_aggregate: boolean;
  present_count: number;
  absent_count: number;
  presence: boolean[] | null;
  values: (number | null)[];
}

export interface ChannelRecord {
  source_name: string;
  content_sha256: string;
  domain: string;
  onboarding_sha256: string | null;
  n_rows: number;
  preview_rows: number;
  rows_withheld: number;
  preview_note: string;
  clock: ClockFacts;
  times_seconds: number[];
  channels: ChannelEntry[];
  domain_limits: {
    declared: Record<string, unknown>;
    precedence_admissible: boolean;
    refuses: { basis: string; consequence: string }[];
    attribution_caveat: string;
  };
  provenance: Record<string, unknown>;
}

/** Browser-only TG11.0 context. The response is a bounded preview; the original File is kept
 * so a later analysis panel can submit the admitted full record without a second file choice. */
export interface ChannelRecordSelection {
  record: ChannelRecord;
  file: File;
  timeColumn: string;
  supportParentPx: Record<string, number>;
}

// ------------------------------------------------------------------ analysis workbench (TG11.1)

export type DomainAnalysisOperation = 'association' | 'precedence' | 'domain_gate';

export interface DomainAnalysisCapabilities {
  operations: Record<DomainAnalysisOperation, string>;
  input: string;
  measure: string;
  read_only: true;
  stores: false;
  moves_rung: false;
  claim_boundary: string;
}

export interface DomainAnalysisResponse {
  schema: 'spectral.analysis.http.v1';
  operation: DomainAnalysisOperation;
  source: {
    name: string;
    content_sha256: string;
    domain: string;
    frames: number;
    channels: string[];
    cadence_seconds: number;
  };
  read_only: true;
  stored: false;
  rung_moved: false;
  result: Record<string, any>;
  claim_boundary: string;
}

// ------------------------------------------------------------------- preregistration (TG11.2)
//
// The generate/confirm split (R18). Nothing here is a result: a seal is a promise about
// ordering, and a confirmation receipt is an input to the evidence write path rather than a
// claim. The client declares a family; it never supplies a digest, a sealing time or a p-value.

export interface PartitionIdentityPayload {
  name: string;
  n_times: number;
  n_channels: number;
  channel_labels: string[];
  frames: [number, number];
  provenance: Record<string, unknown>;
  digest: string;
}

export interface HeldOutLedgerRecord {
  partition_digest: string;
  partition_name: string;
  seal_sha256: string;
  confirm_sha256: string;
  family_size: number;
  sealed_at: string;
  opened_at: string;
  study_id: string;
}

export interface PartitionDescription {
  schema: 'spectral.preregistration.http.v1';
  domain: string;
  frames: number;
  channels: string[];
  cadence_seconds: number;
  train: PartitionIdentityPayload;
  held_out: PartitionIdentityPayload;
  already_opened: boolean;
  opened_record: HeldOutLedgerRecord | null;
  read_only: true;
  stored: false;
  claim_boundary: string;
}

/** What a caller declares. The sealing time, the partition digest and the family labels are
 *  all the server's; a caller-supplied sealing time could be written after the partition was
 *  opened, which is the whole of what a seal claims. */
export interface FamilyDeclaration {
  lags: number[];
  n_surrogates?: number;
  alpha?: number;
  correction?: string;
  estimator?: string;
  bins?: number;
  seed?: number;
}

export interface SealResponse {
  schema: 'spectral.preregistration.http.v1';
  seal_sha256: string;
  sealed_at: string;
  sealed_at_source: string;
  seal: Record<string, any>;
  generate_family_size: number;
  confirm_family_size: number;
  confirm_labels: string[];
  held_out_digest: string;
  records_evidence: false;
  rung_moved: false;
  publication: string;
  claim_boundary: string;
}

export interface SealSummary {
  seal_sha256: string;
  study_id: string;
  sealed_at: string;
  confirm_family_size: number;
  generate_family_size: number;
  held_out_digest: string;
  held_out_frames: [number, number];
  spent: boolean;
  spent_by: string | null;
}

export interface SealListing {
  schema: 'spectral.preregistration.http.v1';
  seals: SealSummary[];
  n_seals: number;
  publication: string;
}

export interface ConfirmationReceipt {
  schema: 'confirmation-receipt/v1';
  stage: 'confirm';
  seal_sha256: string;
  published_sha256: string | null;
  correction_unit: number;
  correction: string;
  dependence_assumption: string;
  alpha: number;
  generate_family_size: number;
  labels: string[];
  p_values: number[];
  adjusted: number[];
  rejected: boolean[];
  rejected_labels: string[];
  n_rejected: number;
  held_out: Record<string, unknown>;
  ledger_record: HeldOutLedgerRecord;
  claim_boundary: string;
}

export interface ConfirmationResponse {
  schema: 'spectral.preregistration.http.v1';
  seal_sha256: string;
  checked_against_publication: boolean;
  receipt: ConfirmationReceipt;
  sweep_warnings: string[];
  records_evidence: false;
  rung_moved: false;
  publication: string;
  claim_boundary: string;
}

export interface PreregistrationCapabilities {
  schema: 'spectral.preregistration.http.v1';
  steps: Record<'partition' | 'seal' | 'confirm', string>;
  narrowing: string;
  once_is_per: string;
  records_evidence: false;
  moves_rung: false;
  publication: string;
  claim_boundary: string;
}

// ------------------------------------------------------- the evidence write path (TG11.3)
//
// Nothing in these types has a field for a rung, and that is the point: the ladder verdict
// arrives computed from the chain the server just wrote, and no request shape here can carry
// one back (R22). `temporal_precedence` is absent from the hand-written payload type for the
// same reason - it is the one payload key the ladder reads.

export type EvidenceCategory =
  | 'observations' | 'effect_sizes' | 'uncertainty' | 'null_results' | 'replication_results'
  | 'holdout_performance' | 'provenance' | 'confounders' | 'contradictory_evidence'
  | 'failure_states';

export type EvidenceStatus = 'PASS' | 'FAIL' | 'INVALID' | 'INCONCLUSIVE' | 'NOT_APPLICABLE';

export interface EvidenceGate {
  name: string;
  rung: string;
  satisfied: boolean;
  requirement: string;
}

export interface ClaimLadderVerdict {
  schema: string;
  study_id: string;
  hypothesis_sha256: string;
  bundle_sha256: string;
  revision: number;
  rung: string;
  rung_index: number;
  unblocked_rung: string;
  gates: EvidenceGate[];
  blocking_entries: number[];
}

export interface EvidenceEntryRecord {
  sequence: number;
  category: EvidenceCategory;
  label: string;
  status: EvidenceStatus;
  summary: string;
  recorded_at: string;
  payload: Record<string, unknown>;
  source_sha256s: string[];
  previous_sha256: string;
  entry_sha256: string;
}

export interface EvidenceState {
  schema: 'spectral.evidence.http.v1';
  study_id: string;
  revision: number;
  head_sha256: string;
  bundle_sha256: string;
  hypothesis_sha256: string;
  ladder: ClaimLadderVerdict;
  assessment_sha256: string;
  blocked: boolean;
  unsatisfied_gates: string[];
  rung_source: string;
  entry?: EvidenceEntryRecord;
  entries?: EvidenceEntryRecord[];
  next_sequence?: number;
  hypothesis?: Record<string, unknown>;
  wording_note?: string | null;
  computed_here?: boolean;
  verdict_source?: string;
}

export interface EvidenceCapabilities {
  schema: 'spectral.evidence.http.v1';
  categories: EvidenceCategory[];
  statuses: EvidenceStatus[];
  append_only: true;
  compare_and_swap: string;
  rung_is_computed: true;
  accepts_a_rung: false;
  computed_not_accepted: Record<string, string>;
  commentary: string;
  claim_boundary: string;
}

/** What a hand-written append may say. No rung, no verdict, no recorded_at: the chronology
 *  is the server's, so an entry cannot be back-dated into an order it did not happen in. */
export interface EvidenceAppend {
  expected_head_sha256: string;
  category: EvidenceCategory;
  label: string;
  status: EvidenceStatus;
  summary: string;
  payload: Record<string, unknown>;
  source_sha256s?: string[];
}

// ----------------------------------------------------------- structure mining (TG11.4)
//
// Nothing in these types has a field for a feature, a coordinate or a graph, and that is
// the point of the surface: a motif is a configuration of extracted features, so a client
// that could send one could draw the shape it wanted the programme to confirm. Scenes are
// addressed by the digest of the field they were extracted from. The tolerance is a digest
// too, for the same reason: it decides which configurations count as repeats.

export interface MiningMatcher {
  name: string;
  description: string;
  declared_invariance: string[];
  capabilities: Record<string, unknown>;
}

export interface MiningCapabilities {
  schema: 'spectral.mining.http.v1';
  sizes: Record<string, string>;
  matchers: MiningMatcher[];
  relations: { name: string; description: string; capabilities: Record<string, unknown> }[];
  extractors: { name: string; description: string; capabilities: Record<string, unknown> }[];
  transforms: string[];
  minable_domains: string[];
  input: string;
  tolerance: string;
  stores: string[];
  moves_rung: false;
  claim_boundary: string;
}

export interface AdmittedFrame {
  frame: number;
  scene: string;
  n_features: number;
  rejected: Record<string, number>;
  threshold: number;
}

export interface AdmittedRecord {
  record_id: string;
  declaration: Record<string, unknown>;
  n_frames: number;
  frames: AdmittedFrame[];
  feature_counts: number[];
  minable: boolean;
  why_not_minable: string | null;
  note: string;
  claim_boundary: string;
}

export interface FamilyPrice {
  generate: {
    family_size: number;
    n_surrogates: number;
    surrogates_required: number;
    p_value_floor: number;
    affordable: boolean;
    correction: string;
    alpha: number;
    claim_boundary: string;
  };
  confirmatory_ceiling: {
    max_affordable_family: number;
    surrogates_for_one_member: number;
    reading: string;
  };
  split_is_not_optional: boolean;
  claim_boundary: string;
}

export interface ToleranceReceipt {
  tolerance_sha256: string;
  receipt: {
    record_id: string;
    frames: number[];
    matcher: string;
    declared_as_replicates_of: string;
    calibrated_at: string;
    tolerance: {
      value: number;
      n_replicates: number;
      components: number;
      worst_component: string;
      basis: string;
    };
  };
  note: string;
  claim_boundary: string;
}

export interface MotifCandidateSummary {
  label: string;
  support: number;
  n_occurrences: number;
  scenes: string[];
  specificity: number;
}

export interface MiningGeneration {
  record_id: string;
  split: Record<string, number[]>;
  train: Record<string, unknown>;
  held_out_identity: Record<string, unknown>;
  held_out_spent: Record<string, unknown> | null;
  mining: {
    scenes: string[];
    size: number;
    matcher: string;
    tolerance: number;
    generate_family_size: number;
    generate_sha256: string;
    n_examined: number;
    n_candidates: number;
    intransitive_pairs: number;
    affordable_in_one_stage: boolean;
    top: MotifCandidateSummary[];
    claim_boundary: string;
  };
  generation_report: Record<string, unknown>;
  chosen: MotifCandidateSummary[];
  confirmatory_ceiling: number;
  next: string;
  claim_boundary: string;
}

export interface MiningSeal {
  seal_sha256: string;
  seal: Record<string, unknown>;
  frozen_labels: string[];
  frozen: MotifCandidateSummary[];
  publication_note: string;
  sealed_settings: Record<string, unknown>;
  next: string;
  claim_boundary: string;
}

export interface MiningConfirmation {
  receipt: {
    seal_sha256: string;
    correction_unit: number;
    correction: string;
    alpha: number;
    labels: string[];
    p_values: number[];
    adjusted: number[];
    rejected: boolean[];
    rejected_labels: string[];
    n_rejected: number;
    supports: number[];
    n_scenes: number;
    motifs: Record<string, unknown>[];
    vacuous: string[];
    claim_boundary: string;
  };
  sealed_settings: Record<string, unknown>;
  published_sha256: string | null;
  publication_note: string;
  vacuous: string[];
  claim_boundary: string;
}

export interface PublishedMotif {
  motif_sha256: string;
  definition_sha256: string;
  motif: Record<string, unknown>;
  origin_licence: string;
  publication_note: string;
  claim_boundary: string;
}

export interface TransferReceipt {
  receipt: {
    motif_sha256: string;
    source_domain: string;
    target_partition_sha256: string;
    n_scenes: number;
    n_examined: number;
    n_matches: number;
    support: number;
    matched_scenes: string[];
    claim_boundary: string;
  };
  target_record_id: string;
  note: string;
  claim_boundary: string;
}

export interface InvarianceAudit {
  record_id: string;
  alpha: number;
  reports: Record<string, {
    matcher: string;
    declared: string[];
    measured: string[];
    overclaimed: string[];
    understated: string[];
    honest: boolean;
    vacuous: string[];
    tests: Record<string, Record<string, unknown>>;
  }>;
  honest: string[];
  overclaimed: Record<string, string[]>;
  declared_transforms_are_the_callers: string;
  claim_boundary: string;
}

/** What a mining run declares. There is no `tolerance` here, only its digest, and no
 *  `features`: both are things the server measured rather than things a client may state. */
export interface MiningRunRequest {
  record_id: string;
  tolerance_sha256: string;
  size?: number;
  matcher?: string;
  n_surrogates?: number;
  alpha?: number;
  correction?: string;
  seed?: number;
  train_ratio?: number;
  embargo_frames?: number;
  study_id?: string;
}

// ---------------------------------------------------- the cross-domain record (TG11.4b)
//
// A lag here is a duration, not a frame count: two native clocks have two frame sizes and a
// family declared in either one is a family the other domain cannot read. Every request below
// therefore carries `lag_seconds` and none carries `lags`.

/** One side's reading. `channels` is required and has no default: R19 says a source's
 *  semantics and units may never be dropped, and a channel table carries neither. */
export interface CrossDomainSource {
  domain: string;
  time_column: string;
  time_units?: string;
  delimiter?: string;
  channels: Record<string, { semantics: string; units: string }>;
  aggregation_window_seconds?: number | null;
}

export interface CrossDomainCapabilities {
  schema: string;
  record_schema: string;
  steps: Record<string, string>;
  interpolation: string;
  alignment: string;
  minimum_common_observations: number;
  lags_declared_in: string;
  measure: string;
  requires_per_channel: string[];
  records_evidence: boolean;
  moves_rung: boolean;
  publication: string;
  claim_boundary: string;
}

export interface CrossDomainAlignment {
  schema: string;
  name: string;
  alignment: string;
  interpolation: string;
  clock_sha256: string;
  clock_start_seconds: number;
  clock_stop_seconds: number;
  common_cadence_seconds: number;
  n_common_observations: number;
  physical_lag_floor_seconds: number;
  retained_native_observations: Record<string, number>;
  discarded_native_observations: Record<string, number>;
  domains: Record<string, Record<string, unknown>>;
  channels: Record<string, { domain: string; native_label: string; semantics: string; units: string }>;
  statistic_units: string;
}

export interface CrossDomainAligned {
  schema: string;
  alignment: CrossDomainAlignment;
  cross_domain_pairs: string[];
  n_frames: number;
  read_only: boolean;
  stored: boolean;
  claim_boundary: string;
}

export interface CrossDomainFamily {
  lag_seconds: number[];
  lag_frames: number[];
  common_cadence_seconds: number;
  cross_domain_pairs: string[];
  n_pairs: number;
  family_size: number;
  n_surrogates: number;
  affordable_member_count: number;
  affordable: boolean;
  required_surrogates: number;
  reading: string;
}

export interface CrossDomainPrice {
  schema: string;
  alignment: CrossDomainAlignment;
  family: CrossDomainFamily;
  read_only: boolean;
  stored: boolean;
  claim_boundary: string;
}

export interface CrossDomainPartition {
  schema: string;
  alignment: CrossDomainAlignment;
  train: { frames: number[]; n_frames: number; digest: string };
  held_out: { frames: number[]; n_frames: number; digest: string };
  embargo_frames: number;
  recommended_embargo_frames: number;
  already_opened: boolean;
  opened_record: Record<string, unknown> | null;
  read_only: boolean;
  stored: boolean;
  claim_boundary: string;
}

export interface CrossDomainCandidate {
  label: string;
  driver: string;
  driven: string;
  lag: number;
  correlation: number;
  n_pairs: number;
  n_effective: number;
  p_naive: number;
  p_effective: number;
  claim_boundary: string;
}

export interface CrossDomainGeneration {
  schema: string;
  alignment: CrossDomainAlignment;
  family: Record<string, unknown>;
  n_examined: number;
  affordable_here: boolean;
  candidates: CrossDomainCandidate[];
  generation: Record<string, unknown>;
  read_only: boolean;
  stored: boolean;
  claim_boundary: string;
}

export interface CrossDomainSeal {
  schema: string;
  seal_sha256: string;
  sealed_at: string;
  sealed_at_source: string;
  seal: Record<string, unknown>;
  generate_family_size: number;
  confirm_family_size: number;
  confirm_labels: string[];
  frozen: CrossDomainCandidate[];
  held_out_digest: string;
  records_evidence: boolean;
  rung_moved: boolean;
  publication: string;
  claim_boundary: string;
}

export interface CrossDomainConfirmation {
  schema: string;
  seal_sha256: string;
  checked_against_publication: boolean;
  alignment: CrossDomainAlignment;
  receipt: Record<string, any>;
  confirmed_labels: string[];
  records_evidence: boolean;
  rung_moved: boolean;
  publication: string;
  claim_boundary: string;
}
