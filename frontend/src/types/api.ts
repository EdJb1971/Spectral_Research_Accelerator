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

export interface ZarrCatalogueResponse {
  stores: Record<string, {
    uri: string;
    resolution_deg: number;
    cadence_hours: number;
    grid: number[];
    levels: number;
    note: string;
  }>;
  network_enabled: boolean;
  network_env_var: string;
  missing_dependencies: string[];
  cache_dir: string;
  r13_minimum_crop: Record<string, number>;
  note: string;
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
  geometry: Record<string, any>;
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
  glossary_sha256: string;
  term_count: number;
  capabilities: Record<string, unknown>;
  defined_in: string;
  declaration: DomainLimits | null;
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
