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

export type AcquisitionShape = 'grid_crop' | 'profile_query' | 'lightcurve_query' | 'channel_table';

export interface AcquisitionDomainLimits {
  declaration: Record<string, any>;
  refuses: Array<Record<string, string>>;
  attribution_caveat: string;
}

export interface AcquisitionOption {
  id: string;
  name: string;
  label?: string;
  provider?: string | null;
  product_family?: string | null;
  shape: AcquisitionShape;
  available: boolean;
  access: string;
  access_means: string;
  unavailable_reason?: string | null;
  store?: ZarrStore;
  profile_source?: ProfileSource;
  lightcurve_source?: LightCurveSource;
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
  operational_routes: Array<{
    id: string; domain: string; label: string; provider: string; product_family: string;
    ui_status: string; execution: string; configuration: string[]; reason: string;
  }>;
  violation_coverage: Record<string, Array<{ domain: string; shape: AcquisitionShape; path: string }>>;
  attribution_caveat: string;
  note: string;
}

export interface CDSPlanRequest {
  variables: string[];
  date_start: string;
  date_end: string;
  hours_utc: number[];
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  pressure_levels: number[];
  grid_degrees: number;
  n_levels_analysis: number;
}

export interface CDSCapabilities {
  schema: string;
  dataset: string;
  variables: Array<{ id: string; cds_name: string }>;
  pressure_levels: number[];
  defaults: CDSPlanRequest;
  network_env_var: string;
  network_enabled: boolean;
  planner_network_used: false;
  execution_status: 'AVAILABLE' | 'NETWORK_DISABLED';
  workflow: string[];
  claim_boundary: string;
}

export interface CDSPlan {
  schema: string;
  request: CDSPlanRequest & { request_sha256: string };
  request_sha256: string;
  monthly_shards: Array<{
    year: number; month: number; days: number[]; request_sha256: string; filename: string;
    request: Record<string, unknown>;
  }>;
  storage_estimate: {
    basis: string; compression_credit_assumed: boolean; frames: number;
    latitude_points_upper_bound: number; longitude_points_upper_bound: number;
    levels: number; variables: number; raw_value_bytes: number;
    artifact_bytes_upper_bound: number; shards: number;
  };
  analysis_geometry: { status: string; assessment: Record<string, any> };
  network_used: false;
  execution_status: 'READY_TO_SUBMIT' | 'NETWORK_DISABLED';
  submission_confirmation: {
    confirm_request_sha256: string; confirm_network_access: true; statement: string;
  };
  next_action: string;
  claim_boundary: string;
}

export interface CDSStoragePreflight {
  status: 'READY';
  completed_shards: number;
  remaining_shards: number;
  volumes: Array<{
    volume: string; roles: string[]; working_bytes_required: number; free_bytes: number;
    reserve_bytes: number; total_free_required: number; passes: boolean;
  }>;
}

export type CDSJobState = 'QUEUED' | 'RUNNING' | 'CANCELLING' | 'INTERRUPTED' |
  'CANCELLED' | 'FAILED' | 'COMPLETE';

export interface CDSAcquisitionRecord {
  schema: 'cds-acquisition-record/v1';
  job_id: string; request_sha256: string; request: CDSPlanRequest & { request_sha256: string };
  completed_at: string; completed_shards: number; total_bytes: number;
  shards: Array<{ filename: string; sha256: string; bytes: number; request_sha256: string }>;
  record_sha256: string; claim_boundary: string;
}

export interface CDSJob {
  schema: 'cds-acquisition-job/v1';
  job_id: string; request_sha256: string; request: CDSPlanRequest & { request_sha256: string };
  state: CDSJobState; created_at: string; updated_at: string; started_at: string | null;
  completed_at: string | null; total_shards: number; completed_shards: number;
  current_shard: string | null; downloaded_shards_this_attempt: number;
  resumed_shards_this_attempt: number; attempts: number; message: string;
  error: { type: string; detail: string } | null; storage_preflight: CDSStoragePreflight;
  acquisition_record: CDSAcquisitionRecord | null; existing_job?: boolean;
  storage: { ownership: 'SERVER_MANAGED'; namespace: string; job_id: string;
    client_path_accepted: false };
  progress: { completed_shards: number; total_shards: number; fraction: number;
    current_shard: string | null };
  resumable: boolean; cancellation_boundary: string; claim_boundary: string;
}

export interface CDSJobList {
  schema: 'cds-acquisition-jobs/v1'; jobs: CDSJob[]; network_enabled: boolean;
  claim_boundary: string;
}

export interface ZarrAnalysisRequest {
  transform_family: 'swt' | 'dtcwt';
  wavelet: 'haar' | 'db2' | 'db3';
  boundary_mode: 'periodic' | 'reflect';
  dtcwt_level1: string;
  dtcwt_qshift: string;
}

export interface LightCurveSource {
  name: string;
  domain: string;
  access: string;
  archive: string;
  collection: string;
  pipeline: string;
  product_subgroup: string;
  network_env_var: string;
  metadata_preflight: boolean;
  maximum_products: number;
  maximum_download_bytes: number;
}

export interface LightCurveCapabilities {
  schema: string;
  domain: Record<string, unknown>;
  refusals: Array<Record<string, unknown>>;
  source: LightCurveSource;
  workflow: string[];
  claim_boundary: string;
}

export interface LightCurveSpecRequest {
  target_id: string;
  sectors: number[];
  flux_column: 'PDCSAP_FLUX' | 'SAP_FLUX';
  quality_policy: 'quality_zero' | 'retain_all';
  max_products: number;
  max_download_bytes: number;
}

export interface LightCurvePlan {
  schema: string;
  request_sha256: string;
  target: { tic_id: string; ra_deg: number; dec_deg: number; frame: string };
  candidate_products: number;
  predicted_download_bytes: number;
  within_product_cap: boolean;
  within_byte_cap: boolean;
  products: Array<{ filename: string; sector: number; size_bytes: number; data_uri: string }>;
  metadata_only: true;
  claim_boundary: string;
}

export interface LightCurveAcquisitionResponse {
  schema: string;
  collection: { collection_sha256: string; request_sha256: string; n_samples: number;
    n_products: number; sectors: number[]; quality_admitted: number; quality_flagged: number;
    time_scale: string; flux_column: string };
  publication: { path: string; collection_sha256: string; bytes: number; publication: string };
  analysis_readiness: Record<string, string>;
  capability_profile: DatasetCapabilityProfile;
  claim_boundary: string;
}

export type SampleRole = 'sample_id' | 'target' | 'nuisance' | 'feature' | 'group' |
  'ordering' | 'ignore';

export interface FileProbe {
  schema: 'spectral.file-probe.v1';
  filename: string;
  content_sha256: string;
  format: string;
  delimiter: string;
  n_rows: number;
  n_columns: number;
  semantic_inference: false;
  columns: Array<{ name: string; storage_type: 'numeric' | 'text' | 'boolean';
    missing_count: number; distinct_count: number; strictly_increasing: boolean;
    minimum: number | null; maximum: number | null }>;
  obligations: string[];
  claim_boundary: string;
}

export interface SampleTableDeclaration {
  roles: Record<string, SampleRole>;
  sample_relationship: 'independent' | 'grouped' | 'ordered';
  units: Record<string, string>;
}

export type CapabilityStatus = 'available' | 'unavailable' | 'needs_declaration' |
  'needs_configuration' | 'insufficient_support';

export interface DatasetOperationDecision {
  name: string;
  description: string;
  status: CapabilityStatus;
  available: boolean;
  reason_code: string;
  reason: string;
  requirements: Array<{ fact: string; label: string; satisfied: boolean | null }>;
}

export interface DatasetCapabilityProfile {
  schema: 'spectral.dataset-capability-profile.v1';
  profile_sha256: string;
  kind: string;
  phase: 'probed' | 'declared' | 'planned' | 'acquired' | 'admitted';
  identity: string;
  domain: string | null;
  facts: Record<string, boolean | null>;
  capabilities: Array<{ name: string; label: string; value: boolean | null }>;
  operations: Record<string, DatasetOperationDecision>;
  basis: Record<string, unknown>;
  claim_boundary: string;
}

export interface RepresentationAuditPlan {
  schema: 'spectral.representation-audit-plan.v1';
  content_sha256: string;
  declaration: SampleTableDeclaration;
  representations: string[];
  pca_components: number;
  bins: number;
  permutations: number;
  generate_fraction: number;
  seed: number;
  alpha: number;
  correction: string;
  candidates: string[];
  n_tests_per_partition: number;
  plan_sha256: string;
  probe: FileProbe;
  admissibility: Record<string, boolean>;
  claim_boundary: string;
}

export interface RepresentationAuditResult {
  schema: 'spectral.representation-audit.v1';
  plan_sha256: string;
  target: string;
  declared_nuisances: string[];
  partitions: { generate_n: number; confirm_n: number; split_seed: number;
    held_out_opened_once: boolean };
  family: { candidates: string[]; n_tests_per_partition: number; correction: string;
    permutations: number; minimum_p_value: number };
  structure: { feature_count: number; effective_dimension: number;
    absolute_correlation: number[][]; pca_singular_values_generate: number[] };
  candidates: Array<{ candidate: string; generate_mi_nats: number; generate_q_value: number;
    confirm_mi_nats: number; confirm_q_value: number; survives_both: boolean;
    nuisance_stability: Array<Record<string, any>> }>;
  stored: false;
  rung_moved: false;
  claim_boundary: string;
}

export interface ProfileSource {
  name: string;
  domain: string;
  shape: 'profile_query';
  access: string;
  description: string;
  licence: string;
  variables: string[];
  defaults: Partial<ProfileSpecRequest>;
}

export interface ProfileSpecRequest {
  source: string;
  time_start: string;
  time_end: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  pressure_min_dbar: number;
  pressure_max_dbar: number;
  variables: string[];
  max_profiles: number;
}

export interface ProfileReductionRequest {
  name: 'per_float_at_pressure' | 'depth_bin_mean';
  configuration: Record<string, any>;
}

export interface ProfileAcquisitionRequest {
  spec: ProfileSpecRequest;
  reduction: ProfileReductionRequest;
}

export interface ProfileCapabilities {
  schema: string;
  sources: ProfileSource[];
  reductions: RegistryEntry[];
  network_enabled: boolean;
  network_env_var: string;
  source_doi: string;
  claim_boundary: string;
}

export interface ProfileQueryPlan {
  schema: string;
  request_sha256: string;
  source: string;
  source_doi: string;
  candidate_profiles: number;
  candidate_platforms: number;
  within_profile_cap: boolean;
  max_profiles: number;
  profiles: Array<{ profile_id: string; platform_id: string; time: string;
                    latitude: number; longitude: number }>;
  profiles_withheld: number;
  claim_boundary: string;
  requested_reduction: Record<string, any>;
}

export interface ProfileAcquisitionResponse {
  schema: string;
  collection: {
    collection_sha256: string;
    request_sha256: string;
    n_profiles: number;
    n_platforms: number;
    variables: string[];
    valid_value_count: Record<string, number>;
    pressure_range_dbar: number[];
    qc_policy: Record<string, any>;
  };
  publication: { path: string; collection_sha256: string; bytes: number; publication: string };
  reduction: {
    reduction: Record<string, any>;
    derived_declaration: Record<string, any>;
    n_frames: number;
    n_channels: number;
    channels: string[];
    channel_records: Array<Record<string, any>>;
    clock: { strictly_increasing: boolean; regular: boolean; cadence_seconds: number | null };
    content_sha256: string;
    claim_boundary: string;
  };
  preview: {
    measure: string;
    n_rows: number;
    preview_rows: number;
    rows_withheld: number;
    times_seconds: number[];
    channels: Array<{ name: string; usable: boolean; present_count: number;
                      absent_count: number; presence: boolean[] | null;
                      values: Array<number | null> }>;
  };
  analysis_readiness: {
    frame_lag_admissible: boolean;
    decision: string;
    basis: string[];
    reason: string;
    contiguous_candidate?: Record<string, any>;
  };
  capability_profile: DatasetCapabilityProfile;
  source_doi: string;
  claim_boundary: string;
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
    // T4C.5i step 6: the threshold is a heuristic, and the dyadic size is reported beside it
    // as a convention that is never refused on.
    heuristic?: boolean;
    dyadic_operational_shape?: number[];
    dyadic_operational_basis?: string;
    limitation?: string;
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
  capability_profile: DatasetCapabilityProfile;
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
  capability_profile: DatasetCapabilityProfile;
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

// ------------------------------------------ configurable experiment manifest (TG17.1)
export interface CrossDomainExperimentManifest {
  schema_id: 'cross-domain-experiment/v1';
  study_id: string;
  title: string;
  mode: 'calendar_aligned' | 'scale_shape_aligned';
  windows: { name: string; start_utc: string; end_utc: string; stride_seconds: number }[];
  observations: {
    domain: string; label: string; role: string; measure: string; semantics: string; units: string;
    acquisition: { source_id: string; source_version: string; identity: Record<string, any>; parameters: Record<string, any> };
    adapter: { adapter_id: string; adapter_version: string; parameters: Record<string, any> };
  }[];
  coverage_policy: { requirement: 'complete_required' | 'partial_permitted'; minimum_fraction: number };
  alignment: AlignmentPolicy;
  scale_normalization: Record<string, any> | null;
  family: FamilyDefinition;
  confirmation: ConfirmationPolicy;
  nulls: { name: string; method: string; replications: number; parameters: Record<string, any> }[];
  correction: 'benjamini_yekutieli' | 'holm' | 'bonferroni';
  alpha: number;
  seeds: Record<string, number>;
  resource_caps: { maximum_family_members: number; maximum_planned_bytes: number; maximum_runtime_seconds: number };
  notes: Record<string, any>;
}

/** Every axis of the declared search (TG17.5). The axes a study is free to vary are the axes
 *  it must declare: a family priced without one is short by exactly the factor nobody wrote
 *  down. */
export interface FamilyDefinition {
  channels: string[];
  scales: number[];
  relationships: string[];
  domain_arities: number[];
  lags_seconds: number[];
  representations: string[];
  motifs: string[];
  maximum_members: number;
}

/** Which stage the manifest declares, and therefore what R18 corrects over (TG17.5). */
export interface ConfirmationPolicy {
  stage: 'confirmatory_only' | 'generate_then_confirm';
  held_out_partition: string | null;
  confirmatory_members: number | null;
}

export interface FamilyExpansionAxis {
  axis: string;
  declared_values: number;
  examples: string[];
  contributes: string;
}

/** What one more value on an axis would cost, priced before the freeze rather than discovered
 *  after acquisition. */
export interface FamilyExpansionCost {
  axis: string;
  adds: string;
  family_size_before: number;
  family_size_after: number;
  members_added: number;
  surrogates_required_before: number;
  surrogates_required_after: number;
  declared_surrogates: number;
  affordable_after: boolean;
}

export interface FamilyCorrectionPlan {
  stage: 'confirmatory_only' | 'generate_then_confirm';
  declared_search_members: number;
  correction_unit_members: number;
  held_out_partition: string | null;
  n_surrogates: number;
  surrogates_required: number;
  p_value_floor: number;
  affordable: boolean;
  warning: string | null;
  generate_stage_makes_claims: boolean | null;
  note: string;
}

export interface PrecedenceAvailability {
  schema: string;
  declared_family_size: number;
  precedence_relationships_declared: string[];
  domains_without_precedence_policy: string[];
  unavailable_precedence_members: number;
  association_members_unaffected: number;
  family_size_unchanged: boolean;
  entire_family_unavailable: boolean;
  basis: string;
  note: string;
}

export interface FamilyExpansion {
  schema: string;
  study_id: string;
  mode: string;
  axes: FamilyExpansionAxis[];
  in_human_terms: string;
  family_size: number;
  specification_sha256: string;
  declared_surrogates: number;
  declared_surrogates_basis: string;
  account: Record<string, any>;
  correction: FamilyCorrectionPlan;
  largest_affordable_family: number;
  resource_requirement: { surrogate_evaluations_declared: number;
    surrogate_evaluations_required: number; note: string };
  expansion_cost: FamilyExpansionCost[];
  screen_and_confirm: { screen_family_size: number; complete_family_size: number;
    correction_unit: number; screen_sha256: string; complete_sha256: string; rule: string };
  precedence: PrecedenceAvailability;
  claim_boundary: string;
}

/** One declared surrogate construction. `preserves` is the list of features that would
 *  otherwise manufacture the structure under test, so a family that preserves fewer of them is
 *  a weaker null and says so. */
export interface NullFamilyDescription {
  name: string;
  modes: string[];
  operates_on: string;
  preserves: string[];
  destroys: string[];
  parameters: string[];
  admissible: boolean;
  inadmissible_reason: string | null;
  definition: string;
  claim_boundary: string;
  admitted_by: string[];
  usable_across_all_registered_domains: boolean;
}

export interface NullFamilyList {
  schema: string;
  families: NullFamilyDescription[];
  modes: string[];
  note: string;
  claim_boundary: string;
}

export interface ExperimentManifestEnvelope {
  schema: string;
  recipe_id?: string;
  manifest_sha256: string;
  run_identity: string;
  canonical_manifest: CrossDomainExperimentManifest;
  immutable: boolean;
}

export interface ExperimentPreflight {
  schema: string;
  manifest_sha256: string;
  status: 'READY' | 'PARTIAL' | 'REFUSED';
  metadata_only: boolean;
  network_used: boolean;
  measurement_values_opened: boolean;
  family: FamilyExpansion & { declared_members: number; maximum_members: number;
    windows_are_one_family: boolean };
  alignment: PreflightAlignment;
  coverage: { domain: string; label?: string; source_id: string; measure?: string; status: string;
    reason: string; support_kind?: string; native_cadence_seconds?: number | null; access?: string;
    opens_measurement_values?: boolean; windows?: Record<string, any>[] }[];
  refusals: { domain: string | null; reason: string }[];
  claim_boundary: string;
}

// ---------------------------------------------------------- registered adapters (TG17.3)

/** One control a domain adapter declares. The Composer renders these and nothing else, so a
 *  domain cannot require a widget only it understands, and a fifth adapter reaches the form
 *  without the form learning its name. */
export interface AdapterControlField {
  name: string;
  label: string;
  kind: 'text' | 'integer' | 'number' | 'boolean' | 'enum' | 'utc_instant' | 'content_record';
  help: string;
  required: boolean;
  default: any;
  choices: any[];
  minimum: number | null;
  maximum: number | null;
  units: string | null;
}

export interface DomainExperimentAdapterDescription {
  schema: 'domain-experiment-adapter/v1';
  adapter_id: string;
  adapter_version: string;
  domain: string;
  definition_sha256: string;
  controls: { schema: string; fields: AdapterControlField[] };
  declaration: Record<string, any>;
  implements: string[];
  onboarding_cost: Record<string, any>;
}

export interface AdapterConformanceReport {
  schema: 'domain-adapter-conformance/v1';
  adapter_id: string;
  domain: string;
  definition_sha256: string;
  conformant: boolean;
  counts: Record<string, number>;
  checks: { check: string; status: 'PASS' | 'FAIL' | 'NOT_APPLICABLE' | 'NOT_PROBED';
    detail: string; evidence: Record<string, any> }[];
  claim_boundary: string;
  record_kind: 'deterministic_known_answer_not_acquired_data';
}

// ------------------------------------------------- clock, support and coverage (TG17.4)

/** How support is compared, frozen in the manifest before any value is opened. The default
 *  kernel transforms nothing; every other one widens, snaps or carries support and must be
 *  admitted by every participating adapter. */
export interface AlignmentPolicy {
  kernel: string;
  parameters: Record<string, number>;
  minimum_overlap_seconds: number;
  minimum_effective_samples: number;
}

export interface AlignmentKernelDescription {
  name: string;
  summary: string;
  required_parameters: string[];
  manufactures_simultaneity: boolean;
  invents_values: boolean;
  refused_over_violations: string[];
  admitted_by: string[];
  usable_across_all_registered_domains: boolean;
}

export interface AlignmentKernelList {
  schema: 'experiment-composer-alignment-kernels/v1';
  kernels: AlignmentKernelDescription[];
  default: string;
  note: string;
  claim_boundary: string;
}

/** One record's coverage of one window. `raw_row_count` is shown next to the numbers that are
 *  actually evidence and is used by nothing: changing row density alone cannot change any other
 *  field here. */
export interface SupportCoverage {
  schema: string;
  label: string;
  domain: string;
  window_start_seconds: number;
  window_end_seconds: number;
  window_seconds: number;
  occupied_seconds: number;
  covered_fraction: number;
  intervals: number[][];
  gaps: number[][];
  gap_count: number;
  largest_gap_seconds: number;
  native_scale_seconds: number;
  raw_row_count: number;
  valid_row_count: number;
  rows_are_not_evidence: string;
  support_is_stationary: boolean;
  support_duration_min_seconds: number | null;
  support_duration_max_seconds: number | null;
  kernel: string;
  kernel_parameters: Record<string, number>;
  manufactured_seconds: number;
}

export interface PairwiseOverlapRow {
  left: string;
  right: string;
  kernel: string;
  kernel_parameters: Record<string, number>;
  overlap_seconds: number;
  exact_overlap_seconds: number;
  manufactured_overlap_seconds: number;
  lost_overlap_seconds: number;
  overlap_intervals: number[][];
  left_occupied_seconds: number;
  right_occupied_seconds: number;
  overlap_fraction_of_shorter: number;
  governing_scale_seconds: number;
  effective_sample_size: number;
  effective_sample_size_basis: string;
  row_counts_not_used: { left: number; right: number };
  status: 'COMPARABLE' | 'REFUSED';
  refusals: string[];
}

export interface AlignmentReport {
  schema: 'structural-alignment/v1';
  mode: 'calendar_aligned' | 'scale_shape_aligned';
  mode_forbids: string;
  admitted_relationships: string[];
  kernel: AlignmentKernelDescription & { parameters?: Record<string, number>; freeze_sha256?: string };
  coverage: SupportCoverage[];
  pairs: PairwiseOverlapRow[];
  refusals: { pair: string[] | null; relationship: string | null; reason: string }[];
  status: 'COMPARABLE' | 'REFUSED';
  row_indices_were_not_compared: true;
  claim_boundary: string;
  manifest_sha256: string;
  record_binding: 'benchmark_known_answer';
  record_kind: 'deterministic_known_answer_not_acquired_data';
  window_seconds: number;
}

/** What the declared windows and clocks imply about shared support, from metadata only. A pair
 *  whose coverage the catalogue cannot establish is reported as bounded by the window rather
 *  than given a number that would later turn out to have been a guess. */
export interface PreflightAlignment {
  schema: 'experiment-preflight-alignment/v1';
  mode: 'calendar_aligned' | 'scale_shape_aligned';
  mode_forbids?: string;
  admitted_relationships?: string[];
  declared_relationships?: string[];
  kernel: (AlignmentKernelDescription & { parameters?: Record<string, number>; freeze_sha256?: string }) | null;
  kernel_refused?: string;
  minimum_overlap_seconds?: number;
  minimum_effective_samples?: number;
  family?: { modes: string[]; multiplier: number; correction_scope: string; why: string };
  row_indices_were_not_compared?: boolean;
  measurement_values_opened?: boolean;
  windows: {
    name: string;
    window_seconds: number;
    clock: { elapsed_seconds: number; nominal_seconds: number; discrepancy_seconds: number; clock_is_uniform: boolean };
    clock_note: string;
    pairs: Record<string, any>[];
  }[];
}

export interface StructuralTrajectoryPreview {
  schema: 'structural-trajectory-preview/v1';
  kind: 'deterministic_known_answer_not_acquired_data';
  seed: number;
  manifest_sha256: string;
  mining_interface: string;
  domain_branch_in_mining: false;
  selected_observations: string[];
  trajectories: {
    trajectory_id: string; domain: string; source_id: string; variable: string;
    native_semantics: string; native_units: string;
    support_start_seconds: number[]; support_end_seconds: number[]; valid_mask: boolean[];
    channel: string; channel_semantics: string; channel_units: string; values: number[];
    structural_scales: { coordinate: number; native_value: number; native_units: string; mapping: string }[];
    adapter: { id: string; version: string; definition_sha256: string; config_sha256: string };
    native_record: { content_sha256: string; locator: string; retained: boolean };
    lineage: { operation: string; source_variable: string; source_indices: number[];
      parameters: Record<string, number>; output_sha256: string };
    assumption_violations: string[];
    peak: { support_start_seconds: number; support_end_seconds: number; value: number; units: string };
  }[];
  claim_boundary: string;
}


// ---------------------------------------------------------------- TG17.6 orchestrated runs

/** The state machine as the backend enforces it, so the browser draws the transitions that
 *  actually exist rather than a picture of them that drifts. */
export interface RunStateMachine {
  schema: string;
  states: string[];
  work_stages: string[];
  terminal_states: string[];
  transitions: Record<string, string[]>;
  component_statuses: string[];
  retryable_statuses: string[];
  note: string;
}

/** A registered stage-worker suite. Everything registered today acquires nothing, and the
 *  capabilities say so rather than the label implying it. */
export interface RunWorkerSuite {
  name: string;
  description: string;
  capabilities: Record<string, any>;
  tags: string[];
}

export interface RunSummary {
  run_id: string;
  study_id: string;
  title: string;
  state: string;
  manifest_sha256: string;
  bounded_work: RunBoundedWork;
}

export interface RunContract {
  schema: string;
  state_machine: RunStateMachine;
  worker_suites: RunWorkerSuite[];
  runs: RunSummary[];
  available_now: string[];
  not_yet_available: string[];
  claim_boundary: string;
}

/** Bounded because the plan is declared. A progress bar over a search whose size is discovered
 *  as it runs is a progress bar that means nothing. */
export interface RunBoundedWork {
  completed_steps: number;
  total_steps: number;
  fraction: number;
  bytes_read: number;
}

/** What a watcher may see mid-run. There is no field here for a measurement value, a statistic
 *  or a p-value, because the backend type it comes from has none either. */
export interface RunComponentProgress {
  component: string;
  status: string;
  artifact_sha256: string | null;
  remediation: string;
  reused: boolean;
}

export interface RunProgress {
  schema: string;
  run_id: string;
  state: string;
  manifest_sha256: string;
  stages: { stage: string; components: RunComponentProgress[] }[];
  bounded_work: RunBoundedWork;
  retryable: boolean;
  results_visible: boolean;
  claim_boundary: string;
}

export interface RunReceipt {
  schema: string;
  run_id: string;
  run_sha256: string;
  manifest_sha256: string;
  study_id: string;
  state: string;
  history: { at: string; from: string; to: string; reason: string }[];
  artefacts: Record<string, string>;
  missing_components: { stage: string; component: string; status: string; detail: string;
    remediation: string }[];
  stage_decisions: { stage: string; verdict: string; reason: string }[];
  bounded_work: RunBoundedWork;
  coverage_policy: { requirement: string; minimum_fraction: number };
  confirmation: Record<string, any>;
  events: number;
  claim_boundary: string;
}

/** Posting a manifest opens the run that manifest identifies. `resumed` says which of the two
 *  happened, because "created" and "resumed" are the same request. */
export interface RunIdentity {
  schema: string;
  manifest_sha256: string;
  run_sha256: string;
  run_id: string;
  study_id: string;
  resumed: boolean;
  progress: RunProgress;
  receipt: RunReceipt;
}

export interface RunEditableCopy {
  draft_id: string;
  manifest_sha256: string;
  copied_from_run: string;
  frozen_run_state: string;
  frozen_run_untouched: boolean;
  note: string;
}

// ------------------------------------------------ TG17.9 portable experiment receipt

export interface ExperimentReceiptField {
  name: string;
  label: string;
  meaning: string;
}

export interface ExperimentReceiptCapabilities {
  schema: string;
  software_version: string;
  operations: { name: string; effect: string; writes_evidence: boolean }[];
  adapters: DomainExperimentAdapterDescription[];
  refusals: { name: string; reason: string }[];
  receipt_fields: ExperimentReceiptField[];
  lineage: string[];
  claim_boundary: string;
}

export interface ExperimentEvidenceHandoff {
  run_complete: boolean;
  eligible_actions: string[];
  automatic_actions: string[];
  categories: { category: string; status: 'PRESENT' | 'ABSENT'; source: string | null }[];
  proposed_study_id: string;
  claim_boundary: string;
}

export interface ExperimentReplayBundle extends Record<string, any> {
  schema: 'cross-domain-experiment-bundle/v1';
  bundle_sha256: string;
  run_receipt: RunReceipt;
  evidence_handoff: ExperimentEvidenceHandoff;
  methods_report: { schema: string; format: string; sha256: string; text: string };
  claim_boundary: string;
}

export interface ExperimentReceiptExport {
  schema: string;
  integrity: 'VERIFIED';
  bundle_sha256: string;
  report_sha256: string;
  bundle: ExperimentReplayBundle;
  claim_boundary: string;
}

export interface ExperimentReceiptReplay {
  schema: string;
  integrity: 'VERIFIED';
  bundle_sha256: string;
  manifest: CrossDomainExperimentManifest;
  run_identity: Record<string, string>;
  run_receipt: RunReceipt;
  results: Record<string, any>;
  refusals: Record<string, any>[];
  evidence_handoff: ExperimentEvidenceHandoff;
  methods_report: { schema: string; format: string; sha256: string; text: string };
  claim_boundary: string;
  bundle: ExperimentReplayBundle;
}

// -------------------------------------------------- TG17.10 release qualification

export interface ExperimentQualificationGate {
  gate_id: string;
  title: string;
  /** `REFUSED` is not `FAIL`. Nothing in the apparatus broke: a domain's declared contract
   *  forbids the plan. Both block release, and collapsing them would hide which one happened. */
  status: 'PASS' | 'FAIL' | 'REFUSED' | 'NOT_RUN' | 'NOT_IMPLEMENTED';
  blocking: boolean;
  detail: string;
}

export interface ExperimentQualificationCell {
  cell_id: string;
  duration: 'week' | 'three_months' | 'six_months';
  mode: 'calendar_aligned' | 'scale_shape_aligned';
  start_utc: string;
  end_utc: string;
  manifest_sha256: string;
  family_correction: string;
  record_kind: string;
  status: 'PASS' | 'FAIL' | 'REFUSED' | 'NOT_RUN';
  /** Absent on a refused cell: a plan the declarations refuse opens no run. */
  run_id?: string;
  bundle_sha256?: string;
  preflight_status?: string;
  checks?: Record<string, boolean>;
  refusals?: { domain?: string; reason: string }[];
}

// TG17.12: whether a rejection is arithmetically reachable at a declared configuration. The
// calendar null's floor is 1/(1+replications), so it is bought with computation and there is no
// enumeration ceiling - the opposite of the scale/shape null one field below.
export interface CalendarNullResolution {
  members: number;
  replications: number;
  p_value_floor: number;
  replications_required: number;
  can_reject_after_correction: boolean;
}

export interface ExperimentQualificationRecord {
  schema: string;
  qualification_sha256: string;
  verdict: 'RELEASEABLE' | 'NOT_RELEASEABLE';
  record_kind: string;
  matrix: ExperimentQualificationCell[];
  gates: ExperimentQualificationGate[];
  recovery?: {
    status: 'PASS' | 'FAIL'; run_id: string; failed_component: string;
    attempts_before_restart: Record<string, number>;
    attempts_after_retry: Record<string, number>;
    checks: Record<string, boolean>;
  };
  // TG17.12. A calibration measured outside orchestration and read here, never run here.
  // `status` is what the gate is entitled to say; `reasons` is why, and is non-empty whenever
  // the recording is absent, unbound from the contract or source it was made against, or failed.
  calendar_calibration?: {
    schema: string;
    entry_point: string;
    executed_here: boolean;
    record_path: string;
    contract_sha256: string;
    status: 'PASS' | 'FAIL' | 'NOT_RUN';
    reasons: string[];
    recorded_utc?: string;
    recorded: {
      recorded_utc: string;
      all_met: boolean;
      replications: number;
      family_size: number;
      cases: Array<{
        case: string; expectation: string;
        minimum_rejections: number; maximum_rejections: number;
        n_rejected_after_correction: number; met: boolean;
      }>;
      claim_boundary: string;
    } | null;
    applicability: {
      floor_is_bought_with: string;
      enumeration_ceiling: number | null;
      calibration_family: CalendarNullResolution;
      declared_plan: CalendarNullResolution & { declared_search_members: number };
      alpha: number;
      correction: string;
      claim_boundary: string;
    };
  };
  // TG17.11: why the scale/shape gate is REFUSED rather than absent. Applicability only - the
  // server does not run that calibration here and this section carries no power number.
  scale_shape_calibration?: {
    calibration: string;
    calibration_executed_here: boolean;
    inference_the_calibration_uses: string;
    inference_the_manifests_declare: string;
    alpha: number;
    correction: string;
    minimum_resolvable_family: number;
    largest_drawable_inventory: number;
    declared_null_can_ever_reject: boolean;
    declared_families: Array<{
      family: string; domains: string[]; pairings: number;
      null_refusal: string; reaches_resolvable_size: boolean;
    }>;
    claim_boundary: string;
  };
  // TG18.5 slice 4. Measured by a rendered run or reported as unmeasured, never synthesized by
  // the server. `MEASURED` carries the counts; `NOT_MEASURED` carries only the reasons it has
  // none. Numbers and strings both appear, so the value type stays open; the durations are named
  // `_unasserted` because nothing compares them to anything.
  scientist_actions: Record<string, string | number | null | string[]>;
  // What the `browser_no_glue` gate read, and why it said what it said.
  browser_evidence?: {
    schema: string;
    measured_by: string;
    executed_here: boolean;
    record_path: string;
    specs_in_this_checkout: number;
    status: 'PASS' | 'FAIL' | 'NOT_RUN';
    reasons: string[];
    recorded: {
      recorded_utc: string;
      playwright_status: string;
      passed: number;
      failed: number;
      skipped: number;
      specs_that_ran: number;
      artefacts: number;
    } | null;
    claim_boundary: string;
  };
  claim_boundary: string;
}

// ------------------------------------------------------- TG17.7 the guided path

/** One step of the workflow, described by the server. The browser renders these; it holds no
 *  copy of the order of operations, because two copies of an order of operations are two
 *  different experiments waiting to happen. */
export interface ComposerPathStep {
  step_id: string;
  ordinal: number;
  title: string;
  question: string;
  settles: string;
  controls: string[];
  action_label: string;
  action_route: string;
  claim_boundary: string;
}

export interface ComposerLadderRung {
  rung: string;
  title: string;
  is: string;
  is_not: string;
  gate: string;
  reached?: boolean;
  why_not?: string;
}

export interface ComposerPathContract {
  schema: string;
  steps: ComposerPathStep[];
  step_statuses: string[];
  duration_presets: string[];
  ladder: ComposerLadderRung[];
  note: string;
  claim_boundary: string;
}

export interface ComposerStepState extends ComposerPathStep {
  status: string;
  reason: string;
  detail: Record<string, any>;
}

/** `next_action` is one field on purpose. Two enabled controls meaning two different scientific
 *  commitments cannot both be the next legitimate act. */
export interface ComposerNextAction {
  step_id: string;
  label: string;
  route: string;
  status: string;
  why: string;
  blocked: boolean;
}

export interface ComposerPathState {
  schema: string;
  manifest_sha256: string;
  study_id: string;
  steps: ComposerStepState[];
  satisfied: number;
  next_action: ComposerNextAction | null;
  ladder: ComposerLadderRung[];
  run_state: string | null;
  claim_boundary: string;
}

export interface ComposerWindowPreset {
  preset: string;
  unit: string;
  amount: number;
  days: number;
  start_utc: string;
  end_utc: string;
  stride_seconds: number;
  label: string;
}

export interface ComposerWindowPresets {
  schema: string;
  anchor_utc: string;
  presets: ComposerWindowPreset[];
  note: string;
  claim_boundary: string;
}

export interface ComposerDomainOption {
  domain: string;
  adapter_id: string;
  label: string;
  licence: string;
  breaks: string[];
  lag_policy: string;
  precedence_admissible: boolean;
  admissible_kernels: string[];
  admissible_nulls: string[];
  selected: boolean;
  selectable: boolean;
  unavailable_reason: string;
  /** The declared observation this domain would join the study with, or null when no recipe
   *  declares one - in which case the row is offered disabled, with the reason. */
  observation: Record<string, any> | null;
  onboarding_cost: Record<string, any>;
}

export interface ComposerDomainMenu {
  schema: string;
  domains: ComposerDomainOption[];
  minimum_domains: number;
  note: string;
  claim_boundary: string;
}

export interface ComposerPreregistrationSummary {
  schema: string;
  manifest_sha256: string;
  study_id: string;
  sentences: string[];
  claim_boundary: string;
}

export interface ComposerManifestEnvelope {
  schema: string;
  manifest_sha256: string;
  envelope_sha256: string;
  exported_from: string;
  manifest: Record<string, any>;
  note: string;
  claim_boundary: string;
}

// ------------------------------------------------------------ TG17.8 comparison views

export interface ComparisonAxis {
  name: string;
  kind: string;
  domains: string[];
  units: string | null;
  /** True when more than one domain occupies the axis. A `native_magnitude` axis can never be
   *  shared: the server raises rather than returning one, so this pair is safe to render. */
  shared: boolean;
  why: string;
}

export interface ComparisonEncoding {
  role: string;
  /** The distinction is carried in three channels - `word`, `colour` and `marker` - so a reader
   *  who cannot use one still has two. Never render the colour alone. */
  word: string;
  colour: string;
  marker: string;
  ordinal: number;
  definition: string;
  admits_claim: boolean;
}

export interface ComparisonMark {
  label: string;
  role: string;
  word: string;
  colour: string;
  marker: string;
  domain: string | null;
  value: number | null;
  display: string;
  artifact_sha256: string | null;
  no_artifact_reason: string;
  admits_claim: boolean;
}

export interface ComparisonTable {
  columns: string[];
  rows: Record<string, any>[];
}

export interface ComparisonViewSummary {
  view_id: string;
  ordinal: number;
  title: string;
  question: string;
  axes: ComparisonAxis[];
  roles: string[];
  may_conclude: string;
  may_not_conclude: string[];
  selectable: boolean;
}

export interface ComparisonViewsContract {
  schema: string;
  views: ComparisonViewSummary[];
  encodings: ComparisonEncoding[];
  axis_kinds: string[];
  shared_axis_kinds: string[];
  coverage_cells: string[];
  readings: {
    declarable_by_mode: Record<string, string[]>;
    renderable_by_mode: Record<string, string[]>;
    never_admissible: string[];
    requires_external_design: string[];
    why_two_lists: string;
  };
  mode_forbids: Record<string, string>;
  refusals: Record<string, string>;
  routes: Record<string, string>;
  not_yet_available: string[];
  claim_boundary: string;
}

export interface ComparisonView {
  schema: string;
  view_id: string;
  ordinal: number;
  title: string;
  question: string;
  mode: string;
  manifest_sha256: string;
  axes: ComparisonAxis[];
  legend: ComparisonEncoding[];
  body: Record<string, any>;
  table: ComparisonTable;
  results_exist: boolean;
  run_state: string;
  may_conclude: string;
  may_not_conclude: string[];
  mode_forbids: string;
  claim_boundary: string;
}

export interface ComparisonViewSet {
  schema: string;
  mode: string;
  manifest_sha256: string;
  views: ComparisonView[];
}

export interface ComparisonContribution {
  domain: string;
  state: string;
  reason: string;
  native_interval: {
    start_utc: string;
    end_utc: string;
    support_kind: string | null;
    native_cadence_seconds: number | null;
    coverage_exact: boolean;
  };
  contributes: boolean;
}

export interface ComparisonLinkedSelection {
  schema: string;
  window: string;
  manifest_sha256: string;
  contributions: ComparisonContribution[];
  /** Always null. Each domain keeps its own native interval; one merged extent would show
   *  agreement about coverage only one domain addresses. */
  merged_interval: null;
  why_not_merged: string;
  claim_boundary: string;
}

export interface ComparisonReadingCheck {
  mode: string;
  reading: string;
  renderable: boolean;
  reason: string;
  claim_boundary?: string;
}

/* ---------------------------------------------------------------- T4C.5j: the gate record
 *
 * The atmospheric gate line is read-only from the browser. There is no request type here
 * because there is no request: every one of these is the shape of a GET response, and the
 * absence of a mutation type is the contract rather than an omission.
 */

/** Why the surface has no acquire button, stated by the server rather than assumed by the UI. */
export interface GateSurface {
  schema: string;
  campaigns: number;
  supersessions: number;
  retired_campaigns: number;
  receipts: number;
  measurement_status: 'MEASURED' | 'NOT_YET_MEASURED';
  unreadable: Record<string, string>[];
  refusals: string[];
  claim_boundary: string;
  network_used: boolean;
}

/** Present only on a retired campaign; `null` is the whole of "this design is still live". */
export interface GateRetirement {
  supersession_id: string;
  successor_campaign_id: string;
  successor_campaign_sha256: string;
  acquisition: string;
  statement: string;
}

export interface GateCampaignSummary {
  campaign_id: string;
  campaign_sha256: string;
  study_plan_sha256: string;
  file: string;
  status: 'ACTIVE' | 'RETIRED';
  retired_by: GateRetirement | null;
  variable: string;
  level_hpa: number;
  date_start: string;
  date_end: string;
  expected_frames: number;
  /** Whether the design can resolve its own declared family. A retired campaign reports
   *  `false` here and is still served, because that is the defect being recorded. */
  resolvable: boolean;
  hypothesis_family_size: number;
  /** D86. The rule by which the two ERA5 routes are declared to agree, frozen with the design.
   *  `null` means the campaign never preregistered one, which is why it may not be acquired:
   *  the decision to spend would rest on a constant in module code that no supersession
   *  governs. It is not a display preference and there is no default to fall back to. */
  overlap_criterion: GateOverlapCriterion | null;
}

export interface GateOverlapCriterion {
  /** `encoding_relative` judges agreement in units of the primary route's own packing step,
   *  which changes frame to frame; `absolute` judges it in the variable's units. */
  name: 'absolute' | 'encoding_relative';
  steps_allowed?: number;
  atol?: Record<string, number>;
}

export interface GateCampaignIndex {
  schema: string;
  campaigns: GateCampaignSummary[];
  unreadable: Record<string, string>[];
  network_used: boolean;
}

export interface GateCampaignReview {
  schema: string;
  campaign_id: string;
  campaign_sha256: string;
  study_plan_sha256: string;
  scientific_design: Record<string, any>;
  decision_rule: string;
  claim_boundary: string;
  network_used: boolean;
  file: string;
  status: 'ACTIVE' | 'RETIRED';
  retired_by: GateRetirement | null;
  refusals: string[];
}

export interface GateSupersessionSummary {
  supersession_id: string;
  supersession_sha256: string;
  file: string;
  superseded_campaign_id: string;
  successor_campaign_id: string;
  defects: string[];
  reason_count: number;
  preserved_count: number;
  deferred_to_run: string[];
}

export interface GateSupersessionIndex {
  schema: string;
  supersessions: GateSupersessionSummary[];
  unreadable: Record<string, string>[];
  network_used: boolean;
}

/** One stated reason, re-run against both campaigns. `passes` is the check's own outcome, so a
 *  reason is admissible exactly where superseded is false and successor is true. */
export interface GateSupersessionOutcome {
  passes: boolean;
  [key: string]: any;
}

export interface GateSupersessionFinding {
  defect?: string;
  check: string;
  parameters: Record<string, any>;
  statement: string;
  superseded: GateSupersessionOutcome;
  successor: GateSupersessionOutcome;
}

export interface GateSupersessionReview {
  schema: string;
  supersession_id: string;
  supersession_sha256: string;
  superseded_campaign_sha256: string;
  successor_campaign_sha256: string;
  superseded_campaign_id: string;
  successor_campaign_id: string;
  reasons: GateSupersessionFinding[];
  preserved: GateSupersessionFinding[];
  /** What the re-freeze does *not* settle. An empty list would be a claim, not a convenience. */
  deferred_to_run: { defect: string; statement: string; adjudicated_by: string }[];
  successor_review: GateCampaignReview;
  claim_boundary: string;
  network_used: boolean;
  file: string;
  refusals: string[];
}

export interface GateReceiptSummary {
  receipt_id: string;
  receipt_sha256: string;
  plan_sha256: string;
  study_id: string | null;
  evidence_role: string | null;
  /** What the replication rule returned. */
  gate_verdict: string;
  /** What a reviewer should read. These differ exactly where the derived power record moved it. */
  scientific_verdict: string;
  power_applied: boolean;
}

export interface GateReceiptIndex {
  schema: string;
  receipts: GateReceiptSummary[];
  unreadable: Record<string, string>[];
  /** `NOT_YET_MEASURED` is an absence of runs. It is not a finding of no relationship. */
  status: 'MEASURED' | 'NOT_YET_MEASURED';
  statement: string;
  network_used: boolean;
}

export interface GateReceiptView {
  schema: string;
  receipt_id: string;
  integrity: string;
  receipt: Record<string, any>;
  gate_verdict: string;
  scientific_verdict: string;
  power_adjudication: Record<string, any>;
  claim_boundary: string;
  network_used: boolean;
}

/** T4E.8 slice 4: the identity declaration surface. A refusal is a value here, not an error. */
export interface IdentityEvidenceCell {
  evidence_class: string;
  admitted: boolean;
  refusal: string | null;
  caveat?: string | null;
  label_boundary?: string;
  evidence_provenance?: string;
  independent_of_record: boolean;
}

export interface IdentityTargetRow {
  identity_target: string;
  recognises: string;
  does_not_license: string;
  evidence: IdentityEvidenceCell[];
}

export interface IdentityEvidenceClass {
  evidence_class: string;
  provenance: string;
  independent_of_record: boolean;
  label_boundary: string;
}

export interface IdentityTargets {
  targets: IdentityTargetRow[];
  evidence_classes: IdentityEvidenceClass[];
  choosing_is_not_automated: string;
  refusals: string[];
  network_used: boolean;
}

export interface IdentityDeclaration {
  identity_target: string;
  recognises: string;
  evidence_class: string;
  evidence_provenance: string;
  evidence_independent_of_record: boolean;
  label_boundary: string;
  does_not_license: string;
  caveat: string | null;
  admissible_evidence: string[];
}

export interface IdentityAuditSummary {
  file: string;
  schema: string | null;
  status: string | null;
  approved_mining_radius: number | null;
  frozen_radius: number | null;
  identity_declaration: IdentityDeclaration | null;
  code_revision: string | null;
  code_dirty: boolean | null;
  design_sha256: string | null;
  windows: number;
  claim_boundary: string | null;
  /** T4E.22. A summary that could not hold a verdict showed nulls where the finding was. */
  verdict: unknown;
  /** Every "what this may not be used for" clause the record carries, under whichever of the
   *  fifteen names this programme has used. Collected rather than normalised: renaming keys in
   *  committed evidence to suit a viewer would be rewriting evidence to fit its display. */
  boundaries: { key: string; text: unknown }[];
  /** Marks left when a record was corrected or superseded in the open. A corrected record that
   *  reads as current is the dangerous case, so the mark travels in the summary. */
  corrected_or_superseded: string[];
}

export interface IdentityMeasurementIndex {
  measurements: IdentityAuditSummary[];
  unreadable: Record<string, string>[];
  corrected_or_superseded: string[];
  correction_note: string;
  measurements_without_a_stated_boundary: string[];
  boundary_note: string;
  refusals: string[];
  network_used: boolean;
}

export interface IdentityMeasurementView {
  measurement: Record<string, any>;
  summary: IdentityAuditSummary;
  refusals: string[];
  network_used: boolean;
}

export interface IdentityStudy {
  task: string;
  declarations: IdentityAuditSummary[];
  adoptions?: IdentityAuditSummary[];
  measurements: IdentityAuditSummary[];
  has_a_result: boolean;
  declared_before_measured: boolean;
  corrected_or_superseded: string[];
}

export interface IdentityStudies {
  studies: IdentityStudy[];
  declared_but_not_measured: string[];
  declared_but_not_measured_note: string;
  files_outside_any_study: string[];
  unreadable: Record<string, string>[];
  refusals: string[];
  network_used: boolean;
}

export interface IdentityAuditIndex {
  audits: IdentityAuditSummary[];
  unreadable: Record<string, string>[];
  audits_without_a_declared_target: string[];
  undeclared_note: string;
  refusals: string[];
  network_used: boolean;
}

export interface IdentityAuditView {
  audit: Record<string, any>;
  summary: IdentityAuditSummary;
  refusals: string[];
  network_used: boolean;
}
