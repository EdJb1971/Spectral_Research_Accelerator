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
  spectral_diagnostics: {
    wavenumbers: number[];
    forecast_psd: number[];
    ground_truth_psd: number[];
    spectral_coherence: number[];
    forecast_slope_analysis?: {
      slope_beta: number;
      intercept_ln_c: number;
      r_squared: number;
      regime_interpretation: string;
    };
    ground_truth_slope_analysis?: {
      slope_beta: number;
      intercept_ln_c: number;
      r_squared: number;
      regime_interpretation: string;
    };
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
  checks: any[];
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
  }>;
}
